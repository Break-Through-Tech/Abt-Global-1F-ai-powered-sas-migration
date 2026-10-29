"""Scenario tests for the fix loop (design doc 6.12.7).

They use a fake chat model, a fake comparator, and a scripted Corrector, and run the real static checks
and executor. They make no API calls.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sas_agent.comparison import CheckResult, RedactedReport
from sas_agent.graph import RECURSION_LIMIT, build_graph, route


def script(a: int = 1, b: int = 2) -> str:
    """A script that writes OUT.csv; the fake comparator expects a=1 and b=2."""
    return (
        "import os\nimport pandas as pd\n"
        f"pd.DataFrame({{'a': [{a}], 'b': [{b}]}}).to_csv(os.path.join(os.environ['SAS_OUTPUT_DIR'], 'OUT.csv'), index=False)\n"
    )


SYNTAX_ERROR = "x = (\n"
# Fails with KeyError: '010001', a hospital identifier read from the input data.
RUNTIME_ERROR = (
    "import os\nimport pandas as pd\n"
    "df = pd.read_csv(os.path.join(os.environ['SAS_INPUT_DIR'], 'in.csv'), dtype={'PROVIDER_ID': str})\n"
    "lookup = {}\n"
    "value = lookup[df['PROVIDER_ID'][0]]\n"
)


def fake_comparator(program_id: str, output_dir: Path, work_dir: Path) -> RedactedReport:
    out = Path(output_dir) / "OUT.csv"
    if not out.exists():
        return RedactedReport([CheckResult("OUT", "file_exists", False)])
    row = pd.read_csv(out).iloc[0]
    checks = []
    for column, expected in (("a", 1), ("b", 2)):
        ok = row[column] == expected
        checks.append(CheckResult("OUT", "values", ok, column=column,
                                  mismatch=None if ok else "<11", error_bin=None if ok else ">=1e-3"))
    return RedactedReport(checks)


class ScriptedCorrector:
    """Returns the given scripts in order and records the context of each call."""

    def __init__(self, *scripts: str):
        self.scripts = list(scripts)
        self.contexts: list[dict] = []

    def __call__(self, context: dict) -> str:
        self.contexts.append(context)
        return self.scripts.pop(0)


@pytest.fixture
def run(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "in.csv").write_text("PROVIDER_ID,x\n010001,1.5\n")

    def _run(first_script: str, corrector=None, **kwargs):
        llm = FakeListChatModel(responses=["## Analysis", f"```python\n{first_script}```"])
        app = build_graph(llm, mode="fix_loop", comparator=fake_comparator, corrector=corrector or ScriptedCorrector(),
                          runs_dir=tmp_path / "runs", migrated_dir=tmp_path / "migrated", **kwargs)
        return app.invoke({"program_id": "program0", "run_id": "r1", "input_dir": str(inputs)},
                          {"recursion_limit": RECURSION_LIMIT})

    return _run


def test_passes_first_time_and_is_promoted(run, tmp_path):
    corrector = ScriptedCorrector()
    state = run(script(), corrector)
    assert (state["status"], state["attempt"]) == ("PASSED", 1)
    assert corrector.contexts == []
    promoted = tmp_path / "migrated" / "program0"
    assert (promoted / "program0.py").read_text() == script()
    provenance = json.loads((promoted / "provenance.json").read_text())
    assert provenance["run_id"] == "r1" and provenance["attempt"] == 1
    assert provenance["comparison"] == {"checks": 2, "failing": 0}


def test_syntax_error_is_corrected_before_execution(run, tmp_path):
    corrector = ScriptedCorrector(script())
    state = run(SYNTAX_ERROR, corrector)
    assert (state["status"], state["attempt"]) == ("PASSED", 2)
    assert "SyntaxError" in corrector.contexts[0]["fix_request"]
    assert not (tmp_path / "runs" / "r1" / "program0" / "attempt-1" / "ai" / "exec_status.json").exists()


def test_runtime_error_is_corrected_without_leaking_data(run):
    corrector = ScriptedCorrector(script())
    state = run(RUNTIME_ERROR, corrector)
    assert state["status"] == "PASSED"
    context = corrector.contexts[0]
    assert "KeyError" in context["fix_request"]
    assert "010001" not in json.dumps(context)


def test_wrong_values_are_corrected(run):
    corrector = ScriptedCorrector(script())
    state = run(script(b=5), corrector)
    assert (state["status"], state["attempt"]) == ("PASSED", 2)
    assert "column b" in corrector.contexts[0]["fix_request"]


def test_escalates_after_three_corrections(run, tmp_path):
    # Every correction improves, so only the correction limit stops the loop.
    corrector = ScriptedCorrector(RUNTIME_ERROR, script(a=9, b=9), script(b=9))
    state = run(SYNTAX_ERROR, corrector)
    assert (state["status"], state["corrections"]) == ("ESCALATED", 3)
    assert state["attempt"] == 4 and state["code"] == script(b=9)  # the best attempt is kept
    assert (tmp_path / "runs" / "r1" / "program0" / "escalation.json").exists()
    assert not (tmp_path / "migrated").exists()


def test_stuck_rule_escalates_after_two_corrections_without_improvement(run):
    corrector = ScriptedCorrector(script(b=9), script(b=9))
    state = run(script(b=9), corrector)
    assert (state["status"], state["corrections"], state["no_improvement"]) == ("ESCALATED", 2, 2)


def test_regression_is_rejected(run):
    first = script(b=9)  # one failing check
    corrector = ScriptedCorrector(RUNTIME_ERROR, script())  # the first correction is worse
    state = run(first, corrector)
    assert (state["status"], state["attempt"]) == ("PASSED", 3)
    assert corrector.contexts[1]["code"] == first  # corrected from the best attempt, not the regression


def test_missing_fix_prompt_names_its_owner(tmp_path):
    # Without a corrector argument the graph uses prompts/fix.md, which the translation team writes.
    llm = FakeListChatModel(responses=["## Analysis", f"```python\n{SYNTAX_ERROR}```"])
    app = build_graph(llm, mode="fix_loop", comparator=fake_comparator,
                      runs_dir=tmp_path / "runs", migrated_dir=tmp_path / "migrated")
    with pytest.raises(FileNotFoundError, match="translation team"):
        app.invoke({"program_id": "program0", "run_id": "r1", "input_dir": str(tmp_path)},
                   {"recursion_limit": RECURSION_LIMIT})


def test_baseline_is_single_pass(tmp_path):
    llm = FakeListChatModel(responses=["## Analysis", f"```python\n{script()}```"])
    app = build_graph(llm, mode="baseline", runs_dir=tmp_path / "runs", migrated_dir=tmp_path / "migrated")
    state = app.invoke({"program_id": "program0", "run_id": "r1"})
    assert state["status"] == "BASELINE"
    assert (tmp_path / "runs" / "r1" / "program0" / "attempt-1" / "ai" / "script.py").exists()
    assert not (tmp_path / "migrated").exists()


def test_tracing_is_refused(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with pytest.raises(RuntimeError, match="LANGSMITH_TRACING"):
        build_graph(FakeListChatModel(responses=[]))


def test_fix_loop_needs_a_comparator():
    with pytest.raises(ValueError, match="comparator"):
        build_graph(FakeListChatModel(responses=[]), mode="fix_loop")


def test_undefined_transition_is_an_error():
    with pytest.raises(RuntimeError, match="No transition"):
        route("execute", {"event": "ALL_PASS"})
