"""Offline tests for the Validation agent: no API key and no model calls (review=False or a fake model).

Most tests replace the SAS results with a small table (OUT) and the run's input with a two-row CSV, so the
scripts are short. The last tests connect the validator to the harness's fix loop and the Correction agent.

validation_agent is imported inside the tests, not at the top of this file: pytest imports every test file
before running any test, and test_graph.py checks what the harness does when validation_agent is missing.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

HEADER = '''import os
import pandas as pd

OUT_DIR = os.environ["SAS_OUTPUT_DIR"]
inp = pd.read_csv(os.path.join(os.environ["SAS_INPUT_DIR"], "alldata_2025jul.csv"),
                  dtype={"PROVIDER_ID": str}, float_precision="round_trip")
'''
WRITE = 'out.to_csv(os.path.join(OUT_DIR, "OUT.csv"), index=False)\n'
GOOD = HEADER + 'out = inp[["PROVIDER_ID", "score"]]\n' + WRITE
WRONG_VALUES = HEADER + 'out = inp[["PROVIDER_ID"]].assign(score=inp["score"] * 2)\n' + WRITE
CRASH_LINE = 'raise KeyError(inp["PROVIDER_ID"][0])'
CRASH_WITH_ID = HEADER + CRASH_LINE + "\n"
CHAINED = HEADER + 'out = inp[["PROVIDER_ID", "score"]].copy()\nout["score"][0] = 5.0\n' + WRITE
SLEEPS = HEADER + 'import time\ntime.sleep(60)\n'
NO_OUTPUT = HEADER + 'out = inp\n'

PRIVATE_ID = "010001"   # a hospital ID: a data value that must never reach ai/ or the fix request


def validate(*args, **kwargs):
    from validation_agent import validate as run
    return run(*args, **kwargs)


@pytest.fixture
def expected(monkeypatch):
    """The SAS result the scripts are compared with: OUT, two hospitals, one missing score."""
    table = {"OUT": pd.DataFrame({"PROVIDER_ID": pd.Series([PRIVATE_ID, "010005"], dtype="str"),
                                  "score": [1.5, float("nan")]})}
    monkeypatch.setattr("validation_agent.compare.expected_outputs",
                        lambda program_id: {k: v.copy() for k, v in table.items()})
    return table


@pytest.fixture
def runs_dir(tmp_path):
    """runs/r1 with a two-row input file, so prepare_inputs() doesn't convert the real dataset."""
    inputs = tmp_path / "runs" / "r1" / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "alldata_2025jul.csv").write_text(f"PROVIDER_ID,score\n{PRIVATE_ID},1.5\n010005,\n")
    return tmp_path / "runs"


def attempt(runs_dir, n=1):
    return runs_dir / "r1" / "program0" / f"attempt-{n}"


def ai_text(attempt_dir) -> str:
    return (attempt_dir / "ai" / "validation.json").read_text()


# --- static checks ---------------------------------------------------------------------------------

def test_a_contract_script_passes_the_static_checks():
    from validation_agent.checks import check_code
    assert check_code(GOOD) == []


@pytest.mark.parametrize("code, rule", [
    ("", "empty"),
    (GOOD + "x = (\n", "SyntaxError"),
    ("import scipy\n" + GOOD, "import of 'scipy'"),
    ("import subprocess\n" + GOOD, "import of 'subprocess'"),
    (GOOD + "eval('1')\n", "eval()"),
    (GOOD + "os.system('dir')\n", "os.system()"),
    (GOOD + "pd.read_sas('x.sas7bdat')\n", ".read_sas()"),
    (GOOD + "ref = pd.read_csv('data/Project_1/SAS Output CSV/OUT.csv')\n", "SAS results"),
    (GOOD.replace("SAS_OUTPUT_DIR", "OUT_DIR"), "SAS_OUTPUT_DIR"),
])
def test_static_checks_name_the_broken_rule(code, rule):
    from validation_agent.checks import check_code
    assert any(rule in v for v in check_code(code))


# --- comparison --------------------------------------------------------------------------------------

def test_column_differences_name_missing_extra_and_case():
    from validation_agent.compare import column_differences
    count, detail = column_differences(["PROVIDER_ID", "Score", "extra"], ["PROVIDER_ID", "score", "total"])
    assert count == 3
    assert "expected score, got Score" in detail and "missing: total" in detail and "extra: extra" in detail
    count, detail = column_differences(["b", "a"], ["a", "b"])
    assert count == 2 and "position 1: expected a, got b" in detail


def test_compare_reports_each_kind_of_difference(expected):
    from validation_agent.compare import compare_dataset
    sas = expected["OUT"]
    assert compare_dataset("OUT", sas.copy(), sas) == []
    assert compare_dataset("OUT", sas.assign(score=sas["score"] + 1e-9), sas) == []   # within tolerance
    kinds = {
        "row_count": sas.iloc[:1],
        "row_order": sas.iloc[::-1],
        "missing_values": sas.assign(score=[1.5, 2.0]),
        "values": sas.assign(score=[2.5, float("nan")]),
    }
    for kind, py in kinds.items():
        assert [c.check for c in compare_dataset("OUT", py, sas)] == [kind], kind
    # Columns in the wrong order are reported, and the values are still compared by name.
    found = compare_dataset("OUT", sas[["score", "PROVIDER_ID"]].assign(score=[9.0, float("nan")]), sas)
    assert [c.check for c in found] == ["columns", "values"]


def test_missing_and_misnamed_output_files(tmp_path, expected):
    from validation_agent.compare import compare_outputs
    report = compare_outputs("program0", tmp_path, expected)
    assert [c.check for c in report.checks] == ["file_missing"] and not report.passed
    expected["OUT"].to_csv(tmp_path / "Out.csv", index=False)
    checks = compare_outputs("program0", tmp_path, expected).checks
    # On a case-insensitive file system (Windows) Out.csv is OUT.csv; elsewhere it is reported by name.
    assert checks == [] or "found Out.csv" in checks[0].detail


# --- the graph, with review=False ------------------------------------------------------------------

def test_a_matching_script_passes(runs_dir, expected):
    result = validate("program0", GOOD, attempt(runs_dir), review=False)
    assert result.next_state == "PASSED" and result.fix_request is None
    assert result.rank == (-2, 0, 0, 0.0)
    assert result.events == ["STATIC_PASS", "EXEC_OK", "COMPARED_DONE", "ALL_PASS"]
    assert (attempt(runs_dir) / "data" / "outputs" / "OUT.csv").exists()
    assert json.loads(ai_text(attempt(runs_dir)))["next_state"] == "PASSED"


def test_wrong_values_are_triaged_without_data_values(runs_dir, expected):
    result = validate("program0", WRONG_VALUES, attempt(runs_dir), review=False)
    assert result.next_state == "TRIAGED" and result.stage_reached == "compare"
    assert result.rank == (-2, 1, 1, 1.5)
    assert "| OUT | score | values | 1 | 1.5 |" in result.fix_request
    assert PRIVATE_ID not in result.fix_request and PRIVATE_ID not in ai_text(attempt(runs_dir))


def test_a_crash_reports_type_and_line_but_not_the_message(runs_dir, expected):
    result = validate("program0", CRASH_WITH_ID, attempt(runs_dir), review=False)
    line = CRASH_WITH_ID.splitlines().index(CRASH_LINE) + 1
    assert result.exception == {"type": "KeyError", "line": line, "source": CRASH_LINE}
    assert result.rank == (-1, 1, 0, 0.0) and result.comparison is None
    assert result.events == ["STATIC_PASS", "EXEC_ERROR", "FAIL"]
    assert PRIVATE_ID not in ai_text(attempt(runs_dir))
    assert PRIVATE_ID in (attempt(runs_dir) / "data" / "stderr.txt").read_text()   # the data zone keeps it


def test_chained_assignment_is_an_error(runs_dir, expected):
    result = validate("program0", CHAINED, attempt(runs_dir), review=False)
    assert result.exception["type"] == "ChainedAssignmentError"
    assert result.exception["source"] == 'out["score"][0] = 5.0'


def test_a_script_that_runs_too_long_times_out(runs_dir, expected):
    result = validate("program0", SLEEPS, attempt(runs_dir), review=False, timeout=3)
    assert result.exception["type"] == "Timeout" and "EXEC_TIMEOUT" in result.events
    assert "time limit" in result.fix_request


def test_a_missing_output_is_triaged(runs_dir, expected):
    result = validate("program0", NO_OUTPUT, attempt(runs_dir), review=False)
    assert [c.check for c in result.comparison.checks] == ["file_missing"]


def test_a_rule_breaking_script_is_not_run(runs_dir, expected):
    result = validate("program0", "import scipy\n" + GOOD, attempt(runs_dir), review=False)
    assert result.stage_reached == "static_check" and result.rank == (0, 1, 0, 0.0)
    assert result.events == ["STATIC_FAIL", "FAIL"]
    assert not (attempt(runs_dir) / "data" / "stdout.txt").exists()


def test_the_script_gets_no_secrets(runs_dir, expected, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "SECRET")
    code = HEADER + 'out = pd.DataFrame({"name": sorted(os.environ)})\n' + WRITE
    validate("program0", code, attempt(runs_dir), review=False)
    names = pd.read_csv(attempt(runs_dir) / "data" / "outputs" / "OUT.csv")["name"].tolist()
    assert "GOOGLE_API_KEY" not in names and {"SAS_INPUT_DIR", "SAS_OUTPUT_DIR"} <= set(names)


# --- the graph, with a fake review model ---------------------------------------------------------

class FakeReviewModel:
    """Stands in for Gemini's structured output: returns the given findings and keeps the prompt."""

    def __init__(self, *findings):
        self.findings, self.prompts = list(findings), []

    def with_structured_output(self, schema):
        self.schema = schema
        return SimpleNamespace(invoke=self._invoke)

    def _invoke(self, messages):
        self.prompts.append(messages[1].content)
        return self.schema(findings=self.findings)


def finding(severity):
    from validation_agent.schema import Finding
    return Finding(kind="hardcoded", severity=severity, python_lines="L6", sas_lines="program L40",
                   problem="the list of measures is written as a literal", what_to_check="derive it from the data")


def test_a_major_review_finding_fails_a_matching_script(runs_dir, expected):
    llm = FakeReviewModel(finding("major"))
    result = validate("program0", GOOD, attempt(runs_dir), llm=llm)
    assert result.next_state == "TRIAGED" and "REVIEW_DONE" in result.events
    assert "[major] hardcoded, Python L6, SAS program L40" in result.fix_request
    prompt = llm.prompts[0]
    assert "every output matches SAS" in prompt and "%macro" in prompt.lower() and PRIVATE_ID not in prompt


def test_a_minor_review_finding_does_not_block(runs_dir, expected):
    result = validate("program0", GOOD, attempt(runs_dir), llm=FakeReviewModel(finding("minor")))
    assert result.next_state == "PASSED" and len(result.review) == 1


def test_the_review_sees_the_tool_findings_of_a_failing_script(runs_dir, expected):
    llm = FakeReviewModel()
    validate("program0", WRONG_VALUES, attempt(runs_dir), llm=llm)
    assert "| OUT | score | values |" in llm.prompts[0]


# --- inputs of a run -------------------------------------------------------------------------------

def test_later_programs_get_the_sas_results_of_earlier_programs(tmp_path):
    from validation_agent.compare import compare_dataset, load_reference
    from validation_agent.runner import seed_upstream
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "OUTCOME_MORTALITY.csv").write_text("stale")   # program1's own output is removed
    seed_upstream("program1", out)
    assert sorted(p.name for p in out.iterdir()) == [
        "LESS100_MEASURE.csv", "MEASURE_AVERAGE_STDDEV_2025JUL.csv", "STD_DATA_2025JUL_ANALYSIS.csv"]
    std = pd.read_csv(out / "STD_DATA_2025JUL_ANALYSIS.csv", dtype={"PROVIDER_ID": str}, float_precision="round_trip")
    assert compare_dataset("STD", std, load_reference("STD_DATA_2025JUL_ANALYSIS")) == []


def test_the_run_input_is_written_once_per_run(tmp_path, monkeypatch):
    from validation_agent import runner
    calls = []
    monkeypatch.setattr(runner.runs, "prepare_inputs", lambda run_id, runs_dir: calls.append((run_id, runs_dir)))
    runner.prepare_inputs(tmp_path / "runs" / "r9" / "program0" / "attempt-1")
    assert calls == [("r9", (tmp_path / "runs").resolve())]


# --- connected to the harness and the Correction agent ---------------------------------------------

class FakeCorrectionModel:
    """The Correction agent's model: returns a prose repair plan."""

    def invoke(self, messages):
        self.prompt = messages[1].content
        return SimpleNamespace(text="Use the score column unchanged; SAS does not double it.",
                               response_metadata={"finish_reason": "STOP"})


def test_the_harness_adapter_builds_this_validator():
    from sas_agent.agents import validation_agent_validator
    assert callable(validation_agent_validator(review=False))   # no MissingAgent: validation_agent.graph exists


def test_fix_loop_with_the_real_validator_and_correction_agent(runs_dir, expected, tmp_path):
    from correction_agent import correct
    from sas_agent.agents import validation_agent_validator
    from sas_agent.graph import RECURSION_LIMIT
    from sas_agent.graph import build_graph as build_harness

    correction_model = FakeCorrectionModel()

    def corrector(program_id, code, validation, *, analysis, prior_code, history):
        return correct(program_id, code, validation, analysis=analysis, prior_code=prior_code,
                       history=history, llm=correction_model).repair_prompt

    repairs = []

    def repairer(prompt):
        repairs.append(prompt)
        return GOOD

    translator = FakeListChatModel(responses=["## Analysis", f"```python\n{WRONG_VALUES}```"])
    app = build_harness(translator, mode="fix_loop", validator=validation_agent_validator(review=False),
                        corrector=corrector, repairer=repairer, runs_dir=runs_dir,
                        migrated_dir=tmp_path / "migrated")
    state = app.invoke({"program_id": "program0", "run_id": "r1"}, {"recursion_limit": RECURSION_LIMIT})

    assert (state["status"], state["attempt"], state["corrections"]) == ("PASSED", 2, 1)
    assert "| OUT | score | values | 1 | 1.5 |" in correction_model.prompt   # the fix request reached the corrector
    assert PRIVATE_ID not in correction_model.prompt
    assert "SAS does not double it" in repairs[0]
    assert json.loads(ai_text(attempt(runs_dir, 1)))["next_state"] == "TRIAGED"
    assert json.loads(ai_text(attempt(runs_dir, 2)))["next_state"] == "PASSED"
    assert (tmp_path / "migrated" / "program0" / "program0.py").read_text() == GOOD
