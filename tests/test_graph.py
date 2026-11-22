"""Scenario tests for the fix loop.

They use a fake chat model and fake validator, corrector, and repairer callables in place of the other
agents, so they need neither an API key nor the teammates' packages.
"""

import json
import sys
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sas_agent.agents import MissingAgent, ValidationView, correction_agent_corrector, validation_agent_validator
from sas_agent.graph import RECURSION_LIMIT, build_graph, route

# Scripts the fake translator and repairer return, and the validation result each one gets.
# Ranks are lower-is-better tuples, like the validation agent's (stage, failing checks, mismatches, error).
GOOD = "result = 'good'\n"
ONE_FAILING = "result = 'one failing check'\n"
TWO_FAILING = "result = 'two failing checks'\n"
CRASHES = "result = 'crashes'\n"
RESULTS = {
    GOOD: ("PASSED", (-2, 0, 0, 0.0)),
    ONE_FAILING: ("TRIAGED", (-2, 1, 4566, 0.5)),
    TWO_FAILING: ("TRIAGED", (-2, 2, 9132, 0.5)),
    CRASHES: ("TRIAGED", (-1, 1, 0, 0.0)),
}


class FakeValidator:
    def __init__(self):
        self.calls = []

    def __call__(self, program_id, code, attempt_dir):
        self.calls.append(code)
        state, rank = RESULTS[code]
        # Extra fields stand for the rest of a ValidationResult; the harness must ignore them.
        return {"program_id": program_id, "next_state": state, "rank": rank, "events": ["..."],
                "fix_request": None if state == "PASSED" else f"checks failed for {code.strip()}"}


class FakeCorrector:
    def __init__(self):
        self.calls = []

    def __call__(self, program_id, code, validation, *, analysis, prior_code, history):
        self.calls.append({"code": code, "validation": validation, "history": history})
        return f"REPAIR PROMPT for {code.strip()}"


class ScriptedRepairer:
    """Returns the given scripts in order; an exception instance is raised instead of returned."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def run(tmp_path):
    def _run(first_reply, *repairs, translation_reply=None):
        validator, corrector, repairer = FakeValidator(), FakeCorrector(), ScriptedRepairer(*repairs)
        reply = translation_reply if translation_reply is not None else f"```python\n{first_reply}```"
        app = build_graph(FakeListChatModel(responses=["## Analysis", reply]), mode="fix_loop",
                          validator=validator, corrector=corrector, repairer=repairer,
                          runs_dir=tmp_path / "runs", migrated_dir=tmp_path / "migrated")
        state = app.invoke({"program_id": "program0", "run_id": "r1"}, {"recursion_limit": RECURSION_LIMIT})
        return state, validator, corrector, repairer

    return _run


def test_passes_first_time_and_is_promoted(run, tmp_path):
    state, validator, corrector, _ = run(GOOD)
    assert (state["status"], state["attempt"]) == ("PASSED", 1)
    assert corrector.calls == []
    promoted = tmp_path / "migrated" / "program0"
    assert (promoted / "program0.py").read_text() == GOOD
    provenance = json.loads((promoted / "provenance.json").read_text())
    assert provenance["attempt"] == 1 and provenance["validation"]["next_state"] == "PASSED"


def test_passes_after_one_correction(run, tmp_path):
    state, validator, corrector, repairer = run(ONE_FAILING, GOOD)
    assert (state["status"], state["attempt"], state["corrections"]) == ("PASSED", 2, 1)
    call = corrector.calls[0]
    assert call["code"] == ONE_FAILING
    assert call["validation"]["next_state"] == "TRIAGED" and call["validation"]["fix_request"]
    assert call["history"] == [{"program_id": "program0", "rank": list(RESULTS[ONE_FAILING][1])}]
    assert repairer.prompts == [f"REPAIR PROMPT for {ONE_FAILING.strip()}"]
    attempt2 = tmp_path / "runs" / "r1" / "program0" / "attempt-2" / "ai"
    assert (attempt2 / "script.py").read_text() == GOOD and (attempt2 / "repair_prompt.md").exists()


def test_regression_is_rejected(run):
    # The first correction makes things worse; the next one starts from the best attempt.
    state, _, corrector, _ = run(ONE_FAILING, TWO_FAILING, GOOD)
    assert (state["status"], state["attempt"]) == ("PASSED", 3)
    assert [c["code"] for c in corrector.calls] == [ONE_FAILING, ONE_FAILING]


def test_correction_limit_escalates_and_keeps_the_best_attempt(run, tmp_path):
    # Every correction improves, so only the limit of 3 corrections stops the loop.
    state, _, corrector, _ = _improving_run(run)
    assert (state["status"], state["corrections"]) == ("ESCALATED", 3)
    assert state["code"] == ONE_FAILING and state["attempt"] == 4
    assert json.loads((tmp_path / "runs" / "r1" / "program0" / "escalation.json").read_text())["reason"] == "LIMIT"
    assert not (tmp_path / "migrated").exists()


def _improving_run(run):
    # Ranks improve at every attempt but never pass: crash -> two failing -> ... -> one failing.
    worse = "result = 'three failing checks'\n"
    RESULTS[worse] = ("TRIAGED", (-2, 3, 13698, 0.5))
    try:
        return run(CRASHES, worse, TWO_FAILING, ONE_FAILING)
    finally:
        del RESULTS[worse]


def test_stuck_rule_escalates_after_two_corrections_without_improvement(run):
    state, *_ = run(ONE_FAILING, ONE_FAILING, ONE_FAILING)
    assert (state["status"], state["corrections"], state["no_improvement"]) == ("ESCALATED", 2, 2)


def test_translation_without_code_block_escalates(run):
    state, validator, *_ = run(None, translation_reply="I cannot translate this.")
    assert state["status"] == "ESCALATED" and state["code"] == ""
    assert validator.calls == []


def test_repair_without_code_block_escalates_with_best_attempt(run):
    state, *_ = run(ONE_FAILING, ValueError("no code block"))
    assert state["status"] == "ESCALATED" and state["code"] == ONE_FAILING


def test_baseline_is_single_pass(tmp_path):
    llm = FakeListChatModel(responses=["## Analysis", f"```python\n{GOOD}```"])
    app = build_graph(llm, mode="baseline", runs_dir=tmp_path / "runs", migrated_dir=tmp_path / "migrated")
    state = app.invoke({"program_id": "program0", "run_id": "r1"})
    assert state["status"] == "BASELINE"
    assert (tmp_path / "runs" / "r1" / "program0" / "attempt-1" / "ai" / "script.py").exists()
    assert not (tmp_path / "migrated").exists()


def test_validation_view_reads_objects_and_dicts():
    obj = SimpleNamespace(next_state="TRIAGED", rank=(-2, 1, 3, 0.1), fix_request="x", extra=1)
    assert ValidationView.of(obj).rank == (-2, 1, 3, 0.1)
    assert ValidationView.of({"next_state": "PASSED", "rank": [0]}).passed
    with pytest.raises(ValueError):
        ValidationView.of({"next_state": "DONE", "rank": [0]})
    with pytest.raises(ValueError):
        ValidationView.of({"next_state": "TRIAGED", "rank": [0], "fix_request": ""})


def test_missing_teammate_packages_name_their_branch(monkeypatch):
    monkeypatch.setitem(sys.modules, "validation_agent", None)
    monkeypatch.setitem(sys.modules, "correction_agent", None)
    with pytest.raises(MissingAgent, match="branch validation_agent"):
        validation_agent_validator()
    with pytest.raises(MissingAgent, match="branch feature/correction_agent"):
        correction_agent_corrector()


def test_tracing_is_refused(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with pytest.raises(RuntimeError, match="LANGSMITH_TRACING"):
        build_graph(FakeListChatModel(responses=[]))


def test_fix_loop_needs_validator_and_corrector():
    with pytest.raises(ValueError, match="validator and a corrector"):
        build_graph(FakeListChatModel(responses=[]), mode="fix_loop")


def test_undefined_transition_is_an_error():
    with pytest.raises(RuntimeError, match="No transition"):
        route("validate", {"event": "REPAIRED"})
