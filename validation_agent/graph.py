"""The Validation agent as a LangGraph graph: checks, runs and compares one translated program.

    START ─► static_check ─► execute ─► compare ─► review ─► verdict ─► END
                  │             │ crash / timeout ───▲          ▲
                  └─ rule broken ┴──────────────────────────────┘  (not run, no review)

The harness (sas_agent/graph.py) calls it through sas_agent.agents.validation_agent_validator():

    build_graph(review=...).invoke({"program_id": ..., "code": ..., "attempt_dir": ...})["result"]

- program_id: "program0", "program1" or "program2" (sas_agent.config.PROGRAMS)
- code: the whole translated script
- attempt_dir: runs/<run-id>/<program>/attempt-<n>; the script runs there (sas_agent/runs.py layout)

The result is a ValidationResult; the harness keeps next_state, rank and fix_request, and the Correction
agent reads a TRIAGED result's fix_request. Retry limits and escalation belong to the harness.
Only the review node uses an LLM; the graph state never holds data values.
"""

import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from sas_agent.runs import AttemptDirs

from . import checks, compare, runner, verdict
from . import review as code_review
from .schema import Finding, RedactedReport, ValidationResult


class ValidationState(TypedDict, total=False):
    # inputs, from the harness
    program_id: str
    code: str
    attempt_dir: str
    # filled in by the nodes
    violations: list[str]
    exception: dict | None
    comparison: RedactedReport | None
    review: list[Finding]
    events: Annotated[list[str], operator.add]   # every node appends its event
    result: ValidationResult


def build_graph(llm: BaseChatModel | None = None, review: bool = True, timeout: int | None = None):
    """Compile the validator graph. Same pattern as sas_agent.graph.build_graph.

    - llm=None builds the review model from the repository's .env; pass any chat model to override it,
      e.g. a fake one for tests.
    - review=False skips the LLM review (tool checks only, no API calls, no key needed).
    - timeout: seconds per script run (default config.TIMEOUT_SECONDS).
    """
    if review:
        from sas_agent.graph import _refuse_tracing   # tracing would send the code and SAS to LangSmith
        _refuse_tracing()
    llm = (llm or code_review.build_llm()) if review else None

    def dirs(state: ValidationState) -> AttemptDirs:
        if not state.get("attempt_dir"):
            raise ValueError("The validator needs an attempt_dir to run the script in.")
        return AttemptDirs(Path(state["attempt_dir"]))

    def static_check(state: ValidationState) -> dict:
        """TRANSLATED / PATCHED -> CHECKED."""
        violations = checks.check_code(state["code"])
        return {"violations": violations, "events": ["STATIC_FAIL" if violations else "STATIC_PASS"]}

    def execute(state: ValidationState) -> dict:
        """CHECKED -> EXECUTED: run the script with SAS_INPUT_DIR and SAS_OUTPUT_DIR set."""
        d = dirs(state)
        d.ai.mkdir(parents=True, exist_ok=True)
        d.outputs.mkdir(parents=True, exist_ok=True)
        script = d.ai / "script.py"
        script.write_text(state["code"], encoding="utf-8")
        inputs = runner.prepare_inputs(d.root)
        runner.seed_upstream(state["program_id"], d.outputs)
        exception = runner.run_script(script, inputs, d.outputs, d.data, timeout)
        event = ("EXEC_OK" if exception is None else
                 "EXEC_TIMEOUT" if exception["type"] == "Timeout" else "EXEC_ERROR")
        return {"exception": exception, "events": [event]}

    def compare_node(state: ValidationState) -> dict:
        """EXECUTED -> COMPARED."""
        program_id = state["program_id"]
        report = compare.compare_outputs(program_id, dirs(state).outputs, compare.expected_outputs(program_id))
        return {"comparison": report, "events": ["COMPARED_DONE"]}

    def review_node(state: ValidationState) -> dict:
        """LLM code review (part of COMPARED): hard-coding, quiet bugs, the cause of each failing check."""
        if llm is None:
            return {"review": []}
        findings = verdict.tool_findings(state["violations"], state.get("exception"), state.get("comparison"))
        found = code_review.llm_review(llm, state["program_id"], state["code"], findings)
        return {"review": found, "events": ["REVIEW_DONE"]}

    def verdict_node(state: ValidationState) -> dict:
        """-> PASSED, or TRIAGED with a fix request for the Correction agent."""
        result = verdict.make_result(state["program_id"], state["events"], state["violations"],
                                     state.get("exception"), state.get("comparison"), state.get("review", []))
        verdict.write_validation(dirs(state).root, result)
        return {"result": result}

    graph = StateGraph(ValidationState)
    graph.add_node("static_check", static_check)
    graph.add_node("execute", execute)
    graph.add_node("compare", compare_node)
    graph.add_node("review", review_node)
    graph.add_node("verdict", verdict_node)
    graph.add_edge(START, "static_check")
    graph.add_conditional_edges("static_check", lambda s: "verdict" if s["violations"] else "execute",
                                ["verdict", "execute"])
    graph.add_conditional_edges("execute", lambda s: "review" if s["exception"] else "compare",
                                ["review", "compare"])   # a crash is reviewed too
    graph.add_edge("compare", "review")
    graph.add_edge("review", "verdict")
    graph.add_edge("verdict", END)
    return graph.compile()
