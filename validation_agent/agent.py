"""The Validation agent's entry point, like correction_agent.correct().

In the fix loop the harness calls the graph itself (sas_agent/agents.py imports validation_agent.graph);
validate() is the same call for scripts, tests and notebooks.
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel

from .graph import build_graph
from .schema import ValidationResult


def validate(program_id: str, code: str, attempt_dir: str | Path, *, llm: BaseChatModel | None = None,
             review: bool = True, timeout: int | None = None) -> ValidationResult:
    """Check, run and compare one translated program; return PASSED or TRIAGED with a fix request.

    attempt_dir is runs/<run-id>/<program>/attempt-<n>. review=False makes no model calls.
    """
    state = build_graph(llm, review, timeout).invoke(
        {"program_id": program_id, "code": code, "attempt_dir": str(attempt_dir)})
    return state["result"]
