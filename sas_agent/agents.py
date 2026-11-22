"""The harness's view of the other agents (validation, correction, translation).

The harness calls these agents through the small interfaces below and never imports their packages at
module level, so it runs and is tested without them. Each interface describes only what the harness
uses; the agents may return more.

Agreed interfaces, as implemented on the teammates' branches (read 2026-11-11):
- Validation agent (branch validation_agent):
      validation_agent.graph.build_graph().invoke({"program_id", "code", "attempt_dir"})["result"]
  returns a ValidationResult with next_state ("PASSED" or "TRIAGED"), rank (lower is better), and
  fix_request (redacted text for the Correction agent).
- Correction agent (branch feature/correction_agent):
      correction_agent.correct(program_id, code, validation, analysis=, prior_code=, history=)
  returns a CorrectionResult whose repair_prompt is handed to the Translator. history items are
  {"program_id", "rank"}.
- Translator (not yet packaged): takes a prompt and returns the whole script in one Python code block.
  Until it is packaged, graph.py uses its own model call (llm_repairer).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class MissingAgent(RuntimeError):
    """A teammate's agent package is not on this branch."""


@dataclass(frozen=True)
class ValidationView:
    """The fields of a validation result the harness uses. All are redacted: no data values."""

    next_state: str           # "PASSED" or "TRIAGED"
    rank: tuple               # lower is better (D-037): stage, failing checks, mismatches, largest error
    fix_request: str | None   # set when next_state is "TRIAGED"

    @property
    def passed(self) -> bool:
        return self.next_state == "PASSED"

    @classmethod
    def of(cls, result: Any) -> "ValidationView":
        """Read the three fields from a ValidationResult, its model_dump(), or any object with them."""
        def get(name: str):
            return result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)

        next_state, rank = get("next_state"), get("rank")
        if next_state not in ("PASSED", "TRIAGED"):
            raise ValueError(f"validation next_state must be PASSED or TRIAGED, not {next_state!r}")
        if rank is None:
            raise ValueError("validation result has no rank")
        fix_request = get("fix_request")
        if next_state == "TRIAGED" and not fix_request:
            raise ValueError("a TRIAGED validation result must carry a fix_request")
        return cls(next_state, tuple(rank), fix_request)

    def as_feedback(self, program_id: str) -> dict:
        """What the Correction agent accepts in place of a ValidationResult."""
        return {"program_id": program_id, "next_state": self.next_state,
                "rank": list(self.rank), "fix_request": self.fix_request}


class Validator(Protocol):
    def __call__(self, program_id: str, code: str, attempt_dir: Path) -> Any:
        """Check, run, and compare the script; return a result ValidationView.of() can read."""
        ...


class Corrector(Protocol):
    def __call__(
        self, program_id: str, code: str, validation: dict, *,
        analysis: str, prior_code: dict[str, str], history: list[dict],
    ) -> str:
        """Return the repair prompt for the Translator. Never returns code."""
        ...


class Repairer(Protocol):
    def __call__(self, prompt: str) -> str:
        """Return the whole repaired script. Raise ValueError if the reply contains no Python code block."""
        ...


def _import(package: str, branch: str):
    """Import a teammate's package, or raise MissingAgent naming the branch it lives on."""
    import importlib

    try:
        return importlib.import_module(package)
    except ImportError as e:
        if (e.name or "").split(".")[0] != package.split(".")[0]:
            raise  # the package exists but one of its own imports failed
        raise MissingAgent(f"{package} is not on this branch; it is on branch {branch}.") from None


def validation_agent_validator(review: bool = True) -> Validator:
    """The validation agent as a Validator. review=False skips its LLM code review."""
    build_graph = _import("validation_agent.graph", "validation_agent").build_graph
    app = build_graph(review=review)

    def validate(program_id: str, code: str, attempt_dir: Path) -> Any:
        state = app.invoke({"program_id": program_id, "code": code, "attempt_dir": str(attempt_dir)})
        return state["result"]

    return validate


def correction_agent_corrector() -> Corrector:
    """The correction agent as a Corrector."""
    correct = _import("correction_agent", "feature/correction_agent").correct

    def corrector(program_id, code, validation, *, analysis, prior_code, history) -> str:
        return correct(program_id, code, validation, analysis=analysis,
                       prior_code=prior_code, history=history).repair_prompt

    return corrector
