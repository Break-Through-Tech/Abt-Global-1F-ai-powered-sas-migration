"""The Validation agent's output contract (Validation Agent Brief v1, item B).

Everything here may be shown to a model: names, counts, code lines and differences, never data values or
error messages.
"""

from typing import Literal

from pydantic import BaseModel


class Check(BaseModel):
    """One failing comparison check."""
    dataset: str                     # output name, e.g. "OUTCOME_MORTALITY"
    column: str = ""                 # e.g. "grp_score"
    check: str                       # file_missing, unreadable, columns, row_count, row_order, missing_values, values
    mismatch_count: int = 0
    max_error: float | None = None   # largest absolute difference (numeric values)
    detail: str = ""                 # column names, row counts or file names; never data values


class RedactedReport(BaseModel):
    """What compare returns."""
    program_id: str
    passed: bool
    failing_checks: int
    total_mismatches: int
    max_error: float | None
    checks: list[Check]


class Finding(BaseModel):
    """One LLM code review finding."""
    kind: Literal["hardcoded", "quiet_bug", "cause"]   # only works for this data / silent logic error / cause of a failing check
    severity: Literal["major", "minor"]                # major: results are or could be wrong
    python_lines: str                                  # e.g. "L5-7"
    sas_lines: str                                     # e.g. "program L40" or "macros L120"
    problem: str
    what_to_check: str                                 # what to inspect; never replacement code


class Review(BaseModel):
    findings: list[Finding]


class ValidationResult(BaseModel):
    program_id: str
    next_state: Literal["PASSED", "TRIAGED"]
    events: list[str]                        # e.g. ["STATIC_PASS", "EXEC_OK", "COMPARED_DONE", "REVIEW_DONE", "FAIL"]
    stage_reached: str                       # static_check, execute or compare
    rank: tuple[int, int, int, float]        # D-037: stage, failing checks, mismatches, largest error; lower is better
    violations: list[str]                    # static-check rules broken
    exception: dict | None                   # {"type", "line", "source"} if the script failed; never the message
    comparison: RedactedReport | None        # None if the script was not run or failed
    review: list[Finding]                    # LLM code review (empty when the review is off or skipped)
    fix_request: str | None                  # for {fix_request} in sas_agent/prompts/fix.md; None when PASSED
