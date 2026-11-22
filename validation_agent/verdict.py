"""Step 5: the verdict, the rank and the fix request (no LLM)."""

from pathlib import Path

from . import config
from .schema import Finding, RedactedReport, ValidationResult


def tool_findings(violations: list[str], exception: dict | None, comparison: RedactedReport | None) -> str:
    """What the tools found, from the first stage that failed; "" if nothing. Never an error message or a data value."""
    if violations:
        return "The script breaks these static-check rules (it was not run):\n" + "\n".join(f"- {v}" for v in violations)
    if exception:
        if exception["type"] == "Timeout":
            return "The script did not finish within the time limit."
        if exception["type"] == "ExitCode":
            return "The script exited with an error status but without a Python traceback."
        lines = ["The script failed when it was run.", f"- error type: {exception['type']}"]
        if exception["line"]:
            lines += [f"- line: {exception['line']}", f"- code line: `{exception['source']}`"]
        return "\n".join(lines)
    if comparison is None or not comparison.checks:
        return ""
    rows = [f"| {c.dataset} | {c.column or '-'} | {c.check} | {c.mismatch_count} | "
            f"{'-' if c.max_error is None else f'{c.max_error:.3g}'} | {c.detail or '-'} |" for c in comparison.checks]
    return "\n".join(["The script ran, but these checks against the SAS outputs failed:", "",
                      "| dataset | column | check | mismatches | max error | detail |",
                      "|---|---|---|---|---|---|", *rows])


def build_fix_request(findings: str, review: list[Finding]) -> str:
    """Item G: the tool findings from the first failing stage, then the review findings."""
    parts = [findings]
    if review:
        parts.append("Code review findings (from reading the code against the SAS program):\n" + "\n".join(
            f"- [{f.severity}] {f.kind}, Python {f.python_lines or '-'}, SAS {f.sas_lines or '-'}: {f.problem}"
            + (f" Check: {f.what_to_check}" if f.what_to_check else "") for f in review))
    return "\n\n".join(p for p in parts if p)


def make_result(program_id: str, events: list[str], violations: list[str], exception: dict | None,
                comparison: RedactedReport | None, review: list[Finding]) -> ValidationResult:
    """PASSED only if every output matches SAS and no review finding is major (REVIEW_BLOCKS)."""
    stage = "static_check" if violations else "compare" if comparison else "execute"
    blocking_review = config.REVIEW_BLOCKS and any(f.severity == "major" for f in review)
    passed = bool(comparison and comparison.passed) and not blocking_review
    failing = comparison.failing_checks if comparison else max(len(violations), 1)
    return ValidationResult(
        program_id=program_id,
        next_state="PASSED" if passed else "TRIAGED",
        events=events + ["ALL_PASS" if passed else "FAIL"],
        stage_reached=stage,
        rank=(-config.STAGES.index(stage), failing, comparison.total_mismatches if comparison else 0,
              (comparison.max_error or 0.0) if comparison else 0.0),
        violations=violations,
        exception=exception,
        comparison=comparison,
        review=review,
        fix_request=None if passed else build_fix_request(tool_findings(violations, exception, comparison), review),
    )


def write_validation(attempt_dir: str | Path, result: ValidationResult) -> Path:
    """<attempt_dir>/ai/validation.json: what the Correction agent may read. No data values."""
    ai = Path(attempt_dir) / "ai"
    ai.mkdir(parents=True, exist_ok=True)
    path = ai / "validation.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path
