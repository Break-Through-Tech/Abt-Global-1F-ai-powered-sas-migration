"""Validation agent: static_check -> execute -> compare -> review -> verdict  (First Agent Brief, items E, F, G).

Input:  program_id and code (the Translator's script: one Python code block).
Output: a ValidationResult with the next state (PASSED or TRIAGED), one event per stage, the attempt's
        rank, the redacted comparison report, the code review findings and the fix request for the Corrector.
Only `review` uses an LLM (Gemini by default). It sees the SAS, the script and the redacted findings, never data values.
Files:  <attempt_dir>/ai   what a model may see (script, result)
        <attempt_dir>/data data values, never shown to a model (outputs, stdout, stderr)
"""

import ast
import json
import operator
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import numpy as np
import pandas as pd
import pyreadstat
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from sas_agent import prompts as translator_prompts   # read-only: the runtime contract text

from . import config, prompts

STAGES = ["static_check", "execute", "compare"]


class Check(BaseModel):
    """One failing comparison check: names and counts only, no data values."""
    dataset: str                     # "OUTCOME_MORTALITY"
    column: str = ""                 # "grp_score"
    check: str                       # file_missing, file_name, columns, row_count, row_order, missing_values, values
    mismatch_count: int = 0
    max_error: float | None = None   # largest absolute difference (numeric values)


class RedactedReport(BaseModel):
    """What compare() returns (brief item I)."""
    program_id: str
    passed: bool
    failing_checks: int
    total_mismatches: int
    max_error: float | None
    checks: list[Check]


class Finding(BaseModel):
    """One code review finding from the LLM."""
    kind: Literal["hardcoded", "quiet_bug", "cause"]   # only works for this data / silent logic error / cause of a failing check
    severity: Literal["major", "minor"]
    python_lines: str                        # "L80-95"
    sas_lines: str                           # "L120-135"
    problem: str
    what_to_check: str                       # never replacement code


class Review(BaseModel):
    findings: list[Finding]


class ValidationResult(BaseModel):
    program_id: str
    next_state: Literal["PASSED", "TRIAGED"]
    events: list[str]                        # e.g. ["STATIC_PASS", "EXEC_OK", "COMPARED_DONE", "REVIEW_DONE", "FAIL"]
    stage_reached: str                       # static_check, execute or compare
    rank: tuple[int, int, int, float]        # D-037: stage, failing checks, mismatches, largest error; lower is better
    violations: list[str]                    # static-check rules broken
    exception: dict | None                   # {"type", "line", "source"}, never the message
    comparison: RedactedReport | None
    review: list[Finding]                    # LLM code review (empty when the review is off)
    fix_request: str | None                  # for {fix_request} in sas_agent/prompts/fix.md


class State(TypedDict, total=False):
    program_id: str
    code: str
    attempt_dir: str
    violations: list[str]
    exception: dict | None
    comparison: RedactedReport
    review: list[Finding]
    events: Annotated[list[str], operator.add]
    result: ValidationResult


def build_llm() -> BaseChatModel:
    """Same setup as the translator: a "provider:model" string; the API key comes from .env."""
    return init_chat_model(config.MODEL, timeout=900)


def build_graph(llm: BaseChatModel | None = None, review: bool = True):
    """review=False skips the LLM node (no API calls)."""
    llm = (llm or build_llm()) if review else None
    macros = config.MACROS_FILE.read_text(errors="replace")

    def static_check(state: State) -> dict:
        """TRANSLATED -> CHECKED: the script parses and imports only pandas, numpy and the standard library."""
        attempt = Path(state.get("attempt_dir") or config.RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S") / state["program_id"])
        for zone in ("ai", "data/outputs"):
            (attempt / zone).mkdir(parents=True, exist_ok=True)
        (attempt / "ai" / "script.py").write_text(state["code"], encoding="utf-8")
        violations = check_code(state["code"])
        return {"attempt_dir": str(attempt), "violations": violations, "events": ["STATIC_FAIL" if violations else "STATIC_PASS"]}

    def execute(state: State) -> dict:
        """CHECKED -> EXECUTED: run the script like the runtime contract says."""
        program = config.get_program(state["program_id"])
        attempt = Path(state["attempt_dir"])
        for earlier in config.PROGRAMS[:config.PROGRAMS.index(program)]:      # outputs of earlier programs
            for name in earlier.outputs:
                shutil.copy(config.SAS_CSV_DIR / f"{name}.csv", attempt / "data" / "outputs")
        status, exception = run_script(attempt)
        return {"exception": exception, "events": [status]}

    def compare_node(state: State) -> dict:
        """EXECUTED -> COMPARED: every required output vs the SAS dataset."""
        report = compare(state["program_id"], Path(state["attempt_dir"]) / "data" / "outputs")
        return {"comparison": report, "events": ["COMPARED_DONE"]}

    def review_node(state: State) -> dict:
        """LLM code review: hard-coding, quiet bugs, and the cause of each failing check."""
        if llm is None:
            return {"review": []}
        program = config.get_program(state["program_id"])
        prompt = prompts.REVIEW.format(
            contract=translator_prompts.RUNTIME_CONTRACT.format(outputs=", ".join(program.outputs)),
            tool_findings=tool_findings(state) or "None: the script runs and every output matches SAS.",
            macros=macros, sas_file=program.sas_file, sas_code=program.path.read_text(errors="replace"),
            code="\n".join(f"{n:4} | {line}" for n, line in enumerate(state["code"].splitlines(), 1)),
        )
        result = llm.with_structured_output(Review).invoke([SystemMessage(prompts.SYSTEM), HumanMessage(prompt)])
        return {"review": result.findings if result else [], "events": ["REVIEW_DONE"]}

    def verdict(state: State) -> dict:
        """COMPARED -> PASSED, or any failure (a check, or a major review finding) -> TRIAGED with a fix request."""
        comparison, review = state.get("comparison"), state.get("review", [])
        stage = "compare" if comparison else "execute" if "exception" in state else "static_check"
        blocking_review = config.REVIEW_BLOCKS and any(f.severity == "major" for f in review)
        passed = bool(comparison and comparison.passed) and not blocking_review
        failing = comparison.failing_checks if comparison else max(len(state["violations"]), 1)
        result = ValidationResult(
            program_id=state["program_id"], next_state="PASSED" if passed else "TRIAGED",
            events=state["events"] + ["ALL_PASS" if passed else "FAIL"], stage_reached=stage,
            rank=(-STAGES.index(stage), failing, comparison.total_mismatches if comparison else 0,
                  (comparison.max_error or 0.0) if comparison else 0.0),
            violations=state["violations"], exception=state.get("exception"), comparison=comparison,
            review=review, fix_request=None if passed else fix_request(state),
        )
        (Path(state["attempt_dir"]) / "ai" / "validation.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return {"result": result}

    graph = StateGraph(State)
    graph.add_node("static_check", static_check)
    graph.add_node("execute", execute)
    graph.add_node("compare", compare_node)
    graph.add_node("review", review_node)
    graph.add_node("verdict", verdict)
    graph.add_edge(START, "static_check")
    graph.add_conditional_edges("static_check", lambda s: "verdict" if s["violations"] else "execute")
    graph.add_conditional_edges("execute", lambda s: "review" if s["exception"] else "compare")   # a crash is reviewed too
    graph.add_edge("compare", "review")
    graph.add_edge("review", "verdict")
    graph.add_edge("verdict", END)
    return graph.compile()


def check_code(code: str) -> list[str]:
    """Static checks (D-046)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"line {e.lineno}: SyntaxError: {e.msg}"]
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or "."]
            for top in {m.split(".")[0] for m in modules}:
                if top not in config.ALLOWED_PACKAGES and top not in sys.stdlib_module_names:
                    violations.append(f"line {node.lineno}: import of '{top or 'relative'}' is not allowed; "
                                      "use only pandas, numpy and the Python standard library")
    if "SAS_OUTPUT_DIR" not in code:
        violations.append("the script never reads SAS_OUTPUT_DIR (runtime contract)")
    return violations


def run_script(attempt: Path) -> tuple[str, dict | None]:
    """Execute (D-047) in a separate process with only the contract's variables set (no API keys)."""
    data = attempt / "data"
    env = {k: os.environ[k] for k in config.PASSED_THROUGH_ENV if k in os.environ}
    env |= {"SAS_INPUT_DIR": str(config.SAS_INPUT_DIR), "SAS_OUTPUT_DIR": str((data / "outputs").resolve())}
    runner = Path(__file__).with_name("runner.py")
    with open(data / "stdout.txt", "w") as out, open(data / "stderr.txt", "w") as err:
        try:
            code = subprocess.run([sys.executable, "-I", str(runner), str((attempt / "ai" / "script.py").resolve()),
                                   str(data / "exception.json")], cwd=data, env=env, stdout=out, stderr=err,
                                  timeout=config.TIMEOUT_S, check=False).returncode
        except subprocess.TimeoutExpired:
            return "TIMEOUT", {"type": "Timeout", "line": None, "source": None}
    if code == 0:
        return "EXEC_OK", None
    report = data / "exception.json"
    return "EXEC_ERROR", json.loads(report.read_text()) if report.exists() else {"type": "ProcessDied", "line": None, "source": None}


def compare(program_id: str, output_dir: Path) -> RedactedReport:
    """Comparator (item I): each required output CSV vs its SAS .sas7bdat."""
    files = {f.name.lower(): f for f in output_dir.iterdir()}
    checks = []
    for name in config.get_program(program_id).outputs:
        found = files.get(f"{name}.csv".lower())
        if found is None:
            checks.append(Check(dataset=name, check="file_missing", mismatch_count=1))
            continue
        if found.name != f"{name}.csv":
            checks.append(Check(dataset=name, check="file_name", mismatch_count=1))   # e.g. .CSV fails on Linux
        py = pd.read_csv(found, dtype={config.ID_COLUMN: str}, float_precision="round_trip")   # D-049
        sas, _ = pyreadstat.read_sas7bdat(str(config.SAS_DATASET_DIR / f"{name.lower()}.sas7bdat"))
        checks += compare_dataset(name, py, sas)
    errors = [c.max_error for c in checks if c.max_error is not None]
    return RedactedReport(program_id=program_id, passed=not checks, failing_checks=len(checks),
                          total_mismatches=sum(c.mismatch_count for c in checks),
                          max_error=max(errors) if errors else None, checks=checks)


def compare_dataset(name: str, py: pd.DataFrame, sas: pd.DataFrame) -> list[Check]:
    """Columns, row count, row order, then every column's missing values and values."""
    if list(py.columns) != list(sas.columns):              # names and order must match SAS
        differ = sum(a != b for a, b in zip(py.columns, sas.columns, strict=False)) + abs(len(py.columns) - len(sas.columns))
        return [Check(dataset=name, check="columns", mismatch_count=differ)]
    if len(py) != len(sas):
        return [Check(dataset=name, check="row_count", mismatch_count=abs(len(py) - len(sas)))]
    key = config.ID_COLUMN
    if key in sas.columns and (wrong := (py[key].astype(str).to_numpy() != sas[key].astype(str).to_numpy()).sum()):
        return [Check(dataset=name, column=key, check="row_order", mismatch_count=int(wrong))]

    checks = []
    for col in sas.columns:
        a, e = py[col], sas[col]
        if not pd.api.types.is_numeric_dtype(e):           # text: a blank equals missing (S-20)
            a, e = (s.astype("string").str.strip().replace("", pd.NA) for s in (a, e))
        a_na, e_na = a.isna().to_numpy(), e.isna().to_numpy()
        if (a_na != e_na).any():
            checks.append(Check(dataset=name, column=col, check="missing_values", mismatch_count=int((a_na != e_na).sum())))
        both = ~a_na & ~e_na
        if pd.api.types.is_numeric_dtype(e) and pd.api.types.is_numeric_dtype(a):
            av, ev = a.to_numpy(float), e.to_numpy(float)
            bad = both & ~np.isclose(av, ev, rtol=config.RTOL, atol=config.ATOL)
            max_error = float(np.abs(av - ev)[bad].max()) if bad.any() else None
        else:
            bad, max_error = both & (a.astype(str).to_numpy() != e.astype(str).to_numpy()), None
        if bad.any():
            checks.append(Check(dataset=name, column=col, check="values", mismatch_count=int(bad.sum()), max_error=max_error))
    return checks


def tool_findings(state: State) -> str:
    """What the tools found, from the first stage that failed; "" if nothing. Never an error message or a data value."""
    if state["violations"]:
        return "The script breaks these static-check rules:\n" + "\n".join(f"- {v}" for v in state["violations"])
    if exc := state.get("exception"):
        lines = ["The script failed when it was run.", f"- error type: {exc['type']}"]
        if exc["line"]:
            lines += [f"- line: {exc['line']}", f"- code line: `{exc['source']}`"]
        return "\n".join(lines)
    if not state["comparison"].checks:
        return ""
    rows = [f"| {c.dataset} | {c.column or '-'} | {c.check} | {c.mismatch_count} | "
            f"{'-' if c.max_error is None else f'{c.max_error:.3g}'} |" for c in state["comparison"].checks]
    return "\n".join(["The script ran, but these checks against the SAS outputs failed:", "",
                      "| dataset | column | check | mismatches | max error |", "|---|---|---|---|---|", *rows])


def fix_request(state: State) -> str:
    """Item G: the tool findings, then the code review findings."""
    parts = [tool_findings(state)]
    if review := state.get("review"):
        parts.append("Code review findings (from reading the code against the SAS program):\n" + "\n".join(
            f"- [{f.severity}] {f.kind}, Python {f.python_lines or '-'}, SAS {f.sas_lines or '-'}: {f.problem}"
            + (f" Check: {f.what_to_check}" if f.what_to_check else "") for f in review))
    return "\n\n".join(p for p in parts if p)
