"""Validation Agent (LangGraph): checks one SAS unit translated by the Translator.

    static_check ─► execute ─► compare ─► review ─► verdict
         │             │ crash ─────────▲          ▲
         └─ rule broken ┴───────────────┴──────────┘ (code not run, review skipped)

Input (state keys; the first three come from the Translator, the rest from the orchestrator):
    translator_input  the Translator's five inputs: specification, sas_source, upstream_signatures,
                      behavior_notes, project_conventions
    translation       the Translator's output: code, checkpoints, conflicts, assumptions, helpers_requested
    test_data         {"inputs": {parameter: value}, "expected": {output name: SAS DataFrame}}
    program_id        optional: the id the Correction Agent checks; default specification.unit.id
    attempt_dir       optional: if given, writes <attempt_dir>/ai/script.py and ai/validation.json

Output (state key `result`): a ValidationResult with the fields of Validation Agent Brief v1, item B:
    program_id, next_state (PASSED or TRIAGED), events, stage_reached, rank, violations, exception,
    comparison, review, fix_request.
    TRIAGED goes to the Correction Agent: correct(program_id, code, result) reads program_id,
    next_state and fix_request.

Pipeline wiring (bottom of this file): build_graph() is the validator node, route_after_validation is
the edge after it (TRIAGED -> "correct", PASSED -> END), and correction_input(state) gives the
arguments of correction_agent.correct().

Only `review` uses an LLM (Gemini through LangChain). It sees the code, the SAS, the specification and the
redacted tool findings (names and counts), never data values or error messages. Everything else is plain Python.

The Gemini key is read from validation_agent/.env (GEMINI_API_KEY or GOOGLE_API_KEY), never from code.
"""

import ast
import json
import operator
import os
import traceback
import warnings
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

# Load the API key from validation_agent/.env, like sas_agent/config.py does with <repo>/.env.
# Variables already set in the environment take precedence over the file.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Same model string and env override style as sas_agent (any init_chat_model "provider:model" works).
MODEL = os.getenv("VALIDATION_AGENT_MODEL", "google_genai:gemini-3.8-flash")
REVIEW_BLOCKS = True                      # a major review finding fails the attempt even if every check passes
RTOL, ATOL = 1e-6, 1e-8                   # numeric tolerance when comparing with SAS
ID_COLUMN = "PROVIDER_ID"                 # row order is checked on this column when an output has it
STAGES = ["static_check", "execute", "compare"]

TRANSLATION_RESULT_FIELDS = (             # the Translator's output contract
    "code",
    "checkpoints",
    "conflicts",
    "assumptions",
    "helpers_requested",
)


# ---------------------------------------------------------------------------------------------
# Output contract (Validation Agent Brief v1, item B)
# ---------------------------------------------------------------------------------------------

class Check(BaseModel):
    """One failing comparison check: names and counts only, no data values."""
    dataset: str                     # output name, e.g. "out" or "OUTCOME_MORTALITY"
    column: str = ""                 # e.g. "grp_score"
    check: str                       # file_missing, columns, row_count, row_order, missing_values, values
    mismatch_count: int = 0
    max_error: float | None = None   # largest absolute difference (numeric values)


class RedactedReport(BaseModel):
    """What compare returns (First Agent Brief, item I)."""
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
    sas_lines: str                                     # e.g. "L4"
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
    exception: dict | None                   # {"type", "line", "source"} if the code crashed; never the message
    comparison: RedactedReport | None        # None if the code was not run or crashed
    review: list[Finding]                    # LLM code review (empty when the review is off or skipped)
    fix_request: str | None                  # for {fix_request} in sas_agent/prompts/fix.md; None when PASSED


# ---------------------------------------------------------------------------------------------
# Step 1: static checks (no LLM, the code is not run)
# ---------------------------------------------------------------------------------------------

ALLOWED_IMPORTS = {
    "pandas", "numpy", "math", "statistics", "decimal", "datetime", "collections",
    "itertools", "functools", "operator", "re", "typing", "dataclasses", "enum",
}
BLOCKED_CALLS = {
    "open", "print", "input", "eval", "exec", "compile", "__import__", "getattr", "globals", "breakpoint",
}
BLOCKED_METHODS = {    # pandas / numpy file reads and writes
    "read_csv", "read_excel", "read_parquet", "read_json", "read_pickle", "read_sql", "read_sas",
    "to_csv", "to_excel", "to_parquet", "to_json", "to_pickle", "to_sql", "tofile", "save", "load",
}


def check_code(translation: dict) -> list[str]:
    """Static checks; returns the rules broken (empty list: the code may be run).

    Output format (the five fields), syntax, one top-level function, imports (D-046), no file I/O,
    printing or dynamic code, and the declared checkpoints in order. A basic safety gate, not a sandbox.
    """
    if not isinstance(translation, dict):
        return ["output format: the translation must be one object with the five fields"]
    problems = []
    missing = set(TRANSLATION_RESULT_FIELDS) - translation.keys()
    extra = translation.keys() - set(TRANSLATION_RESULT_FIELDS)
    if missing or extra:
        problems.append(f"output format: missing fields {sorted(missing)}, extra fields {sorted(extra)}")
    if not isinstance(translation.get("code"), str) or not translation["code"].strip():
        problems.append("output format: code must be a non-empty string")
    if any(not isinstance(translation.get(field), list) for field in TRANSLATION_RESULT_FIELDS[1:]):
        problems.append("output format: checkpoints, conflicts, assumptions and helpers_requested must be lists")
    if problems:
        return problems

    try:
        tree = ast.parse(translation["code"])
    except SyntaxError as error:
        return [f"line {error.lineno}: SyntaxError: {error.msg}"]

    violations = []
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        violations.append(f"expected one top-level function per unit, found {functions}")

    checkpoint_calls = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                modules = ["." * node.level + (node.module or "")]
            violations += [
                f"line {node.lineno}: import of '{m}' is not allowed; use only pandas, numpy and "
                f"the approved standard library modules"
                for m in modules if m.split(".")[0] not in ALLOWED_IMPORTS
            ]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):         # f(...)
            if node.func.id in BLOCKED_CALLS:
                violations.append(f"line {node.lineno}: {node.func.id}() is not allowed in a unit "
                                  "(no file I/O, printing or dynamic code)")
            if node.func.id == "checkpoint" and node.args and isinstance(node.args[0], ast.Constant):
                checkpoint_calls.append((node.lineno, node.col_offset, node.args[0].value))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):    # obj.f(...)
            if node.func.attr in BLOCKED_METHODS:
                violations.append(f"line {node.lineno}: .{node.func.attr}() is not allowed in a unit "
                                  "(the executor does file input and output)")

    called = [name for _, _, name in sorted(checkpoint_calls)]    # in source order
    if called != translation["checkpoints"]:
        violations.append(f"checkpoints: declared {translation['checkpoints']}, the code calls {called} "
                          "(names and order must match)")
    return violations


def open_issues(translation: dict) -> str:
    """The Translator's conflicts and requested helpers: the unit is not finished. Doesn't stop the code from running."""
    if not isinstance(translation, dict):
        return ""
    conflicts = translation.get("conflicts") or []
    helpers = translation.get("helpers_requested") or []
    if not conflicts and not helpers:
        return ""
    lines = ["The Translator reported open issues, so the unit is not finished:"]
    if conflicts:
        lines.append(f"- {len(conflicts)} conflicts between the specification and the SAS source")
    if helpers:
        lines.append(f"- helpers requested: {helpers}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Step 2: execute the function (no LLM)
# ---------------------------------------------------------------------------------------------

def run_function(code: str, inputs: dict) -> tuple[dict | None, dict | None]:
    """Calls the translated function with `inputs` (by parameter name); returns (outputs, exception).

    - checkpoint(name, df) is the pipeline's checkpoint call; here it does nothing.
    - pandas' ChainedAssignmentError is raised as an error (D-047): in pandas 3 a chained assignment
      (df[col][rows] = value) silently does nothing, so the function would finish with wrong values.
    - exception is {"type", "line", "source"}: the error type and the code line, never the message,
      which can contain data values (e.g. KeyError: '010001').
    """
    function_name = next(node.name for node in ast.parse(code).body if isinstance(node, ast.FunctionDef))
    namespace = {"checkpoint": lambda name, df: None}
    inputs = {key: value.copy() if isinstance(value, pd.DataFrame) else value for key, value in inputs.items()}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ChainedAssignmentError)
            exec(compile(code, "<translation>", "exec"), namespace)
            outputs = namespace[function_name](**inputs)
    except Exception as error:
        frames = [f for f in traceback.extract_tb(error.__traceback__) if f.filename == "<translation>"]
        line = frames[-1].lineno if frames else None
        source = code.splitlines()[line - 1].strip() if line else None
        return None, {"type": type(error).__name__, "line": line, "source": source}

    if not isinstance(outputs, dict) or not all(isinstance(df, pd.DataFrame) for df in outputs.values()):
        return None, {"type": "InvalidReturn", "line": None, "source": None}
    return outputs, None


# ---------------------------------------------------------------------------------------------
# Step 3: compare with SAS (no LLM)
# ---------------------------------------------------------------------------------------------

def compare_dataset(name: str, py: pd.DataFrame, sas: pd.DataFrame) -> list[Check]:
    """Columns, row count, row order, then every column's missing values and values."""
    py, sas = py.reset_index(drop=True), sas.reset_index(drop=True)
    if list(py.columns) != list(sas.columns):              # names and order must match SAS
        differ = sum(a != b for a, b in zip(py.columns, sas.columns, strict=False)) + abs(len(py.columns) - len(sas.columns))
        return [Check(dataset=name, check="columns", mismatch_count=int(differ))]
    if len(py) != len(sas):
        return [Check(dataset=name, check="row_count", mismatch_count=abs(len(py) - len(sas)))]
    if ID_COLUMN in sas.columns and (wrong := (py[ID_COLUMN].astype(str).to_numpy() != sas[ID_COLUMN].astype(str).to_numpy()).sum()):
        return [Check(dataset=name, column=ID_COLUMN, check="row_order", mismatch_count=int(wrong))]

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
            bad = both & ~np.isclose(av, ev, rtol=RTOL, atol=ATOL)
            max_error = float(np.abs(av - ev)[bad].max()) if bad.any() else None
        else:
            bad, max_error = both & (a.astype(str).to_numpy() != e.astype(str).to_numpy()), None
        if bad.any():
            checks.append(Check(dataset=name, column=col, check="values", mismatch_count=int(bad.sum()), max_error=max_error))
    return checks


def compare_outputs(program_id: str, outputs: dict, expected: dict) -> RedactedReport:
    """Comparator: each expected output vs the DataFrame the function returned under that name."""
    checks = []
    for name, sas in expected.items():
        py = outputs.get(name)
        if py is None:
            checks.append(Check(dataset=name, check="file_missing", mismatch_count=1))
            continue
        checks += compare_dataset(name, py, sas)
    errors = [c.max_error for c in checks if c.max_error is not None]
    return RedactedReport(program_id=program_id, passed=not checks, failing_checks=len(checks),
                          total_mismatches=sum(c.mismatch_count for c in checks),
                          max_error=max(errors) if errors else None, checks=checks)


def tool_findings(violations: list[str], exception: dict | None, comparison: RedactedReport | None) -> str:
    """What the tools found, from the first stage that failed; "" if nothing. Never an error message or a data value."""
    if violations:
        return "The function breaks these static-check rules:\n" + "\n".join(f"- {v}" for v in violations)
    if exception:
        if exception["type"] == "InvalidReturn":
            return "The function ran, but it must return a dict of named DataFrames."
        lines = ["The function failed when it was run.", f"- error type: {exception['type']}"]
        if exception["line"]:
            lines += [f"- line: {exception['line']}", f"- code line: `{exception['source']}`"]
        return "\n".join(lines)
    if comparison is None or not comparison.checks:
        return ""
    rows = [f"| {c.dataset} | {c.column or '-'} | {c.check} | {c.mismatch_count} | "
            f"{'-' if c.max_error is None else f'{c.max_error:.3g}'} |" for c in comparison.checks]
    return "\n".join(["The function ran, but these checks against the SAS outputs failed:", "",
                      "| dataset | column | check | mismatches | max error |", "|---|---|---|---|---|", *rows])


# ---------------------------------------------------------------------------------------------
# Step 4: LLM code review (Gemini through LangChain)
# ---------------------------------------------------------------------------------------------

REVIEW_SYSTEM = """
You are a senior SAS 9.4 and Python (pandas/numpy) reviewer checking a translation of one unit of the
CMS Overall Hospital Quality Star Rating SAS pack. You find problems; you never rewrite the code.

The Python must behave like the SAS on ANY quarter's data, not only on the data it was tested with.
Watch for SAS semantics that differ from pandas defaults:
- missing numeric values propagate in arithmetic but are ignored by SAS functions like MEAN/SUM/N;
- PROC MEANS / PROC STANDARD use the sample standard deviation (N-1) and skip missing values;
- PROC TABULATE N counts non-missing values;
- PROC TRANSPOSE, BY-group processing, RETAIN, FIRST./LAST., MERGE (match-merge by key);
- WHERE / IF filters, sort order, and the order of rows and columns in outputs;
- macro variables built from data (SELECT ... INTO:, CALL SYMPUTX) must be computed from data in Python too;
- PROC FASTCLUS (k-means) seeding and convergence rules.
"""

REVIEW_PROMPT = """
Review the Python translation of the SAS unit below, step by step against the SAS. Report three kinds of findings:

1. `hardcoded`: the Python only works for this data. Examples: a value, count, list, threshold or column set
   that SAS computes from the data at run time but the Python writes as a literal; logic that skips a SAS
   branch because it happens to be empty now; assumptions about the number of rows, hospitals or measures.
   Lists that SAS itself writes as literals in its code are fine.
2. `quiet_bug`: logic that differs from SAS but raises no error and may still match the current outputs,
   e.g. wrong missing-value handling, ddof, sort or merge semantics, filters, rounding, edge cases such as
   an empty group.
3. `cause`: for each failing check or error in the tool findings below, the Python line(s) that cause it.

Rules:
- The tool findings are exact (the code was run and compared with the real SAS outputs). Do not contradict them.
- Fill `python_lines` (e.g. "L5-7") and `sas_lines` (e.g. "L4") for every finding.
- `severity`: "major" if results are or would be wrong, "minor" otherwise.
- `what_to_check`: what to inspect and the expected behavior. Never write replacement code.
- No style or naming comments. If there is nothing to report, return an empty list.

=== Tool findings ===
{tool_findings}

=== Translator's assumptions ===
{assumptions}

=== Specification, signatures, behavior notes and project conventions ===
{context}

=== SAS source (with line numbers) ===
{sas}

=== Python translation (with line numbers) ===
{code}
"""


def build_llm(model: str = MODEL) -> BaseChatModel:
    """Gemini through LangChain, built the same way as sas_agent.graph.build_llm.

    - The key is not passed here: ChatGoogleGenerativeAI reads GOOGLE_API_KEY or GEMINI_API_KEY from
      the environment, which load_dotenv() above fills from validation_agent/.env.
    - Temperature is left unset (Google's guidance for Gemini 3+), as in sas_agent.
    - Creating the object makes no network call; a bad key only fails on the first review.
    """
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in validation_agent/.env.")
    return init_chat_model(model, timeout=900)


def number_lines(text: str) -> str:
    return "\n".join(f"{n:4} | {line}" for n, line in enumerate(text.splitlines(), 1))


def llm_review(llm: BaseChatModel, translator_input: dict, translation: dict, findings: str) -> list[Finding]:
    """Gemini reads the SAS source and the numbered code, given the tool findings (redacted text)."""
    context = {key: value for key, value in translator_input.items() if key != "sas_source"}
    prompt = REVIEW_PROMPT.format(
        tool_findings=findings or "None: the code runs and every output matches SAS.",
        assumptions=json.dumps(translation["assumptions"], indent=2, default=str),
        context=json.dumps(context, indent=2, default=str),
        sas=number_lines(translator_input["sas_source"]),
        code=number_lines(translation["code"]),
    )
    review = llm.with_structured_output(Review).invoke([SystemMessage(REVIEW_SYSTEM), HumanMessage(prompt)])
    if review is None:
        raise RuntimeError("The model returned no review.")
    return review.findings


# ---------------------------------------------------------------------------------------------
# Step 5: verdict, rank and fix request (no LLM)
# ---------------------------------------------------------------------------------------------

def build_fix_request(findings: str, review: list[Finding], issues: str = "") -> str:
    """Item G: the tool findings from the first failing stage, then the review findings, then open issues."""
    parts = [findings]
    if review:
        parts.append("Code review findings (from reading the code against the SAS program):\n" + "\n".join(
            f"- [{f.severity}] {f.kind}, Python {f.python_lines or '-'}, SAS {f.sas_lines or '-'}: {f.problem}"
            + (f" Check: {f.what_to_check}" if f.what_to_check else "") for f in review))
    parts.append(issues)
    return "\n\n".join(p for p in parts if p)


def make_result(
    program_id: str,
    events: list[str],
    violations: list[str],
    exception: dict | None,
    comparison: RedactedReport | None,
    review: list[Finding],
    translation: dict,
) -> ValidationResult:
    """PASSED only if the outputs match SAS, no review finding is major (REVIEW_BLOCKS) and there are no open issues."""
    stage = "static_check" if violations else "compare" if comparison else "execute"
    blocking_review = REVIEW_BLOCKS and any(f.severity == "major" for f in review)
    issues = "" if violations else open_issues(translation)
    passed = bool(comparison and comparison.passed) and not blocking_review and not issues
    failing = comparison.failing_checks if comparison else max(len(violations), 1)
    findings = tool_findings(violations, exception, comparison)
    return ValidationResult(
        program_id=program_id,
        next_state="PASSED" if passed else "TRIAGED",
        events=events + ["ALL_PASS" if passed else "FAIL"],
        stage_reached=stage,
        rank=(-STAGES.index(stage), failing, comparison.total_mismatches if comparison else 0,
              (comparison.max_error or 0.0) if comparison else 0.0),
        violations=violations,
        exception=exception,
        comparison=comparison,
        review=review,
        fix_request=None if passed else build_fix_request(findings, review, issues),
    )


def write_attempt(attempt_dir: str | Path, code: str, result: ValidationResult) -> Path:
    """<attempt_dir>/ai/script.py and ai/validation.json: what the Correction Agent reads. No data values."""
    ai = Path(attempt_dir) / "ai"
    ai.mkdir(parents=True, exist_ok=True)
    (ai / "script.py").write_text(code, encoding="utf-8")
    (ai / "validation.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return ai


# ---------------------------------------------------------------------------------------------
# LangGraph: the nodes and the graph
# ---------------------------------------------------------------------------------------------

class ValidationState(TypedDict, total=False):
    # inputs (from the Translator / orchestrator)
    translator_input: dict
    translation: dict
    test_data: dict
    program_id: str
    attempt_dir: str
    # filled in by the nodes
    violations: list[str]
    outputs: dict                                   # what the function returned (data, never sent to the LLM)
    exception: dict | None
    comparison: RedactedReport
    review: list[Finding]
    events: Annotated[list[str], operator.add]      # every node appends its event
    result: ValidationResult


def get_program_id(state: ValidationState) -> str:
    """state["program_id"], else specification.unit.id, else "unknown"."""
    if state.get("program_id"):
        return state["program_id"]
    spec = (state.get("translator_input") or {}).get("specification") or {}
    unit = spec.get("unit") if isinstance(spec, dict) else None
    return (unit or {}).get("id") or "unknown"


def build_graph(llm: BaseChatModel | None = None, review: bool = True):
    """Compile the validator graph. Same pattern as sas_agent.graph.build_graph.

    - llm=None builds the default Gemini model from .env (like sas_agent); pass any chat model
      to override it, e.g. a fake one for tests.
    - review=False skips the LLM review (manual checks only, no API calls, no key needed).
    - The compiled graph can be added as one node (a subgraph) of a larger pipeline graph:
      parent.add_node("validate", build_graph()). It shares the parent state keys
      translator_input, translation, test_data, program_id, attempt_dir and result; keep its other
      keys (events, violations, ...) out of the parent state so every attempt starts fresh.
    """
    llm = (llm or build_llm()) if review else None

    def static_check(state: ValidationState) -> dict:
        """TRANSLATED / PATCHED -> CHECKED."""
        violations = check_code(state["translation"])
        return {"violations": violations, "events": ["STATIC_FAIL" if violations else "STATIC_PASS"]}

    def execute(state: ValidationState) -> dict:
        """CHECKED -> EXECUTED."""
        outputs, exception = run_function(state["translation"]["code"], state["test_data"]["inputs"])
        return {"outputs": outputs, "exception": exception, "events": ["EXEC_ERROR" if exception else "EXEC_OK"]}

    def compare(state: ValidationState) -> dict:
        """EXECUTED -> COMPARED."""
        report = compare_outputs(get_program_id(state), state["outputs"], state["test_data"]["expected"])
        return {"comparison": report, "events": ["COMPARED_DONE"]}

    def review_node(state: ValidationState) -> dict:
        """LLM code review (part of COMPARED): hard-coding, quiet bugs, the cause of each failing check."""
        if llm is None:
            return {"review": []}
        findings = tool_findings(state["violations"], state.get("exception"), state.get("comparison"))
        found = llm_review(llm, state["translator_input"], state["translation"], findings)
        return {"review": found, "events": ["REVIEW_DONE"]}

    def verdict(state: ValidationState) -> dict:
        """-> PASSED, or TRIAGED with a fix request for the Correction Agent."""
        result = make_result(
            get_program_id(state), state["events"], state["violations"], state.get("exception"),
            state.get("comparison"), state.get("review", []), state["translation"],
        )
        if state.get("attempt_dir"):
            write_attempt(state["attempt_dir"], state["translation"].get("code", ""), result)
        return {"result": result}

    graph = StateGraph(ValidationState)
    graph.add_node("static_check", static_check)
    graph.add_node("execute", execute)
    graph.add_node("compare", compare)
    graph.add_node("review", review_node)
    graph.add_node("verdict", verdict)
    graph.add_edge(START, "static_check")
    graph.add_conditional_edges("static_check", lambda s: "verdict" if s["violations"] else "execute")
    graph.add_conditional_edges("execute", lambda s: "review" if s["exception"] else "compare")   # a crash is reviewed too
    graph.add_edge("compare", "review")
    graph.add_edge("review", "verdict")
    graph.add_edge("verdict", END)
    return graph.compile()


# ---------------------------------------------------------------------------------------------
# Connecting to the Correction Agent (used by the pipeline graph; not wired until that agent is done)
# ---------------------------------------------------------------------------------------------

CORRECTION_NODE = "correct"     # the node name the pipeline gives the Correction Agent


def route_after_validation(state: dict) -> str:
    """Conditional edge after the validator node: TRIAGED -> the Correction Agent, PASSED -> END.

    pipeline.add_conditional_edges("validate", route_after_validation)
    Retry limits and escalation belong to the orchestrator; wrap this function to add them.
    """
    return CORRECTION_NODE if state["result"].next_state == "TRIAGED" else END


def correction_input(state: dict) -> dict:
    """The arguments of correction_agent.correct(program_id, code, validation), taken from the state.

    def correct_node(state):
        repair = correct(**correction_input(state))
        return {"repair_prompt": repair.repair_prompt}
    """
    result = state["result"]
    translation = state.get("translation")
    code = translation.get("code", "") if isinstance(translation, dict) else ""
    return {"program_id": result.program_id, "code": code, "validation": result}


def validate(
    translator_input: dict,
    translation: dict,
    test_data: dict,
    llm: BaseChatModel | None = None,
    review: bool = True,
    program_id: str | None = None,
    attempt_dir: str | Path | None = None,
) -> dict:
    """Run the whole graph once and return the ValidationResult as a dict."""
    state = {"translator_input": translator_input, "translation": translation, "test_data": test_data}
    if program_id:
        state["program_id"] = program_id
    if attempt_dir:
        state["attempt_dir"] = str(attempt_dir)
    return build_graph(llm, review).invoke(state)["result"].model_dump()
