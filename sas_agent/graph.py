"""LangGraph workflows (design doc 6.12, D-038).

Two graphs share the analyze and translate nodes:

- Single-pass baseline (D-043), the skeleton's original flow:
      analyze -> translate -> save
- Fix loop for the October milestone (design doc 6.12.1):
      analyze -> translate -> static_check -> execute -> compare -> promote
  A failure at static_check, execute, or compare goes to fix, which asks the Corrector for a whole new
  script and sends it back to static_check, until the script passes or a limit is reached (escalate).

Ownership (design doc 13.1): the graph, routing, fix requests, and run records belong to the
orchestration subgroup. The comparator (see comparison.py) and the fix prompt (prompts/fix.md) come
from their owners; the graph calls them through fixed interfaces.
"""

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from . import config, executor, prompts, runs, static_check
from .comparison import ERROR_BINS, Comparator

# Enough steps for analyze, translate, and MAX_CORRECTIONS + 1 attempts of four nodes each.
# Pass as config={"recursion_limit": RECURSION_LIMIT} when invoking the fix loop.
RECURSION_LIMIT = 100

# Failing comparison checks listed in one fix request; the rest are summarized as a count.
MAX_LISTED_CHECKS = 30


class MigrationState(TypedDict, total=False):
    # Everything here may reach the model (design doc 7.7): identifiers, paths, code, redacted results,
    # counters. Never DataFrames, data values, full reports, or exception messages.
    program_id: str
    run_id: str
    input_dir: str              # the run's input folder, passed to scripts as SAS_INPUT_DIR
    prior_code: dict[str, str]  # program_id -> translated Python code of earlier programs
    analysis: str
    code: str                   # the current script
    attempt: int                # attempt number of `code`; 1 is the first translation
    event: str                  # result of the last node; route() reads it
    outcome: dict               # how the current attempt failed, for comparing attempts (D-037)
    fix_request: dict           # built by code from the current attempt's failure
    best: dict | None           # best attempt so far: attempt, code, outcome, fix_request
    corrections: int            # Corrector calls so far
    no_improvement: int         # corrections in a row that did not improve on the best attempt
    history: list[dict]         # one entry per failed attempt
    status: str                 # at the end: "PASSED", "ESCALATED", or "BASELINE"
    output_dir: str             # folder holding this program's result


# Takes the prompt context (the placeholders of prompts/fix.md) and returns a whole new script.
Corrector = Callable[[dict], str]


def build_llm() -> BaseChatModel:
    """Create the chat model used by every LLM node.

    Notes:
    - config.MODEL is "provider:model", e.g. "google_genai:gemini-3.8-flash". init_chat_model
      splits it and builds the matching LangChain class (here ChatGoogleGenerativeAI), so changing
      models or providers only means changing that string.
    - The API key is not passed here: ChatGoogleGenerativeAI reads GOOGLE_API_KEY from the
      environment, which config.py fills from .env via load_dotenv().
    - Defaults left as-is: temperature (unset for Gemini 3+, per Google's guidance) and automatic
      retries on rate limits / temporary server errors.
    - Creating the object makes no network call; a bad key or model name only fails on the first
      .invoke(), i.e. inside the analyze node.
    - build_graph(llm=...) can take any other chat model instead (e.g. a fake one for tests).
    """
    # Long prompts + long scripts with thinking can take minutes; don't time out early.
    return init_chat_model(config.MODEL, timeout=900)


# Gemini finish reasons that mean the response is unusable.
_BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}


def _ask(llm: BaseChatModel, prompt: str) -> str:
    response = llm.invoke([SystemMessage(prompts.SYSTEM), HumanMessage(prompt)])
    finish_reason = response.response_metadata.get("finish_reason")
    if finish_reason in _BLOCKED:
        raise RuntimeError(f"Model blocked the response (finish_reason={finish_reason})")
    if finish_reason == "MAX_TOKENS":
        raise RuntimeError("Model hit its output token limit; the response is truncated.")
    return response.text


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        raise ValueError("Model response contained no ```python code block")
    return max(blocks, key=len).strip() + "\n"


def llm_corrector(llm: BaseChatModel) -> Corrector:
    """Corrector that sends prompts/fix.md to the model.

    Notes:
    - The fix prompt is written by the translation team (design doc 6.12.5). This function only fills
      it and extracts the script, so the graph runs once fix.md exists.
    - Placeholders available to fix.md: {sas_file} {sas_code} {macros} {contract} {prior} {analysis}
      {code} {fix_request} {history}.
    """
    def correct(context: dict) -> str:
        try:
            template = prompts.load("fix")
        except FileNotFoundError:
            raise FileNotFoundError(
                "sas_agent/prompts/fix.md is missing. The fix prompt is written by the translation team "
                "(design doc 6.12.5)."
            ) from None
        return extract_python(_ask(llm, template.format(**context)))

    return correct


# ---------------------------------------------------------------------------------------------------
# Routing (design doc 6.4.4 and 6.12.1, D-033): the next node depends only on the current node and the
# event it returned. promote and escalate lead to END.
TRANSITIONS: dict[tuple[str, str], str] = {
    ("translate", "TRANSLATION_OK"): "static_check",
    ("static_check", "STATIC_PASS"): "execute",
    ("static_check", "STATIC_FAIL"): "fix",
    ("execute", "EXEC_OK"): "compare",
    ("execute", "EXEC_ERROR"): "fix",
    ("execute", "TIMEOUT"): "fix",
    ("compare", "ALL_PASS"): "promote",
    ("compare", "FAIL"): "fix",
    ("fix", "CODE_FAULT"): "static_check",
    ("fix", "LIMIT"): "escalate",
}


def route(node: str, state: MigrationState) -> str:
    """Next node after `node`, read from TRANSITIONS. An undefined pair is an error (D-033)."""
    key = (node, state.get("event"))
    if key not in TRANSITIONS:
        raise RuntimeError(f"No transition for node {node!r} on event {key[1]!r}")
    return TRANSITIONS[key]


# ---------------------------------------------------------------------------------------------------
# Comparing attempts (design doc 6.4.7, D-037)
_STAGES = ("static", "execution", "comparison")  # how far an attempt got, worst first


def outcome_key(outcome: dict) -> tuple:
    """Sort key for attempts; larger is better: the stage reached, then fewer failing checks, fewer
    mismatched rows, and a smaller largest difference."""
    return (
        _STAGES.index(outcome["stage"]),
        -outcome.get("failing_checks", 0),
        -outcome.get("mismatches", 0),
        -outcome.get("error_bin", -1),
    )


def _mismatch_count(mismatch: int | str | None) -> int:
    """Mismatched rows as a number; counts from 1 to 10 are reported as "<11" (design doc 6.6)."""
    if mismatch is None:
        return 0
    return 10 if mismatch == "<11" else int(mismatch)


# ---------------------------------------------------------------------------------------------------
# Fix requests: the text the Corrector receives about a failure. Built only from redacted fields, so
# this is part of the privacy gate (design doc 7.2).
def render_fix_request(fix_request: dict) -> str:
    kind = fix_request["kind"]
    if kind == "static":
        lines = ["The script was rejected before it ran:"]
        lines += [f"- line {v['line']}: {v['detail']}" for v in fix_request["violations"]]
    elif kind == "execution":
        if fix_request["status"] == "timeout":
            lines = [f"The script did not finish within {executor.TIMEOUT_SECONDS} seconds."]
        else:
            e = fix_request["exception"]
            where = f" at line {e['line']}: `{e['source']}`" if e.get("line") else ""
            lines = [
                f"The script failed with {e['type']}{where}.",
                "The error message is withheld because it can contain data values.",
            ]
        if fix_request["files_written"]:
            lines.append("Files written before it stopped: " + ", ".join(fix_request["files_written"]))
    else:  # comparison
        lines = ["The script ran, but its outputs do not match the SAS outputs:"]
        for c in fix_request["failed_checks"]:
            where = c["dataset"] + (f", column {c['column']}" if c.get("column") else "")
            detail = [f"check: {c['check']}"]
            if c.get("mismatch") is not None:
                detail.append(f"mismatched rows: {c['mismatch']}")
            if c.get("error_bin"):
                detail.append(f"largest difference: {c['error_bin']}")
            lines.append(f"- {where} ({'; '.join(detail)})")
        if fix_request.get("more"):
            lines.append(f"- and {fix_request['more']} more failing checks")
    return "\n".join(lines)


def render_history(history: list[dict]) -> str:
    def describe(outcome: dict) -> str:
        if outcome["stage"] == "static":
            return f"rejected by static checks ({outcome['failing_checks']} problems)"
        if outcome["stage"] == "execution":
            return "failed when run"
        return f"ran, with {outcome['failing_checks']} failing checks"

    lines = []
    for h in history:
        verdict = "first attempt" if h["attempt"] == 1 else ("improved" if h["improved"] else "no improvement")
        lines.append(f"- Attempt {h['attempt']}: {describe(h['outcome'])}; {verdict}.")
    return "\n".join(lines) or "None."


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _refuse_tracing() -> None:
    """Guardrail 4 of design doc 7.7: tracing would send prompts and graph state to LangSmith."""
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        if os.environ.get(var):
            raise RuntimeError(f"{var} is set. Unset it: tracing would send prompts and graph state to LangSmith.")


def build_graph(
    llm: BaseChatModel | None = None,
    *,
    mode: str = "baseline",
    comparator: Comparator | None = None,
    corrector: Corrector | None = None,
    runs_dir: Path | None = None,
    migrated_dir: Path | None = None,
):
    """Define the nodes, wire them together, and compile the workflow into a runnable app.

    Notes:
    - mode="baseline" is the single-pass flow (D-043). mode="fix_loop" adds checking and correction
      (design doc 6.12) and needs a comparator.
    - corrector defaults to llm_corrector(llm), which needs prompts/fix.md from the translation team.
    - A node is just a function: it receives the state and returns only the keys it changes.
    - The nodes are defined *inside* this function so they can use `llm`, `comparator`, and
      `corrector` directly (a closure). This keeps them out of the state and lets tests swap in fakes.
    - runs_dir and migrated_dir default to config.RUNS_DIR and config.MIGRATED_DIR.
    - Every run needs "program_id" and "run_id" in its input state; the fix loop also needs "input_dir".
    """
    _refuse_tracing()
    if mode not in ("baseline", "fix_loop"):
        raise ValueError(f"Unknown mode {mode!r}")
    if mode == "fix_loop" and comparator is None:
        raise ValueError("The fix loop needs a comparator (design doc 6.12.4).")
    llm = llm or build_llm()
    corrector = corrector or llm_corrector(llm)
    runs_dir = runs_dir or config.RUNS_DIR
    migrated_dir = migrated_dir or config.MIGRATED_DIR
    # Full text of Star_Macros.sas. Every SAS program %INCLUDEs it, so the model needs it to
    # understand macro calls like %keep_hos. Read once here and reused for every program.
    macros = config.MACROS_FILE.read_text(errors="replace")

    def ctx(state: MigrationState) -> dict:
        """Prompt context: the values that fill the {placeholders} in analyze.md / translate.md.

        Used as prompts.ANALYZE.format(**ctx(state)), so each key must match a placeholder name.
        """
        program = config.get_program(state["program_id"])
        return dict(
            sas_file=program.sas_file,                                  # SAS file name
            sas_code=program.path.read_text(errors="replace"),          # SAS program text
            macros=macros,                                              # Star_Macros.sas text
            contract=prompts.RUNTIME_CONTRACT.format(outputs=", ".join(program.outputs)),  # I/O rules
            prior=prompts.format_prior(state.get("prior_code", {})),    # earlier programs' Python
        )

    def program_dir(state: MigrationState) -> Path:
        out = runs_dir / state["run_id"] / state["program_id"]
        out.mkdir(parents=True, exist_ok=True)
        return out

    def attempt_dirs(state: MigrationState, attempt: int | None = None) -> runs.AttemptDirs:
        return runs.attempt_dirs(state["run_id"], state["program_id"], attempt or state["attempt"], runs_dir)

    # --- shared nodes ------------------------------------------------------------------------------
    def analyze(state: MigrationState) -> dict:
        analysis = _ask(llm, prompts.ANALYZE.format(**ctx(state)))
        # Write immediately so it can be read while translate runs, and survives a translate failure.
        (program_dir(state) / "analysis.md").write_text(analysis)
        return {"analysis": analysis}

    def translate(state: MigrationState) -> dict:
        reply = _ask(llm, prompts.TRANSLATE.format(analysis=state["analysis"], **ctx(state)))
        code = extract_python(reply)
        (attempt_dirs(state, 1).ai / "script.py").write_text(code)
        return {
            "code": code, "attempt": 1, "event": "TRANSLATION_OK",
            "corrections": 0, "no_improvement": 0, "history": [], "best": None,
        }

    # --- baseline ----------------------------------------------------------------------------------
    def save(state: MigrationState) -> dict:
        return {"status": "BASELINE", "output_dir": str(attempt_dirs(state).ai)}

    # --- fix loop ----------------------------------------------------------------------------------
    def check(state: MigrationState) -> dict:
        violations = static_check.check(state["code"])
        if not violations:
            return {"event": "STATIC_PASS"}
        return {
            "event": "STATIC_FAIL",
            "outcome": {"stage": "static", "failing_checks": len(violations)},
            "fix_request": {"kind": "static", "violations": [asdict(v) for v in violations]},
        }

    def execute(state: MigrationState) -> dict:
        dirs = attempt_dirs(state)
        result = executor.execute(dirs.ai / "script.py", Path(state["input_dir"]), dirs.outputs, dirs.data)
        _write_json(dirs.ai / "exec_status.json", asdict(result))
        if result.status == "ok":
            return {"event": "EXEC_OK"}
        return {
            "event": "TIMEOUT" if result.status == "timeout" else "EXEC_ERROR",
            "outcome": {"stage": "execution", "failing_checks": 1},
            "fix_request": {
                "kind": "execution", "status": result.status,
                "exception": result.exception, "files_written": result.files_written,
            },
        }

    def compare(state: MigrationState) -> dict:
        dirs = attempt_dirs(state)
        report = comparator(state["program_id"], dirs.outputs, dirs.data)
        _write_json(dirs.ai / "redacted_report.json", [asdict(c) for c in report.checks])
        if report.passed:
            return {"event": "ALL_PASS", "outcome": {"stage": "comparison", "failing_checks": 0, "checks": len(report.checks)}}
        failed = [c for c in report.checks if not c.passed]
        return {
            "event": "FAIL",
            "outcome": {
                "stage": "comparison",
                "failing_checks": len(failed),
                "mismatches": sum(_mismatch_count(c.mismatch) for c in failed),
                "error_bin": max((ERROR_BINS.index(c.error_bin) for c in failed if c.error_bin), default=-1),
            },
            "fix_request": {
                "kind": "comparison",
                "failed_checks": [asdict(c) for c in failed[:MAX_LISTED_CHECKS]],
                "more": max(0, len(failed) - MAX_LISTED_CHECKS),
            },
        }

    def fix(state: MigrationState) -> dict:
        """Record the failed attempt, apply the limits, and ask the Corrector for a new script."""
        current = {k: state[k] for k in ("attempt", "code", "outcome", "fix_request")}
        best = state.get("best")
        improved = best is None or outcome_key(current["outcome"]) > outcome_key(best["outcome"])
        regressed = best is not None and outcome_key(current["outcome"]) < outcome_key(best["outcome"])
        if improved:
            best = current
        no_improvement = 0 if improved else state["no_improvement"] + 1
        history = state["history"] + [{"attempt": current["attempt"], "outcome": current["outcome"], "improved": improved}]
        updates = {"best": best, "no_improvement": no_improvement, "history": history}
        if state["corrections"] >= config.MAX_CORRECTIONS or no_improvement >= config.MAX_NO_IMPROVEMENT:
            return {**updates, "event": "LIMIT"}

        # A regression is rejected: the Corrector works from the best attempt instead (D-037).
        base = best if regressed else current
        attempt = state["attempt"] + 1
        dirs = attempt_dirs(state, attempt)
        _write_json(dirs.ai / "fix_request.json", {"based_on_attempt": base["attempt"], **base["fix_request"]})
        code = corrector({
            **ctx(state),
            "analysis": state["analysis"],
            "code": base["code"],
            "fix_request": render_fix_request(base["fix_request"]),
            "history": render_history(history),
        })
        (dirs.ai / "script.py").write_text(code)
        return {**updates, "event": "CODE_FAULT", "code": code, "attempt": attempt, "corrections": state["corrections"] + 1}

    def promote(state: MigrationState) -> dict:
        """Copy the passing translation into migrated/ with a record of its origin (D-051)."""
        dest = migrated_dir / state["program_id"]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{state['program_id']}.py").write_text(state["code"])
        (dest / "analysis.md").write_text(state["analysis"])
        prompt_files = sorted(Path(prompts.__file__).parent.glob("*.md"))
        _write_json(dest / "provenance.json", {
            "run_id": state["run_id"],
            "program_id": state["program_id"],
            "attempt": state["attempt"],
            "model": config.MODEL,
            "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prompt_files_sha256": {p.name: _sha256(p.read_bytes()) for p in prompt_files},
            "comparison": {"checks": state["outcome"].get("checks"), "failing": 0},
            "script_sha256": _sha256(state["code"].encode()),
        })
        return {"status": "PASSED", "output_dir": str(dest)}

    def escalate(state: MigrationState) -> dict:
        """End the run for a human to review; the best attempt is kept (design doc 6.12.5)."""
        best = state["best"]
        _write_json(program_dir(state) / "escalation.json", {
            "best_attempt": best["attempt"], "corrections": state["corrections"], "history": state["history"],
        })
        return {"status": "ESCALATED", "code": best["code"], "attempt": best["attempt"], "output_dir": str(program_dir(state))}

    graph = StateGraph(MigrationState)
    graph.add_node("analyze", analyze)
    graph.add_node("translate", translate)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "translate")

    if mode == "baseline":
        graph.add_node("save", save)
        graph.add_edge("translate", "save")
        graph.add_edge("save", END)
        return graph.compile()

    for name, node in [("static_check", check), ("execute", execute), ("compare", compare),
                       ("fix", fix), ("promote", promote), ("escalate", escalate)]:
        graph.add_node(name, node)
    for source in {s for s, _ in TRANSITIONS}:
        targets = sorted({t for (s, _), t in TRANSITIONS.items() if s == source})
        graph.add_conditional_edges(source, partial(route, source), targets)
    graph.add_edge("promote", END)
    graph.add_edge("escalate", END)
    return graph.compile()
