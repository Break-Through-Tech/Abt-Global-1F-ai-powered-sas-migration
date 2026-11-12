"""LangGraph workflows (design doc 6.12, D-038).

Two graphs share the analyze and translate nodes:

- Single-pass baseline (D-043), the skeleton's original flow:
      analyze -> translate -> save
- Fix loop:
      analyze -> translate -> validate -> promote
  When validation fails, correct asks the Correction agent for a repair prompt, repair asks the
  Translator for a whole new script, and the new script is validated again, until it passes or a limit
  is reached (escalate).

Ownership (design doc 13.1): the graph, routing, limits, and run records belong to the orchestration
subgroup. Validation, correction, and translation belong to their agents, which the graph calls through
the interfaces in agents.py.
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from . import config, prompts, runs
from .agents import Corrector, Repairer, ValidationView, Validator

# Enough steps for analyze, translate, and MAX_CORRECTIONS + 1 attempts of three nodes each.
# Pass as config={"recursion_limit": RECURSION_LIMIT} when invoking the fix loop.
RECURSION_LIMIT = 100


class MigrationState(TypedDict, total=False):
    # Everything here may reach the model (design doc 7.7): identifiers, paths, code, redacted results,
    # counters. Never DataFrames, data values, full reports, or exception messages.
    program_id: str
    run_id: str
    prior_code: dict[str, str]  # program_id -> translated Python code of earlier programs
    analysis: str
    code: str                   # the current script
    attempt: int                # attempt number of `code`; 1 is the first translation
    event: str                  # result of the last node; route() reads it
    validation: dict            # ValidationView of the current attempt: next_state, rank, fix_request
    best: dict | None           # best attempt so far: attempt, code, validation
    attempts: list[dict]        # one entry per failed attempt: attempt, rank, improved
    corrections: int            # repair prompts requested so far
    no_improvement: int         # corrections in a row that did not improve on the best attempt
    repair_prompt: str          # from the Correction agent, for the Translator
    status: str                 # at the end: "PASSED", "ESCALATED", or "BASELINE"
    output_dir: str             # folder holding this program's result


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


def llm_repairer(llm: BaseChatModel) -> Repairer:
    """Repairer that sends the Correction agent's repair prompt to the translator's model.

    Stands in for the Translator's repair entry point until the Translator is packaged.
    """
    def repair(prompt: str) -> str:
        return extract_python(_ask(llm, prompt))

    return repair


# ---------------------------------------------------------------------------------------------------
# Routing (design doc 6.4.4, D-033): the next node depends only on the current node and the event it
# returned. promote and escalate lead to END.
TRANSITIONS: dict[tuple[str, str], str] = {
    ("translate", "TRANSLATION_OK"): "validate",
    ("translate", "TRANSLATION_FAILED"): "escalate",
    ("validate", "VALID_PASS"): "promote",
    ("validate", "VALID_FAIL"): "correct",
    ("correct", "CORRECTED"): "repair",
    ("correct", "LIMIT"): "escalate",
    ("repair", "REPAIRED"): "validate",
    ("repair", "TRANSLATION_FAILED"): "escalate",
}


def route(node: str, state: MigrationState) -> str:
    """Next node after `node`, read from TRANSITIONS. An undefined pair is an error (D-033)."""
    key = (node, state.get("event"))
    if key not in TRANSITIONS:
        raise RuntimeError(f"No transition for node {node!r} on event {key[1]!r}")
    return TRANSITIONS[key]


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
    validator: Validator | None = None,
    corrector: Corrector | None = None,
    repairer: Repairer | None = None,
    runs_dir: Path | None = None,
    migrated_dir: Path | None = None,
):
    """Define the nodes, wire them together, and compile the workflow into a runnable app.

    Notes:
    - mode="baseline" is the single-pass flow (D-043). mode="fix_loop" adds validation and correction
      and needs a validator and a corrector (agents.py has adapters for the teammates' agents).
    - repairer defaults to llm_repairer(llm) until the Translator is packaged.
    - A node is just a function: it receives the state and returns only the keys it changes.
    - The nodes are defined *inside* this function so they can use `llm` and the agents directly
      (a closure). This keeps them out of the state and lets tests swap in fakes.
    - runs_dir and migrated_dir default to config.RUNS_DIR and config.MIGRATED_DIR.
    - Every run needs "program_id" and "run_id" in its input state.
    """
    _refuse_tracing()
    if mode not in ("baseline", "fix_loop"):
        raise ValueError(f"Unknown mode {mode!r}")
    if mode == "fix_loop" and (validator is None or corrector is None):
        raise ValueError("The fix loop needs a validator and a corrector (see sas_agent/agents.py).")
    llm = llm or build_llm()
    repairer = repairer or llm_repairer(llm)
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
        start = {"attempt": 1, "corrections": 0, "no_improvement": 0, "attempts": [], "best": None}
        try:
            code = extract_python(reply)
        except ValueError:
            if mode == "baseline":
                raise
            return {**start, "code": "", "event": "TRANSLATION_FAILED"}
        (attempt_dirs(state, 1).ai / "script.py").write_text(code)
        return {**start, "code": code, "event": "TRANSLATION_OK"}

    # --- baseline ----------------------------------------------------------------------------------
    def save(state: MigrationState) -> dict:
        return {"status": "BASELINE", "output_dir": str(attempt_dirs(state).ai)}

    # --- fix loop ----------------------------------------------------------------------------------
    def validate(state: MigrationState) -> dict:
        """Hand the current script to the Validation agent; keep only the redacted view."""
        dirs = attempt_dirs(state)
        view = ValidationView.of(validator(state["program_id"], state["code"], dirs.root))
        validation = {"next_state": view.next_state, "rank": list(view.rank), "fix_request": view.fix_request}
        return {"validation": validation, "event": "VALID_PASS" if view.passed else "VALID_FAIL"}

    def correct(state: MigrationState) -> dict:
        """Record the failed attempt, apply the limits, and ask the Correction agent for a repair prompt."""
        current = {"attempt": state["attempt"], "code": state["code"], "validation": state["validation"]}
        rank, best = tuple(current["validation"]["rank"]), state.get("best")
        improved = best is None or rank < tuple(best["validation"]["rank"])
        regressed = best is not None and rank > tuple(best["validation"]["rank"])
        if improved:
            best = current
        no_improvement = 0 if improved else state["no_improvement"] + 1
        attempts = state["attempts"] + [{"attempt": current["attempt"], "rank": list(rank), "improved": improved}]
        updates = {"best": best, "no_improvement": no_improvement, "attempts": attempts}
        if state["corrections"] >= config.MAX_CORRECTIONS or no_improvement >= config.MAX_NO_IMPROVEMENT:
            return {**updates, "event": "LIMIT"}

        # A regression is rejected: the correction starts from the best attempt instead (D-037).
        base = best if regressed else current
        view = ValidationView.of(base["validation"])
        repair_prompt = corrector(
            state["program_id"], base["code"], view.as_feedback(state["program_id"]),
            analysis=state["analysis"], prior_code=state.get("prior_code", {}),
            history=[{"program_id": state["program_id"], "rank": a["rank"]} for a in attempts],
        )
        dirs = attempt_dirs(state, state["attempt"] + 1)
        (dirs.ai / "repair_prompt.md").write_text(repair_prompt)
        _write_json(dirs.ai / "based_on.json", {"attempt": base["attempt"], "rank": base["validation"]["rank"]})
        return {**updates, "event": "CORRECTED", "repair_prompt": repair_prompt, "corrections": state["corrections"] + 1}

    def repair(state: MigrationState) -> dict:
        """Ask the Translator for a whole new script from the repair prompt."""
        attempt = state["attempt"] + 1
        try:
            code = repairer(state["repair_prompt"])
        except ValueError:  # no code block in the reply
            return {"event": "TRANSLATION_FAILED", "attempt": attempt}
        (attempt_dirs(state, attempt).ai / "script.py").write_text(code)
        return {"event": "REPAIRED", "code": code, "attempt": attempt}

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
            "validation": {"next_state": "PASSED", "rank": state["validation"]["rank"]},
            "script_sha256": _sha256(state["code"].encode()),
        })
        return {"status": "PASSED", "output_dir": str(dest)}

    def escalate(state: MigrationState) -> dict:
        """End the run for a human to review; the best attempt is kept (design doc 6.12.5)."""
        best = state.get("best")
        _write_json(program_dir(state) / "escalation.json", {
            "reason": state["event"],
            "best_attempt": best["attempt"] if best else None,
            "corrections": state.get("corrections", 0),
            "attempts": state.get("attempts", []),
        })
        if best is None:  # no attempt ever produced a script
            return {"status": "ESCALATED", "code": "", "output_dir": str(program_dir(state))}
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

    for name, node in [("validate", validate), ("correct", correct), ("repair", repair),
                       ("promote", promote), ("escalate", escalate)]:
        graph.add_node(name, node)
    for source in {s for s, _ in TRANSITIONS}:
        targets = sorted({t for (s, _), t in TRANSITIONS.items() if s == source})
        graph.add_conditional_edges(source, partial(route, source), targets)
    graph.add_edge("promote", END)
    graph.add_edge("escalate", END)
    return graph.compile()
