"""LangGraph workflow: analyze -> translate -> save -> document."""

import re
from pathlib import Path
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from . import config, prompts
from .mapping import mapping_heading, parse_mapping


class MigrationState(TypedDict, total=False):
    program_id: str
    prior_code: dict[str, str]  # program_id -> translated Python code of earlier programs
    analysis: str
    code: str
    output_dir: str
    documentation: str


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


def build_graph(llm: BaseChatModel | None = None):
    """Define the nodes, wire them together, and compile the workflow into a runnable app.

    Notes:
    - A node is just a function: it receives the state and returns only the keys it adds.
      add_node registers it; add_edge sets the order
      (START -> analyze -> translate -> save -> document -> END).
    - The nodes are defined *inside* this function so they can use `llm` and `macros` directly
      (a closure). LangGraph passes nodes only the state, and this keeps the model out of the
      state and out of globals, which is what lets tests swap in a fake model.
    """
    llm = llm or build_llm()
    # Full text of Star_Macros.sas. Every SAS program %INCLUDEs it, so the model needs it to
    # understand macro calls like %keep_hos. Read once here and reused for every program.
    macros = config.MACROS_FILE.read_text(errors="replace")

    def ctx(state: MigrationState) -> dict:
        """Prompt context: the values that fill the {placeholders} in analyze.md / translate.md.

        Used as prompts.ANALYZE.format(**ctx(state)), so each key must match a placeholder name.
        """
        program = config.get_program(state["program_id"])
        mapping_text = config.MAPPING_FILE.read_text(errors="replace")
        return dict(
            sas_file=program.sas_file,                                  # SAS file name
            sas_code=program.path.read_text(errors="replace"),          # SAS program text
            macros=macros,                                              # Star_Macros.sas text
            contract=prompts.RUNTIME_CONTRACT.format(outputs=", ".join(program.outputs)),  # I/O rules
            prior=prompts.format_prior(state.get("prior_code", {})),    # earlier programs' Python
            outputs=", ".join(program.outputs),
            mapping=parse_mapping(mapping_text, mapping_heading(state["program_id"])),
        )

    def output_dir(state: MigrationState) -> Path:
        out = config.MIGRATED_DIR / state["program_id"]
        out.mkdir(parents=True, exist_ok=True)
        return out

    def analyze(state: MigrationState) -> dict:
        analysis = _ask(llm, prompts.ANALYZE.format(**ctx(state)))
        # Write immediately so it can be read while translate runs, and survives a translate failure.
        (output_dir(state) / "analysis.md").write_text(analysis)
        return {"analysis": analysis}

    def translate(state: MigrationState) -> dict:
        reply = _ask(llm, prompts.TRANSLATE.format(analysis=state["analysis"], **ctx(state)))
        return {"code": extract_python(reply)}

    def save(state: MigrationState) -> dict:
        out = output_dir(state)
        (out / f"{state['program_id']}.py").write_text(state["code"])
        return {"output_dir": str(out)}

    def document(state: MigrationState) -> dict:
        prompt = prompts.fill(
            prompts.DOCUMENT,
            analysis=state["analysis"],
            code=state["code"],
            **ctx(state),
        )
        documentation = _ask(llm, prompt)
        (output_dir(state) / "documentation.md").write_text(documentation)
        return {"documentation": documentation}

    graph = StateGraph(MigrationState)
    graph.add_node("analyze", analyze)
    graph.add_node("translate", translate)
    graph.add_node("save", save)
    graph.add_node("document", document)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "translate")
    graph.add_edge("translate", "save")
    graph.add_edge("save", "document")
    graph.add_edge("document", END)
    return graph.compile()
