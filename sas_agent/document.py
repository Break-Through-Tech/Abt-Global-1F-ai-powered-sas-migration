"""Documentation agent (issue #8): analyst documentation for a promoted translation.

Standalone: it is not a node in graph.py. Run it after the fix loop has promoted a program:

    python -m sas_agent --fix-loop program0     # translate, validate, promote
    python -m sas_agent.document program0       # writes migrated/program0/documentation.md

It reads only what promote wrote to migrated/<program>/ (the script and analysis.md), so it only
ever documents code that passed the comparison with SAS. It never runs code or reads data values.
"""

import argparse
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from . import config, prompts
from .graph import _ask, build_llm
from .mapping import mapping_heading, parse_mapping

MAPPING_FILE = config.REPO_ROOT / "SAS_CSV_Mapping.txt"  # which CSV each program writes, in plain words
PROMPT = (Path(prompts.__file__).parent / "document.md").read_text(encoding="utf-8")


def fill(template: str, **values: str) -> str:
    """Replace {name} placeholders without interpreting the braces inside SAS or Python code."""
    for name, value in values.items():
        template = template.replace("{" + name + "}", value)
    return template


def build_prompt(program_id: str, code: str, analysis: str) -> str:
    program = config.get_program(program_id)
    mapping = parse_mapping(MAPPING_FILE.read_text(encoding="utf-8", errors="replace"), mapping_heading(program_id))
    return fill(
        PROMPT,
        mapping=mapping,
        outputs=", ".join(program.outputs),
        analysis=analysis,
        code=code,
        macros=config.MACROS_FILE.read_text(encoding="utf-8", errors="replace"),
        sas_file=program.sas_file,
        sas_code=program.path.read_text(encoding="utf-8", errors="replace"),
    )


def document(program_id: str, llm: BaseChatModel | None = None, migrated_dir: Path | None = None) -> Path:
    """Write documentation.md next to the promoted script and return its path."""
    folder = (migrated_dir or config.MIGRATED_DIR) / program_id
    script, analysis = folder / f"{program_id}.py", folder / "analysis.md"
    if not script.exists() or not analysis.exists():
        raise FileNotFoundError(
            f"No promoted translation in {folder}. Run `python -m sas_agent --fix-loop {program_id}` first."
        )
    prompt = build_prompt(
        program_id,
        code=script.read_text(encoding="utf-8"),
        analysis=analysis.read_text(encoding="utf-8"),
    )
    out = folder / "documentation.md"
    out.write_text(_ask(llm or build_llm(), prompt), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Write analyst documentation for promoted translations.")
    parser.add_argument(
        "programs", nargs="*", default=[p.id for p in config.PROGRAMS],
        help=f"programs to document (default: all of {[p.id for p in config.PROGRAMS]})",
    )
    args = parser.parse_args()
    for program_id in args.programs:
        config.get_program(program_id)  # fail fast on typos, before building the model
    llm = build_llm()
    for program_id in args.programs:
        print(f"[{program_id}] wrote {document(program_id, llm)}")


if __name__ == "__main__":
    main()
