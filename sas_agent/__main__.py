"""CLI: python -m sas_agent [program0 program1 ...]"""

import argparse

from . import config
from .graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze, translate, and document the SAS Star Rating pipeline.",
    )
    parser.add_argument(
        "programs", nargs="*", default=[p.id for p in config.PROGRAMS],
        help=f"programs to translate, in order (default: all of {[p.id for p in config.PROGRAMS]})",
    )
    args = parser.parse_args()

    app = build_graph()
    prior_code: dict[str, str] = {}
    # Include already-translated earlier programs as context, even if not re-run this time.
    for p in config.PROGRAMS:
        existing = config.MIGRATED_DIR / p.id / f"{p.id}.py"
        if p.id not in args.programs and existing.exists():
            prior_code[p.id] = existing.read_text()

    for program_id in args.programs:
        config.get_program(program_id)  # fail fast on typos
        print(f"=== Translating {program_id} with {config.MODEL} ===")
        state = {}
        for mode, chunk in app.stream(
            {"program_id": program_id, "prior_code": dict(prior_code)},
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                print(f"[{program_id}] finished node: {', '.join(chunk)}")
            else:
                state = chunk
        prior_code[program_id] = state["code"]
        print(
            f"[{program_id}] wrote analysis.md, {program_id}.py, and documentation.md "
            f"-> {state['output_dir']}"
        )


if __name__ == "__main__":
    main()
