"""CLI: python -m sas_agent [--fix-loop] [program0 program1 ...]"""

import argparse

from . import config, runs
from .graph import RECURSION_LIMIT, build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and translate the SAS Star Rating pipeline to Python.")
    parser.add_argument(
        "programs", nargs="*", default=[p.id for p in config.PROGRAMS],
        help=f"programs to translate, in order (default: all of {[p.id for p in config.PROGRAMS]})",
    )
    parser.add_argument(
        "--fix-loop", action="store_true",
        help="run, compare with the SAS outputs, and correct each translation (design doc 6.12); "
             "without it, the single-pass baseline runs (D-043)",
    )
    args = parser.parse_args()

    comparator = None
    if args.fix_loop:
        try:
            from .comparator import compare as comparator
        except ModuleNotFoundError as e:
            if e.name != f"{__package__}.comparator":
                raise
            raise SystemExit("The fix loop needs sas_agent/comparator.py, which is not written yet (design doc 6.12.4).")

    app = build_graph(mode="fix_loop" if args.fix_loop else "baseline", comparator=comparator)
    run_id = runs.new_run_id()
    initial = {"run_id": run_id}
    if args.fix_loop:
        initial["input_dir"] = str(runs.prepare_inputs(run_id))

    prior_code: dict[str, str] = {}
    # Include promoted translations of earlier programs as context, even if not re-run this time.
    # Only translations that passed the comparator are in migrated/ (D-051).
    for p in config.PROGRAMS:
        existing = config.MIGRATED_DIR / p.id / f"{p.id}.py"
        if p.id not in args.programs and existing.exists():
            prior_code[p.id] = existing.read_text()

    for program_id in args.programs:
        config.get_program(program_id)  # fail fast on typos
        print(f"=== Translating {program_id} with {config.MODEL} (run {run_id}) ===")
        state = {}
        for mode, chunk in app.stream(
            {**initial, "program_id": program_id, "prior_code": dict(prior_code)},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                print(f"[{program_id}] finished node: {', '.join(chunk)}")
            else:
                state = chunk
        print(f"[{program_id}] {state['status']} at attempt {state['attempt']} -> {state['output_dir']}")
        if state["status"] == "ESCALATED":
            print(f"[{program_id}] stopping: later programs depend on this one. See escalation.json.")
            break
        prior_code[program_id] = state["code"]
    print(f"Run records: {config.RUNS_DIR / run_id}")


if __name__ == "__main__":
    main()
