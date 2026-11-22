"""CLI: python -m sas_agent.spec UNIT [--facts-only] [--max-revisions N]

Writes runs/<run-id>/specs/<UNIT>.v<n>.json, the redacted validation report of each version, and the
prompts sent to the model.
"""

import argparse
import json

from .. import config, runs
from . import analyzer
from .agent import SpecFailed, SpecUnchanged, generate_spec, revise_spec, save
from .metadata import for_facts
from .validation import validate


def main() -> None:
    analyzed = analyzer.analyze_macros()
    parser = argparse.ArgumentParser(description="Write a checked specification for a SAS unit.")
    parser.add_argument("unit", choices=sorted(analyzed), help="unit id, e.g. M.grp_score")
    parser.add_argument("--facts-only", action="store_true", help="print the facts layer; no model call")
    parser.add_argument("--max-revisions", type=int, default=3, help="revisions after failed validation (D-036)")
    args = parser.parse_args()

    unit, facts = analyzed[args.unit]
    if args.facts_only:
        print(json.dumps({"unit": unit.model_dump(mode="json"), "facts": facts.model_dump(mode="json")}, indent=2))
        return

    from ..graph import build_llm

    folder = config.RUNS_DIR / runs.new_run_id() / "specs"
    folder.mkdir(parents=True, exist_ok=True)
    calls = []

    def record(prompt: str, response: str) -> None:
        calls.append(1)
        (folder / f"{args.unit}.call{len(calls)}.prompt.md").write_text(prompt)
        (folder / f"{args.unit}.call{len(calls)}.response.json").write_text(response)

    llm, metadata = build_llm(), for_facts(facts)
    try:
        spec = generate_spec(args.unit, llm, analyzed=(unit, facts), metadata=metadata, record=record)
        for revision in range(args.max_revisions + 1):
            print(f"[{args.unit}] v{spec.version} -> {save(spec, folder)}")
            items = validate(spec)
            (folder / f"{args.unit}.v{spec.version}.validation.json").write_text(
                json.dumps([i.model_dump() for i in items], indent=2))
            failed = [i.describe() for i in items if not i.passed]
            print(f"[{args.unit}] validation: {len(items) - len(failed)} of {len(items)} checks passed")
            if not failed:
                print(f"[{args.unit}] SPEC_VALIDATED")
                break
            if revision == args.max_revisions:
                print(f"[{args.unit}] ESCALATED: validation still fails after {args.max_revisions} revisions")
                break
            spec = revise_spec(spec, failed, llm, metadata=metadata, record=record)
    except (SpecFailed, SpecUnchanged) as e:
        print(f"[{args.unit}] ESCALATED: {e}")
    print(f"Run records: {folder}")


if __name__ == "__main__":
    main()
