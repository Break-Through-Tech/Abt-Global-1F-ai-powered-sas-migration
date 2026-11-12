"""The Spec agent: the model writes the contract and interpretation layers; code checks them.

generate_spec() and revise_spec() are plain functions. The unit-level harness will call them from its
PENDING, SPECCED, and RESPEC handlers (design doc 6.4.4).

Each call tries up to max_tries times (D-036): the model's layers are merged with the facts and checked
(checks.py); on failure, the next prompt carries the check errors and the previous layers.
"""

import json
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from .. import prompts
from ..graph import _refuse_tracing
from . import analyzer, checks
from .metadata import DatasetMeta, for_facts
from .risks import RISKS
from .schema import AgentLayers, Facts, Spec, Unit

SYSTEM = ("You are a senior SAS 9.4 and Python engineer writing precise specifications of SAS code for a "
          "Translator. You describe behaviour exactly; you never write code; you never include data values.")


class SpecFailed(RuntimeError):
    """The model's layers still failed the spec checks after the last try."""

    def __init__(self, unit_id: str, errors: list[str]):
        super().__init__(f"{unit_id}: spec checks failed after the last try:\n- " + "\n- ".join(errors))
        self.errors = errors


class SpecUnchanged(RuntimeError):
    """A revision returned the same layers as the spec it was asked to revise (D-036)."""


# Called with (prompt, response) after each model call, e.g. to save them in the run folder.
Recorder = Callable[[str, str], None]


def _metadata_text(metadata: dict[str, DatasetMeta]) -> str:
    if not metadata:
        return "None saved."
    return "\n".join(f"- {m.name}: {m.rows} rows; columns in order: {', '.join(m.columns)}" for m in metadata.values())


def build_prompt(unit: Unit, facts: Facts, metadata: dict[str, DatasetMeta], revision: str = "") -> str:
    return prompts.load("spec").format(
        unit_id=unit.id,
        facts=facts.model_dump_json(indent=2),
        metadata=_metadata_text(metadata),
        risks="\n".join(f"- {k}: {v}" for k, v in RISKS.items()),
        sas_source=analyzer.unit_source(unit),
        revision=revision,
    )


def _try_layers(llm, unit: Unit, facts: Facts, metadata, base_revision: str, version: int,
                max_tries: int, record: Recorder | None) -> Spec:
    structured = llm.with_structured_output(AgentLayers)
    revision, errors = base_revision, []
    for _ in range(max_tries):
        prompt = build_prompt(unit, facts, metadata, revision)
        try:
            layers = structured.invoke([SystemMessage(SYSTEM), HumanMessage(prompt)])
            if layers is None:
                raise ValueError("the response did not contain the structured layers")
            if isinstance(layers, dict):
                layers = AgentLayers.model_validate(layers)
            spec = Spec(unit=unit, facts=facts, version=version, **layers.model_dump())
            errors = checks.check(spec, metadata)
            response = layers.model_dump_json(indent=2)
        except ValueError as e:      # includes pydantic ValidationError and LangChain parser errors
            spec, errors, response = None, [f"the response did not match the required structure: {e}"], ""
        if record:
            record(prompt, response)
        if not errors:
            return spec
        revision = (base_revision + "\n=== Your previous answer failed these checks; fix them ===\n- "
                    + "\n- ".join(errors) + ("\n\nPrevious answer:\n" + response if response else ""))
    raise SpecFailed(unit.id, errors)


def generate_spec(unit_id: str, llm, *, analyzed: tuple[Unit, Facts] | None = None,
                  metadata: dict[str, DatasetMeta] | None = None, max_tries: int = 3,
                  record: Recorder | None = None) -> Spec:
    """Version 1 of the spec for a unit, or SpecFailed."""
    _refuse_tracing()
    unit, facts = analyzed or analyzer.analyze_macros()[unit_id]
    metadata = for_facts(facts) if metadata is None else metadata
    return _try_layers(llm, unit, facts, metadata, "", 1, max_tries, record)


def revise_spec(spec: Spec, evidence: list[str], llm, *, metadata: dict[str, DatasetMeta] | None = None,
                max_tries: int = 3, record: Recorder | None = None) -> Spec:
    """The next version of a spec, given redacted evidence that it is wrong (validation items, conflicts)."""
    _refuse_tracing()
    metadata = for_facts(spec.facts) if metadata is None else metadata
    revision = ("\n=== Revise the current specification ===\nThis evidence shows the current specification "
                "is wrong:\n- " + "\n- ".join(evidence) + "\n\nCurrent specification layers:\n"
                + spec.layers().model_dump_json(indent=2))
    new = _try_layers(llm, spec.unit, spec.facts, metadata, revision, spec.version + 1, max_tries, record)
    if new.layers() == spec.layers():
        raise SpecUnchanged(f"{spec.unit.id}: the revision returned the specification unchanged")
    return new


def save(spec: Spec, folder) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{spec.unit.id}.v{spec.version}.json"
    path.write_text(json.dumps(spec.model_dump(mode="json"), indent=2) + "\n")
    return str(path)
