"""Spec checks before merging (design doc §6.5.3). Each error is phrased for the model, which gets it on a retry."""

import re

from .metadata import DatasetMeta
from .risks import RISKS
from .schema import OutputSpec, Spec

_SERIES = re.compile(r"^([A-Za-z_]\w*?)1\.\.\1n$")       # "C1..Cn"


def match_columns(pattern: list[str], actual: list[str]) -> bool:
    """Whether actual columns follow a returns pattern, in order.

    Tokens: a literal column name (case-insensitive); "*param", one or more columns; "C1..Cn",
    the numbered series C1, C2, ... with at least one member.
    """
    def match(i: int, j: int) -> bool:
        if i == len(pattern):
            return j == len(actual)
        token = pattern[i]
        if token.startswith("*"):
            return any(match(i + 1, k) for k in range(j + 1, len(actual) + 1))
        series = _SERIES.match(token)
        if series:
            k, n = j, 1
            while k < len(actual) and actual[k].lower() == f"{series.group(1)}{n}".lower():
                k, n = k + 1, n + 1
                if match(i + 1, k):
                    return True
            return False
        return j < len(actual) and actual[j].lower() == token.lower() and match(i + 1, j + 1)

    return match(0, 0)


def literal_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if not c.startswith("*") and not _SERIES.match(c)]


def output_for(spec: Spec, output: str | None) -> OutputSpec | None:
    returns = spec.contract.returns
    if output is None:
        return returns[0] if returns else None
    return next((r for r in returns if r.name.lower() == output.lower()), None)


def check(spec: Spec, metadata: dict[str, DatasetMeta], behaviour_notes: dict[str, str] | None = None) -> list[str]:
    facts, errors = spec.facts, []
    params = {p.lower(): p for p in facts.params}

    # 2. Logic steps and facts steps correspond one to one.
    step_ids = [s.id for s in facts.steps]
    logic_ids = [l.step for l in spec.logic]
    errors += [f"logic refers to step {s!r}, which is not in the facts (steps are {step_ids})"
               for s in logic_ids if s not in step_ids]
    errors += [f"facts step {s} ({next(x.kind for x in facts.steps if x.id == s)}) has no logic entry"
               for s in step_ids if s not in logic_ids]

    # 3. Returns are the unit's outputs.
    allowed = {p.lower() for p in facts.output_params} | {d.lower() for d in facts.side_effects.read_downstream}
    errors += [f"contract return {r.name!r} is not an output of the unit (outputs: {facts.output_params})"
               for r in spec.contract.returns if r.name.lower() not in allowed]
    returned = {r.name.lower() for r in spec.contract.returns}
    errors += [f"output parameter {p!r} has no contract return" for p in facts.output_params if p.lower() not in returned]
    errors += [f"contract input {i.name!r} is not a parameter of the unit (parameters: {facts.params})"
               for i in spec.contract.inputs if i.name.lower() not in params]

    # 4. Columns against the reference datasets at every call site.
    for site in facts.call_sites:
        for inp in spec.contract.inputs:
            meta = metadata.get(site.arg(inp.name) or "")
            if meta:
                have = {c.lower() for c in meta.columns}
                errors += [f"input {inp.name} requires column {c!r}, which {meta.name} ({site.caller}) does not have"
                           for c in literal_columns(inp.requires) if c.lower() not in have]
        for ret in spec.contract.returns:
            meta = metadata.get(site.arg(ret.name) or "")
            if meta and not match_columns(ret.columns, meta.columns):
                errors += [f"return {ret.name} columns {ret.columns} do not match {meta.name} ({site.caller}), "
                           f"whose columns in order are {meta.columns}"]
    for prop in spec.properties:
        out = output_for(spec, prop.output)
        if out is None:
            errors.append(f"property {prop.id} names output {prop.output!r}, which is not a contract return")
            continue
        known = {c.lower() for c in literal_columns(out.columns)}
        errors += [f"property {prop.id} uses column {c!r}, which return {out.name} does not list"
                   for c in sorted(prop.referenced_columns()) if c.lower() not in known]
    ids = [p.id for p in spec.properties]
    errors += [f"property id {i!r} is used more than once" for i in sorted(set(ids)) if ids.count(i) > 1]

    # 5. Risk ids exist; constructs without a behaviour note are declared unknown.
    errors += [f"logic step {l.step} cites risk {r!r}, which is not in the risk register"
               for l in spec.logic for r in l.risks if r not in RISKS]
    if behaviour_notes:
        declared = {c.lower() for c in spec.unknown_constructs}
        errors += [f"construct {c!r} has no behaviour note; list it in unknown_constructs"
                   for c in facts.constructs if c not in behaviour_notes and c.lower() not in declared]

    # 6. Parameters.
    omitted = {o.param.lower() for o in spec.contract.omitted_params}
    errors += [f"omitted parameter {o.param!r} is not a parameter of the unit"
               for o in spec.contract.omitted_params if o.param.lower() not in params]
    errors += [f"parameter {p!r} is never used in the macro; list it in omitted_params"
               for p in facts.unused_params if p.lower() not in omitted]
    return errors
