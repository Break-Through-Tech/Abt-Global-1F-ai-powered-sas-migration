from pathlib import Path

import pytest

from sas_agent.spec import metadata
from sas_agent.spec.checks import check, match_columns
from sas_agent.spec.schema import LogicStep, OmittedParam, Property, Spec

FIXTURE = Path(__file__).parent / "fixtures" / "grp_score_spec.json"


@pytest.fixture(scope="module")
def meta():
    return metadata.for_facts(Spec.model_validate_json(FIXTURE.read_text()).facts)


@pytest.fixture
def spec():
    return Spec.model_validate_json(FIXTURE.read_text())


def test_fixture_passes(spec, meta):
    assert check(spec, meta) == []


def test_logic_must_match_facts_steps(spec, meta):
    spec.logic[-1] = LogicStep(step="s9", text="x")
    errors = check(spec, meta)
    assert any("'s9'" in e for e in errors) and any("facts step s5" in e for e in errors)


def test_unused_parameter_must_be_omitted(spec, meta):
    spec.contract.omitted_params = [OmittedParam(param="nmeasure", reason="equals len(varlist)")]
    assert any("'gp'" in e and "never used" in e for e in check(spec, meta))


def test_unknown_risk_id(spec, meta):
    spec.logic[0].risks = ["S-99"]
    assert any("S-99" in e for e in check(spec, meta))


def test_returns_must_match_reference_columns(spec, meta):
    spec.contract.returns[0].columns = spec.contract.returns[0].columns[:-1]     # drop grp_score
    errors = check(spec, meta)
    assert any("do not match R.Outcome_mortality" in e for e in errors)
    assert any("'grp_score'" in e for e in errors)                              # properties use it


def test_return_must_be_an_output(spec, meta):
    spec.contract.returns[0].name = "dat0"
    assert any("'dat0' is not an output" in e for e in check(spec, meta))


def test_property_columns_must_be_listed(spec, meta):
    spec.properties.append(Property(id="p6", kind="constant", description="x", columns=["nope"]))
    assert any("'nope'" in e for e in check(spec, meta))


def test_constructs_need_notes_once_notes_exist(spec, meta):
    notes = {"array": "..."}
    errors = check(spec, meta, behaviour_notes=notes)
    assert any("'sum_of'" in e for e in errors) and not any("'array'" in e for e in errors)
    spec.unknown_constructs = [c for c in spec.facts.constructs if c != "array"]
    assert check(spec, meta, behaviour_notes=notes) == []


def test_column_patterns():
    pattern = ["ID", "*varlist", "C1..Cn", "total"]
    assert match_columns(pattern, ["ID", "a", "b", "C1", "C2", "total"])
    assert not match_columns(pattern, ["ID", "C1", "total"])           # *varlist needs one column or more
    assert not match_columns(pattern, ["ID", "a", "C2", "total"])      # the series starts at C1
    assert not match_columns(pattern, ["ID", "a", "C1", "C2"])         # total missing
