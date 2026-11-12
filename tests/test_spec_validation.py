from pathlib import Path

import pytest

from sas_agent.spec.schema import Spec
from sas_agent.spec.validation import validate

FIXTURE = Path(__file__).parent / "fixtures" / "grp_score_spec.json"
OUTPUTS = {"R.Outcome_mortality", "R.Outcome_safety", "R.Outcome_readmission", "R.PtExp", "R.Process"}


@pytest.fixture
def spec():
    return Spec.model_validate_json(FIXTURE.read_text())


def test_fixture_properties_hold_on_all_five_sas_outputs(spec):
    items = validate(spec)
    assert {i.dataset for i in items} == OUTPUTS
    assert len(items) == 5 * (1 + len(spec.properties))
    assert all(i.passed for i in items), [i.describe() for i in items if not i.passed]


def test_wrong_divisor_fails(spec):
    next(p for p in spec.properties if p.id == "p2").ddof = 0
    failed = [i for i in validate(spec) if not i.passed]
    assert {(i.check, i.dataset) for i in failed} == {("p2", d) for d in OUTPUTS}


def test_wrong_missing_rule_fails_with_row_counts(spec):
    next(p for p in spec.properties if p.id == "p1").when = "total_cnt == 1"
    failed = [i for i in validate(spec) if not i.passed]
    assert failed and all(i.check == "p1" and i.failing_rows not in (None, 0) for i in failed)


def test_results_contain_no_data_values(spec):
    for item in validate(spec):
        dumped = item.model_dump()
        assert set(dumped) == {"check", "call_site", "dataset", "passed", "failing_rows"}
        assert dumped["failing_rows"] is None or dumped["failing_rows"] == "<11" or isinstance(dumped["failing_rows"], int)
