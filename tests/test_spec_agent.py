"""The Spec agent with a fake model: no API calls."""

from pathlib import Path

import pytest

from sas_agent.spec import metadata
from sas_agent.spec.agent import SpecFailed, SpecUnchanged, generate_spec, revise_spec
from sas_agent.spec.schema import AgentLayers, LogicStep, Spec

FIXTURE = Path(__file__).parent / "fixtures" / "grp_score_spec.json"


class FakeModel:
    """Stands in for a LangChain chat model's with_structured_output(...).invoke(...)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def with_structured_output(self, schema):
        assert schema is AgentLayers
        return self

    def invoke(self, messages):
        self.prompts.append(messages[1].content)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture(scope="module")
def expected():
    return Spec.model_validate_json(FIXTURE.read_text())


@pytest.fixture(scope="module")
def inputs(expected):
    return {"analyzed": (expected.unit, expected.facts), "metadata": metadata.for_facts(expected.facts)}


def broken(layers: AgentLayers) -> AgentLayers:
    copy = layers.model_copy(deep=True)
    copy.logic[-1] = LogicStep(step="s9", text="x")
    return copy


def test_valid_on_first_try(expected, inputs):
    model = FakeModel(expected.layers())
    spec = generate_spec("M.grp_score", model, **inputs)
    assert spec == expected and len(model.prompts) == 1
    assert "s1" in model.prompts[0] and "R.Outcome_mortality" in model.prompts[0]


def test_check_errors_are_sent_back(expected, inputs):
    model = FakeModel(broken(expected.layers()), expected.layers())
    spec = generate_spec("M.grp_score", model, **inputs)
    assert spec.layers() == expected.layers()
    assert "'s9'" in model.prompts[1] and "failed these checks" in model.prompts[1]


def test_unparseable_answer_counts_as_a_try(expected, inputs):
    model = FakeModel(ValueError("bad json"), expected.layers())
    assert generate_spec("M.grp_score", model, **inputs) == expected
    assert "did not match the required structure" in model.prompts[1]


def test_three_failed_tries_raise(expected, inputs):
    bad = broken(expected.layers())
    with pytest.raises(SpecFailed) as err:
        generate_spec("M.grp_score", FakeModel(bad, bad, bad), **inputs)
    assert any("'s9'" in e for e in err.value.errors)


def test_revision_returns_next_version(expected, inputs):
    changed = expected.layers().model_copy(deep=True)
    changed.assumptions = ["the input is sorted by PROVIDER_ID"]
    model = FakeModel(changed)
    new = revise_spec(expected, ["p2 at program1:L43 on R.Outcome_mortality: FAILED"], model,
                      metadata=inputs["metadata"])
    assert new.version == 2 and new.assumptions == changed.assumptions
    assert "p2 at program1:L43" in model.prompts[0]


def test_unchanged_revision_raises(expected, inputs):
    with pytest.raises(SpecUnchanged):
        revise_spec(expected, ["evidence"], FakeModel(expected.layers()), metadata=inputs["metadata"])
