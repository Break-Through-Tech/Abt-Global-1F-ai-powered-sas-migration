from pathlib import Path

from sas_agent.spec.analyzer import analyze_macros, statements, unit_source
from sas_agent.spec.schema import Spec

FIXTURE = Path(__file__).parent / "fixtures" / "grp_score_spec.json"


def test_all_macros_found_with_line_ranges():
    analyzed = analyze_macros()
    lines = {uid: unit.lines for uid, (unit, _) in analyzed.items()}
    assert lines == {
        "M.keep_hos": (5, 20), "M.grp_score": (28, 63), "M.report": (71, 151), "M.kmeans": (157, 237),
        "M.nation_avg0": (243, 256), "M.nation_avg_peer": (258, 279), "M.nation_avg": (282, 295),
    }


def test_grp_score_facts_match_appendix_a():
    # The fixture's facts were written by hand from design doc Appendix A, not from the analyzer.
    expected = Spec.model_validate_json(FIXTURE.read_text())
    unit, facts = analyze_macros()["M.grp_score"]
    assert facts == expected.facts
    assert unit == expected.unit


def test_keyword_parameters_and_call_sites():
    _, facts = analyze_macros()["M.kmeans"]
    assert facts.params == ["in", "out", "grp"]
    assert [(c.caller, c.arg("in"), c.arg("out")) for c in facts.call_sites] == [
        ("program2:L138", "cnt_s10", "cnt_s1"), ("program2:L144", "cnt_s20", "cnt_s2"), ("program2:L150", "cnt_s30", "cnt_s3"),
    ]
    assert facts.output_params == ["out"]


def test_let_values_are_resolved_in_program_order():
    _, facts = analyze_macros()["M.nation_avg0"]
    assert facts.call_sites[0].arg("indata") == "R.Star_2025Jul"     # &RESULTS, set in Program 0
    _, facts = analyze_macros()["M.keep_hos"]
    assert facts.call_sites[0].arg("indsn") == "Initial_data_2025Jul"


def test_macro_variables_used_are_not_parameters():
    _, facts = analyze_macros()["M.report"]
    assert facts.macro_vars_used == ["measure_OM", "measure_OS", "measure_OR", "measure_PtExp", "measure_Process"]


def test_statements_skip_comments_and_keep_line_numbers():
    text = "/* a\nblock */ data x;\n* a comment;\nset y; run;\n"
    found = [(s.text, s.line) for s in statements(text)]
    assert found == [("data x", 2), ("set y", 4), ("run", 4)]


def test_unit_source_is_the_macro_text():
    unit, _ = analyze_macros()["M.grp_score"]
    source = unit_source(unit)
    assert source.startswith("%macro grp_score") and source.rstrip().endswith("%mend;")
