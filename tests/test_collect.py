from sas_agent.config import get_program
from sas_agent.graph import build_graph
from sas_agent.mapping import mapping_heading, parse_mapping


def test_program0_resolves_to_standardization_sas():
    program = get_program("program0")
    assert program.path.is_file()
    assert program.sas_file == "0 - Data and Measure Standardization_2025Jul.sas"
    assert program.path.name == program.sas_file
    assert "%keep_hos" in program.path.read_text(encoding="utf-8", errors="replace")
    assert program.outputs == [
        "STD_DATA_2025JUL_ANALYSIS",
        "MEASURE_AVERAGE_STDDEV_2025JUL",
        "LESS100_MEASURE",
    ]


def test_program0_mapping_lists_expected_outputs():
    from sas_agent import config

    block = parse_mapping(
        config.MAPPING_FILE.read_text(encoding="utf-8"),
        mapping_heading("program0"),
    )
    for name in get_program("program0").outputs:
        assert name in block or f"{name}.csv" in block
    assert "OUTCOME_MORTALITY" not in block


def test_parse_mapping_returns_program0_block_only():
    text = (
        "SAS Program to CSV Output Mapping\n\n"
        "Program 0: Data and Measure Standardization\n"
        "- LESS100_MEASURE.csv: dropped measures.\n"
        "- MEASURE_AVERAGE_STDDEV_2025JUL.csv: mean and std.\n"
        "- STD_DATA_2025JUL_ANALYSIS.csv: standardized scores.\n\n"
        "Program 1: First Stage\n"
        "- OUTCOME_MORTALITY.csv: mortality domain.\n"
    )
    block = parse_mapping(text, "Program 0:")
    assert "LESS100_MEASURE.csv" in block
    assert "STD_DATA_2025JUL_ANALYSIS.csv" in block
    assert "OUTCOME_MORTALITY" not in block


def test_graph_runs_document_after_save():
    from unittest.mock import MagicMock

    app = build_graph(llm=MagicMock())
    nodes = set(app.get_graph().nodes)
    assert {"analyze", "translate", "save", "document"} <= nodes
    edges = {(edge[0], edge[1]) for edge in app.get_graph().edges}
    assert ("save", "document") in edges