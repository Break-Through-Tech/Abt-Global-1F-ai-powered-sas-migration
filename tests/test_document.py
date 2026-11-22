"""Offline tests for the documentation agent: no API key, fake chat model."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sas_agent import config
from sas_agent.document import MAPPING_FILE, build_prompt, document
from sas_agent.mapping import mapping_heading, parse_mapping


def test_program0_mapping_lists_its_outputs_only():
    block = parse_mapping(MAPPING_FILE.read_text(encoding="utf-8"), mapping_heading("program0"))
    for name in config.get_program("program0").outputs:
        assert name in block
    assert "OUTCOME_MORTALITY" not in block


def test_every_program_has_a_mapping_block():
    text = MAPPING_FILE.read_text(encoding="utf-8")
    for program in config.PROGRAMS:
        assert parse_mapping(text, mapping_heading(program.id))


def test_parse_mapping_rejects_a_missing_program():
    with pytest.raises(ValueError):
        parse_mapping("Program 0: x\n- A.csv\n", "Program 9:")


def test_prompt_is_filled_and_keeps_code_braces():
    code = "spec = {'a': 1}\nprint(f'{spec}')\n"
    prompt = build_prompt("program0", code=code, analysis="## Analysis")
    assert code in prompt and "## Analysis" in prompt
    assert "%keep_hos" in prompt  # SAS program text
    assert "LESS100_MEASURE" in prompt  # outputs and mapping
    for placeholder in ("{mapping}", "{outputs}", "{analysis}", "{code}", "{macros}", "{sas_file}", "{sas_code}"):
        assert placeholder not in prompt


def _promote(migrated_dir):
    folder = migrated_dir / "program0"
    folder.mkdir(parents=True)
    (folder / "program0.py").write_text("result = 1\n", encoding="utf-8")
    (folder / "analysis.md").write_text("## Analysis\n", encoding="utf-8")
    return folder


def test_document_writes_utf8_next_to_the_promoted_script(tmp_path):
    folder = _promote(tmp_path)
    docs = "# Program 0\nSAS → pandas; measures with volume ≤ 100 are dropped.\n"
    out = document("program0", llm=FakeListChatModel(responses=[docs]), migrated_dir=tmp_path)
    assert out == folder / "documentation.md"
    assert out.read_text(encoding="utf-8") == docs


def test_document_needs_a_promoted_translation(tmp_path):
    with pytest.raises(FileNotFoundError, match="--fix-loop program0"):
        document("program0", llm=FakeListChatModel(responses=["x"]), migrated_dir=tmp_path)
