import json
from dataclasses import asdict
from pathlib import Path

import pytest

from sas_agent.executor import execute


@pytest.fixture
def dirs(tmp_path: Path) -> dict:
    """Input folder with a one-row CSV, plus empty output and work folders."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "in.csv").write_text("PROVIDER_ID,x\n010001,1.5\n")
    return {"input_dir": inputs, "output_dir": tmp_path / "outputs", "work_dir": tmp_path / "work"}


def run(tmp_path: Path, dirs: dict, code: str, **kwargs):
    script = tmp_path / "script.py"
    script.write_text(code)
    return execute(script, **dirs, **kwargs)


def test_successful_script_writes_outputs(tmp_path, dirs):
    result = run(tmp_path, dirs, (
        "import os, pandas as pd\n"
        "df = pd.read_csv(os.path.join(os.environ['SAS_INPUT_DIR'], 'in.csv'), dtype={'PROVIDER_ID': str})\n"
        "df.to_csv(os.path.join(os.environ['SAS_OUTPUT_DIR'], 'OUT.csv'), index=False)\n"
    ))
    assert result.status == "ok"
    assert result.exception is None
    assert result.files_written == ["OUT.csv"]


def test_exception_message_with_data_value_stays_in_data_zone(tmp_path, dirs):
    result = run(tmp_path, dirs, (
        "import os, pandas as pd\n"
        "df = pd.read_csv(os.path.join(os.environ['SAS_INPUT_DIR'], 'in.csv'), dtype={'PROVIDER_ID': str})\n"
        "lookup = {}\n"
        "pid = df['PROVIDER_ID'][0]\n"
        "value = lookup[pid]\n"
    ))
    assert result.status == "error"
    assert result.exception == {"type": "KeyError", "line": 5, "source": "value = lookup[pid]"}
    # The message (KeyError: '010001') is not in the result, but the full traceback is kept in the data zone.
    assert "010001" not in json.dumps(asdict(result))
    assert "010001" in (dirs["work_dir"] / "stderr.txt").read_text()


def test_chained_assignment_becomes_an_error(tmp_path, dirs):
    result = run(tmp_path, dirs, (
        "import pandas as pd\n"
        "df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})\n"
        "df['a'][df['b'] > 3] = 99\n"
    ))
    assert result.status == "error"
    assert result.exception["type"] == "ChainedAssignmentError"
    assert result.exception["line"] == 3


def test_api_key_is_not_passed_to_the_script(tmp_path, dirs, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-secret-value")
    run(tmp_path, dirs, (
        "import os\n"
        "open(os.path.join(os.environ['SAS_OUTPUT_DIR'], 'env.txt'), 'w').write(str('GOOGLE_API_KEY' in os.environ))\n"
    ))
    assert (dirs["output_dir"] / "env.txt").read_text() == "False"


def test_syntax_error_is_reported_with_line(tmp_path, dirs):
    result = run(tmp_path, dirs, "x = 1\nheader_ cols = ['a']\n")
    assert result.status == "error"
    assert result.exception["type"] == "SyntaxError"
    assert result.exception["line"] == 2


def test_exit_codes(tmp_path, dirs):
    assert run(tmp_path, dirs, "import sys\nsys.exit(0)\n").status == "ok"
    failed = run(tmp_path, dirs, "import sys\nsys.exit(3)\n")
    assert failed.status == "error"
    assert failed.exception["type"] == "SystemExit"


def test_timeout(tmp_path, dirs):
    result = run(tmp_path, dirs, "import time\ntime.sleep(30)\n", timeout=1)
    assert result.status == "timeout"
    assert result.exception is None
