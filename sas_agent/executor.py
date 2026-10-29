"""Executor: runs a generated script in a separate process (design doc 6.12.3, D-047)."""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RUNNER = Path(__file__).with_name("runner.py")
TIMEOUT_SECONDS = 300

# Only these variables reach the generated script, plus SAS_INPUT_DIR and SAS_OUTPUT_DIR.
# config.py loads .env into this process's environment, so passing os.environ through would hand
# the script the API key. The OS variables are needed on some platforms and hold no secrets.
_PASSED_THROUGH = ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP")


@dataclass(frozen=True)
class ExecResult:
    """How a run ended. Safe for the AI zone: stdout and stderr stay in work_dir."""

    status: str               # "ok", "error", or "timeout"
    duration_s: float
    files_written: list[str]  # names of the files in output_dir
    exception: dict | None    # {"type", "line", "source"} without the message; set only when status is "error"


def execute(
    script: Path, input_dir: Path, output_dir: Path, work_dir: Path, timeout: float = TIMEOUT_SECONDS
) -> ExecResult:
    """Run `script` under the runner and report how it ended.

    Notes:
    - input_dir and output_dir reach the script as SAS_INPUT_DIR and SAS_OUTPUT_DIR (runtime contract).
    - work_dir is in the data zone. It is the script's working directory and receives stdout.txt,
      stderr.txt (with the full traceback), and exception.json.
    - `-I` (isolated mode) makes Python ignore PYTHON* variables and the user's site-packages.
    - On timeout, subprocess.run kills the process; files it had already written are still listed.
    """
    output_dir, work_dir = Path(output_dir), Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    exception_json = work_dir / "exception.json"
    exception_json.unlink(missing_ok=True)

    env = {k: os.environ[k] for k in _PASSED_THROUGH if k in os.environ}
    env["SAS_INPUT_DIR"] = str(Path(input_dir).resolve())
    env["SAS_OUTPUT_DIR"] = str(output_dir.resolve())
    cmd = [sys.executable, "-I", str(RUNNER), str(Path(script).resolve()), str(exception_json)]

    start = time.monotonic()
    with open(work_dir / "stdout.txt", "w") as out, open(work_dir / "stderr.txt", "w") as err:
        try:
            returncode = subprocess.run(cmd, cwd=work_dir, env=env, stdout=out, stderr=err, timeout=timeout).returncode
            status = "ok" if returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            status = "timeout"
    duration_s = round(time.monotonic() - start, 3)

    exception = None
    if status == "error":
        # The runner writes exception.json for every failure it catches. If it is missing, the process
        # died before the runner could report (e.g. it was killed or ran out of memory).
        exception = (
            json.loads(exception_json.read_text()) if exception_json.exists()
            else {"type": "ProcessDied", "line": None, "source": None}
        )
    files_written = sorted(p.name for p in output_dir.iterdir() if p.is_file())
    return ExecResult(status, duration_s, files_written, exception)
