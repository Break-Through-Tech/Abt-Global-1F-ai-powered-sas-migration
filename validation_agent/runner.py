"""Step 2: run a translated program the way the runtime contract describes (no LLM).

- The script runs in a separate process with a timeout. Its environment has only PATH, SAS_INPUT_DIR,
  SAS_OUTPUT_DIR and PYTHONIOENCODING (plus SYSTEMROOT on Windows, which Python needs to start), so API
  keys never reach generated code (sas_agent/README.md, 6.6).
- pandas' ChainedAssignmentError is raised as an error (D-047): in pandas 3 a chained assignment
  (df[col][rows] = value) silently does nothing, so the script would finish with wrong values.
- stdout, stderr (full traceback) and exception.json go to the attempt's data/ folder, which is never sent
  to a model. The returned exception is {"type", "line", "source"}: never the message, which can contain
  data values (e.g. KeyError: '010001').
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from sas_agent import config as harness
from sas_agent import runs

from . import compare, config

# Runs the script as __main__ with chained assignment turned into an error.
_RUNNER = (
    "import runpy, sys, warnings\n"
    "import pandas\n"
    "warnings.simplefilter('error', pandas.errors.ChainedAssignmentError)\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)
_FRAME = re.compile(r'^\s*File "(.+)", line (\d+)')
_EXCEPTION = re.compile(r"^[A-Za-z_][\w.]*$")


def prepare_inputs(attempt_dir: Path) -> Path:
    """The run's input folder (runs/<run-id>/inputs), written once per run by sas_agent.runs."""
    run_dir = Path(attempt_dir).resolve().parents[1]   # runs/<run-id>/<program>/attempt-<n>
    inputs = run_dir / "inputs"
    if not (inputs / f"{harness.INPUT_DATASET.stem}.csv").is_file():
        runs.prepare_inputs(run_dir.name, run_dir.parent)
    return inputs


def seed_upstream(program_id: str, output_dir: Path) -> None:
    """Put the SAS results of the earlier programs in SAS_OUTPUT_DIR, as the runtime contract promises.

    Each program is validated on exactly what SAS gave the next step, so an error in an earlier
    translation can't make a later one fail. This program's own outputs are removed first.
    """
    output_dir = Path(output_dir)
    for program in harness.PROGRAMS:
        if program.id == program_id:
            for name in program.outputs:
                (output_dir / f"{name}.csv").unlink(missing_ok=True)
            return
        for name in program.outputs:
            compare.load_reference(name).to_csv(output_dir / f"{name}.csv", index=False)


def child_env(input_dir: Path, output_dir: Path) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", ""), "SAS_INPUT_DIR": str(input_dir),
           "SAS_OUTPUT_DIR": str(output_dir), "PYTHONIOENCODING": "utf-8"}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def _text(output) -> str:
    if output is None:
        return ""
    return output.decode("utf-8", "replace") if isinstance(output, bytes) else output


def parse_exception(stderr: str, script: Path, code: str) -> dict:
    """{"type", "line", "source"} from a traceback: the error type and the script line, never the message."""
    target = os.path.normcase(str(Path(script).resolve()))
    text = stderr.splitlines()
    frames = [(i, m) for i, m in enumerate(map(_FRAME.match, text)) if m]
    lines = [int(m.group(2)) for _, m in frames if os.path.normcase(str(Path(m.group(1)).resolve())) == target]
    line = lines[-1] if lines else None
    # The exception line is the first unindented line after the last frame; its message may span lines.
    after = text[frames[-1][0] + 1:] if frames else []
    header = next((s for s in after if s.strip() and not s[0].isspace()), "")
    name = header.split(":", 1)[0].strip()
    error_type = name.rsplit(".", 1)[-1] if _EXCEPTION.match(name) else "ExitCode"
    code_lines = code.splitlines()
    source = code_lines[line - 1].strip() if line and line <= len(code_lines) else None
    return {"type": error_type, "line": line, "source": source}


def run_script(script: Path, input_dir: Path, output_dir: Path, data_dir: Path,
               timeout: int | None = None) -> dict | None:
    """Run the script; return None if it exited cleanly, else the redacted exception."""
    script, data_dir = Path(script), Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    timeout = timeout or config.TIMEOUT_SECONDS
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(script)], cwd=data_dir, env=child_env(input_dir, output_dir),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as expired:
        stdout, stderr, returncode = _text(expired.stdout), _text(expired.stderr), None
    (data_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (data_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")
    if returncode == 0:
        return None

    if returncode is None:
        exception = {"type": "Timeout", "line": None, "source": None}
    else:
        exception = parse_exception(stderr, script, script.read_text(encoding="utf-8"))
    last = next((s.strip() for s in reversed((stderr or "").splitlines()) if s.strip()), "")
    (data_dir / "exception.json").write_text(json.dumps(
        {**exception, "returncode": returncode, "timeout_seconds": timeout, "message": last}, indent=2),
        encoding="utf-8")
    return exception
