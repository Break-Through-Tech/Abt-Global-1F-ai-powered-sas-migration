"""Run folders (design doc 6.12.6, D-050).

    runs/<run-id>/inputs/                  input files the scripts read (SAS_INPUT_DIR)
    runs/<run-id>/<program>/attempt-<n>/
        ai/    what the model may see: script, prompts and responses, fix requests, redacted reports
        data/  data zone: outputs/, stdout.txt, stderr.txt, exception.json, full reports

`runs/` is excluded from Git.
"""

import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyreadstat

from . import config


@dataclass(frozen=True)
class AttemptDirs:
    root: Path

    @property
    def ai(self) -> Path:
        return self.root / "ai"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def outputs(self) -> Path:
        return self.data / "outputs"


def new_run_id() -> str:
    """e.g. 20261029-142501-3fa2: sortable by time, with a random suffix so parallel runs don't clash."""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def attempt_dirs(run_id: str, program_id: str, attempt: int, runs_dir: Path | None = None) -> AttemptDirs:
    dirs = AttemptDirs((runs_dir or config.RUNS_DIR) / run_id / program_id / f"attempt-{attempt}")
    dirs.ai.mkdir(parents=True, exist_ok=True)
    dirs.outputs.mkdir(parents=True, exist_ok=True)
    return dirs


def prepare_inputs(run_id: str, runs_dir: Path | None = None) -> Path:
    """Write the run's input folder and return it.

    Notes:
    - The input CSV is generated from the .sas7bdat file, the data SAS actually read. The CSV shipped
      in data/ differs from it in 9,494 of 230,595 values, by at most 1.1e-16 (checked 2026-10-28).
      pandas writes floats exactly, and reading with float_precision="round_trip" reads them back
      exactly (design doc 6.12.4), so scripts see exactly what SAS saw.
    - PROVIDER_ID stays a string; missing values are written as empty fields.
    """
    inputs = (runs_dir or config.RUNS_DIR) / run_id / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    df, _ = pyreadstat.read_sas7bdat(str(config.INPUT_DATASET))
    df.to_csv(inputs / f"{config.INPUT_DATASET.stem}.csv", index=False)
    return inputs
