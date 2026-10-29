"""The interface between the graph and the comparator (design doc 6.6 and 6.12.4).

The comparator itself is written by its owner in the orchestration subgroup (Q-16). This module fixes
only what the graph calls and the redacted result it reads. The comparator may add checks and fields,
but the full report, with any data values, stays in the work folder it is given and never returns here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Ranges for the largest difference in a column, smallest first (design doc 6.6).
ERROR_BINS = ("<1e-12", "<1e-9", "<1e-6", "<1e-3", ">=1e-3")


@dataclass(frozen=True)
class CheckResult:
    """One check on one dataset or column. Every field is safe for the AI zone."""

    dataset: str                       # e.g. "STD_DATA_2025JUL_ANALYSIS"
    check: str                         # e.g. "file_exists", "columns", "row_count", "keys", "missing", "values"
    passed: bool
    column: str | None = None
    mismatch: int | str | None = None  # mismatched rows; counts from 1 to 10 are reported as "<11"
    error_bin: str | None = None       # one of ERROR_BINS

    def __post_init__(self):
        # Comparators built on pandas tend to produce numpy values (numpy.bool, numpy.int64), which
        # the graph cannot write as JSON. Store plain Python values instead.
        object.__setattr__(self, "passed", bool(self.passed))
        if self.mismatch is not None and not isinstance(self.mismatch, str):
            object.__setattr__(self, "mismatch", int(self.mismatch))
        if self.error_bin is not None and self.error_bin not in ERROR_BINS:
            raise ValueError(f"error_bin must be one of {ERROR_BINS}, not {self.error_bin!r}")


@dataclass(frozen=True)
class RedactedReport:
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        # A report with no checks is a comparator fault, not a pass.
        return bool(self.checks) and all(c.passed for c in self.checks)


class Comparator(Protocol):
    def __call__(self, program_id: str, output_dir: Path, work_dir: Path) -> RedactedReport:
        """Compare the files a program wrote in output_dir with its SAS reference outputs.

        Write the full report into work_dir, which is in the data zone, and return only the redacted report.
        """
        ...
