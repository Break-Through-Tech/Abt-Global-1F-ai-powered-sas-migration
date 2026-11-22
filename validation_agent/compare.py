"""Step 3: compare the CSVs a script wrote with the SAS results (no LLM).

The SAS results are the .sas7bdat files in data/Project_1/SAS Output/ (full precision). The report holds
names and counts only, never data values.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from sas_agent import config as harness

from . import config
from .schema import Check, RedactedReport


@lru_cache(maxsize=16)
def _read_reference(name: str) -> pd.DataFrame:
    frame, _ = pyreadstat.read_sas7bdat(str(config.REFERENCE_DIR / f"{name.lower()}.sas7bdat"))
    return frame


def load_reference(name: str) -> pd.DataFrame:
    """The SAS result for an output name (e.g. "OUTCOME_MORTALITY"). Text columns are strings."""
    return _read_reference(name).copy()


def expected_outputs(program_id: str) -> dict[str, pd.DataFrame]:
    """The SAS result for each dataset the program writes to library R."""
    return {name: load_reference(name) for name in harness.get_program(program_id).outputs}


def _names(columns: list[str], limit: int = 20) -> str:
    shown = ", ".join(columns[:limit])
    return shown + (f" and {len(columns) - limit} more" if len(columns) > limit else "")


def column_differences(actual: list[str], expected: list[str]) -> tuple[int, str]:
    """How many column names differ, and which (names only)."""
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    by_lower = {c.lower(): c for c in extra}
    case = [(c, by_lower[c.lower()]) for c in missing if c.lower() in by_lower]
    missing = [c for c in missing if c not in {e for e, _ in case}]
    extra = [c for c in extra if c not in {a for _, a in case}]

    parts = []
    if case:
        parts.append("letter case differs: " + ", ".join(f"expected {e}, got {a}" for e, a in case[:20]))
    if missing:
        parts.append(f"missing: {_names(missing)}")
    if extra:
        parts.append(f"extra: {_names(extra)}")
    if parts:
        return len(case) + len(missing) + len(extra), "; ".join(parts)
    first = next(i for i, (a, e) in enumerate(zip(actual, expected)) if a != e)
    return (sum(a != e for a, e in zip(actual, expected)),
            f"same columns in a different order; first difference at position {first + 1}: "
            f"expected {expected[first]}, got {actual[first]}")


def compare_dataset(name: str, py: pd.DataFrame, sas: pd.DataFrame) -> list[Check]:
    """Columns, row count, row order, then the missing values and values of every shared column."""
    py, sas = py.reset_index(drop=True), sas.reset_index(drop=True)
    checks = []
    if list(py.columns) != list(sas.columns):   # names and order must match SAS
        count, detail = column_differences(list(py.columns), list(sas.columns))
        checks.append(Check(dataset=name, check="columns", mismatch_count=count, detail=detail))
    if len(py) != len(sas):
        return checks + [Check(dataset=name, check="row_count", mismatch_count=abs(len(py) - len(sas)),
                               detail=f"expected {len(sas)} rows, got {len(py)}")]
    id_col = config.ID_COLUMN
    if id_col in sas.columns and id_col in py.columns:
        wrong = int((py[id_col].astype(str).to_numpy() != sas[id_col].astype(str).to_numpy()).sum())
        if wrong:
            return checks + [Check(dataset=name, column=id_col, check="row_order", mismatch_count=wrong,
                                   detail=f"rows are not in the SAS order of {id_col}")]

    for col in [c for c in sas.columns if c in py.columns]:
        a, e = py[col], sas[col]
        if not pd.api.types.is_numeric_dtype(e):           # text: a blank equals missing (S-20)
            a, e = (s.astype("string").str.strip().replace("", pd.NA) for s in (a, e))
        a_na, e_na = a.isna().to_numpy(), e.isna().to_numpy()
        if (a_na != e_na).any():
            checks.append(Check(dataset=name, column=col, check="missing_values",
                                mismatch_count=int((a_na != e_na).sum())))
        both = ~a_na & ~e_na
        if pd.api.types.is_numeric_dtype(e) and pd.api.types.is_numeric_dtype(a):
            av, ev = a.to_numpy(float), e.to_numpy(float)
            bad = both & ~np.isclose(av, ev, rtol=config.RTOL, atol=config.ATOL)
            max_error = float(np.abs(av - ev)[bad].max()) if bad.any() else None
        else:
            bad, max_error = both & (a.astype(str).to_numpy() != e.astype(str).to_numpy()), None
        if bad.any():
            checks.append(Check(dataset=name, column=col, check="values", mismatch_count=int(bad.sum()),
                                max_error=max_error))
    return checks


def read_output(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    """Read a CSV the script wrote: exact numbers, and the SAS text columns (e.g. PROVIDER_ID) as strings."""
    header = pd.read_csv(path, nrows=0).columns
    text = {c: str for c in header if c in expected.columns and not pd.api.types.is_numeric_dtype(expected[c])}
    return pd.read_csv(path, dtype=text, float_precision="round_trip")


def compare_outputs(program_id: str, output_dir: Path, expected: dict[str, pd.DataFrame]) -> RedactedReport:
    """Each expected output vs <output_dir>/<NAME>.csv."""
    files = {p.name.lower(): p for p in Path(output_dir).glob("*") if p.is_file()}
    checks = []
    for name, sas in expected.items():
        path = Path(output_dir) / f"{name}.csv"
        if not path.is_file():
            other = files.get(f"{name}.csv".lower())
            detail = (f"found {other.name}; the file must be named {name}.csv" if other
                      else f"{name}.csv was not written to SAS_OUTPUT_DIR")
            checks.append(Check(dataset=name, check="file_missing", mismatch_count=1, detail=detail))
            continue
        try:
            py = read_output(path, sas)
        except Exception as error:   # empty or malformed CSV; the message may contain data, so keep the type
            checks.append(Check(dataset=name, check="unreadable", mismatch_count=1,
                                detail=f"{name}.csv could not be read as CSV ({type(error).__name__})"))
            continue
        checks += compare_dataset(name, py, sas)
    errors = [c.max_error for c in checks if c.max_error is not None]
    return RedactedReport(program_id=program_id, passed=not checks, failing_checks=len(checks),
                          total_mismatches=sum(c.mismatch_count for c in checks),
                          max_error=max(errors) if errors else None, checks=checks)
