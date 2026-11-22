"""Metadata of the SAS reference datasets a unit's call sites read or write: names, types, row counts.

Only metadata is read here (pyreadstat metadataonly=True). It may be shown to the model (Q-02).
"""

import re
from pathlib import Path

import pyreadstat
from pydantic import BaseModel

from .. import config
from .schema import Facts

REFERENCE_DIR = config.REPO_ROOT / "data" / "Project_1" / "SAS Output"


class DatasetMeta(BaseModel):
    name: str                 # as written at the call site, e.g. "R.Outcome_mortality"
    columns: list[str]
    types: list[str]          # "double" or "string", per column
    rows: int


def reference_path(dataset: str, reference_dir: Path | None = None) -> Path | None:
    """The saved .sas7bdat for a library-R dataset name ("R.Outcome_mortality"), or None."""
    m = re.fullmatch(r"[Rr]\.(\w+)", dataset.strip())
    if not m:
        return None
    path = (reference_dir or REFERENCE_DIR) / f"{m.group(1).lower()}.sas7bdat"
    return path if path.exists() else None


def read_metadata(dataset: str, reference_dir: Path | None = None) -> DatasetMeta | None:
    path = reference_path(dataset, reference_dir)
    if path is None:
        return None
    _, meta = pyreadstat.read_sas7bdat(str(path), metadataonly=True)
    return DatasetMeta(name=dataset, columns=list(meta.column_names),
                       types=[meta.readstat_variable_types[c] for c in meta.column_names],
                       rows=meta.number_rows)


def for_facts(facts: Facts, reference_dir: Path | None = None) -> dict[str, DatasetMeta]:
    """Metadata of every call-site argument that names a saved dataset, keyed by the argument text."""
    found = {}
    for site in facts.call_sites:
        for arg in site.args:
            if arg.value not in found and (meta := read_metadata(arg.value, reference_dir)):
                found[arg.value] = meta
    return found
