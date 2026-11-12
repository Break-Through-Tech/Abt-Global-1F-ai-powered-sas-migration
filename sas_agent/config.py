"""Paths, model settings, and the registry of SAS programs in the pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load secrets (e.g. GOOGLE_API_KEY) from <repo>/.env. Variables already set in the
# environment (such as Codespaces secrets) take precedence over the file.
load_dotenv(REPO_ROOT / ".env")
SAS_PROGRAMS_DIR = REPO_ROOT / "data" / "Project_1" / "SAS Programs"
MACROS_FILE = SAS_PROGRAMS_DIR / "Star_Macros.sas"
MAPPING_FILE = REPO_ROOT / "SAS_CSV_Mapping.txt"

MIGRATED_DIR = REPO_ROOT / "migrated"  # analysis + translated code + documentation per program

# Any LangChain `init_chat_model` string works; "google_genai:" uses the Gemini Developer API
# (reads GOOGLE_API_KEY). Model codes: https://ai.google.dev/gemini-api/docs/models
MODEL = os.getenv("SAS_AGENT_MODEL", "google_genai:gemini-3.8-flash")


@dataclass(frozen=True)
class SasProgram:
    id: str
    sas_file: str
    outputs: list[str] = field(default_factory=list)  # dataset names written to library R

    @property
    def path(self) -> Path:
        return SAS_PROGRAMS_DIR / self.sas_file


PROGRAMS = [
    SasProgram(
        "program0",
        "0 - Data and Measure Standardization_2025Jul.sas",
        ["STD_DATA_2025JUL_ANALYSIS", "MEASURE_AVERAGE_STDDEV_2025JUL", "LESS100_MEASURE"],
    ),
    SasProgram(
        "program1",
        "1 - First stage_Simple Average of Measure Scores_2025Jul.sas",
        ["OUTCOME_MORTALITY", "OUTCOME_SAFETY", "OUTCOME_READMISSION", "PTEXP", "PROCESS"],
    ),
    SasProgram(
        "program2",
        "2 - Second Stage_Weighted Average and Categorize Star_2025Jul.sas",
        ["STAR_2025JUL", "NATIONAL_AVERAGE_2025JUL"],
    ),
]


def get_program(program_id: str) -> SasProgram:
    for p in PROGRAMS:
        if p.id == program_id:
            return p
    raise KeyError(f"Unknown program {program_id!r}; choose from {[p.id for p in PROGRAMS]}")
