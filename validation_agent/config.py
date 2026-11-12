"""Paths and settings. The program list (SAS file, required outputs) comes from sas_agent/config.py."""

import os

from sas_agent.config import MACROS_FILE, PROGRAMS, REPO_ROOT, get_program  # noqa: F401
from sas_agent.config import MODEL as TRANSLATOR_MODEL

DATA_DIR = REPO_ROOT / "data" / "Project_1"
SAS_INPUT_DIR = DATA_DIR / "Starrating"          # SAS_INPUT_DIR for the scripts: alldata_2025jul.csv
SAS_CSV_DIR = DATA_DIR / "SAS Output CSV"        # SAS results as CSV: what later programs read
SAS_DATASET_DIR = DATA_DIR / "SAS Output"        # SAS results as .sas7bdat: the reference we compare with
RUNS_DIR = REPO_ROOT / "runs"                    # default place for attempt folders

ALLOWED_PACKAGES = frozenset({"pandas", "numpy"})   # plus the standard library (D-046)
ID_COLUMN = "PROVIDER_ID"                        # read as text: leading zeros
RTOL, ATOL = 1e-6, 1e-8                          # numeric tolerance vs SAS
TIMEOUT_S = 300                                  # max run time of a script

# The only environment variables the script gets, besides SAS_INPUT_DIR and SAS_OUTPUT_DIR
# (passing os.environ through would hand the script the API key from .env).
PASSED_THROUGH_ENV = ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP")

# Code review (LLM). Any LangChain init_chat_model string; defaults to the translator's Gemini model.
MODEL = os.getenv("VALIDATOR_MODEL", TRANSLATOR_MODEL)
REVIEW_BLOCKS = True                             # a "major" review finding fails the attempt even if all checks pass
