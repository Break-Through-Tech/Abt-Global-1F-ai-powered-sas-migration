"""Settings of the Validation agent.

The API key (GOOGLE_API_KEY or GEMINI_API_KEY) is read from the repository's .env file, the same file the
other agents use: importing sas_agent.config loads it with load_dotenv(). Variables already set in the
environment take precedence over the file. Only the LLM review needs the key.
"""

import os

from sas_agent import config as harness  # loads <repo>/.env

REFERENCE_DIR = harness.REPO_ROOT / "data" / "Project_1" / "SAS Output"  # the SAS results (.sas7bdat)

# Any init_chat_model "provider:model" string; defaults to the translator's model.
MODEL = os.getenv("VALIDATION_AGENT_MODEL", harness.MODEL)

REVIEW_BLOCKS = True       # a major review finding fails the attempt even if every check passes
RTOL, ATOL = 1e-6, 1e-8    # numeric tolerance when comparing with SAS
ID_COLUMN = "PROVIDER_ID"  # row order is checked on this column when an output has it
TIMEOUT_SECONDS = int(os.getenv("VALIDATION_TIMEOUT_SECONDS", "900"))  # per script run
STAGES = ["static_check", "execute", "compare"]  # how far an attempt got; used in the rank
