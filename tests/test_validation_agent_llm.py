"""Validation agent tests that call the real review model (Gemini).

They cost API calls, so they only run when asked:

    RUN_LLM_TESTS=1 uv run pytest tests/test_validation_agent_llm.py -s

The key (GOOGLE_API_KEY or GEMINI_API_KEY) is read from the repository's .env. The model's findings vary
from run to run, so the tests check the shape of the answer and the clearest problems only.
"""

import os

import pytest

import sas_agent.config  # noqa: F401  loads <repo>/.env, so the skip condition sees the key
from test_validation_agent import PRIVATE_ID, WRONG_VALUES, attempt, expected, runs_dir, validate  # noqa: F401

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LLM_TESTS") != "1" or not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="set RUN_LLM_TESTS=1 and an API key in .env to run the tests that call the model",
)

# A program0 translation with obvious hard-coding: a fixed subset of measures, literal national means,
# and the population standard deviation, where SAS computes all of them from the data.
HARDCODED_PROGRAM0 = '''import os
import pandas as pd

inp = pd.read_csv(os.path.join(os.environ["SAS_INPUT_DIR"], "alldata_2025jul.csv"),
                  dtype={"PROVIDER_ID": str}, float_precision="round_trip")
measures = ["MORT_30_AMI", "MORT_30_CABG"]
means = {"MORT_30_AMI": 0.1234, "MORT_30_CABG": 0.0321}
std = inp[["PROVIDER_ID"]].copy()
for m in measures:
    std["std_" + m] = (inp[m] - means[m]) / inp[m].std(ddof=0)
std.to_csv(os.path.join(os.environ["SAS_OUTPUT_DIR"], "STD_DATA_2025JUL_ANALYSIS.csv"), index=False)
'''


def test_review_flags_hard_coded_values():
    from validation_agent.review import build_llm, llm_review

    findings = llm_review(build_llm(), "program0", HARDCODED_PROGRAM0, "")
    print("\n".join(f"[{f.severity}] {f.kind} {f.python_lines} / {f.sas_lines}: {f.problem}" for f in findings))
    assert findings, "the review found nothing in a script with literal means"
    assert any(f.severity == "major" for f in findings)
    assert all(f.python_lines for f in findings)
    assert not any("```" in f.what_to_check for f in findings), "the review must not write code"


def test_graph_with_the_review_on_a_failing_script(runs_dir, expected):  # noqa: F811 (fixtures)
    result = validate("program0", WRONG_VALUES, attempt(runs_dir))
    print(result.fix_request)
    assert result.next_state == "TRIAGED" and "REVIEW_DONE" in result.events
    assert result.fix_request.startswith("The script ran, but these checks")
    assert PRIVATE_ID not in result.fix_request
