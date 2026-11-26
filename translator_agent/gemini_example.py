"""Opt-in live Gemini example for the translator agent.

Run from the repository root with:
    python -m translator_agent.gemini_example

This makes three Gemini requests and requires GEMINI_API_KEY or GOOGLE_API_KEY.
"""

import ast
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langchain.chat_models import init_chat_model

from sas_agent.spec.schema import Spec
from translator_agent import translate

_SPEC_PATH = _REPO_ROOT / "tests" / "fixtures" / "grp_score_spec.json"
_MODEL = "google_genai:gemini-3.8-flash"


def run_sample() -> str:
    """Translate the checked grp_score spec with Gemini and verify its function interface."""
    specification = Spec.model_validate_json(_SPEC_PATH.read_text(encoding="utf-8"))
    llm = init_chat_model(_MODEL, timeout=900)
    code = translate(specification, llm=llm)

    tree = ast.parse(code)
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    expected_function = specification.contract.function
    if expected_function not in function_names:
        raise AssertionError(f"Translation did not define {expected_function!r}.")

    print(code)
    return code


if __name__ == "__main__":
    run_sample()
