"""Prompt templates, one Markdown file per prompt. Placeholders use str.format syntax: {name}."""

from pathlib import Path

_DIR = Path(__file__).parent


def load(name: str) -> str:
    return (_DIR / f"{name}.md").read_text()


SYSTEM = load("system")                      # system prompt shared by LLM nodes
RUNTIME_CONTRACT = load("runtime_contract")  # how the generated script reads/writes data
ANALYZE = load("analyze")                    # node 1: analyze the SAS program
TRANSLATE = load("translate")                # node 2: write the Python script
DOCUMENT = load("document")                  # node 4: scribe documentation for analysts


def fill(template: str, **values: str) -> str:
    """Replace {name} placeholders without interpreting braces inside SAS/Python code."""
    filled = template
    for name, value in values.items():
        filled = filled.replace("{" + name + "}", value)
    return filled


def format_prior(prior_code: dict[str, str]) -> str:
    if not prior_code:
        return ""
    parts = ["\n=== Python translations of earlier programs in this pipeline (for reference) ==="]
    for program_id, code in prior_code.items():
        parts.append(f"--- {program_id} ---\n```python\n{code}\n```")
    return "\n".join(parts)
