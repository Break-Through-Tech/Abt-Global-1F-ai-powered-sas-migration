"""Prompt templates, one Markdown file per prompt. Placeholders use str.format syntax: {name}."""

from pathlib import Path

_DIR = Path(__file__).parent

SYSTEM = (_DIR / "system.md").read_text(encoding="utf-8")   # system prompt for the review node
REVIEW = (_DIR / "review.md").read_text(encoding="utf-8")   # task prompt for the review node
