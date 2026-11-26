"""Translate a Spec agent result into a Python function fragment.

The current Validation agent expects a standalone program script, so this function
source needs a program-level adapter and context before that validator can run it.
"""

import ast
import json
from collections.abc import Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from sas_agent import config
from sas_agent.spec import analyzer
from sas_agent.spec.schema import Spec

TRANSLATION_RESULT_FIELDS = (
    "code",
    "checkpoints",
    "conflicts",
    "assumptions",
    "helpers_requested",
)

_BLOCKED_FINISH_REASONS = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "MAX_TOKENS",
}

_TRANSLATION_SYSTEM = (
    "You are an expert SAS-to-Python translator. Translate exactly the specified SAS unit into "
    "one Python function. The SAS source is authoritative for behavior; report disagreements with "
    "the specification as conflicts. Return JSON only with exactly these fields: code, checkpoints, "
    "conflicts, assumptions, helpers_requested. code must contain the complete translated function; "
    "the other fields must be arrays of strings."
)

_REFLECTION_SYSTEM = (
    "You are reviewing a SAS-to-Python translation. The SAS source is authoritative for behavior. "
    "Return JSON only with exactly these fields: overall_assessment, issues, missing_checkpoints, "
    "semantic_risks, required_changes, unresolved_questions."
)


def _build_llm() -> BaseChatModel:
    return init_chat_model(config.MODEL, timeout=900)


def _invoke_text(llm: BaseChatModel, messages: Sequence[BaseMessage]) -> str:
    response = llm.invoke(messages)
    finish_reason = response.response_metadata.get("finish_reason")
    if finish_reason in _BLOCKED_FINISH_REASONS:
        raise RuntimeError(f"Unusable translation response: {finish_reason}")
    if not isinstance(response.text, str) or not response.text.strip():
        raise RuntimeError("Translation model returned no text.")
    return response.text


def _parse_translation(response: str) -> str:
    try:
        result = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("Final translation was not valid JSON.") from error

    if not isinstance(result, dict):
        raise ValueError("Final translation must be a JSON object.")

    fields = set(TRANSLATION_RESULT_FIELDS)
    missing_fields = fields - result.keys()
    extra_fields = result.keys() - fields
    if missing_fields or extra_fields:
        raise ValueError(
            f"Final translation fields are invalid. Missing: {missing_fields}; Extra: {extra_fields}."
        )

    code = result["code"]
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Final translation code must be a non-empty string.")
    try:
        ast.parse(code)
    except SyntaxError as error:
        raise ValueError("Final translation code is not valid Python.") from error

    for field in TRANSLATION_RESULT_FIELDS[1:]:
        value = result[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"Final translation field {field!r} must be an array of strings.")

    return code


def translate(specification: Spec, *, llm: BaseChatModel | None = None) -> str:
    """Translate a Spec into function source, not a standalone script."""
    if not isinstance(specification, Spec):
        raise TypeError("specification must be a Spec returned by the Spec agent")

    sas_source = analyzer.unit_source(specification.unit)
    spec_json = specification.model_dump_json(indent=2)
    initial_prompt = f"""Translate this checked SAS unit into one Python function.

Use the specification's function signature, inputs, outputs, and logic. Preserve the behavior
of the SAS source when it conflicts with the specification, and record each disagreement in
`conflicts`. Include checkpoint calls required by the project. Do not add a main() entry point,
file input/output, or code unrelated to this unit.

Return one JSON object with exactly these five fields:
- code: the complete translated function.
- checkpoints: checkpoint calls used during translation.
- conflicts: disagreements between the specification and SAS source.
- assumptions: decisions made where the source or requirements are ambiguous.
- helpers_requested: helper functions needed by the translation.

Use an empty array when a metadata field has no entries. Return JSON only, with no Markdown
fences or surrounding commentary.

<SPECIFICATION>
{spec_json}
</SPECIFICATION>

<SAS_SOURCE>
{sas_source}
</SAS_SOURCE>
"""

    reflection_prompt = f"""Carefully compare the initial translation with the SAS source and specification.
Identify concrete issues, missing checkpoints, semantic risks, required changes, and unresolved
questions. The initial translation is in the preceding assistant message.

<SPECIFICATION>
{spec_json}
</SPECIFICATION>

<SAS_SOURCE>
{sas_source}
</SAS_SOURCE>

Return one JSON object with exactly these fields:
- overall_assessment: a concise summary.
- issues: concrete translation issues.
- missing_checkpoints: required checkpoint calls not present.
- semantic_risks: behavior that may differ from the source.
- required_changes: changes needed in the final result.
- unresolved_questions: remaining ambiguities.

Use empty arrays when a field has no entries. Return JSON only, with no Markdown fences or commentary.
"""

    final_prompt = f"""Improve the initial translation using the reflection feedback in the preceding
assistant message. Preserve the source behavior and the specification's function interface.

<SPECIFICATION>
{spec_json}
</SPECIFICATION>

<SAS_SOURCE>
{sas_source}
</SAS_SOURCE>

Return one JSON object with exactly these five fields:
- code: the complete translated function.
- checkpoints: checkpoint calls used during translation.
- conflicts: disagreements between the specification and SAS source.
- assumptions: decisions made where the source or requirements are ambiguous.
- helpers_requested: helper functions needed by the translation.

Use an empty array when a metadata field has no entries. Return JSON only, with no Markdown
fences or surrounding commentary.
"""

    model = llm or _build_llm()
    initial_response = _invoke_text(
        model,
        [SystemMessage(_TRANSLATION_SYSTEM), HumanMessage(initial_prompt)],
    )
    reflection_response = _invoke_text(
        model,
        [
            SystemMessage(_REFLECTION_SYSTEM),
            HumanMessage(initial_prompt),
            AIMessage(initial_response),
            HumanMessage(reflection_prompt),
        ],
    )
    final_response = _invoke_text(
        model,
        [
            SystemMessage(_TRANSLATION_SYSTEM),
            HumanMessage(initial_prompt),
            AIMessage(initial_response),
            HumanMessage(reflection_prompt),
            AIMessage(reflection_response),
            HumanMessage(final_prompt),
        ],
    )

    return _parse_translation(final_response)
