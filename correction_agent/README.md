# Correction Agent

The Correction Agent diagnoses failed translations and prepares repair instructions
for the Translation Agent. It does not write replacement Python code.

## What it does

- Accepts a failed script and the Validation Agent's `ValidationResult` or saved JSON dictionary.
- Requires matching program IDs, a `TRIAGED` state and a nonempty `fix_request`.
- Reads the original SAS program, macros and runtime contract.
- Asks the configured model for a prose repair plan: failures, likely causes,
  suggested changes, supported source references and verification steps.
- Identifies missing evidence instead of inventing details.
- Returns instructions and a prompt containing the context the Translation Agent needs.
- Rejects empty responses, code fences, full Python responses and blocked/truncated responses.

Flow: Validation Agent -> Correction Agent -> Translation Agent -> Validation Agent.
The translator writes the revised script; the validator decides whether it passes.

## Files

| File | Purpose |
|---|---|
| `correction_agent/agent.py` | `correct(...)` and `CorrectionResult`. |
| `correction_agent/__init__.py` | Public imports. |
| `sas_agent/prompts/fix.md` | Diagnosis prompt for the Correction Agent. |
| `tests/test_correction_agent.py` | Offline feedback, response and handoff tests. |

## Input and output

Call `correct(program_id, code, validation, ...)` with the failed script and
validator result. Optional `analysis`, `prior_code` and `history` provide context.
History is restricted to program IDs and attempt ranks.

| Output field | Meaning |
|---|---|
| `program_id` | Program to repair. |
| `instructions` | Model-generated prose repair plan. |
| `repair_prompt` | Translator prompt containing the plan, original SAS, failed script and redacted feedback. |

There is no `.code` or `.code_block` output. Earlier examples that passed
`correct(...).code` directly to validation must be updated.

## Test with an existing failed attempt

Run this Python from the repository root after configuring `.env`:

```python
import json
from pathlib import Path
from correction_agent import correct

attempt = Path("runs/20261104-200438/program0/attempt-1")
feedback = json.loads((attempt / "ai/validation.json").read_text(encoding="utf-8"))
failed = (attempt / "ai/script.py").read_text(encoding="utf-8")
result = correct("program0", failed, feedback)
(attempt / "ai/repair_instructions.md").write_text(result.instructions, encoding="utf-8")
(attempt / "ai/translation_repair_prompt.md").write_text(result.repair_prompt, encoding="utf-8")
```

This makes one real Correction Agent model call. Give `repair_prompt` to the
Translation Agent's model invocation in repair mode, extract its full script,
then send that script to the validator in a fresh attempt directory. Running
`python -m sas_agent` currently starts a new translation; its graph does not
yet accept `repair_prompt`. Automatic graph wiring remains integration work.

## Configuration and limits

Uses `CORRECTOR_MODEL` from `.env`, falling back to the translator's configured
model. Provider credentials follow the existing configuration. `llm=` supports
a fake model for tests. Imports make no model calls.

Supply only redacted feedback, never raw logs, CSV contents or exception messages.
Supplied code and analysis must also be free of embedded private data. Output
checks catch common code responses; the translator must still verify suggestions
against SAS.

The agent never executes scripts, compares datasets, promotes results or manages
retry budgets. Orchestration owns retry limits, best-attempt selection and escalation.
This follows the requested separate correction/translation responsibilities for
issue #7 rather than the attached brief's combined Corrector/Translator role.
Existing translator and validator implementations and the main README are unchanged.

Run offline tests with `python -m unittest discover -s tests -v`.
