# Validation Agent Brief

| Field | Value |
|---|---|
| Version | 1.2 (draft) |
| Date | 2026-11-21 |
| Reflects | Validation Agent Brief v1 output (item B), Translator output contract (`notebooks/translator-*.ipynb`), Correction Agent input (`feature/correction_agent`, c4d9189) |
| Code | `validation_agent/validator.py` (agent), `validation_agent/validator.ipynb` (tests and walkthrough) |
| Maintainer | Haluk Yuzukirmizi |

The Validation Agent checks one SAS unit translated by the Translator and decides where it goes next:
**PASSED**, or **TRIAGED** with a fix request for the Correction Agent. It never edits code. Plain Python
checks decide whether the code follows the project rules and gives the same outputs as SAS. An LLM (Gemini,
through LangChain) reads the code to find problems the data cannot show: hard-coding that only works for
this data, quiet logic bugs, and the cause of each failing check.

It is a LangGraph agent, like `sas_agent`. All the logic, including the edge to the Correction Agent, is in
`validator.py`; the notebook only calls it. It is not wired to the other agents yet (section G).

## A. Input

The keys of the graph state. The first three are required:

| Key | From | Content |
|---|---|---|
| `translator_input` | the agent before the Translator | The five inputs the Translator received: `specification`, `sas_source`, `upstream_signatures`, `behavior_notes`, `project_conventions`. Only the LLM review reads them. |
| `translation` | Translator | The Translator's output: exactly five fields (below). |
| `test_data` | orchestrator / test set | `inputs`: the function's arguments by parameter name. `expected`: the SAS output for each name the function returns. |
| `program_id` | orchestrator (optional) | The id the Correction Agent checks. Default: `specification.unit.id`, else `unknown`. |
| `attempt_dir` | orchestrator (optional) | If given, the result is also written to `<attempt_dir>/ai/` (section E). |

`translation`, the Translator's output contract:

| Field | Type | Meaning |
|---|---|---|
| `code` | string | The complete translated function: one function per unit, DataFrames in, `dict[str, DataFrame]` out, no file I/O |
| `checkpoints` | list of strings | Checkpoint **names**, in the order the code calls `checkpoint(name, df)` |
| `conflicts` | list | Places where the specification and the SAS source disagree |
| `assumptions` | list | Assumptions made by the Translator |
| `helpers_requested` | list | Helper functions the translation needs |

Example (`%nation_avg`, Star_Macros.sas lines 282-295):

```python
translation = {
    "code": "import pandas as pd\n\ndef nation_avg(indata, var): ...  return {'out': out}",
    "checkpoints": ["temp", "out1", "out"],
    "conflicts": [], "assumptions": ["..."], "helpers_requested": [],
}
test_data = {
    "inputs":   {"indata": OUTCOME_MORTALITY_df, "var": "Out_Mrt_Grp_Score_Nat"},
    "expected": {"out": NATIONAL_AVERAGE_2025JUL_df[["Out_Mrt_Grp_Score_Nat"]]},
}
```

## B. Output: `ValidationResult`

The graph state key `result`; `validate(...)` returns it as a dict. The fields are those of Validation Agent
Brief v1, item B, which the Correction Agent was built against.

| Field | Meaning |
|---|---|
| `program_id` | the unit or program checked; the Correction Agent checks it matches |
| `next_state` | `PASSED` or `TRIAGED` (state machine, First Agent Brief item E) |
| `events` | one per stage, e.g. `STATIC_PASS`, `EXEC_OK`, `COMPARED_DONE`, `REVIEW_DONE`, `FAIL` |
| `stage_reached` | `static_check`, `execute` or `compare` |
| `rank` | `(-stage, failing checks, mismatches, largest error)`; lower is better (D-037) |
| `violations` | static-check rules broken |
| `exception` | `{type, line, source}` if the code crashed; never the message |
| `comparison` | redacted report: `program_id`, `passed`, `failing_checks`, `total_mismatches`, `max_error`, `checks` |
| `review` | LLM findings: `{kind, severity, python_lines, sas_lines, problem, what_to_check}` |
| `fix_request` | text for `{fix_request}` in `sas_agent/prompts/fix.md`; `None` when passed |

A comparison check is `{dataset, column, check, mismatch_count, max_error}`, where `check` is `file_missing`
(the output was not returned), `columns`, `row_count`, `row_order`, `missing_values` or `values`. A finding's
`kind` is `hardcoded` (only works for this data), `quiet_bug` (differs from SAS without an error) or `cause`
(the line behind a failing check). `severity` is `major` if results are or could be wrong, otherwise `minor`.
Without `REVIEW_DONE` in `events`, the review was off or skipped.

Example fix request (the buggy `%nation_avg` without the `where total_cnt>=3` filter; the table is real
output, the review finding shows the format):

```
The function ran, but these checks against the SAS outputs failed:

| dataset | column | check | mismatches | max error |
|---|---|---|---|---|
| out | Out_Mrt_Grp_Score_Nat | values | 1 | 0.101 |

Code review findings (from reading the code against the SAS program):
- [major] cause, Python L6, SAS L4: the WHERE total_cnt>=3 filter is missing. Check: ...
```

Its `rank` is `(-2, 1, 1, 0.101)` and its `events` are `STATIC_PASS, EXEC_OK, COMPARED_DONE, REVIEW_DONE, FAIL`.

## C. Workflow

```
START ─► static_check ─► execute ─► compare ─► review ─► verdict ─► END
              │             │ crash ─────────▲          ▲
              └─ rule broken ┴───────────────┴──────────┘  (code not run, review skipped)
```

| Node | State (item E) | Events | LLM? | What happens | Function in `validator.py` |
|---|---|---|---|---|---|
| `static_check` | TRANSLATED / PATCHED → CHECKED | `STATIC_PASS` / `STATIC_FAIL` | no | reads the code without running it (section D). A rule broken goes straight to `verdict` | `check_code` |
| `execute` | CHECKED → EXECUTED | `EXEC_OK` / `EXEC_ERROR` | no | calls the function with `test_data["inputs"]`. A crash skips `compare` | `run_function` |
| `compare` | EXECUTED → COMPARED | `COMPARED_DONE` | no | every expected output vs what the function returned | `compare_outputs`, `compare_dataset` |
| `review` | (part of COMPARED) | `REVIEW_DONE` | Gemini | reads the SAS, the numbered code and the tool findings. Skipped with `review=False` | `tool_findings`, `llm_review` |
| `verdict` | → PASSED / TRIAGED | `ALL_PASS` / `FAIL` | no | the next state, the rank and the fix request; writes `ai/` if `attempt_dir` is given | `make_result`, `build_fix_request`, `open_issues`, `write_attempt` |

## D. Checks

| Check | Node | Passes when |
|---|---|---|
| output format | static_check | the translation has exactly the five fields; `code` is non-empty text, the others are lists |
| syntax | static_check | the code parses |
| one function | static_check | exactly one top-level function |
| imports | static_check | only pandas, numpy and safe standard-library modules (math, statistics, datetime, collections, re, ...) |
| no file I/O | static_check | no `open`, `print`, `eval`/`exec`, `getattr`, `read_*` or `to_*` file calls |
| checkpoints | static_check | the `checkpoint("name", ...)` calls in the code equal `checkpoints`, in order |
| execute | execute | the function returns a dict of DataFrames without an error. pandas' `ChainedAssignmentError` counts as an error (D-047: in pandas 3 a chained assignment silently does nothing). A crash gives the error type and code line, not the message |
| compare | compare | for each expected output: it was returned; columns and their order match; row count matches; rows are in the same `PROVIDER_ID` order (when the output has it); every column has the same missing values (a blank text equals missing, S-20) and the same values (numbers within `RTOL` 1e-6 / `ATOL` 1e-8) |
| review | verdict | no `major` finding (only with `REVIEW_BLOCKS`) |
| open issues | verdict | `conflicts` and `helpers_requested` are empty. It does not stop the code from running |

The `rank` orders attempts by the stage reached, then the failing checks, mismatches and largest error:
`(0, violations, 0, 0.0)` after a static failure, `(-1, 1, 0, 0.0)` after a crash, and
`(-2, failing checks, mismatches, max error)` after `compare`.

## E. Privacy and files

The LLM sees the SAS source, the code, the specification and the tool findings (names and counts), never
data values or error messages. The function's outputs stay in the graph state (`outputs`) and are not sent.

With `attempt_dir`, the verdict writes what the Correction Agent reads:

```
<attempt_dir>/ai/   script.py, validation.json     no data values; may be shown to a model
```

## F. Settings (`validator.py`)

| Setting | Default |
|---|---|
| `MODEL` | `VALIDATION_AGENT_MODEL` from the environment, else `google_genai:gemini-3.8-flash` (same as the Translator) |
| API key | `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `validation_agent/.env` (git-ignored), never in code |
| `REVIEW_BLOCKS` | `True`: a `major` review finding fails the attempt even if every check passes |
| `ALLOWED_IMPORTS`, `BLOCKED_CALLS`, `BLOCKED_METHODS` | see section D |
| `RTOL`, `ATOL` | 1e-6, 1e-8 |
| `ID_COLUMN` | `PROVIDER_ID` (row order check) |

```python
from validation_agent.validator import build_graph, validate

result = validate(translator_input, translation, test_data)                 # Gemini review, key from .env
result = validate(translator_input, translation, test_data, review=False)   # manual checks only, no API calls
app = build_graph()                                                         # the graph, to invoke or add as a node
```

## G. Alignment with the other agents

**Translators**

| | `translator-jared` | `translator-varun` | `sas_agent` |
|---|---|---|---|
| Five-field output | yes, returned directly | yes, as `result["improved_translation"]` | no: one whole script |
| `checkpoints` are names in call order | to agree: described as "checkpoint calls used" | yes | no |
| One function, DataFrames in, dict of DataFrames out, `checkpoint(name, df)` | to agree: not in its prompts | yes | no: reads and writes CSVs |
| Input key `specification` | yes | named `spec` | n/a |

To agree with the translation team: checkpoint names (not call strings), the `specification` key, and
whether scipy / scikit-learn are allowed (`sas_agent` allows them, the validator does not).

**Correction Agent** (`correction_agent.correct(program_id, code, validation)`, branch `feature/correction_agent`)

| It requires | The validator gives |
|---|---|
| `program_id` equal to the call's | `program_id` |
| `next_state == "TRIAGED"` | `next_state` |
| a non-empty `fix_request` | `fix_request` (always set when TRIAGED) |
| the result object, its `model_dump()`, or `ai/validation.json` with the code in `ai/script.py` | all three |

Tested with its own code and a fake model: it accepts a TRIAGED result, rejects a PASSED one, and its
repair prompt contains the fix request.

**Pipeline wiring** (in `validator.py`, used when the agents are connected):

| Name | What it is |
|---|---|
| `build_graph()` | the validator, added to the pipeline graph as one node (a subgraph) |
| `route_after_validation` | the conditional edge after it: TRIAGED → `CORRECTION_NODE` (`"correct"`), PASSED → END |
| `correction_input(state)` | the arguments of `correct(...)`, taken from the state |

```python
pipeline.add_node("validate", build_validator())
pipeline.add_node(CORRECTION_NODE, correct_node)          # correct(**correction_input(state))
pipeline.add_conditional_edges("validate", route_after_validation)
pipeline.add_edge(CORRECTION_NODE, "translate")
```

Only the shared keys (`program_id`, `translator_input`, `translation`, `test_data`, `attempt_dir`, `result`)
pass between the pipeline and the validator; its `events`, `violations`, ... stay inside it, so every attempt
starts fresh. Tested with a fake Translator: TRIAGED → correct → translate → PASSED.

## H. Known limits

- The static checks are a basic safety gate, not a sandbox: the code runs in the same Python process, with
  no timeout (no `TIMEOUT` event).
- Review findings are the model's reading of the code. They can be wrong, and with `REVIEW_BLOCKS` a wrong
  `major` finding sends a correct translation to the Correction Agent.
- `compare` stops at the first structural problem in an output (missing, columns, row count, row order), so
  value differences show up only in the next round.
- Any `conflicts` fail the unit, although under the First Agent Brief a conflict may be a spec fault, not a code fault.
- The graph state holds data values (`test_data`, `outputs`); the First Agent Brief says it should not.
- The Correction Agent only accepts `program0` to `program2`, so a unit id fails its program lookup.
- `sas_agent/prompts/fix.md`, which the Correction Agent fills, is only on `feature/correction_agent`.
- The `test_data` for each unit (inputs and expected SAS outputs) has to be built by someone; no agent
  produces it yet.

## I. Future steps

1. Connect with the Translator and the Correction Agent using the wiring in section G; the orchestrator adds
   the retry limits (D-036) and escalation.
2. Agree with the Correction Agent's owner on the fields it reads and on unit ids.
3. Build `test_data` per unit from the SAS outputs and the SAS log (checkpoint shapes).
4. Run the function in a separate process with a timeout, and keep data values out of the graph state.
5. Keep comparing the shared columns when an output's columns differ, so one round reports everything.
6. Agree the interfaces above with the translation and orchestration subgroups.
