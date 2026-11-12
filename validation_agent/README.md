# Validation Agent Brief

| Field | Value |
|---|---|
| Version | 1 (draft) |
| Date | 2026-10-30 |
| Reflects | First Agent Brief v2 (items E, F, G, I) |
| Code | `validation_agent/` |
| Maintainer | Haluk Yuzukirmizi |

The Validation Agent checks the Translator's script for one SAS program and decides where the program
goes next: **PASSED**, or **TRIAGED** with a fix request for the Corrector. It never edits code.
Deterministic tools decide whether the outputs match SAS. An LLM (Gemini by default) reads the code
to find problems the data cannot show: hard-coding that only works for this quarter, quiet logic bugs,
and the cause of each failing check.

## A. Input

| Item | From | Notes |
|---|---|---|
| `program_id` | orchestrator | `program0`, `program1` or `program2` (`sas_agent/config.py`) |
| `code` | Translator / Corrector | the whole script, one Python code block (brief item G) |
| `attempt_dir` | orchestrator (optional) | where this attempt's files go; default `runs/<timestamp>/<program>` |

The agent also reads the SAS program, `Star_Macros.sas`, the runtime contract
(`sas_agent/prompts/runtime_contract.md`) and the SAS reference outputs in `data/Project_1/`.

## B. Output: `ValidationResult`

| Field | Meaning |
|---|---|
| `next_state` | `PASSED` or `TRIAGED` (state machine, item E) |
| `events` | one per stage, e.g. `STATIC_PASS, EXEC_OK, COMPARED_DONE, REVIEW_DONE, FAIL` |
| `stage_reached` | `static_check`, `execute` or `compare` |
| `rank` | `(-stage, failing checks, mismatches, largest error)`; lower is better (D-037) |
| `violations` | static-check rules broken |
| `exception` | `{type, line, source}` if the script crashed; never the message |
| `comparison` | redacted report: `passed`, `failing_checks`, `total_mismatches`, `max_error`, `checks` |
| `review` | LLM findings: `{kind, severity, python_lines, sas_lines, problem, what_to_check}` |
| `fix_request` | text for `{fix_request}` in `sas_agent/prompts/fix.md`; `None` when passed |

A comparison check is `{dataset, column, check, mismatch_count, max_error}`, where `check` is
`file_missing`, `file_name`, `columns`, `row_count`, `row_order`, `missing_values` or `values`.
A review finding's `kind` is `hardcoded`, `quiet_bug` or `cause`.

Example fix request (a script using the population SD instead of SAS's N−1):

```
The script ran, but these checks against the SAS outputs failed:

| dataset | column | check | mismatches | max error |
|---|---|---|---|---|
| OUTCOME_MORTALITY | grp_score | values | 3682 | 0.000643 |
| OUTCOME_MORTALITY | StdDev | values | 4566 | 9.92e-05 |

Code review findings (from reading the code against the SAS program):
- [major] cause, Python L41, SAS L48: the SD uses ddof=0; PROC STANDARD uses the N-1 divisor. Check: ...
```

## C. Workflow

```
code ─► static_check ─► execute ─► compare ─► review ─► verdict ─► ValidationResult
            │                │ crash ───────────▲          ▲
            └─ rule broken ──┴──────────────────┴──────────┘ (review skipped)
```

| Node | State (item E) | Events | What happens |
|---|---|---|---|
| `static_check` | TRANSLATED / PATCHED → CHECKED | `STATIC_PASS` / `STATIC_FAIL` | parses; imports only pandas, numpy and the standard library; reads `SAS_OUTPUT_DIR`. A failure goes straight to `verdict` |
| `execute` | CHECKED → EXECUTED | `EXEC_OK` / `EXEC_ERROR` / `TIMEOUT` | runs the script as the runtime contract says, with earlier programs' SAS outputs in `SAS_OUTPUT_DIR`. A crash skips `compare` |
| `compare` | EXECUTED → COMPARED | `COMPARED_DONE` | every required output CSV vs its SAS dataset |
| `review` | (part of COMPARED) | `REVIEW_DONE` | the LLM reads the SAS and the numbered script, given the tool findings |
| `verdict` | → PASSED / TRIAGED | `ALL_PASS` / `FAIL` | passes only if every check passes and no review finding is `major`; builds the rank and the fix request |

## D. Tools

| Tool | Where | What it checks or does |
|---|---|---|
| Static checks | `graph.check_code` | syntax (line), imports (D-046), runtime contract |
| Executor | `graph.run_script` + `runner.py` | separate isolated process; only `SAS_INPUT_DIR`, `SAS_OUTPUT_DIR` and OS variables (no API keys); timeout; pandas' `ChainedAssignmentError` is an error (D-047); crash → type, line, code line |
| Comparator | `graph.compare` | file exists and is named `<NAME>.csv`; columns and order; row count; row order by `PROVIDER_ID`; missing values; values within 1e-6. CSVs read with `float_precision="round_trip"` (D-049), SAS with pyreadstat; blank = missing for text (S-20) |
| Code review | `graph.review_node` + `prompts/` | LLM, structured output: `hardcoded` (works only for this data), `quiet_bug` (silent logic difference), `cause` (line causing a failing check) |
| Fix request | `graph.fix_request` | tool findings from the first failing stage, then the review findings |

## E. Privacy and files

The LLM sees the SAS source, the script and the redacted findings (names and counts), never data
values or error messages.

```
<attempt_dir>/ai/     script.py, validation.json                          may be shown to a model
<attempt_dir>/data/   outputs/, stdout.txt, stderr.txt, exception.json    data values, never shown
```

## F. Settings (`config.py`)

| Setting | Default |
|---|---|
| `MODEL` | `VALIDATOR_MODEL` from `.env`, else the translator's model (`google_genai:gemini-3.8-flash`) |
| `REVIEW_BLOCKS` | `True`: a major review finding fails the attempt even if every check passes |
| `ALLOWED_PACKAGES` | pandas, numpy (+ standard library) |
| `RTOL`, `ATOL`, `TIMEOUT_S` | 1e-6, 1e-8, 300 s |

`build_graph(review=False)` runs the tools only (no API calls). `GOOGLE_API_KEY` goes in the root `.env`.

## G. Known limits

- Review findings are the model's reading of the code. They can be wrong, and with `REVIEW_BLOCKS` a
  wrong major finding sends a correct script to the Corrector. Turn it off if that happens often.
- `compare` stops at the first structural problem in a dataset (columns, row count, row order), so
  value differences in that dataset show up only in the next round.
- Program 2's k-means: correct clusters with different cluster numbers would fail `values`.
- Not checked yet: `# SAS:` line tags, printing of data values, dynamic imports.

## H. Future steps

1. Tests: a hand-checked Program 1 script and planted bugs (wrong SD divisor, crash, `.CSV` name,
   missing output, hard-coded list), with the review off and on.
2. Keep comparing the shared columns when a dataset's columns differ, so one round reports everything.
3. Add each output's SAS line range to the fix request.
4. Write the full report (with example rows) to `data/` and keep `ai/` redacted, as brief item I asks.
5. Units (November): spec property checks, checkpoint shapes vs the SAS log, `# SAS:` step pairing,
   and handing the code-vs-spec decision to the Triage agent.
6. Agree the interfaces with the orchestration subgroup (`RedactedReport`, events, attempt folders) and
   with the translation team (`fix.md`).
