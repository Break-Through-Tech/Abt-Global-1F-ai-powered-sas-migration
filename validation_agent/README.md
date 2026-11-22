# Validation Agent

| Field | Value |
|---|---|
| Version | 2.0 |
| Date | 2026-11-22 |
| Fits | The fix loop in `sas_agent/graph.py` (through `sas_agent/agents.py`) and the Correction agent (`correction_agent.correct`) |
| Maintainer | Haluk Yuzukirmizi |

The Validation Agent checks one translated SAS **program** (`program0`, `program1` or `program2`) and decides
where it goes next: **PASSED**, or **TRIAGED** with a fix request for the Correction Agent. It never edits code.
Plain Python checks decide whether the script follows the runtime contract and writes the same outputs as SAS.
An LLM (Gemini, through LangChain) reads the code to find what the data cannot show: hard-coding that only works
for this data, quiet logic bugs, and the line behind each failing check.

Version 1 checked single macro units with in-memory DataFrames. It is in Git history (`validation_agent/validator.py`
before this version) and can come back when a unit-level translator exists.

## A. How the fix loop calls it

The harness owns the loop, the retry limits and escalation. It calls the validator through the adapter in
`sas_agent/agents.py`, which needs nothing else from this package:

```python
from validation_agent.graph import build_graph

app = build_graph(review=True)          # review=False: no model calls, no key needed
result = app.invoke({"program_id": "program0", "code": script, "attempt_dir": "runs/<run-id>/program0/attempt-1"})["result"]
```

```
analyze ─► translate ─► validate ─► promote            (sas_agent/graph.py)
                           │  ▲
                   TRIAGED ▼  │ new script
                        correct ─► repair
```

The harness keeps `next_state`, `rank` and `fix_request`; the Correction Agent turns `fix_request` into a repair
prompt; the translator's model writes a whole new script; the validator checks it again. Run it with:

```bash
uv run python -m sas_agent --fix-loop program0
```

The same call for scripts and tests: `validation_agent.validate(program_id, code, attempt_dir, review=False)`.

## B. Output: `ValidationResult`

| Field | Meaning |
|---|---|
| `program_id` | the program checked; the Correction Agent checks it matches |
| `next_state` | `PASSED` or `TRIAGED` |
| `events` | one per stage, e.g. `STATIC_PASS`, `EXEC_OK`, `COMPARED_DONE`, `REVIEW_DONE`, `FAIL` |
| `stage_reached` | `static_check`, `execute` or `compare` |
| `rank` | `(-stage, failing checks, mismatches, largest error)`; lower is better (D-037) |
| `violations` | static-check rules broken |
| `exception` | `{type, line, source}` if the script failed; never the message |
| `comparison` | redacted report: `passed`, `failing_checks`, `total_mismatches`, `max_error`, `checks` |
| `review` | LLM findings: `{kind, severity, python_lines, sas_lines, problem, what_to_check}` |
| `fix_request` | text for `{fix_request}` in `sas_agent/prompts/fix.md`; `None` when passed |

A comparison check is `{dataset, column, check, mismatch_count, max_error, detail}`. `check` is `file_missing`,
`unreadable`, `columns`, `row_count`, `row_order`, `missing_values` or `values`; `detail` names the missing, extra
or misordered columns and the row counts (never data values). The rank is `(0, violations, 0, 0.0)` after a
static failure, `(-1, 1, 0, 0.0)` after a crash or timeout, and `(-2, failing checks, mismatches, max error)` after
`compare`.

Example fix request:

```
The script ran, but these checks against the SAS outputs failed:

| dataset | column | check | mismatches | max error | detail |
|---|---|---|---|---|---|
| OUTCOME_MORTALITY | grp_score | values | 312 | 0.101 | - |

Code review findings (from reading the code against the SAS program):
- [major] cause, Python L48, SAS macros L286: the WHERE total_cnt>=3 filter is missing. Check: ...
```

## C. Workflow

```
START ─► static_check ─► execute ─► compare ─► review ─► verdict ─► END
              │             │ crash / timeout ───▲          ▲
              └─ rule broken ┴──────────────────────────────┘  (not run, no review)
```

| Node | Events | LLM? | What happens | Module |
|---|---|---|---|---|
| `static_check` | `STATIC_PASS` / `STATIC_FAIL` | no | reads the script without running it (section D) | `checks.py` |
| `execute` | `EXEC_OK` / `EXEC_ERROR` / `EXEC_TIMEOUT` | no | runs the script in a separate process (section E) | `runner.py` |
| `compare` | `COMPARED_DONE` | no | each output CSV vs the SAS `.sas7bdat` result | `compare.py` |
| `review` | `REVIEW_DONE` | Gemini | reads the SAS, the macros, the numbered script and the tool findings | `review.py` |
| `verdict` | `ALL_PASS` / `FAIL` | no | next state, rank, fix request; writes `ai/validation.json` | `verdict.py` |

`graph.py` wires the nodes, `agent.py` has `validate()`, `schema.py` the output models, `config.py` the settings.

## D. Checks

| Check | Node | Passes when |
|---|---|---|
| syntax | static_check | the script parses |
| imports | static_check | only pandas, numpy and the standard library, without process, network or dynamic-import modules (`subprocess`, `socket`, `urllib`, `importlib`, ...) |
| no dynamic code | static_check | no `eval`, `exec`, `compile`, `__import__`, `input`, `breakpoint`; no `os.system`, `os.remove`, `shutil.rmtree`, ... |
| inputs | static_check | no `read_sas` / pickle, and no paths to `SAS Output`, `.sas7bdat` or `Project_1`: data comes only from `SAS_INPUT_DIR` and `SAS_OUTPUT_DIR` |
| contract | static_check | the script uses both `SAS_INPUT_DIR` and `SAS_OUTPUT_DIR` |
| execute | execute | exit code 0 within the time limit. pandas' `ChainedAssignmentError` counts as an error (D-047) |
| compare | compare | for each output of the program: `<NAME>.csv` exists and reads; columns and their order match; row count matches; rows are in the SAS `PROVIDER_ID` order; every column has the same missing values (a blank text equals missing, S-20) and values (numbers within `RTOL` 1e-6 / `ATOL` 1e-8). Columns that differ are reported, and the shared columns are still compared |
| review | verdict | no `major` finding (only with `REVIEW_BLOCKS`) |

## E. Running the script, privacy and files

- The script runs as `python script.py` in `<attempt_dir>/data/`, with a timeout (`VALIDATION_TIMEOUT_SECONDS`,
  default 900). Its environment has only `PATH`, `SAS_INPUT_DIR`, `SAS_OUTPUT_DIR` and `PYTHONIOENCODING`
  (plus `SYSTEMROOT` on Windows, which Python needs to start), so API keys never reach generated code.
- `SAS_INPUT_DIR` is `runs/<run-id>/inputs`, written once per run by `sas_agent.runs.prepare_inputs()`.
- `SAS_OUTPUT_DIR` is `<attempt_dir>/data/outputs`. Before the run it holds the **SAS results of the earlier
  programs**, so each program is checked on exactly what SAS gave it; an error in an earlier translation
  can't fail a later one.
- The model sees the SAS program, Star_Macros.sas, the runtime contract, the script and the tool findings
  (names, counts, code lines), never data values or error messages. The graph state holds no data.

```
runs/<run-id>/<program>/attempt-<n>/
├── ai/     script.py, validation.json                          no data values; may be shown to a model
└── data/   outputs/*.csv, stdout.txt, stderr.txt, exception.json   never sent to a model
```

## F. Settings (`config.py`)

| Setting | Default |
|---|---|
| API key | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in the repository's `.env`, loaded by `sas_agent.config`; only the review needs it |
| `MODEL` | `VALIDATION_AGENT_MODEL` from the environment, else the translator's `SAS_AGENT_MODEL` |
| `REVIEW_BLOCKS` | `True`: a `major` review finding fails the attempt even if every check passes |
| `RTOL`, `ATOL` | 1e-6, 1e-8 |
| `ID_COLUMN` | `PROVIDER_ID` (row order check) |
| `TIMEOUT_SECONDS` | `VALIDATION_TIMEOUT_SECONDS` from the environment, else 900 |
| `REFERENCE_DIR` | `data/Project_1/SAS Output` (the `.sas7bdat` results) |

## G. Tests

| File | What | Model calls |
|---|---|---|
| `tests/test_validation_agent.py` | static rules, comparator, crash redaction, timeout, chained assignment, the child environment, the review with a fake model, and the full fix loop with the real validator and Correction Agent (fake models) | none |
| `tests/test_validation_agent_llm.py` | the Gemini review on a hard-coded program0 script, and the graph with the review on | yes, only with `RUN_LLM_TESTS=1` |

```bash
uv run pytest tests/test_validation_agent.py
RUN_LLM_TESTS=1 uv run pytest tests/test_validation_agent_llm.py -s
```

## H. Known limits

- The static checks are a basic safety gate, not a sandbox: the script runs as the current user.
- Review findings are the model's reading of the code. They can be wrong, and with `REVIEW_BLOCKS` a wrong
  `major` finding sends a correct translation to the Correction Agent. A review-blocked attempt has the same
  rank as a passing one, so the harness's stuck rule ends such a loop after two corrections.
- `compare` stops at a row-count or row-order problem, so value differences show up in the next round.
- Each attempt writes the earlier programs' SAS results to its outputs folder (a few MB per attempt).
