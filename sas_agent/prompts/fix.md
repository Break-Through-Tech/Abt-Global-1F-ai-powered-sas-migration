Diagnose the failed Python translation of {sas_file} using the redacted validation feedback.
Write a repair plan addressed to the Translation Agent, in plain prose.
Do not write replacement Python, code snippets, code blocks or diffs.
For each issue, describe the observed failure, likely cause, suggested change,
relevant SAS/Python file and line references when supported, and how to verify it.
Distinguish evidence from hypotheses. Do not invent line references or column lists.
If feedback lacks required metadata, say what the translator/orchestrator must check.
For column failures, request missing/extra columns and expected versus actual order;
the mismatch count alone does not identify the columns or prove value correctness.

Use SAS source as the authority. Make targeted corrections and preserve unrelated logic.
Use only pandas, numpy and the Python standard library. Do not import scipy or sklearn.
Read CSV numbers with float_precision="round_trip" and identifiers as strings.
Use sample standard deviation (ddof=1) where SAS does. Preserve SAS missing-value,
row-order and column-order semantics. Avoid chained assignment under pandas 3.
Retain or add comments citing SAS file names and line ranges for translated blocks.
Do not print/log dataset values or read reference outputs to manufacture results.
Preserve source-defined constants and lists; fix hard-coding only when unsupported by SAS.
Ask the Translation Agent to return a complete revised script for validation.
Do not claim the repair has passed.

Runtime contract:
{contract}

SAS program ({sas_file}):
```sas
{sas_code}
```

Star_Macros.sas:
```sas
{macros}
```

Earlier accepted translations:
{prior}

Original analysis:
{analysis}

Failed script:
```python
{code}
```

Redacted validation feedback:
{fix_request}

Previous attempt ranks (lower is better):
{history}
