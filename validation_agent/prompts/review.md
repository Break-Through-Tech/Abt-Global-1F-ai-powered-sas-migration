Review the Python translation of the SAS program below, step by step against the SAS. Report three kinds of findings:

1. `hardcoded`: the Python only works for this quarter's data. Examples: a value, count, list, threshold or
   column set that SAS computes from the data at run time but the Python writes as a literal; logic that skips
   a SAS branch because it happens to be empty now (e.g. no measure under 100 hospitals); assumptions about
   the number of rows, hospitals or measures. Lists that SAS itself writes as literals in its code are fine.
2. `quiet_bug`: logic that differs from SAS but raises no error and may still match the current outputs,
   e.g. wrong missing-value handling, ddof, sort or merge semantics, filters, rounding, edge cases such as a
   group with no measures.
3. `cause`: for each failing check in the tool findings below, the line(s) in the Python that cause it.

Rules:
- The tool findings are exact (the script was run and its outputs compared with the real SAS outputs). Do not contradict them.
- Fill `python_lines` (the Python has line numbers, e.g. "L80-95") and `sas_lines` (e.g. "L120-135") for every finding.
- `severity`: "major" if results are or would be wrong, "minor" otherwise.
- `what_to_check`: what to inspect and the expected behaviour. Never write replacement code.
- No style or naming comments. If there is nothing to report, return an empty list.
- Allowed imports are pandas, numpy and the standard library only.

{contract}

=== Tool findings ===
{tool_findings}

=== Star_Macros.sas (included via %INCLUDE) ===
{macros}

=== {sas_file} ===
{sas_code}

=== Python translation (with line numbers) ===
{code}
