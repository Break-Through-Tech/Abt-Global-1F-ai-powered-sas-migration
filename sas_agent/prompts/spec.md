Write the contract and interpretation layers of the specification for the SAS unit {unit_id}.

The facts layer below was written by code from the SAS text. Do not change or restate it; refer to it.
The specification is the contract a Translator follows to write one Python function for this unit, and
code will test your properties on SAS's own results.

Rules:
- `logic`: exactly one entry per facts step, in order, using the step ids from the facts. State exact SAS
  semantics (missing values, divisors, joins, row order), not a paraphrase. Cite the risk ids that apply.
- `contract.inputs` and `contract.returns` use parameter names. A return is an output parameter.
- `contract.returns[].columns`: the output columns in order. Use `*param` for the columns named in a
  parameter (for example `*varlist`) and `C1..Cn` for a numbered series. They must match the column
  lists of the reference datasets below.
- `contract.omitted_params`: every parameter the Python function does not need, with the reason. Unused
  parameters must be listed.
- `properties`: claims about the outputs that hold on any data, each with a unique id (p1, p2, ...).
  Kinds and their fields:
  - `missing_iff`: `column`, `when` (the column is missing exactly where `when` is true)
  - `mean_sd`: `column`, `mean`, `sd`, `ddof` (over non-missing values)
  - `constant`: `columns` (each has a single value across rows)
  - `equals`: `left`, `right`, optional `where` (equal within `tol` where both are non-missing)
  - `in_range`: `column`, `min` and/or `max`
  Expressions may use only column names, numbers, + - * /, comparisons, and `and`/`or`. Every column a
  property uses must appear literally in the return's columns.
- `unknown_constructs`: SAS constructs in this unit whose behaviour you are not sure of.
- `assumptions`: anything you had to assume.
- Never write Python code. Never include data values.

=== Facts (read-only) ===
{facts}

=== Reference datasets at the call sites (metadata only) ===
{metadata}

=== Risk register ===
{risks}

=== SAS source of {unit_id} ===
{sas_source}
{revision}
