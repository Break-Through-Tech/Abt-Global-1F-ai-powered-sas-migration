Analyze the SAS program below before it is translated to Python.

Produce a concise Markdown analysis with these sections:
1. **Purpose** -- what the program computes, in business terms.
2. **Inputs and outputs** -- datasets read and written (library.dataset), with key columns.
3. **Step-by-step logic** -- each DATA step / PROC / macro call in order, what it does, and the equivalent pandas approach.
4. **Macros and macro variables** -- what each one resolves to and how it should become Python.
5. **Translation risks** -- SAS semantics that are easy to get wrong here (missing values, statistics definitions, sorting, clustering, rounding, etc.).

{contract}

=== Star_Macros.sas (included via %INCLUDE) ===
{macros}

=== {sas_file} ===
{sas_code}
{prior}
