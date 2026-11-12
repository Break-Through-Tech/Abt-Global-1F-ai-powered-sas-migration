Write final documentation for analysts after this SAS program has been analyzed and translated.

This is the scribe step: explain the business process, the SAS logic, the Python translation, and the output files. Do not invent datasets, columns, or steps that are not in the supplied materials.

Produce Markdown with these sections:
1. **Purpose** -- what the program computes, in business terms.
2. **Inputs** -- datasets read (library.dataset or file), with key columns when they are named in the code.
3. **Step-by-step logic** -- each DATA step / PROC / macro call in order and what it does.
4. **Macros and macro variables** -- what each one resolves to and how it became Python.
5. **Outputs** -- use the mapping below; explain each CSV in plain language.
6. **Python translation** -- how the generated script corresponds to the SAS steps.
7. **Translation notes** -- SAS semantics that are easy to get wrong here. Always mention when they apply:
   - PROC STANDARD / PROC MEANS use the sample standard deviation (ddof=1) and skip missing values
   - PROC STANDARD leaves missing values missing
   - direction flips after standardization (higher score = better)
   - %keep_hos drops hospitals with no remaining measures
   - measure volume <= 100 is excluded

=== Output mapping ===
{mapping}

Expected SAS library R datasets: {outputs}

=== Analysis (from the analyze node) ===
{analysis}

=== Translated Python ===
```python
{code}
```

=== Star_Macros.sas (included via %INCLUDE) ===
{macros}

=== {sas_file} ===
{sas_code}
