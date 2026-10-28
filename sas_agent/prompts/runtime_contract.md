Runtime contract for the Python script you write:
- It is run as a standalone script: `python script.py`. Do not prompt for input.
- Environment variable SAS_INPUT_DIR is the directory for SAS library HC. The input dataset HC.Alldata_2025Jul is available there as `alldata_2025jul.csv`.
- Environment variable SAS_OUTPUT_DIR is the directory for SAS library R. Every dataset the SAS program writes to library R must be written there as `<UPPERCASE_DATASET_NAME>.csv` (e.g. R.less100_measure -> LESS100_MEASURE.csv) with `index=False`. Datasets written to R by earlier programs are available there as CSVs with the same naming convention.
- Read PROVIDER_ID (and any other identifier) as a string -- it has leading zeros.
- SAS session state from earlier programs (WORK datasets, macro variables) is NOT available. Recompute anything you need from the files on disk.
- Ignore LIBNAME / %INCLUDE / path %LET statements; the contract above replaces them.
- Allowed libraries: Python standard library, pandas, numpy, scipy, scikit-learn.
- Required outputs for this program: {outputs}
