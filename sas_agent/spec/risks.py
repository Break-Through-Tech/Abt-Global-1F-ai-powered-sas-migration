"""The SAS behaviour risk register (design doc §4), cited by id in each spec's logic steps."""

RISKS: dict[str, str] = {
    "S-01": "PROC STANDARD and PROC MEANS use the N-1 standard deviation; missing values are excluded and stay missing.",
    "S-02": "sum(of ...) ignores missing arguments and is missing only if all are missing.",
    "S-03": "A missing number compares as smaller than any number.",
    "S-04": "PROC UNIVARIATE percentiles use definition 5 (PCTLDEF=5).",
    "S-05": "PROC FASTCLUS runs two passes, the second with strict=1; Replace=FULL; negative cluster numbers.",
    "S-06": "Stars are the rank of the cluster's mean, not the cluster number.",
    "S-07": "WORK datasets and macro variables are shared between programs through the SAS session.",
    "S-08": "PROC TABULATE / TRANSPOSE column naming; tranwrd replaces every occurrence.",
    "S-09": "PROC MEANS OUT= writes _TYPE_, _FREQ_, _STAT_ and five statistic rows.",
    "S-10": "A PROC SQL cross join with a one-row table copies that row onto every row.",
    "S-11": "'if _n_=1 then set' keeps the row's values on every later row.",
    "S-12": "DATA step MERGE with BY keeps rows from every input and retains values in many-to-one matches.",
    "S-13": "Group weights are redistributed when a group score is missing.",
    "S-14": "Character comparisons pad the shorter value with blanks.",
    "S-15": "Variable names are case-insensitive.",
    "S-16": "PROVIDER_ID is text with leading zeros.",
    "S-17": "Macro variables are created from data at run time and can generate code conditionally.",
    "S-18": "PROC SORT places missing values first.",
    "S-19": "Output datasets have a defined column order and KEEP/DROP lists.",
    "S-20": "A missing character value is blank.",
}
