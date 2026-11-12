"""Spec validation against SAS's own results (design doc §6.5.4). Runs in the data zone.

For each call site whose output is a saved SAS dataset, the contract's columns are checked and every
property is evaluated on the dataset. Only redacted results leave this module: which check, where, pass
or fail, and how many rows failed (1 to 10 reported as "<11"). No values or statistics are returned.
"""

import ast
import operator

import numpy as np
import pandas as pd
import pyreadstat
from pydantic import BaseModel

from .checks import match_columns, output_for
from .metadata import reference_path
from .schema import Property, Spec


class ValidationItem(BaseModel):
    check: str                       # "columns" or a property id
    call_site: str                   # "program1:L43"
    dataset: str                     # "R.Outcome_mortality"
    passed: bool
    failing_rows: int | str | None = None   # None when the check is not row-based

    def describe(self) -> str:
        rows = "" if self.failing_rows in (None, 0) else f" ({self.failing_rows} rows)"
        return f"{self.check} at {self.call_site} on {self.dataset}: {'passed' if self.passed else 'FAILED'}{rows}"


def _bucket(n: int) -> int | str:
    return "<11" if 0 < n < 11 else n


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
            ast.Gt: operator.gt, ast.GtE: operator.ge}


def evaluate(expr: str, df: pd.DataFrame):
    """Evaluate a property expression (already restricted by the schema) over DataFrame columns."""
    columns = {c.lower(): c for c in df.columns}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Name):
            return df[columns[node.id.lower()]].astype(float) if pd.api.types.is_numeric_dtype(df[columns[node.id.lower()]]) else df[columns[node.id.lower()]]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.BoolOp):
            values = [ev(v) for v in node.values]
            out = values[0]
            for v in values[1:]:
                out = (out & v) if isinstance(node.op, ast.And) else (out | v)
            return out
        if isinstance(node, ast.Compare):
            left, out = ev(node.left), True
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                out = out & _COMPARE[type(op)](left, right)
                left = right
            return out
        raise ValueError(f"unsupported expression node {type(node).__name__}")

    result = ev(ast.parse(expr, mode="eval"))
    return pd.Series(result, index=df.index) if np.isscalar(result) else result


def evaluate_property(prop: Property, df: pd.DataFrame) -> tuple[bool, int | None]:
    """(passed, failing row count or None)."""
    col = {c.lower(): c for c in df.columns}
    if prop.kind == "missing_iff":
        cond = evaluate(prop.when, df).fillna(False).astype(bool)
        bad = int((df[col[prop.column.lower()]].isna() != cond).sum())
        return bad == 0, bad
    if prop.kind == "mean_sd":
        s = df[col[prop.column.lower()]].dropna().astype(float)
        ok = (abs(s.mean() - prop.mean) <= prop.tol * max(1.0, abs(prop.mean))
              and abs(s.std(ddof=prop.ddof) - prop.sd) <= prop.tol * max(1.0, abs(prop.sd)))
        return bool(ok), None
    if prop.kind == "constant":
        bad = 0
        for c in prop.columns:
            s = df[col[c.lower()]]
            bad += int((s != s.iloc[0]).sum()) if len(s) else 0
        return bad == 0, bad
    if prop.kind == "equals":
        left, right = evaluate(prop.left, df).astype(float), evaluate(prop.right, df).astype(float)
        mask = left.notna() & right.notna()
        if prop.where:
            mask &= evaluate(prop.where, df).fillna(False).astype(bool)
        diff = (left - right).abs()
        bad = int((mask & (diff > prop.tol * np.maximum(1.0, right.abs()))).sum())
        return bad == 0, bad
    if prop.kind == "in_range":
        s = df[col[prop.column.lower()]].dropna().astype(float)
        low = s < prop.min if prop.min is not None else False
        high = s > prop.max if prop.max is not None else False
        bad = int((low | high).sum()) if not isinstance(low | high, bool) else 0
        return bad == 0, bad
    raise ValueError(f"unknown property kind {prop.kind}")


def validate(spec: Spec, reference_dir=None) -> list[ValidationItem]:
    items = []
    for site in spec.facts.call_sites:
        for ret in spec.contract.returns:
            dataset = site.arg(ret.name)
            path = reference_path(dataset or "", reference_dir)
            if path is None:
                continue
            df, _ = pyreadstat.read_sas7bdat(str(path))
            items.append(ValidationItem(check="columns", call_site=site.caller, dataset=dataset,
                                        passed=match_columns(ret.columns, list(df.columns))))
            for prop in spec.properties:
                if output_for(spec, prop.output) is not ret:
                    continue
                try:
                    passed, bad = evaluate_property(prop, df)
                except KeyError:          # a column the dataset lacks; the columns check reports why
                    passed, bad = False, None
                items.append(ValidationItem(check=prop.id, call_site=site.caller, dataset=dataset,
                                            passed=passed, failing_rows=None if bad is None else _bucket(bad)))
    return items
