"""The spec format (FirstAgent.md items B and D; design doc §6.5).

A Spec has three layers:
- facts: written by the analyzer from the SAS text; the model never changes them;
- contract: the function signature, inputs, and outputs, proposed by the model and checked by code;
- interpretation: purpose, logic per step, properties, assumptions, unrecognised constructs.

AgentLayers is what the model returns (contract and interpretation). The models avoid dict fields and
unions so that they work with providers' structured-output modes.
"""

import ast
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# --- facts (analyzer) ------------------------------------------------------------------------------


class Unit(BaseModel):
    id: str                          # "M.grp_score"
    kind: Literal["macro"] = "macro"
    source: str                      # "Star_Macros.sas"
    lines: tuple[int, int]
    source_hash: str                 # SHA-256 of the unit's SAS text; the spec is invalid if it changes


class Step(BaseModel):
    id: str                          # "s1"
    kind: str                        # "DATA" or "PROC <NAME>"
    lines: tuple[int, int]
    reads: list[str] = []
    writes: list[str] = []


class Arg(BaseModel):
    param: str
    value: str                       # the argument text, with %LET values resolved


class CallSite(BaseModel):
    caller: str                      # "program1:L43"
    program_id: str
    line: int
    args: list[Arg]

    def arg(self, param: str) -> str | None:
        return next((a.value for a in self.args if a.param.lower() == param.lower()), None)


class SideEffects(BaseModel):
    work_datasets_left: list[str] = []   # WORK datasets the unit writes that are not its outputs
    read_downstream: list[str] = []      # of those, the ones a later program step reads


class Facts(BaseModel):
    params: list[str]
    unused_params: list[str] = []
    output_params: list[str] = []        # parameters the unit writes a dataset to
    steps: list[Step]
    constructs: list[str] = []
    macro_vars_used: list[str] = []      # &x references that are not parameters
    macro_vars_created: list[str] = []   # %LET, INTO:, CALL SYMPUTX
    side_effects: SideEffects = SideEffects()
    call_sites: list[CallSite] = []


# --- contract and interpretation (model) ----------------------------------------------------------


class InputSpec(BaseModel):
    name: str                            # a parameter name
    type: str                            # "DataFrame", "list[str]", "str", "int"
    key: str | None = None
    requires: list[str] = []             # columns; "*param" stands for the columns named in a parameter


class OutputSpec(BaseModel):
    name: str                            # the output parameter (or a dataset read downstream)
    columns: list[str]                   # in order; "*param" = columns named in a parameter; "C1..Cn" = numbered series
    rows: str                            # the row rule, in words


class OmittedParam(BaseModel):
    param: str
    reason: str


class Contract(BaseModel):
    function: str
    inputs: list[InputSpec]
    returns: list[OutputSpec]
    omitted_params: list[OmittedParam] = []


class LogicStep(BaseModel):
    step: str                            # a facts step id
    risks: list[str] = []                # risk register ids, e.g. "S-02"
    text: str                            # what the step does, with exact semantics


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def expression_columns(expr: str) -> set[str]:
    """Column names used by a property expression; raises ValueError if the expression is not allowed.

    Allowed: column names, numbers, + - * /, comparisons, and/or.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"expression {expr!r} is not valid: {e.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"expression {expr!r} uses {type(node).__name__}, which is not allowed")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"expression {expr!r} contains a non-numeric constant")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


# The fields each property kind needs.
_PROPERTY_FIELDS = {
    "missing_iff": ("column", "when"),
    "mean_sd": ("column", "mean", "sd", "ddof"),
    "constant": ("columns",),
    "equals": ("left", "right"),
    "in_range": ("column",),
}


class Property(BaseModel):
    """A claim about an output that code can test on SAS's own results.

    kind and the fields it needs:
    - missing_iff: column, when       (column is missing exactly where the expression `when` is true)
    - mean_sd:     column, mean, sd, ddof   (over non-missing values)
    - constant:    columns            (each column has one value across rows)
    - equals:      left, right, where (left equals right within tol, where `where` holds and both are non-missing)
    - in_range:    column, min, max   (non-missing values lie in [min, max])
    """

    id: str
    kind: Literal["missing_iff", "mean_sd", "constant", "equals", "in_range"]
    description: str
    output: str | None = None            # which return it applies to; the first return if omitted
    column: str | None = None
    columns: list[str] = []
    when: str | None = None
    left: str | None = None
    right: str | None = None
    where: str | None = None
    mean: float | None = None
    sd: float | None = None
    ddof: int | None = None
    min: float | None = None
    max: float | None = None
    tol: float = 1e-9

    @model_validator(mode="after")
    def _fields_for_kind(self):
        missing = [f for f in _PROPERTY_FIELDS[self.kind] if getattr(self, f) in (None, [])]
        if missing:
            raise ValueError(f"property {self.id} ({self.kind}) needs {', '.join(missing)}")
        for expr in self.expressions():
            expression_columns(expr)
        return self

    def expressions(self) -> list[str]:
        return [e for e in (self.when, self.left, self.right, self.where) if e]

    def referenced_columns(self) -> set[str]:
        cols = set(self.columns) | ({self.column} if self.column else set())
        for expr in self.expressions():
            cols |= expression_columns(expr)
        return cols


class AgentLayers(BaseModel):
    """What the Spec agent's model returns."""

    purpose: str
    contract: Contract
    logic: list[LogicStep]
    properties: list[Property] = []
    assumptions: list[str] = []
    unknown_constructs: list[str] = []


class Spec(AgentLayers):
    """A complete, versioned spec: code-written facts plus the model's layers."""

    unit: Unit
    facts: Facts
    version: int = Field(1, ge=1)

    def layers(self) -> AgentLayers:
        return AgentLayers(**{k: getattr(self, k) for k in AgentLayers.model_fields})


def export_json_schema(path: Path) -> None:
    """Write the Spec JSON schema, for the translator owners."""
    path.write_text(json.dumps(Spec.model_json_schema(), indent=2))
