"""Static checks on generated code, run before the code is executed (design doc 6.12.2, D-046)."""

import ast
import sys
from dataclasses import dataclass

# Third-party packages generated code may import (D-026). The standard library is always allowed.
ALLOWED_PACKAGES = frozenset({"pandas", "numpy"})


@dataclass(frozen=True)
class Violation:
    """One broken rule. Every field is safe to show the model: it names a rule, a line, or a module."""

    rule: str         # "syntax" or "import"
    line: int | None  # line number in the script
    detail: str


def check(code: str) -> list[Violation]:
    """Return the rules the code breaks; an empty list means it may be executed.

    Notes:
    - A syntax error stops the check, because the import rule needs a parsed script.
    - Only `import` statements are checked. Dynamic imports (`__import__`, `importlib`) are not
      detected here.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Violation("syntax", e.lineno, f"SyntaxError: {e.msg}")]

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # `from . import x`: a standalone script has no package to import from
                violations.append(Violation("import", node.lineno, "relative imports are not allowed"))
                continue
            modules = [node.module or ""]
        else:
            continue
        for module in modules:
            top = module.split(".")[0]
            if top not in ALLOWED_PACKAGES and top not in sys.stdlib_module_names:
                violations.append(Violation(
                    "import", node.lineno,
                    f"import of '{top}' is not allowed; use only pandas, numpy, and the Python standard library",
                ))
    return violations
