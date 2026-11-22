"""Step 1: static checks of a translated program (no LLM, the script is not run).

The rules follow sas_agent/prompts/runtime_contract.md: a standalone script that reads its input from
SAS_INPUT_DIR, writes CSVs to SAS_OUTPUT_DIR, and imports only pandas, numpy and the standard library.
This is a basic safety gate, not a sandbox.
"""

import ast
import sys

THIRD_PARTY = {"pandas", "numpy"}
BLOCKED_MODULES = {   # standard-library modules a translation never needs: processes, network, dynamic imports
    "subprocess", "socket", "ssl", "urllib", "http", "ftplib", "smtplib", "ctypes", "multiprocessing",
    "webbrowser", "importlib", "runpy",
}
BLOCKED_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
BLOCKED_METHODS = {"read_sas", "read_pickle", "to_pickle"}   # data must come from the CSVs in SAS_INPUT_DIR
BLOCKED_OS_CALLS = {"system", "popen", "startfile", "remove", "unlink", "rmdir", "removedirs", "rmtree", "kill", "fork"}
REFERENCE_MARKERS = ("sas output", ".sas7bdat", "project_1")  # paths to the SAS results or the raw data
REQUIRED_ENV = ("SAS_INPUT_DIR", "SAS_OUTPUT_DIR")


def _module_allowed(module: str) -> bool:
    top = module.split(".")[0]
    return top in THIRD_PARTY or (top in sys.stdlib_module_names and top not in BLOCKED_MODULES)


def check_code(code: str) -> list[str]:
    """Return the rules the script breaks (empty list: the script may be run)."""
    if not isinstance(code, str) or not code.strip():
        return ["the script is empty"]
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return [f"line {error.lineno}: SyntaxError: {error.msg}"]

    violations, strings = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = ["." * node.level + (node.module or "")]
        else:
            modules = []
        violations += [f"line {node.lineno}: import of '{m}' is not allowed; use only pandas, numpy and the "
                       "standard library (no process, network or dynamic-import modules)"
                       for m in modules if not _module_allowed(m)]

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            violations.append(f"line {node.lineno}: {node.func.id}() is not allowed (no dynamic code or prompts)")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr, owner = node.func.attr, node.func.value
            if attr in BLOCKED_METHODS:
                violations.append(f"line {node.lineno}: .{attr}() is not allowed; read the CSV files in "
                                  "SAS_INPUT_DIR and SAS_OUTPUT_DIR")
            elif isinstance(owner, ast.Name) and owner.id in ("os", "shutil") and (
                    attr in BLOCKED_OS_CALLS or attr.startswith(("exec", "spawn"))):
                violations.append(f"line {node.lineno}: {owner.id}.{attr}() is not allowed (no processes or "
                                  "deleting files)")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
            if any(marker in node.value.lower() for marker in REFERENCE_MARKERS):
                violations.append(f"line {node.lineno}: the script refers to the SAS results or the raw data "
                                  "folder; read only from SAS_INPUT_DIR and SAS_OUTPUT_DIR")

    violations += [f"the script never uses the {name} environment variable (runtime contract)"
                   for name in REQUIRED_ENV if not any(name in s for s in strings)]
    return violations
