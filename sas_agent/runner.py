"""Runs one generated script and reports how it ended (design doc 6.12.3, D-047).

Usage: python -I runner.py SCRIPT EXCEPTION_JSON

The executor starts this file in a separate process; the package never imports it.

Notes:
- pandas' ChainedAssignmentError is raised as an error. Under pandas 3 it is only a warning and the
  assignment silently does nothing, so a script would finish with wrong values.
- If the script fails, a sanitized description is written to EXCEPTION_JSON: the exception type, the
  line number in the script, and that line of code. The message is left out because it can contain
  data values (e.g. KeyError: '010001'). The full traceback goes to stderr, which the executor keeps
  in the data zone.
"""

import json
import os
import runpy
import sys
import traceback
import warnings


def sanitize(exc: BaseException, script: str) -> dict:
    """Type, line, and source line of the failure, taken from the script's own frames."""
    if isinstance(exc, SyntaxError) and exc.filename == script:
        return {"type": type(exc).__name__, "line": exc.lineno, "source": (exc.text or "").strip() or None}
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == script]
    last = frames[-1] if frames else None  # the script line that called into the failing code
    return {
        "type": type(exc).__name__,
        "line": last.lineno if last else None,
        "source": last.line if last else None,
    }


def main() -> int:
    script, exception_json = os.path.abspath(sys.argv[1]), sys.argv[2]

    import pandas as pd

    warnings.filterwarnings("error", category=pd.errors.ChainedAssignmentError)
    sys.argv = [script]
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        if exc.code in (None, 0):
            return 0
        error = exc
    except BaseException as exc:  # every failure must be reported
        error = exc
    else:
        return 0

    traceback.print_exception(error)
    with open(exception_json, "w") as f:
        json.dump(sanitize(error, script), f)
    return 1


if __name__ == "__main__":
    sys.exit(main())
