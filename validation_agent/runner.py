"""Runs one translated script and reports how it ended (brief item F, D-047).

Usage: python -I runner.py SCRIPT EXCEPTION_JSON     (started by graph.execute; never imported)

- pandas' ChainedAssignmentError is raised as an error. Under pandas 3 it is only a warning and the
  assignment silently does nothing, so the script would finish with wrong values.
- On a failure, EXCEPTION_JSON gets the exception type, the script line and that line of code. The
  message is left out because it can contain data values (e.g. KeyError: '010001'). The full
  traceback goes to stderr, which stays in the data zone.
"""

import json
import runpy
import sys
import traceback
import warnings


def main() -> int:
    script, exception_json = sys.argv[1], sys.argv[2]

    import pandas as pd

    warnings.filterwarnings("error", category=pd.errors.ChainedAssignmentError)
    sys.argv = [script]
    try:
        runpy.run_path(script, run_name="__main__")
        return 0
    except SystemExit as exc:
        if exc.code in (None, 0):
            return 0
        error = exc
    except BaseException as exc:  # every failure must be reported
        error = exc

    traceback.print_exception(error)
    if isinstance(error, SyntaxError) and error.filename == script:
        line, source = error.lineno, (error.text or "").strip()
    else:
        frames = [f for f in traceback.extract_tb(error.__traceback__) if f.filename == script]
        line, source = (frames[-1].lineno, frames[-1].line) if frames else (None, None)
    with open(exception_json, "w", encoding="utf-8") as f:
        json.dump({"type": type(error).__name__, "line": line, "source": source}, f)
    return 1


if __name__ == "__main__":
    sys.exit(main())
