from sas_agent.static_check import check


def test_allowed_code_passes():
    code = "import os, json\nimport numpy as np\nimport pandas as pd\nfrom numpy.linalg import norm\n"
    assert check(code) == []


def test_syntax_error_reports_rule_and_line():
    # The typo the skeleton guide reports from a real gemini run (sas_agent/README.md, 9.3).
    violations = check("import pandas as pd\nheader_ cols = ['a']\n")
    assert len(violations) == 1
    assert violations[0].rule == "syntax"
    assert violations[0].line == 2


def test_disallowed_imports_are_reported_with_lines():
    code = "import pandas as pd\nfrom sklearn.cluster import KMeans\nimport scipy.stats\n"
    violations = check(code)
    assert [(v.rule, v.line) for v in violations] == [("import", 2), ("import", 3)]
    assert "sklearn" in violations[0].detail and "scipy" in violations[1].detail


def test_relative_import_is_rejected():
    violations = check("from . import helpers\n")
    assert [v.rule for v in violations] == ["import"]
