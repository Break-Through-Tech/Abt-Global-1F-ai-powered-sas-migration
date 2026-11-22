import numpy as np
import pandas as pd
import pyreadstat

from sas_agent import config, runs


def test_attempt_dirs_layout(tmp_path):
    dirs = runs.attempt_dirs("run1", "program0", 2, runs_dir=tmp_path)
    assert dirs.root == tmp_path / "run1" / "program0" / "attempt-2"
    assert dirs.ai.is_dir() and dirs.outputs.is_dir()
    assert dirs.outputs.parent == dirs.data


def test_new_run_ids_are_unique():
    assert len({runs.new_run_id() for _ in range(20)}) == 20


def test_prepared_input_matches_sas7bdat_exactly(tmp_path):
    inputs = runs.prepare_inputs("run1", runs_dir=tmp_path)
    csv = pd.read_csv(inputs / "alldata_2025jul.csv", dtype={"PROVIDER_ID": str}, float_precision="round_trip")
    ref, _ = pyreadstat.read_sas7bdat(str(config.INPUT_DATASET))
    assert list(csv.columns) == list(ref.columns)
    assert (csv["PROVIDER_ID"] == ref["PROVIDER_ID"]).all()
    numeric = [c for c in ref.columns if c != "PROVIDER_ID"]
    assert np.array_equal(csv[numeric].to_numpy(float), ref[numeric].to_numpy(float), equal_nan=True)
