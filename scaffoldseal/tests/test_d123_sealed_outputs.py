from __future__ import annotations

import json

import numpy as np
import pandas as pd

from d123_sealed_outputs import seal_outer_predictions


def _stage():
    return {
        "kind": "d123_pampa_outer_fit_prediction",
        "key": {"variant": "D3", "outer_fold": 7, "seed_index": 2},
        "scientific_identity_sha256": "a" * 64,
        "stage_spec_sha256": "b" * 64,
    }


def test_predictions_are_lossless_label_free_and_exclusive(tmp_path):
    frame = pd.DataFrame(
        {"curated_id": ["x", "y"], "prediction_normalized": [0.25, 0.75]}
    )
    payload = seal_outer_predictions(_stage(), frame, tmp_path)
    observed = json.loads((tmp_path / "predictions.lossless.json").read_text())
    assert observed == payload
    assert observed["contains_observed_labels"] is False
    assert observed["n_predictions"] == 2
    assert observed["records"][0]["prediction_normalized_ieee754_be"] == "3fd0000000000000"
    assert observed["records"][0]["prediction_log10_papp_ieee754_be"] == "c016000000000000"
    csv = pd.read_csv(tmp_path / "predictions.csv")
    assert list(csv.columns) == [
        "curated_id",
        "outer_fold",
        "seed_index",
        "prediction_normalized",
        "prediction_log10_papp",
    ]
    assert not any("label" in column or "observed" in column for column in csv.columns)
    try:
        seal_outer_predictions(_stage(), frame, tmp_path)
    except RuntimeError as error:
        assert "overwrite" in str(error)
    else:
        raise AssertionError("Sealed prediction artifacts were overwritten")


def test_prediction_input_rejects_labels_duplicates_and_nonfinite(tmp_path):
    cases = [
        pd.DataFrame(
            {"curated_id": ["x"], "prediction_normalized": [0.1], "label": [1.0]}
        ),
        pd.DataFrame(
            {"curated_id": ["x", "x"], "prediction_normalized": [0.1, 0.2]}
        ),
        pd.DataFrame(
            {"curated_id": ["x"], "prediction_normalized": [np.nan]}
        ),
    ]
    for index, frame in enumerate(cases):
        target = tmp_path / str(index)
        try:
            seal_outer_predictions(_stage(), frame, target)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe pre-seal prediction input was accepted")
        assert not target.exists()
