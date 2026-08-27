from __future__ import annotations

import pandas as pd
import numpy as np

from d123_metrics import compute_frozen_metrics


def test_frozen_source_macro_and_row_micro_metrics():
    predictions = pd.DataFrame(
        {
            "curated_id": ["a", "b", "c", "a", "b", "c"],
            "outer_fold": [1, 1, 2, 1, 1, 2],
            "seed_index": [0, 0, 0, 1, 1, 1],
            "prediction_log10_papp": [0.0, 2.0, 3.0, 1.0, 1.0, 4.0],
        }
    )
    labels = pd.DataFrame(
        {
            "curated_id": ["a", "b", "c"],
            "source": ["S1", "S1", "S2"],
            "sealed_block_id": ["B1", "B1", "B2"],
            "permeability": [1.0, 1.0, 2.0],
        }
    )
    metrics = compute_frozen_metrics(predictions, labels)
    assert metrics["source_macro_mae_by_seed"] == {"0": 1.0, "1": 1.0}
    assert metrics["source_macro_mae_mean_across_seeds"] == 1.0
    assert metrics["row_micro_mae_by_seed"] == {"0": 1.0, "1": 2.0 / 3.0}
    assert np.isclose(metrics["row_micro_mae_mean_across_seeds"], 5.0 / 6.0)
    assert metrics["n_prediction_rows"] == 6


def test_metrics_reject_missing_labels_and_duplicate_slots():
    labels = pd.DataFrame(
        {
            "curated_id": ["a"],
            "source": ["S"],
            "sealed_block_id": ["B"],
            "permeability": [1.0],
        }
    )
    duplicate = pd.DataFrame(
        {
            "curated_id": ["a", "a"],
            "outer_fold": [1, 1],
            "seed_index": [0, 0],
            "prediction_log10_papp": [1.0, 1.0],
        }
    )
    try:
        compute_frozen_metrics(duplicate, labels)
    except ValueError as error:
        assert "duplicated" in str(error)
    else:
        raise AssertionError("Duplicate D123 metric slots were accepted")
    missing = duplicate.iloc[:1].assign(curated_id="missing")
    try:
        compute_frozen_metrics(missing, labels)
    except RuntimeError as error:
        assert "lacks" in str(error)
    else:
        raise AssertionError("A prediction without a frozen label was accepted")
