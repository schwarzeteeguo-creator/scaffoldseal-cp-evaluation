import copy

import pandas as pd
import pytest

from d123_outer_issuance_resume import authoritative_outer_records
from split_safe import OuterFoldContract, SplitViolation


def _frames():
    return {
        "outer_train": pd.DataFrame(
            {
                "curated_id": ["a", "b", "d", "e"],
                "SMILES": ["CC", "CO", "CCC", "CCO"],
                "normalized_pampa": [0.1, 0.2, 0.3, 0.4],
            }
        ),
        "outer_test_label_free": pd.DataFrame(
            {"curated_id": ["c"], "SMILES": ["CN"]}
        ),
    }


def test_authoritative_outer_records_preserve_label_seal_and_contract():
    frames = _frames()
    combined = authoritative_outer_records(frames)
    assert frames["outer_test_label_free"].columns.tolist() == ["curated_id", "SMILES"]
    assert pd.isna(combined.loc[combined.curated_id.eq("c"), "normalized_pampa"]).all()
    contract = OuterFoldContract(
        1, ["a", "b", "d", "e"], ["c"], {"a": 1, "b": 2, "d": 3, "e": 4}
    )
    batch = contract.outer_frame_training_batch(combined)
    token = contract.validate_authoritative_outer_training_batch(batch)
    prediction, _, _ = contract.authoritative_outer_prediction_frame(
        ["SMILES"], "normalized_pampa", token
    )
    assert prediction.to_dict("records") == [{"curated_id": "c", "SMILES": "CN"}]


def test_authoritative_outer_records_missing_test_fails_closed():
    frames = _frames()
    combined = authoritative_outer_records(frames)
    contract = OuterFoldContract(
        1, ["a", "b", "d", "e"], ["c"], {"a": 1, "b": 2, "d": 3, "e": 4}
    )
    with pytest.raises(SplitViolation, match="all authoritative train/test"):
        contract.outer_frame_training_batch(combined.loc[combined.curated_id.ne("c")])


def test_outer_test_target_materialization_is_rejected():
    frames = copy.deepcopy(_frames())
    frames["outer_test_label_free"]["normalized_pampa"] = 9.9
    with pytest.raises(RuntimeError, match="unexpectedly materialized"):
        authoritative_outer_records(frames)
