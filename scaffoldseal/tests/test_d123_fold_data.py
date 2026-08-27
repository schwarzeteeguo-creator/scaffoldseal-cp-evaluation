from __future__ import annotations

import json
from pathlib import Path

from d123_fold_data import canonical_id_hash, load_fold_frames


def _plan():
    root = Path(__file__).resolve().parents[1]
    return root, json.loads((root / "d123_plan_candidate.json").read_text())


def test_inner_fold_frames_match_locked_hashes_and_test_has_no_target():
    root, plan = _plan()
    stage = next(
        item
        for item in plan["stages"]
        if item["kind"] == "d123_pampa_inner_fit"
        and item["key"]["variant"] == "D3"
        and item["key"]["outer_fold"] == 7
        and item["key"]["inner_basket"] == 2
    )
    frames = load_fold_frames(plan, stage, root)
    assert canonical_id_hash(frames["fit"]["curated_id"]) == stage["key"][
        "fit_ids_sha256"
    ]
    assert canonical_id_hash(frames["validation"]["curated_id"]) == stage["key"][
        "validation_ids_sha256"
    ]
    assert "normalized_pampa" not in frames["outer_test_label_free"].columns
    assert set(frames["fit"]["curated_id"]).isdisjoint(
        frames["validation"]["curated_id"]
    )


def test_outer_fold_frames_have_targets_only_for_training():
    root, plan = _plan()
    stage = next(
        item
        for item in plan["stages"]
        if item["kind"] == "d123_pampa_outer_fit_prediction"
        and item["key"]["variant"] == "D2"
        and item["key"]["outer_fold"] == 3
        and item["key"]["seed_index"] == 4
    )
    frames = load_fold_frames(plan, stage, root)
    assert "normalized_pampa" in frames["outer_train"].columns
    assert "normalized_pampa" not in frames["outer_test_label_free"].columns
    assert len(frames["outer_train"]) == stage["key"]["n_outer_train_ids"]
    assert set(frames["outer_train"]["curated_id"]).isdisjoint(
        frames["outer_test_label_free"]["curated_id"]
    )
