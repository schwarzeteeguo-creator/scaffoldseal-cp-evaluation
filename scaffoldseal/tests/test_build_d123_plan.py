import json
from pathlib import Path
import sys

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_d123_plan import build, canonical_sha256, recompute_contracts


def test_d123_geometry_and_locks():
    root = Path(__file__).resolve().parents[1]
    plan = build(root)
    assert plan["authorization"] == {
        "accepted": False,
        "execution_authorized": False,
        "gpu_training_allowed": False,
    }
    assert plan["counts"]["stages_total"] == 551
    assert plan["counts"]["scientific_fits"] == 491
    stages = plan["stages"]
    assert len(stages) == len({s["scientific_identity_sha256"] for s in stages})
    assert sum(s["kind"] == "d123_pampa_inner_fit" for s in stages) == 216
    assert sum(s["kind"] == "d123_pampa_outer_fit_prediction" for s in stages) == 270
    assert sum(s["kind"] == "d123_sealed_oof_metrics" for s in stages) == 3
    assert sum(s["kind"] == "d123_metric_label_release_gate" for s in stages) == 3
    assert all(
        s["protocol_lock_sha256"] == plan["protocol_lock_sha256"] for s in stages
    )
    assert all(len(s["stage_spec_sha256"]) == 64 for s in stages)
    assert all(type(s["execution"]["gpu"]) is bool for s in stages)
    assert all(s["expected_outputs"] for s in stages)
    assert all(
        s["stage_spec_sha256"]
        == canonical_sha256(
            {
                "kind": s["kind"],
                "key": s["key"],
                "dependencies": s["dependencies"],
                "protocol_lock_sha256": s["protocol_lock_sha256"],
                "execution": s["execution"],
                "expected_outputs": s["expected_outputs"],
                "scientific_identity_sha256": s["scientific_identity_sha256"],
            }
        )
        for s in stages
    )
    assert all(
        s["execution"]["gpu"]
        for s in stages
        if s["kind"]
        in (
            "d23_descriptor_delaney_pretraining",
            "d123_pampa_inner_fit",
            "d123_pampa_outer_fit_prediction",
        )
    )
    assert all(
        not s["execution"]["gpu"]
        for s in stages
        if s["kind"]
        in (
            "d123_stopping_epoch_selection",
            "d123_metric_label_release_gate",
            "d123_sealed_oof_metrics",
        )
    )
    unlocked = {**plan}
    observed = unlocked.pop("plan_sha256")
    assert observed == canonical_sha256(unlocked)


def test_d1_and_d23_pretraining_dependencies_are_frozen():
    root = Path(__file__).resolve().parents[1]
    plan = build(root)
    d0 = json.loads((root / "d0_full_run_manifest.json").read_text())
    d0_ids = {
        s["scientific_identity_sha256"]
        for s in d0["stages"]
        if s["kind"] == "delaney_pretraining"
    }
    descriptor_ids = {
        s["scientific_identity_sha256"]
        for s in plan["stages"]
        if s["kind"] == "d23_descriptor_delaney_pretraining"
    }
    inner = [s for s in plan["stages"] if s["kind"] == "d123_pampa_inner_fit"]
    assert all(
        s["dependencies"][0] in (d0_ids if s["key"]["variant"] == "D1" else descriptor_ids)
        for s in inner
    )
    for relative in (
        "src/split_safe.py",
        "src/r1c0_dmpnn_pilot.py",
        "src/h1_random_cv_runner.py",
        "src/r1a_classical.py",
    ):
        assert relative in plan["locked_files"]
    delaney = plan["scientific_lock"]["d23_delaney_source"]
    assert delaney["size_bytes"] == 97828
    assert delaney["sha256"] == (
        "a4a8ee8be4f368dada6f4d619e74ce3824e48fe4fc119f4d19f461693466f028"
    )


def test_outer_training_ids_are_explicitly_bound():
    root = Path(__file__).resolve().parents[1]
    plan = build(root)
    outer = [
        s for s in plan["stages"] if s["kind"] == "d123_pampa_outer_fit_prediction"
    ]
    assert all(s["key"]["n_outer_train_ids"] > 0 for s in outer)
    assert all(len(s["key"]["outer_train_ids_sha256"]) == 64 for s in outer)
    assert "src/build_d123_plan.py" in plan["locked_files"]
    sources = plan["scientific_lock"]["fold_scoped_training_label_sources"]
    assert set(sources) == {str(fold) for fold in range(1, 19)}
    assert all(
        item["heldout_target_values_materialized"] is False
        and item["n_training_targets"] > 0
        and len(item["training_ids_sha256"]) == 64
        and len(item["heldout_ids_sha256"]) == 64
        for item in sources.values()
    )


def test_metric_labels_require_a_real_dependency_gate():
    root = Path(__file__).resolve().parents[1]
    plan = build(root)
    by_id = {s["scientific_identity_sha256"]: s for s in plan["stages"]}
    metrics = [s for s in plan["stages"] if s["kind"] == "d123_sealed_oof_metrics"]
    for metric in metrics:
        assert len(metric["dependencies"]) == 1
        gate = by_id[metric["dependencies"][0]]
        assert gate["kind"] == "d123_metric_label_release_gate"
        assert len(gate["dependencies"]) == 90
        assert gate["key"]["label_source_sha256"] == plan["scientific_lock"][
            "metric_label_source"
        ]["sha256"]


def test_d1_dependencies_bind_exact_accepted_checkpoint_receipts():
    root = Path(__file__).resolve().parents[1]
    plan = build(root)
    receipts = plan["scientific_lock"]["d1_accepted_checkpoint_receipts"]
    assert set(receipts) == {"0", "1", "2", "3", "4"}
    for seed, receipt in receipts.items():
        assert receipt["attempt_id"].startswith("attempt_")
        assert len(receipt["checkpoint"]["sha256"]) == 64
        assert len(receipt["ledger"]["chain_head_sha256"]) == 64
        assert len(receipt["receipt_sha256"]) == 64
    d1 = [
        s
        for s in plan["stages"]
        if s["kind"] in ("d123_pampa_inner_fit", "d123_pampa_outer_fit_prediction")
        and s["key"]["variant"] == "D1"
    ]
    for item in d1:
        seed = str(item["key"]["seed_index"])
        assert item["key"]["d1_checkpoint_receipt_sha256"] == receipts[seed][
            "receipt_sha256"
        ]


def test_contract_hash_mutation_is_rejected():
    root = Path(__file__).resolve().parents[1]
    outer = pd.read_csv(root / "artifacts/v2_r0/outer_record_assignments.csv")
    baskets = pd.read_csv(root / "artifacts/v2_r0/inner_basket_manifest.csv")
    contracts = pd.read_csv(root / "artifacts/v2_r0/pre_fit_contract_manifest.csv")
    mutated = contracts.copy()
    mutated.loc[0, "fit_ids_sha256"] = "0" * 64
    try:
        recompute_contracts(outer, baskets, mutated)
    except RuntimeError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("Mutated contract hash was accepted")


def test_basket_assignment_mutation_is_rejected_against_frozen_contract():
    root = Path(__file__).resolve().parents[1]
    outer = pd.read_csv(root / "artifacts/v2_r0/outer_record_assignments.csv")
    baskets = pd.read_csv(root / "artifacts/v2_r0/inner_basket_manifest.csv")
    contracts = pd.read_csv(root / "artifacts/v2_r0/pre_fit_contract_manifest.csv")
    mutated = baskets.copy()
    row = mutated.index[mutated["outer_fold"].astype(int).eq(1)][0]
    original = int(mutated.loc[row, "inner_basket"])
    mutated.loc[row, "inner_basket"] = 1 + (original % 4)
    try:
        recompute_contracts(outer, mutated, contracts)
    except RuntimeError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("Mutated inner basket assignment was accepted")
