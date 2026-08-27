"""Build the label-blind, execution-locked D1/D2/D3 stage graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from d123_dmpnn_integration import descriptor_definition_payload


VARIANTS = ("D1", "D2", "D3")
FOLDS = tuple(range(1, 19))
SEEDS = tuple(range(5))
BASKETS = tuple(range(1, 5))
SEED_VALUES = (0, 123, 492, 1107, 1968)


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id_hash(values) -> str:
    ids = sorted({str(value) for value in values})
    return hashlib.sha256("".join(f"{value}\n" for value in ids).encode()).hexdigest()


def recompute_contracts(
    outer: pd.DataFrame, baskets: pd.DataFrame, contracts: pd.DataFrame
) -> dict[tuple[int, int], dict[str, object]]:
    all_ids = set(outer["curated_id"].astype(str))
    expected: dict[tuple[int, int], dict[str, object]] = {}
    for fold in FOLDS:
        test_ids = set(
            outer.loc[outer["outer_fold"].astype(int).eq(fold), "curated_id"].astype(str)
        )
        outer_train = outer.loc[~outer["outer_fold"].astype(int).eq(fold)].copy()
        for basket in BASKETS:
            validation_blocks = set(
                baskets.loc[
                    baskets["outer_fold"].astype(int).eq(fold)
                    & baskets["inner_basket"].astype(int).eq(basket),
                    "sealed_block_id",
                ].astype(str)
            )
            validation_ids = set(
                outer_train.loc[
                    outer_train["sealed_block_id"].astype(str).isin(validation_blocks),
                    "curated_id",
                ].astype(str)
            )
            fit_ids = all_ids - test_ids - validation_ids
            if fit_ids & validation_ids or fit_ids & test_ids or validation_ids & test_ids:
                raise RuntimeError("Recomputed D123 contract roles overlap")
            if fit_ids | validation_ids | test_ids != all_ids:
                raise RuntimeError("Recomputed D123 contract is not a full partition")
            expected[(fold, basket)] = {
                "n_fit_ids": len(fit_ids),
                "fit_ids_sha256": canonical_id_hash(fit_ids),
                "n_inner_validation_ids": len(validation_ids),
                "inner_validation_ids_sha256": canonical_id_hash(validation_ids),
                "n_outer_test_ids": len(test_ids),
                "outer_test_ids_sha256": canonical_id_hash(test_ids),
                "n_outer_train_ids": len(all_ids - test_ids),
                "outer_train_ids_sha256": canonical_id_hash(all_ids - test_ids),
            }
    for row in contracts.itertuples(index=False):
        key = (int(row.outer_fold), int(row.inner_basket))
        if key not in expected:
            raise RuntimeError("Unexpected frozen contract row")
        for field in (
            "n_fit_ids",
            "fit_ids_sha256",
            "n_inner_validation_ids",
            "inner_validation_ids_sha256",
            "n_outer_test_ids",
            "outer_test_ids_sha256",
        ):
            if str(getattr(row, field)) != str(expected[key][field]):
                raise RuntimeError(f"Frozen contract mismatch after independent recomputation: {key} {field}")
    if len(contracts) != len(expected):
        raise RuntimeError("Frozen contract row count drifted")
    return expected


def accepted_d0_checkpoint_receipts(
    root: Path, d0_pretrain: dict[int, str]
) -> tuple[dict[int, dict[str, object]], list[Path]]:
    ledger_root = root / "artifacts/r1c0_d0_full_ledger_v1"
    receipts: dict[int, dict[str, object]] = {}
    locked_paths: list[Path] = []
    for seed_index in SEEDS:
        identity = d0_pretrain[seed_index]
        ledger_path = ledger_root / f"{identity}.json"
        if not ledger_path.is_file():
            raise RuntimeError("Accepted D0 pretraining ledger receipt is absent")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("scientific_identity_sha256") != identity:
            raise RuntimeError("D0 ledger scientific identity drifted")
        accepted = [
            attempt
            for attempt in ledger.get("attempts", [])
            if attempt.get("status") == "COMPLETED_ACCEPTED"
            and attempt.get("accepted_success_artifact") is True
        ]
        if len(accepted) != 1:
            raise RuntimeError("D0 seed lacks one exact accepted checkpoint attempt")
        attempt = accepted[0]
        artifacts = {
            str(item["relative_path"]): item
            for item in attempt.get("details", {}).get("artifacts", [])
        }
        checkpoint_record = artifacts.get("checkpoint1.pt")
        manifest_record = artifacts.get("artifact_manifest.json")
        if not isinstance(checkpoint_record, dict) or not isinstance(manifest_record, dict):
            raise RuntimeError("D0 accepted attempt lacks checkpoint/manifest records")
        output = (root / str(attempt["output_namespace"])).resolve()
        try:
            output.relative_to(root)
        except ValueError as error:
            raise RuntimeError("D0 accepted checkpoint namespace escapes project") from error
        checkpoint_path = output / "checkpoint1.pt"
        manifest_path = output / "artifact_manifest.json"
        for path, record in (
            (checkpoint_path, checkpoint_record),
            (manifest_path, manifest_record),
        ):
            if (
                not path.is_file()
                or path.stat().st_size != int(record["size_bytes"])
                or stream_sha256(path) != str(record["sha256"])
            ):
                raise RuntimeError("D0 accepted checkpoint artifact failed exact validation")
        receipt = {
            "seed_index": seed_index,
            "scientific_identity_sha256": identity,
            "attempt_id": str(attempt["attempt_id"]),
            "attempt_number": int(attempt["attempt_number"]),
            "output_namespace": str(attempt["output_namespace"]),
            "checkpoint": {
                "relative_path": str(checkpoint_path.relative_to(root)).replace("\\", "/"),
                "size_bytes": checkpoint_path.stat().st_size,
                "sha256": stream_sha256(checkpoint_path),
            },
            "artifact_manifest": {
                "relative_path": str(manifest_path.relative_to(root)).replace("\\", "/"),
                "size_bytes": manifest_path.stat().st_size,
                "sha256": stream_sha256(manifest_path),
            },
            "ledger": {
                "relative_path": str(ledger_path.relative_to(root)).replace("\\", "/"),
                "size_bytes": ledger_path.stat().st_size,
                "sha256": stream_sha256(ledger_path),
                "chain_head_sha256": str(ledger["chain_head_sha256"]),
                "summary_sha256": str(ledger["summary_sha256"]),
            },
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        receipts[seed_index] = receipt
        locked_paths.extend((ledger_path, checkpoint_path, manifest_path))
    return receipts, locked_paths


def stage(
    kind: str,
    key: dict[str, object],
    dependencies: list[str],
    protocol_lock_sha256: str,
) -> dict[str, object]:
    variant = str(key.get("variant", ""))
    contracts = {
        "d23_descriptor_delaney_pretraining": (
            True,
            ["checkpoint1.pt", "training_trace.json", "artifact_manifest.json"],
        ),
        "d123_pampa_inner_fit": (
            True,
            [
                "checkpoint1.pt",
                "training_trace.json",
                "artifact_manifest.json",
                *(
                    ["descriptor_transform.json"]
                    if variant in ("D2", "D3")
                    else []
                ),
            ],
        ),
        "d123_stopping_epoch_selection": (
            False,
            ["stopping_epoch.json", "artifact_manifest.json"],
        ),
        "d123_pampa_outer_fit_prediction": (
            True,
            [
                "checkpoint1.pt",
                "training_trace.json",
                "predictions.lossless.json",
                "predictions.csv",
                "artifact_manifest.json",
                *(
                    ["descriptor_transform.json"]
                    if variant in ("D2", "D3")
                    else []
                ),
            ],
        ),
        "d123_metric_label_release_gate": (
            False,
            ["label_release_receipt.json", "artifact_manifest.json"],
        ),
        "d123_sealed_oof_metrics": (
            False,
            ["metrics.json", "artifact_manifest.json"],
        ),
    }
    if kind not in contracts:
        raise ValueError(f"Unknown D123 stage kind: {kind}")
    gpu, expected_outputs = contracts[kind]
    scientific = {
        "kind": kind,
        "key": key,
        "dependencies": dependencies,
        "protocol_lock_sha256": protocol_lock_sha256,
        "execution": {"gpu": gpu},
        "expected_outputs": expected_outputs,
    }
    identity = canonical_sha256(scientific)
    stage_spec_sha256 = canonical_sha256(
        {
            **scientific,
            "scientific_identity_sha256": identity,
        }
    )
    return {
        **scientific,
        "scientific_identity_sha256": identity,
        "stage_spec_sha256": stage_spec_sha256,
        "namespace": {
            "work_attempt_template": (
                f"runs/d123_v1/{kind}/{identity}/attempt_{{attempt_number:03d}}"
            ),
            "output_attempt_template": (
                f"artifacts/d123_v1/{kind}/{identity}/attempt_{{attempt_number:03d}}"
            ),
        },
    }


def build(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    inputs = root / "artifacts/d123_inputs_v1"
    contracts_path = root / "artifacts/v2_r0/pre_fit_contract_manifest.csv"
    outer_path = root / "artifacts/v2_r0/outer_record_assignments.csv"
    baskets_path = root / "artifacts/v2_r0/inner_basket_manifest.csv"
    d0_manifest_path = root / "d0_full_run_manifest.json"
    metric_labels_path = root / "artifacts/v2_r0/analysis_all_labels.csv"
    fold_target_root = root / "artifacts/v2_r0/d0_full_runner_repair1"
    fold_target_manifest_path = fold_target_root / "fold_scoped_target_manifest.json"
    label_free_features_path = fold_target_root / "label_free_features.csv"
    d23_delaney_path = (
        root.parent
        / "baseline_candidates/BenchmarkCycPeptMP/CSV/PreTrainData/delaney-processed.csv"
    )
    sources = [
        root / "src/build_d123_plan.py",
        root / "src/d123_features.py",
        root / "src/d123_dmpnn_integration.py",
        root / "src/d123_runner_governance.py",
        root / "src/d123_runner.py",
        root / "src/d123_locked_adapter.py",
        root / "src/d123_pretraining.py",
        root / "src/d123_sealed_outputs.py",
        root / "src/d123_metrics.py",
        root / "src/d123_fold_data.py",
        root / "src/audit_d123_prefit.py",
        root / "src/build_d123_inputs.py",
        root / "src/build_d123_acceptance.py",
        root / "tests/test_build_d123_plan.py",
        root / "tests/test_d123_features.py",
        root / "tests/test_d123_dmpnn_integration.py",
        root / "tests/test_d123_runner_governance.py",
        root / "tests/test_d123_runner.py",
        root / "tests/test_d123_locked_adapter.py",
        root / "tests/test_d123_pretraining.py",
        root / "tests/test_d123_sealed_outputs.py",
        root / "tests/test_d123_metrics.py",
        root / "tests/test_d123_fold_data.py",
        root / "tests/test_audit_d123_prefit.py",
        root / "src/split_safe.py",
        root / "src/r1c0_dmpnn_pilot.py",
        root / "src/h1_random_cv_runner.py",
        root / "src/r1a_classical.py",
        inputs / "raw_descriptors.csv",
        inputs / "group_metadata.csv",
        inputs / "provenance.json",
        inputs / "manifest.json",
        contracts_path,
        outer_path,
        baskets_path,
        d0_manifest_path,
        metric_labels_path,
        fold_target_manifest_path,
        label_free_features_path,
    ]
    if any(not path.is_file() for path in sources):
        raise RuntimeError("A frozen D123 plan input is absent")
    if not d23_delaney_path.is_file():
        raise RuntimeError("External D23 Delaney source is absent")
    contracts = pd.read_csv(contracts_path)
    outer = pd.read_csv(outer_path)
    baskets = pd.read_csv(baskets_path)
    fold_target_manifest = json.loads(
        fold_target_manifest_path.read_text(encoding="utf-8")
    )
    if len(contracts) != 72 or set(contracts["outer_fold"].astype(int)) != set(FOLDS):
        raise RuntimeError("Frozen 18x4 contract geometry drifted")
    if len(outer) != 6895 or outer["curated_id"].duplicated().any():
        raise RuntimeError("Frozen D123 population drifted")
    recomputed_contracts = recompute_contracts(outer, baskets, contracts)
    target_records = {
        int(item["outer_fold"]): dict(item)
        for item in fold_target_manifest["fold_training_targets"]
    }
    if set(target_records) != set(FOLDS):
        raise RuntimeError("Fold-scoped training-target geometry drifted")
    training_label_sources: dict[str, dict[str, object]] = {}
    all_ids = set(outer["curated_id"].astype(str))
    for fold in FOLDS:
        record = target_records[fold]
        path = fold_target_root / str(record["relative_path"])
        if not path.is_file():
            raise RuntimeError("Fold-scoped training-target file is absent")
        targets = pd.read_csv(path)
        if list(targets.columns) != ["curated_id", "normalized_pampa"]:
            raise RuntimeError("Fold-scoped training-target schema drifted")
        ids = set(targets["curated_id"].astype(str))
        heldout = set(
            outer.loc[
                outer["outer_fold"].astype(int).eq(fold), "curated_id"
            ].astype(str)
        )
        if (
            ids != all_ids - heldout
            or ids & heldout
            or len(ids) != int(record["n_training_targets"])
            or canonical_id_hash(ids) != str(record["training_ids_sha256"])
            or stream_sha256(path) != str(record["sha256"])
            or path.stat().st_size != int(record["size_bytes"])
        ):
            raise RuntimeError("Fold-scoped training-target contract drifted")
        locked_record = {
            "outer_fold": fold,
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": stream_sha256(path),
            "n_training_targets": len(ids),
            "training_ids_sha256": canonical_id_hash(ids),
            "heldout_ids_sha256": canonical_id_hash(heldout),
            "heldout_target_values_materialized": False,
        }
        training_label_sources[str(fold)] = locked_record
        sources.append(path)
    d0 = json.loads(d0_manifest_path.read_text(encoding="utf-8"))
    d0_pretrain = {
        int(item["key"]["seed_index"]): str(item["scientific_identity_sha256"])
        for item in d0["stages"]
        if item["kind"] == "delaney_pretraining"
    }
    if set(d0_pretrain) != set(SEEDS):
        raise RuntimeError("Accepted D0 seed-specific pretraining geometry drifted")
    d0_checkpoint_receipts, d0_receipt_paths = accepted_d0_checkpoint_receipts(
        root, d0_pretrain
    )
    sources.extend(d0_receipt_paths)

    definition = descriptor_definition_payload()
    locked_files = {
        str(path.relative_to(root)).replace("\\", "/"): {
            "size_bytes": path.stat().st_size,
            "sha256": stream_sha256(path),
        }
        for path in sources
    }
    scientific_lock = {
        "variants": list(VARIANTS),
        "outer_folds": list(FOLDS),
        "inner_baskets": list(BASKETS),
        "seed_indices": list(SEEDS),
        "effective_rng_seeds": list(SEED_VALUES),
        "maximum_epochs": 2000,
        "patience": 200,
        "stopping_seed": 0,
        "descriptor_definition": definition,
        "descriptor_definition_sha256": canonical_sha256(definition),
        "d1_pretraining": "reuse_exact_D0_seed_specific_checkpoints",
        "d1_accepted_checkpoint_receipts": {
            str(seed): d0_checkpoint_receipts[seed] for seed in SEEDS
        },
        "d23_pretraining": "shared_seed_specific_DMPNN_27D_all_zero_Delaney",
        "d23_delaney_source": {
            "baseline_relative_path": "CSV/PreTrainData/delaney-processed.csv",
            "size_bytes": d23_delaney_path.stat().st_size,
            "sha256": stream_sha256(d23_delaney_path),
            "target_column": "measured log solubility in mols per litre",
        },
        "heldout_policy": "sealed_predictions_before_metric_only_labels",
        "metric_label_source": {
            "relative_path": "artifacts/v2_r0/analysis_all_labels.csv",
            "size_bytes": metric_labels_path.stat().st_size,
            "sha256": stream_sha256(metric_labels_path),
            "access_role": "METRIC_ONLY_AFTER_VARIANT_PREDICTION_GATE",
        },
        "fold_scoped_training_label_sources": training_label_sources,
        "label_free_feature_source": {
            "relative_path": label_free_features_path.relative_to(root).as_posix(),
            "size_bytes": label_free_features_path.stat().st_size,
            "sha256": stream_sha256(label_free_features_path),
            "access_role": "ALL_ROLES_WITHOUT_TARGET",
        },
    }
    protocol_lock_sha256 = canonical_sha256(
        {"scientific_lock": scientific_lock, "locked_files": locked_files}
    )

    stages: list[dict[str, object]] = []
    descriptor_pretrain: dict[int, str] = {}
    for seed_index, effective_seed in zip(SEEDS, SEED_VALUES):
        item = stage(
            "d23_descriptor_delaney_pretraining",
            {
                "seed_index": seed_index,
                "effective_rng_seed": effective_seed,
                "global_features_size": 27,
                "global_features_policy": "all_zero",
                "shared_by": ["D2", "D3"],
            },
            [],
            protocol_lock_sha256,
        )
        stages.append(item)
        descriptor_pretrain[seed_index] = str(item["scientific_identity_sha256"])

    selections: dict[tuple[str, int], str] = {}
    outer_predictions: dict[str, list[str]] = {variant: [] for variant in VARIANTS}
    for variant in VARIANTS:
        for fold in FOLDS:
            inner_ids = []
            for basket in BASKETS:
                contract = contracts.loc[
                    contracts["outer_fold"].astype(int).eq(fold)
                    & contracts["inner_basket"].astype(int).eq(basket)
                ]
                if len(contract) != 1:
                    raise RuntimeError("Missing exact frozen inner contract")
                row = contract.iloc[0]
                recomputed = recomputed_contracts[(fold, basket)]
                dependency = (
                    d0_pretrain[0]
                    if variant == "D1"
                    else descriptor_pretrain[0]
                )
                d1_receipt = (
                    d0_checkpoint_receipts[0]["receipt_sha256"]
                    if variant == "D1"
                    else None
                )
                item = stage(
                    "d123_pampa_inner_fit",
                    {
                        "variant": variant,
                        "outer_fold": fold,
                        "inner_basket": basket,
                        "seed_index": 0,
                        "effective_rng_seed": 0,
                        "fit_ids_sha256": str(recomputed["fit_ids_sha256"]),
                        "validation_ids_sha256": str(
                            recomputed["inner_validation_ids_sha256"]
                        ),
                        "outer_test_ids_sha256": str(recomputed["outer_test_ids_sha256"]),
                        "d1_checkpoint_receipt_sha256": d1_receipt,
                    },
                    [dependency],
                    protocol_lock_sha256,
                )
                stages.append(item)
                inner_ids.append(str(item["scientific_identity_sha256"]))
            select = stage(
                "d123_stopping_epoch_selection",
                {"variant": variant, "outer_fold": fold, "rule": "ceil_median_4"},
                inner_ids,
                protocol_lock_sha256,
            )
            stages.append(select)
            selections[(variant, fold)] = str(select["scientific_identity_sha256"])

        for fold in FOLDS:
            fold_contract = contracts.loc[
                contracts["outer_fold"].astype(int).eq(fold)
                & contracts["inner_basket"].astype(int).eq(1)
            ].iloc[0]
            recomputed_outer = recomputed_contracts[(fold, 1)]
            for seed_index, effective_seed in zip(SEEDS, SEED_VALUES):
                pretrain = (
                    d0_pretrain[seed_index]
                    if variant == "D1"
                    else descriptor_pretrain[seed_index]
                )
                d1_receipt = (
                    d0_checkpoint_receipts[seed_index]["receipt_sha256"]
                    if variant == "D1"
                    else None
                )
                item = stage(
                    "d123_pampa_outer_fit_prediction",
                    {
                        "variant": variant,
                        "outer_fold": fold,
                        "seed_index": seed_index,
                        "effective_rng_seed": effective_seed,
                        "outer_test_ids_sha256": str(
                            recomputed_outer["outer_test_ids_sha256"]
                        ),
                        "n_outer_train_ids": int(recomputed_outer["n_outer_train_ids"]),
                        "outer_train_ids_sha256": str(
                            recomputed_outer["outer_train_ids_sha256"]
                        ),
                        "d1_checkpoint_receipt_sha256": d1_receipt,
                    },
                    [pretrain, selections[(variant, fold)]],
                    protocol_lock_sha256,
                )
                stages.append(item)
                outer_predictions[variant].append(
                    str(item["scientific_identity_sha256"])
                )
        label_gate = stage(
            "d123_metric_label_release_gate",
            {
                "variant": variant,
                "label_source_sha256": scientific_lock["metric_label_source"]["sha256"],
                "required_prediction_stages": 90,
                "release_condition": "all_dependencies_completed_accepted_and_hash_valid",
            },
            outer_predictions[variant],
            protocol_lock_sha256,
        )
        stages.append(label_gate)
        stages.append(
            stage(
                "d123_sealed_oof_metrics",
                {
                    "variant": variant,
                    "label_gate_identity": label_gate["scientific_identity_sha256"],
                    "n_prediction_stages": 90,
                },
                [str(label_gate["scientific_identity_sha256"])],
                protocol_lock_sha256,
            )
        )

    plan: dict[str, object] = {
        "schema_version": "scaffoldseal-d123-plan-v1",
        "status": "PREFIT_CANDIDATE_NOT_AUTHORIZED",
        "authorization": {
            "accepted": False,
            "execution_authorized": False,
            "gpu_training_allowed": False,
        },
        "scientific_lock": scientific_lock,
        "protocol_lock_sha256": protocol_lock_sha256,
        "locked_files": locked_files,
        "counts": {
            "descriptor_pretraining_fits": 5,
            "inner_fits": 216,
            "stopping_selections": 54,
            "outer_fits_predictions": 270,
            "metric_label_release_gates": 3,
            "sealed_metric_stages": 3,
            "scientific_fits": 491,
            "stages_total": 551,
        },
        "stages": stages,
    }
    if len(stages) != 551 or len({s["scientific_identity_sha256"] for s in stages}) != 551:
        raise RuntimeError("D123 stage count or identity uniqueness drifted")
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("Refusing to overwrite a D123 candidate plan")
    payload = build(args.project_root)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
