"""Build frozen, label-blind H1 molecule-random five-fold execution manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = tuple(range(1, 6))


def canonical_id_hash(values) -> str:
    ids = sorted({str(value) for value in values})
    return hashlib.sha256("".join(f"{value}\n" for value in ids).encode()).hexdigest()


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(project_root: Path, output_root: Path) -> dict[str, object]:
    comparison_path = project_root / "artifacts/v2_r0/comparison_fold_manifest.csv"
    labels_path = project_root / "artifacts/v2_r0/analysis_all_labels.csv"
    comparison = pd.read_csv(comparison_path)
    labels = pd.read_csv(labels_path)
    if output_root.exists():
        raise RuntimeError(f"Output already exists: {output_root}")
    required_comparison = {
        "curated_id",
        "molecule_id",
        "source",
        "analogue_component_id",
        "molecule_id_fold",
    }
    required_labels = {
        "curated_id",
        "canonical_smiles",
        "sealed_block_id",
        "permeability",
    }
    if not required_comparison.issubset(comparison) or not required_labels.issubset(labels):
        raise RuntimeError("Frozen source schema drifted")
    if len(comparison) != 6895 or len(labels) != 6895:
        raise RuntimeError("H1 requires the exact 6,895-record population")
    if comparison["curated_id"].duplicated().any() or labels["curated_id"].duplicated().any():
        raise RuntimeError("Frozen H1 inputs contain duplicate curated IDs")
    if set(comparison["curated_id"]) != set(labels["curated_id"]):
        raise RuntimeError("Comparison folds and labels cover different populations")
    if sorted(comparison["molecule_id_fold"].astype(int).unique()) != list(FOLDS):
        raise RuntimeError("Frozen molecule-random fold coverage drifted")
    if comparison.groupby("molecule_id")["molecule_id_fold"].nunique().max() != 1:
        raise RuntimeError("A molecule crosses frozen molecule-random folds")

    output_root.mkdir(parents=True)
    joined = comparison.merge(
        labels[["curated_id", "canonical_smiles", "sealed_block_id", "permeability"]],
        on="curated_id",
        validate="one_to_one",
    )
    outer = joined[
        [
            "curated_id",
            "molecule_id",
            "source",
            "analogue_component_id",
            "sealed_block_id",
            "molecule_id_fold",
        ]
    ].rename(columns={"molecule_id_fold": "outer_fold"})
    outer["outer_test_block"] = outer["outer_fold"].map(
        lambda fold: f"MOLECULE_RANDOM_FOLD_{int(fold)}"
    )
    outer = outer.sort_values("curated_id", kind="stable").reset_index(drop=True)
    outer.to_csv(output_root / "outer_record_assignments.csv", index=False, lineterminator="\n")

    inner_rows = []
    contract_rows = []
    all_ids = set(outer["curated_id"].astype(str))
    for outer_fold in FOLDS:
        remaining_folds = [fold for fold in FOLDS if fold != outer_fold]
        basket_for_fold = {fold: index + 1 for index, fold in enumerate(remaining_folds)}
        train_outer = outer.loc[outer["outer_fold"] != outer_fold]
        test_ids = set(outer.loc[outer["outer_fold"] == outer_fold, "curated_id"].astype(str))
        for row in train_outer.itertuples(index=False):
            inner_rows.append(
                {
                    "outer_fold": outer_fold,
                    "curated_id": row.curated_id,
                    "inner_basket": basket_for_fold[int(row.outer_fold)],
                    "source_molecule_fold": int(row.outer_fold),
                }
            )
        for basket in range(1, 5):
            validation_ids = {
                str(row["curated_id"])
                for row in inner_rows
                if row["outer_fold"] == outer_fold and row["inner_basket"] == basket
            }
            fit_ids = all_ids - test_ids - validation_ids
            if fit_ids & validation_ids or fit_ids & test_ids or validation_ids & test_ids:
                raise RuntimeError("H1 train/validation/test identities overlap")
            if fit_ids | validation_ids | test_ids != all_ids:
                raise RuntimeError("H1 contract does not partition the frozen population")
            contract_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_basket": basket,
                    "n_fit_ids": len(fit_ids),
                    "fit_ids_sha256": canonical_id_hash(fit_ids),
                    "n_inner_validation_ids": len(validation_ids),
                    "inner_validation_ids_sha256": canonical_id_hash(validation_ids),
                    "n_outer_test_ids": len(test_ids),
                    "outer_test_ids_sha256": canonical_id_hash(test_ids),
                }
            )
    inner = pd.DataFrame(inner_rows).sort_values(
        ["outer_fold", "inner_basket", "curated_id"], kind="stable"
    )
    contracts = pd.DataFrame(contract_rows)
    inner.to_csv(output_root / "inner_id_basket_manifest.csv", index=False, lineterminator="\n")
    contracts.to_csv(
        output_root / "pre_fit_contract_manifest.csv", index=False, lineterminator="\n"
    )

    target_root = output_root / "fold_scoped_targets"
    target_root.mkdir()
    features = joined[
        ["curated_id", "canonical_smiles", "sealed_block_id", "molecule_id_fold"]
    ].rename(columns={"canonical_smiles": "SMILES", "molecule_id_fold": "outer_fold"})
    features = features.sort_values("curated_id", kind="stable")
    features.to_csv(target_root / "label_free_features.csv", index=False, lineterminator="\n")
    normalized = pd.to_numeric(joined["permeability"], errors="coerce")
    if not np.isfinite(normalized).all():
        raise RuntimeError("Frozen H1 endpoint contains non-finite values")
    joined["normalized_pampa"] = (normalized + 6.0) / 2.0
    target_records = []
    for fold in FOLDS:
        train = joined.loc[
            joined["molecule_id_fold"].astype(int) != fold,
            ["curated_id", "normalized_pampa"],
        ].sort_values("curated_id", kind="stable")
        heldout = outer.loc[outer["outer_fold"] == fold, "curated_id"].astype(str)
        path = target_root / f"outer_{fold:02d}_training_targets.csv"
        train.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        target_records.append(
            {
                "outer_fold": fold,
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": stream_sha256(path),
                "n_training_targets": len(train),
                "training_ids_sha256": canonical_id_hash(train["curated_id"]),
                "n_heldout_ids": len(heldout),
                "heldout_ids_sha256": canonical_id_hash(heldout),
                "heldout_target_values_materialized": False,
            }
        )
    governance = {
        "schema_version": "scaffoldseal-h1-fold-scoped-targets-v1",
        "status": "PREFIT_DETERMINISTIC_DERIVATIVE_CANDIDATE",
        "source_labels": {
            "relative_path": "artifacts/v2_r0/analysis_all_labels.csv",
            "size_bytes": labels_path.stat().st_size,
            "sha256": stream_sha256(labels_path),
            "access_role": "METRIC_ONLY_AFTER_ALL_25_PREDICTIONS_ACCEPTED",
        },
        "comparison_source": {
            "relative_path": "artifacts/v2_r0/comparison_fold_manifest.csv",
            "sha256": stream_sha256(comparison_path),
        },
        "outer_assignments": {
            "relative_path": "artifacts/h1_random_cv_r0/outer_record_assignments.csv",
            "sha256": stream_sha256(output_root / "outer_record_assignments.csv"),
        },
        "label_free_features": {
            "relative_path": "label_free_features.csv",
            "size_bytes": (target_root / "label_free_features.csv").stat().st_size,
            "sha256": stream_sha256(target_root / "label_free_features.csv"),
            "columns": ["curated_id", "SMILES", "sealed_block_id", "outer_fold"],
            "contains_endpoint": False,
        },
        "fold_training_targets": target_records,
    }
    governance["derivatives_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "label_free_features": governance["label_free_features"],
                "fold_training_targets": target_records,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    (target_root / "fold_scoped_target_manifest.json").write_text(
        json.dumps(governance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashes = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes.append(f"{stream_sha256(path)}  {path.relative_to(output_root).as_posix()}\n")
    (output_root / "SHA256SUMS").write_text("".join(hashes), encoding="utf-8")
    return {
        "n_records": len(outer),
        "n_outer_folds": len(FOLDS),
        "n_inner_contracts": len(contracts),
        "molecule_cross_fold_violations": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve(), args.output.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
