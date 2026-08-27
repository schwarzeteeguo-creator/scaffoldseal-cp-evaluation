"""Build deterministic fold-scoped D0 target views without model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCHEMA_VERSION = "scaffoldseal-d0-fold-scoped-targets-v1"
OUTER_FOLDS = tuple(range(1, 19))


def stream_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonical_id_hash(ids: Iterable[str]) -> str:
    values = sorted({str(value) for value in ids})
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def build_fold_scoped_governance(
    *,
    labels_path: Path,
    outer_assignments_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Create pre-fit deterministic derivatives and their immutable manifest.

    This builder is a governance/data-preparation operation. It constructs no
    model, optimizer, prediction, metric, or CUDA object. Runtime code must use
    these derivatives and must not call this builder.
    """

    labels_path = labels_path.resolve()
    outer_assignments_path = outer_assignments_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise RuntimeError(f"Governance output already exists: {output_root}")
    output_root.mkdir(parents=True)
    records = pd.read_csv(labels_path)
    outer = pd.read_csv(outer_assignments_path)
    required = {"curated_id", "canonical_smiles", "sealed_block_id", "permeability"}
    if not required.issubset(records.columns):
        raise RuntimeError("Frozen analysis label source schema drifted")
    if records["curated_id"].astype(str).duplicated().any() or len(records) != 6895:
        raise RuntimeError("Frozen analysis label source ID/count drifted")
    if set(records["curated_id"].astype(str)) != set(outer["curated_id"].astype(str)):
        raise RuntimeError("Outer assignments do not cover the frozen label source")
    endpoint = pd.to_numeric(records["permeability"], errors="coerce")
    if not np.isfinite(endpoint).all():
        raise RuntimeError("Frozen endpoint contains a non-finite value")
    joined = records.merge(
        outer[["curated_id", "outer_fold"]],
        on="curated_id",
        how="inner",
        validate="one_to_one",
    )
    joined["curated_id"] = joined["curated_id"].astype(str)
    joined["outer_fold"] = joined["outer_fold"].astype(int)
    if sorted(joined["outer_fold"].unique()) != list(OUTER_FOLDS):
        raise RuntimeError("Outer fold coverage drifted")

    features = joined[
        ["curated_id", "canonical_smiles", "sealed_block_id", "outer_fold"]
    ].rename(columns={"canonical_smiles": "SMILES"})
    features = features.sort_values("curated_id", kind="stable").reset_index(drop=True)
    if str(features["SMILES"].astype(object).dtype) != "object":
        raise RuntimeError("Label-free SMILES view is not object-string compatible")
    feature_path = output_root / "label_free_features.csv"
    features.to_csv(feature_path, index=False, lineterminator="\n")

    target_records = []
    normalized = (pd.to_numeric(joined["permeability"]) + 6.0) / 2.0
    joined = joined.assign(normalized_pampa=normalized)
    for fold in OUTER_FOLDS:
        train = joined.loc[
            joined["outer_fold"].ne(fold), ["curated_id", "normalized_pampa"]
        ].sort_values("curated_id", kind="stable")
        heldout_ids = joined.loc[joined["outer_fold"].eq(fold), "curated_id"].astype(str)
        if set(train["curated_id"].astype(str)) & set(heldout_ids):
            raise RuntimeError(f"Fold {fold} target derivative contains a held-out ID")
        target_path = output_root / f"outer_{fold:02d}_training_targets.csv"
        train.to_csv(
            target_path,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        target_records.append(
            {
                "outer_fold": fold,
                "relative_path": target_path.name,
                "size_bytes": target_path.stat().st_size,
                "sha256": stream_sha256(target_path),
                "n_training_targets": len(train),
                "training_ids_sha256": canonical_id_hash(train["curated_id"].astype(str)),
                "n_heldout_ids": len(heldout_ids),
                "heldout_ids_sha256": canonical_id_hash(heldout_ids),
                "heldout_target_values_materialized": False,
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PREFIT_DETERMINISTIC_DERIVATIVE_CANDIDATE",
        "scientific_geometry_changed": False,
        "source_labels": {
            "relative_path": "artifacts/v2_r0/analysis_all_labels.csv",
            "size_bytes": labels_path.stat().st_size,
            "sha256": stream_sha256(labels_path),
            "access_role": "METRIC_ONLY_AFTER_ALL_90_PREDICTIONS_ACCEPTED",
        },
        "outer_assignments": {
            "relative_path": "artifacts/v2_r0/outer_record_assignments.csv",
            "size_bytes": outer_assignments_path.stat().st_size,
            "sha256": stream_sha256(outer_assignments_path),
        },
        "label_free_features": {
            "relative_path": feature_path.name,
            "size_bytes": feature_path.stat().st_size,
            "sha256": stream_sha256(feature_path),
            "columns": ["curated_id", "SMILES", "sealed_block_id", "outer_fold"],
            "feature_dtype": "object-string SMILES",
            "contains_endpoint": False,
        },
        "fold_training_targets": target_records,
    }
    manifest["derivatives_sha256"] = canonical_json_sha256(
        {
            "label_free_features": manifest["label_free_features"],
            "fold_training_targets": target_records,
        }
    )
    _write_json(output_root / "fold_scoped_target_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--outer-assignments", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_fold_scoped_governance(
        labels_path=args.labels,
        outer_assignments_path=args.outer_assignments,
        output_root=args.output_root,
    )
    print(json.dumps({"status": manifest["status"], "sha256": manifest["derivatives_sha256"]}))


if __name__ == "__main__":
    main()
