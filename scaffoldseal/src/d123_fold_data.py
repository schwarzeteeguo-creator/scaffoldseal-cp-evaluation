"""Fold-scoped D123 frames that never materialize outer-test targets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import pandas as pd


def canonical_id_hash(values) -> str:
    ids = sorted({str(value) for value in values})
    return hashlib.sha256("".join(f"{value}\n" for value in ids).encode()).hexdigest()


def load_fold_frames(
    plan: Mapping[str, object],
    stage: Mapping[str, object],
    project_root: Path,
) -> dict[str, pd.DataFrame]:
    root = project_root.resolve()
    fold = int(stage["key"]["outer_fold"])
    if fold not in range(1, 19):
        raise ValueError("D123 outer fold is outside 1..18")
    feature_record = plan["scientific_lock"]["label_free_feature_source"]
    target_record = plan["scientific_lock"]["fold_scoped_training_label_sources"][
        str(fold)
    ]
    features = pd.read_csv(root / str(feature_record["relative_path"]))
    targets = pd.read_csv(root / str(target_record["relative_path"]))
    if list(features.columns) != [
        "curated_id",
        "SMILES",
        "sealed_block_id",
        "outer_fold",
    ]:
        raise RuntimeError("D123 label-free feature schema drifted")
    if list(targets.columns) != ["curated_id", "normalized_pampa"]:
        raise RuntimeError("D123 fold-target schema drifted")
    features["curated_id"] = features["curated_id"].astype(str)
    targets["curated_id"] = targets["curated_id"].astype(str)
    if (
        len(features) != 6895
        or features["curated_id"].duplicated().any()
        or targets["curated_id"].duplicated().any()
    ):
        raise RuntimeError("D123 fold frame population drifted")
    outer_test = features.loc[features["outer_fold"].astype(int).eq(fold)].copy()
    outer_train_features = features.loc[
        ~features["outer_fold"].astype(int).eq(fold)
    ].copy()
    if set(targets["curated_id"]) != set(outer_train_features["curated_id"]):
        raise RuntimeError("D123 training targets differ from outer-train IDs")
    if set(targets["curated_id"]) & set(outer_test["curated_id"]):
        raise RuntimeError("D123 outer-test target was materialized")
    outer_train = outer_train_features.merge(
        targets,
        on="curated_id",
        how="left",
        validate="one_to_one",
    )
    if outer_train["normalized_pampa"].isna().any():
        raise RuntimeError("D123 outer-train target is missing")
    result = {
        "outer_train": outer_train,
        "outer_test_label_free": outer_test,
    }
    if stage["kind"] == "d123_pampa_inner_fit":
        basket = int(stage["key"]["inner_basket"])
        baskets = pd.read_csv(root / "artifacts/v2_r0/inner_basket_manifest.csv")
        blocks = set(
            baskets.loc[
                baskets["outer_fold"].astype(int).eq(fold)
                & baskets["inner_basket"].astype(int).eq(basket),
                "sealed_block_id",
            ].astype(str)
        )
        validation = outer_train.loc[
            outer_train["sealed_block_id"].astype(str).isin(blocks)
        ].copy()
        fit = outer_train.loc[
            ~outer_train["sealed_block_id"].astype(str).isin(blocks)
        ].copy()
        if (
            canonical_id_hash(fit["curated_id"])
            != str(stage["key"]["fit_ids_sha256"])
            or canonical_id_hash(validation["curated_id"])
            != str(stage["key"]["validation_ids_sha256"])
            or canonical_id_hash(outer_test["curated_id"])
            != str(stage["key"]["outer_test_ids_sha256"])
        ):
            raise RuntimeError("D123 inner frame IDs differ from its scientific identity")
        result.update({"fit": fit, "validation": validation})
    elif stage["kind"] == "d123_pampa_outer_fit_prediction":
        if (
            len(outer_train) != int(stage["key"]["n_outer_train_ids"])
            or canonical_id_hash(outer_train["curated_id"])
            != str(stage["key"]["outer_train_ids_sha256"])
            or canonical_id_hash(outer_test["curated_id"])
            != str(stage["key"]["outer_test_ids_sha256"])
        ):
            raise RuntimeError("D123 outer frame IDs differ from its scientific identity")
    else:
        raise ValueError("Fold frames are available only to D123 fit stages")
    return result
