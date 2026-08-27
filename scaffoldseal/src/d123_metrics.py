"""Gate-after-sealing D123 metric computation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from d123_runner_governance import (
    accepted_output_if_valid,
    canonical_sha256,
    load_metric_labels_after_accepted_gate,
)
from d123_sealed_outputs import read_sealed_predictions


def compute_frozen_metrics(
    predictions: pd.DataFrame, labels: pd.DataFrame
) -> dict[str, object]:
    required_predictions = {
        "curated_id",
        "outer_fold",
        "seed_index",
        "prediction_log10_papp",
    }
    required_labels = {"curated_id", "source", "sealed_block_id", "permeability"}
    if not required_predictions.issubset(predictions.columns):
        raise ValueError("D123 predictions lack frozen metric columns")
    if not required_labels.issubset(labels.columns):
        raise ValueError("D123 labels lack frozen metric columns")
    if predictions.duplicated(["curated_id", "outer_fold", "seed_index"]).any():
        raise ValueError("D123 metric slots are duplicated")
    if labels["curated_id"].astype(str).duplicated().any():
        raise ValueError("D123 metric labels are duplicated")
    evaluated = predictions.merge(
        labels[list(required_labels)],
        on="curated_id",
        how="left",
        validate="many_to_one",
    )
    if evaluated["permeability"].isna().any():
        raise RuntimeError("A sealed D123 prediction lacks its frozen label")
    evaluated["absolute_error"] = np.abs(
        evaluated["prediction_log10_papp"].astype(float)
        - evaluated["permeability"].astype(float)
    )
    if not np.isfinite(evaluated["absolute_error"]).all():
        raise RuntimeError("D123 metric error is non-finite")
    per_source = (
        evaluated.groupby(["seed_index", "source"], sort=True)["absolute_error"]
        .mean()
        .rename("mae")
        .reset_index()
    )
    per_block = (
        evaluated.groupby(
            ["seed_index", "outer_fold", "sealed_block_id"], sort=True
        )["absolute_error"]
        .agg([("mae", "mean"), ("n", "size")])
        .reset_index()
    )
    source_macro = per_source.groupby("seed_index", sort=True)["mae"].mean()
    row_micro = evaluated.groupby("seed_index", sort=True)["absolute_error"].mean()
    return {
        "schema_version": "scaffoldseal-d123-metrics-v1",
        "labels_loaded_after_prediction_sealing": True,
        "n_prediction_rows": len(evaluated),
        "source_macro_mae_by_seed": {
            str(int(seed)): float(value) for seed, value in source_macro.items()
        },
        "source_macro_mae_mean_across_seeds": float(source_macro.mean()),
        "row_micro_mae_by_seed": {
            str(int(seed)): float(value) for seed, value in row_micro.items()
        },
        "row_micro_mae_mean_across_seeds": float(row_micro.mean()),
        "per_source": per_source.to_dict(orient="records"),
        "per_block": per_block.to_dict(orient="records"),
    }


def write_variant_metrics(
    plan: Mapping[str, object],
    metric_stage: Mapping[str, object],
    variant: str,
    ledger,
    project_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Validate all sealed bytes, then open labels through the accepted gate."""

    prediction_stages = [
        stage
        for stage in plan["stages"]
        if stage["kind"] == "d123_pampa_outer_fit_prediction"
        and stage["key"].get("variant") == variant
    ]
    if len(prediction_stages) != 90:
        raise RuntimeError("D123 metrics require exactly 90 prediction stages")
    tables = []
    for stage in prediction_stages:
        output = accepted_output_if_valid(stage, ledger, project_root)
        if output is None:
            raise RuntimeError("D123 metric prediction is not accepted")
        tables.append(
            read_sealed_predictions(
                output / "predictions.lossless.json", stage
            )
        )
    predictions = pd.concat(tables, ignore_index=True)
    outer = pd.read_csv(project_root / "artifacts/v2_r0/outer_record_assignments.csv")
    expected_slots = {
        (str(row.curated_id), int(row.outer_fold), seed)
        for row in outer.itertuples(index=False)
        for seed in range(5)
    }
    observed_slots = set(
        predictions[["curated_id", "outer_fold", "seed_index"]].itertuples(
            index=False, name=None
        )
    )
    if len(predictions) != 34475 or observed_slots != expected_slots:
        raise RuntimeError("D123 sealed OOF slots differ from the frozen manifest")

    # This is the first label read: the helper requires an accepted, hash-valid
    # release gate and revalidates the frozen label source bytes.
    labels = load_metric_labels_after_accepted_gate(
        plan, variant, ledger, project_root
    )
    metrics = compute_frozen_metrics(predictions, labels)
    metrics.update(
        {
            "variant": variant,
            "scientific_identity_sha256": str(
                metric_stage["scientific_identity_sha256"]
            ),
            "stage_spec_sha256": str(metric_stage["stage_spec_sha256"]),
        }
    )
    metrics["metrics_sha256"] = canonical_sha256(metrics)
    target = output_root.resolve() / "metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            str(target),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError("Refusing to overwrite immutable D123 metrics") from error
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return metrics
