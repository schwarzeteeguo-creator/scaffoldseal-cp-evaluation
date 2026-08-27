"""Compute frozen D3 empirical interval coverage from reconstructed residuals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from d3_checkpoint_only_probe import accepted_attempt, sha256


LEVELS = (0.50, 0.80, 0.90)


def finite_sample_quantile(values: np.ndarray, level: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if len(ordered) == 0 or not np.isfinite(ordered).all():
        raise ValueError("Residual pool is empty or non-finite")
    index = min(len(ordered), math.ceil((len(ordered) + 1) * level)) - 1
    return float(ordered[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--reconstruction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.runtime_root.resolve()
    source = args.reconstruction_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((root / "d123_plan_candidate.json").read_text(encoding="utf-8"))
    if plan["plan_sha256"] != "6b73097ee83c000a48ec4dfc646ddbcad3a561d3276c33e3661d3bf8b9d8f5c5":
        raise RuntimeError("Frozen plan binding changed")
    reconstruction = json.loads((source / "reconstruction_summary.json").read_text(encoding="utf-8"))
    residual_path = source / "inner_validation_residuals.csv"
    if reconstruction["status"] != "PASS" or reconstruction["training_performed"] is not False:
        raise RuntimeError("Reconstruction did not pass the zero-training gate")
    if reconstruction["residuals_sha256"] != sha256(residual_path):
        raise RuntimeError("Reconstructed residual file hash changed")
    residuals = pd.read_csv(residual_path)

    sys.path.insert(0, str(root / "src"))
    from d123_sealed_outputs import read_sealed_predictions

    stages = [
        stage for stage in plan["stages"]
        if stage["kind"] == "d123_pampa_outer_fit_prediction" and stage["key"]["variant"] == "D3"
    ]
    if len(stages) != 90:
        raise RuntimeError("Expected 90 accepted D3 outer prediction stages")
    prediction_tables = []
    for stage in stages:
        attempt = accepted_attempt(
            root / "artifacts/d123_v1/d123_pampa_outer_fit_prediction" / stage["scientific_identity_sha256"]
        )
        prediction_tables.append(read_sealed_predictions(attempt / "predictions.lossless.json", stage))
    predictions = pd.concat(prediction_tables, ignore_index=True)
    if len(predictions) != 34475 or predictions.duplicated(["curated_id", "outer_fold", "seed_index"]).any():
        raise RuntimeError("D3 outer prediction slot coverage changed")

    label_record = plan["scientific_lock"]["metric_label_source"]
    label_path = root / label_record["relative_path"]
    if sha256(label_path) != label_record["sha256"]:
        raise RuntimeError("Released metric-label source hash changed")
    labels = pd.read_csv(label_path)
    labels["curated_id"] = labels["curated_id"].astype(str)
    evaluated = predictions.merge(
        labels[["curated_id", "source", "sealed_block_id", "permeability"]],
        on="curated_id", how="left", validate="many_to_one",
    )
    if evaluated["permeability"].isna().any():
        raise RuntimeError("A frozen D3 prediction lacks its released label")

    quantile_rows = []
    for fold in range(1, 19):
        pool = residuals.loc[residuals["outer_fold"].eq(fold), "absolute_residual_log10_papp"].to_numpy()
        for level in LEVELS:
            quantile_rows.append({
                "outer_fold": fold, "nominal_coverage": level, "n_calibration": len(pool),
                "half_width_log10_papp": finite_sample_quantile(pool, level),
            })
    quantiles = pd.DataFrame(quantile_rows)
    long = evaluated.assign(key=1).merge(pd.DataFrame({"nominal_coverage": LEVELS, "key": 1}), on="key").drop(columns="key")
    long = long.merge(quantiles, on=["outer_fold", "nominal_coverage"], validate="many_to_one")
    long["absolute_error"] = np.abs(long["prediction_log10_papp"] - long["permeability"])
    long["covered"] = long["absolute_error"] <= long["half_width_log10_papp"]
    long["interval_width_log10_papp"] = 2.0 * long["half_width_log10_papp"]

    overall = []
    for level, group in long.groupby("nominal_coverage", sort=True):
        coverage = float(group["covered"].mean())
        overall.append({
            "nominal_coverage": float(level), "n_prediction_rows": len(group), "coverage": coverage,
            "calibration_error": coverage - float(level),
            "mean_interval_width_log10_papp": float(group["interval_width_log10_papp"].mean()),
        })
    per_fold = (
        long.groupby(["nominal_coverage", "outer_fold"], sort=True)
        .agg(coverage=("covered", "mean"), n=("covered", "size"), mean_interval_width_log10_papp=("interval_width_log10_papp", "mean"))
        .reset_index()
    )
    coverage_90 = next(item["coverage"] for item in overall if item["nominal_coverage"] == 0.90)
    summary = {
        "schema_version": "scaffoldseal-d3-empirical-interval-coverage-v1", "status": "PASS",
        "frozen_plan_sha256": plan["plan_sha256"], "training_performed": False,
        "reconstruction_summary_sha256": sha256(source / "reconstruction_summary.json"),
        "residuals_sha256": sha256(residual_path), "n_outer_prediction_slots": len(predictions),
        "quantile_definition": "k=min(n,ceil((n+1)*level)); one-indexed upper order statistic",
        "overall": overall, "per_fold": per_fold.to_dict(orient="records"),
        "h2_condition4_90pct_coverage": coverage_90,
        "h2_condition4_required_range_inclusive": [0.85, 0.95],
        "h2_condition4_pass": bool(0.85 <= coverage_90 <= 0.95),
    }
    output.mkdir(parents=True)
    quantiles.to_csv(output / "fold_empirical_quantiles.csv", index=False, lineterminator="\n")
    long.to_csv(output / "interval_rows.csv", index=False, lineterminator="\n")
    (output / "coverage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
