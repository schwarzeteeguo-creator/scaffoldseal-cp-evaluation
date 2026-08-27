"""Assemble the preregistered nested-selected classical comparator.

This module performs no fitting.  For each outer fold it chooses among the
already inner-selected Ridge, Random Forest and XGBoost procedures using only
their concatenated inner-validation metrics, then assembles the corresponding
outer-fold OOF predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "curated_id"
FOLDS = tuple(range(1, 19))
SEEDS = tuple(range(5))
FAMILY_COMPUTE_RANK = {"ridge": 1, "random_forest": 2, "xgboost": 3}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_family_rows(family: str, path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "outer_fold",
        "config_id",
        "source_macro_mae",
        "row_micro_mae",
        "selected",
    }
    if not required.issubset(table.columns):
        raise ValueError(f"{family} selection table lacks required columns")
    selected = table.loc[table["selected"].astype(bool), list(required - {"selected"})].copy()
    counts = selected.groupby("outer_fold").size()
    if tuple(sorted(counts.index.astype(int))) != FOLDS or not (counts == 1).all():
        raise ValueError(f"{family} must have exactly one inner-selected configuration per fold")
    selected["family"] = family
    selected["family_compute_rank"] = FAMILY_COMPUTE_RANK[family]
    return selected


def choose_families(selection_paths: dict[str, Path]) -> pd.DataFrame:
    if set(selection_paths) != set(FAMILY_COMPUTE_RANK):
        raise ValueError("Exactly Ridge, Random Forest and XGBoost are required")
    candidates = pd.concat(
        [selected_family_rows(family, path) for family, path in selection_paths.items()],
        ignore_index=True,
    )
    candidates["selected_family"] = False
    for fold, indices in candidates.groupby("outer_fold", sort=True).groups.items():
        ranked = candidates.loc[indices].sort_values(
            [
                "source_macro_mae",
                "row_micro_mae",
                "family_compute_rank",
                "family",
                "config_id",
            ],
            kind="stable",
        )
        candidates.loc[ranked.index[0], "selected_family"] = True
    if int(candidates["selected_family"].sum()) != len(FOLDS):
        raise RuntimeError("Family selection did not yield exactly one winner per outer fold")
    return candidates.sort_values(["outer_fold", "family"], kind="stable").reset_index(drop=True)


def _prediction_table(family: str, path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        ID_COLUMN,
        "molecule_id",
        "source",
        "analogue_component_id",
        "sealed_block_id",
        "outer_fold",
        "model",
        "config_id",
        "observed",
        "prediction",
    }
    if not required.issubset(table.columns):
        raise ValueError(f"{family} OOF table lacks required columns")
    if family == "ridge":
        table = table.loc[table["model"] == "ridge"].copy()
        table = pd.concat([table.assign(seed=seed) for seed in SEEDS], ignore_index=True)
    else:
        if "seed" not in table.columns:
            raise ValueError(f"{family} OOF table lacks seed")
        table = table.loc[table["model"] == family].copy()
    return table


def assemble_oof(
    family_selection: pd.DataFrame,
    prediction_paths: dict[str, Path],
    frozen_assignments_path: Path,
) -> pd.DataFrame:
    winners = family_selection.loc[
        family_selection["selected_family"], ["outer_fold", "family", "config_id"]
    ]
    pieces: list[pd.DataFrame] = []
    for row in winners.itertuples(index=False):
        family_table = _prediction_table(row.family, prediction_paths[row.family])
        piece = family_table.loc[
            (family_table["outer_fold"].astype(int) == int(row.outer_fold))
            & (family_table["config_id"] == row.config_id)
        ].copy()
        if piece.empty:
            raise ValueError(
                f"No OOF predictions for fold {row.outer_fold}, {row.family}, {row.config_id}"
            )
        piece["selected_family"] = row.family
        pieces.append(piece)
    oof = pd.concat(pieces, ignore_index=True)
    frozen = pd.read_csv(frozen_assignments_path)
    metadata_columns = [
        ID_COLUMN,
        "molecule_id",
        "source",
        "analogue_component_id",
        "sealed_block_id",
        "outer_fold",
    ]
    if len(frozen) != 6895 or frozen[ID_COLUMN].duplicated().any():
        raise ValueError("Frozen assignments must contain exactly 6,895 unique records")
    if not set(metadata_columns).issubset(frozen.columns):
        raise ValueError("Frozen assignments lack required identity metadata")
    frozen = frozen.loc[:, metadata_columns].sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)
    expected_ids = set(frozen[ID_COLUMN])
    reference_seed_metadata: pd.DataFrame | None = None
    for seed, group in oof.groupby("seed", sort=True):
        ids = set(group[ID_COLUMN])
        if len(group) != 6895 or group[ID_COLUMN].duplicated().any() or ids != expected_ids:
            raise ValueError(f"Seed {seed} does not exactly cover the frozen 6,895-record population")
        observed_metadata = (
            group.loc[:, metadata_columns]
            .sort_values(ID_COLUMN, kind="stable")
            .reset_index(drop=True)
        )
        if not observed_metadata.equals(frozen):
            raise ValueError(f"Seed {seed} metadata/fold mapping differs from frozen assignments")
        if reference_seed_metadata is None:
            reference_seed_metadata = observed_metadata
        elif not observed_metadata.equals(reference_seed_metadata):
            raise ValueError("Identity metadata differs across seeds")
    if tuple(sorted(oof["seed"].unique().astype(int))) != SEEDS:
        raise ValueError("Nested comparator requires seeds 0-4")
    observed_counts = oof.groupby(ID_COLUMN)["observed"].nunique()
    if not (observed_counts == 1).all():
        raise ValueError("Observed values disagree across stochastic seeds")
    columns = [
        ID_COLUMN,
        "molecule_id",
        "source",
        "analogue_component_id",
        "sealed_block_id",
        "outer_fold",
        "selected_family",
        "config_id",
        "seed",
        "observed",
        "prediction",
    ]
    return oof.loc[:, columns].sort_values(["seed", ID_COLUMN], kind="stable").reset_index(drop=True)


def metric_outputs(oof: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    work = oof.assign(error=oof["prediction"] - oof["observed"])
    work["absolute_error"] = work["error"].abs()
    work["squared_error"] = work["error"] ** 2
    per_source = (
        work.groupby(["seed", "source"], sort=True)
        .agg(n=(ID_COLUMN, "size"), mae=("absolute_error", "mean"), mse=("squared_error", "mean"))
        .reset_index()
    )
    per_source["rmse"] = np.sqrt(per_source.pop("mse"))
    per_block = (
        work.groupby(["seed", "sealed_block_id", "outer_fold"], sort=True)
        .agg(n=(ID_COLUMN, "size"), mae=("absolute_error", "mean"), mse=("squared_error", "mean"))
        .reset_index()
    )
    per_block["rmse"] = np.sqrt(per_block.pop("mse"))
    by_seed: list[dict[str, object]] = []
    for seed, group in work.groupby("seed", sort=True):
        sources = per_source.loc[per_source["seed"] == seed]
        blocks = per_block.loc[per_block["seed"] == seed]
        by_seed.append(
            {
                "seed": int(seed),
                "n": int(len(group)),
                "source_macro_mae": float(sources["mae"].mean()),
                "source_macro_rmse": float(sources["rmse"].mean()),
                "row_micro_mae": float(group["absolute_error"].mean()),
                "row_micro_rmse": float(np.sqrt(group["squared_error"].mean())),
                "block_median_mae": float(blocks["mae"].median()),
            }
        )
    summary: dict[str, object] = {
        "status": "PROVISIONAL_PENDING_VERIFIER",
        "procedure": "per_outer_fold_inner_selected_classical_family",
        "metrics_by_seed": by_seed,
        "mean_across_seeds": {
            key: float(np.mean([row[key] for row in by_seed]))
            for key in (
                "source_macro_mae",
                "source_macro_rmse",
                "row_micro_mae",
                "row_micro_rmse",
                "block_median_mae",
            )
        },
    }
    return summary, per_source, per_block


def write_outputs(
    output_dir: Path,
    selection_paths: dict[str, Path],
    prediction_paths: dict[str, Path],
    frozen_assignments_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    selection = choose_families(selection_paths)
    oof = assemble_oof(selection, prediction_paths, frozen_assignments_path)
    summary, per_source, per_block = metric_outputs(oof)
    selection.to_csv(output_dir / "family_selection.csv", index=False, lineterminator="\n")
    oof.to_csv(output_dir / "oof_predictions_per_seed.csv", index=False, lineterminator="\n")
    per_source.to_csv(output_dir / "per_source_metrics.csv", index=False, lineterminator="\n")
    per_block.to_csv(output_dir / "per_block_metrics.csv", index=False, lineterminator="\n")
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    provenance = {
        "status": "PROVISIONAL_PENDING_VERIFIER",
        "fit_operations": 0,
        "selection_order": [
            "inner_source_macro_mae",
            "inner_row_micro_mae",
            "lower_family_compute_rank",
            "lexical_family",
            "lexical_config_id",
        ],
        "family_compute_rank": FAMILY_COMPUTE_RANK,
        "inputs": {
            str(path): _sha256(path)
            for path in [
                *selection_paths.values(),
                *prediction_paths.values(),
                frozen_assignments_path,
            ]
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    output_files = sorted(path for path in output_dir.iterdir() if path.name != "SHA256SUMS")
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in output_files),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--frozen-assignments",
        type=Path,
        default=Path("artifacts/v2_r0/outer_record_assignments.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root
    selection_paths = {
        "ridge": root / "r1a_classical" / "inner_selection.csv",
        "random_forest": root / "r1b1_random_forest" / "inner_selection.csv",
        "xgboost": root / "r1b2_xgboost" / "inner_selection.csv",
    }
    prediction_paths = {
        "ridge": root / "r1a_classical" / "oof_predictions.csv",
        "random_forest": root / "r1b1_random_forest" / "oof_predictions_per_seed.csv",
        "xgboost": root / "r1b2_xgboost" / "oof_predictions_per_seed.csv",
    }
    write_outputs(args.output, selection_paths, prediction_paths, args.frozen_assignments)


if __name__ == "__main__":
    main()
