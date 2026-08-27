"""Zero-training stratification of the frozen D3 empirical intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(groups, sort=True)
        .agg(
            n_prediction_slots=("covered", "size"),
            coverage=("covered", "mean"),
            mean_interval_width_log10_papp=("interval_width_log10_papp", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.interval_rows.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, low_memory=False)
    required = {
        "curated_id",
        "outer_fold",
        "seed_index",
        "source",
        "sealed_block_id",
        "nominal_coverage",
        "covered",
        "interval_width_log10_papp",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame["covered"] = frame["covered"].astype(bool)

    outputs = {
        "d3_coverage_by_seed.csv": aggregate(frame, ["nominal_coverage", "seed_index"]),
        "d3_coverage_by_outer_fold.csv": aggregate(frame, ["nominal_coverage", "outer_fold", "sealed_block_id"]),
        "d3_coverage_by_source.csv": aggregate(frame, ["nominal_coverage", "source"]),
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / name, index=False, lineterminator="\n")

    rows = []
    for nominal, part in frame.groupby("nominal_coverage", sort=True):
        seed_table = outputs["d3_coverage_by_seed.csv"].loc[
            outputs["d3_coverage_by_seed.csv"]["nominal_coverage"].eq(nominal)
        ]
        fold_table = outputs["d3_coverage_by_outer_fold.csv"].loc[
            outputs["d3_coverage_by_outer_fold.csv"]["nominal_coverage"].eq(nominal)
        ]
        source_table = outputs["d3_coverage_by_source.csv"].loc[
            outputs["d3_coverage_by_source.csv"]["nominal_coverage"].eq(nominal)
        ]
        rows.append(
            {
                "nominal_coverage": float(nominal),
                "slot_pooled_coverage": float(part["covered"].mean()),
                "minimum_seed_coverage": float(seed_table["coverage"].min()),
                "maximum_seed_coverage": float(seed_table["coverage"].max()),
                "equal_source_macro_coverage": float(source_table["coverage"].mean()),
                "equal_block_macro_coverage": float(fold_table["coverage"].mean()),
                "minimum_block_coverage": float(fold_table["coverage"].min()),
                "maximum_block_coverage": float(fold_table["coverage"].max()),
            }
        )
    aggregation = pd.DataFrame(rows)
    aggregation.to_csv(output_dir / "d3_coverage_aggregation_sensitivity.csv", index=False, lineterminator="\n")

    summary = {
        "schema_version": "scaffoldseal-d3-coverage-stratification-v1",
        "input_interval_rows_sha256": sha256_file(source),
        "n_prediction_slots": int(len(frame) // frame["nominal_coverage"].nunique()),
        "n_unique_records": int(frame["curated_id"].nunique()),
        "n_seeds": int(frame["seed_index"].nunique()),
        "n_outer_folds": int(frame["outer_fold"].nunique()),
        "n_sources": int(frame["source"].nunique()),
        "interpretation_boundary": (
            "Slot-pooled coverage averages five algorithmic seed realizations for each record. "
            "Those slots are not independent experimental observations, so no binomial confidence "
            "interval is attached. Equal-source and equal-block summaries are descriptive estimands."
        ),
        "aggregation_rows": aggregation.to_dict(orient="records"),
    }
    (output_dir / "d3_coverage_stratification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
