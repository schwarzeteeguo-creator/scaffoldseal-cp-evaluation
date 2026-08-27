"""Descriptive, zero-training robustness analysis for the frozen H1 result."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
RUNTIME_ROOT = Path(
    os.environ.get("SCAFFOLDSEAL_RUNTIME_ROOT", str(PROJECT_ROOT / "scaffoldseal"))
).resolve()
SOURCE_DIR = HERE / "source_data"
FIGURE_DIR = HERE / "figures"
THRESHOLD = 0.10
EXPECTED = {
    "joint": 1.00455900469197,
    "random": 0.46721829369813345,
    "gap": 0.5373407109938364,
}

MM = 1 / 25.4
COLORS = {
    "random": "#4C78A8",
    "joint": "#D07A5F",
    "positive": "#4F8F75",
    "negative": "#B85C5C",
    "ink": "#27313A",
    "neutral": "#8B95A1",
    "grid": "#D9DEE3",
    "pale": "#F3F5F7",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_stage_manifest(directory: Path) -> None:
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = directory / record["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["size_bytes"]) or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Frozen artifact mismatch: {path}")


def input_paths() -> dict[str, Path]:
    return {
        "labels": RUNTIME_ROOT / "artifacts/v2_r0/analysis_all_labels.csv",
        "assignments": RUNTIME_ROOT / "artifacts/v2_r0/outer_record_assignments.csv",
        "blocks": RUNTIME_ROOT / "artifacts/source_component_blocks.csv",
        "joint": RUNTIME_ROOT / "artifacts/r1c0_d0_full_v1/metrics/joint_block_lobo/attempt_0001/oof_predictions.csv",
        "random": RUNTIME_ROOT / "artifacts/h1_random_cv_d0_v1/metrics/molecule_random_5fold/attempt_0001/oof_predictions.csv",
    }


def load_frozen_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paths = input_paths()
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    verify_stage_manifest(paths["joint"].parent)
    verify_stage_manifest(paths["random"].parent)

    labels = pd.read_csv(
        paths["labels"],
        usecols=["curated_id", "source", "sealed_block_id", "analogue_component_id", "permeability"],
    )
    assignments = pd.read_csv(
        paths["assignments"], usecols=["curated_id", "sealed_block_id", "outer_fold"]
    )
    blocks = pd.read_csv(paths["blocks"])

    if len(labels) != 6895 or labels["curated_id"].nunique() != 6895:
        raise RuntimeError("Frozen analytical population is not exactly 6,895 unique records")
    if labels["source"].nunique() != 41 or labels["sealed_block_id"].nunique() != 18:
        raise RuntimeError("Frozen source/block universe mismatch")
    mapping = labels[["curated_id", "sealed_block_id"]].merge(
        assignments, on="curated_id", suffixes=("_labels", "_assignments"), validate="one_to_one"
    )
    if not (mapping["sealed_block_id_labels"] == mapping["sealed_block_id_assignments"]).all():
        raise RuntimeError("Frozen block assignments disagree with labels")

    metadata = labels.merge(
        assignments[["curated_id", "outer_fold"]].rename(columns={"outer_fold": "assigned_outer_fold"}),
        on="curated_id",
        validate="one_to_one",
    )
    arms: list[pd.DataFrame] = []
    for arm, key in (("joint", "joint"), ("random", "random")):
        predictions = pd.read_csv(paths[key])
        if len(predictions) != 34475:
            raise RuntimeError(f"Unexpected {arm} OOF prediction count")
        joined = predictions.merge(metadata, on="curated_id", validate="many_to_one")
        if joined[["source", "sealed_block_id", "permeability"]].isna().any().any():
            raise RuntimeError(f"Incomplete metadata join for {arm}")
        joined = joined.rename(columns={"outer_fold": "prediction_fold"})
        if arm == "joint" and not (
            joined["prediction_fold"].astype(int) == joined["assigned_outer_fold"].astype(int)
        ).all():
            raise RuntimeError("Joint-block prediction/assignment fold mismatch")
        joined = joined.rename(columns={"assigned_outer_fold": "outer_fold"})
        joined["absolute_error"] = (
            joined["prediction_log10_papp"].astype(float) - joined["permeability"].astype(float)
        ).abs()
        joined["arm"] = arm
        arms.append(joined)
    rows = pd.concat(arms, ignore_index=True)

    block_summary = (
        labels.groupby("sealed_block_id", sort=True)
        .agg(
            n_rows=("curated_id", "size"),
            n_sources=("source", "nunique"),
            n_analogue_components=("analogue_component_id", "nunique"),
        )
        .reset_index()
        .merge(
            assignments.groupby("sealed_block_id", sort=True)["outer_fold"].nunique().reset_index(name="n_folds"),
            on="sealed_block_id",
            validate="one_to_one",
        )
    )
    if not (block_summary["n_folds"] == 1).all():
        raise RuntimeError("A frozen joint block maps to more than one outer fold")
    fold_map = assignments.groupby("sealed_block_id", sort=True)["outer_fold"].first().reset_index()
    block_summary = block_summary.drop(columns="n_folds").merge(
        fold_map, on="sealed_block_id", validate="one_to_one"
    )

    # Cross-check the aggregate block manifest without relying on it for calculations.
    manifest = blocks[["sealed_block_id", "n_curated_rows", "n_sources", "n_analogue_components"]]
    checked = block_summary.merge(manifest, on="sealed_block_id", suffixes=("_calculated", "_manifest"))
    for name, manifest_name in (
        ("n_rows", "n_curated_rows"),
        ("n_sources_calculated", "n_sources_manifest"),
        ("n_analogue_components_calculated", "n_analogue_components_manifest"),
    ):
        if not (checked[name].astype(int) == checked[manifest_name].astype(int)).all():
            raise RuntimeError(f"Block manifest mismatch for {name}")

    hashes = {name: sha256(path) for name, path in paths.items()}
    return rows, block_summary, hashes


def source_table(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["arm", "seed", "sealed_block_id", "outer_fold", "source"], sort=True)["absolute_error"]
        .mean()
        .reset_index(name="source_mae")
    )


def aggregate_metrics(rows: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, float | str]] = []
    for estimand in ("row_micro", "source_macro", "block_macro"):
        if estimand == "row_micro":
            per_seed = rows.groupby(["arm", "seed"], sort=True)["absolute_error"].mean().reset_index(name="mae")
        elif estimand == "source_macro":
            per_seed = sources.groupby(["arm", "seed"], sort=True)["source_mae"].mean().reset_index(name="mae")
        else:
            per_block = (
                sources.groupby(["arm", "seed", "sealed_block_id"], sort=True)["source_mae"]
                .mean()
                .reset_index(name="block_source_macro_mae")
            )
            per_seed = (
                per_block.groupby(["arm", "seed"], sort=True)["block_source_macro_mae"]
                .mean()
                .reset_index(name="mae")
            )
        values = per_seed.groupby("arm", sort=True)["mae"].mean()
        output.append(
            {
                "estimand": estimand,
                "joint_mae": float(values["joint"]),
                "random_mae": float(values["random"]),
                "gap_joint_minus_random": float(values["joint"] - values["random"]),
            }
        )
    result = pd.DataFrame(output)
    full = result.set_index("estimand").loc["source_macro"]
    for key, value in (("joint", full["joint_mae"]), ("random", full["random_mae"]), ("gap", full["gap_joint_minus_random"])):
        if not np.isclose(float(value), EXPECTED[key], rtol=0, atol=1e-12):
            raise RuntimeError(f"Headline H1 mismatch for {key}: {value} != {EXPECTED[key]}")
    return result


def block_influence(sources: pd.DataFrame, block_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    full_gap = EXPECTED["gap"]
    for block_id in sorted(sources["sealed_block_id"].unique()):
        kept = sources.loc[sources["sealed_block_id"] != block_id]
        per_seed = kept.groupby(["arm", "seed"], sort=True)["source_mae"].mean().reset_index(name="mae")
        values = per_seed.groupby("arm", sort=True)["mae"].mean()
        meta = block_summary.loc[block_summary["sealed_block_id"] == block_id].iloc[0]
        gap = float(values["joint"] - values["random"])
        rows.append(
            {
                "omitted_block_id": block_id,
                "outer_fold": int(meta["outer_fold"]),
                "omitted_n_rows": int(meta["n_rows"]),
                "omitted_n_sources": int(meta["n_sources"]),
                "omitted_n_analogue_components": int(meta["n_analogue_components"]),
                "joint_source_macro_mae": float(values["joint"]),
                "random_source_macro_mae": float(values["random"]),
                "gap_joint_minus_random": gap,
                "gap_change_from_full": gap - full_gap,
                "gap_at_least_0_10": bool(gap >= THRESHOLD),
            }
        )
    return pd.DataFrame(rows).sort_values("gap_joint_minus_random").reset_index(drop=True)


def per_block_effect(sources: pd.DataFrame, block_summary: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        sources.groupby(["arm", "seed", "sealed_block_id"], sort=True)["source_mae"]
        .mean()
        .reset_index(name="mae")
    )
    pivot = (
        per_seed.groupby(["arm", "sealed_block_id"], sort=True)["mae"]
        .mean()
        .unstack("arm")
        .reset_index()
    )
    pivot = pivot.rename(columns={"joint": "joint_source_macro_mae", "random": "random_source_macro_mae"})
    pivot["gap_joint_minus_random"] = pivot["joint_source_macro_mae"] - pivot["random_source_macro_mae"]
    return block_summary.merge(pivot, on="sealed_block_id", validate="one_to_one").sort_values("n_rows", ascending=False)


def top_three_omission(sources: pd.DataFrame, block_summary: pd.DataFrame) -> dict[str, object]:
    top_three = block_summary.nlargest(3, "n_rows")["sealed_block_id"].tolist()
    kept = sources.loc[~sources["sealed_block_id"].isin(top_three)]
    per_seed = kept.groupby(["arm", "seed"], sort=True)["source_mae"].mean().reset_index(name="mae")
    values = per_seed.groupby("arm", sort=True)["mae"].mean()
    remaining_sources = int(kept["source"].nunique())
    remaining_blocks = int(kept["sealed_block_id"].nunique())
    remaining_records = int(block_summary.loc[~block_summary["sealed_block_id"].isin(top_three), "n_rows"].sum())
    gap = float(values["joint"] - values["random"])
    return {
        "omitted_block_ids": top_three,
        "remaining_records": remaining_records,
        "remaining_sources": remaining_sources,
        "remaining_blocks": remaining_blocks,
        "joint_source_macro_mae": float(values["joint"]),
        "random_source_macro_mae": float(values["random"]),
        "gap_joint_minus_random": gap,
        "gap_at_least_0_10": bool(gap >= THRESHOLD),
    }


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def make_figure(influence: pd.DataFrame, blocks: pd.DataFrame, aggregates: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(183 * MM, 86 * MM))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.0, 1.04], wspace=0.48)

    ax = fig.add_subplot(grid[0, 0])
    plot = influence.sort_values("gap_joint_minus_random").reset_index(drop=True)
    y = np.arange(len(plot))
    colors = [COLORS["joint"] if n in set(plot.nlargest(3, "omitted_n_rows")["omitted_n_rows"]) else COLORS["neutral"] for n in plot["omitted_n_rows"]]
    ax.hlines(y, EXPECTED["gap"], plot["gap_joint_minus_random"], color=COLORS["grid"], linewidth=1.2, zorder=1)
    ax.scatter(plot["gap_joint_minus_random"], y, c=colors, s=18, edgecolor="white", linewidth=0.45, zorder=3)
    ax.axvline(EXPECTED["gap"], color=COLORS["ink"], linestyle="--", linewidth=0.85, label="Full population")
    ax.axvline(THRESHOLD, color=COLORS["negative"], linestyle=":", linewidth=0.95, label="H1 threshold")
    ax.set_yticks(y, [f"F{int(v):02d}" for v in plot["outer_fold"]])
    ax.set_xlabel("Gap after omitting one block\n(joint − random MAE)")
    ax.set_ylabel("Omitted outer fold")
    ax.set_title("Single-block influence", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    ax.legend(loc="lower right", fontsize=5.8)
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    block_plot = blocks.sort_values("n_rows", ascending=False).reset_index(drop=True)
    point_colors = [COLORS["joint"] if i < 3 else COLORS["neutral"] for i in range(len(block_plot))]
    ax.scatter(block_plot["n_rows"], block_plot["gap_joint_minus_random"], c=point_colors, s=23, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax.axhline(THRESHOLD, color=COLORS["negative"], linestyle=":", linewidth=0.9)
    for _, row in block_plot.head(3).iterrows():
        ax.annotate(f"F{int(row['outer_fold']):02d}", (row["n_rows"], row["gap_joint_minus_random"]), xytext=(4, 4), textcoords="offset points", fontsize=6, fontweight="bold")
    for _, row in block_plot.loc[block_plot["gap_joint_minus_random"] < 0].iterrows():
        ax.annotate(f"F{int(row['outer_fold']):02d}", (row["n_rows"], row["gap_joint_minus_random"]), xytext=(4, -9), textcoords="offset points", fontsize=6, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Records in block (log scale)")
    ax.set_ylabel("Within-block gap\n(joint − random MAE)")
    ax.set_title("Block heterogeneity", loc="left", fontweight="bold")
    ax.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    panel_label(ax, "b")

    ax = fig.add_subplot(grid[0, 2])
    order = ["row_micro", "source_macro", "block_macro"]
    labels = ["Row micro", "Source macro", "Block macro"]
    lookup = aggregates.set_index("estimand").loc[order]
    y = np.arange(3)
    for i, (_, row) in enumerate(lookup.iterrows()):
        ax.plot([row["random_mae"], row["joint_mae"]], [i, i], color=COLORS["grid"], linewidth=2.0, zorder=1)
        ax.text(row["joint_mae"] + 0.035, i, f"Δ {row['gap_joint_minus_random']:.2f}", va="center", fontsize=5.8, color=COLORS["ink"])
    ax.scatter(lookup["random_mae"], y, color=COLORS["random"], s=25, label="Molecule-random", zorder=3)
    ax.scatter(lookup["joint_mae"], y, color=COLORS["joint"], s=25, label="Joint-block", zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error (log$_{10}$ Papp)")
    ax.set_title("Aggregation estimand", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=5.8)
    panel_label(ax, "c")

    fig.suptitle("H1 remains qualitatively robust to single-block omission and weighting choice", x=0.02, ha="left", y=1.015, fontsize=9, fontweight="bold")
    for suffix, kwargs in {
        "svg": {},
        "pdf": {},
        "tiff": {"dpi": 600},
        "png": {"dpi": 300},
    }.items():
        fig.savefig(FIGURE_DIR / f"figure_s1_h1_block_influence.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(summary: dict[str, object], influence: pd.DataFrame, aggregates: pd.DataFrame, blocks: pd.DataFrame) -> None:
    min_row = influence.loc[influence["gap_joint_minus_random"].idxmin()]
    max_row = influence.loc[influence["gap_joint_minus_random"].idxmax()]
    negative_blocks = blocks.loc[blocks["gap_joint_minus_random"] < 0]
    top_three = summary["top_three_simultaneous_omission"]
    agg_lines = "\n".join(
        f"- {row.estimand}: joint {row.joint_mae:.3f}, random {row.random_mae:.3f}, gap {row.gap_joint_minus_random:.3f}."
        for row in aggregates.itertuples()
    )
    report = f"""# H1 block-influence robustness report

## Scope

This is a descriptive, zero-training analysis of the already frozen H1 out-of-fold predictions. It was specified after the confirmatory H1 result was known and is therefore not treated as a new confirmatory test. No model was fitted, selected, calibrated, or tuned.

## Main result

The full source-macro gap was reproduced exactly: joint-block MAE {EXPECTED['joint']:.6f}, molecule-random MAE {EXPECTED['random']:.6f}, and gap {EXPECTED['gap']:.6f} log10 Papp units.

All 18 leave-one-block-out recalculations remained above the preregistered 0.10 point-gap threshold. The smallest gap was {min_row['gap_joint_minus_random']:.3f} after omitting fold {int(min_row['outer_fold'])} ({int(min_row['omitted_n_rows']):,} records; {int(min_row['omitted_n_sources'])} source(s)), and the largest was {max_row['gap_joint_minus_random']:.3f} after omitting fold {int(max_row['outer_fold'])} ({int(max_row['omitted_n_rows']):,} records; {int(max_row['omitted_n_sources'])} source(s)). Thus, no single frozen joint block alone explains the qualitative H1 contrast.

Simultaneously removing the three largest blocks left only {top_three['remaining_records']:,} records, {top_three['remaining_sources']} sources, and {top_three['remaining_blocks']} blocks. In that heavily altered target population, the source-macro gap was {top_three['gap_joint_minus_random']:.3f}. This stress test is reported only as a sensitivity description because the represented population changes substantially.

## Aggregation estimands

{agg_lines}

The H1 direction is unchanged whether records, sources, or blocks receive equal weight. The magnitude changes because these estimands answer different questions in a highly imbalanced evidence geometry.

## Heterogeneity

{len(negative_blocks)} of 18 individual blocks had a negative within-block gap. A negative within-block value does not contradict the population-level H1 result; it identifies regimes where the two split strategies behaved differently and motivates the next stratified diagnostic.

## Interpretation boundary

- The analysis supports robustness to deleting any one existing block; it does not demonstrate external validity to a new laboratory or chemical regime.
- Leave-one-block-out deletion changes the represented target population and does not yield a replacement confidence interval.
- The original 10,000-replicate block/seed bootstrap remains the inferential analysis for H1.
"""
    (HERE / "H1_BLOCK_INFLUENCE_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows, block_summary, hashes = load_frozen_data()
    sources = source_table(rows)
    aggregates = aggregate_metrics(rows, sources)
    influence = block_influence(sources, block_summary)
    blocks = per_block_effect(sources, block_summary)
    top_three = top_three_omission(sources, block_summary)

    influence.to_csv(SOURCE_DIR / "h1_leave_one_block_out.csv", index=False, float_format="%.12f")
    blocks.to_csv(SOURCE_DIR / "h1_per_block_effect.csv", index=False, float_format="%.12f")
    aggregates.to_csv(SOURCE_DIR / "h1_aggregation_sensitivity.csv", index=False, float_format="%.12f")

    summary: dict[str, object] = {
        "schema_version": "scaffoldseal-h1-descriptive-block-influence-v1",
        "analysis_status": "post-confirmatory descriptive robustness analysis",
        "training_or_tuning_performed": False,
        "n_records": 6895,
        "n_sources": 41,
        "n_blocks": 18,
        "n_seeds": 5,
        "headline_source_macro": EXPECTED,
        "leave_one_block_out": {
            "minimum_gap": float(influence["gap_joint_minus_random"].min()),
            "maximum_gap": float(influence["gap_joint_minus_random"].max()),
            "n_of_18_at_least_0_10": int(influence["gap_at_least_0_10"].sum()),
        },
        "top_three_simultaneous_omission": top_three,
        "input_sha256": hashes,
    }
    (SOURCE_DIR / "h1_block_influence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(influence, blocks, aggregates)
    write_report(summary, influence, aggregates, blocks)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
