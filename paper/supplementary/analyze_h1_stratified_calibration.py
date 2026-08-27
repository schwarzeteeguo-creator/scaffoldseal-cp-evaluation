"""Source/block-stratified error and calibration diagnostics for frozen H1 predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_h1_block_influence import COLORS, load_frozen_data


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "source_data"
FIGURE_DIR = HERE / "figures"
MM = 1 / 25.4

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


def root_mean_square(values: pd.Series) -> float:
    array = values.to_numpy(float)
    return float(np.sqrt(np.mean(np.square(array))))


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = ["_".join(str(part) for part in column if str(part)) for column in output.columns]
    return output


def stratified_tables(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = rows.copy()
    data["signed_error"] = data["prediction_log10_papp"] - data["permeability"]
    data["squared_error"] = np.square(data["signed_error"])

    source_seed = (
        data.groupby(["arm", "seed", "source", "sealed_block_id", "outer_fold"], sort=True)
        .agg(
            n_records=("curated_id", "nunique"),
            mae=("absolute_error", "mean"),
            signed_bias=("signed_error", "mean"),
            mse=("squared_error", "mean"),
        )
        .reset_index()
    )
    source_seed["rmse"] = np.sqrt(source_seed["mse"])
    source_mean = (
        source_seed.groupby(["arm", "source", "sealed_block_id", "outer_fold"], sort=True)
        .agg(
            n_records=("n_records", "first"),
            mae=("mae", "mean"),
            signed_bias=("signed_bias", "mean"),
            rmse=("rmse", "mean"),
        )
        .reset_index()
    )
    source_wide = flatten_columns(
        source_mean.pivot(
            index=["source", "sealed_block_id", "outer_fold", "n_records"],
            columns="arm",
            values=["mae", "signed_bias", "rmse"],
        ).reset_index()
    )
    source_wide["mae_gap_joint_minus_random"] = source_wide["mae_joint"] - source_wide["mae_random"]
    source_wide["absolute_bias_joint"] = source_wide["signed_bias_joint"].abs()
    source_wide["absolute_bias_random"] = source_wide["signed_bias_random"].abs()

    seed_gaps = source_seed.pivot(index=["source", "seed"], columns="arm", values="mae").reset_index()
    seed_gaps["gap"] = seed_gaps["joint"] - seed_gaps["random"]
    seed_summary = (
        seed_gaps.groupby("source", sort=True)["gap"]
        .agg(
            gap_seed_mean="mean",
            gap_seed_sd="std",
            n_positive_seeds=lambda values: int((values > 0).sum()),
            n_negative_seeds=lambda values: int((values < 0).sum()),
        )
        .reset_index()
    )
    source_wide = source_wide.merge(seed_summary, on="source", validate="one_to_one")

    block_seed = (
        data.groupby(["arm", "seed", "sealed_block_id", "outer_fold"], sort=True)
        .agg(
            n_records=("curated_id", "nunique"),
            n_sources=("source", "nunique"),
            mae=("absolute_error", "mean"),
            signed_bias=("signed_error", "mean"),
            mse=("squared_error", "mean"),
        )
        .reset_index()
    )
    block_seed["rmse"] = np.sqrt(block_seed["mse"])
    block_mean = (
        block_seed.groupby(["arm", "sealed_block_id", "outer_fold"], sort=True)
        .agg(
            n_records=("n_records", "first"),
            n_sources=("n_sources", "first"),
            mae=("mae", "mean"),
            signed_bias=("signed_bias", "mean"),
            rmse=("rmse", "mean"),
        )
        .reset_index()
    )
    block_wide = flatten_columns(
        block_mean.pivot(
            index=["sealed_block_id", "outer_fold", "n_records", "n_sources"],
            columns="arm",
            values=["mae", "signed_bias", "rmse"],
        ).reset_index()
    )
    block_wide["mae_gap_joint_minus_random"] = block_wide["mae_joint"] - block_wide["mae_random"]
    block_wide["absolute_bias_joint"] = block_wide["signed_bias_joint"].abs()
    block_wide["absolute_bias_random"] = block_wide["signed_bias_random"].abs()
    return source_wide.sort_values("mae_gap_joint_minus_random"), block_wide.sort_values("outer_fold")


def calibration_tables(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_record = (
        rows.groupby(["arm", "curated_id"], sort=True)
        .agg(
            prediction=("prediction_log10_papp", "mean"),
            observed=("permeability", "first"),
            source=("source", "first"),
            sealed_block_id=("sealed_block_id", "first"),
            outer_fold=("outer_fold", "first"),
        )
        .reset_index()
    )
    curve_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for arm, group in per_record.groupby("arm", sort=True):
        group = group.copy()
        group["prediction_decile"] = pd.qcut(group["prediction"], 10, labels=False, duplicates="raise") + 1
        curve = (
            group.groupby("prediction_decile", sort=True)
            .agg(
                n_records=("curated_id", "size"),
                mean_prediction=("prediction", "mean"),
                mean_observed=("observed", "mean"),
                minimum_prediction=("prediction", "min"),
                maximum_prediction=("prediction", "max"),
            )
            .reset_index()
        )
        curve["mean_signed_bias"] = curve["mean_prediction"] - curve["mean_observed"]
        curve["arm"] = arm
        curve_rows.append(curve)

        slope, intercept = np.polyfit(group["prediction"], group["observed"], 1)
        pearson = np.corrcoef(group["prediction"], group["observed"])[0, 1]
        weighted_decile_error = np.average(
            np.abs(curve["mean_signed_bias"]), weights=curve["n_records"]
        )
        summary_rows.append(
            {
                "arm": arm,
                "n_records": int(len(group)),
                "calibration_slope_observed_on_prediction": float(slope),
                "calibration_intercept": float(intercept),
                "pearson_r": float(pearson),
                "global_signed_bias": float((group["prediction"] - group["observed"]).mean()),
                "decile_weighted_absolute_calibration_error": float(weighted_decile_error),
            }
        )
    return pd.concat(curve_rows, ignore_index=True), pd.DataFrame(summary_rows)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.17, 1.09, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def display_source(name: str) -> str:
    return name.replace("_", " ")


def make_figure(sources: pd.DataFrame, calibration: pd.DataFrame, calibration_summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(183 * MM, 82 * MM))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.02, 1.08], wspace=0.48)

    ax = fig.add_subplot(grid[0, 0])
    ranked = sources.sort_values("mae_gap_joint_minus_random").reset_index(drop=True)
    y = np.arange(len(ranked))
    positive = ranked["mae_gap_joint_minus_random"] >= 0
    ax.scatter(
        ranked.loc[positive, "mae_gap_joint_minus_random"], y[positive],
        color=COLORS["joint"], marker="o", s=16, edgecolor="white", linewidth=0.4, label="Joint worse",
    )
    ax.scatter(
        ranked.loc[~positive, "mae_gap_joint_minus_random"], y[~positive],
        color=COLORS["positive"], marker="D", s=17, edgecolor="white", linewidth=0.4, label="Joint better",
    )
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    ax.set_yticks([])
    ax.set_ylabel("All 41 sources (ranked)")
    ax.set_xlabel("Source MAE gap\n(joint − random)")
    ax.set_title("Breadth across sources", loc="left", fontweight="bold")
    label_indices = set(ranked.nlargest(5, "mae_gap_joint_minus_random").index) | set(ranked.loc[~positive].index)
    source_offsets_a = {
        "2022_Lee": (4, 1),
        "2018_Lee": (4, 0),
        "2019_Ono": (4, 7),
        "2018_García-Pindado": (4, -2),
        "2018_Buckton": (4, -7),
        "2015_Nielsen": (-4, 3),
        "2024_Otani": (-4, 0),
        "2006_Rezai_1": (-4, -3),
    }
    for idx in sorted(label_indices):
        row = ranked.loc[idx]
        offset = source_offsets_a.get(
            str(row["source"]),
            (4, 0) if row["mae_gap_joint_minus_random"] >= 0 else (-4, 0),
        )
        align = "left" if row["mae_gap_joint_minus_random"] >= 0 else "right"
        ax.annotate(
            display_source(str(row["source"])),
            (row["mae_gap_joint_minus_random"], idx),
            xytext=offset,
            textcoords="offset points",
            va="center",
            ha=align,
            fontsize=5.4,
        )
    ax.text(
        0.03, 0.97,
        "38/41 positive on average\n28/41 positive in all 5 seeds",
        transform=ax.transAxes, va="top", fontsize=5.9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.8},
    )
    ax.legend(loc="lower right", fontsize=5.7)
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    limit = 3.0
    ax.plot([-limit, limit], [-limit, limit], color=COLORS["grid"], linestyle="--", linewidth=0.9, zorder=1)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.65)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.65)
    ax.scatter(
        sources["signed_bias_random"], sources["signed_bias_joint"],
        color=COLORS["joint"], s=18, alpha=0.9, edgecolor="white", linewidth=0.45, zorder=3,
    )
    source_offsets_b = {
        "2022_Lee": (6, 5),
        "2018_Lee": (5, -14),
        "2018_Buckton": (6, 5),
        "2018_García-Pindado": (6, -11),
        "2019_Ono": (6, 8),
    }
    for _, row in sources.nlargest(5, "absolute_bias_joint").iterrows():
        ax.annotate(
            display_source(str(row["source"])),
            (row["signed_bias_random"], row["signed_bias_joint"]),
            xytext=source_offsets_b.get(str(row["source"]), (5, 3)),
            textcoords="offset points", fontsize=5.3,
        )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.set_xlabel("Random-split source bias")
    ax.set_ylabel("Joint-block source bias")
    ax.set_title("Signed bias by source", loc="left", fontweight="bold")
    ax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.55)
    ax.text(
        0.03, 0.97,
        "Median |bias|\nrandom 0.08; joint 0.58",
        transform=ax.transAxes, va="top", fontsize=5.9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.8},
    )
    panel_label(ax, "b")

    ax = fig.add_subplot(grid[0, 2])
    lower, upper = -7.3, -4.4
    ax.plot([lower, upper], [lower, upper], color=COLORS["ink"], linestyle="--", linewidth=0.8, label="Ideal")
    for arm, color, label in (
        ("random", COLORS["random"], "Molecule-random"),
        ("joint", COLORS["joint"], "Joint-block"),
    ):
        curve = calibration.loc[calibration["arm"] == arm].sort_values("prediction_decile")
        ax.plot(curve["mean_prediction"], curve["mean_observed"], color=color, marker="o", markersize=3.6, linewidth=1.2, label=label)
    summary = calibration_summary.set_index("arm")
    ax.text(
        0.03, 0.97,
        f"Calibration slope\nrandom {summary.loc['random', 'calibration_slope_observed_on_prediction']:.2f}\njoint {summary.loc['joint', 'calibration_slope_observed_on_prediction']:.2f}",
        transform=ax.transAxes, va="top", fontsize=5.9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.8},
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.set_xlabel("Mean predicted log$_{10}$ Papp")
    ax.set_ylabel("Mean observed log$_{10}$ Papp")
    ax.set_title("Prediction-decile calibration", loc="left", fontweight="bold")
    ax.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.65)
    ax.legend(loc="lower right", fontsize=5.5)
    panel_label(ax, "c")

    fig.suptitle(
        "Joint-block shift broadens source bias and degrades point-prediction calibration",
        x=0.02, ha="left", y=1.015, fontsize=9, fontweight="bold",
    )
    for suffix, kwargs in {
        "svg": {}, "pdf": {}, "tiff": {"dpi": 600}, "png": {"dpi": 300}
    }.items():
        fig.savefig(FIGURE_DIR / f"figure_s2_h1_stratified_calibration.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(
    sources: pd.DataFrame,
    blocks: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    input_hashes: dict[str, str],
) -> dict[str, object]:
    positive_mean = int((sources["mae_gap_joint_minus_random"] > 0).sum())
    positive_all = int((sources["n_positive_seeds"] == 5).sum())
    positive_four = int((sources["n_positive_seeds"] >= 4).sum())
    joint_median_abs_bias = float(sources["absolute_bias_joint"].median())
    random_median_abs_bias = float(sources["absolute_bias_random"].median())
    joint_large_bias = int((sources["absolute_bias_joint"] > 0.5).sum())
    random_large_bias = int((sources["absolute_bias_random"] > 0.5).sum())
    calibration = calibration_summary.set_index("arm")
    top_sources = sources.nlargest(5, "mae_gap_joint_minus_random")
    top_blocks = blocks.nlargest(3, "absolute_bias_joint")

    summary: dict[str, object] = {
        "schema_version": "scaffoldseal-h1-stratified-calibration-v1",
        "analysis_status": "post-confirmatory descriptive diagnosis",
        "training_tuning_or_recalibration_performed": False,
        "n_records": 6895,
        "n_sources": 41,
        "n_blocks": 18,
        "n_seeds": 5,
        "source_gap_breadth": {
            "n_positive_mean_gap": positive_mean,
            "n_positive_all_five_seeds": positive_all,
            "n_positive_at_least_four_seeds": positive_four,
        },
        "source_absolute_bias": {
            "joint_median": joint_median_abs_bias,
            "random_median": random_median_abs_bias,
            "joint_n_above_0_5": joint_large_bias,
            "random_n_above_0_5": random_large_bias,
        },
        "calibration": calibration_summary.to_dict(orient="records"),
        "input_sha256": input_hashes,
    }
    (SOURCE_DIR / "h1_stratified_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    top_source_lines = "\n".join(
        f"- {row.source}: n={int(row.n_records)}, gap {row.mae_gap_joint_minus_random:.3f}, joint bias {row.signed_bias_joint:.3f}."
        for row in top_sources.itertuples()
    )
    top_block_lines = "\n".join(
        f"- Fold {int(row.outer_fold)}: n={int(row.n_records)}, joint bias {row.signed_bias_joint:.3f}, random bias {row.signed_bias_random:.3f}."
        for row in top_blocks.itertuples()
    )
    report = f"""# H1 source-stratified error and point-calibration report

## Scope

This is a post-confirmatory, zero-training diagnosis of the frozen H1 OOF predictions. It did not fit, tune, select or recalibrate any model. Source and block summaries are descriptive five-seed means; calibration curves use one five-seed mean OOF prediction per frozen record.

## Breadth of the random-split optimism gap

The joint-block minus molecule-random MAE gap was positive for {positive_mean}/41 sources on average. It was positive in all five seeds for {positive_all}/41 sources and in at least four seeds for {positive_four}/41 sources. The result is therefore widespread rather than confined to one or two sources, although the magnitude is highly heterogeneous.

The five largest source-level gaps were:

{top_source_lines}

Several extremes have small source sample sizes, so their magnitudes should not be interpreted as stable population estimates. They are localization diagnostics, not independent confirmatory effects.

## Source and block signed bias

The median absolute source-level signed bias increased from {random_median_abs_bias:.3f} under molecule-random evaluation to {joint_median_abs_bias:.3f} under joint-block evaluation. Absolute bias exceeded 0.5 log10 Papp units for {joint_large_bias}/41 joint-block source summaries versus {random_large_bias}/41 molecule-random summaries.

The three blocks with the largest absolute joint-block row-weighted bias were:

{top_block_lines}

Positive bias denotes permeability predictions that are too high (less negative log10 Papp); negative bias denotes predictions that are too low.

## Point-prediction calibration

After averaging the five OOF predictions for each record, the observed-on-predicted calibration slope was {calibration.loc['random', 'calibration_slope_observed_on_prediction']:.3f} for molecule-random evaluation and {calibration.loc['joint', 'calibration_slope_observed_on_prediction']:.3f} for joint-block evaluation. The decile-weighted absolute calibration error was {calibration.loc['random', 'decile_weighted_absolute_calibration_error']:.3f} and {calibration.loc['joint', 'decile_weighted_absolute_calibration_error']:.3f}, respectively. Global signed bias was {calibration.loc['random', 'global_signed_bias']:.3f} under random splitting and {calibration.loc['joint', 'global_signed_bias']:.3f} under joint-block shift.

The joint-block slope below one and the compressed prediction-decile curve indicate regression toward the training-domain mean under source/analogue shift. This is a point-prediction calibration diagnosis and is separate from the D3 empirical interval-coverage result.

## Interpretation boundary

- These analyses explain the released H1 contrast; they do not constitute external validation.
- Prediction bins are arm-specific and descriptive, and no post-hoc recalibration was fitted.
- Sources are highly unequal in size; all 41 are retained, but small-source extremes are labelled as such in the source data.
- The original H1 block/seed bootstrap remains the inferential analysis.
"""
    (HERE / "H1_STRATIFIED_CALIBRATION_REPORT.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows, _, input_hashes = load_frozen_data()
    sources, blocks = stratified_tables(rows)
    calibration_curve, calibration_summary = calibration_tables(rows)

    if len(sources) != 41 or len(blocks) != 18:
        raise RuntimeError("Stratified source/block universe mismatch")
    if set(calibration_curve.groupby("arm")["prediction_decile"].nunique()) != {10}:
        raise RuntimeError("Calibration curves do not contain exactly ten bins per arm")

    sources.to_csv(SOURCE_DIR / "h1_source_stratified_metrics.csv", index=False, float_format="%.12f")
    blocks.to_csv(SOURCE_DIR / "h1_block_stratified_metrics.csv", index=False, float_format="%.12f")
    calibration_curve.to_csv(SOURCE_DIR / "h1_prediction_decile_calibration.csv", index=False, float_format="%.12f")
    calibration_summary.to_csv(SOURCE_DIR / "h1_calibration_summary.csv", index=False, float_format="%.12f")
    summary = write_report(sources, blocks, calibration_summary, input_hashes)
    make_figure(sources, calibration_curve, calibration_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
