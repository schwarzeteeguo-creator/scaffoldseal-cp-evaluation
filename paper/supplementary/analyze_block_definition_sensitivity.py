"""Outcome-blind analogue-threshold sensitivity of source/component block geometry."""

from __future__ import annotations

import hashlib
import json
import sys
from math import comb
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT = WORKSPACE / "scaffoldseal"
SOURCE_DIR = HERE / "source_data"
FIGURE_DIR = HERE / "figures"
THRESHOLDS = (0.70, 0.80, 0.90)
PRIMARY_THRESHOLD = 0.80
MM = 1 / 25.4

sys.path.insert(0, str(PROJECT / "src"))
from build_manifests import build_analogue_graph, build_source_component_blocks  # noqa: E402


COLORS = {
    0.70: "#6F7D8C",
    0.80: "#D07A5F",
    0.90: "#4C78A8",
    "source": "#4F8F75",
    "ink": "#27313A",
    "grid": "#D9DEE3",
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
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
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


def pair_partition_scores(reference: pd.Series, alternative: pd.Series) -> tuple[float, float]:
    contingency = pd.crosstab(reference, alternative).to_numpy(dtype=np.int64)
    index = float(sum(comb(int(value), 2) for value in contingency.ravel()))
    reference_pairs = float(sum(comb(int(value), 2) for value in contingency.sum(axis=1)))
    alternative_pairs = float(sum(comb(int(value), 2) for value in contingency.sum(axis=0)))
    total_pairs = float(comb(int(contingency.sum()), 2))
    expected = reference_pairs * alternative_pairs / total_pairs
    maximum = 0.5 * (reference_pairs + alternative_pairs)
    adjusted_rand = 1.0 if maximum == expected else (index - expected) / (maximum - expected)
    union = reference_pairs + alternative_pairs - index
    pairwise_jaccard = 1.0 if union == 0 else index / union
    return float(adjusted_rand), float(pairwise_jaccard)


def reconstruct() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    curated_path = PROJECT / "artifacts" / "curated_records_public.csv"
    config_path = PROJECT / "config.yaml"
    builder_path = PROJECT / "src" / "build_manifests.py"
    primary_manifest_path = PROJECT / "artifacts" / "split_manifest_public.csv"
    for path in (curated_path, config_path, builder_path, primary_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    curated = pd.read_csv(curated_path)
    if len(curated) != 6895 or curated["molecule_id"].nunique() != 6862 or curated["source"].nunique() != 41:
        raise RuntimeError("Frozen public curated population mismatch")
    forbidden = {"permeability", "label", "target", "outcome", "papp"}
    if forbidden & {column.lower() for column in curated.columns}:
        raise RuntimeError("Outcome-like column found in public graph input")

    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    threshold_rows: list[dict[str, object]] = []
    ranked_rows: list[dict[str, object]] = []
    assignment_frames: list[pd.DataFrame] = []
    component_frames: dict[float, pd.DataFrame] = {}

    for threshold in THRESHOLDS:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["analogue"]["tanimoto_threshold"] = float(threshold)
        analogue = build_analogue_graph(curated, config)
        block_result = build_source_component_blocks(curated, analogue.molecules)
        blocks = block_result.blocks.sort_values("n_curated_rows", ascending=False).reset_index(drop=True)
        shares = blocks["n_curated_rows"].to_numpy(float) / len(curated)
        n_ecfp_edges = int(analogue.edges["edge_types"].str.contains("ecfp4", regex=False).sum())
        n_edit_edges = int(analogue.edges["edge_types"].str.contains("exact_one_token_edit", regex=False).sum())
        threshold_rows.append(
            {
                "tanimoto_threshold": threshold,
                "n_total_analogue_edges": int(len(analogue.edges)),
                "n_ecfp4_edges": n_ecfp_edges,
                "n_exact_one_token_edit_edges": n_edit_edges,
                "n_analogue_components": int(analogue.molecules["analogue_component_id"].nunique()),
                "n_source_groups": int(curated["source"].nunique()),
                "n_joint_blocks": int(len(blocks)),
                "largest_block_records": int(blocks.iloc[0]["n_curated_rows"]),
                "largest_block_share": float(shares[0]),
                "top_three_block_share": float(shares[:3].sum()),
                "effective_number_of_blocks": float(1.0 / np.square(shares).sum()),
                "median_block_records": float(blocks["n_curated_rows"].median()),
            }
        )
        cumulative = 0.0
        for rank, row in enumerate(blocks.itertuples(index=False), start=1):
            share = float(row.n_curated_rows / len(curated))
            cumulative += share
            ranked_rows.append(
                {
                    "tanimoto_threshold": threshold,
                    "block_rank": rank,
                    "sealed_block_id": row.sealed_block_id,
                    "n_records": int(row.n_curated_rows),
                    "n_unique_molecules": int(row.n_unique_molecules),
                    "n_analogue_components": int(row.n_analogue_components),
                    "n_sources": int(row.n_sources),
                    "record_share": share,
                    "cumulative_record_share": cumulative,
                }
            )
        assignments = block_result.rows[
            ["curated_id", "molecule_id", "source", "analogue_component_id", "sealed_block_id"]
        ].copy()
        assignments.insert(0, "tanimoto_threshold", threshold)
        assignment_frames.append(assignments)
        component_frames[threshold] = analogue.molecules[["molecule_id", "analogue_component_id"]].copy()

    threshold_summary = pd.DataFrame(threshold_rows)
    ranked_sizes = pd.DataFrame(ranked_rows)
    assignments = pd.concat(assignment_frames, ignore_index=True)

    primary_new = assignments.loc[assignments["tanimoto_threshold"] == PRIMARY_THRESHOLD]
    primary_old = pd.read_csv(
        primary_manifest_path, usecols=["curated_id", "analogue_component_id", "sealed_block_id"]
    )
    exact = primary_new.merge(primary_old, on="curated_id", suffixes=("_new", "_old"), validate="one_to_one")
    if not (exact["analogue_component_id_new"] == exact["analogue_component_id_old"]).all():
        raise RuntimeError("Reconstructed primary analogue components do not match the frozen manifest")
    if not (exact["sealed_block_id_new"] == exact["sealed_block_id_old"]).all():
        raise RuntimeError("Reconstructed primary joint blocks do not match the frozen manifest")
    primary_summary = threshold_summary.loc[threshold_summary["tanimoto_threshold"] == PRIMARY_THRESHOLD].iloc[0]
    if int(primary_summary["n_total_analogue_edges"]) != 141425 or int(primary_summary["n_analogue_components"]) != 305 or int(primary_summary["n_joint_blocks"]) != 18:
        raise RuntimeError("Reconstructed primary graph headline counts mismatch")

    comparison_rows: list[dict[str, object]] = []
    reference_records = primary_new[["curated_id", "sealed_block_id"]].rename(columns={"sealed_block_id": "primary_block"})
    reference_components = component_frames[PRIMARY_THRESHOLD].rename(columns={"analogue_component_id": "primary_component"})
    for threshold in THRESHOLDS:
        alternative_records = assignments.loc[
            assignments["tanimoto_threshold"] == threshold, ["curated_id", "sealed_block_id"]
        ].rename(columns={"sealed_block_id": "alternative_block"})
        mapping = reference_records.merge(alternative_records, on="curated_id", validate="one_to_one")
        block_ari, block_jaccard = pair_partition_scores(mapping["primary_block"], mapping["alternative_block"])
        primary_splits = mapping.groupby("primary_block")["alternative_block"].nunique()
        alternative_merges = mapping.groupby("alternative_block")["primary_block"].nunique()

        alternative_components = component_frames[threshold].rename(columns={"analogue_component_id": "alternative_component"})
        component_mapping = reference_components.merge(alternative_components, on="molecule_id", validate="one_to_one")
        component_ari, component_jaccard = pair_partition_scores(
            component_mapping["primary_component"], component_mapping["alternative_component"]
        )
        comparison_rows.append(
            {
                "tanimoto_threshold": threshold,
                "joint_block_adjusted_rand_to_primary": block_ari,
                "joint_block_pairwise_jaccard_to_primary": block_jaccard,
                "n_primary_blocks_split": int((primary_splits > 1).sum()),
                "maximum_alternative_fragments_per_primary_block": int(primary_splits.max()),
                "n_alternative_blocks_merging_primary_blocks": int((alternative_merges > 1).sum()),
                "maximum_primary_blocks_per_alternative_block": int(alternative_merges.max()),
                "analogue_component_adjusted_rand_to_primary": component_ari,
                "analogue_component_pairwise_jaccard_to_primary": component_jaccard,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    hashes = {
        "curated_records_public": sha256(curated_path),
        "config": sha256(config_path),
        "graph_builder": sha256(builder_path),
        "primary_split_manifest": sha256(primary_manifest_path),
    }
    return threshold_summary, ranked_sizes, assignments, comparison, hashes


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.15, 1.09, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def make_figure(summary: pd.DataFrame, ranked: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 76 * MM), gridspec_kw={"width_ratios": [0.95, 1.0, 1.18], "wspace": 0.42})

    ax = axes[0]
    ax.plot(summary["tanimoto_threshold"], summary["n_analogue_components"], color=COLORS[0.90], marker="o", linewidth=1.3)
    ax.plot(summary["tanimoto_threshold"], summary["n_joint_blocks"], color=COLORS[0.80], marker="o", linewidth=1.3)
    ax.plot(summary["tanimoto_threshold"], summary["n_source_groups"], color=COLORS["source"], marker="o", linestyle="--", linewidth=1.0)
    for _, row in summary.iterrows():
        ax.annotate(str(int(row["n_analogue_components"])), (row["tanimoto_threshold"], row["n_analogue_components"]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=5.8, color=COLORS[0.90])
        ax.annotate(str(int(row["n_joint_blocks"])), (row["tanimoto_threshold"], row["n_joint_blocks"]), xytext=(0, -10), textcoords="offset points", ha="center", fontsize=5.8, color=COLORS[0.80])
    ax.text(0.905, 43, "41 source groups", fontsize=5.8, color=COLORS["source"], va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(THRESHOLDS, [f"{value:.2f}" for value in THRESHOLDS])
    ax.set_xlabel("Chiral ECFP4 threshold")
    ax.set_ylabel("Number of groups (log scale)")
    ax.set_title("Molecular versus joint grouping", loc="left", fontweight="bold")
    ax.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    panel_label(ax, "a")

    ax = axes[1]
    ax.plot(summary["tanimoto_threshold"], 100 * summary["largest_block_share"], color=COLORS[0.70], marker="o", linewidth=1.3, label="Largest block")
    ax.plot(summary["tanimoto_threshold"], 100 * summary["top_three_block_share"], color=COLORS[0.80], marker="o", linewidth=1.3, label="Three largest")
    ax.set_xticks(THRESHOLDS, [f"{value:.2f}" for value in THRESHOLDS])
    ax.set_ylim(50, 100)
    ax.set_xlabel("Chiral ECFP4 threshold")
    ax.set_ylabel("Share of all records (%)")
    ax.set_title("Persistent concentration", loc="left", fontweight="bold")
    ax.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    ax.legend(loc="center right", fontsize=5.8)
    for _, row in summary.iterrows():
        ax.annotate(f"{100 * row['largest_block_share']:.1f}", (row["tanimoto_threshold"], 100 * row["largest_block_share"]), xytext=(0, -10), textcoords="offset points", ha="center", fontsize=5.6)
        ax.annotate(f"{100 * row['top_three_block_share']:.1f}", (row["tanimoto_threshold"], 100 * row["top_three_block_share"]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=5.6)
    panel_label(ax, "b")

    ax = axes[2]
    for threshold in THRESHOLDS:
        group = ranked.loc[ranked["tanimoto_threshold"] == threshold].sort_values("block_rank")
        linewidth = 1.8 if threshold == PRIMARY_THRESHOLD else 1.0
        markersize = 3.8 if threshold == PRIMARY_THRESHOLD else 3.0
        ax.plot(group["block_rank"], group["n_records"], color=COLORS[threshold], marker="o", markersize=markersize, linewidth=linewidth, label=f"Threshold {threshold:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Joint blocks ranked by size")
    ax.set_ylabel("Records in block (log scale)")
    ax.set_title("Complete block-size profiles", loc="left", fontweight="bold")
    ax.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    ax.legend(loc="upper right", fontsize=5.7)
    panel_label(ax, "c")

    fig.suptitle("Evidence concentration persists across prespecified analogue thresholds", x=0.02, ha="left", y=1.015, fontsize=9, fontweight="bold")
    for suffix, kwargs in {"svg": {}, "pdf": {}, "tiff": {"dpi": 600}, "png": {"dpi": 300}}.items():
        fig.savefig(FIGURE_DIR / f"figure_s3_block_definition_sensitivity.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(summary: pd.DataFrame, comparison: pd.DataFrame, hashes: dict[str, str]) -> dict[str, object]:
    summary_records = summary.to_dict(orient="records")
    comparison_records = comparison.to_dict(orient="records")
    payload: dict[str, object] = {
        "schema_version": "scaffoldseal-block-definition-sensitivity-v1",
        "analysis_status": "post-confirmatory outcome-blind geometry sensitivity",
        "model_training_or_performance_rescoring_performed": False,
        "thresholds": list(THRESHOLDS),
        "fixed_rules": {
            "ecfp_radius": 2,
            "ecfp_bits": 2048,
            "include_chirality": True,
            "molecular_weight_ratio": [0.80, 1.25],
            "token_rule": "exact one-token edit at unchanged topology and ring length",
            "source_provenance_closure": True,
        },
        "threshold_summary": summary_records,
        "partition_comparison_to_primary_0_80": comparison_records,
        "input_sha256": hashes,
    }
    (SOURCE_DIR / "block_definition_sensitivity_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table_lines = "\n".join(
        f"- Threshold {row.tanimoto_threshold:.2f}: {int(row.n_total_analogue_edges):,} analogue edges, {int(row.n_analogue_components)} analogue components, {int(row.n_joint_blocks)} joint blocks, largest share {100 * row.largest_block_share:.2f}%, top-three share {100 * row.top_three_block_share:.2f}%, effective block count {row.effective_number_of_blocks:.3f}."
        for row in summary.itertuples()
    )
    report = f"""# Analogue-threshold sensitivity of block geometry

## Scope and integrity boundary

This post-confirmatory sensitivity analysis rebuilt the outcome-blind analogue and source/component graphs at the prespecified descriptive thresholds 0.70, 0.80 and 0.90. Chiral ECFP4 radius/length, the 0.80–1.25 molecular-weight ratio, the exact one-token cyclic-edit rule, source provenance and all 6,895 public curated records were held fixed. No permeability value, fitted model or prediction was used to form any graph.

The threshold-0.80 reconstruction exactly reproduced the frozen 141,425 edges, 305 analogue components, 18 joint blocks and every record-level primary component/block identifier.

## Results

{table_lines}

Changing the similarity threshold strongly altered the molecular graph: analogue components ranged from 91 at 0.70 to 745 at 0.90. Source/component closure absorbed much of this variation, leaving only 15–20 joint blocks. The largest block remained 57.84–58.80% of all records, the three largest remained 92.07–93.02%, and the inverse-Simpson effective number of blocks remained 2.435–2.504.

Thus, the manuscript's central evidence-geometry limitation—many molecular rows but very few highly concentrated independent source/analogue regimes—is not an artefact of choosing exactly 0.80. The exact identities of smaller components and blocks do change, as quantified in the partition-comparison source data.

## Interpretation boundary

- This result concerns graph and partition geometry only; it does not show that H1 performance is numerically invariant to the threshold.
- The frozen OOF predictions cannot be validly rescored as if they came from the 0.70 or 0.90 partitions because the corresponding training/test boundaries differ.
- A valid threshold-specific performance comparison would require complete nested retraining under separately frozen alternative partitions. Because confirmatory outcomes are already known, such runs would be exploratory and cannot replace the primary 0.80 result.
- The current analysis therefore strengthens the evidence-geometry argument while preserving the original confirmatory boundary.
"""
    (HERE / "BLOCK_DEFINITION_SENSITIVITY_REPORT.md").write_text(report, encoding="utf-8")
    return payload


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    summary, ranked, assignments, comparison, hashes = reconstruct()
    summary.to_csv(SOURCE_DIR / "block_definition_threshold_summary.csv", index=False, float_format="%.12f")
    ranked.to_csv(SOURCE_DIR / "block_definition_ranked_sizes.csv", index=False, float_format="%.12f")
    assignments.to_csv(SOURCE_DIR / "block_definition_record_assignments.csv", index=False)
    comparison.to_csv(SOURCE_DIR / "block_definition_partition_comparison.csv", index=False, float_format="%.12f")
    payload = write_report(summary, comparison, hashes)
    make_figure(summary, ranked)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
