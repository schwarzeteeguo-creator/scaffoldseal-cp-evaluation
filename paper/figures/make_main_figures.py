"""Create publication figures 2-4 from frozen ScaffoldSeal-CP reporting artifacts.

Figure 1 is an author-supplied vector PDF and is intentionally not regenerated
by this script. Its rendered PNG preview and final PDF are maintained under
``paper/figures/output/figure1_study_workflow``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIGDIR = Path(__file__).resolve().parent
OUT = FIGDIR / "output"
SOURCE = FIGDIR / "source_data"

MM = 1 / 25.4
COLORS = {
    "random": "#4C78A8",
    "joint": "#D07A5F",
    "neutral": "#8B95A1",
    "neutral_dark": "#5D6872",
    "classical": "#6F7D8C",
    "d1": "#4F8F75",
    "d2": "#A58B43",
    "d3": "#B85C5C",
    "ink": "#27313A",
    "grid": "#D9DEE3",
    "pale_blue": "#EAF1F7",
    "pale_orange": "#F7ECE8",
    "pale_green": "#E8F2ED",
    "pale_gold": "#F5F0DF",
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
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)


def export(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str = "none",
    fontsize: float = 6.5,
    weight: str = "normal",
    text_color: str | None = None,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color or COLORS["ink"],
        linespacing=1.2,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str | None = None) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color or COLORS["ink"],
            shrinkA=2,
            shrinkB=2,
        )
    )


def study_workflow() -> None:
    """Figure 1: redrawn curation-to-evaluation workflow."""
    fig, ax = plt.subplots(figsize=(183 * MM, 82 * MM))
    ax.set_xlim(0, 18.3)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # Column backgrounds establish the three-stage reading order.
    ax.add_patch(FancyBboxPatch((0.20, 0.35), 5.15, 7.45, boxstyle="round,pad=0.03,rounding_size=0.10", facecolor="#FAFBFC", edgecolor="#D8DEE4", linewidth=0.7))
    ax.add_patch(FancyBboxPatch((5.70, 0.35), 6.15, 7.45, boxstyle="round,pad=0.03,rounding_size=0.10", facecolor="#FAFBFC", edgecolor="#D8DEE4", linewidth=0.7))
    ax.add_patch(FancyBboxPatch((12.20, 0.35), 5.90, 7.45, boxstyle="round,pad=0.03,rounding_size=0.10", facecolor="#FAFBFC", edgecolor="#D8DEE4", linewidth=0.7))

    ax.text(0.38, 7.63, "a", fontsize=9, fontweight="bold", va="top")
    ax.text(0.78, 7.58, "Outcome-blind curation\nand reconciliation", fontsize=7.0, fontweight="bold", va="top", linespacing=1.05)
    ax.text(5.88, 7.63, "b", fontsize=9, fontweight="bold", va="top")
    ax.text(6.28, 7.58, "Dependence graph and\njoint blocks", fontsize=7.4, fontweight="bold", va="top", linespacing=1.05)
    ax.text(12.38, 7.63, "c", fontsize=9, fontweight="bold", va="top")
    ax.text(12.78, 7.58, "Nested leave-one-block-out\nevaluation", fontsize=7.4, fontweight="bold", va="top", linespacing=1.05)

    # a: curation flow.
    rounded_box(ax, 1.22, 5.88, 3.12, 0.92, "Raw PAMPA compilation\n7,298 rows", COLORS["pale_blue"], COLORS["random"], 6.8, "bold")
    arrow(ax, (2.78, 5.83), (2.78, 5.25))
    rounded_box(ax, 0.62, 4.30, 2.45, 0.78, "6,926 uncensored\nusable rows", COLORS["pale"], "#BCC5CD", 6.1, "bold")
    rounded_box(ax, 3.35, 4.30, 1.48, 0.78, "Exclude\n372 censored", COLORS["pale_orange"], COLORS["joint"], 5.9, "bold")
    arrow(ax, (2.78, 4.25), (2.78, 3.75))
    rounded_box(ax, 0.92, 2.82, 3.72, 0.78, "Collapse 31 compatible multi-row\nsource-structure groups by median", COLORS["pale_gold"], COLORS["d2"], 5.75, "bold")
    arrow(ax, (2.78, 2.77), (2.78, 2.29))
    rounded_box(ax, 1.00, 0.78, 3.56, 1.35, "Analytical population\n6,895 source-structure groups\n6,862 unique molecules\n41 sources", COLORS["pale_green"], COLORS["d1"], 5.8, "bold")

    # b: two graph layers merge into joint blocks.
    rounded_box(ax, 6.20, 5.62, 2.18, 1.12, "Peptide-aware\nanalogue graph\n141,425 edges", COLORS["pale_blue"], COLORS["random"], 6.4, "bold")
    rounded_box(ax, 9.12, 5.62, 2.18, 1.12, "Source\nprovenance graph\n41 sources", COLORS["pale_gold"], COLORS["d2"], 6.4, "bold")
    ax.text(7.29, 5.23, "305 transitive\nanalogue components", ha="center", va="top", fontsize=5.8, color=COLORS["neutral_dark"])

    # Miniature bipartite graph.
    comp_xy = [(6.55, 3.95), (7.55, 4.37), (8.20, 3.55), (7.30, 3.20)]
    source_xy = [(9.20, 4.27), (10.23, 3.72), (9.55, 3.12), (10.67, 4.47)]
    links = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (0, 3)]
    for ci, si in links:
        ax.plot([comp_xy[ci][0], source_xy[si][0]], [comp_xy[ci][1], source_xy[si][1]], color="#AAB3BB", linewidth=0.75, zorder=1)
    for x, y in comp_xy:
        ax.add_patch(Circle((x, y), 0.17, facecolor=COLORS["random"], edgecolor="white", linewidth=0.6, zorder=3))
    for x, y in source_xy:
        rounded_box(ax, x - 0.18, y - 0.13, 0.36, 0.26, "", COLORS["d2"], "white")
    ax.text(7.18, 2.68, "analogue components", ha="center", fontsize=5.6, color=COLORS["neutral_dark"])
    ax.text(10.08, 2.68, "sources", ha="center", fontsize=5.6, color=COLORS["neutral_dark"])
    arrow(ax, (8.75, 2.42), (8.75, 1.83))
    rounded_box(ax, 6.98, 0.88, 3.56, 0.92, "18 indivisible connected\njoint blocks", COLORS["pale_orange"], COLORS["joint"], 6.2, "bold")

    # c: nested evaluation and sealed scoring.
    block_x = np.linspace(12.82, 17.50, 18)
    for idx, x in enumerate(block_x):
        is_test = idx == 13
        ax.add_patch(
            FancyBboxPatch(
                (x, 6.05),
                0.19,
                0.64,
                boxstyle="round,pad=0.01,rounding_size=0.025",
                facecolor=COLORS["joint"] if is_test else COLORS["random"],
                edgecolor="none",
                alpha=0.95 if is_test else 0.62,
            )
        )
    ax.text(15.15, 5.76, "17 outer-training blocks    1 unseen test block", ha="center", fontsize=6.0)
    arrow(ax, (15.15, 5.54), (15.15, 4.93))
    rounded_box(ax, 12.82, 3.60, 4.65, 1.18, "Training-only inner evaluation\n4 grouped baskets", COLORS["pale_blue"], COLORS["random"], 7, "bold")
    steps = ["impute/scale", "select", "stop", "calibrate"]
    for idx, step in enumerate(steps):
        x_step = 12.62 + idx * 1.22
        rounded_box(ax, x_step, 2.53, 1.05, 0.57, step, COLORS["pale"], "#BBC4CC", 5.15)
        if idx < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x_step + 1.08, 2.815),
                    (x_step + 1.19, 2.815),
                    arrowstyle="->",
                    mutation_scale=6,
                    linewidth=0.65,
                    color=COLORS["ink"],
                )
            )
    arrow(ax, (15.15, 2.40), (15.15, 1.82))
    rounded_box(ax, 12.73, 0.80, 2.08, 0.87, "Refit on all 17\ntraining blocks", COLORS["pale_green"], COLORS["d1"], 6.2, "bold")
    rounded_box(ax, 15.48, 0.80, 2.08, 0.87, "Seal prediction,\nthen score test block", COLORS["pale_orange"], COLORS["joint"], 6.2, "bold")
    arrow(ax, (14.84, 1.24), (15.44, 1.24))
    ax.text(9.15, 0.12, "Conceptual workflow; outcomes do not define blocks, and icon widths are not to scale", ha="center", va="bottom", fontsize=5.2, color="#737D86")
    export(fig, "figure1_study_workflow")
    plt.close(fig)


def evidence_geometry() -> None:
    """Figure 2: effective evidence geometry of the 18 frozen blocks."""
    frame = pd.read_csv(SOURCE / "figure2_block_geometry.csv").sort_values("n_curated_rows", ascending=False).reset_index(drop=True)
    n = frame["n_curated_rows"].to_numpy(int)
    ranks = np.arange(1, len(frame) + 1)
    cumulative = np.cumsum(n) / n.sum()

    fig = plt.figure(figsize=(183 * MM, 74 * MM), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.05, 1.15], wspace=0.34)
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]

    colors = [COLORS["joint"] if idx < 3 else "#B9C1C8" for idx in range(len(n))]
    axa.bar(ranks, n, width=0.78, color=colors, edgecolor="none")
    axa.set_yscale("log")
    axa.set_xlim(0.2, 18.8)
    axa.set_xticks([1, 3, 6, 9, 12, 15, 18])
    axa.set_xlabel("Joint blocks ranked by record count")
    axa.set_ylabel("Records per block (log scale)")
    axa.grid(axis="y", color=COLORS["grid"], linewidth=0.5, which="major")
    axa.set_axisbelow(True)
    for x, value in zip(ranks[:3], n[:3]):
        axa.text(x, value * 1.18, f"{value:,}", ha="center", va="bottom", fontsize=6.2, fontweight="bold")
    axa.text(11.0, 1450, "remaining 15 blocks", ha="center", fontsize=6.0, color=COLORS["neutral_dark"])
    axa.set_title("Most records occupy three blocks", loc="left", fontweight="bold")
    panel_label(axa, "a")

    axb.plot(ranks, cumulative * 100, color=COLORS["random"], lw=1.7, marker="o", ms=3.0)
    axb.fill_between(ranks, 0, cumulative * 100, color=COLORS["pale_blue"], alpha=0.8)
    axb.axhline(100, color=COLORS["neutral"], lw=0.7, ls="--")
    axb.scatter([1, 3], [cumulative[0] * 100, cumulative[2] * 100], color=COLORS["joint"], s=24, zorder=4)
    axb.annotate("largest\n58.16%", (1, cumulative[0] * 100), xytext=(3.2, 49), fontsize=6.0, ha="center", arrowprops=dict(arrowstyle="-", color=COLORS["neutral_dark"], lw=0.6))
    axb.annotate("top three\n92.39%", (3, cumulative[2] * 100), xytext=(7.0, 79), fontsize=6.0, ha="center", arrowprops=dict(arrowstyle="-", color=COLORS["neutral_dark"], lw=0.6))
    axb.set_xlim(0.5, 18.5)
    axb.set_ylim(0, 104)
    axb.set_xticks([1, 3, 6, 9, 12, 15, 18])
    axb.set_yticks([0, 25, 50, 75, 100])
    axb.set_xlabel("Number of ranked blocks included")
    axb.set_ylabel("Cumulative record share (%)")
    axb.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    axb.set_axisbelow(True)
    axb.set_title("Evidence is highly concentrated", loc="left", fontweight="bold")
    panel_label(axb, "b")

    components = frame["n_analogue_components"].to_numpy(int)
    sources = frame["n_sources"].to_numpy(int)
    sizes = 22 + 145 * np.sqrt(n / n.max())
    colors_c = [COLORS["joint"] if idx < 3 else COLORS["random"] for idx in range(len(n))]
    axc.scatter(components, sources, s=sizes, c=colors_c, alpha=0.78, edgecolor="white", linewidth=0.6)
    axc.set_xscale("log")
    axc.set_xlim(0.8, 230)
    axc.set_ylim(0.3, 22.5)
    axc.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    axc.set_yticks([1, 5, 10, 15, 20])
    axc.set_xlabel("Analogue components per block (log scale)")
    axc.set_ylabel("Sources per block")
    axc.grid(color=COLORS["grid"], linewidth=0.5)
    axc.set_axisbelow(True)
    label_offsets = [(18, -4), (10, 35), (-12, 22)]
    for idx, offset in zip(range(3), label_offsets):
        axc.annotate(
            f"n={n[idx]:,}",
            (components[idx], sources[idx]),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.8,
            ha="left" if offset[0] > 0 else "right",
            arrowprops=dict(arrowstyle="-", color=COLORS["neutral_dark"], lw=0.55),
        )
    axc.text(0.98, 0.03, "point area = records", transform=axc.transAxes, ha="right", va="bottom", fontsize=5.6, color=COLORS["neutral_dark"])
    axc.set_title("Connectivity creates unequal regimes", loc="left", fontweight="bold")
    panel_label(axc, "c")

    export(fig, "figure2_evidence_geometry")
    plt.close(fig)


def main_results() -> None:
    """Figure 3: H1, H2, and D3 calibration results."""
    table = pd.read_csv(ROOT / "paper" / "tables" / "main_results.csv")
    coverage = json.loads((ROOT / "scaffoldseal" / "artifacts" / "d3_interval_coverage" / "coverage_summary.json").read_text(encoding="utf-8"))
    SOURCE.mkdir(parents=True, exist_ok=True)

    h1 = table.loc[table["analysis"].eq("H1")].copy()
    h2 = table.loc[table["analysis"].eq("H2")].copy()
    cov = pd.DataFrame(coverage["overall"])
    h1.to_csv(SOURCE / "figure3a_h1.csv", index=False)
    h2.to_csv(SOURCE / "figure3b_h2_models.csv", index=False)
    cov.to_csv(SOURCE / "figure3c_coverage.csv", index=False)

    fig = plt.figure(figsize=(183 * MM, 76 * MM), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.35, 1.15], wspace=0.28)
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]

    values = [
        float(h1.loc[h1["model_or_comparison"].eq("molecule-random D0"), "value"].iloc[0]),
        float(h1.loc[h1["model_or_comparison"].eq("joint-block D0"), "value"].iloc[0]),
    ]
    axa.bar([0, 1], values, width=0.62, color=[COLORS["random"], COLORS["joint"]], edgecolor="none")
    axa.set_xticks([0, 1], ["Molecule-\nrandom", "Joint source/\nanalogue block"])
    axa.set_ylabel("Source-macro MAE\n(log$_{10}$ Papp)")
    axa.set_ylim(0, 1.18)
    axa.grid(axis="y", color=COLORS["grid"], linewidth=0.5, zorder=0)
    axa.set_axisbelow(True)
    for x, value in enumerate(values):
        axa.text(x, value + 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    axa.plot([0, 1], [1.105, 1.105], color=COLORS["ink"], lw=0.8, clip_on=False)
    axa.plot([0, 0], [1.085, 1.125], color=COLORS["ink"], lw=0.8, clip_on=False)
    axa.plot([1, 1], [1.085, 1.125], color=COLORS["ink"], lw=0.8, clip_on=False)
    axa.text(0.5, 1.13, "gap = 0.537 (95% CI 0.324-0.753)", ha="center", va="bottom", fontsize=6.0)
    axa.set_title("Evaluation estimates diverge", loc="left", fontweight="bold")
    panel_label(axa, "a")

    order = ["D0", "nested-selected classical", "D1", "D2", "D3"]
    labels = ["D0", "Classical", "D1", "D2", "D3\n(fixed)"]
    cmap = [COLORS["neutral"], COLORS["classical"], COLORS["d1"], COLORS["d2"], COLORS["d3"]]
    values_b = [float(h2.loc[h2["model_or_comparison"].eq(name), "value"].iloc[0]) for name in order]
    y = np.arange(len(order))
    axb.axvline(values_b[0], color=COLORS["neutral"], lw=0.8, ls="--", zorder=1)
    for yi, value, color in zip(y, values_b, cmap):
        axb.scatter(value, yi, s=34, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        axb.text(value + 0.004, yi, f"{value:.3f}", ha="left", va="center", fontsize=6.2)
    axb.set_yticks(y, labels)
    axb.set_xlim(0.88, 1.03)
    axb.set_ylim(-0.65, len(order) - 0.35)
    axb.invert_yaxis()
    axb.set_xlabel("Source-macro MAE\n(log$_{10}$ Papp; lower is better)")
    axb.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
    axb.set_axisbelow(True)
    axb.text(0.998, 4.43, "H2 failed", ha="center", va="center", color=COLORS["d3"], fontsize=6.2, fontweight="bold")
    axb.set_title("Fixed D3 does not close\nthe joint-block gap", loc="left", fontweight="bold", fontsize=7.5)
    panel_label(axb, "b")

    nominal = cov["nominal_coverage"].to_numpy(float)
    observed = cov["coverage"].to_numpy(float)
    axc.plot([0.4, 1.0], [0.4, 1.0], ls="--", lw=0.9, color=COLORS["neutral"], label="Ideal")
    axc.plot(nominal, observed, marker="o", ms=5, lw=1.5, color=COLORS["d3"], label="D3")
    axc.plot([0.9, 0.9], [0.85, 0.95], color=COLORS["d1"], lw=8, alpha=0.20, solid_capstyle="butt")
    for index, (xn, yo) in enumerate(zip(nominal, observed)):
        offset = 0.025 if index == 0 else -0.035
        valign = "bottom" if index == 0 else "top"
        axc.text(xn, yo + offset, f"{yo:.2f}", ha="center", va=valign, fontsize=6.5)
    axc.set_xlim(0.40, 1.00)
    axc.set_ylim(0.40, 1.00)
    axc.set_xticks([0.5, 0.8, 0.9, 1.0])
    axc.set_yticks([0.4, 0.6, 0.8, 1.0])
    axc.set_xlabel("Nominal coverage")
    axc.set_ylabel("Observed coverage")
    axc.grid(color=COLORS["grid"], linewidth=0.5)
    axc.legend(loc="upper left", handlelength=1.5)
    axc.text(0.93, 0.90, "required\n85-95%", ha="left", va="center", fontsize=5.8, color="#4F7667")
    axc.set_title("D3 intervals are\nunder-covered", loc="left", fontweight="bold", fontsize=7.5)
    panel_label(axc, "c")

    export(fig, "figure3_main_results")
    plt.close(fig)


def failure_heterogeneity() -> None:
    """Figure 4: seed stability and source/block heterogeneity of D3 failure."""
    seed = pd.read_csv(SOURCE / "figure4a_seed_metrics.csv")
    sources = pd.read_csv(SOURCE / "figure4b_source_deltas.csv").sort_values("d3_minus_d1").reset_index(drop=True)
    blocks = pd.read_csv(SOURCE / "figure4c_block_deltas.csv")

    fig = plt.figure(figsize=(183 * MM, 80 * MM), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.45, 1.2], wspace=0.32)
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for variant, color, marker in [("D1", COLORS["d1"], "o"), ("D2", COLORS["d2"], "s"), ("D3", COLORS["d3"], "^")]:
        part = seed.loc[seed["variant"].eq(variant)].sort_values("seed_index")
        x = part["seed_index"].to_numpy(int) + 1
        y = part["source_macro_mae"].to_numpy(float)
        axa.plot(x, y, color=color, marker=marker, ms=4.2, lw=1.4, label=variant)
    axa.set_xticks([1, 2, 3, 4, 5])
    axa.set_ylim(0.82, 1.08)
    axa.set_xlabel("Scheduled seed")
    axa.set_ylabel("Source-macro MAE\n(log$_{10}$ Papp)")
    axa.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    axa.set_axisbelow(True)
    axa.legend(loc="upper left", ncol=3, handlelength=1.1, columnspacing=0.8)
    axa.text(0.98, 0.06, "D2 and D3 worse\nthan D1 at every seed", transform=axa.transAxes, ha="right", va="bottom", fontsize=5.8, color=COLORS["d3"], fontweight="bold")
    axa.set_title("Failure is stable across seeds", loc="left", fontweight="bold")
    panel_label(axa, "a")

    delta = sources["d3_minus_d1"].to_numpy(float)
    rank = np.arange(1, len(delta) + 1)
    colors_b = np.where(delta > 0, COLORS["d3"], COLORS["d1"])
    axb.vlines(rank, 0, delta, color=colors_b, linewidth=1.2, alpha=0.72)
    axb.scatter(rank, delta, color=colors_b, s=12, edgecolor="white", linewidth=0.35, zorder=3)
    axb.axhline(0, color=COLORS["neutral_dark"], lw=0.8)
    axb.set_xlim(0, len(delta) + 1)
    axb.set_ylim(min(-0.42, delta.min() - 0.10), delta.max() + 0.38)
    axb.set_xticks([1, 10, 20, 30, 41])
    axb.set_xlabel("Sources ranked by D3-D1 change")
    axb.set_ylabel("D3-D1 MAE (log$_{10}$ Papp)")
    axb.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    axb.set_axisbelow(True)
    top = sources.nlargest(5, "d3_minus_d1")
    top_lines = ["Largest degradations"] + [f"{row.source}: +{row.d3_minus_d1:.3f}" for row in top.itertuples()]
    axb.text(
        0.03,
        0.96,
        "\n".join(top_lines),
        transform=axb.transAxes,
        va="top",
        ha="left",
        fontsize=5.2,
        linespacing=1.18,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D5DBE0", linewidth=0.5, alpha=0.94),
    )
    axb.text(2.0, 0.035, "D3 better", va="bottom", fontsize=5.4, color=COLORS["d1"])
    axb.text(40.5, 0.035, "D3 worse", va="bottom", ha="right", fontsize=5.4, color=COLORS["d3"])
    axb.set_title("Degradation clusters by source", loc="left", fontweight="bold")
    panel_label(axb, "b")

    delta_c = blocks["d3_minus_d1"].to_numpy(float)
    n = blocks["n"].to_numpy(int)
    colors_c = np.where(delta_c > 0, COLORS["d3"], COLORS["d1"])
    sizes_c = 24 + 75 * np.sqrt(n / n.max())
    axc.scatter(n, delta_c, s=sizes_c, c=colors_c, alpha=0.80, edgecolor="white", linewidth=0.55)
    axc.axhline(0, color=COLORS["neutral_dark"], lw=0.8)
    axc.set_xscale("log")
    axc.set_xlim(2, 6500)
    axc.set_ylim(min(-0.65, delta_c.min() - 0.10), max(0.90, delta_c.max() + 0.08))
    axc.set_xticks([3, 10, 30, 100, 300, 1000, 4000], ["3", "10", "30", "100", "300", "1k", "4k"])
    axc.set_xlabel("Outer-block records (log scale)")
    axc.set_ylabel("D3-D1 MAE (log$_{10}$ Papp)")
    axc.grid(color=COLORS["grid"], linewidth=0.5)
    axc.set_axisbelow(True)
    callouts = {
        3: ("fold 3\nn=1,518", (-65, 20)),
        7: ("fold 7\nn=842", (-15, 37)),
        5: ("fold 5\nn=4,010", (-60, -26)),
    }
    for fold, (label, offset) in callouts.items():
        row = blocks.loc[blocks["outer_fold"].eq(fold)].iloc[0]
        axc.annotate(
            label,
            (row["n"], row["d3_minus_d1"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.6,
            ha="left",
            arrowprops=dict(arrowstyle="-", color=COLORS["neutral_dark"], lw=0.5),
        )
    axc.set_title("Large blocks fail differently", loc="left", fontweight="bold")
    panel_label(axc, "c")

    export(fig, "figure4_failure_heterogeneity")
    plt.close(fig)


if __name__ == "__main__":
    # Figure 1 is supplied as a reviewed vector asset; do not overwrite it.
    evidence_geometry()
    main_results()
    failure_heterogeneity()
