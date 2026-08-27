"""Create the ACS Table-of-Contents graphic for the v0.7 submission draft.

The graphic uses only the two completed D0 source-macro MAE values reported in
the manuscript. It is schematic and does not display chemical structures or
unreleased row-level data.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent / "output"
BASE = OUT / "toc_graphic_evaluation_boundary"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.7,
    }
)

INK = "#173F62"
MUTED = "#5B6775"
PALE = "#F4F7FA"
TEAL = "#177E89"
LAVENDER = "#8D7BB8"
ORANGE = "#E28B3B"
RED = "#B64B4B"
GREEN = "#45845C"


def node(ax, x, y, color, radius=0.045):
    ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="white", linewidth=0.9, zorder=3))


def curved_link(ax, start, end, color, rad=0.0, width=1.1, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-",
            connectionstyle=f"arc3,rad={rad}",
            linewidth=width,
            linestyle=style,
            color=color,
            alpha=0.82,
            zorder=1,
        )
    )


def draw_dependency_cartoon(ax):
    ax.add_patch(FancyBboxPatch((0.04, 0.25), 0.66, 1.13, boxstyle="round,pad=0.025,rounding_size=0.06", facecolor=PALE, edgecolor="#D6E0E8", linewidth=0.8))
    ax.text(0.37, 1.28, "sources +\nanalogues", ha="center", va="center", fontsize=6.1, color=INK, weight="bold", linespacing=0.9)
    points = [(0.19, 0.92, TEAL), (0.33, 1.08, LAVENDER), (0.49, 0.95, TEAL), (0.25, 0.55, LAVENDER), (0.46, 0.55, TEAL), (0.59, 0.74, LAVENDER)]
    for a, b, c in [(0, 1, MUTED), (1, 2, MUTED), (0, 3, MUTED), (3, 4, MUTED), (4, 5, MUTED), (2, 5, MUTED), (1, 5, LAVENDER)]:
        curved_link(ax, points[a][:2], points[b][:2], c, rad=0.12 if a == 1 and b == 5 else 0)
    for x, y, color in points:
        node(ax, x, y, color)


def draw_model(ax):
    ax.add_patch(FancyBboxPatch((0.82, 0.57), 0.48, 0.48, boxstyle="round,pad=0.03,rounding_size=0.07", facecolor="white", edgecolor=INK, linewidth=1.1))
    ax.text(1.06, 0.80, "same\nDMPNN", ha="center", va="center", fontsize=7.4, color=INK, weight="bold", linespacing=1.05)
    ax.add_patch(FancyArrowPatch((0.72, 0.82), (0.81, 0.82), arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color=INK))
    ax.add_patch(FancyArrowPatch((1.31, 0.82), (1.41, 0.82), arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color=INK))


def draw_result(ax, x, label, value, color, crossed):
    ax.add_patch(FancyBboxPatch((x, 0.25), 0.78, 1.13, boxstyle="round,pad=0.025,rounding_size=0.06", facecolor="white", edgecolor="#D6E0E8", linewidth=0.8))
    ax.text(x + 0.39, 1.28, label, ha="center", va="center", fontsize=6.5, color=INK, weight="bold")
    if crossed:
        ax.add_patch(Rectangle((x + 0.11, 0.75), 0.24, 0.27, facecolor="#FFF1E6", edgecolor=ORANGE, linewidth=0.8))
        ax.add_patch(Rectangle((x + 0.43, 0.75), 0.24, 0.27, facecolor="#F3EEF9", edgecolor=LAVENDER, linewidth=0.8))
        for s, e, c, r in [((x + 0.18, 0.94), (x + 0.55, 0.85), TEAL, 0.15), ((x + 0.27, 0.81), (x + 0.50, 0.96), LAVENDER, -0.18)]:
            curved_link(ax, s, e, c, r, 1.2)
        for px, py, pc in [(x + 0.18, 0.94, TEAL), (x + 0.27, 0.81, ORANGE), (x + 0.55, 0.85, LAVENDER), (x + 0.50, 0.96, TEAL)]:
            node(ax, px, py, pc, 0.032)
        ax.text(x + 0.39, 0.66, "crossing\nlinks", ha="center", va="center", fontsize=5.1, color=RED, linespacing=0.9)
    else:
        for dx, dy, c in [(0.15, 0.94, TEAL), (0.27, 0.82, LAVENDER), (0.52, 0.94, TEAL), (0.64, 0.82, LAVENDER)]:
            node(ax, x + dx, dy, c, 0.035)
        curved_link(ax, (x + 0.15, 0.94), (x + 0.27, 0.82), MUTED, 0.08)
        curved_link(ax, (x + 0.52, 0.94), (x + 0.64, 0.82), MUTED, -0.08)
        ax.add_patch(Rectangle((x + 0.08, 0.74), 0.27, 0.30, fill=False, edgecolor=GREEN, linewidth=0.9))
        ax.add_patch(Rectangle((x + 0.44, 0.74), 0.27, 0.30, fill=False, edgecolor=GREEN, linewidth=0.9))
        ax.text(x + 0.39, 0.66, "blocked\nlinks", ha="center", va="center", fontsize=5.1, color=GREEN, linespacing=0.9)
    ax.text(x + 0.39, 0.45, f"MAE  {value:.2f}", ha="center", va="center", fontsize=10.2, color=color, weight="bold")


def make_figure():
    fig, ax = plt.subplots(figsize=(3.25, 1.75), dpi=600)
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 3.25)
    ax.set_ylim(0, 1.75)
    ax.axis("off")
    draw_dependency_cartoon(ax)
    draw_model(ax)
    draw_result(ax, 1.47, "molecule-random", 0.467, ORANGE, crossed=True)
    draw_result(ax, 2.40, "joint blocks", 1.005, TEAL, crossed=False)
    ax.text(1.72, 0.10, "Evaluation boundary changes the estimate", ha="center", va="center", fontsize=6.5, color=INK, weight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(BASE.with_suffix(".pdf"))
    fig.savefig(BASE.with_suffix(".svg"))
    fig.savefig(BASE.with_suffix(".png"), dpi=600)
    fig.savefig(BASE.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
