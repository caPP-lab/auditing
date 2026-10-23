#!/usr/bin/env python3
"""Render the final-paper Kenya biodiversity mislabel-rate comparison figure."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd


DEFAULT_RESULTS = PROJECT_ROOT / "results" / "kenya_mislabel"
DEFAULT_COMPARISON = (
    PROJECT_ROOT / "results" / "mislabel_method_comparison" / "kenya"
)
DEFAULT_CONFUSION = (
    PROJECT_ROOT
    / "results"
    / "figure_data"
    / "kenya_compact_confusion_matrix.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "figures"

PAS_COLOR = "#D55E00"
SRS_COLOR = "#222222"
PPR_COLOR = "#0072B2"
WALD_COLOR = "#7A5195"
SERFLING_COLOR = "#5C6F82"
TARGET_COLOR = "#767676"


def _style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 9.6,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 7.7,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "figure.dpi": 180,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _summarize(checkpoints: pd.DataFrame) -> pd.DataFrame:
    return (
        checkpoints.groupby(["policy_id", "policy", "audit"], as_index=False)
        .half_width.agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_values(["policy_id", "audit"])
    )


def _plot_compact_confusion(axis: plt.Axes, counts: np.ndarray):
    blues = matplotlib.colormaps["Blues"]
    cmap = LinearSegmentedColormap.from_list(
        "compact_dark_blues", blues(np.linspace(0.30, 1.0, 256))
    )
    cmap.set_bad("white")
    image = axis.imshow(
        np.ma.masked_equal(counts, 0),
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=max(2, int(counts.max()))),
        aspect="equal",
        interpolation="nearest",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.7}
    axis.text(
        0.50,
        0.012,
        "Verified species",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.3,
        bbox=label_box,
    )
    axis.set_ylabel("Submitted species", fontsize=9.3, labelpad=2)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#333333")
    return image


def create_figure(
    results: Path,
    comparison: Path,
    confusion: Path,
    output: Path,
) -> None:
    verification = results / "VERIFICATION.json"
    if not verification.is_file():
        raise RuntimeError("run and aggregate the mislabel study before reporting")
    _style()
    checkpoints = pd.read_csv(results / "checkpoints.csv.gz")
    serfling = pd.read_csv(results / "serfling.csv.gz")
    ppr = pd.read_csv(comparison / "ppr_summary.csv")
    wald = pd.read_csv(comparison / "wald.csv")
    targets = pd.read_csv(comparison / "method_target_summary.csv")
    confusion_counts = pd.read_csv(confusion, index_col=0).to_numpy(dtype=int)
    summary = _summarize(checkpoints)
    source = output / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    summary.to_csv(source / "kenya_mislabel_betting_cs_summary.csv", index=False)
    serfling.to_csv(source / "kenya_mislabel_serfling.csv", index=False)
    ppr.to_csv(source / "kenya_mislabel_ramdas_ppr_summary.csv", index=False)
    wald.to_csv(source / "kenya_mislabel_wald.csv", index=False)
    targets.to_csv(source / "kenya_mislabel_two_point_targets.csv", index=False)
    pd.DataFrame(confusion_counts).to_csv(
        source / "kenya_compact_confusion_matrix.csv", index=False
    )

    figure = plt.figure(figsize=(7.05, 2.75))
    matrix_axis = figure.add_axes([0.030, 0.018, 0.375, 0.964])
    colorbar_axis = figure.add_axes([0.411, 0.045, 0.014, 0.910])
    axis = figure.add_axes([0.505, 0.17, 0.485, 0.74])
    axes = (matrix_axis, axis)
    image = _plot_compact_confusion(matrix_axis, confusion_counts)
    styles = {
        "srs": {"color": SRS_COLOR, "linestyle": ":", "label": "SRS"},
        "pas": {"color": PAS_COLOR, "linestyle": "-", "label": "PAS"},
    }
    for policy in ("srs", "pas"):
        part = summary[(summary.policy_id == policy) & (summary.audit >= 100)]
        style = styles[policy]
        x = part.audit.to_numpy(dtype=float)
        center = 100.0 * part["median"].to_numpy(dtype=float)
        lower = 100.0 * part.q25.to_numpy(dtype=float)
        upper = 100.0 * part.q75.to_numpy(dtype=float)
        axis.fill_between(x, lower, upper, color=style["color"], alpha=0.11, linewidth=0)
        axis.plot(
            x,
            center,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
            dash_capstyle="round",
        )
    ppr_plot = ppr[(ppr.audit >= 100) & (ppr.audit <= 5_000)]
    axis.fill_between(
        ppr_plot.audit.to_numpy(dtype=float),
        100.0 * ppr_plot.q25.to_numpy(dtype=float),
        100.0 * ppr_plot.q75.to_numpy(dtype=float),
        color=PPR_COLOR,
        alpha=0.08,
        linewidth=0,
    )
    axis.plot(
        ppr_plot.audit,
        100.0 * ppr_plot["median"],
        color=PPR_COLOR,
        linestyle=(0, (5, 1.8)),
        label="PPR",
    )
    wald_plot = wald[(wald.audit >= 100) & (wald.audit <= 5_000)]
    axis.plot(
        wald_plot.audit,
        100.0 * wald_plot.half_width,
        color=WALD_COLOR,
        linestyle="-.",
        label="Wald",
    )
    serfling_plot = serfling[(serfling.audit >= 100) & (serfling.audit <= 5_000)]
    axis.plot(
        serfling_plot.audit,
        100.0 * serfling_plot.half_width,
        color=SERFLING_COLOR,
        linestyle="--",
        label="Serfling",
    )
    axis.axhline(2.0, color=TARGET_COLOR, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    axis.text(
        4_970,
        2.12,
        r"target: $\pm2$ points",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=TARGET_COLOR,
    )

    colors = {
        "PAS": PAS_COLOR,
        "SRS": SRS_COLOR,
        "Ramdas PPR": PPR_COLOR,
        "Wald": WALD_COLOR,
        "Serfling": SERFLING_COLOR,
    }
    offsets = {
        "PAS": (-9, 8),
        "SRS": (10, -13),
        "Wald": (0, 8),
        "Ramdas PPR": (-28, 8),
        "Serfling": (28, 8),
    }
    for row in targets.itertuples():
        x = float(row.median_reviews)
        color = colors[row.method]
        axis.scatter([x], [2.0], s=18, color=color, edgecolors="white", linewidths=0.45, zorder=5)
        label = f"{int(math.floor(x + 0.5)):,}"
        axis.annotate(
            label,
            xy=(x, 2.0),
            xytext=offsets[row.method],
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.1,
            color=color,
            fontweight="semibold",
        )

    axis.set_xlim(0, 5_050)
    axis.set_ylim(0, 7.0)
    axis.set_xticks(np.arange(0, 5_001, 1_000))
    axis.set_yticks(np.arange(0, 7.1, 1.0))
    axis.set_xlabel("Audits")
    axis.set_ylabel("Mislabel rate CS half-width (%)", labelpad=2.0)
    axis.set_title("Mislabel rate certification", fontsize=9.3, pad=3.0)
    axis.grid(axis="y", color="#D7DBE0", linewidth=0.45, linestyle="--", dashes=(2, 2))
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper right",
        ncol=5,
        columnspacing=0.75,
        handlelength=1.8,
        fontsize=7.2,
        borderaxespad=0.2,
    )
    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.ax.tick_params(labelsize=7.4, width=0.5, length=2)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "supplementary code Kenya biodiversity mislabel-CS pipeline"}
    figure.savefig(
        output / "kenya_mislabel_rate_cs_comparison.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata=metadata,
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--confusion", type=Path, default=DEFAULT_CONFUSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_figure(args.results, args.comparison, args.confusion, args.output)
