#!/usr/bin/env python3
"""Create final-paper Kenya biodiversity figures for the optimized entropy study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESULTS = PROJECT_ROOT / "results" / "kenya_entropy"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures"

SRS_COLOR = "#222222"
PAS_COLOR = "#D55E00"


def _style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "lines.linewidth": 1.7,
            "figure.dpi": 180,
            "savefig.dpi": 400,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _axis_style(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D7DBE0", linewidth=0.45, linestyle="--", dashes=(2, 2))
    axis.set_axisbelow(True)
    axis.margins(x=0.015)


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "supplementary code optimized entropy figure pipeline"}
    figure.savefig(
        output / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata=metadata,
    )
    plt.close(figure)


def _relative_width_summary(checkpoints: pd.DataFrame) -> pd.DataFrame:
    data = checkpoints.assign(
        relative_half_width_pct=100.0 * checkpoints.relative_error_bound
    )
    return (
        data.groupby(["policy_id", "policy", "audit"], as_index=False)
        .relative_half_width_pct.agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_values(["policy_id", "audit"])
    )


def _plot_width(
    axis: plt.Axes,
    summary: pd.DataFrame,
    *,
    target_fontsize: float = 6.1,
) -> None:
    styles = {
        "srs": {"color": SRS_COLOR, "linestyle": ":", "label": "SRS"},
        "pas": {"color": PAS_COLOR, "linestyle": "-", "label": "PAS"},
    }
    for policy in ("srs", "pas"):
        part = summary[summary.policy_id == policy]
        style = styles[policy]
        x = part.audit.to_numpy(dtype=float)
        median = part["median"].to_numpy(dtype=float)
        lower = part.q25.to_numpy(dtype=float)
        upper = part.q75.to_numpy(dtype=float)
        band_alpha = 0.16 if policy == "srs" else 0.23
        axis.fill_between(
            x,
            lower,
            upper,
            color=style["color"],
            alpha=band_alpha,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x,
            median,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
            marker="o",
            markersize=2.5,
            markeredgewidth=0,
            markevery=3,
            dash_capstyle="round",
            zorder=2,
        )
    axis.set_yscale("log")
    axis.axhline(
        5.0,
        color="#6B7280",
        linewidth=0.75,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    axis.annotate(
        "5% target",
        xy=(1.0, 5.0),
        xycoords=("axes fraction", "data"),
        xytext=(-3, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="#5F6368",
        fontsize=target_fontsize,
    )
    axis.set_xlabel("Audits")
    axis.set_ylabel("Relative CS half-width (%)")
    _axis_style(axis)


def _add_pillar_title(axis: plt.Axes, *, fontsize: float = 8.3) -> None:
    axis.set_title("Pillar 2 diversity", fontsize=fontsize, pad=3.0)


def create_figures(results: Path, output: Path) -> None:
    _style()
    source = output / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    checkpoints = pd.read_csv(results / "checkpoints.csv.gz")
    summary = _relative_width_summary(checkpoints)
    summary.to_csv(source / "kenya_relative_width_summary.csv", index=False)

    figure, axis = plt.subplots(figsize=(2.55, 2.55), constrained_layout=True)
    _plot_width(axis, summary, target_fontsize=8.2)
    _add_pillar_title(axis, fontsize=10.5)
    axis.set_xticks([500, 1_500, 2_500, 3_000])
    axis.tick_params(labelsize=9.0)
    axis.set_xlabel("Audits", fontsize=10.5)
    axis.set_ylabel("Relative CS half-width (%)", fontsize=10.5)
    axis.legend(loc="upper right", handlelength=2.0, fontsize=9.0)
    _save(figure, output, "phase4_kenya_width_trajectories")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    create_figures(arguments.results, arguments.output)
