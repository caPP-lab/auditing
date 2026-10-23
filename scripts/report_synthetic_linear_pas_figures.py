#!/usr/bin/env python3
"""Create final-paper figures for the revised favorable synthetic study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "synthetic"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures"
SRS_COLOR = "#222222"
PAS_COLOR = "#D55E00"
GRID_COLOR = "#D7DBE0"


def _style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.3,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "figure.dpi": 180,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    metadata = {"Creator": "supplementary code synthetic direct-CS pipeline"}
    figure.savefig(
        output / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata=metadata,
    )
    plt.close(figure)


def _summary(frame: pd.DataFrame, value: str, groups: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(groups, as_index=False)[value]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_values(groups)
    )


def _axis_style(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.45, linestyle="--", dashes=(2, 2))
    axis.set_axisbelow(True)


def _line_band(
    axis: plt.Axes,
    part: pd.DataFrame,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    x = part.audit.to_numpy(dtype=float)
    axis.fill_between(
        x,
        part.q25.to_numpy(dtype=float),
        part.q75.to_numpy(dtype=float),
        color=color,
        alpha=0.12,
        linewidth=0,
    )
    axis.plot(x, part["median"], color=color, linestyle=linestyle, label=label)


def _prepared(results: Path, output: Path) -> dict[str, object]:
    verification = json.loads((results / "VERIFICATION.json").read_text(encoding="utf-8"))
    if verification.get("status") != "pass":
        raise RuntimeError("synthetic study validation did not pass")
    checkpoints = pd.read_csv(results / "confirmation" / "checkpoints.csv.gz")
    development = pd.read_csv(results / "development" / "paired_terminal.csv")
    artifact = json.loads((results / "CONFIRMATION_POPULATION.json").read_text(encoding="utf-8"))
    widths = _summary(checkpoints, "width", ["policy_id", "audit"])
    scaling = (
        development.groupby("rows", as_index=False)
        .relative_reduction_percent.agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_values("rows")
    )
    scaling["theoretical_variance_ratio"] = scaling.rows / int(artifact["active_rows"])
    scaling["theoretical_width_reduction_percent"] = 100.0 * (
        1.0 - np.sqrt(int(artifact["active_rows"]) / scaling.rows)
    )
    source = output / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    widths.to_csv(source / "phase5_new_synthetic_width_summary.csv", index=False)
    scaling.to_csv(source / "phase5_new_synthetic_scaling_summary.csv", index=False)
    pd.DataFrame(
        artifact["confusion"],
        index=artifact["row_labels"],
        columns=artifact["category_labels"],
    ).to_csv(source / "phase5_new_synthetic_confusion_matrix.csv")
    return {
        "verification": verification,
        "checkpoints": checkpoints,
        "artifact": artifact,
        "widths": widths,
        "scaling": scaling,
    }


def _plot_width_only(axis: plt.Axes, prepared: dict[str, object]) -> None:
    widths = prepared["widths"]
    for policy, color, label, style in (
        ("srs", SRS_COLOR, "SRS", ":"),
        ("pas", PAS_COLOR, "PAS", "-"),
    ):
        _line_band(
            axis,
            widths[widths.policy_id == policy],
            color=color,
            label=label,
            linestyle=style,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Audits")
    axis.set_ylabel("Confidence-sequence width")
    _axis_style(axis)


def _plot_scaling(axis: plt.Axes, prepared: dict[str, object]) -> None:
    scaling = prepared["scaling"]
    axis.fill_between(
        scaling.rows,
        scaling.q25,
        scaling.q75,
        color=PAS_COLOR,
        alpha=0.12,
        linewidth=0,
    )
    axis.plot(
        scaling.rows,
        scaling["median"],
        color=PAS_COLOR,
        marker="o",
        markersize=3.0,
        label="Observed median",
    )
    axis.plot(
        scaling.rows,
        scaling.theoretical_width_reduction_percent,
        color=SRS_COLOR,
        linestyle=":",
        label=r"$1-\sqrt{k/R}$ reference",
    )
    axis.set_xlabel("Total rows $R$ (four variable-risk rows)")
    axis.set_ylabel("Terminal width reduction vs. SRS (%)")
    axis.set_ylim(20.0, 85.0)
    axis.legend(loc="lower right", handlelength=2.2)
    _axis_style(axis)


def _plot_confusion(
    axis: plt.Axes, prepared: dict[str, object], *, annotate: bool = True
) -> None:
    artifact = prepared["artifact"]
    counts = np.asarray(artifact["confusion"], dtype=int)
    active = set(int(value) for value in artifact["influential_indices"])
    transformed = np.ma.masked_where(counts == 0, np.log10(counts + 1))
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("white")
    image = axis.imshow(transformed, cmap=cmap, aspect="equal")
    if annotate:
        for row, column in zip(*np.nonzero(counts), strict=True):
            value = counts[row, column]
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=4.8,
                color="white" if value >= counts.max() / 2 else "black",
            )
    labels = [f"R{index + 1:02d}{'*' if index in active else ''}" for index in range(counts.shape[0])]
    axis.set_xticks(range(counts.shape[1]), [f"V{i+1:02d}" for i in range(counts.shape[1])], rotation=55, ha="right")
    axis.set_yticks(range(counts.shape[0]), labels)
    axis.set_xlabel("Verified label")
    axis.set_ylabel("Submitted-label row")
    axis.tick_params(length=0, labelsize=5.2)
    if annotate:
        axis.text(
            0.99,
            1.015,
            "* variable-risk row",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.0,
            color="#555555",
        )
    return image


def create_figures(results: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _style()
    prepared = _prepared(results, output)

    figure = plt.figure(figsize=(7.05, 4.75))
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.15, 0.8),
        left=0.07,
        right=0.985,
        bottom=0.09,
        top=0.91,
        hspace=0.36,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=(0.88, 1.50), wspace=0.32)
    confusion_axis = figure.add_subplot(top[0, 0])
    width_axis = figure.add_subplot(top[0, 1])
    _plot_confusion(confusion_axis, prepared, annotate=False)
    confusion_axis.set_title("Synthetic confusion matrix", pad=4.0)
    _plot_width_only(width_axis, prepared)
    width_axis.set_title("Certified width contraction", pad=4.0)
    scaling_axis = figure.add_subplot(outer[1, 0])
    _plot_scaling(scaling_axis, prepared)
    scaling_axis.set_title("Priority advantage scales with row heterogeneity", pad=4.0)
    for label, axis in zip(
        ("(a)", "(b)", "(c)"),
        (confusion_axis, width_axis, scaling_axis),
        strict=True,
    ):
        axis.text(
            -0.10 if axis is confusion_axis else 0.015,
            1.04 if axis is confusion_axis else 0.985,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=7.6,
        )
    legend = (
        Line2D([], [], color=SRS_COLOR, linestyle=":", label="SRS"),
        Line2D([], [], color=PAS_COLOR, linestyle="-", label="PAS"),
    )
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        handlelength=2.3,
    )
    _save(figure, output, "phase5_new_synthetic_appendix_composite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_figures(args.results, args.output)
