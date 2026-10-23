#!/usr/bin/env python3
"""Render the final-paper CIFAR worst-three betting-CS comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "cifar_worst_three"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures"

PAS_COLOR = "#D55E00"
SRS_COLOR = "#222222"
GRID_COLOR = "#D7DBE0"
SCORES = (
    ("worst3_precision", "Worst-three precision"),
    ("worst3_taxonomic_risk", "Worst-three taxonomic risk"),
)
WIDTH_TARGETS = {
    "worst3_precision": 0.08,
    "worst3_taxonomic_risk": 0.05,
}


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
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
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


def _summary(checkpoints: pd.DataFrame) -> pd.DataFrame:
    return (
        checkpoints.groupby(["score_id", "policy_id", "audit"], as_index=False)
        .width.agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_values(["score_id", "policy_id", "audit"])
    )


def _paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby(["score_id", "audit"], as_index=False)
        .relative_reduction_percent.agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_values(["score_id", "audit"])
    )


def create_figure(results: Path, output: Path) -> None:
    verification_path = results / "VERIFICATION.json"
    if not verification_path.is_file():
        raise RuntimeError("run and aggregate the CIFAR tail study before reporting")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "pass":
        raise RuntimeError("the CIFAR tail study did not pass validation")
    checkpoints = pd.read_csv(results / "checkpoints.csv.gz")
    paired = pd.read_csv(results / "paired_width_effects.csv.gz")
    if checkpoints.groupby(["score_id", "policy_id", "seed"]).ngroups != 40:
        raise RuntimeError("the plotted CIFAR panel is incomplete")
    widths = _summary(checkpoints)
    reductions = _paired_summary(paired)

    source = output / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    widths.to_csv(source / "cifar_worst3_betting_cs_width_summary.csv", index=False)
    paired.to_csv(source / "cifar_worst3_betting_cs_paired_paths.csv", index=False)
    reductions.to_csv(source / "cifar_worst3_betting_cs_reduction_summary.csv", index=False)

    _style()
    savings_rows: list[dict[str, float | int | str]] = []
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(6.85, 2.55),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    method_styles = {
        "srs": {"color": SRS_COLOR, "linestyle": ":", "label": "SRS"},
        "pas": {"color": PAS_COLOR, "linestyle": "-", "label": "PAS"},
    }
    panel_labels = []
    for column, (score_id, title) in enumerate(SCORES):
        axis = axes[column]
        for policy in ("srs", "pas"):
            part = widths[(widths.score_id == score_id) & (widths.policy_id == policy)]
            style = method_styles[policy]
            x = part.audit.to_numpy(dtype=float)
            median = part["median"].to_numpy(dtype=float)
            q25 = part.q25.to_numpy(dtype=float)
            q75 = part.q75.to_numpy(dtype=float)
            axis.fill_between(
                x,
                q25,
                q75,
                color=style["color"],
                alpha=0.11,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                x,
                median,
                color=style["color"],
                linestyle=style["linestyle"],
                label=style["label"],
                dash_capstyle="round",
                zorder=3,
            )
        target = WIDTH_TARGETS[score_id]
        axis.axhline(
            target,
            color="#6B7280",
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            zorder=0,
        )
        axis.annotate(
            rf"target: $W={target:.2f}$",
            xy=(1.0, target),
            xycoords=("axes fraction", "data"),
            xytext=(-3, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color="#5F6368",
            fontsize=6.1,
        )
        first_audits: dict[str, int] = {}
        for policy in ("srs", "pas"):
            part = widths[
                (widths.score_id == score_id)
                & (widths.policy_id == policy)
                & (widths["median"] <= target)
            ]
            if part.empty:
                raise RuntimeError(f"{policy} never reaches the frozen width target for {score_id}")
            first_audits[policy] = int(part.audit.min())
        saved = first_audits["srs"] - first_audits["pas"]
        savings_rows.append(
            {
                "score_id": score_id,
                "width_target": target,
                "pas_first_stored_audit": first_audits["pas"],
                "srs_first_stored_audit": first_audits["srs"],
                "audits_saved": saved,
            }
        )
        axis.set_yscale("log")
        axis.set_title(title, pad=3.0)
        panel_labels.append(axis.text(
            0.02,
            0.96,
            f"({chr(97 + column)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        ))
        axis.set_xlabel("Audits")
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.45, linestyle="--", dashes=(2, 2))
        axis.set_axisbelow(True)

    pd.DataFrame(savings_rows).to_csv(
        source / "cifar_worst3_betting_cs_audit_savings.csv", index=False
    )
    axes[0].set_ylabel("Confidence-sequence width")
    axes[1].tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    shared_legend = figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        handlelength=2.8,
        columnspacing=1.5,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.19,
        top=0.79,
        wspace=0.08,
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "supplementary code CIFAR finite-envelope betting-CS pipeline"}
    # Narrow, large-type standalone panels for inline LaTeX composition.
    shared_ylim = axes[0].get_ylim()
    for column, ((score_id, title), stem) in enumerate(
        zip(
            SCORES,
            (
                "phase6_cifar_worst3_precision",
                "phase6_cifar_worst3_taxonomic_risk",
            ),
            strict=True,
        )
    ):
        standalone, standalone_axis = plt.subplots(
            figsize=(2.45, 2.55), constrained_layout=True
        )
        for policy in ("srs", "pas"):
            part = widths[
                (widths.score_id == score_id) & (widths.policy_id == policy)
            ]
            style = method_styles[policy]
            x = part.audit.to_numpy(dtype=float)
            standalone_axis.fill_between(
                x,
                part.q25.to_numpy(dtype=float),
                part.q75.to_numpy(dtype=float),
                color=style["color"],
                alpha=0.11,
                linewidth=0,
                zorder=1,
            )
            standalone_axis.plot(
                x,
                part["median"].to_numpy(dtype=float),
                color=style["color"],
                linestyle=style["linestyle"],
                dash_capstyle="round",
                zorder=3,
            )
        target = WIDTH_TARGETS[score_id]
        standalone_axis.axhline(
            target,
            color="#6B7280",
            linewidth=0.85,
            linestyle=(0, (3, 2)),
            zorder=0,
        )
        standalone_axis.annotate(
            rf"target: $W={target:.2f}$",
            xy=(1.0, target),
            xycoords=("axes fraction", "data"),
            xytext=(-3, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color="#5F6368",
            fontsize=8.2,
        )
        standalone_axis.set_yscale("log")
        standalone_axis.set_ylim(shared_ylim)
        standalone_axis.set_xlim(0, 10_250)
        standalone_axis.set_xticks([0, 5_000, 10_000])
        standalone_axis.set_title(title, fontsize=10.5, pad=3.0)
        standalone_axis.set_xlabel("Audits", fontsize=10.5)
        standalone_axis.tick_params(labelsize=9.0)
        if column == 0:
            standalone_axis.set_ylabel("Confidence-sequence width", fontsize=10.5)
        else:
            standalone_axis.tick_params(labelleft=False)
        standalone_axis.grid(
            axis="y",
            color=GRID_COLOR,
            linewidth=0.45,
            linestyle="--",
            dashes=(2, 2),
        )
        standalone_axis.set_axisbelow(True)
        standalone.savefig(
            output / f"{stem}.pdf",
            bbox_inches="tight",
            pad_inches=0.025,
            metadata=metadata,
        )
        plt.close(standalone)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    create_figure(arguments.results, arguments.output)
