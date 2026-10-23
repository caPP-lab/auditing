#!/usr/bin/env python3
"""Compute PPR and Wald baselines for both final-paper mislabel studies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from auditcs.finite_population_baselines import (
    ppr_running_intervals,
    wald_half_width,
    wald_target_reviews,
)
from auditcs.mislabel_cs import POLICY_PAS, POLICY_SRS


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "mislabel_method_comparison"
DATASETS = {
    "kenya": PROJECT_ROOT / "results" / "kenya_mislabel",
    "cifar10n": PROJECT_ROOT / "results" / "cifar_mislabel",
}
TARGET_HALF_WIDTH = 0.02
ALPHA = 0.045
REPORT_AUDITS = (250, 500, 750, 1_000, 1_500, 2_000, 3_000, 4_000, 5_000)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "mtime": 0} if gzip else None,
    )
    os.replace(temporary, path)


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sequential_method_rows(
    stopping: pd.DataFrame,
    *,
    dataset: str,
) -> list[dict[str, object]]:
    rows = []
    for policy, method in ((POLICY_PAS, "PAS"), (POLICY_SRS, "SRS")):
        values = stopping.loc[stopping.policy_id == policy, "first_target_audit"].dropna()
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "selection": "priority-aware" if policy == POLICY_PAS else "random records",
                "guarantee": "finite-sample, anytime",
                "paths": int(len(values)),
                "reached": int(len(values)),
                "mean_reviews": float(values.mean()),
                "median_reviews": float(values.median()),
                "q10_reviews": float(values.quantile(0.1)),
                "q90_reviews": float(values.quantile(0.9)),
            }
        )
    return rows


def compute_dataset(dataset: str, results: Path, output: Path) -> dict[str, object]:
    required = (
        results / "actions.csv.gz",
        results / "checkpoints.csv.gz",
        results / "stopping.csv.gz",
        results / "serfling.csv.gz",
        results / "VERIFICATION.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing prerequisite files for {dataset}: {missing}")
    actions = pd.read_csv(results / "actions.csv.gz")
    checkpoints = pd.read_csv(results / "checkpoints.csv.gz")
    stopping = pd.read_csv(results / "stopping.csv.gz")
    serfling = pd.read_csv(results / "serfling.csv.gz")
    truth_values = checkpoints.truth.unique()
    if len(truth_values) != 1:
        raise RuntimeError(f"{dataset} has multiple truth values")
    truth = float(truth_values[0])
    population_size = int(
        json.loads((results / "VERIFICATION.json").read_text(encoding="utf-8"))[
            "records"
        ]
    )
    srs_actions = actions[actions.policy_id == POLICY_SRS].copy()
    if srs_actions.empty:
        raise RuntimeError(f"{dataset} has no SRS actions")

    ppr_frames = []
    ppr_stopping = []
    for seed, frame in srs_actions.groupby("seed"):
        frame = frame.sort_values("audit")
        errors = frame.error.to_numpy(dtype=int)
        intervals = ppr_running_intervals(errors, population_size, alpha=ALPHA)
        audits = np.arange(1, len(errors) + 1)
        path = pd.DataFrame(
            {
                "dataset": dataset,
                "seed": int(seed),
                "audit": audits,
                "lower_count": intervals["lower_count"],
                "upper_count": intervals["upper_count"],
                "lower": intervals["lower"],
                "upper": intervals["upper"],
                "half_width": intervals["half_width"],
            }
        )
        path["truth"] = truth
        path["truth_covered"] = (
            (path.lower <= truth) & (truth <= path.upper) & path.half_width.notna()
        )
        target = path[(path.half_width.notna()) & (path.half_width <= TARGET_HALF_WIDTH)]
        first_target = None if target.empty else int(target.audit.iloc[0])
        ppr_frames.append(path)
        ppr_stopping.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "first_target_audit": first_target,
                "empty_at": int(intervals["empty_at"][0]),
                "always_covered": bool(path.truth_covered.all()),
            }
        )
    ppr_paths = pd.concat(ppr_frames, ignore_index=True)
    ppr_stopping_frame = pd.DataFrame(ppr_stopping)
    ppr_summary = (
        ppr_paths.groupby("audit", as_index=False)
        .half_width.agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            available="count",
        )
    )

    maximum_audit = int(checkpoints.audit.max())
    wald = pd.DataFrame({"audit": np.arange(1, maximum_audit + 1)})
    wald["half_width"] = wald_half_width(
        wald.audit.to_numpy(), truth, alpha=ALPHA
    )

    local = output / dataset
    _atomic_csv(ppr_paths, local / "ppr_paths.csv.gz", gzip=True)
    _atomic_csv(ppr_summary, local / "ppr_summary.csv")
    _atomic_csv(ppr_stopping_frame, local / "ppr_stopping.csv")
    _atomic_csv(wald, local / "wald.csv")

    target_rows = _sequential_method_rows(stopping, dataset=dataset)
    ppr_reached = ppr_stopping_frame.first_target_audit.dropna()
    target_rows.append(
        {
            "dataset": dataset,
            "method": "Ramdas PPR",
            "selection": "random records",
            "guarantee": "finite-sample, anytime",
            "paths": int(len(ppr_stopping_frame)),
            "reached": int(len(ppr_reached)),
            "mean_reviews": float(ppr_reached.mean()),
            "median_reviews": float(ppr_reached.median()),
            "q10_reviews": float(ppr_reached.quantile(0.1)),
            "q90_reviews": float(ppr_reached.quantile(0.9)),
        }
    )
    wald_target = wald_target_reviews(TARGET_HALF_WIDTH, truth, alpha=ALPHA)
    target_rows.append(
        {
            "dataset": dataset,
            "method": "Wald",
            "selection": "random records",
            "guarantee": "asymptotic, fixed-n; oracle variance",
            "paths": 1,
            "reached": 1,
            "mean_reviews": float(wald_target),
            "median_reviews": float(wald_target),
            "q10_reviews": float(wald_target),
            "q90_reviews": float(wald_target),
        }
    )
    serfling_target = int(serfling.loc[serfling.half_width <= TARGET_HALF_WIDTH, "audit"].min())
    target_rows.append(
        {
            "dataset": dataset,
            "method": "Serfling",
            "selection": "random records",
            "guarantee": "finite-sample, fixed-n",
            "paths": 1,
            "reached": 1,
            "mean_reviews": float(serfling_target),
            "median_reviews": float(serfling_target),
            "q10_reviews": float(serfling_target),
            "q90_reviews": float(serfling_target),
        }
    )
    targets = pd.DataFrame(target_rows)
    _atomic_csv(targets, local / "method_target_summary.csv")

    betting_summary = (
        checkpoints.groupby(["policy_id", "audit"], as_index=False)
        .half_width.agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            paths="count",
        )
    )
    checkpoint_rows: list[dict[str, object]] = []
    for audit in REPORT_AUDITS:
        for policy_id, method in ((POLICY_PAS, "PAS"), (POLICY_SRS, "SRS")):
            row = betting_summary[
                (betting_summary.policy_id == policy_id)
                & (betting_summary.audit == audit)
            ]
            if not row.empty:
                checkpoint_rows.append(
                    {
                        "dataset": dataset,
                        "audit": audit,
                        "method": method,
                        "median_half_width": float(row.iloc[0]["median"]),
                        "q25_half_width": float(row.iloc[0]["q25"]),
                        "q75_half_width": float(row.iloc[0]["q75"]),
                        "paths": int(row.iloc[0]["paths"]),
                    }
                )
        ppr_row = ppr_summary[ppr_summary.audit == audit]
        if not ppr_row.empty:
            checkpoint_rows.append(
                {
                    "dataset": dataset,
                    "audit": audit,
                    "method": "Ramdas PPR",
                    "median_half_width": float(ppr_row.iloc[0]["median"]),
                    "q25_half_width": float(ppr_row.iloc[0]["q25"]),
                    "q75_half_width": float(ppr_row.iloc[0]["q75"]),
                    "paths": int(ppr_row.iloc[0]["available"]),
                }
            )
        wald_row = wald[wald.audit == audit]
        if not wald_row.empty:
            value = float(wald_row.iloc[0].half_width)
            checkpoint_rows.append(
                {
                    "dataset": dataset,
                    "audit": audit,
                    "method": "Wald",
                    "median_half_width": value,
                    "q25_half_width": value,
                    "q75_half_width": value,
                    "paths": 1,
                }
            )
        serfling_row = serfling[serfling.audit == audit]
        if not serfling_row.empty:
            value = float(serfling_row.iloc[0].half_width)
            checkpoint_rows.append(
                {
                    "dataset": dataset,
                    "audit": audit,
                    "method": "Serfling",
                    "median_half_width": value,
                    "q25_half_width": value,
                    "q75_half_width": value,
                    "paths": 1,
                }
            )
    checkpoint_summary = pd.DataFrame(checkpoint_rows)
    _atomic_csv(checkpoint_summary, local / "method_checkpoint_summary.csv")
    return {
        "dataset": dataset,
        "population_size": population_size,
        "truth": truth,
        "srs_paths": int(srs_actions.seed.nunique()),
        "ppr_empty_paths": int((ppr_stopping_frame.empty_at >= 0).sum()),
        "ppr_always_covered_paths": int(ppr_stopping_frame.always_covered.sum()),
        "ppr_target_reached_paths": int(len(ppr_reached)),
        "target_rows": target_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets",
        default="cifar10n",
        help="comma-separated subset of: kenya,cifar10n",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    unknown = sorted(set(selected).difference(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    reports = [
        compute_dataset(dataset, DATASETS[dataset], args.output)
        for dataset in selected
    ]
    combined = pd.concat(
        [pd.DataFrame(report["target_rows"]) for report in reports],
        ignore_index=True,
    )
    _atomic_csv(combined, args.output / "method_target_summary.csv")
    combined_checkpoints = pd.concat(
        [
            pd.read_csv(args.output / dataset / "method_checkpoint_summary.csv")
            for dataset in selected
        ],
        ignore_index=True,
    )
    _atomic_csv(
        combined_checkpoints,
        args.output / "method_checkpoint_summary.csv",
    )
    _atomic_json(
        {
            "status": "pass",
            "alpha": ALPHA,
            "confidence": 1.0 - ALPHA,
            "target_half_width": TARGET_HALF_WIDTH,
            "ppr": "uniform BetaBinomial(N,1,1) prior-posterior-ratio confidence sequence with running intersection",
            "wald": "oracle finite-population truth variance; asymptotic fixed-n; no finite-population correction",
            "datasets": [
                {key: value for key, value in report.items() if key != "target_rows"}
                for report in reports
            ],
        },
        args.output / "VERIFICATION.json",
    )
    print(combined.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
