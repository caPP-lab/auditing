#!/usr/bin/env python3
"""Run the final matched SRS/PAS Kenya biodiversity entropy-CS experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from auditcs.core_loader import (
    load_reference_population,
    verify_reference_bundle,
)
from auditcs.entropy_cs import (
    EntropyAuditConfig,
    OptimizedEntropyAudit,
    POLICIES,
    POLICY_PAS,
    POLICY_SRS,
    build_station_strata,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "kenya_entropy"
DEFAULT_SEEDS = tuple(range(95_001, 95_021))
DEFAULT_BUDGET = 3_000
DEFAULT_ENDPOINT_PERIOD = 25
DEFAULT_CHECKPOINTS = tuple(range(100, DEFAULT_BUDGET + 1, 100))
TARGET_ENTROPY_WIDTH = float(np.log(21.0 / 19.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    os.replace(temporary, path)


def _task_directory(output: Path, seed: int, policy: str) -> Path:
    return output / "shards" / f"seed_{seed}__{policy}"


def _certificate_summary(value: object) -> str:
    if not isinstance(value, dict):
        return json.dumps({"present": False}, sort_keys=True)
    keep = {"present": True}
    for key in (
        "certified",
        "log_profile_lower",
        "log_threshold_upper",
        "certified_global_gap_upper",
        "min_factor_lower",
    ):
        if key in value:
            item = value[key]
            if isinstance(item, (bool, np.bool_)):
                keep[key] = bool(item)
            elif item is not None:
                keep[key] = float(item)
    return json.dumps(keep, sort_keys=True)


def run_task(
    *,
    seed: int,
    policy: str,
    budget: int,
    checkpoints: tuple[int, ...],
    output: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    task = _task_directory(output, seed, policy)
    complete = task / "COMPLETE.json"
    if complete.is_file() and not overwrite:
        return json.loads(complete.read_text(encoding="utf-8"))
    population = load_reference_population()
    strata = build_station_strata()
    labels = np.asarray(population["v_label_index"], dtype=int)
    config = EntropyAuditConfig(endpoint_period=DEFAULT_ENDPOINT_PERIOD)
    audit = OptimizedEntropyAudit(
        strata.record_strata,
        strata.stratum_to_row,
        strata.stratum_to_submitted_label,
        len(strata.category_labels),
        seed=seed,
        policy=policy,
        config=config,
    )
    counts = np.asarray(population["C_star"].sum(axis=0), dtype=float)
    probabilities = counts / counts.sum()
    positive = probabilities > 0
    truth_entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    truth_p2 = float(np.exp(truth_entropy))
    checkpoint_set = set(checkpoints)
    actions: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    started = perf_counter()
    first_target_audit = None
    for review in range(1, budget + 1):
        request = audit.prepare()
        if request is None:
            break
        record_id = int(request["record_id"])
        stratum = int(request["stratum"])
        represented = np.asarray(request["row_probabilities"], dtype=float)
        selected_probability = float(represented[stratum])
        if selected_probability <= 0 or abs(float(represented.sum()) - 1.0) > 1e-12:
            raise RuntimeError("Invalid represented stratum probabilities")
        audit.observe(record_id, int(labels[record_id]))
        if review % config.endpoint_period == 0:
            before = perf_counter()
            interval = audit.interval()
            endpoint_seconds = perf_counter() - before
            if (
                first_target_audit is None
                and interval["entropy_width"] <= TARGET_ENTROPY_WIDTH
            ):
                first_target_audit = review
        else:
            interval = None
            endpoint_seconds = 0.0
        actions.append(
            {
                "seed": seed,
                "policy_id": policy,
                "audit": review,
                "record_id": record_id,
                "stratum": stratum,
                "submitted_row": int(request["submitted_row"]),
                "destination": int(labels[record_id]),
                "selected_stratum_probability": selected_probability,
                "minimum_active_probability": float(represented[represented > 0].min()),
                "probability_sum": float(represented.sum()),
                "selected_proportional_component": float(
                    request["allocation_proportional"][stratum]
                ),
                "selected_range_component": float(request["allocation_range"][stratum]),
                "selected_neyman_component": float(
                    request["allocation_neyman"][stratum]
                ),
            }
        )
        if review in checkpoint_set:
            if interval is None:
                before = perf_counter()
                interval = audit.interval(force=True)
                endpoint_seconds = perf_counter() - before
            records.append(
                {
                    "seed": seed,
                    "policy_id": policy,
                    "policy": "SRS" if policy == POLICY_SRS else "PAS",
                    "audit": review,
                    "entropy_lower": float(interval["entropy_lower"]),
                    "entropy_upper": float(interval["entropy_upper"]),
                    "entropy_width": float(interval["entropy_width"]),
                    "lower": float(interval["lower"]),
                    "upper": float(interval["upper"]),
                    "width": float(interval["full_width"]),
                    "estimate": float(interval["estimate"]),
                    "harmonic_report": float(interval["harmonic_report"]),
                    "relative_error_bound": float(
                        interval["harmonic_relative_error_bound"]
                    ),
                    "truth_entropy": truth_entropy,
                    "truth": truth_p2,
                    "truth_covered": bool(
                        float(interval["entropy_lower"])
                        <= truth_entropy
                        <= float(interval["entropy_upper"])
                    ),
                    "numeric_verified": bool(interval["numeric_verified"]),
                    "target_met": bool(
                        float(interval["entropy_width"]) <= TARGET_ENTROPY_WIDTH
                    ),
                    "endpoint_seconds": endpoint_seconds,
                    "elapsed_seconds": perf_counter() - started,
                    "lower_certificate": _certificate_summary(
                        interval.get("lower_certificate")
                    ),
                }
            )
    if not records or records[-1]["audit"] != max(checkpoints):
        raise RuntimeError(f"Task {seed}/{policy} did not reach all checkpoints")
    task.mkdir(parents=True, exist_ok=True)
    checkpoint_frame = pd.DataFrame(records)
    action_frame = pd.DataFrame(actions)
    _atomic_csv(checkpoint_frame, task / "checkpoints.csv.gz")
    _atomic_csv(action_frame, task / "actions.csv.gz")
    allocations = (
        action_frame.groupby(["stratum", "submitted_row"], as_index=False)
        .size()
        .rename(columns={"size": "audits"})
    )
    allocations.insert(0, "policy_id", policy)
    allocations.insert(0, "seed", seed)
    _atomic_csv(allocations, task / "allocations.csv.gz")
    result = {
        "status": "complete",
        "seed": seed,
        "policy_id": policy,
        "reviews": len(actions),
        "first_target_audit": first_target_audit,
        "all_truth_covered": bool(checkpoint_frame.truth_covered.all()),
        "all_numeric_verified": bool(checkpoint_frame.numeric_verified.all()),
        "seconds": perf_counter() - started,
    }
    _atomic_json(result, complete)
    return result


def aggregate(
    output: Path,
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, object]:
    checkpoint_frames = []
    action_frames = []
    allocation_frames = []
    completion_rows = []
    missing = []
    for seed in seeds:
        for policy in policies:
            task = _task_directory(output, seed, policy)
            if not (task / "COMPLETE.json").is_file():
                missing.append(f"{seed}/{policy}")
                continue
            checkpoint_frames.append(pd.read_csv(task / "checkpoints.csv.gz"))
            action_frames.append(pd.read_csv(task / "actions.csv.gz"))
            allocation_frames.append(pd.read_csv(task / "allocations.csv.gz"))
            completion_rows.append(
                json.loads((task / "COMPLETE.json").read_text(encoding="utf-8"))
            )
    if missing:
        raise RuntimeError(f"Incomplete tasks: {missing}")
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)
    allocations = pd.concat(allocation_frames, ignore_index=True)
    _atomic_csv(checkpoints, output / "checkpoints.csv.gz")
    _atomic_csv(actions, output / "actions.csv.gz")
    _atomic_csv(allocations, output / "allocations.csv.gz")

    wide = checkpoints.pivot(
        index=["seed", "audit"], columns="policy_id", values="width"
    ).reset_index()
    wide["absolute_width_reduction"] = wide[POLICY_SRS] - wide[POLICY_PAS]
    wide["relative_width_reduction_percent"] = 100.0 * (
        1.0 - wide[POLICY_PAS] / wide[POLICY_SRS]
    )
    _atomic_csv(wide, output / "paired_width_effects.csv.gz")

    stopping = pd.DataFrame(completion_rows)
    _atomic_csv(stopping, output / "stopping.csv.gz")
    stop_wide = stopping.pivot(
        index="seed", columns="policy_id", values="first_target_audit"
    ).reset_index()
    if stop_wide[[POLICY_SRS, POLICY_PAS]].notna().all().all():
        stop_wide["audits_saved"] = stop_wide[POLICY_SRS] - stop_wide[POLICY_PAS]
        stop_wide["relative_audit_saving_percent"] = 100.0 * (
            1.0 - stop_wide[POLICY_PAS] / stop_wide[POLICY_SRS]
        )
    _atomic_csv(stop_wide, output / "paired_stopping.csv.gz")

    population = load_reference_population()
    strata = build_station_strata()
    stratum_sizes = np.bincount(strata.record_strata, minlength=len(strata.stratum_labels))
    streams: dict[tuple[int, str], dict[tuple[int, int], int]] = {}
    max_srs_probability_error = 0.0
    for (seed, policy), group in actions.groupby(["seed", "policy_id"]):
        local = np.zeros(len(stratum_sizes), dtype=int)
        remaining = stratum_sizes.copy()
        stream = {}
        for row in group.sort_values("audit").itertuples():
            stratum = int(row.stratum)
            stream[(stratum, int(local[stratum]))] = int(row.record_id)
            if policy == POLICY_SRS:
                expected = float(remaining[stratum] / remaining.sum())
                max_srs_probability_error = max(
                    max_srs_probability_error,
                    abs(float(row.selected_stratum_probability) - expected),
                )
            local[stratum] += 1
            remaining[stratum] -= 1
        streams[(int(seed), str(policy))] = stream
    shared_prefixes = 0
    conflicts = 0
    for seed in seeds:
        left = streams[(seed, POLICY_SRS)]
        right = streams[(seed, POLICY_PAS)]
        shared = set(left).intersection(right)
        shared_prefixes += len(shared)
        conflicts += sum(left[key] != right[key] for key in shared)
    if conflicts:
        raise RuntimeError("SRS and PAS do not share within-stratum replay prefixes")
    if max_srs_probability_error > 1e-12:
        raise RuntimeError("The SRS policy is not uniform over remaining records")
    if not checkpoints.numeric_verified.all():
        raise RuntimeError("At least one endpoint was not numerically verified")
    if not (
        checkpoints.sort_values("audit")
        .groupby(["seed", "policy_id"])
        .apply(
            lambda frame: bool(
                np.all(np.diff(frame.entropy_lower) >= -1e-12)
                and np.all(np.diff(frame.entropy_upper) <= 1e-12)
            ),
            include_groups=False,
        )
        .all()
    ):
        raise RuntimeError("A reported confidence sequence is not nested")

    final = checkpoints[checkpoints.audit == checkpoints.audit.max()]
    terminal = (
        final.groupby(["policy_id", "policy"], as_index=False)
        .agg(
            paths=("seed", "nunique"),
            median_width=("width", "median"),
            mean_width=("width", "mean"),
            median_entropy_width=("entropy_width", "median"),
            all_truth_covered=("truth_covered", "all"),
            all_numeric_verified=("numeric_verified", "all"),
        )
    )
    _atomic_csv(terminal, output / "terminal_summary.csv.gz")
    verification = {
        "status": "pass",
        "tasks": len(seeds) * len(policies),
        "checkpoint_rows": len(checkpoints),
        "action_rows": len(actions),
        "all_checkpoint_truth_covered": bool(checkpoints.truth_covered.all()),
        "all_numeric_verified": bool(checkpoints.numeric_verified.all()),
        "maximum_probability_sum_error": float(
            np.abs(actions.probability_sum - 1.0).max()
        ),
        "maximum_srs_record_probability_error": max_srs_probability_error,
        "shared_within_stratum_prefixes": shared_prefixes,
        "within_stratum_record_conflicts": conflicts,
        "strata": len(strata.stratum_labels),
        "records": len(population["v_label_index"]),
        "target_entropy_width": TARGET_ENTROPY_WIDTH,
    }
    _atomic_json(verification, output / "VERIFICATION.json")
    return verification


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run", "aggregate", "all"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--replicates", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--checkpoints",
        type=_parse_csv_ints,
        default=DEFAULT_CHECKPOINTS,
    )
    parser.add_argument(
        "--policies",
        type=lambda value: tuple(value.split(",")),
        default=POLICIES,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = tuple(args.policies)
    unknown = sorted(set(policies).difference(POLICIES))
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}")
    if args.stage == "preflight":
        args.replicates = 1
        args.budget = 500
        args.checkpoints = tuple(range(100, 501, 100))
        args.workers = 1
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_OUTPUT / "preflight"
    seeds = tuple(range(args.seed_start, args.seed_start + args.replicates))
    if max(args.checkpoints) > args.budget:
        raise ValueError("All checkpoints must lie within the audit budget")
    if args.stage in ("preflight", "run", "all"):
        bundle = verify_reference_bundle()
        strata = build_station_strata()
        config = EntropyAuditConfig(endpoint_period=DEFAULT_ENDPOINT_PERIOD)
        lock = {
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "optimized finite-sample convex-profile entropy confidence sequence",
            "comparison": "uniform random-record SRS versus score-aware PAS; matched inferential certificate",
            "seeds": list(seeds),
            "policies": list(policies),
            "budget": args.budget,
            "checkpoints": list(args.checkpoints),
            "endpoint_period": config.endpoint_period,
            "target_entropy_width": TARGET_ENTROPY_WIDTH,
            "confidence_at_least": 1 - config.alpha_lower - config.alpha_upper,
            "config": config.__dict__,
            "strata": len(strata.stratum_labels),
            "stratification": "submitted label by camera station",
            "reference_bundle": bundle.__dict__,
            "code_sha256": {
                "audit": _sha256(
                    PROJECT_ROOT
                    / "auditcs/entropy_cs.py"
                ),
                "runner": _sha256(Path(__file__)),
            },
        }
        _atomic_json(lock, args.output / "RUN_LOCK.json")
        tasks = [(seed, policy) for seed in seeds for policy in policies]
        if args.workers == 1:
            for seed, policy in tasks:
                result = run_task(
                    seed=seed,
                    policy=policy,
                    budget=args.budget,
                    checkpoints=tuple(args.checkpoints),
                    output=args.output,
                    overwrite=args.overwrite,
                )
                print(json.dumps(result), flush=True)
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=context
            ) as pool:
                futures = {
                    pool.submit(
                        run_task,
                        seed=seed,
                        policy=policy,
                        budget=args.budget,
                        checkpoints=tuple(args.checkpoints),
                        output=args.output,
                        overwrite=args.overwrite,
                    ): (seed, policy)
                    for seed, policy in tasks
                }
                for future in as_completed(futures):
                    seed, policy = futures[future]
                    result = future.result()
                    print(json.dumps({"seed": seed, "policy": policy, **result}), flush=True)
    if args.stage in ("aggregate", "all"):
        verification = aggregate(args.output, seeds, policies)
        lock_path = args.output / "RUN_LOCK.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["status"] = "complete"
        lock["completed_utc"] = datetime.now(timezone.utc).isoformat()
        lock["verification"] = verification
        _atomic_json(lock, lock_path)
        print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
