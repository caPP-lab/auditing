#!/usr/bin/env python3
"""Run the paired Kenya biodiversity mislabel-rate confidence-sequence study."""

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
from auditcs.mislabel_cs import (
    MislabelAuditConfig,
    MislabelRateAudit,
    POLICIES,
    POLICY_PAS,
    POLICY_SRS,
    serfling_half_width,
    serfling_target_reviews,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "kenya_mislabel"
DEFAULT_SEEDS = tuple(range(20))
DEFAULT_BUDGET = 5_000
DEFAULT_CHECKPOINTS = tuple(range(25, DEFAULT_BUDGET + 1, 25))
TARGET_HALF_WIDTH = 0.02


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
    return output / "shards" / f"seed_{seed:02d}__{policy}"


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
    labels = np.asarray(population["v_label_index"], dtype=int)
    rows = np.asarray(population["y_row_index"], dtype=int)
    row_to_label = np.asarray(population["row_to_label_index"], dtype=int)
    audit = MislabelRateAudit(
        rows,
        labels,
        row_to_label,
        seed=seed,
        policy=policy,
    )
    submitted = row_to_label[rows]
    truth = float(np.mean(labels != submitted))
    checkpoint_set = set(checkpoints)
    actions: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    first_target_audit = None
    started = perf_counter()
    for review in range(1, budget + 1):
        request = audit.prepare()
        if request is None:
            break
        probabilities = np.asarray(request["row_probabilities"], dtype=float)
        selected = int(request["row"])
        record_id = int(request["record_id"])
        observed = audit.observe(record_id)
        interval = audit.interval()
        if first_target_audit is None and interval["half_width"] <= TARGET_HALF_WIDTH:
            first_target_audit = review
        actions.append(
            {
                "seed": seed,
                "policy_id": policy,
                "audit": review,
                "record_id": record_id,
                "submitted_row": selected,
                "within_row_audit": int(audit.audits[selected]),
                "error": int(observed["error"]),
                "observation": float(observed["observation"]),
                "selected_probability": float(probabilities[selected]),
                "probability_sum": float(probabilities.sum()),
                "plugin": float(request["plugin"]),
                "predicted_error": float(request["predicted_error"][selected]),
                "observation_lower": float(request["lower_range"]),
                "observation_upper": float(request["upper_range"]),
                "variance_estimate": float(request["variance_estimate"]),
                "base_stake": float(request["base_stake"]),
                "selected_weight": float(request["selected_weight"]),
            }
        )
        if review in checkpoint_set:
            records.append(
                {
                    "seed": seed,
                    "policy_id": policy,
                    "policy": "SRS" if policy == POLICY_SRS else "PAS",
                    "audit": review,
                    "lower": float(interval["lower"]),
                    "upper": float(interval["upper"]),
                    "half_width": float(interval["half_width"]),
                    "center": float(interval["center"]),
                    "truth": truth,
                    "truth_covered": bool(
                        float(interval["lower"]) <= truth <= float(interval["upper"])
                    ),
                    "alive_candidates": int(interval["alive_candidates"]),
                    "target_met": bool(
                        float(interval["half_width"]) <= TARGET_HALF_WIDTH
                    ),
                    "elapsed_seconds": perf_counter() - started,
                }
            )
    if not records or records[-1]["audit"] != max(checkpoints):
        raise RuntimeError(f"task {seed}/{policy} did not reach every checkpoint")
    task.mkdir(parents=True, exist_ok=True)
    checkpoint_frame = pd.DataFrame(records)
    action_frame = pd.DataFrame(actions)
    _atomic_csv(checkpoint_frame, task / "checkpoints.csv.gz")
    _atomic_csv(action_frame, task / "actions.csv.gz")
    allocations = (
        action_frame.groupby("submitted_row", as_index=False)
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
        "all_checkpoint_truth_covered": bool(checkpoint_frame.truth_covered.all()),
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
        raise RuntimeError(f"incomplete tasks: {missing}")
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)
    allocations = pd.concat(allocation_frames, ignore_index=True)
    stopping = pd.DataFrame(completion_rows)
    _atomic_csv(checkpoints, output / "checkpoints.csv.gz")
    _atomic_csv(actions, output / "actions.csv.gz")
    _atomic_csv(allocations, output / "allocations.csv.gz")
    _atomic_csv(stopping, output / "stopping.csv.gz")

    population = load_reference_population()
    population_size = len(population["y_row_index"])
    serfling = pd.DataFrame(
        {
            "audit": np.arange(
                1,
                min(population_size, int(checkpoints.audit.max())) + 1,
            ),
        }
    )
    serfling["half_width"] = serfling_half_width(
        serfling.audit.to_numpy(), population_size
    )
    _atomic_csv(serfling, output / "serfling.csv.gz")

    # Verify exact remaining-record SRS and paired within-row replay prefixes.
    row_sizes = np.asarray(population["N_r"], dtype=int)
    streams: dict[tuple[int, str], dict[tuple[int, int], int]] = {}
    max_srs_probability_error = 0.0
    for (seed, policy), group in actions.groupby(["seed", "policy_id"]):
        remaining = row_sizes.copy()
        local = np.zeros_like(row_sizes)
        stream = {}
        for row in group.sort_values("audit").itertuples():
            submitted_row = int(row.submitted_row)
            stream[(submitted_row, int(local[submitted_row]))] = int(row.record_id)
            if policy == POLICY_SRS:
                expected = remaining[submitted_row] / remaining.sum()
                max_srs_probability_error = max(
                    max_srs_probability_error,
                    abs(float(row.selected_probability) - float(expected)),
                )
            local[submitted_row] += 1
            remaining[submitted_row] -= 1
        streams[(int(seed), str(policy))] = stream
    conflicts = 0
    shared_prefixes = 0
    for seed in seeds:
        left = streams[(seed, POLICY_SRS)]
        right = streams[(seed, POLICY_PAS)]
        shared = set(left).intersection(right)
        shared_prefixes += len(shared)
        conflicts += sum(left[key] != right[key] for key in shared)
    if max_srs_probability_error > 1e-12:
        raise RuntimeError("SRS was not uniform over remaining records")
    if conflicts:
        raise RuntimeError("paired policies disagree on within-row record prefixes")
    nested = (
        checkpoints.sort_values("audit")
        .groupby(["seed", "policy_id"])
        .apply(
            lambda frame: bool(
                np.all(np.diff(frame.lower) >= -1e-12)
                and np.all(np.diff(frame.upper) <= 1e-12)
            ),
            include_groups=False,
        )
    )
    if not nested.all():
        raise RuntimeError("a reported interval was not nested")

    target = (
        stopping.groupby(["policy_id"], as_index=False)
        .first_target_audit.agg(
            paths="count",
            mean="mean",
            median="median",
            q10=lambda values: values.quantile(0.1),
            q90=lambda values: values.quantile(0.9),
            minimum="min",
            maximum="max",
        )
    )
    target = pd.concat(
        [
            target,
            pd.DataFrame(
                [
                    {
                        "policy_id": "serfling",
                        "paths": 1,
                        "mean": serfling_target_reviews(TARGET_HALF_WIDTH, population_size),
                        "median": serfling_target_reviews(TARGET_HALF_WIDTH, population_size),
                        "q10": serfling_target_reviews(TARGET_HALF_WIDTH, population_size),
                        "q90": serfling_target_reviews(TARGET_HALF_WIDTH, population_size),
                        "minimum": serfling_target_reviews(TARGET_HALF_WIDTH, population_size),
                        "maximum": serfling_target_reviews(TARGET_HALF_WIDTH, population_size),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    _atomic_csv(target, output / "target_summary.csv.gz")
    verification = {
        "status": "pass",
        "tasks": len(seeds) * len(policies),
        "records": population_size,
        "rows": len(row_sizes),
        "checkpoint_rows": len(checkpoints),
        "action_rows": len(actions),
        "paths_ever_missing_truth_at_checkpoint": int(
            checkpoints.groupby(["seed", "policy_id"]).truth_covered.all().eq(False).sum()
        ),
        "maximum_probability_sum_error": float(
            np.abs(actions.probability_sum - 1.0).max()
        ),
        "maximum_srs_record_probability_error": max_srs_probability_error,
        "shared_within_row_prefixes": shared_prefixes,
        "within_row_record_conflicts": conflicts,
        "all_intervals_nested": bool(nested.all()),
        "serfling_reviews_at_two_points": serfling_target_reviews(
            TARGET_HALF_WIDTH, population_size
        ),
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
        "--checkpoints", type=_parse_csv_ints, default=DEFAULT_CHECKPOINTS
    )
    parser.add_argument(
        "--policies", type=lambda value: tuple(value.split(",")), default=POLICIES
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = tuple(args.policies)
    unknown = sorted(set(policies).difference(POLICIES))
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    if args.stage == "preflight":
        args.replicates = 1
        args.budget = 500
        args.checkpoints = tuple(range(25, 501, 25))
        args.workers = 1
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_OUTPUT / "preflight"
    seeds = tuple(range(args.seed_start, args.seed_start + args.replicates))
    if max(args.checkpoints) > args.budget:
        raise ValueError("all checkpoints must lie within the audit budget")
    if args.stage in ("preflight", "run", "all"):
        bundle = verify_reference_bundle()
        population = load_reference_population()
        config = MislabelAuditConfig()
        lock = {
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "direct linear-score betting confidence sequence",
            "comparison": "PAS versus uniform remaining-record SRS with matched inference; fixed-n Serfling reference",
            "seeds": list(seeds),
            "policies": list(policies),
            "budget": args.budget,
            "checkpoints": list(args.checkpoints),
            "target_half_width": TARGET_HALF_WIDTH,
            "confidence_at_least": 1.0 - config.alpha,
            "config": config.__dict__,
            "rows": int(len(population["N_r"])),
            "records": int(len(population["y_row_index"])),
            "stratification": "submitted labels",
            "reference_bundle": bundle.__dict__,
            "code_sha256": {
                "audit": _sha256(
                    PROJECT_ROOT / "auditcs/mislabel_cs.py"
                ),
                "runner": _sha256(Path(__file__)),
            },
        }
        _atomic_json(lock, args.output / "RUN_LOCK.json")
        tasks = [(seed, policy) for seed in seeds for policy in policies]
        if args.workers == 1:
            for seed, policy in tasks:
                print(
                    json.dumps(
                        run_task(
                            seed=seed,
                            policy=policy,
                            budget=args.budget,
                            checkpoints=tuple(args.checkpoints),
                            output=args.output,
                            overwrite=args.overwrite,
                        )
                    ),
                    flush=True,
                )
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
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
