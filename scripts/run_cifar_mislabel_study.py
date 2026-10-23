#!/usr/bin/env python3
"""Tune and run the paired CIFAR-10N mislabel-rate CS study."""

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

from auditcs.cifar_mislabel_cs import (
    AGREEMENT_GROUPS,
    CifarMislabelConfig,
    CifarMislabelRateAudit,
    calibrate_stratum_priors,
)
from auditcs.mislabel_cs import (
    POLICY_PAS,
    POLICY_SRS,
    serfling_half_width,
    serfling_target_reviews,
)


DEFAULT_DATA = PROJECT_ROOT / "data" / "cifar" / "CIFAR-10_human_ordered.npy"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "cifar_mislabel"
DEFAULT_SCREEN_SEEDS = tuple(range(2026101100, 2026101105))
DEFAULT_CONFIRMATION_SEEDS = tuple(range(2026101200, 2026101220))
DEFAULT_SCREEN_BUDGET = 1_500
DEFAULT_CONFIRMATION_BUDGET = 5_000
DEFAULT_CHECKPOINTS = tuple(range(25, DEFAULT_CONFIRMATION_BUDGET + 1, 25))
TARGET_HALF_WIDTH = 0.02

POLICY_VARIANTS = {
    "floor05_fast": CifarMislabelConfig(
        srs_floor=0.05, prior_strength=20.0, stake_effective_n=500.0
    ),
    "floor10_fast": CifarMislabelConfig(
        srs_floor=0.10, prior_strength=20.0, stake_effective_n=500.0
    ),
    "floor10_base": CifarMislabelConfig(
        srs_floor=0.10, prior_strength=20.0, stake_effective_n=800.0
    ),
    "floor20_base": CifarMislabelConfig(
        srs_floor=0.20, prior_strength=20.0, stake_effective_n=800.0
    ),
    "floor30_base": CifarMislabelConfig(
        srs_floor=0.30, prior_strength=20.0, stake_effective_n=800.0
    ),
    "floor10_stable": CifarMislabelConfig(
        srs_floor=0.10, prior_strength=50.0, stake_effective_n=1_200.0
    ),
}


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


def _load_fields(path: Path) -> dict[str, np.ndarray]:
    source = np.load(path, allow_pickle=True).item()
    required = {
        "clean_label",
        "random_label1",
        "random_label2",
        "random_label3",
    }
    missing = required.difference(source)
    if missing:
        raise KeyError(f"missing CIFAR fields: {sorted(missing)}")
    return {key: np.asarray(source[key], dtype=int) for key in required}


def _development_priors(fields: dict[str, np.ndarray]) -> np.ndarray:
    return calibrate_stratum_priors(
        fields["random_label2"],
        fields["clean_label"],
        fields["random_label1"],
        fields["random_label3"],
    )


def _task_directory(
    output: Path,
    phase: str,
    variant: str,
    seed: int,
    policy: str,
) -> Path:
    return output / "shards" / phase / variant / f"seed_{seed}__{policy}"


def run_task(
    *,
    phase: str,
    variant: str,
    seed: int,
    policy: str,
    budget: int,
    checkpoints: tuple[int, ...],
    output: Path,
    data_path: Path,
    config: CifarMislabelConfig,
    overwrite: bool = False,
) -> dict[str, object]:
    task = _task_directory(output, phase, variant, seed, policy)
    complete = task / "COMPLETE.json"
    if complete.is_file() and not overwrite:
        return json.loads(complete.read_text(encoding="utf-8"))
    fields = _load_fields(data_path)
    priors = _development_priors(fields)
    if phase == "development":
        submitted = fields["random_label2"]
        auxiliary_one = fields["random_label1"]
        auxiliary_two = fields["random_label3"]
    elif phase == "confirmation":
        submitted = fields["random_label1"]
        auxiliary_one = fields["random_label2"]
        auxiliary_two = fields["random_label3"]
    else:
        raise ValueError(f"unknown phase {phase!r}")
    verified = fields["clean_label"]
    truth = float(np.mean(submitted != verified))
    audit = CifarMislabelRateAudit(
        submitted,
        verified,
        auxiliary_one,
        auxiliary_two,
        prior_error=priors,
        seed=seed,
        policy=policy,
        config=config,
    )
    checkpoint_set = set(checkpoints)
    actions: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    first_target_audit: int | None = None
    previous_lower, previous_upper = 0.0, 1.0
    started = perf_counter()
    for review in range(1, budget + 1):
        request = audit.prepare()
        if request is None:
            break
        probabilities = np.asarray(request["row_probabilities"], dtype=float)
        selected = int(request["row"])
        record_id = int(request["record_id"])
        within_before = int(audit.audits[selected])
        observed = audit.observe(record_id)
        interval = audit.interval()
        lower, upper = float(interval["lower"]), float(interval["upper"])
        if lower < previous_lower - 1e-12 or upper > previous_upper + 1e-12:
            raise RuntimeError("running-intersection interval expanded")
        previous_lower, previous_upper = lower, upper
        if first_target_audit is None and float(interval["half_width"]) <= TARGET_HALF_WIDTH:
            first_target_audit = review
        actions.append(
            {
                "phase": phase,
                "variant": variant,
                "seed": seed,
                "policy_id": policy,
                "audit": review,
                "record_id": record_id,
                "audit_stratum": selected,
                "submitted_class": int(audit.row_to_submitted_label[selected]),
                "agreement_group": AGREEMENT_GROUPS[selected % len(AGREEMENT_GROUPS)],
                "within_stratum_audit": within_before,
                "error": int(observed["error"]),
                "observation": float(observed["observation"]),
                "selected_probability": float(probabilities[selected]),
                "probability_sum": float(probabilities.sum()),
                "plugin": float(request["plugin"]),
                "predicted_error": float(request["predicted_error"][selected]),
                "selected_weight": float(request["selected_weight"]),
            }
        )
        if review in checkpoint_set:
            records.append(
                {
                    "phase": phase,
                    "variant": variant,
                    "seed": seed,
                    "policy_id": policy,
                    "policy": "SRS" if policy == POLICY_SRS else "PAS",
                    "audit": review,
                    "lower": lower,
                    "upper": upper,
                    "half_width": float(interval["half_width"]),
                    "center": float(interval["center"]),
                    "truth": truth,
                    "truth_covered": bool(lower <= truth <= upper),
                    "alive_candidates": int(interval["alive_candidates"]),
                    "target_met": bool(float(interval["half_width"]) <= TARGET_HALF_WIDTH),
                    "elapsed_seconds": perf_counter() - started,
                }
            )
    if not records or int(records[-1]["audit"]) != max(checkpoints):
        raise RuntimeError(f"task {phase}/{variant}/{seed}/{policy} missed checkpoints")
    task.mkdir(parents=True, exist_ok=True)
    checkpoints_frame = pd.DataFrame(records)
    actions_frame = pd.DataFrame(actions)
    _atomic_csv(checkpoints_frame, task / "checkpoints.csv.gz")
    _atomic_csv(actions_frame, task / "actions.csv.gz")
    result = {
        "status": "complete",
        "phase": phase,
        "variant": variant,
        "seed": seed,
        "policy_id": policy,
        "reviews": len(actions),
        "first_target_audit": first_target_audit,
        "all_checkpoint_truth_covered": bool(checkpoints_frame.truth_covered.all()),
        "seconds": perf_counter() - started,
    }
    _atomic_json(result, complete)
    return result


def _run_tasks(tasks: list[dict[str, object]], workers: int) -> None:
    if workers == 1:
        for task in tasks:
            print(json.dumps(run_task(**task)), flush=True)
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(run_task, **task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            print(
                json.dumps(
                    {
                        "phase": task["phase"],
                        "variant": task["variant"],
                        "seed": task["seed"],
                        "policy": task["policy"],
                        **result,
                    }
                ),
                flush=True,
            )


def aggregate_screen(output: Path, seeds: tuple[int, ...]) -> str:
    rows = []
    checkpoints = []
    for variant in POLICY_VARIANTS:
        for seed in seeds:
            task = _task_directory(output, "development", variant, seed, POLICY_PAS)
            if not (task / "COMPLETE.json").is_file():
                raise RuntimeError(f"missing development task {variant}/{seed}")
            rows.append(json.loads((task / "COMPLETE.json").read_text(encoding="utf-8")))
            checkpoints.append(pd.read_csv(task / "checkpoints.csv.gz"))
    completion = pd.DataFrame(rows)
    checkpoint_frame = pd.concat(checkpoints, ignore_index=True)
    terminal_audit = int(checkpoint_frame.audit.max())
    terminal = checkpoint_frame[checkpoint_frame.audit == terminal_audit]
    summary = (
        terminal.groupby("variant", as_index=False)
        .agg(
            median_half_width=("half_width", "median"),
            mean_half_width=("half_width", "mean"),
            paths=("seed", "nunique"),
            truth_coverage=("truth_covered", "mean"),
        )
        .sort_values(["median_half_width", "mean_half_width", "variant"])
        .reset_index(drop=True)
    )
    selected = str(summary.iloc[0].variant)
    _atomic_csv(summary, output / "screen_summary.csv.gz")
    _atomic_csv(checkpoint_frame, output / "screen_checkpoints.csv.gz")
    _atomic_csv(completion, output / "screen_completions.csv.gz")
    decision = {
        "status": "pass",
        "development_view": "random_label2 versus clean_label",
        "auxiliary_fields": ["random_label1", "random_label3"],
        "selection_rule": "minimum median terminal half-width; then mean; then variant id",
        "selected_variant": selected,
        "terminal_audit": terminal_audit,
        "candidates": {key: value.__dict__ for key, value in POLICY_VARIANTS.items()},
    }
    _atomic_json(decision, output / "SCREEN_DECISION.json")
    return selected


def aggregate_confirmation(
    output: Path,
    data_path: Path,
    seeds: tuple[int, ...],
    variant: str,
) -> dict[str, object]:
    checkpoint_frames = []
    action_frames = []
    completion_rows = []
    for seed in seeds:
        for policy in (POLICY_SRS, POLICY_PAS):
            task = _task_directory(output, "confirmation", variant, seed, policy)
            if not (task / "COMPLETE.json").is_file():
                raise RuntimeError(f"missing confirmation task {seed}/{policy}")
            checkpoint_frames.append(pd.read_csv(task / "checkpoints.csv.gz"))
            action_frames.append(pd.read_csv(task / "actions.csv.gz"))
            completion_rows.append(
                json.loads((task / "COMPLETE.json").read_text(encoding="utf-8"))
            )
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)
    stopping = pd.DataFrame(completion_rows)
    _atomic_csv(checkpoints, output / "checkpoints.csv.gz")
    _atomic_csv(actions, output / "actions.csv.gz")
    _atomic_csv(stopping, output / "stopping.csv.gz")

    fields = _load_fields(data_path)
    submitted = fields["random_label1"]
    auxiliary_one = fields["random_label2"]
    auxiliary_two = fields["random_label3"]
    # Reconstruct the same 40 frozen strata for exact SRS and pairing checks.
    from auditcs.cifar_mislabel_cs import stratify_votes

    record_rows, _, _ = stratify_votes(submitted, auxiliary_one, auxiliary_two)
    row_sizes = np.bincount(record_rows, minlength=40).astype(int)
    streams: dict[tuple[int, str], dict[tuple[int, int], int]] = {}
    maximum_srs_error = 0.0
    for (seed, policy), frame in actions.groupby(["seed", "policy_id"]):
        remaining = row_sizes.copy()
        local = np.zeros_like(row_sizes)
        stream = {}
        for row in frame.sort_values("audit").itertuples():
            stratum = int(row.audit_stratum)
            stream[(stratum, int(local[stratum]))] = int(row.record_id)
            if policy == POLICY_SRS:
                expected = remaining[stratum] / remaining.sum()
                maximum_srs_error = max(
                    maximum_srs_error,
                    abs(float(row.selected_probability) - float(expected)),
                )
            local[stratum] += 1
            remaining[stratum] -= 1
        streams[(int(seed), str(policy))] = stream
    conflicts = 0
    shared = 0
    for seed in seeds:
        left = streams[(seed, POLICY_SRS)]
        right = streams[(seed, POLICY_PAS)]
        keys = set(left).intersection(right)
        shared += len(keys)
        conflicts += sum(left[key] != right[key] for key in keys)
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
    if maximum_srs_error > 1e-12 or conflicts or not nested.all():
        raise RuntimeError("confirmation replay validation failed")

    population_size = len(submitted)
    serfling = pd.DataFrame({"audit": np.arange(1, int(checkpoints.audit.max()) + 1)})
    serfling["half_width"] = serfling_half_width(
        serfling.audit.to_numpy(), population_size
    )
    _atomic_csv(serfling, output / "serfling.csv.gz")
    verification = {
        "status": "pass",
        "selected_variant": variant,
        "tasks": len(seeds) * 2,
        "records": population_size,
        "audit_strata": int(len(row_sizes)),
        "checkpoint_rows": int(len(checkpoints)),
        "action_rows": int(len(actions)),
        "paths_ever_missing_truth_at_checkpoint": int(
            checkpoints.groupby(["seed", "policy_id"]).truth_covered.all().eq(False).sum()
        ),
        "maximum_probability_sum_error": float(
            np.abs(actions.probability_sum - 1.0).max()
        ),
        "maximum_srs_record_probability_error": float(maximum_srs_error),
        "shared_within_stratum_prefixes": int(shared),
        "within_stratum_record_conflicts": int(conflicts),
        "all_intervals_nested": bool(nested.all()),
        "serfling_reviews_at_two_points": int(
            serfling_target_reviews(TARGET_HALF_WIDTH, population_size)
        ),
    }
    _atomic_json(verification, output / "VERIFICATION.json")
    return verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("screen", "confirmation", "aggregate", "all")
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.stage in ("screen", "all"):
        screen_checkpoints = tuple(range(25, DEFAULT_SCREEN_BUDGET + 1, 25))
        tasks = [
            {
                "phase": "development",
                "variant": variant,
                "seed": seed,
                "policy": POLICY_PAS,
                "budget": DEFAULT_SCREEN_BUDGET,
                "checkpoints": screen_checkpoints,
                "output": args.output,
                "data_path": args.data,
                "config": config,
                "overwrite": args.overwrite,
            }
            for variant, config in POLICY_VARIANTS.items()
            for seed in DEFAULT_SCREEN_SEEDS
        ]
        _run_tasks(tasks, args.workers)
        selected = aggregate_screen(args.output, DEFAULT_SCREEN_SEEDS)
        print(json.dumps({"selected_variant": selected}), flush=True)
    else:
        decision_path = args.output / "SCREEN_DECISION.json"
        if not decision_path.is_file():
            raise RuntimeError("run the screen before confirmation")
        selected = json.loads(decision_path.read_text(encoding="utf-8"))[
            "selected_variant"
        ]

    if args.stage in ("confirmation", "all"):
        config = POLICY_VARIANTS[selected]
        lock = {
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "direct finite-population linear-score betting CS",
            "comparison": "vote-stratified PAS versus uniform remaining-record SRS; fixed-n Serfling reference",
            "confirmation_view": "random_label1 versus clean_label",
            "auxiliary_fields": ["random_label2", "random_label3"],
            "development_view": "random_label2 versus clean_label",
            "selected_variant": selected,
            "selected_config": config.__dict__,
            "confirmation_seeds": list(DEFAULT_CONFIRMATION_SEEDS),
            "budget": DEFAULT_CONFIRMATION_BUDGET,
            "checkpoints": list(DEFAULT_CHECKPOINTS),
            "target_half_width": TARGET_HALF_WIDTH,
            "confidence_at_least": 1.0 - config.alpha,
            "data_sha256": _sha256(args.data),
            "code_sha256": {
                "audit": _sha256(
                    PROJECT_ROOT / "auditcs/cifar_mislabel_cs.py"
                ),
                "runner": _sha256(Path(__file__)),
            },
        }
        _atomic_json(lock, args.output / "RUN_LOCK.json")
        tasks = [
            {
                "phase": "confirmation",
                "variant": selected,
                "seed": seed,
                "policy": policy,
                "budget": DEFAULT_CONFIRMATION_BUDGET,
                "checkpoints": DEFAULT_CHECKPOINTS,
                "output": args.output,
                "data_path": args.data,
                "config": config,
                "overwrite": args.overwrite,
            }
            for seed in DEFAULT_CONFIRMATION_SEEDS
            for policy in (POLICY_SRS, POLICY_PAS)
        ]
        _run_tasks(tasks, args.workers)

    if args.stage in ("aggregate", "all"):
        verification = aggregate_confirmation(
            args.output, args.data, DEFAULT_CONFIRMATION_SEEDS, selected
        )
        lock_path = args.output / "RUN_LOCK.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["status"] = "complete"
        lock["completed_utc"] = datetime.now(timezone.utc).isoformat()
        lock["verification"] = verification
        _atomic_json(lock, lock_path)
        print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
