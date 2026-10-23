#!/usr/bin/env python3
"""Run paired CIFAR-10N betting CSs for the two worst-three scores."""

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

from auditcs.cifar_data import load_cifar10n_population
from auditcs.cifar_scores import (
    canonical_multimetric_scores,
    evaluate_score,
    load_taxonomy_inputs,
)
from auditcs.cifar_tail_betting_cs import (
    CifarWorstThreeAudit,
    POLICIES,
    POLICY_PAS,
    POLICY_SRS,
    SCORES,
    TailBettingConfig,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "cifar_worst_three"
DEFAULT_DATA = PROJECT_ROOT / "data" / "cifar" / "CIFAR-10_human_ordered.npy"
DEFAULT_TAXONOMY = PROJECT_ROOT / "data/taxonomy/cifar10_wordnet_3_0.json"
DEFAULT_SEEDS = tuple(range(2026100300, 2026100310))
DEFAULT_BUDGET = 10_000
DEFAULT_CHECKPOINTS = tuple(sorted({50, *range(250, DEFAULT_BUDGET + 1, 250)}))


PAS_VARIANTS = {
    "active_neyman": TailBettingConfig(),
    "balanced30": TailBettingConfig(
        allocation_mode="endpoint_balanced",
        proportional_weight=0.4,
        range_weight=0.3,
        neyman_weight=0.3,
        active_endpoint_weight=0.5,
    ),
    "conservative": TailBettingConfig(
        allocation_mode="endpoint_balanced",
        proportional_weight=0.6,
        range_weight=0.2,
        neyman_weight=0.2,
        active_endpoint_weight=0.5,
    ),
    "robust70": TailBettingConfig(
        allocation_mode="endpoint_balanced",
        proportional_weight=0.7,
        range_weight=0.15,
        neyman_weight=0.15,
        active_endpoint_weight=0.5,
    ),
    "robust75": TailBettingConfig(
        allocation_mode="endpoint_balanced",
        proportional_weight=0.75,
        range_weight=0.125,
        neyman_weight=0.125,
        active_endpoint_weight=0.5,
    ),
    "range_heavy": TailBettingConfig(
        allocation_mode="endpoint_balanced",
        proportional_weight=0.4,
        range_weight=0.4,
        neyman_weight=0.2,
        active_endpoint_weight=0.5,
    ),
    "profile_heavy": TailBettingConfig(
        allocation_mode="endpoint_balanced",
        proportional_weight=0.4,
        range_weight=0.3,
        neyman_weight=0.3,
        active_endpoint_weight=0.3,
    ),
    "neyman_heavy": TailBettingConfig(
        allocation_mode="endpoint_balanced",
        proportional_weight=0.4,
        range_weight=0.2,
        neyman_weight=0.4,
        active_endpoint_weight=0.5,
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


def _task_directory(output: Path, score_id: str, seed: int, policy: str) -> Path:
    return output / "shards" / score_id / f"seed_{seed}__{policy}"


def run_task(
    *,
    score_id: str,
    seed: int,
    policy: str,
    budget: int,
    checkpoints: tuple[int, ...],
    output: Path,
    data_path: Path,
    taxonomy_path: Path,
    config: TailBettingConfig,
    overwrite: bool = False,
) -> dict[str, object]:
    task = _task_directory(output, score_id, seed, policy)
    complete = task / "COMPLETE.json"
    if complete.is_file() and not overwrite:
        return json.loads(complete.read_text(encoding="utf-8"))

    population = load_cifar10n_population(data_path)
    taxonomy = load_taxonomy_inputs(taxonomy_path)
    score = canonical_multimetric_scores()[score_id]
    truth = evaluate_score(score, population.true_confusion, taxonomy)
    audit = CifarWorstThreeAudit(
        population.submitted,
        population.verified,
        score_id=score_id,
        policy=policy,
        seed=seed,
        taxonomy_distances=taxonomy.integer_distances,
        taxonomy_d_max=taxonomy.d_max,
        config=config,
    )
    checkpoint_set = set(checkpoints)
    records: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    started = perf_counter()
    previous_lower, previous_upper = 0.0, 1.0

    for review in range(1, budget + 1):
        request = audit.prepare()
        if request is None:
            break
        probabilities = np.asarray(request["row_probabilities"], dtype=float)
        predicted = np.asarray(request["predicted_row_values"], dtype=float)
        membership = np.asarray(request["tail_membership"], dtype=float)
        sigma = np.asarray(request["row_sigma"], dtype=float)
        selected = int(request["row"])
        record_id = int(request["record_id"])
        within_row_before = int(audit.audits_by_row[selected])
        observed = audit.observe(record_id)
        interval = audit.interval()
        lower = float(interval["lower"])
        upper = float(interval["upper"])
        if lower < previous_lower - 1e-12 or upper > previous_upper + 1e-12:
            raise RuntimeError("running-intersection interval expanded")
        previous_lower, previous_upper = lower, upper
        actions.append(
            {
                "score_id": score_id,
                "seed": seed,
                "policy_id": policy,
                "audit": review,
                "record_id": record_id,
                "submitted_row": selected,
                "within_row_audit": within_row_before,
                "verified_label": int(observed["destination"]),
                "selected_probability": float(probabilities[selected]),
                "probability_sum": float(probabilities.sum()),
                "predicted_row_value": float(predicted[selected]),
                "tail_membership": float(membership[selected]),
                "predicted_row_sigma": float(sigma[selected]),
                "selected_weight": float(request["selected_weight"]),
                "active_subset": "|".join(map(str, request["active_subset"])),
                "selected_proportional_component": float(
                    np.asarray(request["allocation_proportional"])[selected]
                ),
                "selected_range_component": float(
                    np.asarray(request["allocation_range"])[selected]
                ),
                "selected_neyman_component": float(
                    np.asarray(request["allocation_neyman"])[selected]
                ),
                "selected_profile_membership": float(
                    np.asarray(request["profile_bottleneck_membership"])[selected]
                ),
                "selected_endpoint_relevance": float(
                    np.asarray(request["endpoint_relevance"])[selected]
                ),
            }
        )
        if review in checkpoint_set:
            records.append(
                {
                    "score_id": score_id,
                    "seed": seed,
                    "policy_id": policy,
                    "policy": "SRS" if policy == POLICY_SRS else "PAS",
                    "audit": review,
                    "lower": lower,
                    "upper": upper,
                    "width": float(interval["width"]),
                    "center": float(interval["center"]),
                    "truth": float(truth),
                    "truth_covered": bool(lower <= truth <= upper),
                    "alive_candidates": int(interval["alive_candidates"]),
                    "profile_subsets": int(interval["profile_subsets"]),
                    "empty_grid_event": bool(interval["empty_grid_event"]),
                    "empty_grid_events": int(interval["empty_grid_events"]),
                    "elapsed_seconds": perf_counter() - started,
                }
            )

    if not records or int(records[-1]["audit"]) != max(checkpoints):
        raise RuntimeError(f"task {score_id}/{seed}/{policy} missed a checkpoint")
    task.mkdir(parents=True, exist_ok=True)
    checkpoint_frame = pd.DataFrame(records)
    action_frame = pd.DataFrame(actions)
    _atomic_csv(checkpoint_frame, task / "checkpoints.csv.gz")
    _atomic_csv(action_frame, task / "actions.csv.gz")
    allocation = (
        action_frame.groupby("submitted_row", as_index=False)
        .size()
        .rename(columns={"size": "audits"})
    )
    allocation.insert(0, "policy_id", policy)
    allocation.insert(0, "seed", seed)
    allocation.insert(0, "score_id", score_id)
    _atomic_csv(allocation, task / "allocations.csv.gz")
    result = {
        "status": "complete",
        "score_id": score_id,
        "seed": seed,
        "policy_id": policy,
        "reviews": len(actions),
        "checkpoint_truth_coverage": bool(checkpoint_frame.truth_covered.all()),
        "seconds": perf_counter() - started,
    }
    _atomic_json(result, complete)
    return result


def aggregate(
    output: Path,
    *,
    scores: tuple[str, ...],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    data_path: Path,
    pas_floor_weight: float,
) -> dict[str, object]:
    checkpoint_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    allocation_frames: list[pd.DataFrame] = []
    completion_rows: list[dict[str, object]] = []
    missing = []
    for score_id in scores:
        for seed in seeds:
            for policy in policies:
                task = _task_directory(output, score_id, seed, policy)
                if not (task / "COMPLETE.json").is_file():
                    missing.append(f"{score_id}/{seed}/{policy}")
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
    completion = pd.DataFrame(completion_rows)
    _atomic_csv(checkpoints, output / "checkpoints.csv.gz")
    _atomic_csv(actions, output / "actions.csv.gz")
    _atomic_csv(allocations, output / "allocations.csv.gz")
    _atomic_csv(completion, output / "completion.csv.gz")

    population = load_cifar10n_population(data_path)
    row_sizes = population.row_sizes
    probability_error = float(np.abs(actions.probability_sum - 1.0).max())
    srs_error = 0.0
    floor_margin = np.inf
    prefix_conflicts = 0
    shared_prefixes = 0
    streams: dict[tuple[str, int, str], dict[tuple[int, int], int]] = {}
    for (score_id, seed, policy), frame in actions.groupby(
        ["score_id", "seed", "policy_id"]
    ):
        remaining = row_sizes.copy()
        stream: dict[tuple[int, int], int] = {}
        for row in frame.sort_values("audit").itertuples():
            selected = int(row.submitted_row)
            local_index = int(row.within_row_audit)
            stream[(selected, local_index)] = int(row.record_id)
            srs_selected = float(remaining[selected] / remaining.sum())
            if policy == POLICY_SRS:
                srs_error = max(
                    srs_error,
                    abs(float(row.selected_probability) - srs_selected),
                )
            else:
                floor_margin = min(
                    floor_margin,
                    float(row.selected_probability) - pas_floor_weight * srs_selected,
                )
            remaining[selected] -= 1
        streams[(str(score_id), int(seed), str(policy))] = stream
    for score_id in scores:
        for seed in seeds:
            left = streams[(score_id, seed, POLICY_SRS)]
            right = streams[(score_id, seed, POLICY_PAS)]
            shared = set(left).intersection(right)
            shared_prefixes += len(shared)
            prefix_conflicts += sum(left[key] != right[key] for key in shared)

    nesting = (
        checkpoints.sort_values("audit")
        .groupby(["score_id", "seed", "policy_id"])
        .apply(
            lambda frame: bool(
                np.all(np.diff(frame.lower) >= -1e-12)
                and np.all(np.diff(frame.upper) <= 1e-12)
            ),
            include_groups=False,
        )
    )
    expected_tasks = len(scores) * len(seeds) * len(policies)
    expected_checkpoint_rows = expected_tasks * checkpoints.audit.nunique()
    if len(completion) != expected_tasks or len(checkpoints) != expected_checkpoint_rows:
        raise RuntimeError("the aggregate panel is incomplete")
    if probability_error > 1e-12 or srs_error > 1e-12:
        raise RuntimeError("a saved acquisition probability is inconsistent")
    if floor_margin < -1e-12:
        raise RuntimeError("PAS violated its proportional remaining-record floor")
    if prefix_conflicts:
        raise RuntimeError("paired paths do not share within-row record prefixes")
    if not nesting.all():
        raise RuntimeError("a checkpoint confidence sequence is not nested")

    paired = checkpoints.pivot_table(
        index=["score_id", "seed", "audit"],
        columns="policy_id",
        values="width",
    ).reset_index()
    paired["absolute_reduction"] = paired[POLICY_SRS] - paired[POLICY_PAS]
    paired["relative_reduction_percent"] = 100.0 * (
        1.0 - paired[POLICY_PAS] / paired[POLICY_SRS]
    )
    _atomic_csv(paired, output / "paired_width_effects.csv.gz")
    verification = {
        "status": "pass",
        "method": "finite-envelope direct betting confidence sequence",
        "tasks": expected_tasks,
        "scores": list(scores),
        "seeds": list(seeds),
        "policies": list(policies),
        "checkpoint_rows": len(checkpoints),
        "action_rows": len(actions),
        "paths_missing_truth_at_any_checkpoint": int(
            checkpoints.groupby(["score_id", "seed", "policy_id"])
            .truth_covered.all().eq(False).sum()
        ),
        "paths_with_empty_grid_event": int(
            checkpoints.groupby(["score_id", "seed", "policy_id"])
            .empty_grid_events.max().gt(0).sum()
        ),
        "maximum_probability_sum_error": probability_error,
        "maximum_srs_selected_probability_error": srs_error,
        "minimum_pas_floor_margin": float(floor_margin),
        "shared_within_row_prefixes": shared_prefixes,
        "within_row_record_conflicts": prefix_conflicts,
        "all_checkpoint_intervals_nested": bool(nesting.all()),
    }
    _atomic_json(verification, output / "VERIFICATION.json")
    return verification


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _parse_csv_text(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run", "aggregate", "all"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--seeds", type=_parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--scores", type=_parse_csv_text, default=SCORES)
    parser.add_argument("--policies", type=_parse_csv_text, default=POLICIES)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--checkpoints", type=_parse_csv_ints, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--pas-variant",
        choices=tuple(PAS_VARIANTS),
        default="robust70",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = tuple(args.scores)
    policies = tuple(args.policies)
    seeds = tuple(args.seeds)
    config = PAS_VARIANTS[args.pas_variant]
    if set(scores).difference(SCORES):
        raise ValueError("unknown score requested")
    if set(policies).difference(POLICIES):
        raise ValueError("unknown policy requested")
    if max(args.checkpoints) > args.budget:
        raise ValueError("all checkpoints must be at or below the budget")
    if args.stage == "preflight":
        seeds = seeds[:1]
        args.budget = 500
        args.checkpoints = (50, 250, 500)
        args.workers = 1
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_OUTPUT / "preflight"

    if args.stage in ("preflight", "run", "all"):
        population = load_cifar10n_population(args.data)
        taxonomy = load_taxonomy_inputs(args.taxonomy)
        lock = {
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "direct finite-envelope betting confidence sequence",
            "comparison": "PAS versus uniform remaining-record SRS with identical anytime-valid inference",
            "score_structure": {
                "worst3_precision": "minimum over all 120 three-row mean precisions",
                "worst3_taxonomic_risk": "maximum over all 120 three-row mean normalized WordNet risks",
            },
            "scores": list(scores),
            "policies": list(policies),
            "seeds": list(seeds),
            "budget": args.budget,
            "checkpoints": list(args.checkpoints),
            "confidence_at_least": 1.0 - config.alpha,
            "config": config.__dict__,
            "pas_variant": args.pas_variant,
            "records": population.n,
            "row_sizes": population.row_sizes.tolist(),
            "data": {"path": str(args.data.relative_to(PROJECT_ROOT)), "sha256": _sha256(args.data)},
            "taxonomy": {
                "path": str(args.taxonomy.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(args.taxonomy),
                "matrix_sha256": taxonomy.matrix_sha256,
            },
            "code_sha256": {
                "audit": _sha256(PROJECT_ROOT / "auditcs/cifar_tail_betting_cs.py"),
                "runner": _sha256(Path(__file__)),
            },
        }
        _atomic_json(lock, args.output / "RUN_LOCK.json")
        tasks = [
            (score_id, seed, policy)
            for score_id in scores
            for seed in seeds
            for policy in policies
        ]
        call = dict(
            budget=args.budget,
            checkpoints=tuple(args.checkpoints),
            output=args.output,
            data_path=args.data,
            taxonomy_path=args.taxonomy,
            config=config,
            overwrite=args.overwrite,
        )
        if args.workers == 1:
            for score_id, seed, policy in tasks:
                print(json.dumps(run_task(score_id=score_id, seed=seed, policy=policy, **call)), flush=True)
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
                futures = {
                    pool.submit(
                        run_task,
                        score_id=score_id,
                        seed=seed,
                        policy=policy,
                        **call,
                    ): (score_id, seed, policy)
                    for score_id, seed, policy in tasks
                }
                for future in as_completed(futures):
                    score_id, seed, policy = futures[future]
                    print(
                        json.dumps({"task": [score_id, seed, policy], **future.result()}),
                        flush=True,
                    )

    if args.stage in ("aggregate", "all"):
        verification = aggregate(
            args.output,
            scores=scores,
            seeds=seeds,
            policies=policies,
            data_path=args.data,
            pas_floor_weight=(
                config.srs_floor
                if config.allocation_mode == "active_neyman"
                else config.proportional_weight
            ),
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
