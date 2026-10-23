#!/usr/bin/env python3
"""Run the revised favorable synthetic direct-betting-CS experiment."""

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

from auditcs.synthetic_linear_pas import (
    POLICIES,
    POLICY_PAS,
    POLICY_SRS,
    SyntheticLinearRiskAudit,
    SyntheticRiskConfig,
    initial_oracle_variances,
    make_synthetic_risk_scenario,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "synthetic"
DEVELOPMENT_ROWS = (8, 12, 20, 32, 48)
ACTIVE_ROWS = 4
ROW_SIZE = 250
BUDGET = 800
CHECKPOINTS = tuple(range(25, BUDGET + 1, 25))
DEVELOPMENT_SEEDS = tuple(range(2_026_092_400, 2_026_092_405))
CONFIRMATION_SEEDS = tuple(range(2_026_101_000, 2_026_101_100))
CONFIRMATION_ROWS = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
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
        compression={"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None,
    )
    os.replace(temporary, path)


def _task_dir(output: Path, stage: str, rows: int, seed: int, policy: str) -> Path:
    return output / stage / "shards" / f"rows_{rows:02d}" / f"seed_{seed}__{policy}"


def _run_task(
    output: Path,
    stage: str,
    rows: int,
    seed: int,
    policy: str,
    overwrite: bool = False,
) -> dict[str, object]:
    task = _task_dir(output, stage, rows, seed, policy)
    complete = task / "COMPLETE.json"
    if complete.is_file() and not overwrite:
        return json.loads(complete.read_text(encoding="utf-8"))
    scenario = make_synthetic_risk_scenario(
        rows, active_rows=ACTIVE_ROWS, row_size=ROW_SIZE
    )
    audit = SyntheticLinearRiskAudit(
        scenario,
        seed=seed,
        policy=policy,
        config=SyntheticRiskConfig(),
    )
    checkpoint_set = set(CHECKPOINTS)
    records: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    started = perf_counter()
    for review in range(1, BUDGET + 1):
        request = audit.prepare()
        if request is None:
            break
        probabilities = np.asarray(request["row_probabilities"], dtype=float)
        row = int(request["row"])
        observed = audit.observe(int(request["record_id"]))
        interval = audit.interval()
        actions.append(
            {
                "stage": stage,
                "rows": rows,
                "active_rows": ACTIVE_ROWS,
                "seed": seed,
                "policy_id": policy,
                "audit": review,
                "record_id": int(request["record_id"]),
                "row": row,
                "within_row_audit": int(audit.audits[row]),
                "influential": bool(row in scenario.influential_indices),
                "contribution": float(observed["value"]),
                "observation": float(observed["observation"]),
                "selected_probability": float(probabilities[row]),
                "probability_sum": float(probabilities.sum()),
                "plugin": float(request["plugin"]),
                "observation_lower": float(request["lower_range"]),
                "observation_upper": float(request["upper_range"]),
                "selected_weight": float(request["selected_weight"]),
            }
        )
        if review in checkpoint_set:
            records.append(
                {
                    "stage": stage,
                    "rows": rows,
                    "active_rows": ACTIVE_ROWS,
                    "row_size": ROW_SIZE,
                    "population_size": scenario.population_size,
                    "seed": seed,
                    "policy_id": policy,
                    "policy": "SRS" if policy == POLICY_SRS else "PAS",
                    "audit": review,
                    "lower": float(interval["lower"]),
                    "upper": float(interval["upper"]),
                    "width": float(interval["width"]),
                    "truth": scenario.truth,
                    "truth_covered": bool(
                        float(interval["lower"])
                        <= scenario.truth
                        <= float(interval["upper"])
                    ),
                    "active_audits": int(
                        audit.audits[list(scenario.influential_indices)].sum()
                    ),
                    "active_audit_fraction": float(
                        audit.audits[list(scenario.influential_indices)].sum()
                        / review
                    ),
                    "rows_audited": int(np.sum(audit.audits > 0)),
                    "elapsed_seconds": perf_counter() - started,
                }
            )
    if not records or int(records[-1]["audit"]) != BUDGET:
        raise RuntimeError(f"task {stage}/{rows}/{seed}/{policy} ended early")
    task.mkdir(parents=True, exist_ok=True)
    checkpoints = pd.DataFrame(records)
    action_frame = pd.DataFrame(actions)
    _atomic_csv(checkpoints, task / "checkpoints.csv.gz")
    _atomic_csv(action_frame, task / "actions.csv.gz")
    allocation = np.bincount(action_frame.row, minlength=rows)
    metadata = {
        "status": "complete",
        "stage": stage,
        "rows": rows,
        "active_rows": ACTIVE_ROWS,
        "row_size": ROW_SIZE,
        "seed": seed,
        "policy_id": policy,
        "budget": BUDGET,
        "truth": scenario.truth,
        "terminal_width": float(checkpoints.iloc[-1].width),
        "truth_covered_all_checkpoints": bool(checkpoints.truth_covered.all()),
        "row_audits": allocation.tolist(),
        "influential_indices": list(scenario.influential_indices),
        "runtime_seconds": perf_counter() - started,
    }
    _atomic_json(complete, metadata)
    return metadata


def _aggregate(output: Path, stage: str, rows_grid: tuple[int, ...], seeds: tuple[int, ...]) -> dict[str, object]:
    checkpoint_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    completions: list[dict[str, object]] = []
    for rows in rows_grid:
        for seed in seeds:
            for policy in POLICIES:
                task = _task_dir(output, stage, rows, seed, policy)
                if not (task / "COMPLETE.json").is_file():
                    raise RuntimeError(f"missing task {stage}/{rows}/{seed}/{policy}")
                completions.append(
                    json.loads((task / "COMPLETE.json").read_text(encoding="utf-8"))
                )
                checkpoint_frames.append(pd.read_csv(task / "checkpoints.csv.gz"))
                action_frames.append(pd.read_csv(task / "actions.csv.gz"))
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)
    completion = pd.DataFrame(completions)
    stage_dir = output / stage
    _atomic_csv(checkpoints, stage_dir / "checkpoints.csv.gz")
    _atomic_csv(actions, stage_dir / "actions.csv.gz")
    _atomic_csv(completion, stage_dir / "completion.csv")
    terminal = checkpoints[checkpoints.audit == BUDGET].pivot_table(
        index=["rows", "seed"], columns="policy_id", values="width"
    ).reset_index()
    terminal["absolute_reduction"] = terminal[POLICY_SRS] - terminal[POLICY_PAS]
    terminal["relative_reduction_percent"] = (
        100.0 * terminal.absolute_reduction / terminal[POLICY_SRS]
    )
    _atomic_csv(terminal, stage_dir / "paired_terminal.csv")
    return {
        "checkpoints": checkpoints,
        "actions": actions,
        "completion": completion,
        "paired": terminal,
    }


def _validate(
    output: Path,
    development: dict[str, object],
    confirmation: dict[str, object],
) -> None:
    checkpoints = pd.concat(
        [development["checkpoints"], confirmation["checkpoints"]],
        ignore_index=True,
    )
    actions = pd.concat(
        [development["actions"], confirmation["actions"]], ignore_index=True
    )
    path_coverage = (
        checkpoints.groupby(["stage", "rows", "seed", "policy_id"])
        .truth_covered.all()
    )
    nesting_failures = 0
    for _, group in checkpoints.groupby(["stage", "rows", "seed", "policy_id"]):
        ordered = group.sort_values("audit")
        if np.any(np.diff(ordered.lower) < -1e-12) or np.any(
            np.diff(ordered.upper) > 1e-12
        ):
            nesting_failures += 1
    if nesting_failures:
        raise RuntimeError("a stored confidence sequence was not nested")
    probability_error = float(np.max(np.abs(actions.probability_sum - 1.0)))
    if probability_error > 1e-12:
        raise RuntimeError("selection probabilities did not sum to one")
    conflicts = 0
    prefix: dict[tuple[str, int, int, int, int], int] = {}
    for record in actions.itertuples(index=False):
        key = (
            str(record.stage),
            int(record.rows),
            int(record.seed),
            int(record.row),
            int(record.within_row_audit),
        )
        old = prefix.setdefault(key, int(record.record_id))
        conflicts += old != int(record.record_id)
    if conflicts:
        raise RuntimeError("paired policies did not share within-row record prefixes")
    scenario = make_synthetic_risk_scenario(
        CONFIRMATION_ROWS, active_rows=ACTIVE_ROWS, row_size=ROW_SIZE
    )
    proportional, neyman = initial_oracle_variances(scenario)
    verification = {
        "status": "pass",
        "method": "direct finite-population betting confidence sequence for a linear score",
        "development_rows": list(DEVELOPMENT_ROWS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_rows": CONFIRMATION_ROWS,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "active_rows": ACTIVE_ROWS,
        "row_size": ROW_SIZE,
        "budget": BUDGET,
        "checkpoint_rows": int(len(checkpoints)),
        "action_rows": int(len(actions)),
        "checkpoint_intervals_missing_truth": int((~checkpoints.truth_covered).sum()),
        "paths_ever_missing_truth": int((~path_coverage).sum()),
        "pathwise_simultaneous_coverage": float(path_coverage.mean()),
        "nesting_failures": nesting_failures,
        "maximum_probability_sum_error": probability_error,
        "within_row_prefix_conflicts": conflicts,
        "initial_oracle_proportional_variance": proportional,
        "initial_oracle_neyman_variance": neyman,
        "initial_oracle_variance_ratio": proportional / neyman,
    }
    _atomic_json(output / "VERIFICATION.json", verification)


def _write_population(output: Path) -> None:
    scenario = make_synthetic_risk_scenario(
        CONFIRMATION_ROWS, active_rows=ACTIVE_ROWS, row_size=ROW_SIZE
    )
    _atomic_json(
        output / "CONFIRMATION_POPULATION.json",
        {
            "rows": scenario.rows,
            "active_rows": scenario.active_rows,
            "row_size": scenario.row_size,
            "population_size": scenario.population_size,
            "truth": scenario.truth,
            "influential_indices": scenario.influential_indices,
            "row_labels": scenario.row_labels,
            "category_labels": scenario.category_labels,
            "confusion": scenario.confusion.tolist(),
            "row_support_values": scenario.row_support_values,
            "confusion_sha256": hashlib.sha256(
                scenario.confusion.astype(np.int64).tobytes()
            ).hexdigest(),
        },
    )


def _run_tasks(
    output: Path,
    tasks: list[tuple[str, int, int, str]],
    *,
    workers: int,
    overwrite: bool,
) -> None:
    call = lambda task: _run_task(output, *task, overwrite=overwrite)
    started = perf_counter()
    if workers == 1:
        for index, task in enumerate(tasks, 1):
            call(task)
            if index % max(1, len(tasks) // 10) == 0:
                print(f"[{index}/{len(tasks)}] elapsed={perf_counter()-started:.1f}s", flush=True)
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_run_task, output, *task, overwrite): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % max(1, len(tasks) // 20) == 0 or index == len(tasks):
                elapsed = perf_counter() - started
                eta = elapsed / index * (len(tasks) - index)
                print(f"[{index}/{len(tasks)}] elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)


def run_all(output: Path, *, workers: int, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write_population(output)
    lock = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_rows": DEVELOPMENT_ROWS,
        "active_rows": ACTIVE_ROWS,
        "row_size": ROW_SIZE,
        "budget": BUDGET,
        "checkpoints": CHECKPOINTS,
        "development_seeds": DEVELOPMENT_SEEDS,
        "confirmation_rows": CONFIRMATION_ROWS,
        "confirmation_seeds": CONFIRMATION_SEEDS,
        "policies": POLICIES,
        "score": "population-wide linear taxonomic risk in [0,1]",
        "source_hashes": {
            name: _sha256(PROJECT_ROOT / name)
            for name in (
                "auditcs/synthetic_linear_pas.py",
                "scripts/run_synthetic_linear_pas_study.py",
            )
        },
    }
    _atomic_json(output / "RUN_LOCK.json", lock)
    development_tasks = [
        ("development", rows, seed, policy)
        for rows in DEVELOPMENT_ROWS
        for seed in DEVELOPMENT_SEEDS
        for policy in POLICIES
    ]
    _run_tasks(output, development_tasks, workers=workers, overwrite=overwrite)
    development = _aggregate(output, "development", DEVELOPMENT_ROWS, DEVELOPMENT_SEEDS)
    confirmation_tasks = [
        ("confirmation", CONFIRMATION_ROWS, seed, policy)
        for seed in CONFIRMATION_SEEDS
        for policy in POLICIES
    ]
    _run_tasks(output, confirmation_tasks, workers=workers, overwrite=overwrite)
    confirmation = _aggregate(
        output, "confirmation", (CONFIRMATION_ROWS,), CONFIRMATION_SEEDS
    )
    _validate(output, development, confirmation)
    _atomic_json(
        output / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_tasks": len(development_tasks),
            "confirmation_tasks": len(confirmation_tasks),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all",))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_all(arguments.output, workers=arguments.workers, overwrite=arguments.overwrite)
