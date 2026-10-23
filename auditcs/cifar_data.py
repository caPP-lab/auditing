"""CIFAR-10N Phase-2 scores and predictable audit-policy replay."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .categorical_ppr_linear import LinearMatrixScore
from .population import AuditPopulation


CIFAR10_LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
VEHICLE_CODES = (0, 1, 8, 9)
ANIMAL_CODES = (2, 3, 4, 5, 6, 7)


def load_cifar10n_population(
    path: str | Path,
    *,
    submitted_field: str = "random_label1",
    reference_field: str = "clean_label",
) -> AuditPopulation:
    source = np.load(Path(path), allow_pickle=True).item()
    if submitted_field not in source or reference_field not in source:
        raise KeyError("requested CIFAR-10N label field is absent")
    submitted = np.asarray(source[submitted_field], dtype=int)
    reference = np.asarray(source[reference_field], dtype=int)
    full = tuple(range(len(CIFAR10_LABELS)))
    return AuditPopulation(
        submitted=submitted,
        verified=reference,
        labels=CIFAR10_LABELS,
        row_labels=CIFAR10_LABELS,
        row_supports=tuple(full for _ in CIFAR10_LABELS),
        name=f"CIFAR-10N {submitted_field} versus {reference_field}",
    )


def canonical_cifar_scores(population: AuditPopulation) -> dict[str, LinearMatrixScore]:
    if population.n_rows != 10 or population.n_categories != 10:
        raise ValueError("canonical CIFAR scores require a 10 by 10 confusion matrix")
    disagreement = np.ones((10, 10), dtype=int) - np.eye(10, dtype=int)

    group = np.empty(10, dtype=int)
    group[list(VEHICLE_CODES)] = 0
    group[list(ANIMAL_CODES)] = 1
    semantic = np.zeros((10, 10), dtype=int)
    for row in range(10):
        for destination in range(10):
            if row == destination:
                semantic[row, destination] = 0
            elif group[row] == group[destination]:
                semantic[row, destination] = 1
            else:
                semantic[row, destination] = 2

    airplane_precision = np.zeros((10, 10), dtype=int)
    airplane_precision[0, 0] = 1
    return {
        "overall_disagreement": LinearMatrixScore(
            score_id="overall_disagreement",
            integer_costs=disagreement,
            denominator=population.n,
            description="submitted/reference disagreement fraction",
            higher_is_better=False,
        ),
        "symmetric_semantic_severity": LinearMatrixScore(
            score_id="symmetric_semantic_severity",
            integer_costs=semantic,
            denominator=2 * population.n,
            description=(
                "normalized severity: exact=0, within animal/vehicle superclass=1/2, "
                "cross-superclass=1"
            ),
            higher_is_better=False,
        ),
        "airplane_class_precision": LinearMatrixScore(
            score_id="airplane_class_precision",
            integer_costs=airplane_precision,
            denominator=int(population.row_sizes[0]),
            description=(
                "precision of submitted airplane labels relative to the designated "
                "reference; class 0 was frozen before policy screening"
            ),
            higher_is_better=True,
        ),
    }


@dataclass(frozen=True)
class CanonicalPolicy:
    policy_id: str
    selection: Literal["srs", "pas", "posterior_presence", "oracle"]
    pilot_per_row: int = 5
    srs_floor: float = 0.0
    max_row_fraction: float | None = 0.5
    declared_support: bool = False
    implementable: bool = True

    def __post_init__(self) -> None:
        if self.pilot_per_row < 0:
            raise ValueError("pilot_per_row must be nonnegative")
        if not 0.0 <= self.srs_floor <= 1.0:
            raise ValueError("srs_floor must lie in [0, 1]")
        if self.selection == "oracle" and self.implementable:
            raise ValueError("oracle policy must be marked non-implementable")


def canonical_policy_catalog() -> dict[str, CanonicalPolicy]:
    return {
        "global_srs": CanonicalPolicy(
            "global_srs", "srs", pilot_per_row=0, max_row_fraction=None
        ),
        "pilot_then_srs": CanonicalPolicy(
            "pilot_then_srs", "srs", pilot_per_row=5, max_row_fraction=None
        ),
        "vanilla_pas": CanonicalPolicy("vanilla_pas", "pas"),
        "posterior_presence_pas": CanonicalPolicy(
            "posterior_presence_pas", "posterior_presence"
        ),
        "declared_support_pas": CanonicalPolicy(
            "declared_support_pas", "pas", declared_support=True
        ),
        "pas_srs75": CanonicalPolicy("pas_srs75", "pas", srs_floor=0.75),
        "pas_srs90": CanonicalPolicy("pas_srs90", "pas", srs_floor=0.90),
        "oracle_pas": CanonicalPolicy(
            "oracle_pas", "oracle", implementable=False
        ),
    }


@dataclass(frozen=True)
class RowSelectionDecision:
    row: int
    probabilities: np.ndarray
    selected_probability: float
    reason: str
    randomized_tie_size: int
    uses_oracle: bool


def _srs_distribution(row_sizes: np.ndarray, audits: np.ndarray) -> np.ndarray:
    remaining = np.maximum(0, row_sizes - audits).astype(float)
    if remaining.sum() <= 0:
        raise StopIteration
    return remaining / remaining.sum()


def _planning_radius(
    row_size: int,
    audits: int,
    *,
    effective_support: float,
    maximum_support: float,
    n_rows: int,
    maximum_row_size: int,
    alpha: float,
) -> float:
    if audits >= row_size:
        return 0.0
    m = max(1, audits)
    rho = 1.0 - (m - 1.0) / row_size
    log_term = np.log(
        2.0 * n_rows * max(1.0, maximum_support) * maximum_row_size / alpha
    )
    return float(
        2.0
        * max(1.0, effective_support)
        * np.sqrt(max(0.0, rho) * log_term / (2.0 * m))
    )


def _posterior_presence_priority(
    row_sizes: np.ndarray,
    audits: np.ndarray,
    counts: np.ndarray,
    score: LinearMatrixScore,
) -> tuple[np.ndarray, np.ndarray]:
    rows, categories = counts.shape
    priorities = np.zeros(rows, dtype=float)
    support_sizes = np.zeros(rows, dtype=float)
    for row in range(rows):
        prior = np.full(categories, 0.5 / max(1, categories - 1), dtype=float)
        prior[row] = 0.5
        posterior = counts[row] + prior
        posterior /= posterior.sum()
        horizon = max(1, min(int(row_sizes[row] - audits[row]), int(np.ceil(np.sqrt(row_sizes[row])))))
        presence = np.where(
            counts[row] > 0,
            1.0,
            1.0 - np.power(1.0 - posterior, horizon),
        )
        support_sizes[row] = max(1.0, float(presence.sum()))
        levels = np.unique(score.integer_costs[row])
        level_presence = np.asarray(
            [1.0 - np.prod(1.0 - presence[score.integer_costs[row] == level]) for level in levels]
        )
        expected_range = 0.0
        for mask in range(1, 1 << levels.size):
            selected = np.asarray([(mask >> index) & 1 for index in range(levels.size)], dtype=bool)
            probability = float(
                np.prod(np.where(selected, level_presence, 1.0 - level_presence))
            )
            if selected.sum() >= 2:
                expected_range += probability * float(np.ptp(levels[selected]))
        priorities[row] = (
            row_sizes[row] * expected_range / (2.0 * score.denominator)
        )
    return priorities, support_sizes


def _oracle_priority(
    row_sizes: np.ndarray,
    score: LinearMatrixScore,
    true_confusion: np.ndarray,
) -> np.ndarray:
    priorities = np.zeros(row_sizes.size, dtype=float)
    for row, size in enumerate(row_sizes):
        probabilities = true_confusion[row] / size
        costs = score.integer_costs[row].astype(float)
        mean = float(probabilities @ costs)
        variance = float(probabilities @ np.square(costs - mean))
        priorities[row] = size * np.sqrt(max(0.0, variance)) / score.denominator
    return priorities


def selection_probabilities(
    policy: CanonicalPolicy,
    row_sizes: np.ndarray,
    audits: np.ndarray,
    counts: np.ndarray,
    score: LinearMatrixScore,
    *,
    alpha: float,
    true_confusion: np.ndarray | None = None,
) -> tuple[np.ndarray, str, int, bool, np.ndarray, np.ndarray]:
    """Return a predictable row distribution and its planning diagnostics."""

    row_sizes = np.asarray(row_sizes, dtype=int)
    audits = np.asarray(audits, dtype=int)
    srs = _srs_distribution(row_sizes, audits)
    if policy.selection == "srs":
        return srs, "residual_global_srs", 0, False, np.zeros_like(srs), np.full_like(srs, 10.0)

    caps = row_sizes.copy()
    if policy.max_row_fraction is not None:
        caps = np.maximum(1, np.floor(policy.max_row_fraction * row_sizes)).astype(int)
    available = np.flatnonzero((audits < row_sizes) & (audits < caps))
    if available.size == 0:
        raise StopIteration
    eligible_srs = np.zeros_like(srs)
    eligible_srs[available] = srs[available]
    eligible_srs /= eligible_srs.sum()

    if policy.selection == "posterior_presence":
        priorities, support_sizes = _posterior_presence_priority(
            row_sizes, audits, counts, score
        )
        reason = "posterior_support_presence_pas"
        uses_oracle = False
    elif policy.selection == "oracle":
        if true_confusion is None:
            raise ValueError("oracle selection requires the full confusion matrix")
        priorities = _oracle_priority(row_sizes, score, true_confusion)
        support_sizes = np.count_nonzero(true_confusion, axis=1).astype(float)
        reason = "truth_informed_variance_pas"
        uses_oracle = True
    else:
        priorities = score.row_influences(row_sizes)
        support_sizes = np.full(row_sizes.size, score.integer_costs.shape[1], dtype=float)
        reason = "declared_support_pas" if policy.declared_support else "vanilla_pas"
        uses_oracle = False

    maximum_support = float(support_sizes.max(initial=1.0))
    gains = np.full(row_sizes.size, -np.inf)
    for row in available:
        current = _planning_radius(
            int(row_sizes[row]),
            int(audits[row]),
            effective_support=float(support_sizes[row]),
            maximum_support=maximum_support,
            n_rows=row_sizes.size,
            maximum_row_size=int(row_sizes.max()),
            alpha=alpha,
        )
        future = _planning_radius(
            int(row_sizes[row]),
            int(audits[row] + 1),
            effective_support=float(support_sizes[row]),
            maximum_support=maximum_support,
            n_rows=row_sizes.size,
            maximum_row_size=int(row_sizes.max()),
            alpha=alpha,
        )
        gains[row] = priorities[row] * max(0.0, current - future)
    maximum = float(np.max(gains[available]))
    ties = available[np.isclose(gains[available], maximum, rtol=1e-12, atol=1e-15)]
    if not np.isfinite(maximum) or maximum <= 0.0:
        pas = eligible_srs
        reason = f"{reason}_srs_fallback"
        tie_size = 0
    else:
        pas = np.zeros(row_sizes.size, dtype=float)
        pas[int(ties[0])] = 1.0
        tie_size = int(ties.size)
    distribution = policy.srs_floor * eligible_srs + (1.0 - policy.srs_floor) * pas
    distribution[audits >= row_sizes] = 0.0
    distribution /= distribution.sum()
    if policy.srs_floor:
        reason = f"{reason}_with_{int(100 * policy.srs_floor)}pct_srs_floor"
    return distribution, reason, tie_size, uses_oracle, priorities, support_sizes


def _stable_policy_seed(path_seed: int, policy_id: str) -> int:
    digest = sha256(f"{path_seed}:{policy_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def row_destination_orders(
    population: AuditPopulation,
    path_seed: int,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(path_seed)
    return tuple(
        population.verified[rng.permutation(indices)]
        for indices in population.row_indices
    )


@dataclass(frozen=True)
class PolicyReplay:
    checkpoints: pd.DataFrame
    allocations: pd.DataFrame
    actions: pd.DataFrame


def replay_policy(
    population: AuditPopulation,
    score: LinearMatrixScore,
    policy: CanonicalPolicy,
    *,
    path_seed: int,
    checkpoints: Sequence[int],
    alpha: float = 0.05,
    record_actions: bool = False,
) -> PolicyReplay:
    checkpoints = tuple(sorted(set(int(value) for value in checkpoints)))
    if not checkpoints or checkpoints[0] <= 0 or checkpoints[-1] > population.n:
        raise ValueError("checkpoints must lie inside the population")
    orders = row_destination_orders(population, path_seed)
    row_sizes = population.row_sizes
    audits = np.zeros(population.n_rows, dtype=int)
    counts = np.zeros((population.n_rows, population.n_categories), dtype=int)
    rng = np.random.default_rng(_stable_policy_seed(path_seed, policy.policy_id))
    checkpoint_records: list[dict[str, object]] = []
    allocation_records: list[dict[str, object]] = []
    action_records: list[dict[str, object]] = []
    checkpoint_set = set(checkpoints)

    for audit_index in range(1, checkpoints[-1] + 1):
        pilot_rows = np.flatnonzero(audits < np.minimum(policy.pilot_per_row, row_sizes))
        if pilot_rows.size:
            row = int(pilot_rows[0])
            probabilities = np.zeros(population.n_rows, dtype=float)
            probabilities[row] = 1.0
            reason = "fixed_five_per_row_pilot"
            tie_size = 1
            uses_oracle = False
            priorities = np.zeros(population.n_rows)
            support_sizes = np.full(population.n_rows, population.n_categories, dtype=float)
        else:
            (
                probabilities,
                reason,
                tie_size,
                uses_oracle,
                priorities,
                support_sizes,
            ) = selection_probabilities(
                policy,
                row_sizes,
                audits,
                counts,
                score,
                alpha=alpha,
                true_confusion=(
                    population.true_confusion if policy.selection == "oracle" else None
                ),
            )
            row = int(rng.choice(population.n_rows, p=probabilities))
        destination = int(orders[row][audits[row]])
        selected_probability = float(probabilities[row])
        within_row_audit = int(audits[row] + 1)
        counts[row, destination] += 1
        audits[row] += 1

        if record_actions:
            record: dict[str, object] = {
                "seed": path_seed,
                "score_id": score.score_id,
                "policy_id": policy.policy_id,
                "audit": audit_index,
                "selected_row": row,
                "selected_probability": selected_probability,
                "selection_reason": reason,
                "randomized_tie_size": tie_size,
                "uses_oracle": uses_oracle,
                "within_row_audit": within_row_audit,
                "observed_destination": destination,
            }
            record.update(
                {f"probability_row_{index}": value for index, value in enumerate(probabilities)}
            )
            action_records.append(record)

        if audit_index in checkpoint_set:
            checkpoint_records.append(
                {
                    "seed": path_seed,
                    "score_id": score.score_id,
                    "policy_id": policy.policy_id,
                    "audit": audit_index,
                    "population_fraction": audit_index / population.n,
                    "pilot_complete": bool(
                        np.all(audits >= np.minimum(policy.pilot_per_row, row_sizes))
                    ),
                    "implementable": policy.implementable,
                    "oracle_status": "oracle_only" if not policy.implementable else "none",
                }
            )
            for row_index in range(population.n_rows):
                allocation_records.append(
                    {
                        "seed": path_seed,
                        "score_id": score.score_id,
                        "policy_id": policy.policy_id,
                        "audit": audit_index,
                        "row": row_index,
                        "row_label": population.row_labels[row_index],
                        "row_audits": int(audits[row_index]),
                        "row_fraction": float(audits[row_index] / row_sizes[row_index]),
                        "planning_priority": float(priorities[row_index]),
                        "effective_support_size": float(support_sizes[row_index]),
                    }
                )

    return PolicyReplay(
        checkpoints=pd.DataFrame.from_records(checkpoint_records),
        allocations=pd.DataFrame.from_records(allocation_records),
        actions=pd.DataFrame.from_records(action_records),
    )
