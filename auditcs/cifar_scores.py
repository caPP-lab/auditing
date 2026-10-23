"""Phase-3 CIFAR multimetric scores and predictable audit policies."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .cifar_data import row_destination_orders
from .population import AuditPopulation
from .taxonomy import load_cifar10_taxonomy


ScoreFamily = Literal[
    "tail_precision",
    "tail_taxonomic",
    "tail_neighborhood",
    "rms_error",
    "rms_taxonomic",
    "harmonic_precision",
    "macro_precision",
    "mean_taxonomic",
    "overall_disagreement",
    "weighted_kappa",
]


@dataclass(frozen=True, eq=False)
class MultimetricScore:
    score_id: str
    family: ScoreFamily
    description: str
    higher_is_better: bool
    k: int = 3
    tail: Literal["bottom", "top"] | None = None


@dataclass(frozen=True)
class TaxonomyInputs:
    integer_distances: np.ndarray
    d_max: int
    neighborhoods: tuple[tuple[int, ...], ...]
    artifact_sha256: str
    matrix_sha256: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.integer_distances, dtype=int)
        if matrix.shape != (10, 10):
            raise ValueError("CIFAR taxonomy matrix must be 10 by 10")
        frozen = np.array(matrix, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "integer_distances", frozen)


def load_taxonomy_inputs(path: str | Path) -> TaxonomyInputs:
    artifact = load_cifar10_taxonomy(path)
    return TaxonomyInputs(
        integer_distances=np.asarray(artifact["integer_distance_matrix"], dtype=int),
        d_max=int(artifact["d_max"]),
        neighborhoods=tuple(
            tuple(int(value) for value in row) for row in artifact["neighborhoods"]
        ),
        artifact_sha256=sha256(Path(path).read_bytes()).hexdigest(),
        matrix_sha256=str(artifact["integer_distance_matrix_sha256"]),
    )


def canonical_multimetric_scores() -> dict[str, MultimetricScore]:
    scores = (
        MultimetricScore(
            "worst3_precision",
            "tail_precision",
            "mean precision of the three least precise submitted classes",
            True,
            tail="bottom",
        ),
        MultimetricScore(
            "best3_precision",
            "tail_precision",
            "mean precision of the three most precise submitted classes",
            True,
            tail="top",
        ),
        MultimetricScore(
            "worst3_taxonomic_risk",
            "tail_taxonomic",
            "mean WordNet risk of the three highest-risk submitted classes",
            False,
            tail="top",
        ),
        MultimetricScore(
            "best3_taxonomic_risk",
            "tail_taxonomic",
            "mean WordNet risk of the three lowest-risk submitted classes",
            False,
            tail="bottom",
        ),
        MultimetricScore(
            "worst3_neighborhood_accuracy",
            "tail_neighborhood",
            "mean semantic-neighborhood accuracy of the three weakest classes",
            True,
            tail="bottom",
        ),
        MultimetricScore(
            "best3_neighborhood_accuracy",
            "tail_neighborhood",
            "mean semantic-neighborhood accuracy of the three strongest classes",
            True,
            tail="top",
        ),
        MultimetricScore(
            "rms_class_error",
            "rms_error",
            "population-weighted RMS class-conditional error",
            False,
        ),
        MultimetricScore(
            "rms_taxonomic_risk",
            "rms_taxonomic",
            "population-weighted RMS class-conditional WordNet risk",
            False,
        ),
        MultimetricScore(
            "harmonic_macro_precision",
            "harmonic_precision",
            "harmonic mean of submitted-class precisions",
            True,
        ),
        MultimetricScore(
            "macro_precision",
            "macro_precision",
            "unweighted macro average of submitted-class precisions",
            True,
        ),
        MultimetricScore(
            "mean_taxonomic_risk",
            "mean_taxonomic",
            "population mean submitted/reference WordNet distance",
            False,
        ),
        MultimetricScore(
            "overall_disagreement",
            "overall_disagreement",
            "population submitted/reference disagreement fraction",
            False,
        ),
        MultimetricScore(
            "wordnet_weighted_kappa",
            "weighted_kappa",
            "WordNet-distance-weighted Cohen's kappa",
            True,
        ),
    )
    return {score.score_id: score for score in scores}


def row_functional(score: MultimetricScore) -> str:
    if score.family in {
        "tail_precision",
        "rms_error",
        "harmonic_precision",
        "macro_precision",
        "overall_disagreement",
    }:
        return "precision"
    if score.family in {
        "tail_taxonomic",
        "rms_taxonomic",
        "mean_taxonomic",
    }:
        return "taxonomic"
    if score.family == "tail_neighborhood":
        return "neighborhood"
    if score.family == "weighted_kappa":
        return "weighted_kappa"
    raise ValueError(f"unknown score family: {score.family}")


def row_values(
    confusion: np.ndarray,
    taxonomy: TaxonomyInputs,
) -> dict[str, np.ndarray]:
    matrix = np.asarray(confusion, dtype=float)
    row_sizes = matrix.sum(axis=1)
    if matrix.shape != (10, 10) or np.any(row_sizes <= 0):
        raise ValueError("multimetric scores require ten nonempty CIFAR rows")
    precision = np.diag(matrix) / row_sizes
    taxonomic = np.sum(matrix * taxonomy.integer_distances, axis=1) / (
        row_sizes * taxonomy.d_max
    )
    neighborhood = np.asarray(
        [
            matrix[row, list(taxonomy.neighborhoods[row])].sum() / row_sizes[row]
            for row in range(10)
        ]
    )
    return {
        "precision": precision,
        "taxonomic": taxonomic,
        "neighborhood": neighborhood,
    }


def _tail_mean(values: np.ndarray, k: int, tail: str) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    selected = ordered[:k] if tail == "bottom" else ordered[-k:]
    return float(selected.mean())


def harmonic_macro_precision(values: Sequence[float]) -> float:
    precision = np.asarray(values, dtype=float)
    if np.any(precision <= 0.0):
        return 0.0
    return float(precision.size / np.sum(1.0 / precision))


def weighted_kappa_components(
    confusion: np.ndarray,
    taxonomy: TaxonomyInputs,
) -> tuple[float, float, float]:
    matrix = np.asarray(confusion, dtype=float)
    total = float(matrix.sum())
    submitted = matrix.sum(axis=1) / total
    reference = matrix.sum(axis=0) / total
    distance = taxonomy.integer_distances / taxonomy.d_max
    observed = float(np.sum(matrix * distance) / total)
    expected = float(submitted @ distance @ reference)
    if expected <= 0.0:
        raise ValueError("weighted-kappa chance disagreement must be positive")
    return observed, expected, float(1.0 - observed / expected)


def evaluate_score(
    score: MultimetricScore,
    confusion: np.ndarray,
    taxonomy: TaxonomyInputs,
) -> float:
    values = row_values(confusion, taxonomy)
    row_sizes = np.asarray(confusion).sum(axis=1).astype(float)
    weights = row_sizes / row_sizes.sum()
    if score.family == "tail_precision":
        return _tail_mean(values["precision"], score.k, str(score.tail))
    if score.family == "tail_taxonomic":
        return _tail_mean(values["taxonomic"], score.k, str(score.tail))
    if score.family == "tail_neighborhood":
        return _tail_mean(values["neighborhood"], score.k, str(score.tail))
    if score.family == "rms_error":
        return float(np.sqrt(weights @ np.square(1.0 - values["precision"])))
    if score.family == "rms_taxonomic":
        return float(np.sqrt(weights @ np.square(values["taxonomic"])))
    if score.family == "harmonic_precision":
        return harmonic_macro_precision(values["precision"])
    if score.family == "macro_precision":
        return float(values["precision"].mean())
    if score.family == "mean_taxonomic":
        return float(weights @ values["taxonomic"])
    if score.family == "overall_disagreement":
        return float(1.0 - weights @ values["precision"])
    if score.family == "weighted_kappa":
        return weighted_kappa_components(confusion, taxonomy)[2]
    raise ValueError(f"unknown score family: {score.family}")


def aggregate_interval(
    score: MultimetricScore,
    lower: Sequence[float],
    upper: Sequence[float],
    row_sizes: Sequence[int],
) -> tuple[float, float]:
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    weights = np.asarray(row_sizes, dtype=float)
    weights /= weights.sum()
    if np.any(lower_values > upper_values + 1e-12):
        raise ValueError("row endpoint order is invalid")
    if score.family in {"tail_precision", "tail_taxonomic", "tail_neighborhood"}:
        return (
            _tail_mean(lower_values, score.k, str(score.tail)),
            _tail_mean(upper_values, score.k, str(score.tail)),
        )
    if score.family == "rms_taxonomic":
        return (
            float(np.sqrt(weights @ np.square(lower_values))),
            float(np.sqrt(weights @ np.square(upper_values))),
        )
    if score.family == "rms_error":
        lower_error = 1.0 - upper_values
        upper_error = 1.0 - lower_values
        return (
            float(np.sqrt(weights @ np.square(lower_error))),
            float(np.sqrt(weights @ np.square(upper_error))),
        )
    if score.family == "harmonic_precision":
        return (
            harmonic_macro_precision(lower_values),
            harmonic_macro_precision(upper_values),
        )
    if score.family == "macro_precision":
        return float(lower_values.mean()), float(upper_values.mean())
    if score.family == "mean_taxonomic":
        return float(weights @ lower_values), float(weights @ upper_values)
    if score.family == "overall_disagreement":
        return (
            float(1.0 - weights @ upper_values),
            float(1.0 - weights @ lower_values),
        )
    raise ValueError("weighted kappa requires the fractional projection solver")


def row_integer_costs(
    functional: str,
    taxonomy: TaxonomyInputs,
) -> np.ndarray:
    if functional == "precision":
        return np.eye(10, dtype=int)
    if functional == "taxonomic":
        return np.asarray(taxonomy.integer_distances, dtype=int)
    if functional == "neighborhood":
        costs = np.zeros((10, 10), dtype=int)
        for row, neighbors in enumerate(taxonomy.neighborhoods):
            costs[row, list(neighbors)] = 1
        return costs
    raise ValueError(f"unknown row functional: {functional}")


def normalize_row_numerators(
    functional: str,
    numerators: Sequence[int],
    row_sizes: Sequence[int],
    taxonomy: TaxonomyInputs,
) -> np.ndarray:
    scale = np.asarray(row_sizes, dtype=float)
    if functional == "taxonomic":
        scale *= taxonomy.d_max
    return np.asarray(numerators, dtype=float) / scale


@dataclass(frozen=True)
class MultimetricPolicy:
    policy_id: str
    selection: Literal["srs", "metric", "oracle"]
    pilot_per_row: int = 5
    srs_floor: float = 0.0
    max_row_fraction: float | None = 0.5
    implementable: bool = True


def multimetric_policy_catalog() -> dict[str, MultimetricPolicy]:
    policies = (
        MultimetricPolicy(
            "global_srs", "srs", pilot_per_row=0, max_row_fraction=None
        ),
        MultimetricPolicy(
            "pilot_then_srs", "srs", pilot_per_row=5, max_row_fraction=None
        ),
        MultimetricPolicy("metric_pas", "metric"),
        MultimetricPolicy("metric_pas_srs50", "metric", srs_floor=0.5),
        MultimetricPolicy(
            "oracle_pas", "oracle", implementable=False
        ),
    )
    return {policy.policy_id: policy for policy in policies}


def _stable_seed(*parts: object) -> int:
    digest = sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _srs_distribution(row_sizes: np.ndarray, audits: np.ndarray) -> np.ndarray:
    remaining = np.maximum(0, row_sizes - audits).astype(float)
    if remaining.sum() <= 0:
        raise StopIteration
    return remaining / remaining.sum()


def _planning_radius(row_size: int, audits: int, *, alpha: float) -> float:
    if audits >= row_size:
        return 0.0
    m = max(1, audits)
    rho = 1.0 - (m - 1.0) / row_size
    log_term = np.log(20.0 * 10.0 * int(row_size) / alpha)
    return float(np.sqrt(max(0.0, rho) * log_term / (2.0 * m)))


def _fractional_active_membership(
    values: np.ndarray,
    *,
    k: int,
    tail: str,
) -> np.ndarray:
    ordered = np.sort(values)
    cutoff = ordered[k - 1] if tail == "bottom" else ordered[-k]
    strict = values < cutoff if tail == "bottom" else values > cutoff
    tied = np.isclose(values, cutoff, rtol=0.0, atol=1e-15)
    result = strict.astype(float)
    remaining = k - int(strict.sum())
    result[tied] = remaining / max(1, int(tied.sum()))
    return result


def _posterior_row_samples(
    counts: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 10, 10), dtype=float)
    for row in range(10):
        samples[:, row, :] = rng.dirichlet(counts[row] + 0.5, size=draws)
    return samples


def _tail_membership_probabilities(
    score: MultimetricScore,
    counts: np.ndarray,
    taxonomy: TaxonomyInputs,
    *,
    seed: int,
    draws: int = 512,
) -> np.ndarray:
    samples = _posterior_row_samples(counts, draws=draws, seed=seed)
    if score.family == "tail_precision":
        values = np.diagonal(samples, axis1=1, axis2=2)
    elif score.family == "tail_taxonomic":
        values = np.sum(
            samples * taxonomy.integer_distances[None, :, :], axis=2
        ) / taxonomy.d_max
    elif score.family == "tail_neighborhood":
        values = np.asarray(
            [
                [
                    samples[draw, row, list(taxonomy.neighborhoods[row])].sum()
                    for row in range(10)
                ]
                for draw in range(draws)
            ]
        )
    else:
        raise ValueError("tail membership requested for a non-tail score")
    membership = np.zeros(10, dtype=float)
    for draw_values in values:
        membership += _fractional_active_membership(
            draw_values, k=score.k, tail=str(score.tail)
        )
    return membership / draws


def score_row_priorities(
    score: MultimetricScore,
    row_sizes: np.ndarray,
    counts: np.ndarray,
    taxonomy: TaxonomyInputs,
    *,
    oracle_confusion: np.ndarray | None,
    posterior_seed: int,
) -> np.ndarray:
    weights = row_sizes / row_sizes.sum()
    if oracle_confusion is None:
        probabilities = (counts + 0.5) / (counts.sum(axis=1, keepdims=True) + 5.0)
    else:
        probabilities = oracle_confusion / row_sizes[:, None]
    precision = np.diag(probabilities)
    taxonomic = np.sum(probabilities * taxonomy.integer_distances, axis=1) / taxonomy.d_max
    neighborhood = np.asarray(
        [
            probabilities[row, list(taxonomy.neighborhoods[row])].sum()
            for row in range(10)
        ]
    )

    if score.family.startswith("tail_"):
        if oracle_confusion is None:
            membership = _tail_membership_probabilities(
                score, counts, taxonomy, seed=posterior_seed
            )
        else:
            source = {
                "tail_precision": precision,
                "tail_taxonomic": taxonomic,
                "tail_neighborhood": neighborhood,
            }[score.family]
            membership = _fractional_active_membership(
                source, k=score.k, tail=str(score.tail)
            )
        if score.family == "tail_taxonomic":
            ranges = np.ptp(taxonomy.integer_distances, axis=1) / taxonomy.d_max
        else:
            ranges = np.ones(10)
        return membership * ranges / score.k

    if score.family == "rms_error":
        values = 1.0 - precision
        aggregate = max(1e-9, float(np.sqrt(weights @ np.square(values))))
        return weights * values / aggregate
    if score.family == "rms_taxonomic":
        aggregate = max(1e-9, float(np.sqrt(weights @ np.square(taxonomic))))
        ranges = np.ptp(taxonomy.integer_distances, axis=1) / taxonomy.d_max
        return weights * taxonomic * ranges / aggregate
    if score.family == "harmonic_precision":
        stabilized = np.maximum(precision, 1e-3)
        harmonic = harmonic_macro_precision(stabilized)
        return np.square(harmonic) / (10.0 * np.square(stabilized))
    if score.family == "macro_precision":
        return np.full(10, 0.1)
    if score.family == "mean_taxonomic":
        ranges = np.ptp(taxonomy.integer_distances, axis=1) / taxonomy.d_max
        return weights * ranges
    if score.family == "overall_disagreement":
        return weights.copy()
    if score.family == "weighted_kappa":
        distance = taxonomy.integer_distances.astype(float)
        observed = float(np.sum(weights[:, None] * probabilities * distance))
        reference = np.sum(weights[:, None] * probabilities, axis=0)
        expected_by_destination = weights @ distance
        expected = float(expected_by_destination @ reference)
        rho = observed / max(expected, 1e-9)
        coefficients = distance - rho * expected_by_destination[None, :]
        return weights * np.ptp(coefficients, axis=1) / max(expected, 1e-9)
    raise ValueError(f"unknown score family: {score.family}")


def selection_probabilities(
    policy: MultimetricPolicy,
    score: MultimetricScore,
    row_sizes: np.ndarray,
    audits: np.ndarray,
    counts: np.ndarray,
    taxonomy: TaxonomyInputs,
    *,
    alpha: float,
    posterior_seed: int,
    true_confusion: np.ndarray | None = None,
    priority_override: np.ndarray | None = None,
) -> tuple[np.ndarray, str, np.ndarray, bool]:
    srs = _srs_distribution(row_sizes, audits)
    if policy.selection == "srs":
        return srs, "residual_global_srs", np.zeros(10), False
    caps = row_sizes.copy()
    if policy.max_row_fraction is not None:
        caps = np.maximum(1, np.floor(policy.max_row_fraction * row_sizes)).astype(int)
    available = np.flatnonzero((audits < row_sizes) & (audits < caps))
    if available.size == 0:
        raise StopIteration
    eligible_srs = np.zeros_like(srs)
    eligible_srs[available] = srs[available]
    eligible_srs /= eligible_srs.sum()
    uses_oracle = policy.selection == "oracle"
    if uses_oracle and true_confusion is None:
        raise ValueError("oracle policy requires the full confusion matrix")
    priorities = (
        np.asarray(priority_override, dtype=float)
        if priority_override is not None
        else score_row_priorities(
            score,
            row_sizes,
            counts,
            taxonomy,
            oracle_confusion=true_confusion if uses_oracle else None,
            posterior_seed=posterior_seed,
        )
    )
    gains = np.full(10, -np.inf)
    for row in available:
        current = _planning_radius(int(row_sizes[row]), int(audits[row]), alpha=alpha)
        future = _planning_radius(int(row_sizes[row]), int(audits[row] + 1), alpha=alpha)
        gains[row] = priorities[row] * max(0.0, current - future)
    maximum = float(np.max(gains[available]))
    ties = available[np.isclose(gains[available], maximum, rtol=1e-12, atol=1e-15)]
    pas = np.zeros(10, dtype=float)
    if not np.isfinite(maximum) or maximum <= 0.0:
        pas = eligible_srs
        reason = "metric_pas_srs_fallback"
    else:
        pas[ties] = 1.0 / ties.size
        reason = "oracle_gradient_pas" if uses_oracle else "predictable_metric_pas"
    distribution = policy.srs_floor * eligible_srs + (1.0 - policy.srs_floor) * pas
    distribution[audits >= row_sizes] = 0.0
    distribution /= distribution.sum()
    if policy.srs_floor:
        reason = f"{reason}_with_{int(100 * policy.srs_floor)}pct_srs_floor"
    return distribution, reason, priorities, uses_oracle


@dataclass(frozen=True)
class MultimetricReplay:
    checkpoints: pd.DataFrame
    allocations: pd.DataFrame
    actions: pd.DataFrame


def replay_multimetric_policy(
    population: AuditPopulation,
    score: MultimetricScore,
    policy: MultimetricPolicy,
    taxonomy: TaxonomyInputs,
    *,
    path_seed: int,
    checkpoints: Sequence[int],
    alpha: float = 0.05,
    record_actions: bool = False,
    posterior_update_every: int = 25,
) -> MultimetricReplay:
    checkpoints = tuple(sorted(set(int(value) for value in checkpoints)))
    if not checkpoints or checkpoints[0] <= 0 or checkpoints[-1] > population.n:
        raise ValueError("checkpoints must lie inside the population")
    orders = row_destination_orders(population, path_seed)
    row_sizes = population.row_sizes
    audits = np.zeros(10, dtype=int)
    counts = np.zeros((10, 10), dtype=int)
    rng = np.random.default_rng(
        _stable_seed(
            path_seed,
            policy.policy_id,
            "score_independent" if policy.selection == "srs" else score.score_id,
        )
    )
    checkpoint_records: list[dict[str, object]] = []
    allocation_records: list[dict[str, object]] = []
    action_records: list[dict[str, object]] = []
    checkpoint_set = set(checkpoints)
    cached_priorities = np.zeros(10)
    cached_priority_epoch = -1

    for audit_index in range(1, checkpoints[-1] + 1):
        pilot_rows = np.flatnonzero(audits < np.minimum(policy.pilot_per_row, row_sizes))
        if pilot_rows.size:
            row = int(pilot_rows[0])
            probabilities = np.zeros(10)
            probabilities[row] = 1.0
            reason = "fixed_five_per_row_pilot"
            uses_oracle = False
            cached_priorities = np.zeros(10)
        else:
            update_epoch = audit_index // posterior_update_every
            probabilities, reason, priorities, uses_oracle = selection_probabilities(
                policy,
                score,
                row_sizes,
                audits,
                counts,
                taxonomy,
                alpha=alpha,
                posterior_seed=_stable_seed(
                    path_seed, policy.policy_id, score.score_id, update_epoch
                ),
                true_confusion=(
                    population.true_confusion if policy.selection == "oracle" else None
                ),
                priority_override=(
                    cached_priorities
                    if cached_priority_epoch == update_epoch
                    else None
                ),
            )
            cached_priorities = priorities
            cached_priority_epoch = update_epoch
            row = int(rng.choice(10, p=probabilities))
        destination = int(orders[row][audits[row]])
        selected_probability = float(probabilities[row])
        counts[row, destination] += 1
        audits[row] += 1

        if record_actions:
            action_records.append(
                {
                    "seed": path_seed,
                    "score_id": score.score_id,
                    "policy_id": policy.policy_id,
                    "audit": audit_index,
                    "selected_row": row,
                    "selected_probability": selected_probability,
                    "selection_reason": reason,
                    "uses_oracle": uses_oracle,
                    "within_row_audit": int(audits[row]),
                    "observed_destination": destination,
                    **{
                        f"probability_row_{index}": float(value)
                        for index, value in enumerate(probabilities)
                    },
                }
            )
        if audit_index in checkpoint_set:
            checkpoint_records.append(
                {
                    "seed": path_seed,
                    "score_id": score.score_id,
                    "policy_id": policy.policy_id,
                    "audit": audit_index,
                    "implementable": policy.implementable,
                    "oracle_status": "oracle_only" if uses_oracle else "none",
                }
            )
            for row_index in range(10):
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
                        "planning_priority": float(cached_priorities[row_index]),
                    }
                )
    return MultimetricReplay(
        checkpoints=pd.DataFrame.from_records(checkpoint_records),
        allocations=pd.DataFrame.from_records(allocation_records),
        actions=pd.DataFrame.from_records(action_records),
    )


__all__ = [
    "MultimetricPolicy",
    "MultimetricReplay",
    "MultimetricScore",
    "TaxonomyInputs",
    "aggregate_interval",
    "canonical_multimetric_scores",
    "evaluate_score",
    "harmonic_macro_precision",
    "load_taxonomy_inputs",
    "multimetric_policy_catalog",
    "normalize_row_numerators",
    "replay_multimetric_policy",
    "row_functional",
    "row_integer_costs",
    "row_values",
    "score_row_priorities",
    "selection_probabilities",
    "weighted_kappa_components",
]
