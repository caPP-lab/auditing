"""Anytime-valid mislabel-rate audit for a fixed finite population.

Submitted labels define 46 sampling rows.  Both policies use the same
predictable control variate and the same two-sided betting confidence
sequence; they differ only in whether the next row is selected uniformly over
remaining records (SRS) or by the predicted Bernoulli standard deviation
(PAS).  The implementation keeps the audit population fixed and treats only
the audit randomization as random.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from auditcs.core_loader import (
    DEFAULT_REFERENCE_ROOT,
    _load_reference_modules,
)


POLICY_SRS = "srs"
POLICY_PAS = "pas"
POLICIES = (POLICY_SRS, POLICY_PAS)


@dataclass(frozen=True)
class MislabelAuditConfig:
    """Frozen settings for the submitted-row error-rate experiment."""

    alpha: float = 0.045
    submitted_label_probability: float = 0.8
    row_shrinkage: float = 2.0
    stratum_shrinkage: float = 1.0
    srs_floor: float = 0.3
    stake_effective_n: float = 800.0
    cap_fraction: float = 0.9
    grid_step: float = 0.0005

    def validate(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0,1)")
        if not 0.0 <= self.submitted_label_probability <= 1.0:
            raise ValueError("submitted-label probability must lie in [0,1]")
        if self.row_shrinkage <= 0.0 or self.stratum_shrinkage <= 0.0:
            raise ValueError("shrinkage strengths must be positive")
        if not 0.0 <= self.srs_floor <= 1.0:
            raise ValueError("SRS floor must lie in [0,1]")
        if self.stake_effective_n <= 0.0:
            raise ValueError("stake effective sample size must be positive")
        if not 0.0 < self.cap_fraction < 1.0:
            raise ValueError("cap fraction must lie in (0,1)")
        reciprocal = 1.0 / self.grid_step
        if self.grid_step <= 0.0 or not np.isclose(reciprocal, round(reciprocal)):
            raise ValueError("grid_step must divide [0,1] exactly")


def serfling_half_width(
    reviews: int | np.ndarray,
    population_size: int,
    *,
    alpha: float = 0.045,
) -> np.ndarray:
    """Two-sided fixed-n Serfling half-width for values in [0,1]."""

    n = np.asarray(reviews, dtype=float)
    if np.any(n <= 0) or np.any(n > population_size):
        raise ValueError("reviews must lie in [1, population_size]")
    finite_population = 1.0 - (n - 1.0) / float(population_size)
    return np.sqrt(finite_population * np.log(2.0 / alpha) / (2.0 * n))


def serfling_target_reviews(
    half_width: float,
    population_size: int,
    *,
    alpha: float = 0.045,
) -> int:
    """Smallest fixed sample size whose Serfling half-width meets a target."""

    if not 0.0 < half_width < 1.0:
        raise ValueError("half_width must lie in (0,1)")
    scale = np.log(2.0 / alpha) / (2.0 * half_width**2)
    answer = int(math.ceil(scale * (population_size + 1.0) / (population_size + scale)))
    while answer > 1 and serfling_half_width(
        answer - 1, population_size, alpha=alpha
    ) <= half_width:
        answer -= 1
    while serfling_half_width(answer, population_size, alpha=alpha) > half_width:
        answer += 1
    return answer


class MislabelRateAudit:
    """Sequential direct betting CS for the finite-population error rate."""

    def __init__(
        self,
        record_rows: Sequence[int],
        verified_labels: Sequence[int],
        row_to_submitted_label: Sequence[int],
        *,
        seed: int,
        policy: str,
        config: MislabelAuditConfig | None = None,
        reference_root: Path = DEFAULT_REFERENCE_ROOT,
        row_sampler: Callable | None = None,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        self.config = config or MislabelAuditConfig()
        self.config.validate()
        self.policy = policy
        self.record_rows = np.asarray(record_rows, dtype=int)
        self.verified_labels = np.asarray(verified_labels, dtype=int)
        self.row_to_submitted_label = np.asarray(row_to_submitted_label, dtype=int)
        if self.record_rows.ndim != 1 or self.verified_labels.shape != self.record_rows.shape:
            raise ValueError("record rows and verified labels must be aligned vectors")
        self.N = int(len(self.record_rows))
        self.R = int(self.row_to_submitted_label.size)
        if self.N <= 0 or self.R <= 0:
            raise ValueError("the population and row vocabulary must be nonempty")
        if self.record_rows.min() < 0 or self.record_rows.max() >= self.R:
            raise ValueError("record row lies outside the row vocabulary")
        self.N_r = np.bincount(self.record_rows, minlength=self.R).astype(int)
        if np.any(self.N_r <= 0):
            raise ValueError("rows must be consecutive and nonempty")

        if row_sampler is None:
            modules = _load_reference_modules(reference_root)
            self._sample_dyadic_row = modules[
                "directional_observations"
            ].sample_dyadic_row
        else:
            self._sample_dyadic_row = row_sampler
        order_rng = np.random.default_rng(seed)
        self._orders = [
            list(order_rng.permutation(np.flatnonzero(self.record_rows == row))[::-1])
            for row in range(self.R)
        ]
        self._row_rng = np.random.default_rng(seed + 7121)

        self.audits = np.zeros(self.R, dtype=int)
        self.errors = np.zeros(self.R, dtype=int)
        self.grid = np.linspace(
            0.0,
            1.0,
            int(round(1.0 / self.config.grid_step)) + 1,
        )
        self._log_wealth_plus = np.zeros_like(self.grid)
        self._log_wealth_minus = np.zeros_like(self.grid)
        self._rejected_plus = np.zeros_like(self.grid, dtype=bool)
        self._rejected_minus = np.zeros_like(self.grid, dtype=bool)
        self._variance_numerator = 0.25
        self._observation_mean = self.config.submitted_label_probability
        self._observations: list[float] = []
        self._pending: dict[str, object] | None = None

    @property
    def reviews(self) -> int:
        return int(self.audits.sum())

    def _predict_error(self) -> np.ndarray:
        # Appendix K.1: rows and strata coincide, but both shrinkage stages are
        # retained to reproduce the declared predictive control variate.
        prior_error = 1.0 - self.config.submitted_label_probability
        row_mean = (
            self.errors + self.config.row_shrinkage * prior_error
        ) / (self.audits + self.config.row_shrinkage)
        return (
            self.errors + self.config.stratum_shrinkage * row_mean
        ) / (self.audits + self.config.stratum_shrinkage)

    def _desired_probabilities(self, predicted_error: np.ndarray) -> np.ndarray:
        remaining = self.N_r - self.audits
        active = remaining > 0
        proportional = np.zeros(self.R, dtype=float)
        proportional[active] = remaining[active] / remaining[active].sum()
        if self.policy == POLICY_SRS:
            return proportional
        sigma = np.sqrt(np.maximum(predicted_error * (1.0 - predicted_error), 0.0))
        neyman_raw = remaining * sigma
        neyman = np.zeros(self.R, dtype=float)
        if neyman_raw[active].sum() > 0.0:
            neyman[active] = neyman_raw[active] / neyman_raw[active].sum()
        else:
            neyman = proportional.copy()
        desired = self.config.srs_floor * proportional + (1.0 - self.config.srs_floor) * neyman
        desired[~active] = 0.0
        desired /= desired.sum()
        return desired

    def prepare(self) -> dict[str, object] | None:
        """Freeze the allocation, observation range, and stakes before review."""

        if self._pending is not None:
            public = dict(self._pending["public"])
            public["row_probabilities"] = np.asarray(public["row_probabilities"]).copy()
            return public
        if self.reviews >= self.N:
            return None
        predicted_error = self._predict_error()
        remaining = self.N_r - self.audits
        desired = self._desired_probabilities(predicted_error)
        selected, probabilities, _ = self._sample_dyadic_row(
            self._row_rng,
            desired,
            remaining > 0,
        )
        selected = int(selected)
        if probabilities[selected] <= 0.0:
            raise RuntimeError("selected a row with zero represented probability")
        weights = np.zeros(self.R, dtype=float)
        active = remaining > 0
        weights[active] = remaining[active] / (self.N * probabilities[active])
        observed_errors = int(self.errors.sum())
        plugin = float(
            (observed_errors + np.dot(remaining, predicted_error)) / self.N
        )
        lower_range = float(plugin - np.max(weights[active] * predicted_error[active]))
        upper_range = float(
            plugin + np.max(weights[active] * (1.0 - predicted_error[active]))
        )
        review_number = self.reviews + 1
        variance_estimate = float(self._variance_numerator / review_number)
        base_stake = float(
            np.sqrt(
                2.0
                * np.log(2.0 / self.config.alpha)
                / (self.config.stake_effective_n * variance_estimate)
            )
        )
        lambda_plus = np.full_like(self.grid, base_stake)
        positive = self.grid > lower_range
        lambda_plus[positive] = np.minimum(
            lambda_plus[positive],
            self.config.cap_fraction / (self.grid[positive] - lower_range),
        )
        lambda_minus = np.full_like(self.grid, base_stake)
        negative = self.grid < upper_range
        lambda_minus[negative] = np.minimum(
            lambda_minus[negative],
            self.config.cap_fraction / (upper_range - self.grid[negative]),
        )
        record_id = int(self._orders[selected][-1])
        public = {
            "record_id": record_id,
            "row": selected,
            "row_probabilities": probabilities.copy(),
            "desired_probabilities": desired.copy(),
            "predicted_error": predicted_error.copy(),
            "plugin": plugin,
            "lower_range": lower_range,
            "upper_range": upper_range,
            "variance_estimate": variance_estimate,
            "base_stake": base_stake,
            "selected_weight": float(weights[selected]),
        }
        self._pending = {
            "public": public,
            "record_id": record_id,
            "row": selected,
            "predicted_error": predicted_error,
            "plugin": plugin,
            "weights": weights,
            "lambda_plus": lambda_plus,
            "lambda_minus": lambda_minus,
        }
        return dict(public)

    def observe(self, record_id: int) -> dict[str, object]:
        """Reveal the fixed record, update the betting CS, and return diagnostics."""

        pending = self._pending
        if pending is None or int(record_id) != int(pending["record_id"]):
            raise ValueError("record_id must match the pending review")
        row = int(pending["row"])
        submitted = int(self.row_to_submitted_label[row])
        error = int(self.verified_labels[int(record_id)] != submitted)
        predicted_error = np.asarray(pending["predicted_error"], dtype=float)
        weight = float(np.asarray(pending["weights"], dtype=float)[row])
        observation = float(
            pending["plugin"] + weight * (error - predicted_error[row])
        )
        centered = observation - self.grid
        plus_factor = 1.0 + np.asarray(pending["lambda_plus"]) * centered
        minus_factor = 1.0 - np.asarray(pending["lambda_minus"]) * centered
        if np.any(plus_factor < 0.1 - 1e-12) or np.any(minus_factor < 0.1 - 1e-12):
            raise RuntimeError("a range-capped wealth factor fell below 0.1")
        self._log_wealth_plus += np.log(plus_factor)
        self._log_wealth_minus += np.log(minus_factor)
        threshold = math.log(2.0 / self.config.alpha)
        self._rejected_plus |= self._log_wealth_plus >= threshold
        self._rejected_minus |= self._log_wealth_minus >= threshold

        old_mean = self._observation_mean
        self._variance_numerator += (observation - old_mean) ** 2
        self._observation_mean = old_mean + (observation - old_mean) / (self.reviews + 1)
        self._observations.append(observation)
        self.audits[row] += 1
        self.errors[row] += error
        popped = int(self._orders[row].pop())
        if popped != int(record_id):
            raise RuntimeError("within-row replay order changed")
        self._pending = None
        return {
            "review": self.reviews,
            "record_id": int(record_id),
            "row": row,
            "error": error,
            "observation": observation,
        }

    def interval(self) -> dict[str, float | bool]:
        """Return the running-intersection confidence sequence on the frozen grid."""

        alive = ~(self._rejected_plus | self._rejected_minus)
        if not np.any(alive):
            raise RuntimeError("the betting confidence sequence became empty")
        retained = self.grid[alive]
        lower_grid = float(retained[0])
        upper_grid = float(retained[-1])
        lower = max(0.0, lower_grid - self.config.grid_step)
        upper = min(1.0, upper_grid + self.config.grid_step)
        return {
            "lower": lower,
            "upper": upper,
            "half_width": 0.5 * (upper - lower),
            "center": 0.5 * (upper + lower),
            "grid_lower": lower_grid,
            "grid_upper": upper_grid,
            "alive_candidates": int(alive.sum()),
            "nested_by_running_intersection": True,
        }
