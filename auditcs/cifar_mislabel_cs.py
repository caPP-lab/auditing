"""CIFAR-10N linear mislabel-rate confidence sequences.

The target is the population-wide disagreement rate between ``random_label1``
and ``clean_label``.  The audit rows cross the submitted class with a frozen
four-level auxiliary-vote pattern formed from ``random_label2`` and
``random_label3``.  PAS uses development-calibrated error predictions and a
remaining-size-aware Neyman allocation; SRS samples uniformly from all
remaining records.  Both policies share the same direct finite-population
betting confidence sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from auditcs.mislabel_cs import (
    MislabelAuditConfig,
    MislabelRateAudit,
)


AGREEMENT_GROUPS = (
    "both_support",
    "one_support",
    "both_oppose",
    "all_different",
)
GROUP_BOTH_SUPPORT = 0
GROUP_ONE_SUPPORT = 1
GROUP_BOTH_OPPOSE = 2
GROUP_ALL_DIFFERENT = 3


@dataclass(frozen=True)
class CifarMislabelConfig:
    """Frozen policy and inference settings for CIFAR confirmation."""

    srs_floor: float = 0.10
    prior_strength: float = 20.0
    stake_effective_n: float = 800.0
    cap_fraction: float = 0.9
    grid_step: float = 0.0005
    alpha: float = 0.045

    def validate(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0,1)")
        if not 0.0 <= self.srs_floor <= 1.0:
            raise ValueError("srs_floor must lie in [0,1]")
        if self.prior_strength <= 0.0 or self.stake_effective_n <= 0.0:
            raise ValueError("strength parameters must be positive")
        if not 0.0 < self.cap_fraction < 1.0:
            raise ValueError("cap_fraction must lie in (0,1)")
        reciprocal = 1.0 / self.grid_step
        if self.grid_step <= 0.0 or not np.isclose(reciprocal, round(reciprocal)):
            raise ValueError("grid_step must divide [0,1] exactly")

    def base(self) -> MislabelAuditConfig:
        self.validate()
        return MislabelAuditConfig(
            alpha=self.alpha,
            submitted_label_probability=0.8,
            row_shrinkage=1.0,
            stratum_shrinkage=1.0,
            srs_floor=self.srs_floor,
            stake_effective_n=self.stake_effective_n,
            cap_fraction=self.cap_fraction,
            grid_step=self.grid_step,
        )


def agreement_groups(
    submitted: Sequence[int],
    auxiliary_one: Sequence[int],
    auxiliary_two: Sequence[int],
) -> np.ndarray:
    """Return the prespecified four-level auxiliary-vote pattern."""

    submitted = np.asarray(submitted, dtype=int)
    one = np.asarray(auxiliary_one, dtype=int)
    two = np.asarray(auxiliary_two, dtype=int)
    if submitted.ndim != 1 or one.shape != submitted.shape or two.shape != submitted.shape:
        raise ValueError("submitted and auxiliary labels must be aligned vectors")
    groups = np.full(submitted.shape, GROUP_ALL_DIFFERENT, dtype=int)
    both_support = (one == submitted) & (two == submitted)
    both_oppose = (one == two) & (one != submitted)
    one_support = ((one == submitted) | (two == submitted)) & ~both_support
    groups[both_support] = GROUP_BOTH_SUPPORT
    groups[one_support] = GROUP_ONE_SUPPORT
    groups[both_oppose] = GROUP_BOTH_OPPOSE
    return groups


def stratify_votes(
    submitted: Sequence[int],
    auxiliary_one: Sequence[int],
    auxiliary_two: Sequence[int],
    *,
    classes: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return record strata, row-to-class mapping, and agreement groups."""

    submitted = np.asarray(submitted, dtype=int)
    if submitted.min() < 0 or submitted.max() >= classes:
        raise ValueError("submitted labels lie outside the declared vocabulary")
    groups = agreement_groups(submitted, auxiliary_one, auxiliary_two)
    strata = submitted * len(AGREEMENT_GROUPS) + groups
    row_to_submitted = np.repeat(np.arange(classes, dtype=int), len(AGREEMENT_GROUPS))
    return strata, row_to_submitted, groups


def calibrate_stratum_priors(
    submitted: Sequence[int],
    verified: Sequence[int],
    auxiliary_one: Sequence[int],
    auxiliary_two: Sequence[int],
    *,
    classes: int = 10,
    smoothing: float = 12.0,
) -> np.ndarray:
    """Calibrate class-by-vote-pattern error priors on a development view."""

    submitted = np.asarray(submitted, dtype=int)
    verified = np.asarray(verified, dtype=int)
    if verified.shape != submitted.shape:
        raise ValueError("submitted and verified labels must align")
    strata, _, groups = stratify_votes(
        submitted, auxiliary_one, auxiliary_two, classes=classes
    )
    errors = (submitted != verified).astype(float)
    group_counts = np.bincount(groups, minlength=len(AGREEMENT_GROUPS)).astype(float)
    group_errors = np.bincount(
        groups, weights=errors, minlength=len(AGREEMENT_GROUPS)
    ).astype(float)
    group_means = (group_errors + 0.5) / (group_counts + 1.0)
    rows = classes * len(AGREEMENT_GROUPS)
    counts = np.bincount(strata, minlength=rows).astype(float)
    totals = np.bincount(strata, weights=errors, minlength=rows).astype(float)
    priors = np.empty(rows, dtype=float)
    for row in range(rows):
        group = row % len(AGREEMENT_GROUPS)
        priors[row] = (
            totals[row] + smoothing * group_means[group]
        ) / (counts[row] + smoothing)
    return np.clip(priors, 1e-4, 1.0 - 1e-4)


def _categorical_row_sampler(
    rng: np.random.Generator,
    desired: np.ndarray,
    active: np.ndarray,
) -> tuple[int, np.ndarray, None]:
    """Sample from the frozen predictable row distribution."""

    probabilities = np.asarray(desired, dtype=float).copy()
    probabilities[~np.asarray(active, dtype=bool)] = 0.0
    if probabilities.sum() <= 0.0:
        probabilities = np.asarray(active, dtype=float)
    probabilities /= probabilities.sum()
    selected = int(rng.choice(probabilities.size, p=probabilities))
    return selected, probabilities, None


class CifarMislabelRateAudit(MislabelRateAudit):
    """Vote-stratified CIFAR audit with a calibrated predictable prior."""

    def __init__(
        self,
        submitted: Sequence[int],
        verified: Sequence[int],
        auxiliary_one: Sequence[int],
        auxiliary_two: Sequence[int],
        *,
        prior_error: Sequence[float],
        seed: int,
        policy: str,
        config: CifarMislabelConfig | None = None,
    ) -> None:
        self.cifar_config = config or CifarMislabelConfig()
        self.cifar_config.validate()
        strata, row_to_submitted, groups = stratify_votes(
            submitted, auxiliary_one, auxiliary_two
        )
        self.agreement_group = groups
        self.prior_error = np.asarray(prior_error, dtype=float)
        if self.prior_error.shape != row_to_submitted.shape:
            raise ValueError("prior_error must contain one value per audit stratum")
        if np.any(self.prior_error <= 0.0) or np.any(self.prior_error >= 1.0):
            raise ValueError("prior errors must lie strictly between zero and one")
        super().__init__(
            strata,
            verified,
            row_to_submitted,
            seed=seed,
            policy=policy,
            config=self.cifar_config.base(),
            row_sampler=_categorical_row_sampler,
        )

    def _predict_error(self) -> np.ndarray:
        strength = self.cifar_config.prior_strength
        return (self.errors + strength * self.prior_error) / (self.audits + strength)
