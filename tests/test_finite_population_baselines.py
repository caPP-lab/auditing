from __future__ import annotations

import math

import numpy as np

from auditcs.finite_population_baselines import (
    _hypergeom_logpmf_errors,
    ppr_current_interval,
    ppr_running_intervals,
    wald_half_width,
    wald_target_reviews,
)


def test_ppr_interval_matches_brute_force() -> None:
    population_size = 35
    reviews = 11
    observed = 4
    alpha = 0.045
    threshold = math.log(alpha) - math.log(reviews + 1.0)
    retained = [
        total
        for total in range(population_size + 1)
        if _hypergeom_logpmf_errors(
            total,
            population_size=population_size,
            reviews=reviews,
            observed_errors=observed,
        )
        > threshold
    ]
    assert ppr_current_interval(
        population_size=population_size,
        reviews=reviews,
        observed_errors=observed,
        alpha=alpha,
    ) == (min(retained), max(retained))


def test_running_ppr_is_intersection_of_all_looks() -> None:
    errors = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 0])
    result = ppr_running_intervals(errors, 50)
    assert np.all(np.diff(result["lower_count"]) >= 0)
    assert np.all(np.diff(result["upper_count"]) <= 0)
    assert np.all(result["lower_count"] <= result["upper_count"])


def test_wald_target_is_first_integer_to_cross() -> None:
    mean = 0.18
    target = 0.02
    reviews = wald_target_reviews(target, mean)
    assert float(wald_half_width(reviews, mean)) <= target
    assert float(wald_half_width(reviews - 1, mean)) > target
