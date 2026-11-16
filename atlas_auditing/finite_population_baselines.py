from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from scipy.special import gammaln

def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

def _hypergeom_logpmf_errors(total_errors: int, *, population_size: int, reviews: int, observed_errors: int) -> float:
    return _log_choose(total_errors, observed_errors) + _log_choose(population_size - total_errors, reviews - observed_errors) - _log_choose(population_size, reviews)

def _hypergeom_mode_in_total(*, population_size: int, reviews: int, observed_errors: int) -> int:
    lower = observed_errors
    upper = population_size - reviews + observed_errors
    if observed_errors == 0:
        return lower
    if observed_errors == reviews:
        return upper
    left, right = (lower, upper)
    while left < right:
        current = (left + right) // 2
        if current >= upper:
            right = current
            continue
        log_ratio = math.log(current + 1) - math.log(current + 1 - observed_errors) + math.log(population_size - current - reviews + observed_errors) - math.log(population_size - current)
        if log_ratio >= 0.0:
            left = current + 1
        else:
            right = current
    candidates = range(max(lower, left - 2), min(upper, left + 2) + 1)
    return max(candidates, key=lambda value: _hypergeom_logpmf_errors(value, population_size=population_size, reviews=reviews, observed_errors=observed_errors))

def ppr_current_interval(*, population_size: int, reviews: int, observed_errors: int, alpha: float=0.045) -> tuple[int, int] | None:
    if population_size <= 0:
        raise ValueError('population_size must be positive')
    if not 1 <= reviews <= population_size:
        raise ValueError('reviews must lie in [1,N]')
    if not 0 <= observed_errors <= reviews:
        raise ValueError('observed_errors must lie in [0,reviews]')
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha must lie in (0,1)')
    support_lower = observed_errors
    support_upper = population_size - reviews + observed_errors
    threshold = math.log(alpha) - math.log(reviews + 1.0)

    def retained(total: int) -> bool:
        return _hypergeom_logpmf_errors(total, population_size=population_size, reviews=reviews, observed_errors=observed_errors) > threshold
    mode = _hypergeom_mode_in_total(population_size=population_size, reviews=reviews, observed_errors=observed_errors)
    if not retained(mode):
        return None
    if retained(support_lower):
        lower = support_lower
    else:
        lo, hi = (support_lower, mode)
        while lo + 1 < hi:
            middle = (lo + hi) // 2
            if retained(middle):
                hi = middle
            else:
                lo = middle
        lower = hi
    if retained(support_upper):
        upper = support_upper
    else:
        lo, hi = (mode, support_upper)
        while lo + 1 < hi:
            middle = (lo + hi) // 2
            if retained(middle):
                lo = middle
            else:
                hi = middle
        upper = lo
    return (int(lower), int(upper))

def ppr_running_intervals(errors: Sequence[int], population_size: int, *, alpha: float=0.045) -> dict[str, np.ndarray]:
    observations = np.asarray(errors, dtype=int)
    if observations.ndim != 1 or observations.size > population_size:
        raise ValueError('errors must be a vector no longer than the population')
    if np.any((observations != 0) & (observations != 1)):
        raise ValueError('errors must be binary')
    lower = np.empty(observations.size, dtype=int)
    upper = np.empty(observations.size, dtype=int)
    running_lower, running_upper = (0, population_size)
    observed = 0
    empty_at = -1
    for index, value in enumerate(observations, start=1):
        observed += int(value)
        current = ppr_current_interval(population_size=population_size, reviews=index, observed_errors=observed, alpha=alpha)
        if current is None:
            running_lower, running_upper = (1, 0)
        else:
            running_lower = max(running_lower, current[0])
            running_upper = min(running_upper, current[1])
        if running_lower > running_upper and empty_at < 0:
            empty_at = index
        lower[index - 1] = running_lower
        upper[index - 1] = running_upper
    widths = (upper - lower) / (2.0 * population_size)
    widths[upper < lower] = np.nan
    return {'lower_count': lower, 'upper_count': upper, 'lower': lower / float(population_size), 'upper': upper / float(population_size), 'half_width': widths, 'empty_at': np.asarray([empty_at], dtype=int)}
