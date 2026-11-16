from functools import lru_cache
import math
import numpy as np
from scipy.special import gammaln, xlogy

@lru_cache(maxsize=32)
def _log_factorial(n):
    result = gammaln(np.arange(n + 1, dtype=float) + 1)
    result.flags.writeable = False
    return result

def ppr_count_intervals(population_size, reviews, observed_counts, alpha):
    N, n = (int(population_size), int(reviews))
    k = np.asarray(observed_counts, dtype=np.int64)
    if N <= 0 or not 0 <= n <= N or k.ndim != 1 or np.any(k < 0) or np.any(k > n) or (not 0 < alpha < 1):
        raise ValueError('invalid count-CS inputs')
    if n == 0:
        return (np.zeros_like(k), np.full_like(k, N))
    lf = _log_factorial(N)
    constant = -lf[k] - lf[n - k] - lf[N] + lf[n] + lf[N - n]
    threshold = math.log(alpha) - math.log(n + 1) - 1e-10

    def accepted(K):
        return lf[K] - lf[K - k] + lf[N - K] - lf[N - K - n + k] + constant > threshold
    lower, upper = (k.copy(), N - n + k)
    mode = np.clip(k * (N + 1) // n, lower, upper)
    if not np.all(accepted(mode)):
        raise RuntimeError('empty PPR acceptance component')
    left, right = (lower.copy(), mode.copy())
    keep_lower = accepted(lower)
    while np.any(right - left > 1):
        mid = (left + right) // 2
        keep = accepted(mid)
        right = np.where(keep, mid, right)
        left = np.where(keep, left, mid)
    lower = np.where(keep_lower, lower, right)
    left, right = (mode.copy(), upper.copy())
    keep_upper = accepted(upper)
    while np.any(right - left > 1):
        mid = (left + right + 1) // 2
        keep = accepted(mid)
        left = np.where(keep, mid, left)
        right = np.where(keep, right, mid)
    upper = np.where(keep_upper, upper, left)
    return (lower, upper)

def box_simplex_linear(lower, upper, coefficients, *, total=1.0, maximize=False):
    lo, hi, c = [np.asarray(v, dtype=float) for v in (lower, upper, coefficients)]
    if lo.shape != hi.shape or lo.shape != c.shape or lo.ndim != 1 or (not np.isfinite([lo, hi, c]).all()):
        raise ValueError('invalid arrays')
    tol = 1e-10 * max(1.0, abs(total))
    if np.any(lo < 0) or np.any(hi < lo) or lo.sum() > total + tol or (hi.sum() < total - tol):
        raise ValueError('empty box/simplex intersection')
    x = lo.copy()
    remaining = max(0.0, total - float(x.sum()))
    for j in np.argsort(-c if maximize else c, kind='stable'):
        added = min(remaining, float(hi[j] - lo[j]))
        x[j] += added
        remaining -= added
    if remaining > tol:
        raise ArithmeticError('mass allocation failed')
    return (float(c @ x), x)

def entropy_projection(lower, upper):
    lo, hi = (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))
    h = lambda p: -xlogy(p, p)
    width = hi - lo
    slopes = np.divide(h(hi) - h(lo), width, out=np.zeros_like(lo), where=width > 0)
    intercepts = h(lo) - slopes * lo
    chord_min, vertex = box_simplex_linear(lo, hi, slopes)
    raw_lower = float(chord_min + intercepts.sum())
    feasible_upper = float(h(vertex).sum())
    left, right = (0.0, 1.0)
    for _ in range(70):
        level = (left + right) / 2
        if np.clip(level, lo, hi).sum() < 1:
            left = level
        else:
            right = level
    p = np.clip((left + right) / 2, lo, hi)
    grad = np.zeros_like(p)
    positive = hi > 0
    if np.any(p[positive] <= 0):
        raise ArithmeticError('nonpositive entropy tangent reference')
    grad[positive] = -np.log(p[positive]) - 1
    linear_max, _ = box_simplex_linear(lo, hi, grad, maximize=True)
    raw_upper = float(h(p).sum() + linear_max - grad @ p)
    padding = 1e-11
    lower_bound = max(0.0, raw_lower - padding)
    upper_bound = min(math.log(len(lo)), raw_upper + padding)
    return dict(lower=lower_bound, upper=upper_bound, min_feasible_upper=feasible_upper + padding, min_gap_upper=max(0.0, feasible_upper + padding - lower_bound), max_gap_upper=max(0.0, upper_bound - float(h(p).sum()) + padding))

def tail_projection(lower, upper, *, size=3, largest=False):
    lo, hi = (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))
    if lo.shape != hi.shape or np.any(lo > hi) or (not 1 <= size <= len(lo)):
        raise ValueError('invalid row-score bounds')
    select = slice(-size, None) if largest else slice(None, size)
    return (float(np.sort(lo)[select].mean()), float(np.sort(hi)[select].mean()))
