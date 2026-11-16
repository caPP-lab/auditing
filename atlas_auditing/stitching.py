from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import math
import numpy as np
from scipy.special import xlogy
from atlas_auditing.stitching_penalty import reference_log_stitching

@dataclass(frozen=True)
class MRConfig:
    alpha: float = 0.045
    epoch_base: int = 2
    outward_pad: float = 1e-12

    def validate(self):
        if not 0 < self.alpha < 1:
            raise ValueError('alpha must lie in (0,1)')
        if not isinstance(self.epoch_base, int) or self.epoch_base < 2:
            raise ValueError('epoch_base must be an integer >= 2')
        if not np.isfinite(self.outward_pad) or self.outward_pad < 0:
            raise ValueError('outward_pad must be finite and nonnegative')
        if self.epoch_base != 2:
            raise ValueError('reference construction uses dyadic stitching')

def shannon(probabilities):
    p = np.asarray(probabilities, dtype=float)
    return float(-xlogy(p, p).sum())

def finite_population_range_factor(N):
    if int(N) != N or N < 1:
        raise ValueError('N must be a positive integer')
    N = int(N)
    result = np.zeros(N + 1)
    n = np.arange(1, N, dtype=float)
    result[1:N] = (N - n) ** 2 * np.cumsum(1 / (N - n) ** 2)
    return result

def reference_stitched_boundary(bias, variance_proxy, alpha, *, log_multiplicity=0.0):
    bias, variance_proxy = (np.asarray(bias), np.asarray(variance_proxy))
    if bias.ndim != 1 or bias.shape != variance_proxy.shape or len(bias) < 2 or (not 0 < alpha < 1) or (not np.isfinite(log_multiplicity)) or (log_multiplicity < 0) or np.any(bias < 0) or np.any(variance_proxy < 0) or (not np.isfinite(bias).all()) or (not np.isfinite(variance_proxy).all()):
        raise ValueError('invalid fixed-time moment bounds')
    N = len(bias) - 1
    result = np.full(N + 1, np.inf)
    if N > 1:
        proof_horizon = 1 << (N - 1).bit_length()
        times = np.arange(1, proof_horizon + 1)
        anchors = (times + 1) // 2
        penalty = reference_log_stitching(times) + math.log(2 / alpha) + log_multiplicity
        raw = bias[anchors] + np.sqrt(2 * variance_proxy[anchors] * penalty)
        majorant = np.maximum.accumulate(raw[::-1])[::-1]
        result[1:N] = majorant[:N - 1]
    result[N] = 0.0
    result.setflags(write=False)
    return result

def entropy_one_change(n):
    return 0.0 if n == 1 else math.log(n) - (n - 1) * math.log(n - 1) / n

@lru_cache(maxsize=64)
def entropy_boundaries(N, categories, config):
    config.validate()
    rho = finite_population_range_factor(N)
    n = np.arange(1, N + 1, dtype=float)
    low_var, high_var, bias = (np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1))
    low_var[1:] = math.log(N) ** 2 * rho[1:] / (4 * n ** 2)
    sensitivity = np.array([entropy_one_change(int(t)) for t in n])
    high_var[1:] = sensitivity ** 2 * rho[1:] / 4
    if N > 1:
        bias[1:] = np.minimum(math.log(categories), np.log1p((categories - 1) * (N - n) / (n * (N - 1))))
    return (reference_stitched_boundary(np.zeros(N + 1), low_var, config.alpha), reference_stitched_boundary(bias, high_var, config.alpha))

def entropy_completion_bounds(counts, N):
    counts = np.asarray(counts, dtype=float)
    remaining = N - counts.sum()
    if remaining < 0 or np.any(counts < 0):
        raise ValueError('invalid observed counts')
    low_counts = counts.copy()
    low_counts[int(np.argmax(counts))] += remaining
    floors = np.sort(counts / N)[::-1]
    prefix = np.r_[0.0, np.cumsum(floors)]
    k = len(floors)
    for m in range(k):
        level = (1 - prefix[m]) / (k - m)
        if level >= floors[m] - 1e-14:
            high_p = np.maximum(counts / N, max(0.0, level))
            return (shannon(low_counts / N), shannon(high_p))
    raise ArithmeticError('entropy water-filling failed')

class MREntropyCS:

    def __init__(self, N, categories, *, config=None):
        if int(N) != N or N < 1 or int(categories) != categories or (categories < 1):
            raise ValueError('positive integer N and categories required')
        self.N, self.categories = (int(N), int(categories))
        self.config = config or MRConfig()
        self.config.validate()
        self.counts = np.zeros(self.categories, dtype=int)
        self.t = 0
        self.lower, self.upper = (0.0, math.log(self.categories))
        self.plugin = 0.0
        self.radii = entropy_boundaries(self.N, self.categories, self.config)

    def observe(self, label):
        if self.t == self.N or int(label) != label or (not 0 <= label < self.categories):
            raise ValueError('invalid observation or census already reached')
        self.counts[int(label)] += 1
        self.t += 1
        self.plugin = shannon(self.counts / self.t)
        logical_lower, logical_upper = entropy_completion_bounds(self.counts, self.N)
        if self.t == self.N:
            lo = hi = self.plugin
        else:
            lo = max(logical_lower, self.plugin - self.radii[0][self.t]) - self.config.outward_pad
            hi = min(logical_upper, self.plugin + self.radii[1][self.t]) + self.config.outward_pad
        self.lower = max(self.lower, lo)
        self.upper = min(self.upper, hi)
        return self.interval()

    def interval(self):
        empty = self.lower > self.upper
        return dict(lower=math.exp(self.lower), upper=math.exp(self.upper), entropy_lower=self.lower, entropy_upper=self.upper, plugin_entropy=self.plugin, empty=empty)

@lru_cache(maxsize=64)
def envelope_boundaries(N, branches, value_range, config):
    config.validate()
    rho = finite_population_range_factor(N)
    variance = np.zeros(N + 1)
    variance[1:] = value_range ** 2 * rho[1:] / (4 * np.arange(1, N + 1) ** 2)
    bias = np.zeros(N + 1)
    return (reference_stitched_boundary(bias, variance, config.alpha, log_multiplicity=math.log(branches)), reference_stitched_boundary(bias, variance, config.alpha))

class MRTailCS:

    def __init__(self, row_sizes, costs, *, k=3, largest=False, config=None):
        self.row_sizes = np.asarray(row_sizes, dtype=int).copy()
        self.costs = np.asarray(costs, dtype=float).copy()
        if self.row_sizes.ndim != 1 or np.any(self.row_sizes <= 0):
            raise ValueError('row sizes must be positive')
        R = len(self.row_sizes)
        if self.costs.ndim != 2 or self.costs.shape[0] != R or (not np.isfinite(self.costs).all()):
            raise ValueError('cost matrix must be finite and match rows')
        if np.any(self.costs < 0) or np.any(self.costs > 1) or (not 1 <= k <= R):
            raise ValueError('costs must be in [0,1] and 1 <= k <= rows')
        self.N, self.k, self.largest = (int(self.row_sizes.sum()), int(k), largest)
        self.config = config or MRConfig()
        self.config.validate()
        self.minimum, self.maximum = (self.costs.min(axis=1), self.costs.max(axis=1))
        self.mid = (self.minimum + self.maximum) / 2
        self.scale = self.N / self.row_sizes
        self.subsets = np.asarray(list(combinations(range(R), self.k)), dtype=int)
        self.value_range = float(np.max(self.scale * (self.maximum - self.minimum)) / self.k)
        self.radii = envelope_boundaries(self.N, len(self.subsets), self.value_range, self.config)
        self.audits = np.zeros(R, dtype=int)
        self.sums = np.zeros(R)
        self.t, self.lower, self.upper, self.plugin = (0, 0.0, 1.0, 0.0)

    def _tail(self, row_means):
        ordered = np.sort(row_means)
        return float(np.mean(ordered[-self.k:] if self.largest else ordered[:self.k]))

    def observe(self, row, label):
        if int(row) != row or int(label) != label or (not 0 <= row < len(self.row_sizes)) or (not 0 <= label < self.costs.shape[1]):
            raise ValueError('invalid row or label')
        row, label = (int(row), int(label))
        if self.audits[row] >= self.row_sizes[row]:
            raise ValueError('row already exhausted')
        self.audits[row] += 1
        self.sums[row] += self.costs[row, label]
        self.t += 1
        row_estimate = self.mid + self.scale * (self.sums - self.mid * self.audits) / self.t
        self.plugin = self._tail(row_estimate)
        remaining = self.row_sizes - self.audits
        logical_lower = self._tail((self.sums + remaining * self.minimum) / self.row_sizes)
        logical_upper = self._tail((self.sums + remaining * self.maximum) / self.row_sizes)
        over, under = (self.radii[0][self.t], self.radii[1][self.t])
        if self.t == self.N:
            lo = hi = self._tail(self.sums / self.row_sizes)
        else:
            lo = max(logical_lower, self.plugin - (over if self.largest else under)) - self.config.outward_pad
            hi = min(logical_upper, self.plugin + (under if self.largest else over)) + self.config.outward_pad
        self.lower, self.upper = (max(self.lower, lo), min(self.upper, hi))
        return self.interval()

    def interval(self):
        return dict(lower=self.lower, upper=self.upper, plugin=self.plugin, empty=self.lower > self.upper)
