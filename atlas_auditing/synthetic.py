from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
from atlas_auditing.reference import DEFAULT_REFERENCE_ROOT, _load_reference_modules
POLICY_SRS = 'srs'
POLICY_PAS = 'pas'
POLICIES = (POLICY_SRS, POLICY_PAS)

@dataclass(frozen=True)
class SyntheticRiskConfig:
    alpha: float = 0.045
    prior_strength: float = 2.0
    stake_effective_n: float = 800.0
    cap_fraction: float = 0.9
    grid_step: float = 0.0005

    def validate(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError('alpha must lie in (0,1)')
        if self.prior_strength <= 0.0:
            raise ValueError('prior_strength must be positive')
        if self.stake_effective_n <= 0.0:
            raise ValueError('stake_effective_n must be positive')
        if not 0.0 < self.cap_fraction < 1.0:
            raise ValueError('cap_fraction must lie in (0,1)')
        reciprocal = 1.0 / self.grid_step
        if self.grid_step <= 0.0 or not np.isclose(reciprocal, round(reciprocal)):
            raise ValueError('grid_step must divide [0,1] exactly')

@dataclass(frozen=True)
class SyntheticRiskScenario:
    rows: int
    active_rows: int
    row_size: int
    record_rows: np.ndarray
    verified_labels: np.ndarray
    contributions: np.ndarray
    row_support_values: tuple[tuple[float, ...], ...]
    influential_indices: tuple[int, ...]
    row_labels: tuple[str, ...]
    category_labels: tuple[str, ...]

    @property
    def population_size(self) -> int:
        return int(self.record_rows.size)

    @property
    def truth(self) -> float:
        return float(np.mean(self.contributions))

    @property
    def confusion(self) -> np.ndarray:
        matrix = np.zeros((self.rows, self.rows), dtype=int)
        np.add.at(matrix, (self.record_rows, self.verified_labels), 1)
        return matrix

def make_synthetic_risk_scenario(rows: int, *, active_rows: int=4, row_size: int=250, design_seed: int=2026092401) -> SyntheticRiskScenario:
    if rows < 4 or rows % 2:
        raise ValueError('rows must be even and at least four')
    if active_rows < 2 or active_rows >= rows or active_rows % 2:
        raise ValueError('active_rows must be even and lie in [2, rows)')
    if row_size < 20 or row_size % 2:
        raise ValueError('row_size must be even and at least 20')
    permutation = np.random.default_rng(design_seed + rows).permutation(rows)
    influential = tuple(sorted((int(value) for value in permutation[:active_rows])))
    influential_set = set(influential)
    record_rows: list[np.ndarray] = []
    destinations: list[np.ndarray] = []
    contributions: list[np.ndarray] = []
    supports: list[tuple[float, ...]] = []
    half = row_size // 2
    for row in range(rows):
        record_rows.append(np.full(row_size, row, dtype=int))
        if row in influential_set:
            destinations.append(np.concatenate((np.full(half, row, dtype=int), np.full(half, (row + rows // 2) % rows, dtype=int))))
            contributions.append(np.concatenate((np.zeros(half), np.ones(half))))
            supports.append((0.0, 1.0))
        else:
            destinations.append(np.concatenate((np.full(half, (row + 1) % rows, dtype=int), np.full(half, (row + 2) % rows, dtype=int))))
            contributions.append(np.full(row_size, 0.25, dtype=float))
            supports.append((0.25, 0.25))
    return SyntheticRiskScenario(rows=rows, active_rows=active_rows, row_size=row_size, record_rows=np.concatenate(record_rows), verified_labels=np.concatenate(destinations), contributions=np.concatenate(contributions), row_support_values=tuple(supports), influential_indices=influential, row_labels=tuple((f'Submitted {index + 1:02d}' for index in range(rows))), category_labels=tuple((f'Verified {index + 1:02d}' for index in range(rows))))

class SyntheticLinearRiskAudit:

    def __init__(self, scenario: SyntheticRiskScenario, *, seed: int, policy: str, config: SyntheticRiskConfig | None=None, reference_root: Path=DEFAULT_REFERENCE_ROOT) -> None:
        if policy not in POLICIES:
            raise ValueError(f'unknown policy {policy!r}')
        self.scenario = scenario
        self.config = config or SyntheticRiskConfig()
        self.config.validate()
        self.policy = policy
        self.N = scenario.population_size
        self.R = scenario.rows
        self.N_r = np.bincount(scenario.record_rows, minlength=self.R).astype(int)
        self._minimum = np.asarray([min(values) for values in scenario.row_support_values], dtype=float)
        self._maximum = np.asarray([max(values) for values in scenario.row_support_values], dtype=float)
        self._prior_mean = np.asarray([np.mean(values) for values in scenario.row_support_values], dtype=float)
        self._prior_second = np.asarray([np.mean(np.square(values)) for values in scenario.row_support_values], dtype=float)
        modules = _load_reference_modules(reference_root)
        self._sample_dyadic_row = modules['directional_observations'].sample_dyadic_row
        order_rng = np.random.default_rng(seed)
        self._orders = [list(order_rng.permutation(np.flatnonzero(scenario.record_rows == row))[::-1]) for row in range(self.R)]
        self._row_rng = np.random.default_rng(seed + 7121)
        self.audits = np.zeros(self.R, dtype=int)
        self.observed_sum = np.zeros(self.R, dtype=float)
        self.observed_second = np.zeros(self.R, dtype=float)
        self.grid = np.linspace(0.0, 1.0, int(round(1.0 / self.config.grid_step)) + 1)
        self._log_wealth_plus = np.zeros_like(self.grid)
        self._log_wealth_minus = np.zeros_like(self.grid)
        self._rejected_plus = np.zeros_like(self.grid, dtype=bool)
        self._rejected_minus = np.zeros_like(self.grid, dtype=bool)
        self._variance_numerator = 0.25
        self._observation_mean = float(np.dot(self.N_r, self._prior_mean) / self.N)
        self._pending: dict[str, object] | None = None

    @property
    def reviews(self) -> int:
        return int(self.audits.sum())

    def _predict(self) -> tuple[np.ndarray, np.ndarray]:
        strength = self.config.prior_strength
        denominator = self.audits + strength
        mean = (self.observed_sum + strength * self._prior_mean) / denominator
        second = (self.observed_second + strength * self._prior_second) / denominator
        variance = np.maximum(second - np.square(mean), 0.0)
        return (mean, variance)

    def _desired_probabilities(self, variance: np.ndarray) -> np.ndarray:
        remaining = self.N_r - self.audits
        physical = remaining > 0
        proportional = np.zeros(self.R, dtype=float)
        proportional[physical] = remaining[physical] / remaining[physical].sum()
        if self.policy == POLICY_SRS:
            return proportional
        score_uncertain = physical & (self._maximum > self._minimum)
        raw = remaining * np.sqrt(variance)
        raw[~score_uncertain] = 0.0
        if raw.sum() <= 0.0:
            return np.zeros(self.R, dtype=float)
        return raw / raw.sum()

    def prepare(self) -> dict[str, object] | None:
        if self._pending is not None:
            public = dict(self._pending['public'])
            public['row_probabilities'] = np.asarray(public['row_probabilities']).copy()
            return public
        if self.reviews >= self.N:
            return None
        predicted_mean, predicted_variance = self._predict()
        remaining = self.N_r - self.audits
        desired = self._desired_probabilities(predicted_variance)
        eligible = (remaining > 0) & (desired > 0.0)
        if not np.any(eligible):
            return None
        selected, probabilities, _ = self._sample_dyadic_row(self._row_rng, desired, eligible)
        selected = int(selected)
        weights = np.zeros(self.R, dtype=float)
        weights[eligible] = remaining[eligible] / (self.N * probabilities[eligible])
        plugin = float((self.observed_sum.sum() + np.dot(remaining, predicted_mean)) / self.N)
        lower_range = float(plugin + np.min(weights[eligible] * (self._minimum[eligible] - predicted_mean[eligible])))
        upper_range = float(plugin + np.max(weights[eligible] * (self._maximum[eligible] - predicted_mean[eligible])))
        review_number = self.reviews + 1
        variance_estimate = float(self._variance_numerator / review_number)
        base_stake = float(np.sqrt(2.0 * np.log(2.0 / self.config.alpha) / (self.config.stake_effective_n * variance_estimate)))
        lambda_plus = np.full_like(self.grid, base_stake)
        positive = self.grid > lower_range
        lambda_plus[positive] = np.minimum(lambda_plus[positive], self.config.cap_fraction / (self.grid[positive] - lower_range))
        lambda_minus = np.full_like(self.grid, base_stake)
        negative = self.grid < upper_range
        lambda_minus[negative] = np.minimum(lambda_minus[negative], self.config.cap_fraction / (upper_range - self.grid[negative]))
        record_id = int(self._orders[selected][-1])
        public = {'record_id': record_id, 'row': selected, 'row_probabilities': probabilities.copy(), 'desired_probabilities': desired.copy(), 'predicted_mean': predicted_mean.copy(), 'predicted_variance': predicted_variance.copy(), 'plugin': plugin, 'lower_range': lower_range, 'upper_range': upper_range, 'variance_estimate': variance_estimate, 'base_stake': base_stake, 'selected_weight': float(weights[selected])}
        self._pending = {'public': public, 'record_id': record_id, 'row': selected, 'predicted_mean': predicted_mean, 'plugin': plugin, 'weights': weights, 'lambda_plus': lambda_plus, 'lambda_minus': lambda_minus}
        return dict(public)

    def observe(self, record_id: int) -> dict[str, float | int]:
        pending = self._pending
        if pending is None or int(record_id) != int(pending['record_id']):
            raise ValueError('record_id must match the pending review')
        row = int(pending['row'])
        value = float(self.scenario.contributions[int(record_id)])
        predicted = np.asarray(pending['predicted_mean'], dtype=float)
        weight = float(np.asarray(pending['weights'], dtype=float)[row])
        observation = float(pending['plugin'] + weight * (value - predicted[row]))
        centered = observation - self.grid
        plus_factor = 1.0 + np.asarray(pending['lambda_plus']) * centered
        minus_factor = 1.0 - np.asarray(pending['lambda_minus']) * centered
        if np.any(plus_factor < 0.1 - 1e-12) or np.any(minus_factor < 0.1 - 1e-12):
            raise RuntimeError('a range-capped wealth factor fell below 0.1')
        self._log_wealth_plus += np.log(plus_factor)
        self._log_wealth_minus += np.log(minus_factor)
        threshold = math.log(2.0 / self.config.alpha)
        self._rejected_plus |= self._log_wealth_plus >= threshold
        self._rejected_minus |= self._log_wealth_minus >= threshold
        old_mean = self._observation_mean
        self._variance_numerator += (observation - old_mean) ** 2
        self._observation_mean = old_mean + (observation - old_mean) / (self.reviews + 1)
        self.audits[row] += 1
        self.observed_sum[row] += value
        self.observed_second[row] += value * value
        popped = int(self._orders[row].pop())
        if popped != int(record_id):
            raise RuntimeError('within-row replay order changed')
        self._pending = None
        return {'review': self.reviews, 'record_id': int(record_id), 'row': row, 'value': value, 'observation': observation}

    def interval(self) -> dict[str, float | int | bool]:
        alive = ~(self._rejected_plus | self._rejected_minus)
        if not np.any(alive):
            raise RuntimeError('the betting confidence sequence became empty')
        retained = self.grid[alive]
        lower_grid = float(retained[0])
        upper_grid = float(retained[-1])
        lower = max(0.0, lower_grid - self.config.grid_step)
        upper = min(1.0, upper_grid + self.config.grid_step)
        return {'lower': lower, 'upper': upper, 'width': upper - lower, 'center': 0.5 * (lower + upper), 'alive_candidates': int(alive.sum()), 'nested_by_running_intersection': True}

