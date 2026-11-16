from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Literal, Sequence
import numpy as np
POLICY_SRS = 'srs'
POLICY_PAS = 'pas'
POLICIES = (POLICY_SRS, POLICY_PAS)
SCORE_WORST3_PRECISION = 'worst3_precision'
SCORE_WORST3_TAXONOMIC_RISK = 'worst3_taxonomic_risk'
SCORES = (SCORE_WORST3_PRECISION, SCORE_WORST3_TAXONOMIC_RISK)

@dataclass(frozen=True)
class TailBettingConfig:
    alpha: float = 0.045
    submitted_label_probability: float = 0.8
    row_shrinkage: float = 2.0
    stratum_shrinkage: float = 1.0
    stake_effective_n: float = 800.0
    cap_fraction: float = 0.9
    grid_step: float = 0.001
    tail_size: int = 3
    proportional_weight: float = 0.7
    range_weight: float = 0.15
    neyman_weight: float = 0.15
    active_endpoint_weight: float = 0.5
    profile_temperature: float = 1.0

    def validate(self, rows: int) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError('alpha must lie in (0,1)')
        if not 0.0 <= self.submitted_label_probability <= 1.0:
            raise ValueError('submitted-label probability must lie in [0,1]')
        if self.row_shrinkage <= 0.0 or self.stratum_shrinkage <= 0.0:
            raise ValueError('shrinkage strengths must be positive')
        if self.stake_effective_n <= 0.0:
            raise ValueError('stake effective sample size must be positive')
        if not 0.0 < self.cap_fraction < 1.0:
            raise ValueError('cap_fraction must lie in (0,1)')
        if not 0 < self.tail_size < rows:
            raise ValueError('tail_size must lie strictly between zero and rows')
        mixture = np.asarray([self.proportional_weight, self.range_weight, self.neyman_weight])
        if np.any(mixture < 0.0) or not np.isclose(mixture.sum(), 1.0):
            raise ValueError('allocation mixture weights must be nonnegative and sum to one')
        if not 0.0 <= self.active_endpoint_weight <= 1.0:
            raise ValueError('active_endpoint_weight must lie in [0,1]')
        if self.profile_temperature <= 0.0:
            raise ValueError('profile_temperature must be positive')
        reciprocal = 1.0 / self.grid_step
        if self.grid_step <= 0.0 or not np.isclose(reciprocal, round(reciprocal)):
            raise ValueError('grid_step must divide [0,1] exactly')

def _fractional_tail_membership(values: np.ndarray, *, size: int, tail: Literal['bottom', 'top']) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=float))
    cutoff = ordered[size - 1] if tail == 'bottom' else ordered[-size]
    strict = values < cutoff if tail == 'bottom' else values > cutoff
    tied = np.isclose(values, cutoff, atol=1e-15, rtol=0.0)
    result = strict.astype(float)
    result[tied] = (size - int(strict.sum())) / max(1, int(tied.sum()))
    return result

class CifarWorstThreeAudit:

    def __init__(self, submitted: Sequence[int], verified: Sequence[int], *, score_id: str, policy: str, seed: int, taxonomy_distances: np.ndarray | None=None, taxonomy_d_max: int | None=None, config: TailBettingConfig | None=None) -> None:
        if score_id not in SCORES:
            raise ValueError(f'unknown score {score_id!r}')
        if policy not in POLICIES:
            raise ValueError(f'unknown policy {policy!r}')
        self.score_id = score_id
        self.policy = policy
        self.submitted = np.asarray(submitted, dtype=int)
        self.verified = np.asarray(verified, dtype=int)
        if self.submitted.ndim != 1 or self.verified.shape != self.submitted.shape:
            raise ValueError('submitted and verified labels must be aligned vectors')
        self.N = int(self.submitted.size)
        self.R = int(max(self.submitted.max(), self.verified.max()) + 1)
        if self.R != 10:
            raise ValueError('the canonical study requires ten CIFAR classes')
        self.config = config or TailBettingConfig()
        self.config.validate(self.R)
        self.N_r = np.bincount(self.submitted, minlength=self.R).astype(int)
        if np.any(self.N_r <= 0):
            raise ValueError('every submitted CIFAR row must be nonempty')
        if score_id == SCORE_WORST3_PRECISION:
            self.row_costs = np.eye(self.R, dtype=float)
            self.tail: Literal['bottom', 'top'] = 'bottom'
            self.envelope: Literal['min', 'max'] = 'min'
        else:
            if taxonomy_distances is None or taxonomy_d_max is None:
                raise ValueError('taxonomic risk requires the frozen distance matrix')
            distance = np.asarray(taxonomy_distances, dtype=float)
            if distance.shape != (self.R, self.R) or taxonomy_d_max <= 0:
                raise ValueError('invalid CIFAR taxonomy inputs')
            self.row_costs = distance / float(taxonomy_d_max)
            self.tail = 'top'
            self.envelope = 'max'
        if np.any(self.row_costs < 0.0) or np.any(self.row_costs > 1.0):
            raise ValueError('row contributions must lie in [0,1]')
        self.row_ranges = np.ptp(self.row_costs, axis=1)
        subset_rows = tuple(combinations(range(self.R), self.config.tail_size))
        self.subsets = np.asarray(subset_rows, dtype=int)
        self.membership = np.zeros((len(subset_rows), self.R), dtype=float)
        for index, rows in enumerate(subset_rows):
            self.membership[index, list(rows)] = 1.0
        self.subset_lookup = {rows: index for index, rows in enumerate(subset_rows)}
        self.J = int(len(subset_rows))
        order_rng = np.random.default_rng(seed)
        self._orders = [list(order_rng.permutation(np.flatnonzero(self.submitted == row))[::-1]) for row in range(self.R)]
        self._row_rng = np.random.default_rng(seed + 7121)
        self.A = np.zeros((self.R, self.R), dtype=np.int64)
        self.grid = np.linspace(0.0, 1.0, int(round(1.0 / self.config.grid_step)) + 1)
        self._profile_log_wealth = np.zeros((self.J, self.grid.size), dtype=float)
        self._profile_crossed = np.zeros((self.J, self.grid.size), dtype=bool)
        self._adaptive_log_wealth = np.zeros(self.grid.size, dtype=float)
        self._adaptive_crossed = np.zeros(self.grid.size, dtype=bool)
        self._reported_alive = np.ones(self.grid.size, dtype=bool)
        self._empty_grid_events = 0
        initial_q = self._predict()
        initial_rows = np.sum(initial_q * self.row_costs, axis=1)
        initial_subsets = self.membership @ initial_rows / self.config.tail_size
        self._profile_variance_numerator = np.full(self.J, 0.25, dtype=float)
        self._profile_observation_mean = initial_subsets.copy()
        initial_active = self._active_subset_index(initial_rows)
        self._adaptive_variance_numerator = 0.25
        self._adaptive_observation_mean = float(initial_subsets[initial_active])
        self._pending: dict[str, object] | None = None

    @property
    def reviews(self) -> int:
        return int(self.A.sum())

    @property
    def audits_by_row(self) -> np.ndarray:
        return self.A.sum(axis=1).astype(int)

    def _predict(self) -> np.ndarray:
        total = self.A.sum(axis=0).astype(float)
        pooled = (total + 1.0 / self.R) / (float(total.sum()) + 1.0)
        prior = np.tile((1.0 - self.config.submitted_label_probability) * pooled, (self.R, 1))
        prior[np.arange(self.R), np.arange(self.R)] += self.config.submitted_label_probability
        audited = self.A.sum(axis=1).astype(float)
        row_mean = (self.A + self.config.row_shrinkage * prior) / (audited[:, None] + self.config.row_shrinkage)
        return (self.A + self.config.stratum_shrinkage * row_mean) / (audited[:, None] + self.config.stratum_shrinkage)

    def _active_subset_index(self, row_values: np.ndarray) -> int:
        if self.tail == 'bottom':
            selected = tuple(sorted(np.argsort(row_values, kind='stable')[:self.config.tail_size]))
        else:
            selected = tuple(sorted(np.argsort(-row_values, kind='stable')[:self.config.tail_size]))
        return int(self.subset_lookup[selected])

    def _selection_probabilities(self, q: np.ndarray, row_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        audits = self.audits_by_row
        remaining = self.N_r - audits
        active = remaining > 0
        srs = np.zeros(self.R, dtype=float)
        srs[active] = remaining[active] / remaining[active].sum()
        membership = _fractional_tail_membership(row_values, size=self.config.tail_size, tail=self.tail)
        centered = self.row_costs - row_values[:, None]
        sigma = np.sqrt(np.maximum(np.sum(q * centered ** 2, axis=1), 0.0))
        if self.policy == POLICY_SRS:
            return (srs, membership, sigma, {'proportional': srs.copy(), 'range': srs.copy(), 'neyman': srs.copy(), 'profile_pressure': np.full(self.R, self.config.tail_size / self.R), 'relevance': membership.copy()})
        profile_pressure = self._profile_bottleneck_pressure()
        active_weight = self.config.active_endpoint_weight
        relevance = active_weight * membership + (1.0 - active_weight) * profile_pressure
        range_raw = remaining / self.N_r * relevance * self.row_ranges
        neyman_raw = remaining / self.N_r * relevance * sigma
        range_component = np.zeros(self.R, dtype=float)
        neyman_component = np.zeros(self.R, dtype=float)
        if range_raw[active].sum() > 0.0:
            range_component[active] = range_raw[active] / range_raw[active].sum()
        else:
            range_component = srs.copy()
        if neyman_raw[active].sum() > 0.0:
            neyman_component[active] = neyman_raw[active] / neyman_raw[active].sum()
        else:
            neyman_component = srs.copy()
        probabilities = self.config.proportional_weight * srs + self.config.range_weight * range_component + self.config.neyman_weight * neyman_component
        components = {'proportional': srs.copy(), 'range': range_component, 'neyman': neyman_component, 'profile_pressure': profile_pressure, 'relevance': relevance}
        probabilities[~active] = 0.0
        probabilities /= probabilities.sum()
        return (probabilities, membership, sigma, components)

    def _profile_bottleneck_pressure(self) -> np.ndarray:
        if self.reviews == 0:
            weights = np.full(self.J, 1.0 / self.J)
        else:
            if self.envelope == 'min':
                rejected_profile = np.all(self._profile_crossed, axis=0)
                rejected_other = self._adaptive_crossed
                alive = ~(rejected_profile | rejected_other)
                if not np.any(alive):
                    alive = self._reported_alive
                candidate = int(np.flatnonzero(alive)[0])
            else:
                rejected_other = self._adaptive_crossed
                rejected_profile = np.all(self._profile_crossed, axis=0)
                alive = ~(rejected_other | rejected_profile)
                if not np.any(alive):
                    alive = self._reported_alive
                candidate = int(np.flatnonzero(alive)[-1])
            blockers = ~self._profile_crossed[:, candidate]
            if not np.any(blockers):
                blockers = np.ones(self.J, dtype=bool)
            evidence = self._profile_log_wealth[:, candidate]
            selected = evidence[blockers]
            logits = -(selected - float(selected.min())) / self.config.profile_temperature
            logits -= float(logits.max())
            local = np.exp(logits)
            local /= local.sum()
            weights = np.zeros(self.J, dtype=float)
            weights[blockers] = local
        pressure = weights @ self.membership
        if not np.isclose(pressure.sum(), self.config.tail_size):
            raise RuntimeError('profile pressure lost its tail mass')
        return pressure

    def prepare(self) -> dict[str, object] | None:
        if self._pending is not None:
            public = dict(self._pending['public'])
            for key in ('row_probabilities', 'predicted_row_values', 'tail_membership', 'row_sigma', 'allocation_proportional', 'allocation_range', 'allocation_neyman', 'profile_bottleneck_membership', 'endpoint_relevance'):
                public[key] = np.asarray(public[key]).copy()
            return public
        if self.reviews >= self.N:
            return None
        q = self._predict()
        row_values = np.sum(q * self.row_costs, axis=1)
        probabilities, tail_membership, sigma, components = self._selection_probabilities(q, row_values)
        row = int(self._row_rng.choice(self.R, p=probabilities))
        record_id = int(self._orders[row][-1])
        audits = self.audits_by_row
        remaining = self.N_r - audits
        observed_sums = np.sum(self.A * self.row_costs, axis=1)
        row_plugin = (observed_sums + remaining * row_values) / self.N_r
        subset_plugin = self.membership @ row_plugin / self.config.tail_size
        scale = np.zeros(self.R, dtype=float)
        available = remaining > 0
        scale[available] = remaining[available] / (self.config.tail_size * self.N_r[available] * probabilities[available])
        minimum = self.row_costs.min(axis=1)
        maximum = self.row_costs.max(axis=1)
        low_by_row = scale * (minimum - row_values)
        high_by_row = scale * (maximum - row_values)
        subset_lower = subset_plugin + np.min(self.membership * low_by_row[None, :], axis=1)
        subset_upper = subset_plugin + np.max(self.membership * high_by_row[None, :], axis=1)
        active_subset = self._active_subset_index(row_values)
        review_number = self.reviews + 1
        profile_variance = self._profile_variance_numerator / review_number
        adaptive_variance = self._adaptive_variance_numerator / review_number
        constant = 2.0 * np.log(2.0 / self.config.alpha) / self.config.stake_effective_n
        profile_base_stake = np.sqrt(constant / profile_variance)
        adaptive_base_stake = float(np.sqrt(constant / adaptive_variance))
        public = {'record_id': record_id, 'row': row, 'row_probabilities': probabilities.copy(), 'predicted_row_values': row_values.copy(), 'tail_membership': tail_membership.copy(), 'row_sigma': sigma.copy(), 'allocation_proportional': components['proportional'].copy(), 'allocation_range': components['range'].copy(), 'allocation_neyman': components['neyman'].copy(), 'profile_bottleneck_membership': components['profile_pressure'].copy(), 'endpoint_relevance': components['relevance'].copy(), 'active_subset': tuple((int(value) for value in self.subsets[active_subset])), 'selected_weight': float(scale[row]), 'adaptive_variance_estimate': float(adaptive_variance)}
        self._pending = {'public': public, 'record_id': record_id, 'row': row, 'q': q, 'row_values': row_values, 'subset_plugin': subset_plugin, 'subset_lower': subset_lower, 'subset_upper': subset_upper, 'scale': scale, 'profile_base_stake': profile_base_stake, 'adaptive_base_stake': adaptive_base_stake, 'active_subset_index': active_subset}
        return dict(public)

    @staticmethod
    def _range_capped_stakes(base: np.ndarray, grid: np.ndarray, bound: np.ndarray, *, side: Literal['plus', 'minus'], cap_fraction: float) -> np.ndarray:
        base_2d = np.asarray(base, dtype=float).reshape(-1, 1)
        bound_2d = np.asarray(bound, dtype=float).reshape(-1, 1)
        if side == 'plus':
            denominator = grid.reshape(1, -1) - bound_2d
        else:
            denominator = bound_2d - grid.reshape(1, -1)
        cap = np.full_like(denominator, np.inf)
        positive = denominator > 0.0
        cap[positive] = cap_fraction / denominator[positive]
        return np.minimum(base_2d, cap)

    def observe(self, record_id: int) -> dict[str, object]:
        pending = self._pending
        if pending is None or int(record_id) != int(pending['record_id']):
            raise ValueError('record_id must match the pending review')
        row = int(pending['row'])
        destination = int(self.verified[int(record_id)])
        q = np.asarray(pending['q'], dtype=float)
        row_values = np.asarray(pending['row_values'], dtype=float)
        subset_plugin = np.asarray(pending['subset_plugin'], dtype=float)
        scale = float(np.asarray(pending['scale'], dtype=float)[row])
        residual = float(self.row_costs[row, destination] - row_values[row])
        subset_observation = subset_plugin + self.membership[:, row] * scale * residual
        active = int(pending['active_subset_index'])
        adaptive_observation = float(subset_observation[active])
        threshold = math.log(2.0 / self.config.alpha)
        if self.envelope == 'min':
            stakes = self._range_capped_stakes(np.asarray(pending['profile_base_stake']), self.grid, np.asarray(pending['subset_lower']), side='plus', cap_fraction=self.config.cap_fraction)
            factor = 1.0 + stakes * (subset_observation[:, None] - self.grid[None, :])
            if np.any(factor < 0.1 - 1e-11):
                raise RuntimeError('profile plus factor fell below the cap margin')
            self._profile_log_wealth += np.log(factor)
            self._profile_crossed |= self._profile_log_wealth >= threshold
            adaptive_stake = self._range_capped_stakes(np.array([pending['adaptive_base_stake']]), self.grid, np.array([np.asarray(pending['subset_upper'])[active]]), side='minus', cap_fraction=self.config.cap_fraction)[0]
            adaptive_factor = 1.0 - adaptive_stake * (adaptive_observation - self.grid)
        else:
            stakes = self._range_capped_stakes(np.asarray(pending['profile_base_stake']), self.grid, np.asarray(pending['subset_upper']), side='minus', cap_fraction=self.config.cap_fraction)
            factor = 1.0 - stakes * (subset_observation[:, None] - self.grid[None, :])
            if np.any(factor < 0.1 - 1e-11):
                raise RuntimeError('profile minus factor fell below the cap margin')
            self._profile_log_wealth += np.log(factor)
            self._profile_crossed |= self._profile_log_wealth >= threshold
            adaptive_stake = self._range_capped_stakes(np.array([pending['adaptive_base_stake']]), self.grid, np.array([np.asarray(pending['subset_lower'])[active]]), side='plus', cap_fraction=self.config.cap_fraction)[0]
            adaptive_factor = 1.0 + adaptive_stake * (adaptive_observation - self.grid)
        if np.any(adaptive_factor < 0.1 - 1e-11):
            raise RuntimeError('adaptive factor fell below the cap margin')
        self._adaptive_log_wealth += np.log(adaptive_factor)
        self._adaptive_crossed |= self._adaptive_log_wealth >= threshold
        old_profile_mean = self._profile_observation_mean.copy()
        self._profile_variance_numerator += np.square(subset_observation - old_profile_mean)
        self._profile_observation_mean += (subset_observation - old_profile_mean) / (self.reviews + 1)
        old_adaptive_mean = self._adaptive_observation_mean
        self._adaptive_variance_numerator += (adaptive_observation - old_adaptive_mean) ** 2
        self._adaptive_observation_mean += (adaptive_observation - old_adaptive_mean) / (self.reviews + 1)
        self.A[row, destination] += 1
        popped = int(self._orders[row].pop())
        if popped != int(record_id):
            raise RuntimeError('within-row replay order changed')
        self._pending = None
        return {'review': self.reviews, 'record_id': int(record_id), 'row': row, 'destination': destination, 'row_contribution': float(self.row_costs[row, destination]), 'adaptive_observation': adaptive_observation}

    def interval(self) -> dict[str, float | int | bool]:
        if self.envelope == 'min':
            rejected_low = np.all(self._profile_crossed, axis=0)
            rejected_high = self._adaptive_crossed
        else:
            rejected_low = self._adaptive_crossed
            rejected_high = np.all(self._profile_crossed, axis=0)
        alive = ~(rejected_low | rejected_high)
        empty_grid_event = not np.any(alive)
        if empty_grid_event:
            self._empty_grid_events += 1
            alive = self._reported_alive.copy()
        else:
            self._reported_alive = alive.copy()
        retained = self.grid[alive]
        lower_grid = float(retained[0])
        upper_grid = float(retained[-1])
        lower = max(0.0, lower_grid - self.config.grid_step)
        upper = min(1.0, upper_grid + self.config.grid_step)
        return {'lower': lower, 'upper': upper, 'width': upper - lower, 'center': 0.5 * (lower + upper), 'alive_candidates': int(alive.sum()), 'profile_subsets': self.J, 'nested_by_running_intersection': True, 'empty_grid_event': empty_grid_event, 'empty_grid_events': self._empty_grid_events}
