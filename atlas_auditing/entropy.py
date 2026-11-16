from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from atlas_auditing.reference import DEFAULT_REFERENCE_ROOT, _load_reference_modules
POLICY_SRS = 'srs'
POLICY_PAS = 'pas'
POLICIES = (POLICY_SRS, POLICY_PAS)

@dataclass(frozen=True)
class EntropyAuditConfig:
    alpha_lower: float = 0.0225
    alpha_upper: float = 0.0225
    reference_smoothing: float = 0.01
    row_shrinkage: float = 2.0
    stratum_shrinkage: float = 1.0
    submitted_label_prior: float = 0.8
    proportional_weight: float = 0.4
    range_weight: float = 0.3
    neyman_weight: float = 0.3
    cap_fraction: float = 0.9
    entropy_resolution: float = 0.0003
    profile_max_iterations: int = 60
    endpoint_period: int = 25
    upper_starts: tuple[int, ...] = (200, 750, 2250, 5250, 11250, 23250)
    upper_restart_weights: tuple[float, ...] = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125)
    bet_scales: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    lower_bet_scales: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)

    def validate(self) -> None:
        if not (0 < self.alpha_lower < 1 and 0 < self.alpha_upper < 1):
            raise ValueError('Tail error budgets must lie in (0,1)')
        if self.alpha_lower + self.alpha_upper >= 1:
            raise ValueError('Tail error budgets must sum to less than one')
        if not 0 < self.reference_smoothing < 1:
            raise ValueError('Reference smoothing must lie in (0,1)')
        if self.row_shrinkage <= 0 or self.stratum_shrinkage <= 0:
            raise ValueError('Shrinkage strengths must be positive')
        if not 0 <= self.submitted_label_prior <= 1:
            raise ValueError('Submitted-label prior must lie in [0,1]')
        weights = np.array([self.proportional_weight, self.range_weight, self.neyman_weight])
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError('Allocation mixture weights must be nonnegative and sum to one')
        if not 0 < self.cap_fraction < 1:
            raise ValueError('Cap fraction must lie in (0,1)')
        if self.endpoint_period <= 0:
            raise ValueError('Endpoint period must be positive')

class _CrossCappedProfile:

    def __init__(self, base_class, X: np.ndarray, K: np.ndarray, D: np.ndarray, upper_caps: np.ndarray, N: int, *, alpha: float, scales: Sequence[float], cap_fraction: float) -> None:

        class Profile(base_class):

            def at(inner_self, h):
                inner_self.h = float(h)
                inner_self.m = inner_self.h + inner_self.c
                denominator = inner_self.c + inner_self.D + upper_caps
                if np.any(denominator <= 0):
                    raise ArithmeticError('Nonpositive cross-cap denominator')
                inner_self.lam = np.minimum(inner_self.ks[None, :], cap_fraction / denominator[:, None])
                inner_self.last = None
                return inner_self
        self.model = Profile(X, K, D, N, ks=tuple(scales), alpha=alpha, eta=cap_fraction)

    def endpoint(self, *, lower: np.ndarray, upper_entropy: float, resolution: float, max_iterations: int) -> dict[str, object]:
        left = 0.0
        right = max(0.0, float(upper_entropy))
        history: list[dict[str, object]] = []
        best = None
        start = None
        while right - left > resolution:
            h = 0.5 * (left + right)
            answer = self.model.solve(h, lower=lower, start=start, maxiter=max_iterations)
            start = np.asarray(answer['p'], dtype=float)
            history.append(answer)
            if answer['affine_lower'] > self.model.threshold:
                left = h
                best = answer
            else:
                right = h
        return {'entropy_lower': left, 'best': best, 'history': history, 'resolution': resolution}

class OptimizedEntropyAudit:

    def __init__(self, record_strata: Sequence[int], stratum_to_row: Sequence[int], stratum_to_submitted_label: Sequence[int], categories: int, *, seed: int, policy: str, config: EntropyAuditConfig | None=None, reference_root: Path=DEFAULT_REFERENCE_ROOT) -> None:
        if policy not in POLICIES:
            raise ValueError(f'Unknown policy {policy!r}')
        self.config = config or EntropyAuditConfig()
        self.config.validate()
        self.policy = policy
        self.record_strata = np.asarray(record_strata, dtype=int)
        self.stratum_to_row = np.asarray(stratum_to_row, dtype=int)
        self.stratum_to_submitted_label = np.asarray(stratum_to_submitted_label, dtype=int)
        self.N = len(self.record_strata)
        self.G = int(self.record_strata.max()) + 1
        self.R = int(self.stratum_to_row.max()) + 1
        self.L = int(categories)
        if self.stratum_to_row.shape != (self.G,):
            raise ValueError('Expected one submitted row per stratum')
        if self.stratum_to_submitted_label.shape != (self.G,):
            raise ValueError('Expected one submitted label per stratum')
        self.N_g = np.bincount(self.record_strata, minlength=self.G).astype(int)
        if np.any(self.N_g <= 0):
            raise ValueError('Strata must be consecutive and nonempty')
        modules = _load_reference_modules(reference_root)
        online = modules['convex_audit_online']
        self._positive_observation = modules['convex_profile_replay'].positive_observation
        self._control_observation = online.control_observation
        self._sample_dyadic_row = modules['directional_observations'].sample_dyadic_row
        self._point = online._point
        self._profile_class = online.ConvexEntropyProfile
        self._certify = online.certify
        self._interval = online._I
        self._decimal = online._D
        self._exp = online._exp
        self._fdown = online._fdown
        self._fup = online._fup
        self._wealth_lower = online.wealth_lower
        self.A = np.zeros((self.G, self.L), dtype=np.int64)
        order_rng = np.random.default_rng(seed)
        self._orders = [list(order_rng.permutation(np.flatnonzero(self.record_strata == stratum))[::-1]) for stratum in range(self.G)]
        self._row_rng = np.random.default_rng(seed + 7121)
        self._pending: dict[str, object] | None = None
        self._seen: list[int] = []
        self._labels: list[int] = []
        self._X: list[np.ndarray] = []
        self._K: list[float] = []
        self._D: list[float] = []
        self._upper_cap_history: list[float] = []
        self._Y_upper: list[float] = []
        self._Y_max: list[float] = []
        self._lower_cap_history: list[float] = []
        self._entropy_lower = 0.0
        self._entropy_upper = math.log(self.L)
        self._last_lower_certificate: dict[str, object] | None = None
        submitted_counts = np.bincount(self.stratum_to_submitted_label, weights=self.N_g, minlength=self.L)
        submitted_probabilities = submitted_counts / self.N
        positive = submitted_probabilities > 0
        self.submitted_entropy = float(-np.sum(submitted_probabilities[positive] * np.log(submitted_probabilities[positive])))

    @property
    def reviews(self) -> int:
        return len(self._seen)

    def _predict(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total = self.A.sum(axis=0).astype(float)
        reviewed = float(total.sum())
        pooled = (total + 1.0 / self.L) / (reviewed + 1.0)
        row_counts = np.zeros((self.R, self.L), dtype=float)
        np.add.at(row_counts, self.stratum_to_row, self.A)
        row_reviews = row_counts.sum(axis=1)
        row_prior = np.tile((1.0 - self.config.submitted_label_prior) * pooled, (self.R, 1))
        row_labels = np.zeros(self.R, dtype=int)
        for row in range(self.R):
            labels = self.stratum_to_submitted_label[self.stratum_to_row == row]
            if not len(labels) or np.any(labels != labels[0]):
                raise RuntimeError('A submitted row maps to inconsistent labels')
            row_labels[row] = int(labels[0])
        row_prior[np.arange(self.R), row_labels] += self.config.submitted_label_prior
        row_prediction = (row_counts + self.config.row_shrinkage * row_prior) / (row_reviews[:, None] + self.config.row_shrinkage)
        stratum_reviews = self.A.sum(axis=1).astype(float)
        q = (self.A + self.config.stratum_shrinkage * row_prediction[self.stratum_to_row]) / (stratum_reviews[:, None] + self.config.stratum_shrinkage)
        remaining = self.N_g - self.A.sum(axis=1)
        p_hat = (total + (remaining[:, None] * q).sum(axis=0)) / self.N
        reference = (1.0 - self.config.reference_smoothing) * p_hat + self.config.reference_smoothing / self.L
        reference /= reference.sum()
        influence = -np.log(reference)
        return (q, p_hat, influence)

    def _desired_probabilities(self, q: np.ndarray, influence: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        remaining = self.N_g - self.A.sum(axis=1)
        active = remaining > 0
        proportional = np.zeros(self.G)
        proportional[active] = remaining[active] / remaining[active].sum()
        if self.policy == POLICY_SRS:
            return (proportional, {'proportional': proportional, 'range': proportional, 'neyman': proportional})
        predicted = q @ influence
        centered = influence[None, :] - predicted[:, None]
        sigma = np.sqrt(np.maximum(np.sum(q * centered ** 2, axis=1), 0.0))
        range_weight = np.zeros(self.G)
        range_raw = remaining * (1.0 + predicted)
        range_weight[active] = range_raw[active] / range_raw[active].sum()
        neyman = np.zeros(self.G)
        neyman_raw = remaining * sigma
        if neyman_raw[active].sum() > 0:
            neyman[active] = neyman_raw[active] / neyman_raw[active].sum()
        else:
            neyman = proportional.copy()
        desired = self.config.proportional_weight * proportional + self.config.range_weight * range_weight + self.config.neyman_weight * neyman
        desired[~active] = 0.0
        desired /= desired.sum()
        return (desired, {'proportional': proportional, 'range': range_weight, 'neyman': neyman})

    def prepare(self) -> dict[str, object] | None:
        if self._pending is not None:
            result = dict(self._pending['public'])
            result['row_probabilities'] = np.asarray(result['row_probabilities']).copy()
            return result
        if self.reviews == self.N:
            return None
        q, p_hat, influence = self._predict()
        desired, components = self._desired_probabilities(q, influence)
        remaining = self.N_g - self.A.sum(axis=1)
        stratum, probabilities, _ = self._sample_dyadic_row(self._row_rng, desired, remaining > 0)
        record_id = int(self._orders[stratum][-1])
        reviews_by_stratum = self.A.sum(axis=1)
        beta = q * (reviews_by_stratum + 1.0)[:, None] - self.A
        frozen = SimpleNamespace(N=self.N, L=self.L, N_r=self.N_g.copy(), A=self.A.copy(), beta=beta, dl=influence.copy(), bl=0.0, du=influence.copy(), bu=0.0)
        public = {'record_id': record_id, 'stratum': int(stratum), 'submitted_row': int(self.stratum_to_row[stratum]), 'row_probabilities': probabilities.copy(), 'policy': self.policy, 'reference_composition': p_hat.copy(), 'influence': influence.copy(), 'allocation_proportional': components['proportional'].copy(), 'allocation_range': components['range'].copy(), 'allocation_neyman': components['neyman'].copy()}
        self._pending = {'public': public, 'record_id': record_id, 'stratum': int(stratum), 'probabilities': probabilities, 'state': frozen}
        return dict(public)

    def observe(self, record_id: int, verified_label: int) -> dict[str, object]:
        pending = self._pending
        if pending is None or int(record_id) != pending['record_id']:
            raise ValueError('record_id must match the pending request')
        if not 0 <= int(verified_label) < self.L:
            raise ValueError('verified_label is outside the declared vocabulary')
        stratum = int(pending['stratum'])
        probabilities = np.asarray(pending['probabilities'], dtype=float)
        state = pending['state']
        zero = self._point(0.0)
        zeros = np.zeros(self.G)
        X, _, K, _, D = self._positive_observation(state, stratum, int(verified_label), probabilities, 0.0, zero, zeros)
        upper = self._control_observation(state, stratum, int(verified_label), probabilities, 0.0, zero, zeros, zeros)
        self._X.append(np.asarray(X, dtype=float))
        self._K.append(float(K))
        self._D.append(float(D))
        self._upper_cap_history.append(float(self._entropy_upper))
        self._Y_upper.append(float(upper['entropy_yhi']))
        self._Y_max.append(float(upper['entropy_upper']))
        self._lower_cap_history.append(float(self._entropy_lower))
        self.A[stratum, int(verified_label)] += 1
        popped = int(self._orders[stratum].pop())
        if popped != int(record_id):
            raise RuntimeError('Within-stratum replay order changed')
        self._seen.append(int(record_id))
        self._labels.append(int(verified_label))
        self._pending = None
        return {'reviews': self.reviews, 'record_id': int(record_id), 'stratum': stratum, 'verified_label': int(verified_label)}

    def _upper_log_wealth(self, h: float) -> float:
        y = np.asarray(self._Y_upper, dtype=float)
        maximum = np.asarray(self._Y_max, dtype=float)
        lower_caps = np.asarray(self._lower_cap_history, dtype=float)
        scales = np.asarray(self.config.bet_scales, dtype=float)
        denominator = np.maximum(maximum - lower_caps, 1e-15)
        lambdas = np.minimum(scales[None, :], self.config.cap_fraction / denominator[:, None])
        factors = 1.0 + lambdas * (float(h) - y[:, None])
        if np.any(factors <= 0):
            return -np.inf
        log_factors = np.log(factors)
        epochs = []
        for start in self.config.upper_starts:
            epochs.append(logsumexp(log_factors[min(start, len(y)):].sum(axis=0)) - math.log(len(scales)))
        return float(logsumexp(np.log(np.asarray(self.config.upper_restart_weights)) + np.asarray(epochs)))

    def _upper_is_certified(self, h: float) -> bool:
        y = np.asarray(self._Y_upper, dtype=float)
        maximum = np.asarray(self._Y_max, dtype=float)
        lower_caps = np.asarray(self._lower_cap_history, dtype=float)
        adjusted = maximum + float(h) - lower_caps
        scaled = self._wealth_lower(float(h), y, y, np.zeros_like(y), adjusted, np.asarray(self.config.bet_scales), self.config.upper_starts, self.config.upper_restart_weights, 'hi', return_scaled=True)
        alpha_effective = float(np.nextafter(self.config.alpha_upper, 0.0))
        target = self._fup((self._interval(1) / self._interval(self._decimal(alpha_effective))).hi)
        return bool(scaled.exceeds(target))

    def _upper_endpoint(self) -> tuple[float, bool]:
        threshold = math.log(1.0 / self.config.alpha_upper)
        left = min(self._entropy_lower, self._entropy_upper)
        right = math.log(self.L)
        if self._upper_log_wealth(right) < threshold:
            return (self._entropy_upper, True)
        if self._upper_log_wealth(left) >= threshold:
            proposal = left
        else:
            proposal = brentq(lambda h: self._upper_log_wealth(h) - threshold, left, right)
        proposal = min(right, proposal + 1e-08)
        if not self._upper_is_certified(proposal):
            lo = proposal
            hi = right
            if not self._upper_is_certified(hi):
                return (self._entropy_upper, False)
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                if self._upper_is_certified(mid):
                    hi = mid
                else:
                    lo = mid
            proposal = hi
        return (min(self._entropy_upper, proposal), True)

    def _lower_endpoint(self) -> tuple[float, dict[str, object] | None]:
        X = np.asarray(self._X, dtype=float)
        K = np.asarray(self._K, dtype=float)
        D = np.asarray(self._D, dtype=float)
        upper_caps = np.asarray(self._upper_cap_history, dtype=float)
        observed = self.A.sum(axis=0).astype(np.int64)
        lower = observed / self.N
        model = _CrossCappedProfile(self._profile_class, X, K, D, upper_caps, self.N, alpha=self.config.alpha_lower, scales=self.config.lower_bet_scales, cap_fraction=self.config.cap_fraction)
        answer = model.endpoint(lower=lower, upper_entropy=min(self._entropy_upper, self.submitted_entropy + 0.5), resolution=self.config.entropy_resolution, max_iterations=self.config.profile_max_iterations)
        proposed = max(self._entropy_lower, float(answer['entropy_lower']))
        if proposed <= self._entropy_lower or answer['best'] is None:
            return (self._entropy_lower, self._last_lower_certificate)
        point = np.asarray(answer['best']['p'], dtype=float)
        h = proposed
        step = self.config.entropy_resolution
        certificate = None
        while h > self._entropy_lower:
            adjusted_D = D + upper_caps - h
            if np.any(adjusted_D < -1e-12):
                h = max(self._entropy_lower, h - step)
                step *= 2
                continue
            adjusted_D = np.maximum(adjusted_D, 0.0)
            certificate = self._certify(X, K, adjusted_D, self.N, point, observed, h, alpha=self.config.alpha_lower, ks=self.config.lower_bet_scales, eta=self.config.cap_fraction)
            if certificate['certified']:
                return (h, certificate)
            h = max(self._entropy_lower, h - step)
            step *= 2
        return (self._entropy_lower, self._last_lower_certificate)

    def interval(self, *, force: bool=False) -> dict[str, object]:
        if not self.reviews:
            return self._interval_result(True, True)
        if force or self.reviews % self.config.endpoint_period == 0:
            lower, certificate = self._lower_endpoint()
            self._entropy_lower = max(self._entropy_lower, lower)
            if certificate is not None:
                self._last_lower_certificate = certificate
            upper, upper_verified = self._upper_endpoint()
            self._entropy_upper = min(self._entropy_upper, upper)
        else:
            upper_verified = True
        return self._interval_result(self._last_lower_certificate is not None or self._entropy_lower == 0, upper_verified)

    def _interval_result(self, lower_verified: bool, upper_verified: bool) -> dict[str, object]:
        lower_p2 = max(1.0, self._fdown(self._exp(self._interval(self._decimal(float(self._entropy_lower)))).lo))
        upper_p2 = min(float(self.L), self._fup(self._exp(self._interval(self._decimal(float(self._entropy_upper)))).hi))
        empty = bool(lower_p2 > upper_p2)
        harmonic = float(2 * lower_p2 * upper_p2 / (lower_p2 + upper_p2))
        relative = float((upper_p2 - lower_p2) / (upper_p2 + lower_p2))
        q, p_hat, _ = self._predict()
        positive = p_hat > 0
        estimate = float(np.exp(-np.sum(p_hat[positive] * np.log(p_hat[positive]))))
        return {'reviews': self.reviews, 'entropy_lower': float(self._entropy_lower), 'entropy_upper': float(self._entropy_upper), 'entropy_width': float(self._entropy_upper - self._entropy_lower), 'lower': float(lower_p2), 'upper': float(upper_p2), 'full_width': float(upper_p2 - lower_p2), 'harmonic_report': harmonic, 'harmonic_relative_error_bound': relative, 'estimate': estimate, 'numeric_verified': bool(lower_verified and upper_verified and (not empty)), 'empty': empty, 'confidence_at_least': float(1.0 - self.config.alpha_lower - self.config.alpha_upper), 'lower_certificate': self._last_lower_certificate, 'prediction_rows': q}
