from dataclasses import dataclass
import math
import numpy as np
from atlas_auditing.audit import LabelBlindAudit
from atlas_auditing.strata import _categorical_row_sampler
from atlas_auditing.allocation import rlfa_probabilities

@dataclass(frozen=True)
class AuthorsRLFAConfig:
    alpha: float = 0.045
    grid_step: float = 0.0005
    lambda_max: float = 2.5
    tol: float = 1e-10

    def validate(self):
        if not 0 < self.alpha < 1 or not np.isfinite(self.lambda_max) or self.lambda_max <= 0:
            raise ValueError('invalid alpha or lambda_max')
        if not 0 < self.grid_step <= 1 or not np.isclose(1 / self.grid_step, round(1 / self.grid_step)):
            raise ValueError('grid_step must divide [0,1]')
        if self.tol <= 0:
            raise ValueError('invalid numerical tolerances')

def source_bet(sum_payoff, sum_squared, minimum, maximum, config):
    if np.any(maximum < minimum):
        raise ValueError('invalid payoff range')
    lower = np.full_like(minimum, -config.lambda_max)
    upper = np.full_like(maximum, config.lambda_max)
    np.divide(-1.0, maximum + config.tol, out=lower, where=maximum > 0)
    np.divide(1.0, -minimum + config.tol, out=upper, where=minimum < 0)
    lower = np.maximum(lower, -config.lambda_max)
    upper = np.minimum(upper, config.lambda_max)
    return np.clip(sum_payoff / (sum_squared + config.tol), lower, upper)

class AuthorsRLFAConfidenceSequence:

    def __init__(self, config=None):
        self.config = config or AuthorsRLFAConfig()
        self.config.validate()
        self.grid = np.linspace(0.0, 1.0, int(round(1 / self.config.grid_step)) + 1)
        middle = (self.grid[:-1] + self.grid[1:]) / 2
        self.left, self.right = (np.r_[0.0, middle], np.r_[middle, 1.0])
        self.sum_d = np.zeros_like(self.grid)
        self.sum_d2 = np.zeros_like(self.grid)
        self.log_left = np.zeros_like(self.grid)
        self.log_right = np.zeros_like(self.grid)
        self.rejected = np.zeros(len(self.grid), dtype=bool)
        self.lower, self.upper = (0.0, 1.0)
        self.minimum_factor = 1.0
        self._pending = None

    def prepare(self, *, audited_total, remaining_weight, max_importance):
        if self._pending is not None:
            raise RuntimeError('observe the pending draw first')
        inputs = np.array([audited_total, remaining_weight, max_importance])
        if not np.isfinite(inputs).all() or np.any(inputs < 0) or audited_total + remaining_weight > 1 + 1e-12:
            raise ValueError('invalid public support')
        stake = source_bet(self.sum_d, self.sum_d2, audited_total - self.right, audited_total + max_importance - self.left, self.config)
        self._pending = dict(audited_total=audited_total, remaining_weight=remaining_weight, max_importance=max_importance, stake=stake)
        return dict(stake=stake.copy())

    def observe(self, *, importance_observation, revealed_weight, revealed_value):
        p = self._pending
        if p is None:
            raise RuntimeError('prepare before observing')
        if not 0 <= revealed_value <= 1 or not 0 <= revealed_weight <= p['remaining_weight'] + 1e-12:
            raise ValueError('invalid revealed observation')
        if not np.isfinite([importance_observation]).all():
            raise ValueError('nonfinite importance observation')
        if not -1e-12 <= importance_observation <= p['max_importance'] + 1e-12:
            raise ValueError('importance observation outside predictable range')
        base = p['audited_total'] + importance_observation
        factors = (1 + p['stake'] * (base - self.left), 1 + p['stake'] * (base - self.right))
        self.minimum_factor = min(self.minimum_factor, *(float(f.min()) for f in factors))
        if self.minimum_factor <= 0:
            raise ArithmeticError('nonpositive factor: inspect support/numerical failure')
        self.log_left += np.log(factors[0])
        self.log_right += np.log(factors[1])
        self.rejected |= np.minimum(self.log_left, self.log_right) >= math.log(1 / self.config.alpha) + 1e-12
        d = base - self.grid
        self.sum_d += d
        self.sum_d2 += d * d
        logical_lower = p['audited_total'] + revealed_weight * revealed_value
        logical_upper = logical_lower + max(0.0, p['remaining_weight'] - revealed_weight)
        alive = np.flatnonzero(~self.rejected)
        if len(alive):
            self.lower = max(self.lower, logical_lower, float(self.left[alive[0]]))
            self.upper = min(self.upper, logical_upper, float(self.right[alive[-1]]))
        else:
            self.lower, self.upper = (1.0, 0.0)
        self._pending = None

    def interval(self):
        empty = self.lower > self.upper
        return dict(lower=self.lower, upper=self.upper, empty=empty, half_width=(self.upper - self.lower) / 2 if not empty else np.nan)

class LabelBlindAuthorsRLFA(LabelBlindAudit):

    def __init__(self, record_rows, row_to_submitted_label, *, seed, config=None, rlfa_config=None):
        super().__init__(record_rows, row_to_submitted_label, seed=seed, policy='srs', config=config, row_sampler=_categorical_row_sampler)
        self.cs = AuthorsRLFAConfidenceSequence(rlfa_config or AuthorsRLFAConfig(alpha=self.config.alpha, grid_step=self.config.grid_step))

    def prepare(self):
        if self._pending is not None:
            return dict(self._pending['public'])
        if self.reviews == self.N:
            return None
        remaining = self.N_r - self.audits
        active = remaining > 0
        prediction = self._predict_error()
        desired = rlfa_probabilities(remaining, prediction, 0.0)
        selected, q, _ = self._sample_dyadic_row(self._row_rng, desired, active)
        selected = int(selected)
        weights = np.zeros(self.R)
        weights[active] = remaining[active] / (self.N * q[active])
        self.cs.prepare(audited_total=float(self.errors.sum() / self.N), remaining_weight=float(remaining.sum() / self.N), max_importance=float(weights.max()))
        record = int(self._orders[selected][-1])
        public = dict(record_id=record, row=selected, row_probabilities=q.copy(), predicted_error=prediction.copy(), selected_weight=float(weights[selected]), record_probability=float(q[selected] / remaining[selected]))
        self._pending = dict(public=public, record_id=record, row=selected)
        return dict(public)

    def observe_error(self, record_id, error):
        if error not in (0, 1) or self._pending is None or record_id != self._pending['record_id']:
            raise ValueError('reveal must match the selected record and be binary')
        p = self._pending['public']
        row = p['row']
        z = p['selected_weight'] * error
        self.cs.observe(importance_observation=z, revealed_weight=1 / self.N, revealed_value=error)
        self.audits[row] += 1
        self.errors[row] += error
        assert int(self._orders[row].pop()) == record_id
        self._pending = None
        return dict(error=int(error), observation=z)

    def interval(self):
        return self.cs.interval()
