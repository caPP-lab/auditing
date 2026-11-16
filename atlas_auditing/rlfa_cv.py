from dataclasses import dataclass
import math
import numpy as np
from atlas_auditing.audit import LabelBlindAudit
from atlas_auditing.rlfa import AuthorsRLFAConfig, AuthorsRLFAConfidenceSequence, source_bet

@dataclass(frozen=True)
class AuthorsRLFACVConfig(AuthorsRLFAConfig):
    beta_max: float = 0.5
    smoothing_term: float = 1.0

    def validate(self):
        super().validate()
        if not np.isfinite(self.beta_max) or not 0 <= self.beta_max <= 1:
            raise ValueError('beta_max must be in [0,1]')
        if not np.isfinite(self.smoothing_term) or self.smoothing_term <= 0:
            raise ValueError('smoothing_term must be positive')

class AuthorsRLFACVConfidenceSequence(AuthorsRLFAConfidenceSequence):

    def __init__(self, config=None):
        super().__init__(config or AuthorsRLFACVConfig())
        self.sum_cross = np.zeros_like(self.grid)
        self.sum_control2 = 0.0

    def _coefficient(self, cross):
        cfg = self.config
        return np.clip((cross + cfg.smoothing_term) / (self.sum_control2 + cfg.tol + 2 * cfg.smoothing_term), -cfg.beta_max, cfg.beta_max)

    def prepare(self, *, audited_total, remaining_weight, max_importance, control_min, control_max):
        if self._pending is not None:
            raise RuntimeError('observe the pending draw first')
        values = [audited_total, remaining_weight, max_importance, control_min, control_max]
        if not np.isfinite(values).all() or min(values[:3]) < 0 or audited_total + remaining_weight > 1 + 1e-12 or (not control_min <= 0 <= control_max):
            raise ValueError('invalid public support')
        beta = self._coefficient(self.sum_cross)

        def support(coefficient, left, right, bound):
            c0, c1 = (coefficient * control_min, coefficient * control_max)
            return (audited_total - right - np.maximum(c0, c1), audited_total + bound - left - np.minimum(c0, c1))
        lo, hi = support(beta, self.left, self.right, max_importance)
        stake = source_bet(self.sum_d, self.sum_d2, lo, hi, self.config)
        self._pending = dict(audited_total=audited_total, remaining_weight=remaining_weight, max_importance=max_importance, control_min=control_min, control_max=control_max, stake=stake, beta=beta)
        return dict(stake=stake.copy(), beta=beta.copy())

    def observe(self, *, importance_observation, control, revealed_weight, revealed_value):
        p = self._pending
        if p is None:
            raise RuntimeError('prepare before observing')
        if not np.isfinite([importance_observation, control, revealed_weight, revealed_value]).all():
            raise ValueError('nonfinite observation')
        if not 0 <= revealed_value <= 1 or not 0 <= revealed_weight <= p['remaining_weight'] + 1e-12 or (not -1e-12 <= importance_observation <= p['max_importance'] + 1e-12) or (not p['control_min'] - 1e-12 <= control <= p['control_max'] + 1e-12):
            raise ValueError('observation outside predictable support')
        base = p['audited_total'] + importance_observation
        adjusted = base - p['beta'] * control
        factors = (1 + p['stake'] * (adjusted - self.left), 1 + p['stake'] * (adjusted - self.right))
        self.minimum_factor = min(self.minimum_factor, *(float(f.min()) for f in factors))
        if self.minimum_factor <= 0:
            raise ArithmeticError('nonpositive factor: no outcome-dependent bet reset permitted')
        self.log_left += np.log(factors[0])
        self.log_right += np.log(factors[1])
        self.rejected |= np.minimum(self.log_left, self.log_right) >= math.log(1 / self.config.alpha) + 1e-12
        raw = base - self.grid
        self.sum_cross += control * raw
        self.sum_control2 += control * control
        d = raw - p['beta'][-1] * control
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

class LabelBlindAuthorsRLFACV(LabelBlindAudit):

    def __init__(self, record_rows, row_to_submitted_label, *, seed, config=None, rlfa_config=None, row_sampler=None):
        super().__init__(record_rows, row_to_submitted_label, seed=seed, policy='srs', config=config, row_sampler=row_sampler)
        self.cs = AuthorsRLFACVConfidenceSequence(rlfa_config or AuthorsRLFACVConfig(alpha=self.config.alpha, grid_step=self.config.grid_step))

    def prepare(self):
        if self._pending is not None:
            return dict(self._pending['public'])
        if self.reviews == self.N:
            return None
        remaining = self.N_r - self.audits
        active = remaining > 0
        prediction = self._predict_error()
        desired = remaining / remaining.sum()
        selected, q, _ = self._sample_dyadic_row(self._row_rng, desired, active)
        selected = int(selected)
        weights = np.zeros(self.R)
        weights[active] = remaining[active] / (self.N * q[active])
        expected = float(q @ prediction)
        controls = prediction[active] - expected
        self.cs.prepare(audited_total=float(self.errors.sum() / self.N), remaining_weight=float(remaining.sum() / self.N), max_importance=float(weights.max()), control_min=float(min(0.0, controls.min())), control_max=float(max(0.0, controls.max())))
        record = int(self._orders[selected][-1])
        public = dict(record_id=record, row=selected, row_probabilities=q.copy(), predicted_error=prediction.copy(), expected_prediction=expected, selected_weight=float(weights[selected]), record_probability=float(q[selected] / remaining[selected]))
        self._pending = dict(public=public, record_id=record, row=selected)
        return dict(public)

    def observe_error(self, record_id, error):
        if error not in (0, 1) or self._pending is None or record_id != self._pending['record_id']:
            raise ValueError('reveal must match the selected record and be binary')
        p = self._pending['public']
        row = p['row']
        z = p['selected_weight'] * error
        control = float(p['predicted_error'][row] - p['expected_prediction'])
        beta_at_one = float(self.cs._pending['beta'][-1])
        self.cs.observe(importance_observation=z, control=control, revealed_weight=1 / self.N, revealed_value=error)
        self.audits[row] += 1
        self.errors[row] += error
        assert int(self._orders[row].pop()) == record_id
        self._pending = None
        return dict(error=int(error), observation=z, control=control, shared_history_beta=beta_at_one)

    def interval(self):
        return self.cs.interval()
