from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from atlas_auditing.mislabel import MislabelAuditConfig

class SubsetF1Audit:

    def __init__(self, submitted: Sequence[int], target_classes: Sequence[int], categories: int, *, seed: int, policy: str, config: MislabelAuditConfig | None=None):
        self.config = config or MislabelAuditConfig()
        self.config.validate()
        if policy not in {'pas', 'srs'}:
            raise ValueError('policy must be pas or srs')
        self.policy = policy
        self.submitted = np.asarray(submitted, dtype=int)
        self.L = int(categories)
        if self.submitted.ndim != 1 or not len(self.submitted) or self.submitted.min() < 0 or (self.submitted.max() >= self.L):
            raise ValueError('invalid submitted labels')
        target = np.unique(np.asarray(target_classes, dtype=int))
        if not len(target) or target.min() < 0 or target.max() >= self.L:
            raise ValueError('invalid target classes')
        self.target = np.zeros(self.L, dtype=bool)
        self.target[target] = True
        self.N = len(self.submitted)
        self.row_labels, self.record_rows = np.unique(self.submitted, return_inverse=True)
        self.R = len(self.row_labels)
        self.in_target = self.target[self.row_labels]
        self.N_r = np.bincount(self.record_rows, minlength=self.R)
        self.submitted_target_count = int(self.N_r @ self.in_target)
        if self.submitted_target_count == 0:
            raise ValueError('require a positive known denominator contribution')
        self.audits = np.zeros(self.R, dtype=int)
        self.matches = np.zeros(self.R, dtype=int)
        self.target_counts = np.zeros(self.R, dtype=int)
        self.label_counts = np.zeros(self.L, dtype=int)
        self.audited_u = 0.0
        self.audited_w = 0.0
        self.seen: list[int] = []
        self.observed_u: list[float] = []
        self.observed_w: list[float] = []
        order_rng = np.random.default_rng(seed)
        self._orders = [list(order_rng.permutation(np.flatnonzero(self.record_rows == g))[::-1]) for g in range(self.R)]
        self._rng = np.random.default_rng(seed + 7121)
        self.grid = np.linspace(0.0, 1.0, int(round(1 / self.config.grid_step)) + 1)
        self.log_plus = np.zeros_like(self.grid)
        self.log_minus = np.zeros_like(self.grid)
        self.rejected_cells = np.zeros(len(self.grid) - 1, dtype=bool)
        self._pending = None
        self._variance_numerator = 1.0
        self._mean_observation = None
        self.minimum_factor = 1.0
        self.atom_u = np.zeros((self.R, 3))
        self.atom_w = np.zeros((self.R, 3))
        self.atom_u[self.in_target, 0] = 2.0
        self.atom_w[self.in_target] = [2.0, 2.0, 1.0]
        self.atom_w[~self.in_target] = [0.0, 1.0, 0.0]
        self.possible = np.ones((self.R, 3), dtype=bool)
        self.possible[~self.in_target, 0] = False
        if len(target) == 1:
            self.possible[self.in_target, 1] = False
        if len(target) == self.L:
            self.possible[:, 2] = False

    @property
    def reviews(self):
        return len(self.seen)

    def _predict(self):
        pooled = (self.label_counts + 1 / self.L) / (self.reviews + 1)
        trust = self.config.submitted_label_probability
        prior_match = trust + (1 - trust) * pooled[self.row_labels]
        prior_target = trust * self.in_target + (1 - trust) * pooled[self.target].sum()
        kr, ks = (self.config.row_shrinkage, self.config.stratum_shrinkage)
        row_match = (self.matches + kr * prior_match) / (self.audits + kr)
        row_target = (self.target_counts + kr * prior_target) / (self.audits + kr)
        pm = (self.matches + ks * row_match) / (self.audits + ks)
        pa = (self.target_counts + ks * row_target) / (self.audits + ks)
        q = np.column_stack((np.where(self.in_target, pm, 0.0), pa - np.where(self.in_target, pm, 0.0), 1 - pa))
        q = np.maximum(q, 0.0)
        q[~self.possible] = 0.0
        q /= q.sum(axis=1, keepdims=True)
        means = np.column_stack(((q * self.atom_u).sum(1), (q * self.atom_w).sum(1)))
        remaining = self.N_r - self.audits
        plugin = (np.array([self.audited_u, self.audited_w]) + remaining @ means) / self.N
        fhat = float(np.clip(plugin[0] / plugin[1], 0.0, 1.0))
        return (q, means, plugin, fhat)

    def _allocation(self, q, fhat):
        remaining = self.N_r - self.audits
        proportional = remaining / remaining.sum()
        d = self.atom_u - fhat * self.atom_w
        mean = (q * d).sum(1)
        sigma = np.sqrt(np.maximum((q * (d - mean[:, None]) ** 2).sum(1), 0.0))
        neyman = remaining * sigma
        neyman = neyman / neyman.sum() if neyman.sum() else proportional
        if self.policy == 'srs':
            return proportional
        floor = self.config.srs_floor
        p = floor * proportional + (1 - floor) * neyman
        return p / p.sum()

    def prepare(self):
        if self._pending is not None:
            return self._pending
        if self.reviews == self.N:
            return None
        q, means, plugin, fhat = self._predict()
        pi = self._allocation(q, fhat)
        remaining = self.N_r - self.audits
        active = remaining > 0
        scale = np.zeros(self.R)
        scale[active] = remaining[active] / (self.N * pi[active])
        possible = self.possible & active[:, None]
        extremes = []
        for f in [0.0, 1.0]:
            atom = self.atom_u - f * self.atom_w
            pred = means[:, 0] - f * means[:, 1]
            z = plugin[0] - f * plugin[1] + scale[:, None] * (atom - pred[:, None])
            extremes.extend([float(z[possible].min()), float(z[possible].max())])
        lo, hi = (min(extremes), max(extremes))
        variance = self._variance_numerator / (self.reviews + 1)
        base = math.sqrt(2 * math.log(2 / self.config.alpha) / (self.config.stake_effective_n * max(variance, 1e-15)))
        lp = min(base, self.config.cap_fraction / -lo) if lo < 0 else base
        lm = min(base, self.config.cap_fraction / hi) if hi > 0 else base
        row = int(self._rng.choice(self.R, p=pi))
        self._pending = {'record_id': int(self._orders[row][-1]), 'row': row, 'pi': pi, 'scale': scale, 'q': q, 'means': means, 'plugin': plugin, 'fhat': fhat, 'lambda_plus': lp, 'lambda_minus': lm, 'global_range': (lo, hi)}
        return self._pending

    def observe(self, record_id: int, verified_label: int):
        p = self._pending
        if p is None or record_id != p['record_id'] or (not 0 <= verified_label < self.L):
            raise ValueError('invalid requested observation')
        row = p['row']
        a = bool(self.in_target[row])
        b = bool(self.target[verified_label])
        match = bool(self.row_labels[row] == verified_label)
        u, w = (2.0 * (a and match), float(a) + float(b))
        z = p['plugin'] + p['scale'][row] * (np.array([u, w]) - p['means'][row])
        centered = z[0] - self.grid * z[1]
        plus = 1 + p['lambda_plus'] * centered
        minus = 1 - p['lambda_minus'] * centered
        minimum = min(float(plus.min()), float(minus.min()))
        if minimum < 1 - self.config.cap_fraction - 1e-10:
            raise ArithmeticError('range-capped factor violated')
        self.minimum_factor = min(self.minimum_factor, minimum)
        self.log_plus += np.log(plus)
        self.log_minus += np.log(minus)
        threshold = math.log(2 / self.config.alpha) + 1e-10
        reject_plus = np.minimum(self.log_plus[:-1], self.log_plus[1:]) >= threshold
        reject_minus = np.minimum(self.log_minus[:-1], self.log_minus[1:]) >= threshold
        self.rejected_cells |= reject_plus | reject_minus
        old = p['plugin'] if self._mean_observation is None else self._mean_observation
        delta = z - old
        self._variance_numerator += float(delta[0] - p['fhat'] * delta[1]) ** 2
        self._mean_observation = old + delta / (self.reviews + 1)
        self.audits[row] += 1
        self.matches[row] += match
        self.target_counts[row] += b
        self.label_counts[verified_label] += 1
        self.audited_u += u
        self.audited_w += w
        assert self._orders[row].pop() == record_id
        self.seen.append(int(record_id))
        self.observed_u.append(u)
        self.observed_w.append(w)
        self._pending = None
        return {'u': u, 'w': w, 'observation': z}

    def interval(self):
        if self.reviews == self.N:
            exact = self.audited_u / self.audited_w
            return {'lower': exact, 'upper': exact, 'width': 0.0, 'half_width': 0.0, 'empty': False}
        alive = np.flatnonzero(~self.rejected_cells)
        if not len(alive):
            return {'lower': 1.0, 'upper': 0.0, 'width': -1.0, 'half_width': -0.5, 'empty': True}
        lo, hi = (float(self.grid[alive[0]]), float(self.grid[alive[-1] + 1]))
        return {'lower': lo, 'upper': hi, 'width': hi - lo, 'half_width': (hi - lo) / 2, 'empty': False}
