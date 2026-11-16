from __future__ import annotations
import numpy as np
from atlas_auditing.entropy import OptimizedEntropyAudit
from atlas_auditing.tail_scores import CifarWorstThreeAudit
from atlas_auditing.f1 import SubsetF1Audit

def categorical_sigma(q):
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('Expected finite, nonnegative categorical probabilities')
    if not np.allclose(q.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise ValueError('Categorical probabilities must sum to one')
    return np.sqrt(np.maximum(1.0 - np.sum(q * q, axis=1), 0.0))

def uncertainty_allocation(remaining, sigma, epsilon):
    remaining, sigma = (np.asarray(remaining), np.asarray(sigma, dtype=float))
    if remaining.ndim != 1 or sigma.shape != remaining.shape or (not np.isfinite(remaining).all()) or (not np.isfinite(sigma).all()) or np.any(remaining < 0) or np.any(sigma < 0) or (not 0 <= epsilon <= 1):
        raise ValueError('Invalid allocation inputs')
    active = remaining > 0
    if not active.any():
        raise ValueError('Cannot sample an exhausted population')
    proportional = np.zeros(len(remaining), dtype=float)
    proportional[active] = remaining[active] / remaining[active].sum()
    raw = remaining * sigma
    priority = np.zeros_like(proportional)
    if raw[active].sum() > 0:
        priority[active] = raw[active] / raw[active].sum()
    else:
        priority = proportional.copy()
    probabilities = epsilon * proportional + (1.0 - epsilon) * priority
    probabilities[~active] = 0.0
    probabilities /= probabilities.sum()
    return (probabilities, proportional, priority)

class UncertaintyOnlyEntropyAudit(OptimizedEntropyAudit):

    def _desired_probabilities(self, q, influence):
        remaining = self.N_g - self.A.sum(axis=1)
        probabilities, proportional, priority = uncertainty_allocation(remaining, categorical_sigma(q), self.config.proportional_weight)
        return (probabilities, dict(proportional=proportional, range=priority, neyman=priority))

class UncertaintyOnlyCifarAudit(CifarWorstThreeAudit):

    def __init__(self, submitted, **kwargs):
        super().__init__(submitted, np.zeros(len(submitted), dtype=int), **kwargs)

    def _selection_probabilities(self, q, row_values):
        sigma = categorical_sigma(q)
        epsilon = self.config.proportional_weight
        probabilities, proportional, priority = uncertainty_allocation(self.N_r - self.audits_by_row, sigma, epsilon)
        neutral = np.ones(self.R)
        return (probabilities, neutral, sigma, dict(proportional=proportional, range=priority, neyman=priority, profile_pressure=neutral, relevance=neutral))

    def observe(self, record_id, verified_label):
        if self._pending is None or int(record_id) != self._pending['public']['record_id']:
            raise ValueError('Reveal must match the selected record')
        if not 0 <= verified_label < self.R:
            raise ValueError('Invalid verified label')
        self.verified[record_id] = verified_label
        return super().observe(record_id)

class UncertaintyOnlyF1Audit(SubsetF1Audit):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._label_counts_by_row = np.zeros((self.R, self.L), dtype=np.int64)
        self._row_count_norm2 = np.zeros(self.R, dtype=np.int64)
        self._row_pooled_dot = np.zeros(self.R, dtype=np.int64)

    def _label_sigma(self):
        n = self.audits.astype(float)
        kr, ks = (self.config.row_shrinkage, self.config.stratum_shrinkage)
        denominator = (n + kr) * (n + ks)
        a, b = ((n + kr + ks) / denominator, kr * ks / denominator)
        pooled = (self.label_counts + 1.0 / self.L) / (self.reviews + 1.0)
        trust = self.config.submitted_label_probability
        prior_norm2 = trust ** 2 + 2 * trust * (1 - trust) * pooled[self.row_labels] + (1 - trust) ** 2 * np.sum(pooled ** 2)
        count_prior_dot = trust * self.matches + (1 - trust) * (self._row_pooled_dot + n / self.L) / (self.reviews + 1.0)
        q_norm2 = a * a * self._row_count_norm2 + b * b * prior_norm2 + 2 * a * b * count_prior_dot
        return np.sqrt(np.clip(1.0 - q_norm2, 0.0, 1.0))

    def _allocation(self, q, fhat):
        return uncertainty_allocation(self.N_r - self.audits, self._label_sigma(), self.config.srs_floor)[0]

    def observe(self, record_id, verified_label):
        if self._pending is None:
            raise ValueError('Call prepare before observe')
        row = self._pending['row']
        old_global = int(self.label_counts[verified_label]) if 0 <= verified_label < self.L else 0
        result = super().observe(record_id, verified_label)
        counts = self._label_counts_by_row
        self._row_pooled_dot += counts[:, verified_label]
        self._row_pooled_dot[row] += old_global + 1
        self._row_count_norm2[row] += 2 * counts[row, verified_label] + 1
        counts[row, verified_label] += 1
        return result
