from functools import lru_cache
from atlas_auditing.stitching import MRConfig, envelope_boundaries

@lru_cache(maxsize=16)
def f1_mean_boundary(N, config):
    return envelope_boundaries(N, 1, 2.0, config)[0]

class MRF1CS:

    def __init__(self, N, submitted_target_count, *, config=None):
        if int(N) != N or int(submitted_target_count) != submitted_target_count or (not 1 <= submitted_target_count <= N):
            raise ValueError('require integer 1 <= submitted target count <= N')
        self.N, self.A = (int(N), int(submitted_target_count))
        self.config = config or MRConfig(stitching='reference_closed_form')
        self.config.validate()
        self.radii = f1_mean_boundary(self.N, self.config)
        self.t = self.audited_a = self.audited_b = self.tp = 0
        self.lower, self.upper = (0.0, 1.0)

    def observe(self, submitted_in_target, verified_in_target, exact_match):
        a, b, match = (bool(submitted_in_target), bool(verified_in_target), bool(exact_match))
        if self.t >= self.N or (match and a != b) or self.audited_a + a > self.A or (self.t - self.audited_a + (not a) > self.N - self.A):
            raise ValueError('inconsistent observation or census already reached')
        self.t += 1
        self.audited_a += a
        self.audited_b += b
        self.tp += a and match
        denominator = self.A / self.N + self.audited_b / self.t
        numerator = 2 * self.tp / self.t
        radius = self.radii[self.t]
        fp, fn = (self.audited_a - self.tp, self.audited_b - self.tp)
        fp_max = fp + self.A - self.audited_a
        fn_max = fn + self.N - self.t
        logical_lo = 2 * (self.A - fp_max) / (2 * self.A - fp_max + fn_max)
        logical_hi = 2 * (self.A - fp) / (2 * self.A - fp + fn)
        if self.t == self.N:
            lo = hi = 2 * self.tp / (self.A + self.audited_b)
        else:
            lo = max(logical_lo, (numerator - radius) / denominator) - self.config.outward_pad
            hi = min(logical_hi, (numerator + radius) / denominator) + self.config.outward_pad
        self.lower, self.upper = (max(self.lower, lo), min(self.upper, hi))
        return dict(lower=self.lower, upper=self.upper, empty=self.lower > self.upper, plugin=numerator / denominator, residual_radius=float(radius))
