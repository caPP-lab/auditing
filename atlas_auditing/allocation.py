import numpy as np

def rlfa_probabilities(remaining, prediction, srs_floor):
    remaining = np.asarray(remaining, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if remaining.shape != prediction.shape or np.any(remaining < 0):
        raise ValueError('invalid remaining counts')
    if not 0 <= srs_floor <= 1 or np.any((prediction < 0) | (prediction > 1)):
        raise ValueError('invalid predictions or floor')
    if not np.isfinite(prediction).all() or remaining.sum() <= 0:
        raise ValueError('nonfinite predictions or exhausted population')
    uniform = remaining / remaining.sum()
    raw = remaining * prediction
    weighted = raw / raw.sum() if raw.sum() > 0 else uniform
    q = srs_floor * uniform + (1 - srs_floor) * weighted
    if np.any(q[remaining > 0] <= 0):
        raise ValueError('RLFA requires positive sampling support')
    return q / q.sum()

def chebyshev_half_width(n, population_size, alpha=0.045):
    n = np.asarray(n, dtype=float)
    if population_size < 1 or not 0 < alpha < 1:
        raise ValueError('invalid population or alpha')
    if np.any(~np.isfinite(n) | (n < 1) | (n > population_size) | (n != np.floor(n))):
        raise ValueError('n must be an integer in [1,N]')
    if population_size == 1:
        return np.zeros_like(n)
    return np.sqrt((population_size - n) / (4 * alpha * n * (population_size - 1)))

def chebyshev_target_n(population_size, half_width=0.02, alpha=0.045):
    if not 0 < half_width < 1:
        raise ValueError('invalid target')
    chebyshev_half_width(1, population_size, alpha)
    return int(np.ceil(population_size / (1 + 4 * alpha * half_width ** 2 * (population_size - 1))))
