from __future__ import annotations
import numpy as np

def _point(x):
    x = np.asarray(x, dtype=float)
    return (x, x)

def _down(x):
    return np.nextafter(x, -np.inf)

def _up(x):
    return np.nextafter(x, np.inf)

def _add(a, b):
    return (_down(a[0] + b[0]), _up(a[1] + b[1]))

def _neg(a):
    return (-a[1], -a[0])

def _sub(a, b):
    return _add(a, _neg(b))

def _mul(a, b):
    p = [a[i] * b[j] for i in range(2) for j in range(2)]
    return (_down(np.minimum.reduce(p)), _up(np.maximum.reduce(p)))

def _div(a, b):
    if np.any((b[0] <= 0) & (b[1] >= 0)):
        raise ZeroDivisionError('interval denominator contains zero')
    v = [a[i] / b[j] for i in range(2) for j in range(2)]
    return (_down(np.minimum.reduce(v)), _up(np.maximum.reduce(v)))

def _sum(a, axis=0):
    lo = np.moveaxis(np.asarray(a[0], float), axis, 0)
    hi = np.moveaxis(np.asarray(a[1], float), axis, 0)
    if len(lo) == 0:
        return _point(np.zeros(lo.shape[1:]))
    while len(lo) > 1:
        pairs = len(lo) // 2
        ll = _down(lo[:2 * pairs:2] + lo[1:2 * pairs:2])
        hh = _up(hi[:2 * pairs:2] + hi[1:2 * pairs:2])
        if len(lo) % 2:
            ll = np.concatenate([ll, lo[-1:]], axis=0)
            hh = np.concatenate([hh, hi[-1:]], axis=0)
        lo, hi = (ll, hh)
    return (lo[0], hi[0])

def _finite(*xs):
    if any((not np.isfinite(x).all() for x in xs)):
        raise ArithmeticError('nonfinite interval input or output')

def dyadic_probabilities(weights, active=None, bits=53):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not 1 <= bits <= 53:
        raise ValueError('invalid shape or precision')
    _finite(w)
    active = w > 0 if active is None else np.asarray(active, dtype=bool)
    if active.shape != w.shape or not active.any():
        raise ValueError('no active rows')
    if np.any(w < 0) or np.any(w[active] <= 0):
        raise ValueError('active weights must be positive')
    indices = np.flatnonzero(active)
    T = 1 << bits
    if len(indices) > T:
        raise ValueError('not enough positive bins')
    ratios = [float(w[r]).as_integer_ratio() for r in indices]
    power = max((den.bit_length() - 1 for _, den in ratios))
    numerators = [num << power - (den.bit_length() - 1) for num, den in ratios]
    denominator = sum(numerators)
    divisions = [divmod(T * num, denominator) for num in numerators]
    bins = [max(1, int(whole)) for whole, _ in divisions]
    missing = T - sum(bins)
    if missing > 0:
        order = sorted(range(len(bins)), key=lambda i: divisions[i][1], reverse=True)
        for i in order[:missing]:
            bins[i] += 1
    elif missing < 0:
        order = sorted(range(len(bins)), key=lambda i: divisions[i][1])
        remaining = -missing
        for i in order:
            take = min(remaining, bins[i] - 1)
            bins[i] -= take
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise ArithmeticError('could not balance positive bins')
    counts = np.zeros(len(w), dtype=np.int64)
    counts[indices] = bins
    if sum(map(int, counts)) != T or np.any(counts[active] < 1):
        raise ArithmeticError('invalid dyadic allocation')
    probabilities = counts.astype(float) / float(T)
    return (probabilities, counts)

def sample_dyadic_row(rng, weights, active=None, bits=53):
    probabilities, counts = dyadic_probabilities(weights, active, bits)
    draw = int(rng.integers(0, 1 << bits))
    row = int(np.searchsorted(np.cumsum(counts, dtype=np.int64), draw, side='right'))
    if counts[row] <= 0:
        raise ArithmeticError('selected inactive row')
    return (row, probabilities, counts)
