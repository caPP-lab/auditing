from dataclasses import dataclass
from decimal import Decimal, Context, localcontext
import numpy as np
_EPS = np.float64(2.0 ** (-52))
_INF = np.inf
_MAX = np.finfo(np.float64).max

def _down(x):
    return np.nextafter(x, -_INF)

def _up(x):
    return np.nextafter(x, _INF)

def _positive_down(x):
    return np.maximum(0.0, np.nextafter(x, 0.0))

@dataclass
class ScaledLower:
    mantissa: np.ndarray
    exponent: np.ndarray

    def to_float(self):
        with np.errstate(over='ignore', under='ignore'):
            x = np.ldexp(self.mantissa, self.exponent)
        x = np.where(np.isposinf(x), _MAX, _positive_down(x))
        return float(x) if np.ndim(x) == 0 else x

    def exceeds(self, threshold):
        t = np.asarray(threshold, float)
        if np.any(t < 0) or not np.isfinite(t).all():
            raise ValueError('finite nonnegative threshold required')
        mt, et = np.frexp(t)
        result = (self.mantissa > 0) & ((self.exponent > et) | (self.exponent == et) & (self.mantissa > mt))
        result = np.where(t == 0, self.mantissa > 0, result)
        return bool(result) if np.ndim(result) == 0 else result

def _guard(n):
    return np.nextafter(np.float64(1) - np.float64(n) * _EPS, 0.0)

def _scaled_product_lower(factors, block_size=128):
    shape = factors.shape[1:]
    mantissa = np.ones(shape, float)
    exponent = np.zeros(shape, np.int64)
    for start in range(0, len(factors), block_size):
        block = factors[start:start + block_size]
        m, e = np.frexp(block)
        bm = _positive_down(np.prod(m, axis=0, dtype=np.float64) * _guard(len(block)))
        be = np.sum(e, axis=0, dtype=np.int64)
        bm, shift = np.frexp(bm)
        be += shift
        combined = _positive_down(mantissa * bm)
        mantissa, shift = np.frexp(combined)
        exponent += be + shift
    mantissa, shift = np.frexp(mantissa)
    exponent += shift
    return (mantissa, exponent)

def _scaled_sum_lower(mantissas, exponents):
    m = np.asarray(mantissas)
    e = np.asarray(exponents, np.int64)
    emax = np.max(e, axis=0)
    with np.errstate(under='ignore'):
        aligned = _positive_down(np.ldexp(m, e - emax))
    total = _positive_down(np.sum(aligned, axis=0, dtype=np.float64) * _guard(len(m)))
    total, shift = np.frexp(total)
    return ScaledLower(total, emax + shift)

def wealth_lower(h, Ylo_bounds, Yupper_bounds, minlo, maxhi, ks, starts=(0,), weights=(1.0,), side='lo', *, denominator_floor=1e-15, block_size=128, return_scaled=False):
    if side not in ('lo', 'hi'):
        raise ValueError("side must be 'lo' or 'hi'")
    if not isinstance(block_size, int) or not 1 <= block_size <= 512:
        raise ValueError('block_size must be in1..512')
    ks = np.asarray(ks, np.float64)
    if ks.ndim != 1 or len(ks) == 0 or np.any(ks < 0) or (not np.isfinite(ks).all()):
        raise ValueError('invalid bets')
    starts = np.asarray(starts, np.int64)
    weights = np.asarray(weights, np.float64)
    if starts.ndim != 1 or weights.shape != starts.shape or (not len(starts)) or np.any(starts < 0):
        raise ValueError('invalid restarts')
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError('invalid mixture weights')
    with localcontext(Context(prec=1600)):
        total_weight = sum((Decimal.from_float(float(w)) for w in weights), Decimal(0))
    if total_weight > 1:
        raise ValueError('represented restart weights sum to more than one')
    if not np.isfinite(denominator_floor) or denominator_floor <= 0:
        raise ValueError('positive finite denominator floor required')
    y = np.asarray(Ylo_bounds if side == 'lo' else Yupper_bounds, np.float64)
    b = np.asarray(minlo if side == 'lo' else maxhi, np.float64)
    h = np.asarray(h, np.float64)
    if y.ndim not in (1, 2):
        raise ValueError('observations need time axis and optional coordinate axis')
    b = np.broadcast_to(b, y.shape)
    for v in (y, b, h):
        if not np.isfinite(v).all() or np.any(np.abs(v) > 1e+200):
            raise ValueError('input magnitude exceeds supported finite range1e200')
    if denominator_floor < 1e-100:
        raise ValueError('denominator_floor below supported1e-100')
    if side == 'lo':
        y = np.maximum(y, b)
        delta = _down(y - h)
        dlo = _down(h - b)
        dhi = _up(h - b)
    else:
        y = np.minimum(y, b)
        delta = _down(h - y)
        dlo = _down(b - h)
        dhi = _up(b - h)
    dlo = np.maximum(dlo, denominator_floor)
    dhi = np.maximum(dhi, denominator_floor)
    c_lo = np.nextafter(np.float64(9) / np.float64(10), 0.0)
    c_hi = np.nextafter(np.float64(9) / np.float64(10), _INF)
    caplo = _positive_down(c_lo / dhi)
    caphi = _up(c_hi / dlo)
    kshape = (1, len(ks)) + (1,) * (y.ndim - 1)
    lamlo = np.minimum(ks.reshape(kshape), caplo[:, None, ...])
    lamhi = np.minimum(ks.reshape(kshape), caphi[:, None, ...])
    dl = delta[:, None, ...]
    chosen = np.where(dl < 0, lamhi, lamlo)
    factor = _down(1.0 + _down(chosen * dl))
    factor = np.maximum(factor, np.nextafter(np.float64(1) / np.float64(10), 0.0))
    if not np.isfinite(factor).all():
        raise ArithmeticError('factor overflow: reduce input magnitude or bet scales')
    ms = []
    es = []
    for start, weight in zip(starts, weights):
        if weight == 0:
            continue
        m, e = _scaled_product_lower(factor[min(int(start), len(factor)):], block_size)
        wm, we = np.frexp(weight)
        m = _positive_down(m * wm)
        m = _positive_down(m / len(ks))
        m, shift = np.frexp(m)
        e = e + we + shift
        ms.extend(m)
        es.extend(e)
    if ms:
        result = _scaled_sum_lower(ms, es)
    else:
        shape = np.broadcast_shapes(y.shape[1:], h.shape)
        result = ScaledLower(np.zeros(shape), np.zeros(shape, np.int64))
    return result if return_scaled else result.to_float()
