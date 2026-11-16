from __future__ import annotations
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING
import math
import numpy as np
_PREC = 42
_DOWN = Context(prec=_PREC, rounding=ROUND_FLOOR)
_UP = Context(prec=_PREC, rounding=ROUND_CEILING)

def _D(x):
    return x if isinstance(x, Decimal) else Decimal(int(x)) if isinstance(x, (int, np.integer)) else Decimal.from_float(float(x))

class _I:
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi=None):
        self.lo, self.hi = (_D(lo), _D(lo if hi is None else hi))

    def __add__(self, v):
        v = v if isinstance(v, _I) else _I(v)
        return _I(_DOWN.add(self.lo, v.lo), _UP.add(self.hi, v.hi))
    __radd__ = __add__

    def __neg__(self):
        return _I(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, v):
        return self + -(v if isinstance(v, _I) else _I(v))

    def __rsub__(self, v):
        return (v if isinstance(v, _I) else _I(v)) + -self

    def __mul__(self, v):
        v = v if isinstance(v, _I) else _I(v)
        return _I(min((_DOWN.multiply(a, b) for a in (self.lo, self.hi) for b in (v.lo, v.hi))), max((_UP.multiply(a, b) for a in (self.lo, self.hi) for b in (v.lo, v.hi))))
    __rmul__ = __mul__

    def __truediv__(self, v):
        v = v if isinstance(v, _I) else _I(v)
        if v.lo <= 0 <= v.hi:
            raise ZeroDivisionError('interval contains zero')
        return self * _I(_DOWN.divide(Decimal(1), v.hi), _UP.divide(Decimal(1), v.lo))

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError('nonnegative integer power required')
        if self.lo < 0:
            raise ValueError('positive interval required')
        result = _I(1)
        base = self
        while n:
            if n & 1:
                result = result * base
            n //= 2
            if n:
                base = base * base
        return result

    def log(self):
        if self.lo <= 0:
            raise ValueError('log of nonpositive interval')
        a, b = (_DOWN.ln(self.lo), _UP.ln(self.hi))
        ea = Decimal(1).scaleb(a.adjusted() - _PREC + 1)
        eb = Decimal(1).scaleb(b.adjusted() - _PREC + 1)
        return _I(_DOWN.subtract(a, ea), _UP.add(b, eb))

def _fdown(x):
    r = float(x)
    if Decimal.from_float(r) > x:
        r = math.nextafter(r, -math.inf)
    return r

def _fup(x):
    r = float(x)
    if Decimal.from_float(r) < x:
        r = math.nextafter(r, math.inf)
    return r
