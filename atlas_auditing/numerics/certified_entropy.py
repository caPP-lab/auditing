from decimal import Decimal
from atlas_auditing.numerics.ppr_convex import _I, _DOWN, _UP, _PREC

def _exp(x):
    x = x if isinstance(x, _I) else _I(x)
    a, b = (_DOWN.exp(x.lo), _UP.exp(x.hi))
    ea = Decimal(1).scaleb(a.adjusted() - _PREC + 1)
    eb = Decimal(1).scaleb(b.adjusted() - _PREC + 1)
    return _I(max(Decimal(0), _DOWN.subtract(a, ea)), _UP.add(b, eb))
