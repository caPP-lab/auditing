from __future__ import annotations
import numpy as np
from atlas_auditing.numerics.ppr_convex import _I, _D, _fdown, _fup
from atlas_auditing.numerics.directional_observations import _point, _add, _sub, _mul, _div, _sum, _down, _up
from atlas_auditing.numerics.directional_numeric import _scaled_product_lower, _guard

def dyadic_simplex_point(p, N, observed, bits=53):
    if not isinstance(bits, int) or not 1 <= bits <= 53:
        raise ValueError('bits must be in 1..53')
    p = np.asarray(p, float)
    observed = np.asarray(observed, np.int64)
    T = 1 << bits
    base = np.array([(int(a) * T + N - 1) // N for a in observed], dtype=np.int64)
    if sum(map(int, base)) > T:
        raise ValueError('no dyadic feasible point; use exact census or general cube proposal')
    bins = np.maximum(base, np.floor(np.clip(p, 0, 1) * T).astype(np.int64))
    missing = T - sum(map(int, bins))
    if missing > 0:
        bins[int(np.argmax(p))] += missing
    elif missing < 0:
        for j in np.argsort(-(bins - base), kind='stable'):
            take = min(-missing, int(bins[j] - base[j]))
            bins[j] -= take
            missing += take
            if not missing:
                break
    if sum(map(int, bins)) != T or np.any(bins < base):
        raise ArithmeticError('dyadic simplex construction failed')
    return bins.astype(float) / float(T)

def parameters(h, N, L, D, ks, eta=0.875):
    c = (_I(N + L) / _I(N)).log()
    mi = _I(_D(h)) + c
    m = (_fdown(mi.lo), _fup(mi.hi))
    D = np.asarray(D, float)
    ks = np.asarray(ks, float)
    denominator = _add(m, _point(D))
    if np.any(denominator[0] <= 0):
        raise ValueError('positive cap denominator required')
    if not 0 < eta <= 1:
        raise ValueError('eta must lie in (0,1]')
    cap = _div(_point(float(eta)), denominator)
    lam = (np.minimum(ks[None, :], cap[0][:, None]), np.minimum(ks[None, :], cap[1][:, None]))
    if np.any(lam[0] < 0) or not np.isfinite(lam[1]).all():
        raise ValueError('invalid stakes')
    return (m, lam)

def _scaled_product_upper(factors, block_size=128):
    shape = factors.shape[1:]
    mant = np.ones(shape)
    exponent = np.zeros(shape, np.int64)
    for start in range(0, len(factors), block_size):
        block = factors[start:start + block_size]
        m, e = np.frexp(block)
        bm = _up(np.prod(m, axis=0, dtype=float) / _guard(len(block)))
        be = np.sum(e, axis=0, dtype=np.int64)
        bm, sh = np.frexp(bm)
        be += sh
        combined = _up(mant * bm)
        mant, sh = np.frexp(combined)
        exponent += be + sh
    mant, sh = np.frexp(mant)
    return (mant, exponent + sh)

def certify(X, K, D, N, p, observed, h, alpha=0.025, ks=(0.03, 0.1, 0.3, 1.0), eta=0.875):
    X = np.asarray(X, float)
    K = np.asarray(K, float)
    D = np.asarray(D, float)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 2:
        raise ValueError('nonempty time-by-category array required')
    n, L = X.shape
    if not np.isfinite(N) or N != int(N) or (not 0 < N <= 2 ** 53):
        raise ValueError('integer N in (0,2**53] required')
    N = int(N)
    raw = np.asarray(observed)
    if raw.shape != (L,) or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw != np.floor(raw)) or (sum(map(int, raw)) > N):
        raise ValueError('invalid observed counts')
    observed = raw.astype(np.int64)
    ks = np.asarray(ks, float)
    if ks.ndim != 1 or not len(ks) or (not np.isfinite(ks).all()) or np.any(ks < 0) or (not np.isfinite(h)) or (h < 0) or (not 0 < float(alpha) < 1):
        raise ValueError('invalid error level, threshold or betting magnitudes')
    if K.shape != (n,) or D.shape != (n,) or observed.shape != (L,):
        raise ValueError('shape mismatch')
    if np.any(X < 0) or not all((np.isfinite(v).all() for v in [X, K, D])):
        raise ValueError('invalid coefficients')
    p = np.asarray(p, float)
    if p.shape != (L,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('proposal outside convex cube')
    try:
        p = dyadic_simplex_point(p, N, observed)
        feasible_proposal = True
    except ValueError:
        feasible_proposal = False
    m, lam = parameters(float(h), N, L, D, ks, eta)
    d = []
    denominators = []
    for v in p:
        denom = _I(N) * _I(_D(v)) + _I(1)
        denominators.append(denom)
        d.append((_I(N + L) / denom).log())
    di = (np.array([_fdown(v.lo) for v in d]), np.array([_fup(v.hi) for v in d]))
    y = _add(_sum(_mul(_point(X), di), axis=1), _point(K))
    diff = _sub(y, m)
    factors = _add(_point(1.0), _mul(lam, (diff[0][:, None], diff[1][:, None])))
    if np.any(factors[0] <= 0):
        raise ArithmeticError('nonpositive certified factor')
    ml, el = _scaled_product_lower(factors[0])
    mu, eu = _scaled_product_upper(factors[1])
    E = int(max(eu))
    with np.errstate(under='ignore'):
        wl = np.maximum(0.0, _down(np.ldexp(ml, el - E)))
        wu = _up(np.ldexp(mu, eu - E))
    total = _sum((wl, wu))
    Slo, Shi = map(float, total)
    if Slo <= 0:
        raise ArithmeticError('zero lower scaled wealth')
    F = (_I(_D(Slo), _D(Shi)) / _I(len(ks))).log() + _I(E) * _I(2).log()
    weights = _div((wl, wu), _point(1.0))
    weights = (np.maximum(0.0, _down(wl / Shi)), _up(wu / Slo))
    inv = _div(lam, factors)
    sx = _sum(_mul((inv[0][:, :, None], inv[1][:, :, None]), _point(X[:, None, :])), axis=0)
    deriv = [-_I(N) / v for v in denominators]
    derivative = (np.array([_fdown(v.lo) for v in deriv]), np.array([_fup(v.hi) for v in deriv]))
    G = _mul(sx, (derivative[0][None, :], derivative[1][None, :]))
    gradient = _sum(_mul((weights[0][:, None], weights[1][:, None]), G), axis=0)
    delta = _sub(_div(_point(observed.astype(float)), _point(float(N))), _point(p))
    linear = _sum(_mul(gradient, delta))
    remain = _I(N - int(observed.sum())) / _I(N)
    lower = _I(F.lo) + _I(_D(float(linear[0]))) + remain * _I(_D(float(min(gradient[0]))))
    alpha_effective = float(np.nextafter(float(alpha), 0.0))
    if alpha_effective <= 0:
        raise ValueError('error level too small for supported numeric budget')
    threshold = (_I(1) / _I(_D(alpha_effective))).log()
    return {'certified': bool(lower.lo > threshold.hi), 'log_profile_lower': _fdown(lower.lo), 'log_profile_lower_decimal': str(lower.lo), 'log_value_lower': _fdown(F.lo), 'log_value_upper': _fup(F.hi), 'log_threshold_upper': _fup(threshold.hi), 'mass_shift_interval': list(m), 'gradient_max_width': float(np.max(gradient[1] - gradient[0])), 'effective_alpha': alpha_effective, 'requested_alpha': float(alpha), 'feasible_proposal': feasible_proposal, 'certified_global_gap_upper': _fup((_I(F.hi) - _I(lower.lo)).hi) if feasible_proposal else None, 'min_factor_lower': float(np.min(factors[0])), 'proposal': p.tolist(), 'numerical_contract': 'outward F, gradient, rational-simplex supporting-plane certificate'}
