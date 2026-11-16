import numpy as np
from atlas_auditing.numerics.directional_observations import _point, _add, _sub, _mul, _div, _sum, _finite

def positive_observation(state, r, s, pi, c_i, c_total, c_max):
    N = state.N
    A = state.A
    u = A.sum(1)
    M = state.N_r - u
    active = M > 0
    Q = (A + state.beta) / (u + 1)[:, None]
    base = Q @ state.du
    nn = _point(float(N))
    mm = _point(M.astype(float))
    ratio = _div(_div(mm, nn), _point(np.where(active, pi, 1.0)))
    ratio = (np.where(active, ratio[0], 0.0), np.where(active, ratio[1], 0.0))
    rr = (ratio[0][r], ratio[1][r])
    one = np.zeros(state.L)
    one[s] = 1.0
    X = _add(_div(_point(A.sum(0).astype(float)), nn), _mul(rr, _point(one)))
    S = _add(_sum(_mul(mm, _point(base))), c_total)
    K = _sub(_div(S, nn), _mul(rr, _add(_point(float(base[r])), _point(float(c_i)))))
    maxb = _add(_point(base), _point(c_max))
    minK = _sub(_div(S, nn), _mul(ratio, maxb))
    mass = _add(_div(_point(float(A.sum())), nn), ratio)
    D = float(np.max(_sub(mass, minK)[1][active]))
    xl = np.maximum(0.0, X[0])
    xu = X[1]
    kl = float(K[0])
    ku = float(K[1])
    _finite(xl, xu, kl, ku, D)
    if np.sum(xl) - kl > D + 1e-10:
        raise ArithmeticError('Realized profile exceeds predictable cap')
    return (xl, xu, kl, ku, D)
