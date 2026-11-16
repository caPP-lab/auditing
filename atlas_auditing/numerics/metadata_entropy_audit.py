import numpy as np
from atlas_auditing.numerics.directional_observations import _point, _add, _sub, _mul, _div, _sum, _finite

def control_observation(state, r, s, pi, c_value, c_total, c_min, c_max):
    N = state.N
    A = state.A
    rem = state.N_r - A.sum(axis=1)
    active = rem > 0
    pred = (A + state.beta) / (A.sum(axis=1) + 1)[:, None]
    n = _point(float(N))
    m = _div(_point(rem.astype(float)), n)
    ratios = _div(m, _point(np.where(active, pi, 1.0)))
    ratios = (np.where(active, ratios[0], 0.0), np.where(active, ratios[1], 0.0))
    obs = A.sum(axis=0).astype(float)
    out = {}
    for name, d, b, lower_side in [('lo', state.dl, state.bl, True), ('hi', state.du, state.bu, False)]:
        baseline = pred @ d
        known = _div(_sum(_mul(_point(obs), _point(d))), n)
        center = _add(_add(_point(float(b)), known), _sum(_mul(m, _point(baseline))))
        center = _add(center, _div(c_total, n))
        delta = _sub(_sub(_point(float(d[s])), _point(float(baseline[r]))), _point(float(c_value)))
        value = _add(center, _mul((ratios[0][r], ratios[1][r]), delta))
        min_delta = _sub(_sub(_point(float(d.min())), _point(baseline)), _point(c_max))
        max_delta = _sub(_sub(_point(float(d.max())), _point(baseline)), _point(c_min))
        minimum = _add(center, _mul(ratios, min_delta))
        maximum = _add(center, _mul(ratios, max_delta))
        lower = float(np.min(minimum[0][active]))
        upper = float(np.max(maximum[1][active]))
        out['entropy_y' + name] = max(float(value[0]), lower) if lower_side else min(float(value[1]), upper)
        out['entropy_' + ('lower' if lower_side else 'upper')] = lower if lower_side else upper
    for v in out.values():
        _finite(v)
    return out
