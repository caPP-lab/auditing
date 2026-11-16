import numpy as np
import pandas as pd

def censored_quantiles(values, probabilities=(0.25, 0.5, 0.75)):
    ordered = np.sort(np.asarray(values, dtype=float), axis=0)
    output = []
    for probability in probabilities:
        position = probability * (len(ordered) - 1)
        lo, hi = (int(np.floor(position)), int(np.ceil(position)))
        weight = position - lo
        a, b = (ordered[lo], ordered[hi])
        finite = np.isfinite(b)
        value = np.full_like(b, np.inf, dtype=float)
        np.add(a, weight * np.subtract(b, a, where=finite, out=np.zeros_like(b)), where=finite, out=value)
        output.append(value)
    return np.asarray(output)

def first_attainment(audits, widths, levels, *, tolerance=0.0):
    audits = np.asarray(audits, dtype=float)
    widths = np.asarray(widths, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if widths.ndim != 2 or widths.shape[1] != len(audits):
        raise ValueError('Expected path-by-checkpoint widths')
    if not len(audits) or np.any(np.diff(audits) <= 0):
        raise ValueError('Checkpoints must be strictly increasing')
    if np.any(widths[np.isfinite(widths)] < 0):
        raise ValueError('Empty intervals must be marked invalid, not negative')
    best = np.minimum.accumulate(np.where(np.isfinite(widths), widths, np.inf), axis=1)
    output = np.full((len(widths), len(levels)), np.inf)
    for i, row in enumerate(best):
        indices = np.searchsorted(-row, -(levels + tolerance), side='left')
        valid = indices < len(audits)
        output[i, valid] = audits[indices[valid]]
    return output

def precision_levels(widths, target, *, maximum, count=2400):
    finite = np.asarray(widths)[np.isfinite(widths) & (np.asarray(widths) > 0)]
    minimum = max(0.001, min(float(finite.min()) if len(finite) else target, target) * 0.95)
    levels = np.unique(np.r_[np.geomspace(minimum, maximum, count), np.linspace(minimum, maximum, count), float(target)])
    return levels[::-1]

def summarize_attainment(audits, widths, levels, *, tolerance=0.0):
    times = first_attainment(audits, widths, levels, tolerance=tolerance)
    quantiles = censored_quantiles(times)
    return pd.DataFrame(dict(precision=levels, q25_audits=quantiles[0], median_audits=quantiles[1], q75_audits=quantiles[2], reached=np.isfinite(times).sum(axis=0), paths=len(widths), budget=float(np.max(audits))))

def plot_attainment(ax, data, *, color, linestyle, label=None, linewidth=2.0, alpha=0.14):
    part = data.sort_values('precision', ascending=False)
    y = part.precision.to_numpy(float)
    x = part.median_audits.to_numpy(float)
    lo, hi = (part.q25_audits.to_numpy(float), part.q75_audits.to_numpy(float))
    budget = part.budget.to_numpy(float)
    ax.fill_betweenx(y, lo, np.minimum(hi, budget), where=np.isfinite(lo), color=color, alpha=alpha, linewidth=0)
    line, = ax.plot(np.where(np.isfinite(x), x, np.nan), y, color=color, linestyle=linestyle, linewidth=linewidth, label=label, dash_capstyle='round')
    line.set_gid('precision_attainment_' + str(label))
    return line

def target_count(data, target):
    rows = data.loc[data.precision.eq(target), 'median_audits']
    if len(rows) != 1:
        raise ValueError('Exactly one explicit target precision is required')
    return float(rows.iloc[0])
