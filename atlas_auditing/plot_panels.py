"""Final precision-attainment panels."""
import numpy as np
from .precision_attainment import plot_attainment,target_count
METHODS=('pas','srs','prop_ms','prop_m_cv','ppr','wald','serfling','chebyshev')
def number(value, budget, short=False):
    if not np.isfinite(value):
        return f'$>{int(budget / 1000)}$k' if short else f'$>${int(budget):,}'
    return f'{int(np.floor(value + 0.5)):,}'

def relative_panel(ax, summary, index, *, show_y=True, tax_log=False, title_prefix=True, show_legend=None):
    from . import plot_style as final
    from matplotlib.ticker import FuncFormatter, NullLocator
    score = ('pillar2_diversity', 'worst3_precision', 'worst3_taxonomic_risk')[index]
    titles = ('Pooled Hill-Shannon entropy', 'Worst-three precision', 'Worst-three taxonomic risk')
    data = summary.loc[summary.score_id.eq(score)]
    target, maximum = (25 if index == 2 else 5, 3000 if index == 0 else 10000)
    for policy in ('mr_convex', 'ppr_projection', 'srs', 'uncertainty_only', 'pas'):
        part = data.loc[data.policy_id.eq(policy)]
        if part.empty:
            continue
        color, line, label = final.NL_STYLE[policy]
        plot_attainment(ax, part, color=color, linestyle=line, label=label)
    crossings = {p: target_count(data.loc[data.policy_id.eq(p)], target) for p in ('pas', 'srs')}
    ticks = sorted(set([0, maximum] + [x for x in crossings.values() if np.isfinite(x) and x <= maximum]))
    ax.set_xticks(ticks, [number(x, maximum) for x in ticks])
    for policy in data.policy_id.unique():
        cross = target_count(data.loc[data.policy_id.eq(policy)], target)
        if not np.isfinite(cross) or cross > maximum:
            continue
        color = final.NL_STYLE[policy][0]
        ax.scatter(cross, target, color=color, edgecolor='white', s=35, linewidth=0.7, zorder=6)
        if policy not in ('pas', 'srs'):
            ax.annotate(number(cross, maximum), xy=(cross, target), xytext=(min(cross + 0.1 * maximum, 0.9 * maximum), target * (1.3 if index == 2 else 1.45)), ha='center', va='bottom', color=color, fontsize=10.5, arrowprops=dict(arrowstyle='-', color=color, linewidth=0.65, shrinkA=3, shrinkB=5))
    ax.set_xlim(0, maximum * 1.02)
    ax.set_yscale('log' if index != 2 or tax_log else 'linear')
    ax.set_ylim(10 if index == 2 else 2.5, 105)
    ax.set_yticks([10, 25, 50, 100] if index == 2 else [5, 10, 30, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.axhline(target, color='#7C818B', linewidth=0.9, linestyle='--', zorder=0)
    ax.annotate(f'$\\leq {target}\\%$ target', xy=(0.02 if index else 0.98, target), xycoords=('axes fraction', 'data'), xytext=(0, -5 if index else 5), textcoords='offset points', ha='left' if index else 'right', va='top' if index else 'bottom', fontsize=10.5, color='#62666D')
    prefix = ('Kenya: ' if index == 0 else 'CIFAR-10N: ') if title_prefix else ''
    ax.set_title(prefix + titles[index], fontsize=13, pad=10)
    if show_y:
        ax.set_ylabel('Relative margin of error (%)', fontsize=12)
    final.axes_style(ax, show_y)
    if show_legend is True or (show_legend is None and index == 0):
        anchor = 0.7 if index == 0 else 1.0 if index == 1 else 0.65 if tax_log else 0.46
        final.nonlinear_legend(ax, data, loc='upper right', bbox_to_anchor=(0.99, anchor))

def taxonomic_risk_appendix(summary):
    """Standalone, full-axis appendix view of the four primary CS methods."""
    from . import plot_style as final
    import matplotlib.pyplot as plt
    data = summary.loc[summary.score_id.eq('worst3_taxonomic_risk') & ~summary.policy_id.eq('uncertainty_only')].copy()
    assert set(data.policy_id) == {'pas', 'srs', 'ppr_projection', 'mr_convex'}
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.9)
    relative_panel(ax, data, 2, show_y=True, tax_log=False, show_legend=False)
    ax.set_xlim(0, 10000)
    ax.set_yticks([10, 25, 50, 75, 100])
    ax.set_title('CIFAR-10N: worst-three taxonomic risk', fontsize=16, weight='bold', fontfamily=['Times New Roman', 'STIXGeneral', 'DejaVu Serif'], pad=8)
    ax.set_xlabel('Audits', fontsize=14, labelpad=5)
    ax.set_ylabel('Relative margin of error (%)', fontsize=14, labelpad=5)
    ax.tick_params(labelsize=12, pad=4)
    for label in ax.texts:
        label.set_fontsize(12)
    for policy in ('pas', 'srs'):
        crossing = target_count(data.loc[data.policy_id.eq(policy)], 25.0)
        ax.plot([crossing, crossing], [ax.get_ylim()[0], 25.0], color=final.NL_STYLE[policy][0], linestyle=':', linewidth=1.4, zorder=2, gid=f'{policy}_crossing_guide')
    final.nonlinear_legend(ax, data, loc='upper right', bbox_to_anchor=(0.99, 0.52), fontsize=12, labelspacing=0.5, frameon=False, linewidth=2.4, handlelength=3.0)
    return (fig, ax)

def f1_panel(ax, summary, targets, dataset, *, show_y=True, show_legend=False):
    from . import plot_style as final
    from matplotlib.ticker import FuncFormatter, NullLocator
    data = summary.loc[summary.dataset.eq(dataset)]
    maximum = 3000 if dataset == 'cleanconll' else 10000
    ticks = [0, maximum]
    for policy in ('mr_ratio', 'ppr_projection', 'srs', 'uncertainty_only', 'pas'):
        part = data.loc[data.policy_id.eq(policy)]
        if part.empty:
            continue
        color, line, label = final.NL_STYLE[policy]
        plot_attainment(ax, part, color=color, linestyle=line, label=label)
        cross = target_count(part, 2.0)
        if not np.isfinite(cross) or cross > maximum:
            continue
        ax.scatter(cross, 2, color=color, edgecolor='white', s=35, linewidth=0.7, zorder=6)
        if policy in ('pas', 'srs'):
            ticks.append(cross)
        else:
            position = (2200, 1.35) if dataset == 'cleanconll' else (cross + 400, 4.4) if policy == 'ppr_projection' else (cross + 850, 3.5)
            ax.annotate(number(cross, maximum), xy=(cross, 2), xytext=position, ha='center', va='center', color=color, fontsize=10.5, arrowprops=dict(arrowstyle='-', color=color, linewidth=0.65, shrinkA=3, shrinkB=5))
    ticks = sorted(set(ticks))
    ax.set_xticks(ticks, [number(t, maximum) for t in ticks])
    ax.set(xlim=(0, maximum), yscale='log', ylim=(1, 52), yticks=[1, 2, 5, 10, 20, 50])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.axhline(2, color='#7C818B', linewidth=0.9, linestyle='--', zorder=0)
    ax.annotate('$\\leq 2$ pp target', xy=(0.04, 2), xycoords=('axes fraction', 'data'), xytext=(0, -13 if dataset == 'reimagenet' else -4), textcoords='offset points', ha='left', va='top', fontsize=10.5, color='#62666D')
    ax.set_title(f'{final.LABELS[dataset]}: F1 score', fontsize=13, pad=10)
    if show_y:
        ax.set_ylabel('F1 margin of error (percentage points)', fontsize=12)
    final.axes_style(ax, show_y)
    if dataset == 'cleanconll':
        ax.get_xticklabels()[-2].set_ha('right')
        ax.get_xticklabels()[-1].set_ha('right')
    if show_legend:
        final.nonlinear_legend(ax, data, loc='upper right', fontsize=10, labelspacing=0.32, borderpad=0.25, borderaxespad=0.3)

def mislabel_panel(ax, curves, dataset):
    from . import plot_style as final
    from . import plot_style as preview
    ml = final.ml
    labels = ['ATLAS-PAS', 'ATLAS-SRS', 'RLFA\n(Prop-MS)', 'RLFA\n(Prop-M + CV)', 'PPR', 'Wald', 'Serfling']
    methods = ('ATLAS-PAS', 'ATLAS-SRS', 'RLFA', 'CV', 'PPR', 'Wald', 'Serfling')
    colors = [ml.COLORS[m] if m != 'CV' else preview.CV_COLOR for m in methods]
    styles = [ml.STYLES[m] if m != 'CV' else '--' for m in methods]
    maximum = preview.AUDIT_VIEW_LIMITS.get(dataset, 5000)
    minimum = preview.MARGIN_VIEW_MINIMA.get(dataset, 0)
    positions = {'kenya': [(390, 2.22), (980, 2.14), (1700, 2.4), (2160, 2.4), (3850, 1.64), (1500, 2.55), (4450, 1.55)], 'cifar10n': [(480, 2.2), (900, 2.1), (1700, 2.4), (2700, 3.0), (3850, 1.64), (1400, 3.15), (4490, 0.58)], 'reimagenet': [(1070, 1.34), (1600, 1.58), (1900, 2.65), (2850, 2.72), (2350, 1.25), (550, 1.8), (4120, 1.58)], 'cleanconll': [(510, 2.2), (440, 2.65), (1460, 2.4), (250, 1.07), (710, 2.02), (150, 1.35), (2370, 2.2)]}[dataset]
    handles = []
    for policy, label, color, line, position in zip(METHODS[:-1], labels, colors, styles, positions):
        part = curves.loc[curves.setting.eq(dataset + '_mislabel') & curves.policy.eq(policy)]
        handles.append(plot_attainment(ax, part, color=color, linestyle=line, label=label, linewidth=1.8, alpha=0.085))
        cross = target_count(part, 2.0)
        if np.isfinite(cross) and cross <= maximum:
            ax.scatter(cross, 2, s=18, color=color, edgecolor='white', linewidth=0.45, zorder=5)
            ax.annotate(number(cross, 5000), xy=(cross, 2), xytext=position, ha='center', va='bottom', color=color, fontsize=preview.ANNOTATION_FONTSIZE, arrowprops=dict(arrowstyle='-', color=color, linewidth=0.55, shrinkA=2, shrinkB=5))
    ax.axhline(2, color='#767676', linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    ax.text(0.98 * maximum, 2.12 if dataset != 'cifar10n' else 2.45, '$\\leq 2$ pp target', ha='right', va='bottom', fontsize=preview.ANNOTATION_FONTSIZE, color='#767676')
    ax.set(xlim=(0, maximum), ylim=(minimum, 5), xticks=np.arange(0, maximum + 1, 500 if maximum < 5000 else 1000), yticks=sorted(set([minimum] + list(range(1, 6)))))
    ax.set_title(f'{final.LABELS[dataset]}: mislabel rate certification', fontsize=14, pad=5)
    ax.set_xlabel('Audits', fontsize=13, labelpad=4)
    ax.set_ylabel('Margin of error (percentage points)', fontsize=13, labelpad=3)
    final.axes_style(ax)
    ax.set_xlabel('Audits', fontsize=13, labelpad=4)
    ax.tick_params(labelsize=preview.TICK_FONTSIZE, pad=3)
    preview.grouped_legend(ax, handles, labels, dataset)
