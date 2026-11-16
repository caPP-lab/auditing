"""The two sensitivity figures presented in the appendix."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from .plot_style import _style,PAS_COLOR,SRS_COLOR
BUDGETS=dict(linear=5000,entropy=3000)
COLORS = dict(pas=PAS_COLOR, srs=SRS_COLOR)

LABELS = dict(pas='ATLAS-PAS', srs='ATLAS-SRS')

STYLES = dict(pas='-', srs=':')

TITLES = dict(linear='Kenya: mislabel rate', entropy='Kenya: Hill-Shannon diversity')

XLABELS = dict(linear='Initial correctness probability (%)', entropy='Initial uniform-mixture weight (%)')

def plot_quantiles(ax, data, field, *, scale=1.0, budget=None):
    for policy in ('pas', 'srs'):
        part = data[data.policy_id == policy].sort_values('percent')
        x = part.percent.to_numpy()
        med, lo, hi = [scale * part[f'{q}_{field}'].to_numpy() for q in ('median', 'q25', 'q75')]
        if budget is not None:
            censored = ~np.isfinite(med)
            ax.scatter(x[censored], np.full(censored.sum(), budget), marker='^', s=55, facecolors='none', edgecolors=COLORS[policy], zorder=5)
            lo = np.where(np.isfinite(lo), lo, budget)
            upper_censored = ~np.isfinite(hi)
            hi = np.where(upper_censored, budget, hi)
            if upper_censored.any():
                ax.scatter(x[upper_censored], hi[upper_censored], marker='|', s=50, color=COLORS[policy])
        ax.fill_between(x, lo, hi, color=COLORS[policy], alpha=0.16, linewidth=0)
        ax.plot(x, med, STYLES[policy], color=COLORS[policy], lw=2, marker='o' if policy == 'pas' else 's', ms=4.8, label=LABELS[policy], zorder=3)

def build_audit_coverage_multiplot(data):
    """Two audit-count panels above aligned, whisker-free coverage panels."""
    _style()
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10})
    fig = plt.figure(figsize=(12, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.08, 0.92), left=0.08, right=0.985, bottom=0.1, top=0.95, wspace=0.28, hspace=0.12)
    top = [fig.add_subplot(grid[0, 0])]
    top.append(fig.add_subplot(grid[0, 1], sharey=top[0]))
    bottom = [fig.add_subplot(grid[1, 0], sharex=top[0]), fig.add_subplot(grid[1, 1], sharex=top[1], sharey=None)]
    bottom[1].sharey(bottom[0])

    def decorate(ax, study, *, label_default=False, xlabel=False):
        ax.set_xlim(-3, 103)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.grid(axis='y', color='#D7DBE0', linewidth=0.65, linestyle=':')
        ax.set_axisbelow(True)
        current = 50 if study == 'linear' else 20
        ax.axvline(current, color='#777777', linestyle=':', lw=1, zorder=0)
        if label_default:
            ax.text(current + 1.5, 0.98, 'Current default', transform=ax.get_xaxis_transform(), fontsize=9, color='#555555', ha='left', va='top', rotation=90)
        if xlabel:
            ax.set_xlabel(XLABELS[study])
        else:
            ax.tick_params(axis='x', labelbottom=False)
    for col, study in enumerate(('linear', 'entropy')):
        part = data[data.study == study]
        ax = top[col]
        plot_quantiles(ax, part, 'audits', budget=BUDGETS[study])
        decorate(ax, study, label_default=True)
        ax.set_title(TITLES[study], pad=12)
        ax.set_ylabel('Audits to reach 2 pp\nmargin of error' if study == 'linear' else 'Audits to reach 5% relative\nmargin of error')
        ax.set_ylim(bottom=0)
        if part.median_audits.isna().any():
            ax.set_ylim(top=BUDGETS[study] * 1.15)
            ax.axhline(BUDGETS[study], color='#888888', ls='--', lw=0.7)
        ax = bottom[col]
        for policy in ('pas', 'srs'):
            sub = part[part.policy_id == policy].sort_values('percent')
            ax.plot(sub.percent, 100 * sub.pathwise_coverage, STYLES[policy], color=COLORS[policy], marker='o' if policy == 'pas' else 's', ms=4.8, lw=2, label=LABELS[policy], clip_on=False)
        decorate(ax, study, xlabel=True)
        ax.axhline(95.5, color='#555555', ls='--', lw=1)
        ax.set_ylim(90, 100)
    top[0].set_ylim(0, 3200)
    bottom[0].set_ylabel('Whole-path coverage (%)')
    bottom[1].tick_params(axis='y', labelleft=False)
    bottom[0].text(0.02, 0.05, 'Nominal: 95.5%', transform=bottom[0].transAxes, fontsize=9, color='#555555')
    handles, labels = top[0].get_legend_handles_labels()
    top[0].legend(handles[::-1], labels[::-1], loc='lower left', frameon=True, facecolor='white', edgecolor='none', framealpha=0.6)
    return fig
