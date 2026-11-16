"""Reported exploration-weight sensitivity figure."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .experiments import EPSILONS
DATASETS = ('kenya', 'cifar10n', 'reimagenet', 'cleanconll')

LABELS = dict(kenya='Kenya', cifar10n='CIFAR-10N', reimagenet='ReImageNet', cleanconll='CleanCoNLL')

COLORS = dict(kenya='#D55E00', cifar10n='#ED9A45', reimagenet='#B52B35', cleanconll='#B58B00')

MARKERS = dict(kenya='o', cifar10n='s', reimagenet='D', cleanconll='^')

def axes_common(ax):
    ax.set(xlim=(0.025, 1.04), xticks=EPSILONS, xlabel='SRS mixing weight $\\epsilon$')
    ax.set_xticklabels(['0.05'] + [f'{value:.1f}' for value in EPSILONS[1:]])
    ax.grid(axis='y', color='#D7DBE0', linewidth=0.55, linestyle='--', dashes=(2, 3))
    ax.set_axisbelow(True)

def aggregate_figure(combined):
    fig, ax = plt.subplots(figsize=(7.1, 4.05))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.84)
    for dataset in DATASETS:
        data = combined.loc[combined.dataset == dataset]
        color = COLORS[dataset]
        x = data.epsilon.to_numpy()
        ax.fill_between(x, data.normalized_q25.to_numpy(), data.normalized_q75.to_numpy(), color=color, alpha=0.13, linewidth=0)
        ax.plot(x, data.normalized_median.to_numpy(), color=color, marker=MARKERS[dataset], linewidth=1.8, markersize=4.2, label=LABELS[dataset], zorder=3)
    for value in (0.3, 1.0):
        ax.axvline(value, color='#666666', linewidth=0.75, linestyle=':', zorder=0)
    ax.text(0.3, 1.015, 'Default', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=9)
    ax.text(1.0, 1.015, 'SRS', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=9)
    axes_common(ax)
    upper = max(1.05, float(combined.normalized_q75.max()) * 1.08)
    ax.set(ylim=(0, upper), ylabel='Normalized audits to reach 2 pp')
    ax.set_title('ATLAS-PAS sensitivity to $\\epsilon$ across four datasets', fontsize=12, pad=29)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor='#999999', alpha=0.2, edgecolor='none')]
    labels += ['IQR (100 paths)']
    ax.legend(handles, labels, loc='lower left', ncol=3, fontsize=8.4, columnspacing=1.3, handlelength=2.0, borderaxespad=0.8, frameon=True, facecolor='white', edgecolor='none', framealpha=1.0)
    return fig
