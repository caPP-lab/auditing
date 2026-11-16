"""Publication styling shared by the reported figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import HPacker, VPacker
from matplotlib.transforms import ScaledTranslation
from types import SimpleNamespace
import numpy as np
import pandas as pd
from .precision_attainment import target_count
PAS_COLOR='#C65300'
SRS_COLOR='#F2A65A'
UNCERTAINTY_COLOR='#B39B00'
LABELS=dict(kenya='Kenya',cifar10n='CIFAR-10N',reimagenet='ReImageNet',cleanconll='CleanCoNLL')
SHORT=dict(kenya='kenya',cifar10n='cifar',reimagenet='reimagenet',cleanconll='cleanconll')
def _style() -> None:
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIX Two Text', 'Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 9.0, 'axes.labelsize': 9.6, 'axes.linewidth': 0.65, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.labelsize': 8.2, 'ytick.labelsize': 8.2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.fontsize': 7.7, 'legend.frameon': False, 'lines.linewidth': 1.8, 'figure.dpi': 180, 'savefig.dpi': 400, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
NL_STYLE = dict(pas=(PAS_COLOR, '-', 'ATLAS-PAS'), srs=(SRS_COLOR, ':', 'ATLAS-SRS'), uncertainty_only=(UNCERTAINTY_COLOR, ':', 'Uncertainty-only'), ppr_projection=('#0072B2', '--', 'PPR + projection'), mr_convex=('#008C95', '-.', 'MR stitching'), mr_ratio=('#008C95', '-.', 'MR stitching'))

def legend(ax, handles, labels, **kwargs):
    defaults = dict(fontsize=10.5, handlelength=2.8, handletextpad=0.8, columnspacing=1.5, labelspacing=0.5, borderaxespad=0.55, borderpad=0.4, frameon=True, facecolor='white', edgecolor='none', framealpha=0.96)
    defaults.update(kwargs)
    return ax.legend(handles, labels, **defaults)

class CenteredLegendRow(HPacker):
    """Center text and samples in a shared, renderer-dependent row height."""

    def __init__(self, original, peers):
        super().__init__(pad=original.pad, sep=original.sep, align='center', children=original.get_children())
        self.peers = peers

    def _get_bbox_and_child_offsets(self, renderer):
        self.height = max((child.get_bbox(renderer).height for child in self.peers))
        return super()._get_bbox_and_child_offsets(renderer)

def nonlinear_legend(ax, data, *, order=None, linewidth=2.0, **kwargs):
    mr = 'mr_convex' if 'mr_convex' in set(data.policy_id) else 'mr_ratio'
    if order is None:
        order = (mr, 'ppr_projection', 'srs', 'uncertainty_only', 'pas')
    order = [p for p in order if p in set(data.policy_id)]
    handles = [Line2D([], [], color=NL_STYLE[p][0], linestyle=NL_STYLE[p][1], linewidth=linewidth, dash_capstyle='round') for p in order]
    return legend(ax, handles, [NL_STYLE[p][2] for p in order], **kwargs)

def axes_style(ax, show_y=True):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_visible(show_y)
    ax.tick_params(axis='y', which='both', left=show_y, labelleft=show_y)
    ax.tick_params(axis='both', labelsize=10.5)
    ax.grid(axis='y', linestyle=':', linewidth=0.65, color='#CBD1DB', alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel('Audits', fontsize=12)
    if not show_y:
        ax.set_ylabel('')

def narrow_panel_style(ax, *, kenya=False):
    """Typography shared by narrow standalone panels and the horizontal row."""
    dataset, score = ax.get_title().split(': ', 1)
    ax.set_title(dataset, fontsize=14, pad=4)
    ax.set_xlabel('Audits', fontsize=12, labelpad=3)
    ax.tick_params(labelsize=11.5, pad=3)
    ax.get_xticklabels()[0].set_ha('left')
    ax.get_xticklabels()[-1].set_ha('right')
    for annotation in ax.texts:
        annotation.set_fontsize(11)
    subtitle = ax.text(0.98, 0.965, score, transform=ax.transAxes, ha='right', va='top', fontsize=12, zorder=8)
    subtitle.set_gid('inside_score_title')
    if kenya:
        ax.set_xlim(0, 3000)
        ax.get_xticklabels()[-2].set_ha('right')
    ticks = ax.get_xticks()
    if len(ticks) > 2 and (ticks[-1] - ticks[-2]) / (ticks[-1] - ticks[0]) < 0.13:
        label = ax.get_xticklabels()[-2]
        label.set_transform(label.get_transform() + ScaledTranslation(-6 / 72, 0, ax.figure.dpi_scale_trans))
    if ax.get_legend() is not None:
        ax.get_legend().remove()

def compact_nonlinear_row(nl, fs, ft):
    """Tight horizontal row with pairwise y-axes and one bottom legend."""
    nl = nl.loc[nl.policy_id != 'uncertainty_only']
    fs = fs.loc[fs.policy_id != 'uncertainty_only']
    fig = plt.figure(figsize=(13.6, 3.7))
    fig._fixed_publication_canvas = True
    left_margin, right_margin = (0.56, 0.08)
    gaps = (0.12, 0.57, 0.12)
    width = (fig.get_figwidth() - left_margin - right_margin - sum(gaps)) / 4
    axes = []
    cursor = left_margin
    for index in range(4):
        ax = fig.add_axes([cursor / fig.get_figwidth(), 0.25, width / fig.get_figwidth(), 0.66], sharey=axes[-1] if index in (1, 3) else None)
        axes.append(ax)
        if index < 3:
            cursor += width + gaps[index]
    relative_panel(axes[0], nl, 0, show_legend=False)
    relative_panel(axes[1], nl, 1, show_y=False)
    f1_panel(axes[2], fs, ft, 'reimagenet')
    f1_panel(axes[3], fs, ft, 'cleanconll', show_y=False)
    for index, ax in enumerate(axes):
        narrow_panel_style(ax, kenya=ax is axes[0])
        bold_family = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif']
        ax.set_title(f"({'abcd'[index]}) {ax.get_title()}", fontweight='bold', fontfamily=bold_family, fontsize=14, pad=4)
        score_title = next((t for t in ax.texts if t.get_gid() == 'inside_score_title'))
        score_title.set_fontweight('bold')
        score_title.set_fontfamily(bold_family)
        ax.set_xlabel('')
        if index in (1, 3):
            for text in list(ax.texts):
                if 'target' in text.get_text():
                    text.remove()
        elif index == 2:
            target_label = next((t for t in ax.texts if 'target' in t.get_text()))
            target_label.xy = (0.98, 2)
            target_label.set_position((0, 4))
            target_label.set_ha('right')
            target_label.set_va('bottom')
            for text in ax.texts:
                if text.get_color() == NL_STYLE['ppr_projection'][0] and hasattr(text, 'xy'):
                    text.set_position((text.xy[0] - 900, 4.4))
        for policy in ('pas', 'srs'):
            if index < 2:
                crossing = _curve_target_intersection(nl, score_id=('pillar2_diversity', 'worst3_precision')[index], policy_id=policy, target=5, log_y=True)
            else:
                target_column = 'median_audits'
                crossing = ft.loc[(ft.dataset == ('reimagenet', 'cleanconll')[index - 2]) & (ft.policy_id == policy), target_column].iloc[0]
            if pd.notna(crossing) and ax.get_xlim()[0] <= crossing <= ax.get_xlim()[1]:
                guide, = ax.plot([crossing, crossing], [ax.get_ylim()[0], 5 if index < 2 else 2], color=NL_STYLE[policy][0], linestyle=':', linewidth=1.25, dash_capstyle='round', zorder=2.5)
                guide.set_gid(f'{policy}_crossing_guide')
    axes[0].set_ylabel('Relative margin of error (%)', fontsize=12, labelpad=1)
    axes[2].set_ylabel('F1 margin of error (pp)', fontsize=12, labelpad=1)
    fig.supxlabel('Audits', fontsize=13, y=0.115)
    nonlinear_legend(fig, nl[nl.score_id == 'pillar2_diversity'], order=('pas', 'srs', 'uncertainty_only', 'ppr_projection', 'mr_convex'), linewidth=3.0, loc='upper center', bbox_to_anchor=(0.5, 0.103), ncol=5, fontsize=16, handlelength=3.2, handletextpad=0.65, columnspacing=1.3, labelspacing=0.2, borderpad=0.1, borderaxespad=0, frameon=False)
    return (fig, axes)
CV_COLOR = '#009E73'

AUDIT_VIEW_LIMITS = {'reimagenet': 3000, 'cleanconll': 2000}

MARGIN_VIEW_MINIMA = {'kenya': 0.5, 'cifar10n': 0.5, 'reimagenet': 1.0, 'cleanconll': 0.0}

SINGLE_FIGSIZE = (5.9, 4.5)

PAIRED_FIGSIZE = (10.8, 5.1)

LEGEND_FONTSIZE = 10.0

TICK_FONTSIZE = 11.5

ANNOTATION_FONTSIZE = 11.5

def grouped_legend(ax, handles, labels, dataset):
    """Three rows: ATLAS, other anytime-valid methods, fixed-n methods."""
    compact = dataset == 'cleanconll'
    leg = legend(ax, handles, labels, ncol=7, loc='upper right', fontsize=LEGEND_FONTSIZE, handlelength=1.2 if compact else 1.9, handletextpad=0.4 if compact else 0.55, columnspacing=0.6 if compact else 0.95, labelspacing=0.15 if compact else 0.3, borderpad=0.1 if compact else 0.2, borderaxespad=0 if compact else 0.3, frameon=False)
    entries = [column.get_children()[0] for column in leg._legend_handle_box.get_children()]
    rows = []
    for group in (entries[:2], entries[2:5], entries[5:]):
        peers = [child for entry in group for child in entry.get_children()]
        centered = [CenteredLegendRow(entry, peers) for entry in group]
        rows.append(HPacker(pad=0, sep=leg.columnspacing * leg._fontsize, align='center', children=centered))
    body = VPacker(pad=0, sep=leg.labelspacing * leg._fontsize, align='right', children=rows)
    body.set_figure(ax.figure)
    leg._legend_box._children[1] = body
    leg._legend_handle_box = body
    return leg

def compact_matrix_text(ax, dataset):
    """Keep companion matrices readable at the narrower export width."""
    ax.set_title(ax.get_title(), fontsize=14, pad=22 if dataset == 'reimagenet' else 5)
    ax.set_xlabel(ax.get_xlabel(), fontsize=12, labelpad=4)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12, labelpad=3)
    ax.tick_params(labelsize=10, pad=3)
    for text in ax.texts:
        text.set_fontsize(text.get_fontsize() + 0.5)
    colorbar = ax.images[0].colorbar
    colorbar.ax.tick_params(labelsize=10, pad=3)
    colorbar.set_label('Records', fontsize=11, labelpad=4)

def style():
    _style()
    plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':12,
        'xtick.labelsize':10.5,'ytick.labelsize':10.5,'axes.labelpad':7,
        'axes.titlepad':10,'legend.fontsize':10.5,'lines.linewidth':2.,
        'figure.dpi':120,'savefig.dpi':300})

def relative_panel(*args,**kwargs):
    from .plot_panels import relative_panel as draw
    return draw(*args,**kwargs)

def f1_panel(*args,**kwargs):
    from .plot_panels import f1_panel as draw
    return draw(*args,**kwargs)

def _curve_target_intersection(summary,*,score_id,policy_id,target,log_y):
    return target_count(summary.loc[summary.score_id.eq(score_id)&summary.policy_id.eq(policy_id)],target)

COLORS={'ATLAS-PAS':PAS_COLOR,'ATLAS-SRS':SRS_COLOR,'RLFA':'#008C95','PPR':'#0072B2','Wald':'#444444','Serfling':'#9A9A9A'}
DOT_DASH=(0,(.1,1.8,3.2,1.8))
DOTTED=(0,(.1,1.9))
STYLES={'ATLAS-PAS':'-','ATLAS-SRS':':','RLFA':DOT_DASH,'PPR':DOTTED,'Wald':DOT_DASH,'Serfling':DOTTED}
ml=SimpleNamespace(COLORS=COLORS,STYLES=STYLES)
