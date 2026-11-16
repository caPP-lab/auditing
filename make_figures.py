"""Render the experimental figures in the camera-ready main text and appendix."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from atlas_auditing import plot_style as style
from atlas_auditing.plot_panels import mislabel_panel, taxonomic_risk_appendix
from atlas_auditing.plot_epsilon import aggregate_figure
from atlas_auditing.plot_initialization import build_audit_coverage_multiplot
from atlas_auditing import plot_synthetic

ROOT=Path(__file__).resolve().parent


def matrix_panel(ax,counts,dataset,stats=None):
    display=counts.to_numpy(int)
    cmap=matplotlib.colormaps['Blues'].copy();cmap.set_bad('white')
    im=ax.imshow(np.ma.masked_equal(display,0),cmap=cmap,
        norm=LogNorm(vmin=1,vmax=max(2,display.max())),aspect='equal',interpolation='nearest')
    if dataset=='kenya':
        ax.set(xticks=[],yticks=[],xlabel='Verified species',ylabel='Submitted species')
    else:
        ax.set_xticks(range(len(counts.columns)),counts.columns,rotation=60 if dataset=='reimagenet' else 45,ha='right',fontsize=9)
        ax.set_yticks(range(len(counts.index)),counts.index,fontsize=9)
        ax.set(xlabel='Reference class',ylabel='Submitted class')
    ax.tick_params(length=0,pad=3)
    ax.set_title('ReImageNet: feline subset' if dataset=='reimagenet' else style.LABELS[dataset],fontsize=13,pad=28 if dataset=='reimagenet' else 10)
    if dataset=='reimagenet':
        ax.text(.5,1.015,f'{stats["selected_classes"]} of {stats["observed_submitted_classes"]} classes; '
            f'{stats["internal_errors"]}/{stats["selected_errors"]} mislabels within subset',
            transform=ax.transAxes,ha='center',va='bottom',fontsize=8.5,color='#62666D')
        ax.axvline(len(counts.index)-.5,color='#777777',linewidth=.65)
    if dataset!='kenya':
        threshold=.56*np.log10(display.max()+1)
        for r,c in np.ndindex(display.shape):
            if display[r,c]:
                size=6 if dataset=='cleanconll' and display[r,c]>=100000 else 8
                ax.text(c,r,f'{display[r,c]:,}',ha='center',va='center',fontsize=size,
                    color='white' if np.log10(display[r,c]+1)>threshold else 'black')
    for spine in ax.spines.values(): spine.set_visible(True)
    cb=ax.figure.colorbar(im,ax=ax,fraction=.035,pad=.025)
    cb.ax.tick_params(labelsize=9);cb.set_label('Records',fontsize=10)


def synthetic_data(results):
    main=pd.read_csv(results/'synthetic/main.csv.gz')
    scale=pd.read_csv(results/'synthetic/scaling.csv.gz')
    widths=main.groupby(['policy_id','audit']).width.agg(median='median',q25=lambda v:v.quantile(.25),q75=lambda v:v.quantile(.75)).reset_index()
    terminal=scale.loc[scale.audit.eq(scale.audit.max())].pivot(index=['rows','seed'],columns='policy_id',values='width')
    terminal['reduction']=100*(1-terminal.pas/terminal.srs)
    scaling=terminal.groupby('rows').reduction.agg(median='median',q25=lambda v:v.quantile(.25),q75=lambda v:v.quantile(.75)).reset_index()
    scaling['theoretical_width_reduction_percent']=100*(1-np.sqrt(4/scaling.rows))
    return dict(artifact=json.loads((results/'synthetic/population.json').read_text()),widths=widths,scaling=scaling)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'figures')
    args=parser.parse_args();results,output=args.results,args.output
    output.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        bounds=None if getattr(fig,'_fixed_publication_canvas',False) else 'tight'
        fig.savefig(output/(name+'.pdf'),bbox_inches=bounds,pad_inches=.065,
            metadata={'Author':'','Creator':'ATLAS experiment figures','CreationDate':None,'ModDate':None})
        plt.close(fig);print(name,flush=True)
    style.style()
    linear=pd.read_csv(results/'linear_curves.csv.gz')
    for dataset in ('kenya','cifar10n','reimagenet','cleanconll'):
        counts=pd.read_csv(results/'confusion'/f'{dataset}.csv',index_col=0)
        stats=json.loads((results/'confusion/reimagenet.json').read_text()) if dataset=='reimagenet' else None
        if dataset=='kenya':
            fig,ax=plt.subplots(figsize=style.SINGLE_FIGSIZE)
            fig.subplots_adjust(left=.12,right=.98,bottom=.15,top=.9)
            mislabel_panel(ax,linear,dataset);save(fig,'mislabel_kenya')
            fig,ax=plt.subplots(figsize=(4.8,4.5))
            matrix_panel(ax,counts,dataset)
            ax.set_title('');save(fig,'kenya_confusion_matrix_panel')
        else:
            fig,axes=plt.subplots(1,2,figsize=style.PAIRED_FIGSIZE,gridspec_kw={'width_ratios':[1,1.6]})
            matrix_panel(axes[0],counts,dataset,stats);style.compact_matrix_text(axes[0],dataset)
            fig.subplots_adjust(left=.10 if dataset=='reimagenet' else .07,right=.99,
                bottom=.25 if dataset=='reimagenet' else .19,top=.88,wspace=.33)
            mislabel_panel(axes[1],linear,dataset)
            save(fig,f'mislabel_{style.SHORT[dataset]}_with_confusion')
    nonlinear=pd.read_csv(results/'nonlinear_curves.csv.gz');targets=pd.read_csv(results/'target_summary.csv')
    score_ids=dict(kenya_entropy='pillar2_diversity',cifar_worst3_precision='worst3_precision',
                   cifar_worst3_taxonomic_risk='worst3_taxonomic_risk')
    nl=nonlinear[nonlinear.setting.isin(score_ids)].copy()
    nl['score_id']=nl.setting.map(score_ids);nl['policy_id']=nl.policy.replace({'mr_stitching':'mr_convex'})
    fs=nonlinear[nonlinear.setting.str.endswith('_f1')].copy()
    fs['dataset']=fs.setting.str.removesuffix('_f1');fs['policy_id']=fs.policy.replace({'mr_stitching':'mr_ratio'})
    ft=targets[targets.setting.str.endswith('_f1')].copy()
    ft['dataset']=ft.setting.str.removesuffix('_f1');ft['policy_id']=ft.policy.replace({'mr_stitching':'mr_ratio'})
    fig,_=style.compact_nonlinear_row(nl,fs,ft);save(fig,'nonlinear_four_datasets_compact_row')
    fig,_=taxonomic_risk_appendix(nl);save(fig,'nonlinear_cifar_worst3_taxonomic_risk_appendix')
    plot_synthetic._style();save(plot_synthetic.composite(synthetic_data(results)),'synthetic')
    style._style()
    save(aggregate_figure(pd.read_csv(results/'sensitivity/epsilon_summary.csv')),'epsilon_all_datasets')
    save(build_audit_coverage_multiplot(pd.read_csv(results/'sensitivity/initialization_summary.csv')),
         'kenya_prior_sensitivity_audits_coverage_multiplot')


if __name__=='__main__':
    main()
