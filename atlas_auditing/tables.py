"""LaTeX tables for the reported endpoints and robustness studies."""
from pathlib import Path
import numpy as np
import pandas as pd
from .reporting import DATASETS, SETTINGS, METHODS, NL_METHODS

LABELS=dict(pas='ATLAS-PAS',srs='ATLAS-SRS',prop_ms='Prop-MS',prop_m_cv='Prop-M + CV',
    ppr='PPR',wald='Wald',serfling='Serfling',chebyshev='Chebyshev',
    uncertainty_only='Uncertainty-only',ppr_projection='PPR + projection',mr_stitching='MR stitching')


def number(value,budget=5000):
    return rf'$>{budget:,}$' if not np.isfinite(value) else f'{int(np.floor(value+.5)):,}'


def cell(row,bold=False):
    mid=number(row.median_audits,int(row.budget))
    if bold: mid=rf'\textbf{{{mid}}}'
    return rf'\auditresult{{{mid}}}{{{number(row.q25_audits,int(row.budget))}}}{{{number(row.q75_audits,int(row.budget))}}}'


def sensitivity_cell(row):
    def precise(value):
        return r'$>5{,}000$' if not np.isfinite(value) else f'{value:,.1f}'.rstrip('0').rstrip('.')
    return rf'\auditresult{{{precise(row.median_audits)}}}{{{precise(row.q25_audits)}}}{{{precise(row.q75_audits)}}}'


def table(path,caption,headers,rows,footer=''):
    text=r'''% Requires booktabs and graphicx.
\providecommand{\auditresult}[3]{\shortstack[r]{#1\\[-2pt]{\scriptsize[#2, #3]}}}
\begin{table}[t]
\centering\small
'''+r'\caption{'+caption+'}\n'+r'\resizebox{\linewidth}{!}{%'+'\n'
    text+=r'\begin{tabular}{@{}l'+'c'*(len(headers)-1)+r'@{}}'+'\n'+r'\toprule'+'\n'
    text+=' & '.join(r'\textbf{'+h+'}' for h in headers)+r'\\'+'\n'+r'\midrule'+'\n'
    text+='\n'.join(' & '.join(map(str,row))+r'\\[3pt]' for row in rows)
    text+='\n'+r'\bottomrule'+'\n'+r'\end{tabular}}'+'\n'
    if footer: text+=r'\par\smallskip{\footnotesize '+footer+r'\par}'+'\n'
    text+=r'\end{table}'+'\n'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)


def write_tables(targets,output):
    output=Path(output);dest=output/'tables'
    caption=(r'Median [25th, 75th percentiles] audits to first attain the precision target '
             r'at 95.5\% nominal confidence over 100 paired paths. Counts are rounded to the nearest audit. ')
    rows=[]
    for policy in METHODS:
        row=[LABELS[policy]]
        for dataset in DATASETS:
            group=targets[targets.setting.eq(dataset+'_mislabel')]
            part=group[group.policy.eq(policy)].iloc[0]
            best=group[group.policy.isin(METHODS[:5])].median_audits.min()
            row.append(cell(part,part.median_audits==best) if policy in METHODS[:5] else number(part.median_audits))
        rows.append(row)
    table(dest/'mislabel_audit_table.tex',caption+r'Target: 2 percentage points; checks begin at audit 100. $>5{,}000$ denotes nonattainment within budget.',
        ['Method','Kenya','CIFAR-10N','ReImageNet','CleanCoNLL'],rows,
        r'Bold: smallest median among anytime-valid methods. Wald uses oracle variance; Wald, Serfling, and Chebyshev are fixed-$n$ references. Chebyshev uses variance bound $1/4$.')
    rows=[]
    for policy in NL_METHODS:
        row=[LABELS[policy]]
        for setting in SETTINGS:
            group=targets[targets.setting.eq(setting)]
            part=group[group.policy.eq(policy)].iloc[0]
            row.append(cell(part,part.median_audits==group.median_audits.min()))
        rows.append(row)
    headers=['Method',r'\shortstack{Kenya\\Hill--Shannon}',r'\shortstack{CIFAR-10N\\Worst-three precision}',
             r'\shortstack{CIFAR-10N\\Worst-three taxonomic risk}',r'\shortstack{ReImageNet\\F1}',r'\shortstack{CleanCoNLL\\NER F1}']
    table(dest/'nonlinear_audit_table.tex',caption+r'Checks occur every 25 audits. Targets are 5\% relative half-width for diversity and precision, 25\% for taxonomic risk, and 2 percentage points for F1. $>B$ denotes nonattainment: $B=3{,}000$ for Kenya ATLAS and uncertainty-only, and $10{,}000$ otherwise.',headers,rows)
    coverage=pd.read_csv(output/'nonlinear_coverage.csv')
    rows=[]
    for policy in NL_METHODS:
        row=[LABELS[policy]]
        for setting in SETTINGS:
            r=coverage[coverage.setting.eq(setting)&coverage.policy.eq(policy)].iloc[0]
            row.append(f'{r.covered_paths}/{r.paths}')
        rows.append(row)
    table(dest/'nonlinear_coverage_table.tex','Paths whose intervals cover the true nonlinear score at every reported update.',headers,rows)
    bayes=pd.read_csv(output/'bayesian_summary.csv').set_index('policy')
    rows=[]
    for p in ('pas','srs'):
        r=bayes.loc[p]
        rows.append([f'Bayesian + {p.upper()}',number(r['mean']),
            rf'\auditresult{{{number(r["median"])}}}{{{number(r.q25)}}}{{{number(r.q75)}}}',
            f'{int(r.coverage_at_stopping)}/100',f'{int(r.whole_path_coverage)}/100'])
    table(dest/'bayesian_table.tex',r'Kenya Hill--Shannon diversity: audits to 5\% relative accuracy at 95.5\% nominal credibility over 100 paths. Brackets give the 25th and 75th percentiles. Whole-path coverage is evaluated through stopping.',
        ['Sampling strategy','Mean','Median','Coverage at stopping','Whole-path coverage'],rows)
    stress=pd.read_csv(output/'sensitivity/misspecification_summary.csv')
    # Identifiers are defined once, alongside the predictor transformations.
    from .predictor_stress import SCENARIOS
    rows=[]
    for scenario in SCENARIOS:
        group=stress[stress.scenario.eq(scenario.name)].set_index('policy')
        a,b=group.loc['pas'].copy(),group.loc['srs'].copy()
        a['budget']=b['budget']=5000
        saving=100*(1-a.median_audits/b.median_audits)
        rows.append([scenario.label,sensitivity_cell(a),sensitivity_cell(b),f'{saving:.1f}'+r'\%',
            f'{int(a.reached)}/{int(b.reached)}',f'{int(round(a.pathwise_coverage*100))}/{int(round(b.pathwise_coverage*100))}'])
    table(dest/'misspecification_table.tex',r'Kenya mislabel-rate sensitivity over 100 paired paths. Entries are median [25th, 75th percentiles] audits to 2 percentage points. Savings compare PAS and SRS medians. Reached and whole-path coverage are path counts for PAS/SRS through audit 5,000.',
          ['Predictor','PAS','SRS','PAS savings','Reached',r'\shortstack{Whole-path\\coverage}'],rows)
