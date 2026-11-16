"""Aggregate observed checkpoints without interpolating within an audit path."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .precision_attainment import first_attainment, precision_levels, summarize_attainment
from .allocation import chebyshev_target_n

DATASETS = ('kenya', 'cifar10n', 'reimagenet', 'cleanconll')
POPULATIONS = dict(zip(DATASETS, (33403, 50000, 30908, 301353)))
SETTINGS = ('kenya_entropy', 'cifar_worst3_precision', 'cifar_worst3_taxonomic_risk',
            'reimagenet_f1', 'cleanconll_f1')
METHODS = ('pas', 'srs', 'prop_ms', 'prop_m_cv', 'ppr', 'wald', 'serfling', 'chebyshev')
NL_METHODS = ('pas', 'srs', 'uncertainty_only', 'ppr_projection', 'mr_stitching')


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={'method':'gzip', 'mtime':0} if path.suffix=='.gz' else None)


def quantiles(values):
    """Linear order-statistic quantiles, retaining right-censored paths."""
    x = np.sort(np.asarray(values, float))
    result = []
    for q in (.25, .5, .75):
        position = (len(x)-1)*q
        lo, hi = int(np.floor(position)), int(np.ceil(position))
        result.append(x[lo] if lo==hi or x[lo]==x[hi] else
                      np.inf if not np.isfinite(x[hi]) else x[lo]+(position-lo)*(x[hi]-x[lo]))
    return result


def summarize_paths(frame, keys):
    rows = []
    for group, part in frame.groupby(keys, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        values = part.first_target_audit.fillna(np.inf).to_numpy()
        q25, median, q75 = quantiles(values)
        row = dict(zip(keys, group), paths=len(part), reached=int(np.isfinite(values).sum()),
                   q25_audits=q25, median_audits=median, q75_audits=q75)
        coverage = next((name for name in ('always_covered','whole_path_covered') if name in part), None)
        if coverage:
            row['pathwise_coverage'] = float(part[coverage].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def report(source, output):
    source, output = Path(source), Path(output)
    curves, targets, paths = [], [], []

    def add(setting, policy, audits, widths, seeds, target, maximum, tolerance):
        result = summarize_attainment(audits, widths,
            precision_levels(widths, target, maximum=maximum), tolerance=tolerance)
        result['setting'], result['policy'] = setting, policy
        result['target'], result['is_target'] = target, result.precision.eq(target)
        curves.append(result)
        targets.append(result.loc[result.is_target].iloc[0].to_dict())
        stops = first_attainment(audits, widths, [target], tolerance=tolerance)[:,0]
        paths.extend(dict(setting=setting, policy=policy, seed=int(seed), target=target,
                          first_target_audit=stop, budget=int(audits[-1])) for seed,stop in zip(seeds,stops))

    for dataset in DATASETS:
        for policy in METHODS[:-1]:
            with np.load(source/'mislabel'/dataset/f'{policy}.npz') as data:
                keep = data['audit']>=100
                widths = 100*data['half_width'][:,keep]
                widths[data['empty'][:,keep] | (data['lower'][:,keep]>data['upper'][:,keep])] = np.inf
                seeds = data['seed']
                add(dataset+'_mislabel',policy,data['audit'][keep],widths,seeds,2.,50.,1e-10)
        count = chebyshev_target_n(POPULATIONS[dataset],half_width=.02,alpha=.045)
        targets.append(dict(setting=dataset+'_mislabel',policy='chebyshev',target=2.,precision=2.,
            median_audits=count,q25_audits=count,q75_audits=count,reached=len(seeds),
            paths=len(seeds),budget=POPULATIONS[dataset],is_target=True,analytic=True))
        paths.extend(dict(setting=dataset+'_mislabel',policy='chebyshev',seed=int(seed),target=2.,
                          first_target_audit=count,budget=POPULATIONS[dataset]) for seed in seeds)
    write_csv(pd.concat(curves,ignore_index=True),output/'linear_curves.csv.gz')
    curves = []
    nonlinear = pd.read_csv(source/'nonlinear/checkpoints.csv.gz')
    for (setting,policy), part in nonlinear.groupby(['setting','policy'],sort=False):
        part = part.copy()
        relative = not setting.endswith('_f1')
        part['width'] = (100*(part.upper-part.lower)/np.maximum(part.upper+part.lower,1e-300)
                         if relative else 50*(part.upper-part.lower))
        part.loc[part['empty'] | (part.lower>part.upper),'width'] = np.inf
        pivot = part.pivot(index='seed',columns='audit',values='width').sort_index(axis=1)
        if not relative and 0 not in pivot.columns:
            # The pre-audit F1 interval is [0,1], hence 50 percentage points.
            pivot.insert(0,0,50.)
        assert not pivot.isna().any().any(), 'Incomplete checkpoint grid'
        add(setting,policy,pivot.columns.to_numpy(),pivot.to_numpy(),pivot.index.to_numpy(),
            25. if setting.endswith('taxonomic_risk') else 5. if relative else 2.,
            100. if relative else 50.,1e-12)
    write_csv(pd.concat(curves,ignore_index=True),output/'nonlinear_curves.csv.gz')
    write_csv(pd.DataFrame(targets),output/'target_summary.csv')
    write_csv(pd.DataFrame(paths),output/'per_path_target_times.csv.gz')
    coverage = nonlinear.groupby(['setting','policy','seed']).truth_covered.all().groupby(['setting','policy']).agg(['sum','count']).reset_index()
    write_csv(coverage.rename(columns={'sum':'covered_paths','count':'paths'}),output/'nonlinear_coverage.csv')
    for name,keys in [('epsilon',['dataset','epsilon']),('initialization',['study','percent','policy_id']),
                      ('misspecification',['scenario','policy'])]:
        frame = pd.read_csv(source/'sensitivity'/f'{name}_paths.csv')
        summary = summarize_paths(frame,keys)
        if name=='epsilon':
            denominator = summary.groupby('dataset').median_audits.transform('max')
            for q in ('q25','median','q75'):
                summary['normalized_'+q] = summary[q+'_audits']/denominator
        write_csv(summary,output/'sensitivity'/f'{name}_summary.csv')
    records = [json.loads(line) for line in (source/'bayesian/paths.jsonl').read_text().splitlines()]
    bayes = pd.DataFrame(records)
    bayes_rows=[]
    for policy,part in bayes.groupby('policy'):
        q25,median,q75=quantiles(part.t_stop)
        bayes_rows.append(dict(policy=policy,paths=len(part),mean=part.t_stop.mean(),median=median,
            q25=q25,q75=q75,coverage_at_stopping=int(part.cover_stop.sum()),
            whole_path_coverage=int(part.cover_path.sum())))
    write_csv(pd.DataFrame(bayes_rows),output/'bayesian_summary.csv')
    return pd.DataFrame(targets)


def collect_runs(runs, destination):
    """Collect the runner's per-path files into the same schema as saved results."""
    from run_experiments import tasks, STUDIES
    runs, destination = Path(runs), Path(destination)
    linear, nonlinear, bayes, synthetic = {}, [], [], []
    sensitivity = {name:[] for name in ('epsilon','initialization','misspecification')}
    for study in STUDIES:
        for task in tasks(study,DATASETS):
            policy,seed=task['policy'],task['seed']
            parts=[study]+[str(task[k]) for k in ('dataset','score','rows','epsilon','percent','scenario') if k in task]
            path=runs.joinpath(*parts,policy,f'{seed}.json' if study=='bayesian' else f'{seed}.npz')
            if study=='mislabel' and policy=='chebyshev':
                continue  # Fixed-n count is analytic.
            if not path.exists():
                raise FileNotFoundError(f'Complete the frozen run before collecting: {path}')
            if study=='bayesian':
                bayes.append(json.loads(path.read_text())); continue
            with np.load(path) as saved:
                data={k:saved[k] for k in saved.files}
            audits,lo,hi,empty=data['audit'],data['lower'],data['upper'],data['empty']
            tolerance=1e-12 if study in ('mislabel','epsilon','misspecification') or (study=='initialization' and task['score']=='mislabel') else 0
            covered=(lo<=data['truth']+tolerance)&(data['truth']<=hi+tolerance)&~empty
            if study=='mislabel':
                linear.setdefault((task['dataset'],policy),[]).append((seed,data)); continue
            if study=='nonlinear':
                prefix='cifar' if task['dataset']=='cifar10n' else task['dataset']
                nonlinear.append(pd.DataFrame(dict(setting=prefix+'_'+task['score'],policy=policy,
                    seed=seed,audit=audits,lower=lo,upper=hi,empty=empty,truth=float(data['truth']),truth_covered=covered)))
                continue
            if study=='synthetic':
                synthetic.append(pd.DataFrame(dict(rows=task['rows'],active_rows=4,row_size=250,
                    population_size=250*task['rows'],seed=seed,policy_id=policy,audit=audits,lower=lo,
                    upper=hi,width=hi-lo,truth=float(data['truth']),truth_covered=covered,
                    active_audit_fraction=np.cumsum(data['influential'])[audits-1]/audits)))
                continue
            relative=study=='initialization' and task['score']=='entropy'
            widths=(hi-lo)/np.maximum(hi+lo,1e-300) if relative else (hi-lo)/2
            target=.05 if relative else .02
            eligible=(audits>=25 if relative else audits>=100)&~empty
            hits=np.flatnonzero(eligible&(widths<=target+1e-12))
            stop=float(audits[hits[0]]) if len(hits) else np.inf
            row=dict(seed=seed,first_target_audit=stop,always_covered=bool(covered.all()))
            if study=='epsilon': row.update(dataset=task['dataset'],epsilon=task['epsilon'])
            elif study=='initialization': row.update(study='entropy' if relative else 'linear',percent=task['percent'],policy_id=policy)
            else: row.update(scenario=task['scenario'],policy=policy,covered_at_stop=bool(covered[hits[0]]) if len(hits) else False,whole_path_covered=row.pop('always_covered'))
            sensitivity[study].append(row)
    for (dataset,policy),records in linear.items():
        records.sort(key=lambda x:x[0])
        data=dict(seed=np.array([s for s,d in records]),audit=records[0][1]['audit'])
        for field in ('lower','upper','empty'):
            data[field]=np.stack([d[field] for s,d in records])
        data['half_width']=np.stack([d.get('reporting_half_width',(d['upper']-d['lower'])/2) for s,d in records])
        path=destination/'mislabel'/dataset/f'{policy}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**data)
    write_csv(pd.concat(nonlinear,ignore_index=True),destination/'nonlinear/checkpoints.csv.gz')
    path=destination/'bayesian/paths.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in bayes))
    synthetic=pd.concat(synthetic,ignore_index=True)
    main=synthetic.seed.between(2026101000,2026101099)
    for name,part in [('main',synthetic[main]),('scaling',synthetic[~main])]:
        write_csv(part,destination/'synthetic'/f'{name}.csv.gz')
    for name,rows in sensitivity.items():
        write_csv(pd.DataFrame(rows),destination/'sensitivity'/f'{name}_paths.csv')
