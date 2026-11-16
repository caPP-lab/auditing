"""Run the frozen camera-ready experiments; each completed path is resumable."""
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import argparse
import json
import numpy as np
from atlas_auditing import experiments as ex
from atlas_auditing.datasets import DATASETS

ROOT = Path(__file__).resolve().parent
STUDIES = ('mislabel','nonlinear','bayesian','synthetic','epsilon','initialization','misspecification')


def tasks(study, datasets, seed_index=None):
    def seeds(values):
        return values if seed_index is None else [values[seed_index]]
    if study == 'mislabel':
        for d in datasets:
            for p in ex.LINEAR_POLICIES[:-1]:  # Chebyshev's fixed-n count is analytic.
                for s in seeds(ex.LINEAR_SEEDS):
                    yield dict(study=study,dataset=d,policy=p,seed=s)
    elif study == 'nonlinear':
        for d in datasets:
            for score in ex.SCORES[d]:
                for p in ex.NONLINEAR_POLICIES:
                    for s in seeds(ex.NONLINEAR_SEEDS[d]):
                        yield dict(study=study,dataset=d,score=score,policy=p,seed=s)
    elif study == 'bayesian' and 'kenya' in datasets:
        for p in ('pas','srs'):
            for s in seeds(tuple(range(8000,8100))):
                yield dict(study=study,policy=p,seed=s)
    elif study == 'synthetic':
        for rows, values in [(20,tuple(range(2026101000,2026101100))),
                             *[(r,tuple(range(2026092400,2026092405))) for r in (8,12,20,32,48)]]:
            if seed_index is not None and seed_index >= len(values):
                continue
            for p in ('pas','srs'):
                for s in seeds(values):
                    yield dict(study=study,rows=rows,policy=p,seed=s)
    elif study == 'epsilon':
        for d in datasets:
            for epsilon in ex.EPSILONS:
                for s in seeds(ex.LINEAR_SEEDS):
                    yield dict(study=study,dataset=d,epsilon=epsilon,policy='pas',seed=s)
    elif study == 'initialization' and 'kenya' in datasets:
        for score in ('mislabel','entropy'):
            for percent in range(0,101,10):
                for p in ('pas','srs'):
                    for s in seeds(ex.LINEAR_SEEDS if score=='mislabel' else ex.NONLINEAR_SEEDS['kenya']):
                        yield dict(study=study,score=score,percent=percent,policy=p,seed=s)
    elif study == 'misspecification' and 'kenya' in datasets:
        for scenario in ex.SCENARIO_BY_NAME:
            for p in ('pas','srs'):
                for s in seeds(ex.LINEAR_SEEDS):
                    yield dict(study=study,scenario=scenario,policy=p,seed=s)


def run(task, output):
    study=task['study'];policy=task['policy'];seed=task['seed']
    parts=[study]+[str(task[k]) for k in ('dataset','score','rows','epsilon','percent','scenario') if k in task]
    path=Path(output).joinpath(*parts,policy,f'{seed}.json' if study=='bayesian' else f'{seed}.npz')
    if path.exists():
        return str(path)
    if study=='mislabel': result=ex.mislabel(task['dataset'],policy,seed)
    elif study=='nonlinear': result=ex.nonlinear(task['dataset'],task['score'],policy,seed)
    elif study=='bayesian': result=ex.bayesian(seed,policy)
    elif study=='synthetic': result=ex.synthetic(task['rows'],seed,policy)
    elif study=='epsilon':
        result=ex.mislabel(task['dataset'],policy,seed,config=replace(ex.common_config(),srs_floor=task['epsilon']))
    elif study=='initialization':
        if task['score']=='mislabel':
            result=ex.mislabel('kenya',policy,seed,config=replace(ex.common_config(),submitted_label_probability=task['percent']/100))
        else:
            result=ex.nonlinear('kenya','entropy',policy,seed,
                config=replace(ex.EntropyAuditConfig(),submitted_label_prior=(100-task['percent'])/100))
    elif study=='misspecification':
        result=ex.mislabel('kenya',policy,seed,stress=task['scenario'])
    else: raise ValueError(study)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    if study=='bayesian':
        tmp.write_text(json.dumps(result,separators=(',',':'))+'\n')
    else:
        with tmp.open('wb') as stream:
            np.savez_compressed(stream,**result)
    tmp.replace(path)
    return str(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=('all',)+STUDIES,default='all')
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--seed-index',type=int,help='Run one of the frozen seeds (0--99).')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--output',type=Path,default=ROOT/'runs')
    args=parser.parse_args()
    if args.seed_index is not None and not 0<=args.seed_index<100:
        parser.error('--seed-index must be between 0 and 99')
    studies=STUDIES if args.study=='all' else (args.study,)
    jobs=[job for study in studies for job in tasks(study,args.datasets,args.seed_index)]
    if args.workers==1:
        for index,job in enumerate(jobs,1):
            print(f'{index}/{len(jobs)} {run(job,args.output)}',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run,job,args.output) for job in jobs]
            for index,future in enumerate(as_completed(futures),1):
                print(f'{index}/{len(jobs)} {future.result()}',flush=True)


if __name__=='__main__':
    main()
