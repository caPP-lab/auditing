"""Frozen experimental procedures for the reported main and appendix studies."""
from dataclasses import replace
from statistics import NormalDist
import numpy as np

from .datasets import load, truth, bayesian_population
from .audit import LabelBlindAudit, common_config
from .mislabel import MislabelAuditConfig, serfling_half_width
from .allocation import chebyshev_half_width
from .strata import _categorical_row_sampler
from .reference import _load_reference_modules
from .rlfa import LabelBlindAuthorsRLFA
from .rlfa_cv import LabelBlindAuthorsRLFACV
from .entropy import OptimizedEntropyAudit, EntropyAuditConfig
from .tail_scores import CifarWorstThreeAudit, TailBettingConfig
from .f1 import SubsetF1Audit
from .uncertainty_only import UncertaintyOnlyEntropyAudit, UncertaintyOnlyCifarAudit, UncertaintyOnlyF1Audit
from .finite_population_baselines import ppr_running_intervals, ppr_current_interval
from .ppr_projection import ppr_count_intervals, entropy_projection, tail_projection, box_simplex_linear
from .stitching import MRConfig, MREntropyCS, MRTailCS
from .stitching_f1 import MRF1CS
from .predictor_stress import SCENARIO_BY_NAME, MisspecifiedMislabelAudit
from .synthetic import make_synthetic_risk_scenario, SyntheticLinearRiskAudit
from .bayesian import run_path as bayesian_path

ALPHA = .045
LINEAR_SEEDS = tuple(range(2026110900, 2026111000))
NONLINEAR_SEEDS = {'kenya': tuple(range(95001, 95101)),
                   'cifar10n': tuple(range(2026100300, 2026100400)),
                   'reimagenet': tuple(range(96201, 96301)),
                   'cleanconll': tuple(range(96201, 96301))}
LINEAR_POLICIES = ('pas', 'srs', 'prop_ms', 'prop_m_cv', 'ppr', 'wald', 'serfling', 'chebyshev')
NONLINEAR_POLICIES = ('pas', 'srs', 'uncertainty_only', 'ppr_projection', 'mr_stitching')
SCORES = {'kenya': ('entropy',), 'cifar10n': ('worst3_precision', 'worst3_taxonomic_risk'),
          'reimagenet': ('f1',), 'cleanconll': ('f1',)}
EPSILONS = (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.)


def srs_records(rows, seed, budget, *, dyadic=False):
    rows = np.asarray(rows, int)
    sizes = np.bincount(rows)
    rng = np.random.default_rng(seed)
    orders = [list(rng.permutation(np.flatnonzero(rows == g))[::-1]) for g in range(len(sizes))]
    rng = np.random.default_rng(seed + 7121)
    sampler = _load_reference_modules()['directional_observations'].sample_dyadic_row if dyadic else _categorical_row_sampler
    remaining = sizes.copy()
    result = np.empty(budget, int)
    for t in range(budget):
        group, _, _ = sampler(rng, remaining / remaining.sum(), remaining > 0)
        result[t] = orders[int(group)].pop()
        remaining[group] -= 1
    return result


def pack(audits, endpoints, *, empty=None, records=None, value=None, **extra):
    endpoints = np.asarray(endpoints, float)
    empty = endpoints[:,0] > endpoints[:,1] if empty is None else np.asarray(empty, bool) | (endpoints[:,0] > endpoints[:,1])
    result = dict(audit=np.asarray(audits, int), lower=endpoints[:,0], upper=endpoints[:,1], empty=empty, **extra)
    if records is not None:
        result['record_ids'] = np.asarray(records, np.int32)
    if value is not None:
        result['truth'] = np.asarray(value)
    return result


def mislabel(dataset, policy, seed, *, budget=5000, config=None, stress=None):
    pop = load(dataset)
    rows, mapping = pop['rows'], pop['row_labels']
    errors = (pop['y'] != pop['v']).astype(int)
    cfg = config or common_config()
    sampler = None if dataset == 'kenya' else _categorical_row_sampler
    if policy in ('pas', 'srs', 'prop_ms', 'prop_m_cv'):
        kwargs = dict(seed=seed, config=cfg)
        if policy == 'prop_ms':
            engine = LabelBlindAuthorsRLFA(rows, mapping, **kwargs)
        elif policy == 'prop_m_cv':
            engine = LabelBlindAuthorsRLFACV(rows, mapping, row_sampler=sampler, **kwargs)
        elif stress is not None:
            engine = MisspecifiedMislabelAudit(rows, mapping, policy=policy, row_sampler=sampler,
                                               stress=SCENARIO_BY_NAME[stress], **kwargs)
        else:
            engine = LabelBlindAudit(rows, mapping, policy=policy, row_sampler=sampler, **kwargs)
        ids, endpoints, empty = [], [], []
        for _ in range(budget):
            request = engine.prepare()
            rid = int(request['record_id'])
            engine.observe_error(rid, int(errors[rid]))
            ci = engine.interval()
            ids.append(rid); endpoints.append((ci['lower'], ci['upper'])); empty.append(ci['empty'])
        return pack(np.arange(1,budget+1), endpoints, empty=empty, records=ids, value=float(errors.mean()))
    ids = srs_records(rows, seed, budget, dyadic=dataset == 'kenya')
    n = np.arange(1, budget+1)
    if policy == 'ppr':
        ci = ppr_running_intervals(errors[ids], len(rows), alpha=cfg.alpha)
        return pack(n, np.column_stack((ci['lower'],ci['upper'])), records=ids, value=float(errors.mean()))
    center = np.cumsum(errors[ids]) / n
    if policy == 'wald':
        raw = NormalDist().inv_cdf(1-cfg.alpha/2) * np.sqrt(np.var(errors, ddof=0) / n)
        raw[0] = np.inf
    elif policy == 'serfling':
        raw = serfling_half_width(n, len(rows), alpha=cfg.alpha)
    elif policy == 'chebyshev':
        raw = chebyshev_half_width(n, len(rows), alpha=cfg.alpha)
    else:
        raise ValueError(policy)
    lo, hi = np.maximum(0, center-raw), np.minimum(1, center+raw)
    width = raw if (dataset == 'cleanconll' and policy in ('wald','serfling')) or policy == 'chebyshev' else (hi-lo)/2
    return pack(n, np.column_stack((lo,hi)), records=ids, value=float(errors.mean()), reporting_half_width=width)


def coupling_budget(N, groups):
    return float(N * 16 * (groups+1)**2 * np.finfo(float).eps)


class ErrorPPR:
    def __init__(self, N, A, alpha):
        self.N, self.A, self.alpha = N, A, alpha
        self.bounds = [[0,N],[0,N]]

    def update(self, t, fp, fn):
        for j, count in enumerate((fp, fn)):
            ci = ppr_current_interval(population_size=self.N, reviews=t, observed_errors=count, alpha=self.alpha/2)
            if ci is None:
                self.bounds[j] = [1,0]
            else:
                self.bounds[j][0] = max(self.bounds[j][0],ci[0])
                self.bounds[j][1] = min(self.bounds[j][1],ci[1])
        (fl,fu),(nl,nu) = self.bounds
        fu = min(fu,self.A)
        if fl > fu or nl > nu:
            return 1.,0.
        lo = 2*(self.A-fu)/(2*self.A-fu+nu) if 2*self.A-fu+nu else 0.
        hi = 2*(self.A-fl)/(2*self.A-fl+nl) if 2*self.A-fl+nl else 1.
        return max(0.,lo),min(1.,hi)


def projected_path(dataset, score, seed, budget):
    pop = load(dataset)
    y,v,K = pop['y'],pop['v'],pop['categories']
    rows = pop['strata'] if score == 'entropy' else (y if dataset == 'cifar10n' else pop['rows'])
    ids = srs_records(rows, seed, budget, dyadic=score == 'entropy')
    N = len(y); checks=[]; endpoints=[]; failures=[]
    running = (1.,float(K)) if score == 'entropy' else (0.,1.)
    if score == 'entropy':
        G = len(pop['stratum_rows'])
        alpha = ALPHA - N*(G*G/2**54 + 4*np.finfo(float).eps)
        counts=np.zeros(K,int); low=np.zeros(K,int); high=np.full(K,N,int)
        h=(0.,np.log(K))
    elif score == 'f1':
        target=np.isin(np.arange(K),pop['target']); a=target[y]
        engine=ErrorPPR(N,int(a.sum()),ALPHA); fp=fn=0
    else:
        sizes=np.bincount(y,minlength=10); audits=np.zeros(10,int)
        counts=np.zeros((10,10),int)
        low=np.zeros((10,10),int); high=np.broadcast_to(sizes[:,None],(10,10)).copy()
        pl=np.zeros(10,int); pu=sizes.copy()
        costs=pop['distances']/pop['d_max']
        failed=False
    for t,rid in enumerate(ids,1):
        label=int(v[rid])
        if score == 'entropy':
            counts[label]+=1
            lo,hi=ppr_count_intervals(N,t,counts,alpha/K)
            low=np.maximum(low,lo);high=np.minimum(high,hi)
            empty=bool(np.any(low>high) or low.sum()>N or high.sum()<N)
        elif score == 'f1':
            fp+=int(a[rid] and y[rid]!=label)
            fn+=int(target[label] and y[rid]!=label)
        else:
            r=int(y[rid]);audits[r]+=1;counts[r,label]+=1
            if score == 'worst3_precision':
                lo,hi=ppr_count_intervals(int(sizes[r]),int(audits[r]),np.array([counts[r,r]]),ALPHA/10)
                pl[r]=max(pl[r],lo[0]);pu[r]=min(pu[r],hi[0])
                failed |= bool(np.any(pl>pu))
            else:
                lo,hi=ppr_count_intervals(int(sizes[r]),int(audits[r]),counts[r],ALPHA/100)
                low[r]=np.maximum(low[r],lo);high[r]=np.minimum(high[r],hi)
                failed |= bool(np.any(low>high) or np.any(low.sum(1)>sizes) or np.any(high.sum(1)<sizes))
        if t%25:
            continue
        if score == 'entropy':
            if not empty:
                ci=entropy_projection(low/N,high/N)
                h=max(h[0],ci['lower']),min(h[1],ci['upper'])
            empty |= h[0]>h[1]
            interval=(1.,float(K)) if empty else tuple(np.exp(h))
        elif score == 'f1':
            interval=engine.update(t,fp,fn);empty=interval[0]>interval[1]
        else:
            if not failed:
                if score == 'worst3_precision':
                    ci=tail_projection(pl/sizes,pu/sizes)
                else:
                    rowlo=np.empty(10);rowhi=np.empty(10)
                    for j in range(10):
                        rowlo[j]=max(0.,box_simplex_linear(low[j],high[j],costs[j],total=int(sizes[j]))[0]/sizes[j]-1e-12)
                        rowhi[j]=min(1.,box_simplex_linear(low[j],high[j],costs[j],total=int(sizes[j]),maximize=True)[0]/sizes[j]+1e-12)
                    ci=tail_projection(rowlo,rowhi,largest=True)
                running=max(running[0],ci[0]),min(running[1],ci[1])
                failed |= running[0]>running[1]
            empty=failed;interval=(0.,1.) if failed else running
        checks.append(t);endpoints.append(interval);failures.append(empty)
    return pack(checks,endpoints,empty=failures,records=ids,value=truth(dataset,score))


def stitched_path(dataset, score, seed, budget):
    p=load(dataset);y,v,K=p['y'],p['v'],p['categories']
    rows=p['strata'] if score=='entropy' else (y if dataset=='cifar10n' else p['rows'])
    ids=srs_records(rows,seed,budget,dyadic=score=='entropy')
    cfg=MRConfig(alpha=ALPHA-coupling_budget(len(y),len(np.unique(rows))))
    if score=='entropy':
        engine=MREntropyCS(len(y),K,config=cfg)
    elif score=='f1':
        target=np.isin(np.arange(K),p['target']);a=target[y]
        engine=MRF1CS(len(y),int(a.sum()),config=cfg)
    else:
        cost=np.eye(10) if score=='worst3_precision' else p['distances']/p['d_max']
        engine=MRTailCS(np.bincount(y,minlength=10),cost,k=3,largest=score=='worst3_taxonomic_risk',config=cfg)
    checks=[];endpoints=[];empty=[]
    for t,rid in enumerate(ids,1):
        label=int(v[rid])
        if score=='entropy': ci=engine.observe(label)
        elif score=='f1': ci=engine.observe(a[rid],target[label],y[rid]==label)
        else: ci=engine.observe(int(y[rid]),label)
        if t%25==0:
            checks.append(t);endpoints.append((ci['lower'],ci['upper']));empty.append(ci['empty'])
    return pack(checks,endpoints,empty=empty,records=ids,value=truth(dataset,score))


def nonlinear(dataset, score, policy, seed, *, budget=None, config=None):
    if budget is None:
        budget=3000 if dataset=='kenya' and policy in ('pas','srs','uncertainty_only') else 10000
    if policy=='ppr_projection': return projected_path(dataset,score,seed,budget)
    if policy=='mr_stitching': return stitched_path(dataset,score,seed,budget)
    p=load(dataset);y,v=p['y'],p['v'];uncertainty=policy=='uncertainty_only'
    actual_policy='pas' if uncertainty else policy
    if score=='entropy':
        cls=UncertaintyOnlyEntropyAudit if uncertainty else OptimizedEntropyAudit
        engine=cls(p['strata'],p['stratum_rows'],p['stratum_labels'],p['categories'],seed=seed,
                   policy=actual_policy,config=config or EntropyAuditConfig())
    elif score=='f1':
        cls=UncertaintyOnlyF1Audit if uncertainty else SubsetF1Audit
        engine=cls(y,p['target'],p['categories'],seed=seed,policy=actual_policy,config=config or MislabelAuditConfig())
    else:
        kwargs=dict(seed=seed,score_id=score,policy=actual_policy,config=config or TailBettingConfig(),
                    taxonomy_distances=p['distances'],taxonomy_d_max=p['d_max'])
        engine=UncertaintyOnlyCifarAudit(y,**kwargs) if uncertainty else CifarWorstThreeAudit(y,v,**kwargs)
    ids=[];checks=[];endpoints=[];empty=[]
    for t in range(1,budget+1):
        rid=int(engine.prepare()['record_id']);ids.append(rid)
        if score in ('entropy','f1') or uncertainty: engine.observe(rid,int(v[rid]))
        else: engine.observe(rid)
        # Tail/F1 engines retain their every-audit running intersection.
        if score!='entropy': ci=engine.interval()
        elif t%25==0: ci=engine.interval(force=True)
        if t%25==0:
            checks.append(t);endpoints.append((ci['lower'],ci['upper']))
            empty.append(ci.get('empty',ci.get('empty_grid_event',False)))
    return pack(checks,endpoints,empty=empty,records=ids,value=truth(dataset,score))


def synthetic(rows, seed, policy, *, budget=800):
    scenario=make_synthetic_risk_scenario(rows)
    engine=SyntheticLinearRiskAudit(scenario,seed=seed,policy=policy)
    ids=[];selected=[];checks=[];endpoints=[];influential=[]
    for t in range(1,budget+1):
        request=engine.prepare();rid=int(request['record_id']);ids.append(rid)
        selected.append(int(request['row']));influential.append(int(request['row']) in scenario.influential_indices)
        engine.observe(rid);ci=engine.interval()
        if t%25==0:
            checks.append(t);endpoints.append((ci['lower'],ci['upper']))
    return pack(checks,endpoints,records=ids,value=scenario.truth,
                selected_rows=np.asarray(selected,np.int16), influential=np.asarray(influential,bool))


def bayesian(seed,policy):
    return bayesian_path((bayesian_population(),seed,policy))
