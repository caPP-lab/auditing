"""Exact-mean metadata control variates for the original entropy CS.

Only the 748 revealed pilot labels fit the frozen predictor. The same observed
labels and coordinate CSs feed baseline and metadata-adjusted entropy tests.
The correction has zero conditional mean for any fixed predictor values.
"""
from pathlib import Path
import argparse, os, json, time
import numpy as np
from composition_audit import CompositionAudit
from efficiency_directional import EntropyBet, sample_dyadic_row, census_hill
from directional_observations import _point,_add,_sub,_mul,_div,_sum,_finite
from adaptive_entropy_betting import CHECKPOINTS,atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/conservatism_research'

def control_observation(state,r,s,pi,c_value,c_total,c_min,c_max):
    """Outward observation and global predictable range with frozen c_i.

    c_total is an interval enclosing the sum of c_i over ALL remaining records.
    c_min/c_max are exact represented extrema over each row's remaining IDs.
    No predictor accuracy or confidence statement is assumed.
    """
    N=state.N;A=state.A;rem=state.N_r-A.sum(axis=1);active=rem>0
    pred=(A+state.beta)/(A.sum(axis=1)+1)[:,None]
    n=_point(float(N));m=_div(_point(rem.astype(float)),n)
    ratios=_div(m,_point(np.where(active,pi,1.)))
    ratios=(np.where(active,ratios[0],0.),np.where(active,ratios[1],0.))
    obs=A.sum(axis=0).astype(float);out={}
    for name,d,b,lower_side in [('lo',state.dl,state.bl,True),('hi',state.du,state.bu,False)]:
        baseline=pred@d
        known=_div(_sum(_mul(_point(obs),_point(d))),n)
        center=_add(_add(_point(float(b)),known),_sum(_mul(m,_point(baseline))))
        center=_add(center,_div(c_total,n))
        delta=_sub(_sub(_point(float(d[s])),_point(float(baseline[r]))),_point(float(c_value)))
        value=_add(center,_mul((ratios[0][r],ratios[1][r]),delta))
        min_delta=_sub(_sub(_point(float(d.min())),_point(baseline)),_point(c_max))
        max_delta=_sub(_sub(_point(float(d.max())),_point(baseline)),_point(c_min))
        minimum=_add(center,_mul(ratios,min_delta));maximum=_add(center,_mul(ratios,max_delta))
        lower=float(np.min(minimum[0][active]));upper=float(np.max(maximum[1][active]))
        out['entropy_y'+name]=max(float(value[0]),lower) if lower_side else min(float(value[1]),upper)
        out['entropy_'+('lower' if lower_side else 'upper')]=lower if lower_side else upper
    for v in out.values():_finite(v)
    return out

def predictor(seed,pop,pilot_ids):
    """Fit function receives only pilot verified labels, never full truth."""
    path=OUT/f'metadata_predictions_{seed}.npz'
    if not path.exists():
        from metadata_prediction_research import fit_boosted_metadata_scores
        fit=fit_boosted_metadata_scores(pilot_ids,pop['v_label_index'][pilot_ids])
        return fit['correction'],pilot_ids
    with np.load(path) as p:
        return p['metadata_boost_score']-p['row_only'],p['pilot_indices']

def replay(seed,max_audits=10000,scales=(1.,)):
    start=time.perf_counter();pop=np.load(ROOT/'work/replay_population.npz')
    nr=pop['N_r'];row=pop['y_row_index'];labels=pop['v_label_index'];L=len(pop['labels'])
    rng=np.random.default_rng(seed);rnglabels=np.random.default_rng(seed);arng=np.random.default_rng(seed+7121)
    orders=[]
    for r in range(len(nr)):
        ids=np.flatnonzero(row==r);perm=rng.permutation(ids)
        assert np.array_equal(labels[perm],rnglabels.permutation(labels[ids]))
        orders.append(perm)
    pilot=np.minimum(nr,np.maximum(5,np.ceil(np.sqrt(nr)).astype(int)))
    pilot_ids=np.concatenate([orders[r][:pilot[r]] for r in range(len(nr))])
    c,fit_ids=predictor(seed,pop,pilot_ids)
    assert np.array_equal(fit_ids,pilot_ids),'Fitted labels are not exactly the common randomized pilot'
    if len(c)!=len(labels) or not np.isfinite(c).all():raise ValueError('Invalid known prediction values')
    state=CompositionAudit(nr,pop['row_to_label_index'],L)
    for r in range(len(nr)):state.A[r]=np.bincount(labels[orders[r][:pilot[r]]],minlength=L)
    state.update();t=int(pilot.sum());pilott=t
    cache_path=OUT/f'adaptive_cache_{seed}.npz'
    prior_audits=np.load(cache_path)['audits'] if cache_path.exists() else None
    variants={};remaining=np.ones(len(labels),bool);remaining[pilot_ids]=False
    for scale in scales:
        cc=np.asarray(float(scale)*c,float)
        variants[float(scale)]={'c':cc,'bet':EntropyBet(L,.025,restarts=True),
            'total':_sum(_point(cc[remaining])),
            'mins':[np.minimum.accumulate(cc[ids][::-1])[::-1] for ids in orders],
            'maxs':[np.maximum.accumulate(cc[ids][::-1])[::-1] for ids in orders]}
    tl,tu,truth=census_hill(pop['C_star'].sum(axis=0),state.N);results=[];audit_ids=[];predictions=[];gaps=[]
    for goal in [v for v in CHECKPOINTS if v<=max_audits]:
        while t<goal:
            ph=(state.A+(state.N_r-state.A.sum(axis=1))[:,None]*(state.A+state.beta)/(state.A.sum(axis=1)+1)[:,None]).sum(axis=0)/state.N
            hh=float(np.log(state.estimate()));predictions.append(hh)
            gaps.append(max(0.,hh-state.bl-float(state.dl@ph)))
            r,pi,_=sample_dyadic_row(arng,state.probabilities(),state.N_r-state.A.sum(axis=1)>0)
            us=state.A.sum(axis=1);i=int(orders[r][us[r]]);s=int(labels[i]);audit_ids.append(i)
            if prior_audits is not None and t-pilott<len(prior_audits):
                assert np.array_equal(prior_audits[t-pilott],(r,s)),'Trajectory diverged from cached baseline'
            active=us<nr
            for scale,v in variants.items():
                mins=np.array([v['mins'][rr][us[rr]] if active[rr] else 0. for rr in range(len(nr))])
                maxs=np.array([v['maxs'][rr][us[rr]] if active[rr] else 0. for rr in range(len(nr))])
                ob=control_observation(state,r,s,pi,v['c'][i],v['total'],mins,maxs)
                v['bet'].add(ob['entropy_ylo'],ob['entropy_yhi'],ob['entropy_lower'],ob['entropy_upper'])
                v['total']=_sub(v['total'],_point(float(v['c'][i])))
            state.observation(r,s,pi);t+=1
            if t%500==0:state.update()
        for name,bet in [('baseline',state.bet)]+[(f'metadata_scale_{s}',v['bet']) for s,v in variants.items()]:
            lo,hi=bet.interval()
            results.append(dict(seed=seed,variant=name,t=t,lo=lo,hi=hi,width=hi-lo,estimate=state.estimate(),
                true=truth,coverage=bool(lo<=tl and tu<=hi),numeric_verified=True,pilot=pilott,
                alpha_coordinates=.020,alpha_entropy=.025,seconds=time.perf_counter()-start,
                predictor='pilot-only histogram gradient boosting, frozen after common pilot',
                bad=state.bad,crossed=bet.crossed))
    atomic_json(OUT/f'metadata_cs_{seed}.json',results)
    path=OUT/f'metadata_cs_streams_{seed}.npz';tmp=path.with_name(path.name+'.pending')
    arrays={'audit_ids':np.array(audit_ids),'pilot_ids':pilot_ids}
    arrays.update(prediction=np.array(predictions),gap_prediction=np.array(gaps),pilot=np.array(pilott),
        L=np.array(L),truth=np.array([tl,tu,truth]),seed=np.array(seed),
        estimate=np.array([(r['t'],r['estimate']) for r in results if r['variant']=='baseline']))
    for k in ['Ylo','Yhi','minlo','maxhi']:arrays['baseline_'+k]=np.array(getattr(state.bet,k))
    for scale,v in variants.items():
        for k in ['Ylo','Yhi','minlo','maxhi']:
            arrays[f'{scale}_{k}']=np.array(getattr(v['bet'],k))
            if scale==1.:arrays[k]=arrays[f'{scale}_{k}']
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    print(json.dumps({'seed':seed,'seconds':time.perf_counter()-start,
        'widths':{name:{r['t']:r['width'] for r in results if r['variant']==name} for name in sorted(set(r['variant'] for r in results))}}),flush=True)
    return results

def summarize():
    import hashlib
    seeds=list(range(92001,92021));methods=['baseline','metadata_scale_1.0'];rows=[]
    for seed in seeds:rows+=json.loads((OUT/f'metadata_cs_{seed}.json').read_text())
    mapping={(r['seed'],r['variant'],r['t']):r for r in rows}
    expected={(s,m,t) for s in seeds for m in methods for t in CHECKPOINTS}
    if set(mapping)!=expected or len(rows)!=len(expected):raise ValueError('Incomplete/duplicate metadata validation')
    rng=np.random.default_rng(9202101);idx=rng.integers(0,20,size=(20000,20))
    ans=dict(seeds=seeds,development_seed=1729,methods=methods,scaling_frozen=1.,
        truth=rows[0]['true'],pilot=748,confidence_failure_bound=.045,
        checkpoint_count=len(rows),covered_checkpoints=sum(r['coverage'] for r in rows),
        numeric_verified=all(r['numeric_verified'] for r in rows),widths={},stopping={},
        bad_or_crossed_count=sum(r['bad'] or r['crossed'] for r in rows),
        scope='Paired review-randomization paths on one fixed population; identical reviews, allocation, coordinateCS, envelopes, pointestimates, alpha; only scalar entropy control variate differs',
        source_sha256={p:hashlib.sha256((ROOT/'work'/p).read_bytes()).hexdigest() for p in ['metadata_entropy_audit.py','metadata_prediction_research.py','directional_observations.py','efficiency_directional.py']})
    elapsed=[max(r['seconds'] for r in rows if r['seed']==s) for s in seeds]
    ans['runtime']={'joint_fit_and_paired_replay_to10000_median_seconds':float(np.median(elapsed)),
        'range_seconds':[float(min(elapsed)),float(max(elapsed))],
        'scope':'Fit plus simultaneous baseline/metadata inference on identical10000review replay; two validation processes in sharedruntime; excludes actual humanreview; trainingtime not separately measured'}
    for t in CHECKPOINTS:
        a=np.array([mapping[s,methods[0],t]['width'] for s in seeds])
        b=np.array([mapping[s,methods[1],t]['width'] for s in seeds])
        boot=100*(1-b[idx].mean(1)/a[idx].mean(1))
        ans['widths'][str(t)]={'baseline_mean_width':float(a.mean()),'metadata_mean_width':float(b.mean()),
            'percent_reduction':float(100*(1-b.mean()/a.mean())),
            'paired_bootstrap_95_percent_reduction':np.quantile(boot,[.025,.975]).tolist(),
            'baseline_halfwidth_percent_truth':float(50*a.mean()/ans['truth']),
            'metadata_halfwidth_percent_truth':float(50*b.mean()/ans['truth']),
            'mean_absolute_estimation_error':float(np.mean([abs(mapping[s,methods[0],t]['estimate']-ans['truth']) for s in seeds]))}
        assert all(mapping[s,methods[0],t]['estimate']==mapping[s,methods[1],t]['estimate'] for s in seeds)
    criteria={'full_width_1':lambda r:r['width']<=1,
        'full_width_0.5':lambda r:r['width']<=.5,
        'midpoint_relative_5pct':lambda r:r['width']<=.1*r['lo'],
        'plugin_absolute_0.5':lambda r:max(r['estimate']-r['lo'],r['hi']-r['estimate'])<=.5,
        'plugin_absolute_0.25':lambda r:max(r['estimate']-r['lo'],r['hi']-r['estimate'])<=.25}
    for name,test in criteria.items():
        res={}
        for m in methods:
            stops=[next((t for t in CHECKPOINTS if not(mapping[s,m,t]['bad'] or mapping[s,m,t]['crossed']) and test(mapping[s,m,t])),None) for s in seeds]
            res[m]={'audit_counts':stops,'censored':sum(v is None for v in stops),
                'mean_audits':float(np.mean(stops)) if all(v is not None for v in stops) else None}
        if all(res[m]['mean_audits'] is not None for m in methods):
            a=np.array(res[methods[0]]['audit_counts']);b=np.array(res[methods[1]]['audit_counts'])
            res['reviews_saved']=float(a.mean()-b.mean());res['percent_reduction']=float(100*(1-b.mean()/a.mean()))
            res['paired_bootstrap_95_reviews_saved']=np.quantile((a[idx]-b[idx]).mean(1),[.025,.975]).tolist()
        ans['stopping'][name]=res
    for m in methods:ans[m+'_path_coverage']=sum(all(mapping[s,m,t]['coverage'] for t in CHECKPOINTS) for s in seeds)
    atomic_json(OUT/'metadata_validation_summary.json',ans);return ans

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=1729)
    ap.add_argument('--max-audits',type=int,default=10000);ap.add_argument('--scales',nargs='+',type=float,default=[1.])
    args=ap.parse_args();replay(args.seed,args.max_audits,args.scales)
