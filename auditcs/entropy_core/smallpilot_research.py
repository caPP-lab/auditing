"""Review-one inference with a predeclared later metadata fit.

Every reviewed label enters the confidence process. At a fixed review count,
fit the metadata score using labels seen so far and apply its exact-zero-mean
control correction to future observations only. Alternative fits are reported
as separate research methods, never intersected or selected for an online claim.
"""
from pathlib import Path
import argparse, json, os, time
import numpy as np
from composition_audit import CompositionAudit
from efficiency_directional import EntropyBet, census_hill, sample_dyadic_row
from directional_observations import _point, _sum, _sub
from metadata_entropy_audit import control_observation
from metadata_prediction_research import fit_boosted_metadata_scores

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'optimal_revision'
CHECKPOINTS=[100,200,300,500,750,1000,1500,2000,2500,3000,4000,5000]

def save(path,obj):
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('w') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def make_bet(L,restart):
    b=EntropyBet(L,.025,restarts=True)
    if restart=='early':
        b.starts=np.array([0,100,300,700,1500,3100,6300,12700,25500])
        b.weights=np.array([.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.00390625])
    elif restart!='baseline':raise ValueError(restart)
    return b

def floor_variance_allocation(rem,a,epsilon=.5):
    """Minimize sum a_r^2/pi_r subject to pi>=epsilon*remaining fraction.

    Floating probabilities are merely proposals: the existing dyadic sampler
    reports the represented probabilities actually used by all observations.
    """
    rem=np.asarray(rem,float);a=np.asarray(a,float);active=rem>0
    if not 0<epsilon<=1 or np.any(a<0) or not np.isfinite(a).all():
        raise ValueError('Invalid variance weights/exploration floor')
    base=rem/rem.sum();floor=epsilon*base;pi=np.zeros(len(rem));free=active.copy()
    if not np.any(a[active]>0):return base
    while np.any(free):
        mass=1-pi[~free].sum();candidate=np.zeros(len(rem))
        denom=a[free].sum()
        candidate[free]=mass*(a[free]/denom if denom>0 else rem[free]/rem[free].sum())
        freeze=free&(candidate<floor)
        if not np.any(freeze):pi[free]=candidate[free];break
        pi[freeze]=floor[freeze];free[freeze]=False
    return pi

def fit_state_scores(state,ids,verified_labels,pop):
    """Same fixed metadata learner, with the audit state's working prior.

    ``verified_labels`` contains only the passed reviewed IDs. CSV reads are
    restricted to pre-audit metadata; no verified-label column is opened.
    """
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    nr=state.N_r;row=pop['y_row_index'];u=state.A.sum(1)
    q=(state.A+state.beta)/(u+1)[:,None]
    ph=(state.A+(nr-u)[:,None]*q).sum(0)/state.N
    d=-np.log(ph);rb=q@d
    allowed=['station_name','object_class','treatment','observation_datetime','megadetector_confidence']
    df=pd.read_csv(ROOT/'upload'/'labels_keyword_cleaned(4).csv',usecols=allowed)
    df=df.iloc[pop['csv_line_number']-2].reset_index(drop=True)
    stamp=pd.to_datetime(df.observation_datetime,utc=True)
    hour=stamp.dt.hour.to_numpy()+stamp.dt.minute.to_numpy()/60
    cat=np.column_stack((row.astype(str),df.station_name.astype(str),df.object_class.astype(str),
                         df.treatment.astype(str),np.floor(hour/6).astype(int).astype(str)))
    numeric=np.column_stack((df.megadetector_confidence.to_numpy(float),
                             np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)))
    X=np.column_stack((OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1).fit_transform(cat),numeric))
    target=d[np.asarray(verified_labels,int)]-rb[row[ids]]
    weights=nr[row[ids]]/u[row[ids]];weights/=weights.mean()
    model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,max_depth=4,
        min_samples_leaf=15,l2_regularization=10.,categorical_features=[0,1,2,3,4],random_state=93746)
    with threadpool_limits(limits=2):
        model.fit(X[ids],target,sample_weight=weights)
        scores=np.clip(rb[row]+model.predict(X),d.min(),d.max())
    return scores-rb[row]

def replay(seed=1729,train_at=(0,100,200,400),max_audits=5000,
           restarts=('baseline','early'),allocation='original',tag='development',refit_at=(),prior='catalogue'):
    """All allocation and inferential choices use revealed data only.

    ``train_at=0`` means no metadata control. Baseline allocation is shared by
    all variants. Residual allocation, if enabled, requires a single nonzero
    training count; its rowwise residual moments use revealed records only.
    """
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    if allocation!='original' and len(train_at)!=1:
        raise ValueError('A residual allocation experiment has one frozen training time.')
    if refit_at and len(train_at)!=1:
        raise ValueError('A repeated-fit experiment requires one initial fit time.')
    p=np.load(ROOT/'work'/'replay_population.npz');nr=p['N_r'];row=p['y_row_index'];labels=p['v_label_index'];L=len(p['labels'])
    rng=np.random.default_rng(seed);arng=np.random.default_rng(seed+7121)
    orders=[rng.permutation(np.flatnonzero(row==r)) for r in range(len(nr))]
    state=CompositionAudit(nr,p['row_to_label_index'],L)
    if prior in ('submitted','submitted_laplace'):
        counts=np.bincount(p['row_to_label_index'],weights=nr,minlength=L)
        g=(counts+1)/(nr.sum()+L) if prior=='submitted_laplace' else .999*counts/nr.sum()+.001/L
        state.beta=np.tile(.5*g,(len(nr),1))
        state.beta[np.arange(len(nr)),p['row_to_label_index']]+=.5
    elif prior!='catalogue':raise ValueError(prior)
    state.update();variants={};seen=[];rows=[];estimates=[];predictions=[];gaps=[]
    for b in train_at:
        variants[int(b)]={'c':None,'bets':{k:make_bet(L,k) for k in restarts}}
    tl,tu,truth=census_hill(p['C_star'].sum(0),int(nr.sum()))
    for t0 in range(max_audits):
        t=t0+1;us=state.A.sum(1);rem=nr-us;active=rem>0
        ph=(state.A+rem[:,None]*(state.A+state.beta)/(us+1)[:,None]).sum(0)/state.N
        hh=float(np.log(state.estimate()));predictions.append(hh)
        gaps.append(max(0.,hh-state.bl-float(state.dl@ph)))
        desired=state.probabilities()
        if allocation in ('residual','waterfill') and variants[int(train_at[0])]['c'] is not None:
            v=variants[int(train_at[0])]
            # Predictable shrinkage of observed residual second moments. The
            # original row predictor supplies a strictly positive fallback.
            rr=row[np.asarray(seen)];vv=labels[np.asarray(seen)]
            baseline=(state.A+state.beta)/(us+1)[:,None]@state.du
            resid=state.du[vv]-baseline[rr]-v['c'][seen]
            second=np.bincount(rr,weights=resid**2,minlength=len(nr))
            var=(second+10*state.sd**2)/(us+10)
            a=rem*np.sqrt(np.maximum(var,1e-12))
            desired=floor_variance_allocation(rem,a) if allocation=='waterfill' else .5*rem/rem.sum()+.5*a/a.sum()
        elif allocation not in ('original','residual','waterfill'):raise ValueError(allocation)
        r,pi,_=sample_dyadic_row(arng,desired,active)
        i=int(orders[r][us[r]]);s=int(labels[i]);seen.append(i)
        # state.observation computes the original coordinate and entropy
        # observations BEFORE incorporating the new verified label.
        for b,v in variants.items():
            if v['c'] is None:
                continue
            mins=np.array([v['mins'][rr][us[rr]] if active[rr] else 0. for rr in range(len(nr))])
            maxs=np.array([v['maxs'][rr][us[rr]] if active[rr] else 0. for rr in range(len(nr))])
            ob=control_observation(state,r,s,pi,v['c'][i],v['total'],mins,maxs)
            for bet in v['bets'].values():
                bet.add(ob['entropy_ylo'],ob['entropy_yhi'],ob['entropy_lower'],ob['entropy_upper'])
            v['total']=_sub(v['total'],_point(float(v['c'][i])))
        state.observation(r,s,pi)
        for b,v in variants.items():
            if v['c'] is None:
                for bet in v['bets'].values():
                    bet.add(state.bet.Ylo[-1],state.bet.Yhi[-1],state.bet.minlo[-1],state.bet.maxhi[-1])
        # Refresh times are fixed before the audit, with denser early updates.
        if (t<=1000 and (t%100==0 or t==750)) or t%500==0:state.update()
        fit_target=t if t in variants else (int(train_at[0]) if t in refit_at and t>=int(train_at[0]) else None)
        if fit_target is not None and fit_target>0:
            ids=np.asarray(seen)
            c=fit_state_scores(state,ids,labels[ids],p) if prior!='catalogue' else np.asarray(fit_boosted_metadata_scores(ids,labels[ids])['correction'],float)
            v=variants[fit_target];v['c']=c
            remaining=np.ones(state.N,bool);remaining[ids]=False
            v['total']=_sum(_point(c[remaining]))
            v['mins']=[np.minimum.accumulate(c[ids][::-1])[::-1] for ids in orders]
            v['maxs']=[np.maximum.accumulate(c[ids][::-1])[::-1] for ids in orders]
        if t in CHECKPOINTS:
            estimates.append((t,state.estimate()))
            # Auxiliary point estimate: average the design-unbiased vector
            # observations and Euclidean-project onto the probability simplex.
            # This changes no betting factor, endpoint, allocation or stopping.
            ph_ht=(np.mean(state.coordinates.y,axis=0)+np.mean(state.coordinates.y_upper,axis=0))/2
            sorted_ht=np.sort(ph_ht)[::-1];css=np.cumsum(sorted_ht)-1
            rho=np.flatnonzero(sorted_ht-css/np.arange(1,L+1)>0)[-1]
            ph_ht=np.maximum(ph_ht-css[rho]/(rho+1),0)
            ht_estimate=float(np.exp(-np.sum(ph_ht[ph_ht>0]*np.log(ph_ht[ph_ht>0]))))
            reporting=variants if 0 in variants else {0:{'bets':{'baseline':state.bet}},**variants}
            for b,v in reporting.items():
                for restart,bet in v['bets'].items():
                    lo,hi=bet.interval()
                    rec=dict(seed=seed,train_at=b,t=t,restart=restart,allocation=allocation,
                        lo=lo,hi=hi,width=hi-lo,estimate=state.estimate(),partial=state.partial(),
                        ht_estimate=ht_estimate,
                        truth=truth,coverage=bool(lo<=tl and tu<=hi),numeric_verified=True,
                        discarded_pilot=0,training_reviews_included_in_inference=True,
                        refit_at=list(refit_at),
                        prior=prior,
                        alpha_coordinates=.020,alpha_entropy=.025,bad=state.bad,crossed=bet.crossed,
                        seconds=time.perf_counter()-start)
                    rows.append(rec)
            print(json.dumps({'seed':seed,'t':t,'widths':{f"fit{r['train_at']}_{r['restart']}":r['width'] for r in rows if r['t']==t},'seconds':time.perf_counter()-start}),flush=True)
            save(OUT/f'smallpilot_{tag}_{seed}_{allocation}.json',rows)
    for b,v in variants.items():
        bet=v['bets'][restarts[0]]
        path=OUT/f'smallpilot_stream_{tag}_{seed}_{allocation}_fit{b}.npz'
        tmp=path.with_name(path.name+'.pending')
        data={k:np.asarray(getattr(bet,k)) for k in ('Ylo','Yhi','minlo','maxhi')}
        data.update(seed=seed,L=L,pilot=0,train_at=b,truth=np.array([tl,tu,truth]),
            estimate=np.array(estimates),prediction=np.array(predictions),gap_prediction=np.array(gaps),
            audit_ids=np.array(seen))
        with tmp.open('wb') as f:np.savez_compressed(f,**data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=1729)
    ap.add_argument('--train-at',type=int,nargs='+',default=[0,100,200,400])
    ap.add_argument('--max-audits',type=int,default=5000)
    ap.add_argument('--restarts',nargs='+',default=['baseline','early'])
    ap.add_argument('--allocation',default='original');ap.add_argument('--tag',default='development')
    ap.add_argument('--refit-at',nargs='*',type=int,default=[])
    ap.add_argument('--prior',default='catalogue')
    a=ap.parse_args();replay(a.seed,a.train_at,a.max_audits,a.restarts,a.allocation,a.tag,a.refit_at,a.prior)
