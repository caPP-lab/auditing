"""Prespecified, predictable scalar betting refinements; core audit unchanged.

Replay stores only outward observations, predictable ranges and audit-based
point predictions as inferential inputs. Population truth is evaluation only.
Every nontrivial endpoint is independently checked with outward product wealth.
"""
from pathlib import Path
import argparse, json, os, time
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brentq
from composition_audit import CompositionAudit
from efficiency_directional import EntropyBet, census_hill, sample_dyadic_row, _I, _D, _fup, _fdown, _exp
from directional_numeric import (_down, _up, _positive_down, _scaled_product_lower,
    _scaled_sum_lower)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/conservatism_research'
CHECKPOINTS=[1000,1500,2000,2500,3000,4000,5000,6000,7500,10000]

class ReviewAlignedEntropyBet(EntropyBet):
    """Optional drop-in restart ablation, with inherited outward certificates.

    Create once before processing the first future observation. Initial audits
    are known bookkeeping and are not retroactively inserted into a new bet.
    This preserves validity, but validation shows no material late width gain.
    """
    def __init__(self,L,alpha,initial_audits,total_records,refresh_every=500):
        super().__init__(L,alpha,restarts=False)
        if not (0<=initial_audits<total_records and refresh_every>=1):
            raise ValueError('Invalid initial audits, population, or refresh frequency')
        first=(int(initial_audits)//int(refresh_every)+1)*int(refresh_every)
        self.starts=np.array([0]+[t-int(initial_audits) for t in range(first,int(total_records),int(refresh_every))])
        self.weights=np.array([np.floor(2.**40/((j+1)*(j+2)))*2.**-40 for j in range(len(self.starts))])

def atomic_json(path,value):
    path=Path(path); tmp=path.with_name(path.name+'.pending')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def cache(seed,max_audits=10000):
    path=OUT/f'adaptive_cache_{seed}.npz'
    if path.exists():
        with np.load(path) as p:
            if 'gap_prediction' in p and len(p['Ylo'])>=max_audits-int(p['pilot']):return {k:p[k] for k in p.files}
    pop=np.load(Path(__file__).with_name('replay_population.npz'))
    nr=pop['N_r'];L=len(pop['labels']);rng=np.random.default_rng(seed)
    arng=np.random.default_rng(seed+7121)
    orders=[rng.permutation(pop['v_label_index'][pop['y_row_index']==r]) for r in range(len(nr))]
    state=CompositionAudit(nr,pop['row_to_label_index'],L)
    pilot=np.minimum(nr,np.maximum(5,np.ceil(np.sqrt(nr)).astype(int)))
    for r in range(len(nr)):state.A[r]=np.bincount(orders[r][:pilot[r]],minlength=L)
    t=int(pilot.sum());state.update();pred=[];est=[];audits=[];gap_prediction=[]
    while t<max_audits:
        pred.append(float(np.log(state.estimate())))
        rem=state.N_r-state.A.sum(axis=1)
        ph=(state.A+rem[:,None]*(state.A+state.beta)/(state.A.sum(axis=1)+1)[:,None]).sum(axis=0)/state.N
        gap_prediction.append(max(0.,pred[-1]-state.bl-float(state.dl@ph)))
        r,pi,_=sample_dyadic_row(arng,state.probabilities(),state.N_r-state.A.sum(axis=1)>0)
        s=int(orders[r][int(state.A[r].sum())]);audits.append((r,s))
        state.observation(r,s,pi);t+=1
        if t%500==0:state.update()
        if t in CHECKPOINTS:est.append((t,state.estimate()))
    tl,tu,truth=census_hill(pop['C_star'].sum(axis=0),int(nr.sum()))
    ans=dict(Ylo=np.array(state.bet.Ylo),Yhi=np.array(state.bet.Yhi),
             minlo=np.array(state.bet.minlo),maxhi=np.array(state.bet.maxhi),
             prediction=np.array(pred),gap_prediction=np.array(gap_prediction),estimate=np.array(est),audits=np.array(audits),
             truth=np.array([tl,tu,truth]),pilot=np.array(pilot.sum()),L=np.array(L),seed=np.array(seed))
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**ans);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);return ans

def restart_spec(kind,pilot):
    if kind=='baseline':return np.array([0,750,2250,5250,11250,23250]),np.array([.5,.25,.125,.0625,.03125,.03125])
    # Restarts are fixed functions of review count, before data are observed.
    if kind=='aligned':
        starts=np.array([0]+[max(0,t-pilot) for t in [1000,1500,2000,3000,4500,6500,9500,14500,22000]])
    elif kind in ('dense','polynomial'):starts=np.array([0]+[max(0,t-pilot) for t in range(1000,33500,500)])
    else:raise ValueError(kind)
    # Exactly representable dyadic prior; total is exactly one.
    if kind=='polynomial':
        weights=np.array([np.floor(2.**40/((j+1)*(j+2)))*2.**-40 for j in range(len(starts))])
    else:
        weights=np.array([2.**(-j-1) for j in range(len(starts))]);weights[-1]=weights[-2]
    return starts,weights

def schedule(y,pred,mode,start,n,gaps=None,side='lo'):
    y=y[start:n];pred=pred[start:n];a=np.arange(1,len(y)+1,dtype=float)
    if mode.startswith('fixed') or mode.startswith('attenuate') or mode.startswith('skip'):
        grid=np.array([.01,.03,.1,.3,1.,3.,10.]) if mode!='fixed_compact' else np.array([.03,.1,.3,1.])
        kk=np.broadcast_to(grid,(len(y),len(grid)))
        if side=='lo' and mode.startswith('attenuate'):
            target=float(mode.split('_')[1]);kk=kk*(target/(target+gaps[start:n]))[:,None]
        if side=='lo' and mode.startswith('skip'):
            target=float(mode.split('_')[1]);kk=kk*(gaps[start:n]<=target)[:,None]
        return kk
    # Exponential smoothing is a tuning device only, with no variance claim.
    # Each variance value is computed BEFORE its own observation is revealed.
    v=1.;m=float(pred[0]) if len(pred) else 0.;vv=[]
    for yy,pp in zip(y,pred):
        vv.append(max(v,.0001))
        center=pp if 'pred' in mode else m
        v=.99*v+.01*(yy-center)**2;m=.99*m+.01*yy
    vv=np.array(vv)
    scale=np.sqrt(2*np.log(80)/(a*vv))
    cs=np.array([.25,.5,1.,2.]) if 'mix' in mode else np.array([.5])
    return scale[:,None]*cs[None,:]

def wealth(h,y,b,pred,side,starts,weights,mode,n,verified=False,kscache=None,gaps=None):
    epochs=[];ms=[];es=[]
    for start,weight in zip(starts,weights):
        s=min(int(start),n);yy=y[s:n];bb=b[s:n]
        if kscache is None:kk=schedule(y,pred,mode,s,n,gaps,side)
        else:
            key=(side,mode,s,n)
            if key not in kscache:kscache[key]=schedule(y,pred,mode,s,n,gaps,side)
            kk=kscache[key]
        if side=='lo':delta=yy-h;den=h-bb
        else:delta=h-yy;den=bb-h
        if not verified:
            cap=.9/np.maximum(den,1e-15)
            lf=np.log1p(np.minimum(kk,cap[:,None])*delta[:,None])
            epochs.append(np.log(weight)+logsumexp(lf.sum(axis=0))-np.log(kk.shape[1]))
            continue
        dd=_down(delta);dlo=np.maximum(_down(den),1e-15);dhi=np.maximum(_up(den),1e-15)
        cl=np.nextafter(np.float64(9)/10,0.);cu=np.nextafter(np.float64(9)/10,np.inf)
        lamlo=np.minimum(kk,_positive_down(cl/dhi)[:,None])
        lamhi=np.minimum(kk,_up(cu/dlo)[:,None])
        chosen=np.where(dd[:,None]<0,lamhi,lamlo)
        factor=np.maximum(_down(1+_down(chosen*dd[:,None])),np.nextafter(.1,0.))
        m,e=_scaled_product_lower(factor)
        wm,we=np.frexp(weight);m=_positive_down(m*wm);m=_positive_down(m/kk.shape[1])
        m,shift=np.frexp(m);e=e+we+shift;ms.extend(m);es.extend(e)
    return _scaled_sum_lower(ms,es) if verified else float(logsumexp(epochs))

def evaluate(data,name,restart,mode,verify=True):
    pilot=int(data['pilot']);L=int(data['L']);starts,weights=restart_spec(restart,pilot)
    logL=_fup(_I(L).log().hi);low=0.;high=logL;alpha=np.nextafter(.025,0.)
    threshold=np.log(2/alpha);target=_fup((_I(2)/_I(_D(alpha))).hi)
    result=[]
    for t,estimate in data['estimate']:
        t=int(t);n=t-pilot;end={};kscache={}
        for side in ['lo','hi']:
            y=data['Ylo' if side=='lo' else 'Yhi'];b=data['minlo' if side=='lo' else 'maxhi']
            w=lambda h:wealth(h,y,b,data['prediction'],side,starts,weights,mode,n,kscache=kscache,gaps=data['gap_prediction'])-threshold
            w0=w(0.);w1=w(logL)
            if side=='lo':value=0. if w0<=0 else (logL if w1>0 else brentq(w,0,logL))
            else:value=logL if w1<=0 else (0. if w0>0 else brentq(w,0,logL))
            value=max(0.,value-1e-8) if side=='lo' else min(logL,value+1e-8)
            if verify and ((side=='lo' and value>0) or (side=='hi' and value<logL)):
                if not wealth(value,y,b,data['prediction'],side,starts,weights,mode,n,True,kscache,data['gap_prediction']).exceeds(target):
                    value=0. if side=='lo' else logL
            end[side]=value
        low=max(low,end['lo']);high=min(high,end['hi'])
        lo=max(1.,_fdown(_exp(_I(_D(low))).lo));hi=min(float(L),_fup(_exp(_I(_D(high))).hi))
        result.append(dict(seed=int(data['seed']),variant=name,t=t,lo=lo,hi=hi,width=hi-lo,
            estimate=float(estimate),truth=float(data['truth'][2]),
            coverage=bool(lo<=data['truth'][0] and data['truth'][1]<=hi),
            numeric_verified=verify,alpha_coordinates=.020,alpha_entropy=.025,
            restart=restart,mode=mode,pilot=pilot))
    return result

VARIANTS={
    'baseline':('baseline','fixed'),
    'aligned_fixed':('aligned','fixed'),
    'dense_fixed':('dense','fixed'),
    'baseline_adaptive':('baseline','mean'),
    'aligned_adaptive':('aligned','mean'),
    'aligned_adaptive_mix':('aligned','mean_mix'),
    'aligned_prediction_mix':('aligned','pred_mix'),
    'polynomial_fixed':('polynomial','fixed'),
    'polynomial_compact':('polynomial','fixed_compact'),
    'baseline_attenuate02':('baseline','attenuate_0.02'),
    'baseline_attenuate05':('baseline','attenuate_0.05'),
    'baseline_skip05':('baseline','skip_0.05'),
    'baseline_skip10':('baseline','skip_0.10'),
    'baseline_skip20':('baseline','skip_0.20'),
}

def cell_kelly(data,cell_width=.005,scale=.5):
    """Diagnostic grid-cell-frozen predictable Kelly test; no root assumption.

    Cell [a,b] uses a predictable magnitude independent of h within that cell.
    Lower product is minimized at b; upper product is minimized at a. A cell
    is discarded only if at least one MINIMUM wealth exceeds 2/alpha. Thus
    truth remains in the surviving-cell hull on the joint event. Ordinary
    floating results are exploratory pending the outward product check below.
    """
    L=int(data['L']);logL=_fup(_I(L).log().hi);pilot=int(data['pilot'])
    edges=np.linspace(0,logL,int(np.ceil(logL/cell_width))+1)
    aa=edges[:-1];bb=edges[1:];mid=(aa+bb)/2
    mus={'lo':data['prediction']-data['gap_prediction'],'hi':data['prediction']}
    vars={}
    for side in ('lo','hi'):
        y=data['Ylo' if side=='lo' else 'Yhi'];v=1.;vs=[]
        for yy,mm in zip(y,mus[side]):
            vs.append(max(v,.0001));v=.99*v+.01*(yy-mm)**2
        vars[side]=np.array(vs)
    alive=np.ones(len(aa),bool);rows=[];alpha=np.nextafter(.025,0.)
    target=_fup((_I(2)/_I(_D(alpha))).hi);threshold=np.log(target)
    for t,estimate in data['estimate']:
        t=int(t);n=t-pilot
        for ix in range(0,len(aa),128):
            sl=slice(ix,ix+128);a=aa[sl];b=bb[sl];m=mid[sl]
            rejected=np.zeros(len(a),bool)
            for side in ('lo','hi'):
                yy=data['Ylo' if side=='lo' else 'Yhi'][:n,None]
                bound=data['minlo' if side=='lo' else 'maxhi'][:n,None]
                edge=(mus[side][:n,None]-m) if side=='lo' else (m-mus[side][:n,None])
                kk=scale*np.maximum(0.,edge)/(vars[side][:n,None]+edge**2)
                # Each represented kk and cap is predictable. The exact cap
                # uses the far cell endpoint, guaranteeing positivity over it.
                hh=b if side=='lo' else a
                delta=yy-hh if side=='lo' else hh-yy
                den=hh-bound if side=='lo' else bound-hh
                cap=.9/np.maximum(den,1e-15);lam=np.minimum(kk,cap)
                logw=np.log1p(lam*delta).sum(axis=0)
                proposals=logw>threshold
                if np.any(proposals):
                    dd=_down(delta[:,proposals]);dlo=np.maximum(_down(den[:,proposals]),1e-15)
                    dhi=np.maximum(_up(den[:,proposals]),1e-15)
                    cl=np.nextafter(np.float64(9)/10,0.);cu=np.nextafter(np.float64(9)/10,np.inf)
                    kl=kk[:,proposals]
                    lamlo=np.minimum(kl,_positive_down(cl/dhi));lamhi=np.minimum(kl,_up(cu/dlo))
                    chosen=np.where(dd<0,lamhi,lamlo)
                    factors=np.maximum(_down(1+_down(chosen*dd)),np.nextafter(.1,0.))
                    from directional_numeric import ScaledLower
                    mant,exp=_scaled_product_lower(factors)
                    checked=ScaledLower(mant,exp).exceeds(target)
                    rejected[np.flatnonzero(proposals)]=rejected[np.flatnonzero(proposals)]|checked
            alive[sl]&=~rejected
        indices=np.flatnonzero(alive)
        if len(indices):
            lo=max(1.,_fdown(_exp(_I(_D(aa[indices[0]]))).lo))
            hi=min(float(L),_fup(_exp(_I(_D(bb[indices[-1]]))).hi))
        else:lo,hi=1.,float(L)
        rows.append(dict(seed=int(data['seed']),variant=f'cell_kelly_{scale}',t=t,lo=lo,hi=hi,
            width=hi-lo,estimate=float(estimate),truth=float(data['truth'][2]),
            coverage=bool(lo<=data['truth'][0] and data['truth'][1]<=hi),
            numeric_verified=True,alpha_coordinates=.020,alpha_entropy=.025,
            pilot=pilot,cell_width=float(np.max(bb-aa)),empty=not len(indices)))
    return rows

def summarize_validation():
    """Twenty fresh paired paths, frozen polynomial restart schedule."""
    import hashlib
    seeds=list(range(91001,91021));rows=[]
    for seed in seeds:
        rows+=json.loads((OUT/f'adaptive_results_{seed}.json').read_text())
    methods=['baseline','polynomial_fixed']
    mapping={(r['seed'],r['variant'],r['t']):r for r in rows}
    expected={(s,m,t) for s in seeds for m in methods for t in CHECKPOINTS}
    if set(mapping)!=expected or len(rows)!=len(expected):raise ValueError('Missing or duplicate validation records')
    rng=np.random.default_rng(19093026);idx=rng.integers(0,len(seeds),size=(20000,len(seeds)))
    ans=dict(seeds=seeds,development_seed=1729,methods=methods,checkpoint_count=len(rows),
        coverage_count=sum(r['coverage'] for r in rows),numeric_verified=all(r['numeric_verified'] for r in rows),
        error_budget={'coordinates':.020,'entropy':.025,'joint_failure_bound':.045},
        scope='Same fixed population and catalogue; paired randomization paths; methods use exactly identical observed labels',
        widths={},stopping={},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for t in CHECKPOINTS:
        a=np.array([mapping[s,'baseline',t]['width'] for s in seeds])
        b=np.array([mapping[s,'polynomial_fixed',t]['width'] for s in seeds])
        boot=100*(1-b[idx].mean(axis=1)/a[idx].mean(axis=1))
        ans['widths'][str(t)]={'baseline':float(a.mean()),'polynomial':float(b.mean()),
            'percent_reduction':float(100*(1-b.mean()/a.mean())),
            'paired_bootstrap_95_percent_reduction':np.quantile(boot,[.025,.975]).tolist()}
    criteria={
        'full_width_1':lambda r:r['width']<=1,
        'full_width_0.5':lambda r:r['width']<=.5,
        'midpoint_relative_5pct':lambda r:r['width']<=.1*r['lo'],
        'plugin_abs_0.5':lambda r:max(r['estimate']-r['lo'],r['hi']-r['estimate'])<=.5,
        'plugin_abs_0.25':lambda r:max(r['estimate']-r['lo'],r['hi']-r['estimate'])<=.25,
    }
    for name,test in criteria.items():
        rec={}
        for m in methods:
            stops=[next((t for t in CHECKPOINTS if test(mapping[s,m,t])),None) for s in seeds]
            rec[m]={'audit_counts':stops,'mean':float(np.mean(stops)) if all(v is not None for v in stops) else None,
                'censored':sum(v is None for v in stops)}
        if all(rec[m]['mean'] is not None for m in methods):
            rec['mean_reviews_saved']=rec['baseline']['mean']-rec['polynomial_fixed']['mean']
        ans['stopping'][name]=rec
    for m in methods:
        ans[m+'_path_coverage']=sum(all(mapping[s,m,t]['coverage'] for t in CHECKPOINTS) for s in seeds)
    atomic_json(OUT/'adaptive_validation_summary.json',ans)
    return ans

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=1729)
    ap.add_argument('--max-audits',type=int,default=10000);ap.add_argument('--variants',nargs='+',default=list(VARIANTS))
    args=ap.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    start=time.perf_counter();data=cache(args.seed,args.max_audits);allrows=[]
    print(json.dumps({'seed':args.seed,'cached_seconds':time.perf_counter()-start}),flush=True)
    for name in args.variants:
        rows=cell_kelly(data,scale=float(name.split('_')[-1])) if name.startswith('cell_kelly_') else evaluate(data,name,*VARIANTS[name]);allrows+=rows
        print(json.dumps({'seed':args.seed,'variant':name,'widths':{r['t']:round(r['width'],6) for r in rows},'seconds':time.perf_counter()-start}),flush=True)
        atomic_json(OUT/f'adaptive_results_{args.seed}.json',allrows)

if __name__=='__main__':main()
