"""Audit-only positive-HT profile cache with inference from review one.

This is a retrospective replay driver. The profile inputs are nonnegative
downward HT coefficients X, downward zero-mean scalar corrections K, and
predictable upward bounds D on coefficient mass minus correction. They never
use the completed population except to simulate the single selected reveal.
The accompanying upper entropy stream is globally valid and independently
certified at one-sided alpha=.0225. Coordinates guide prediction/allocation
only; the new two-sided P2 argument does not condition on their coverage.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,os,time,hashlib
import numpy as np
from composition_audit import CompositionAudit
from efficiency_directional import census_hill,sample_dyadic_row,EntropyBet,_fup,_logsumexp
from metadata_prediction_research import fit_boosted_metadata_scores
from metadata_entropy_audit import control_observation
from directional_observations import _point,_add,_sub,_mul,_div,_sum,_finite
from profile_upper import upper_endpoint
from smallpilot_research import floor_variance_allocation

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/optimal_revision'
CHECKPOINTS=[100,200,300,500,750,1000,1500,2000,2500,3000,4000,5000]

def projected_upper_direction(state):
    """Positive variational reference from past vector observations only."""
    L=state.L
    if not state.coordinates.y:
        ph=np.bincount(state.row_to_label,weights=state.N_r,minlength=L)/state.N
    else:
        ph=(np.mean(state.coordinates.y,axis=0)+np.mean(state.coordinates.y_upper,axis=0))/2
        z=np.sort(ph)[::-1];cs=np.cumsum(z)-1;rho=np.flatnonzero(z-cs/np.arange(1,L+1)>0)[-1]
        ph=np.maximum(ph-cs[rho]/(rho+1),0)
    q=(state.N*ph+1)/(state.N+L)
    d=-np.log(q);b=_fup(_logsumexp(-d).hi)
    return d,b

def positive_observation(state,r,s,pi,c_i,c_total,c_max):
    """Return outward X,K,D using candidate-independent b_i=base_r+c_i.

    base_r is the current audit-only prediction of the represented upper-score
    direction. Additions base_r+c_i are enclosed, not silently rounded and
    then treated inconsistently in the remaining-record sum.
    """
    N=state.N;A=state.A;u=A.sum(1);M=state.N_r-u;active=M>0
    Q=(A+state.beta)/(u+1)[:,None];base=Q@state.du
    nn=_point(float(N));mm=_point(M.astype(float))
    ratio=_div(_div(mm,nn),_point(np.where(active,pi,1.)))
    ratio=(np.where(active,ratio[0],0.),np.where(active,ratio[1],0.))
    rr=(ratio[0][r],ratio[1][r]);one=np.zeros(state.L);one[s]=1.
    X=_add(_div(_point(A.sum(0).astype(float)),nn),_mul(rr,_point(one)))
    S=_add(_sum(_mul(mm,_point(base))),c_total)
    K=_sub(_div(S,nn),_mul(rr,_add(_point(float(base[r])),_point(float(c_i)))))
    maxb=_add(_point(base),_point(c_max))
    minK=_sub(_div(S,nn),_mul(ratio,maxb))
    mass=_add(_div(_point(float(A.sum())),nn),ratio)
    D=float(np.max(_sub(mass,minK)[1][active]))
    xl=np.maximum(0.,X[0]);xu=X[1];kl=float(K[0]);ku=float(K[1])
    _finite(xl,xu,kl,ku,D)
    if np.sum(xl)-kl>D+1e-10:raise ArithmeticError('Realized profile exceeds predictable cap')
    return xl,xu,kl,ku,D

def replay(seed=1729,allocation='residual',max_audits=5000,tag='development',
           fit_at=(200,500,1000,2000,3000),upper_direction='row',upper_start=0):
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    pop=np.load(ROOT/'work/replay_population.npz');nr=pop['N_r'];row=pop['y_row_index'];labels=pop['v_label_index']
    N=len(row);L=len(pop['labels']);R=len(nr)
    rng=np.random.default_rng(seed);arng=np.random.default_rng(seed+7121)
    orders=[rng.permutation(np.flatnonzero(row==r)) for r in range(R)]
    st=CompositionAudit(nr,pop['row_to_label_index'],L);st.update()
    if upper_direction not in ('row','projectedht'):raise ValueError(upper_direction)
    if upper_start not in (0,200):raise ValueError('Development tested only upper starts0and200')
    upper_starts=(upper_start,750,2250,5250,11250,23250)
    preserve_projected=upper_direction=='projectedht' or tag=='development'
    alt_du,alt_bu=projected_upper_direction(st) if preserve_projected else (None,None)
    old_metadata=EntropyBet(L,.025,restarts=True)
    c=np.zeros(N);c_total=_point(0.);mins=[np.zeros(len(o)) for o in orders];maxs=[np.zeros(len(o)) for o in orders]
    seen=[];xs=[];xu=[];ks=[];ku=[];ds=[];ys=[];ubs=[];pys=[];pubs=[];snapshots=[];rs=[];ss=[];pis=[]
    truth=np.array(census_hill(pop['C_star'].sum(0),N))
    upper_intersection=float(L);upper_entropy=np.log(float(L))
    for t in range(1,max_audits+1):
        u=st.A.sum(1);M=nr-u;active=M>0;desired=st.probabilities()
        if allocation in ('residual','waterfill') and t>fit_at[0]:
            ids=np.asarray(seen);rr=row[ids];vv=labels[ids]
            base=(st.A+st.beta)/(u+1)[:,None]@st.du
            residual=st.du[vv]-base[rr]-c[ids]
            second=np.bincount(rr,weights=residual**2,minlength=R)
            qhat=(second+10*st.sd**2)/(u+10)
            a=M*np.sqrt(np.maximum(qhat,1e-12))
            desired=floor_variance_allocation(M,a) if allocation=='waterfill' else .5*M/M.sum()+.5*a/a.sum()
        elif allocation not in ('original','residual','waterfill'):raise ValueError(allocation)
        r,pi,_=sample_dyadic_row(arng,desired,active)
        i=int(orders[r][u[r]]);s=int(labels[i])
        cmin=np.array([mins[j][u[j]] if active[j] else 0. for j in range(R)])
        cmax=np.array([maxs[j][u[j]] if active[j] else 0. for j in range(R)])
        xlo,xhi,klo,khi,dup=positive_observation(st,r,s,pi,c[i],c_total,cmax)
        ob=control_observation(st,r,s,pi,c[i],c_total,cmin,cmax)
        old_metadata.add(ob['entropy_ylo'],ob['entropy_yhi'],ob['entropy_lower'],ob['entropy_upper'])
        if preserve_projected:
            upst=SimpleNamespace(N=st.N,N_r=st.N_r,A=st.A,beta=st.beta,dl=st.dl,bl=st.bl,du=alt_du,bu=alt_bu)
            pob=control_observation(upst,r,s,pi,c[i],c_total,cmin,cmax)
            pys.append(pob['entropy_yhi']);pubs.append(pob['entropy_upper'])
        upper_ob=pob if upper_direction=='projectedht' else ob
        xs.append(xlo);xu.append(xhi);ks.append(klo);ku.append(khi);ds.append(dup)
        ys.append(upper_ob['entropy_yhi']);ubs.append(upper_ob['entropy_upper']);rs.append(r);ss.append(s);pis.append(pi)
        seen.append(i);c_total=_sub(c_total,_point(float(c[i])))
        st.observation(r,s,pi)
        if (t<=1000 and (t%100==0 or t==750)) or t%500==0:
            st.update()
            if preserve_projected:alt_du,alt_bu=projected_upper_direction(st)
        if t in fit_at:
            ids=np.asarray(seen);fit=fit_boosted_metadata_scores(ids,labels[ids]);c=np.asarray(fit['correction'],float)
            remaining=np.ones(N,bool);remaining[ids]=False;c_total=_sum(_point(c[remaining]))
            mins=[np.minimum.accumulate(c[o][::-1])[::-1] for o in orders]
            maxs=[np.maximum.accumulate(c[o][::-1])[::-1] for o in orders]
        if t in CHECKPOINTS:
            cert=upper_endpoint(ys,ubs,L,alpha=.0225,starts=upper_starts)
            cert0=upper_endpoint(ys,ubs,L,alpha=.0225) if upper_start else cert
            upper_intersection=min(upper_intersection,cert['pillar_upper']);upper_entropy=min(upper_entropy,cert['entropy_upper'])
            # The existing projected-vector estimate is an evaluation
            # diagnostic only; averaging positive X itself need not sum to1.
            ph=(np.mean(st.coordinates.y,axis=0)+np.mean(st.coordinates.y_upper,axis=0))/2
            q=np.sort(ph)[::-1];css=np.cumsum(q)-1;rho=np.flatnonzero(q-css/np.arange(1,L+1)>0)[-1]
            ph=np.maximum(ph-css[rho]/(rho+1),0);ht=float(np.exp(-np.sum(ph[ph>0]*np.log(ph[ph>0]))))
            oldlo,oldhi=st.bet.interval();cvlo,cvhi=old_metadata.interval()
            snap={'t':t,'A':st.A.tolist(),'estimate':st.estimate(),'partial':st.partial(),'ht_estimate':ht,
                  'upper':upper_intersection,'pillar_upper':upper_intersection,'entropy_upper':upper_entropy,'upper_numeric_verified':cert['numeric_verified'],
                  'old_no_control_lower':oldlo,'old_no_control_upper':oldhi,
                  'old_chord_control_lower':cvlo,'old_chord_control_upper':cvhi,
                  'old_no_control_crossed':st.bet.crossed,'old_chord_control_crossed':old_metadata.crossed,
                  'upper_start0':cert0['pillar_upper'],'upper_start0_verified':cert0['numeric_verified'],
                  'seconds':time.perf_counter()-started,'bad_planning_coordinates':st.bad}
            snapshots.append(snap)
            print(json.dumps({'seed':seed,'allocation':allocation,'t':t,'upper':upper_intersection,'seconds':snap['seconds']}),flush=True)
    suffix='' if upper_direction=='row' else '_'+upper_direction
    if upper_start:suffix+=f'_start{upper_start}'
    path=OUT/f'convex_profile_{tag}_{seed}_{allocation}{suffix}.npz';tmp=path.with_name(path.name+'.pending')
    data=dict(X=np.array(xs),X_upper=np.array(xu),K=np.array(ks),K_upper=np.array(ku),D=np.array(ds),
              N=N,L=L,pilot=0,seed=seed,truth=truth,snapshots=json.dumps(snapshots),audit_ids=np.array(seen),
              r=np.array(rs),s=np.array(ss),pi=np.array(pis),Yhi=np.array(ys),maxhi=np.array(ubs),
              Ylo=np.asarray(old_metadata.Ylo),minlo=np.asarray(old_metadata.minlo),
              baseline_Ylo=np.asarray(st.bet.Ylo),baseline_Yhi=np.asarray(st.bet.Yhi),
              baseline_minlo=np.asarray(st.bet.minlo),baseline_maxhi=np.asarray(st.bet.maxhi),
              row_Yhi=np.asarray(old_metadata.Yhi),row_maxhi=np.asarray(old_metadata.maxhi),
              predictor='candidate-independent current row upper-score prediction plus predictable metadata correction',
              training_schedule=json.dumps(list(fit_at)),allocation=allocation,alpha_lower=.0225,alpha_upper=.0225,
              upper_direction=upper_direction,
              upper_starts=json.dumps(list(upper_starts)),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if preserve_projected:data.update(projectedht_Yhi=np.asarray(pys),projectedht_maxhi=np.asarray(pubs))
    with tmp.open('wb') as f:np.savez_compressed(f,**data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    print(json.dumps({'path':str(path),'reviews':max_audits,'Xmin':float(np.min(xs)),'Dmin':float(min(ds)),
                      'upper_verified':all(s['upper_numeric_verified'] for s in snapshots)}),flush=True)
    return path

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=1729)
    ap.add_argument('--allocation',default='residual');ap.add_argument('--max-audits',type=int,default=5000)
    ap.add_argument('--tag',default='development')
    ap.add_argument('--upper-direction',default='row')
    ap.add_argument('--upper-start',type=int,default=0);a=ap.parse_args()
    replay(a.seed,a.allocation,a.max_audits,a.tag,upper_direction=a.upper_direction,upper_start=a.upper_start)
