"""Pilot/online affine entropy envelopes and variance-directed scalar betting.

Experimental finite-population procedure, distinct from the original product
projection. The only complete labels occur in replay(); online code receives
known row sizes, submitted labels, and one revealed destination per review.
"""
from pathlib import Path
import sys, json, argparse, time
import numpy as np
from scipy.special import xlogy, logsumexp
from scipy.optimize import brentq
from ppr_convex import RowPPR, _I, _D, _isum, _fdown, _fup
from certified_entropy import _entropy_term, _logsumexp, _exp
from directional_numeric import wealth_lower
from directional_observations import outward_observation, sample_dyadic_row

def hill(p): return float(np.exp(-xlogy(p,p).sum()))

def census_hill(counts,N):
    H=_isum(_entropy_term(_I(int(v))/_I(int(N))) for v in counts if v)
    expH=_exp(H)
    return _fdown(expH.lo),_fup(expH.hi),float((expH.lo+expH.hi)/2)

def envelopes(N_r,A,beta,alpha,extra=None):
    """Audit-only predictable affine entropy envelopes, outward intercepts."""
    N=int(sum(N_r)); L=A.shape[1]; R=len(N_r)
    lower=[];upper=[]
    for r in range(R):
        row=RowPPR(N_r[r],A[r],beta[r],np.nextafter(alpha/R,0.))
        lo,hi=row.coordinate_bounds();lower.append(lo);upper.append(hi)
    lower=np.array(lower);upper=np.array(upper)
    ds=np.zeros(L);bs=_I(0);pl=[];pu=[]
    for s in range(L):
        cl=max(int(A[:,s].sum()),int(_isum(_I(_D(v)) for v in lower[:,s]).lo.to_integral_value(rounding='ROUND_CEILING')))
        cu=min(N,int(A[:,s].sum())+N-int(A.sum()),int(_isum(_I(_D(v)) for v in upper[:,s]).hi.to_integral_value(rounding='ROUND_FLOOR')))
        if cl>cu: return np.zeros(L),0.,np.zeros(L),_fup(_I(L).log().hi),True
        if extra is not None:
            cl=max(cl,int((_I(_D(extra[0][s]))*_I(N)).lo.to_integral_value(rounding='ROUND_CEILING')))
            cu=min(cu,int((_I(_D(extra[1][s]))*_I(N)).hi.to_integral_value(rounding='ROUND_FLOOR')))
        if cl>cu: return np.zeros(L),0.,np.zeros(L),_fup(_I(L).log().hi),True
        l=(_I(cl)/_I(N)).lo;u=(_I(cu)/_I(N)).hi
        hl=_entropy_term(_I(l));hu=_entropy_term(_I(u))
        if u>l: ds[s]=float((hu.lo-hl.lo)/(u-l))
        intercept=min((hl-_I(_D(ds[s]))*_I(l)).lo,(hu-_I(_D(ds[s]))*_I(u)).lo)
        bs+=_I(intercept);pl.append(float(l));pu.append(float(u))
    rem=N_r-A.sum(axis=1)
    ph=(A+rem[:,None]*(A+beta)/(A.sum(axis=1)+beta.sum(axis=1))[:,None]).sum(axis=0)/N
    au=-np.log(np.maximum(ph,1e-12));zu=_fup(_logsumexp(-au).hi)
    return ds,_fdown(bs.lo),au,zu,False

class EntropyBet:
    """Two monotone mixture e-values for composite entropy hypotheses.

    For lower-tail rejection, E[Ylo|past] <= H and minlo <= Ylo.
    For upper-tail rejection, E[Yhi|past] >= H and maxhi >= Yhi.
    Candidate-dependent capped bets remain predictable and monotone in h.
    """
    def __init__(self,L,alpha,restarts=False):
        self.L=L;self.logL=_fup(_I(L).log().hi);self.alpha=np.nextafter(alpha,0.);self.Ylo=[];self.Yhi=[]
        self.minlo=[];self.maxhi=[];self.ks=np.array([.01,.03,.1,.3,1.,3.,10.])
        self.low=0.;self.high=self.logL;self.crossed=False
        self.starts=np.array([0,750,2250,5250,11250,23250]) if restarts else np.array([0])
        self.weights=np.array([.5,.25,.125,.0625,.03125,.03125]) if restarts else np.array([1.])
    def add(self,yl,yu,ll,uu):
        if yl<ll-1e-10 or yu>uu+1e-10: raise ValueError('observation outside predictable range')
        self.Ylo.append(yl);self.Yhi.append(yu);self.minlo.append(ll);self.maxhi.append(uu)
    def wealth(self,h,side):
        if side=='lo':
            y=np.asarray(self.Ylo);b=np.asarray(self.minlo);delta=y-h;cap=.9/np.maximum(h-b,1e-15)
        else:
            y=np.asarray(self.Yhi);b=np.asarray(self.maxhi);delta=h-y;cap=.9/np.maximum(b-h,1e-15)
        lam=np.minimum(self.ks[:,None],cap[None,:])
        lf=np.log1p(lam*delta[None,:]);epoch=[]
        for start in self.starts:
            epoch.append(logsumexp(lf[:,start:].sum(axis=1))-np.log(len(self.ks)))
        return float(logsumexp(np.log(self.weights)+epoch))
    def interval(self):
        threshold=np.log(2/self.alpha)
        fl=lambda h:self.wealth(h,'lo')-threshold
        fu=lambda h:self.wealth(h,'hi')-threshold
        lo=0. if fl(0.)<=0 else (self.logL if fl(self.logL)>0 else brentq(fl,0,self.logL))
        hi=self.logL if fu(self.logL)<=0 else (0. if fu(0.)>0 else brentq(fu,0,self.logL))
        # Ordinary roots only propose rejecting endpoints. Every nontrivial
        # endpoint must pass an independent outward PRODUCT-wealth check.
        target=_fup((_I(2)/_I(_D(self.alpha))).hi)
        lo=max(0.,lo-1e-8);hi=min(self.logL,hi+1e-8)
        if lo>0 and not wealth_lower(lo,self.Ylo,self.Ylo,self.minlo,self.maxhi,self.ks,
            self.starts,self.weights,'lo',return_scaled=True).exceeds(target):lo=0.
        if hi<self.logL and not wealth_lower(hi,self.Yhi,self.Yhi,self.minlo,self.maxhi,self.ks,
            self.starts,self.weights,'hi',return_scaled=True).exceeds(target):hi=self.logL
        self.low=max(self.low,lo);self.high=min(self.high,hi)
        lower=max(1.,_fdown(_exp(_I(_D(self.low))).lo));upper=min(float(self.L),_fup(_exp(_I(_D(self.high))).hi))
        self.crossed = self.crossed or lower > upper
        return (lower,upper) if not self.crossed else (1.,float(self.L))

class CoordinateBet:
    """Simultaneous scalar CSs for column sums of the unknown matrix."""
    def __init__(self,L,alpha):
        self.L=L;self.alpha=np.nextafter(alpha,0.);self.y=[];self.y_upper=[];self.lower=[];self.upper=[]
        self.ks=np.array([.03,.1,.3,1.,3.,10.,30.,100.])
        self.lo=np.zeros(L);self.hi=np.ones(L)
    def add(self,y,lower,upper,yupper=None):
        self.y.append(y);self.y_upper.append(y if yupper is None else yupper)
        self.lower.append(lower);self.upper.append(upper)
    def bounds(self):
        if not self.y:return self.lo,self.hi
        y=np.asarray(self.y);yu=np.asarray(self.y_upper);low=np.asarray(self.lower);up=np.asarray(self.upper)
        threshold=np.log(2*self.L/self.alpha)
        def wealth(h,side):
            if side=='lo': delta=y-h;cap=.9/np.maximum(h-low,1e-15)
            else:delta=h-yu;cap=.9/np.maximum(up-h,1e-15)
            totals=[]
            for k in self.ks:
                totals.append(np.log1p(np.minimum(k,cap)*delta).sum(axis=0))
            return logsumexp(totals,axis=0)-np.log(len(self.ks))-threshold
        for side in ['lo','hi']:
            left=np.zeros(self.L);right=np.ones(self.L)
            w0=wealth(left,side);w1=wealth(right,side)
            for _ in range(32):
                mid=(left+right)/2;v=wealth(mid,side)
                move=v>0 if side=='lo' else v<=0
                left=np.where(move,mid,left);right=np.where(move,right,mid)
            proposal=np.maximum(0.,np.where(w0<=0,0.,left)-1e-8) if side=='lo' else np.minimum(1.,np.where(w1<=0,1.,right)+1e-8)
            target=_fup((_I(2*self.L)/_I(_D(self.alpha))).hi)
            verified=wealth_lower(proposal,y,yu,low,up,self.ks,side=side,return_scaled=True).exceeds(target)
            if side=='lo':self.lo=np.maximum(self.lo,np.where(verified,proposal,0.))
            else:self.hi=np.minimum(self.hi,np.where(verified,proposal,1.))
        return self.lo,self.hi

class DirectionalAudit:
    def __init__(self,N_r,row_to_label,L,alpha_matrix=.025,alpha_bet=.025,explore=.5,pooled_coordinates=False,restarts=False):
        self.N_r=np.asarray(N_r,int);self.N=int(sum(N_r));self.R=len(N_r);self.L=L
        self.row_to_label=np.asarray(row_to_label,int);self.A=np.zeros((self.R,L),int)
        self.beta=np.full((self.R,L),.5/(L-1));self.beta[np.arange(self.R),row_to_label]=.5
        self.alpha_matrix=np.nextafter(alpha_matrix if not pooled_coordinates else .005,0.)
        self.bet=EntropyBet(L,alpha_bet,restarts=restarts);self.explore=explore
        self.coordinates=CoordinateBet(L,.02) if pooled_coordinates else None
        self.dl=np.zeros(L);self.bl=0.;self.du=np.zeros(L);self.bu=_fup(_I(L).log().hi)
        self.sd=np.ones(self.R);self.bad=False
    def update(self):
        extra=self.coordinates.bounds() if self.coordinates is not None else None
        self.dl,self.bl,self.du,self.bu,self.bad=envelopes(self.N_r,self.A,self.beta,self.alpha_matrix,extra)
        pred=(self.A+self.beta)/(self.A.sum(axis=1)+1)[:,None]
        # Variance allocation is directed to the entropy upper affine score.
        means=pred@self.du;var=pred@(self.du**2)-means**2
        self.sd=np.sqrt(np.maximum(var,1e-12))
    def probabilities(self):
        rem=self.N_r-self.A.sum(axis=1);base=rem/rem.sum()
        opt=rem*self.sd;opt/=opt.sum()
        return self.explore*base+(1-self.explore)*opt
    def observation(self,r,s,pi):
        pred=(self.A+self.beta)/(self.A.sum(axis=1)+1)[:,None]
        obs=outward_observation(self.N_r,self.A,pred,self.dl,self.bl,self.du,self.bu,pi,r,s,
            coordinates=self.coordinates is not None)
        self.bet.add(obs['entropy_ylo'],obs['entropy_yhi'],obs['entropy_lower'],obs['entropy_upper'])
        if self.coordinates is not None:
            self.coordinates.add(obs['coordinate_ylo'],obs['coordinate_lower'],obs['coordinate_upper'],obs['coordinate_yhi'])
        self.A[r,s]+=1
    def estimate(self):
        rem=self.N_r-self.A.sum(axis=1)
        C=self.A+rem[:,None]*(self.A+self.beta)/(self.A.sum(axis=1)+1)[:,None]
        return hill(C.sum(axis=0)/self.N)
    def partial(self):
        C=self.A.copy();C[np.arange(self.R),self.row_to_label]+=self.N_r-self.A.sum(axis=1)
        return hill(C.sum(axis=0)/self.N)

def replay(seed,goals,out,explore=.5,pooled_coordinates=False,restarts=False,max_audits=None):
    pop=np.load(Path(__file__).with_name('replay_population.npz'),allow_pickle=False)
    N_r=pop['N_r'];L=len(pop['labels']);N=int(sum(N_r));R=len(N_r)
    rng=np.random.default_rng(seed);arng=np.random.default_rng(seed+7121)
    orders=[rng.permutation(pop['v_label_index'][pop['y_row_index']==r]) for r in range(R)]
    state=DirectionalAudit(N_r,pop['row_to_label_index'],L,explore=explore,pooled_coordinates=pooled_coordinates,restarts=restarts)
    pilot=np.minimum(N_r,np.maximum(5,np.ceil(np.sqrt(N_r)).astype(int)))
    for r in range(R):
        state.A[r]=np.bincount(orders[r][:pilot[r]],minlength=L)
    t=int(sum(pilot));start=time.perf_counter();state.update()
    ptrue=pop['C_star'].sum(axis=0)/N;true_lo,true_hi,true=census_hill(pop['C_star'].sum(axis=0),N);H=-float(xlogy(ptrue,ptrue).sum())
    def record():
        if t==N:lo,hi,_=census_hill(state.A.sum(axis=0),N)
        elif t==int(sum(pilot)) or state.bad: lo,hi=1.,float(L)
        else: lo,hi=state.bet.interval()
        row=dict(seed=seed,t=t,lo=lo,hi=hi,estimate=state.estimate(),partial=state.partial(),
            true=true,width=hi-lo,coverage=lo<=true_lo and true_hi<=hi,lower_gap=H-state.bl-float(state.dl@ptrue),
            upper_gap=state.bu+float(state.du@ptrue)-H,seconds=time.perf_counter()-start,
            bad=state.bad,explore=explore,pooled_coordinates=pooled_coordinates,restarts=restarts,numeric_verified=True)
        out.write(json.dumps(row)+'\n');out.flush();print(json.dumps(row),flush=True)
    record()
    goals=sorted(set([v for v in goals if v>t]+[N]))
    if max_audits:goals=sorted(set([v for v in goals if v<=max_audits]+[min(N,max_audits)]))
    for goal in goals:
        while t<goal:
            desired=state.probabilities()
            r,pi,_=sample_dyadic_row(arng,desired,state.N_r-state.A.sum(axis=1)>0)
            u=int(state.A[r].sum())
            state.observation(r,int(orders[r][u]),pi);t+=1
            if t%500==0 and t<N:state.update()
        record()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=1);ap.add_argument('--seed-start',type=int,default=1729)
    ap.add_argument('--output',default='output/efficiency/directional.jsonl');ap.add_argument('--explore',type=float,default=.5)
    ap.add_argument('--pooled-coordinates',action='store_true')
    ap.add_argument('--restarts',action='store_true');ap.add_argument('--max-audits',type=int)
    ap.add_argument('--checkpoints',nargs='+',type=int,default=[1000,1500,2500,5000,7500,10000,12500,15000,16702,17500,20000,25000,30000])
    args=ap.parse_args();Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,'w') as out:
        for seed in range(args.seed_start,args.seed_start+args.seeds):replay(seed,args.checkpoints,out,args.explore,args.pooled_coordinates,args.restarts,args.max_audits)
