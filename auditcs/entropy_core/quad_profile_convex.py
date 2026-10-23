"""Convex direct lower-entropy profile, with candidate-independent scalar CV.

X>=0 is an HT vector with conditional mean p. K is any predictable-model
zero-mean scalar control. D>=max_possible(sum(X)-K) is predictable. For a
candidate entropy h, smooth q(p)=(p+1/N)/(1+L/N), m=h+log(1+L/N), and use
lambda_tk=min(k,.875/(m+D_t)). The log factors are globally convex in p,
nonincreasing in h, and define a null e-process at every true p with H(p)<=h.
The convex profile over the simplex therefore gives an anytime-valid lower
entropy test. No entropy chord or coordinate CS is used.

Ordinary numerical optimizer proposals and supporting-plane gaps are research
diagnostics. An independent directed evaluation must certify final endpoints.
"""
from pathlib import Path
import argparse,os,json,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,xlogy
from certified_entropy import _exp
from ppr_convex import _I,_D,_fdown
ROOT=Path(__file__).resolve().parents[1]

class ConvexEntropyProfile:
    def __init__(self,X,K,D,N,ks=(.03,.1,.3,1.),alpha=.025,eta=.875):
        self.X=np.asarray(X);self.K=np.asarray(K);self.D=np.asarray(D)
        self.N=int(N);self.L=self.X.shape[1];self.eps=1/self.N;self.c=np.log1p(self.L/self.N)
        self.ks=np.array(ks);self.alpha=alpha;self.eta=float(eta);self.threshold=np.log(1/alpha)
        if self.X.min() < -1e-15:raise ValueError('Positive HT coefficients required')
        if len(self.K)!=len(self.X) or len(self.D)!=len(self.X):raise ValueError('Stream mismatch')
        self.last=None
    def at(self,h):
        self.h=float(h);self.m=h+self.c;self.lam=np.minimum(self.ks[None,:],self.eta/(self.m+self.D[:,None]))
        self.last=None;return self
    def calc(self,p):
        if self.last is not None and np.array_equal(self.last[0],p):return self.last[1:]
        d=-np.log((p+self.eps)/(1+self.L*self.eps));Y=self.X@d+self.K
        f=1+self.lam*(Y[:,None]-self.m)
        if f.min()<=0:raise ArithmeticError('Invalid factor')
        lw=np.log(f).sum(axis=0);val=logsumexp(lw)-np.log(len(lw));mix=np.exp(lw-logsumexp(lw))
        allgrad=-(self.X.T@(self.lam/f))/(p[:,None]+self.eps)
        grad=allgrad@mix
        self.last=(p.copy(),float(val),grad,float(f.min()))
        return self.last[1:]
    def value(self,p):return self.calc(p)[0]
    def grad(self,p):return self.calc(p)[1]
    def solve(self,h,lower=None,start=None,maxiter=500):
        self.at(h);L=self.L;lower=np.zeros(L) if lower is None else np.asarray(lower,float)
        active=np.sum(self.X,axis=0)>0
        # Coordinates with zero coefficient in EVERY factor have zero lower
        # bound. Removing their mass can only decrease the objective.
        active|=lower>0
        inds=np.flatnonzero(active)
        if start is None:
            start=lower+(1-lower.sum())*self.X.sum(axis=0)/self.X.sum()
        else:
            start=np.maximum(np.asarray(start),lower)
            start=lower+(1-lower.sum())*(start-lower)/max(np.sum(start-lower),1e-100)
        start[~active]=0.;start=lower+(1-lower.sum())*(start-lower)/max(np.sum(start-lower),1e-100)
        def full(v):p=np.zeros(L);p[inds]=v;return p
        fit=minimize(lambda v:self.value(full(v)),start[inds],jac=lambda v:self.grad(full(v))[inds],
                     method='SLSQP',bounds=[(lower[j],1.) for j in inds],
                     constraints=[{'type':'eq','fun':lambda v:np.sum(v)-1,'jac':lambda v:np.ones(len(v))}],
                     options={'ftol':1e-11,'maxiter':maxiter})
        p=full(fit.x);f,g,mf=self.calc(p)
        # This supporting-plane bound is global for all p in the original
        # simplex, including removed coordinates, irrespective of fit.success.
        affine=f-g@p+g@lower+(1-lower.sum())*g.min()
        return dict(h=float(h),value=f,affine_lower=float(affine),duality_gap=float(f-affine),
                    p=p.tolist(),iterations=int(fit.nit),success=bool(fit.success),minfactor=mf)
    def endpoint(self,lower=None,start=None,htol=2e-4,maxiter=500):
        l=0.;u=np.log(self.L);history=[];best=None
        while u-l>htol:
            h=.5*(l+u);ans=self.solve(h,lower,start,maxiter);start=np.array(ans['p']);history.append(ans)
            if ans['affine_lower']>self.threshold:
                l=h;best=ans
            else:u=h
        return dict(entropy_lower=l,pillar_lower=float(np.exp(l)),h_resolution=htol,
                    threshold=self.threshold,alpha=self.alpha,certification='floating supporting-plane bound; needs independent outward certificate',
                    best=best,history=history)

def run(cache,out=None,goals=(1500,2000,2500,3000,4000,5000),alpha=.025,eta=.875):
    z=np.load(cache);N=int(z['N']);L=int(z['L']);pilot=int(z['pilot'])
    X=z['positiveX'] if 'positiveX' in z else z['X'];K=z['K'];D=z['D']
    snaps=json.loads(str(z['snapshots']));res=[]
    for snap in snaps:
        t=snap['t']
        if t not in goals:continue
        n=t-pilot;A=np.array(snap['A']);lower=A.sum(0)/N
        m=ConvexEntropyProfile(X[:n],K[:n],D[:n],N,alpha=alpha,eta=eta)
        now=time.perf_counter();a=m.endpoint(lower=lower,htol=3e-4)
        # Independently certify the real-valued convex family. Any failed
        # numerical certificate only widens the proposed lower endpoint.
        from direct_convex_numeric import certify
        proposed_h=a['entropy_lower'];h=proposed_h;step=3e-4;attempts=[]
        point=np.array(a['best']['p']) if a['best'] is not None else lower+(1-lower.sum())/L
        while h>0:
            cert=certify(X[:n],K[:n],D[:n],N,point,A.sum(0),h,alpha=alpha,eta=eta)
            attempts.append({'h':h,'certified':cert['certified'],'log_profile_lower':cert['log_profile_lower']})
            if cert['certified']:break
            h=max(0.,h-step);step*=2
        a.update(proposed_entropy_lower=proposed_h,entropy_lower=h,pillar_lower=_fdown(_exp(_I(_D(float(h)))).lo),
                 numeric_certificate=cert if h>0 else None,certificate_attempts=attempts,
                 certification='independent outward supporting-plane certificate' if h>0 else 'deterministic H>=0',
                 t=t,pilot=pilot,n=n,eta=eta,seconds=time.perf_counter()-now)
        res.append(a);print(json.dumps({k:a[k] for k in ['t','pillar_lower','seconds']}),flush=True)
    dest=Path(out or ROOT/'output/optimal_revision/quad_profile_convex.json');tmp=dest.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump({'input':str(cache),'records':res},f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,dest)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cache');p.add_argument('--out');p.add_argument('--goals',nargs='*',type=int,default=[1500,2000,2500,3000,4000,5000]);p.add_argument('--alpha',type=float,default=.025);p.add_argument('--eta',type=float,default=.875)
    a=p.parse_args();run(a.cache,a.out,a.goals,a.alpha,a.eta)
