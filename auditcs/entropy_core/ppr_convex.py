"""Certified outer linear optimization of categorical finite-population PPR sets.

Original paper notation: N=N_r, A=A_{r,u}, beta=beta_r, alpha=alpha_r.
``support(d)`` bounds sup_q d @ q over the *integer* raw PPR set, via
its convex real relaxation. ``update`` retains valid supporting halfspaces
from earlier times, hence bounds the running intersection as well.

Numerical certificate: convex supporting planes are enclosed using Decimal
interval arithmetic and rigorous positive-real Stirling remainder bounds;
LP duals are recomputed with directed rounding. SLSQP is only a proposal.
No statistical decision trusts its termination status or objective value.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, localcontext
from functools import lru_cache
import math
import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import minimize, linprog, brentq

_PREC = 42
_DOWN = Context(prec=_PREC, rounding=ROUND_FLOOR)
_UP = Context(prec=_PREC, rounding=ROUND_CEILING)


def _D(x):
    return x if isinstance(x, Decimal) else (Decimal(int(x)) if isinstance(x, (int, np.integer)) else Decimal.from_float(float(x)))


class _I:
    """Closed Decimal interval; elementary operations round outward."""
    __slots__ = ('lo', 'hi')
    def __init__(self, lo, hi=None):
        self.lo, self.hi = _D(lo), _D(lo if hi is None else hi)
    def __add__(self, v):
        v = v if isinstance(v, _I) else _I(v)
        return _I(_DOWN.add(self.lo, v.lo), _UP.add(self.hi, v.hi))
    __radd__ = __add__
    def __neg__(self):
        return _I(self.hi.copy_negate(), self.lo.copy_negate())
    def __sub__(self, v): return self + -(v if isinstance(v, _I) else _I(v))
    def __rsub__(self, v): return (v if isinstance(v, _I) else _I(v)) + -self
    def __mul__(self, v):
        v = v if isinstance(v, _I) else _I(v)
        return _I(min(_DOWN.multiply(a,b) for a in (self.lo,self.hi) for b in (v.lo,v.hi)),
                  max(_UP.multiply(a,b) for a in (self.lo,self.hi) for b in (v.lo,v.hi)))
    __rmul__ = __mul__
    def __truediv__(self, v):
        v = v if isinstance(v, _I) else _I(v)
        if v.lo <= 0 <= v.hi: raise ZeroDivisionError('interval contains zero')
        return self * _I(_DOWN.divide(Decimal(1),v.hi), _UP.divide(Decimal(1),v.lo))
    def __pow__(self, n):
        if not isinstance(n,int) or n < 0: raise ValueError('nonnegative integer power required')
        if self.lo < 0: raise ValueError('positive interval required')
        # Binary exponentiation uses only the elementary correctly rounded
        # multiplications already covered by this interval implementation.
        result=_I(1);base=self
        while n:
            if n&1: result=result*base
            n//=2
            if n: base=base*base
        return result
    def log(self):
        if self.lo <= 0: raise ValueError('log of nonpositive interval')
        # Decimal ln is documented correctly rounded (ROUND_HALF_EVEN).
        # Expanding by a full ulp handles that documented rounding mode.
        a,b = _DOWN.ln(self.lo), _UP.ln(self.hi)
        ea = Decimal(1).scaleb(a.adjusted()-_PREC+1)
        eb = Decimal(1).scaleb(b.adjusted()-_PREC+1)
        return _I(_DOWN.subtract(a,ea), _UP.add(b,eb))


_HALF_LOG_2PI = _I(Decimal('0.91893853320467274178032973640561763986139747363778'),
                  Decimal('0.91893853320467274178032973640561763986139747363779'))
_BERNOULLI = [(1,6),(-1,30),(1,42),(-1,30),(5,66),(-691,2730),(7,6),(-3617,510),(43867,798)]


@lru_cache(maxsize=65536)
def _gamma_psi(x: Decimal):
    """Outward intervals for log Gamma(x) and psi(x), x>0.

    Shift to x>=16, use eight Stirling terms, then enclose the remainder
    between zero and the ninth term, valid on the positive real axis.
    These expansions follow the standard Stieltjes remainder inequalities.
    """
    if x <= 0: raise ValueError('Gamma argument must be positive')
    z = _I(x)
    logshift, psishift = _I(0), _I(0)
    while z.lo < 16:
        logshift += z.log()
        psishift += _I(1)/z
        z += 1
    lg = (z-_I(Decimal('0.5')))*z.log()-z+_HALF_LOG_2PI
    psi = z.log()-_I(1)/(2*z)
    for k,(num,den) in enumerate(_BERNOULLI[:8],1):
        lg += _I(num)/(_I(den*(2*k)*(2*k-1))*(z**(2*k-1)))
        psi -= _I(num)/(_I(den*(2*k))*(z**(2*k)))
    k=9; num,den=_BERNOULLI[8]
    rem_g = _I(num)/(_I(den*(2*k)*(2*k-1))*(z**(2*k-1)))
    rem_p = _I(num)/(_I(den*(2*k))*(z**(2*k)))
    lg += _I(0,rem_g.hi)
    psi += _I(rem_p.hi.copy_negate(),0)
    return lg-logshift, psi-psishift


def _fdown(x):
    r=float(x)
    if Decimal.from_float(r)>x: r=math.nextafter(r,-math.inf)
    return r


def _fup(x):
    r=float(x)
    if Decimal.from_float(r)<x: r=math.nextafter(r,math.inf)
    return r


def _isum(xs):
    y=_I(0)
    for x in xs: y += x
    return y


@dataclass
class SupportResult:
    upper: float
    proposal: np.ndarray | None
    nonlinear_success: bool
    cuts: int
    method: str


class RowPPR:
    """Reusable row oracle. Counts and working-prior parameters are public.

    Parameters
    ----------
    N : int
        Known complete row size.
    A : length-L integer vector
        Observed audited counts, not unobserved ground truth.
    beta : positive scalar or length-L vector
        Fixed working Dirichlet parameters.
    alpha : float in (0,1)
        Predeclared row error budget.
    max_cuts : int
        Number of additional LP-point cuts if SLSQP's first cut is loose.
    """
    def __init__(self, N, A, beta=.5, alpha=.05, max_cuts=3):
        self.N=int(N)
        self.A=np.asarray(A,dtype=np.int64).copy()
        self.L=len(self.A)
        self.beta=np.broadcast_to(np.asarray(beta,dtype=float),(self.L,)).copy()
        self.alpha=float(alpha)
        self.max_cuts=int(max_cuts)
        if self.N<0 or np.any(self.A<0) or self.A.sum()>self.N: raise ValueError('invalid counts')
        if np.any(self.beta<=0) or not np.isfinite(self.beta).all(): raise ValueError('beta must be positive finite')
        if not 0<self.alpha<1: raise ValueError('alpha must be in (0,1)')
        self._G=[] # valid inequalities in unshifted q: G @ q <= b
        self._b=[]
        self._coordinate_cache=None
        self.last_result=None
        self._setup()

    def _setup(self):
        self.u=int(self.A.sum()); self.M=self.N-self.u
        self.active=np.flatnonzero(self.A)
        self.unseen=np.flatnonzero(self.A==0)
        with localcontext(Context(prec=1600)):
            Bd=sum((_D(v) for v in self.beta),Decimal(0))
            Bplusu=Bd+_D(self.u)
        self._kappa=(_gamma_psi(_D(self.N+1))[0]-_gamma_psi(_D(self.M+1))[0]
                     +_gamma_psi(Bd)[0]-_gamma_psi(Bplusu)[0])
        for b,a in zip(self.beta,self.A):
            if a:
                bd=_D(b)
                with localcontext(Context(prec=1600)): bplus=bd+_D(a)
                self._kappa += _gamma_psi(bplus)[0]-_gamma_psi(bd)[0]
        self._threshold=-_I(_D(self.alpha)).log()-self._kappa
        Bf=float(Bd)
        self._kappa_float=(gammaln(self.N+1)-gammaln(self.M+1)+gammaln(Bf)-gammaln(Bf+self.u)
            +sum(gammaln(self.beta[self.active]+self.A[self.active])-gammaln(self.beta[self.active])))
        self._threshold_float=-math.log(self.alpha)-self._kappa_float
        self._single_lower=None

    def update(self,A):
        """Add a checkpoint, preserving all old cuts for running intersections."""
        A=np.asarray(A,dtype=np.int64)
        if A.shape!=self.A.shape or np.any(A<self.A) or A.sum()>self.N: raise ValueError('counts must increase')
        self.A=A.copy(); self._coordinate_cache=None; self._setup()
        return self

    def log_ppr(self,q):
        q=np.asarray(q,dtype=float)
        if np.any(q<self.A) or abs(q.sum()-self.N)>1e-6: return math.inf
        a=self.A[self.active]; v=q[self.active]
        return float(self._kappa_float-sum(gammaln(v+1)-gammaln(v-a+1)))

    def _validated_f_grad(self,q):
        f=_I(0); g=[]
        for s in self.active:
            # Decimal exact subtraction is needed: q can be very close to A.
            qd=_D(q[s]); ad=_D(self.A[s])
            with localcontext(Context(prec=80)):
                x1=qd+1; x0=qd-ad+1
            lg1,ps1=_gamma_psi(x1); lg0,ps0=_gamma_psi(x0)
            f += lg0-lg1; g.append(ps0-ps1)
        return f,g

    def _add_cut(self,q):
        q=np.maximum(np.asarray(q,dtype=float),self.A)
        # Convex tangents hold throughout the domain even if sum q != N.
        f,giv=self._validated_f_grad(q)
        g=np.zeros(self.L)
        rhs=_I(self._threshold.hi)-_I(f.lo)
        for s,gi in zip(self.active,giv):
            gm=float((gi.lo+gi.hi)/2)
            g[s]=gm
            mid=_D(gm)
            err=max(_UP.subtract(mid,gi.lo),_UP.subtract(gi.hi,mid),Decimal(0))
            # Every feasible q_s is in [A_s,A_s+M].
            dist=max(_UP.subtract(_D(q[s]),_D(self.A[s])),
                     _UP.subtract(_D(self.A[s]+self.M),_D(q[s])),Decimal(0))
            rhs += _I(mid)*_I(_D(q[s]))+_I(err)*_I(dist)
        self._G.append(g); self._b.append(_fup(rhs.hi))

    def _dual_bound(self,d):
        """Bound support using arbitrary nonnegative LP dual multipliers.

        For q=A+z, z>=0, sum z=M and Gq<=b:
        d'q <= d'A + lambda'(b-GA) + M max_s(d_s-(G'lambda)_s).
        This inequality needs lambda>=0 only, not LP feasibility/optimality.
        """
        if not self._G: return self._box_bound(d)
        G=np.asarray(self._G); b=np.asarray(self._b)
        scaled=b-G@self.A
        res=linprog(-d,A_ub=G,b_ub=scaled,A_eq=np.ones((1,self.L)),b_eq=[self.M],
                    bounds=(0,None),method='highs')
        if not res.success or res.ineqlin.marginals is None:
            return self._box_bound(d)
        lam=np.maximum(0,-np.asarray(res.ineqlin.marginals))
        # One exact Decimal accumulation is substantially faster than a full
        # interval matrix multiply; all inputs here are binary floats.
        with localcontext(Context(prec=100,rounding=ROUND_CEILING)):
            # products exact at precision 100 for ordinary binary64 inputs;
            # use interval products below to remain valid for all scales.
            rhs=_isum(_I(_D(d[s]))*_I(self.A[s]) for s in range(self.L))
            residual=[_I(_D(x)) for x in d]
            for k in np.flatnonzero(lam):
                lk=_I(_D(lam[k]))
                ba=_I(_D(b[k]))-_isum(_I(_D(G[k,s]))*_I(self.A[s]) for s in self.active)
                rhs += lk*ba
                for s in np.flatnonzero(G[k]): residual[s] -= lk*_I(_D(G[k,s]))
            rhs += _I(self.M)*_I(max(v.hi for v in residual))
        return min(self._box_bound(d),_fup(rhs.hi))

    def _box_bound(self,d):
        rhs=_isum(_I(_D(v))*_I(a) for v,a in zip(d,self.A)) + _I(self.M)*_I(_D(max(d)))
        return _fup(rhs.hi)

    def _single_bound(self,d):
        """One observed category: a scalar root/tangent solves the relaxation."""
        s=int(self.active[0]); a=int(self.A[s])
        if self._single_lower is None:
            def f(q): return float(gammaln(q-a+1)-gammaln(q+1)-self._threshold_float)
            if f(a)<=0: lower=float(a)
            elif f(self.N)>0:
                # An empty confidence set can occur off the coverage event;
                # retain deterministic box rather than fabricate precision.
                lower=float(a)
            else:
                x=brentq(f,float(a),float(self.N),xtol=2e-11,rtol=1e-13)
                q=self.A.astype(float); q[s]=x
                self._add_cut(q)
                g=_D(self._G[-1][s]); b=_D(self._b[-1])
                # g<0: g q_s <= b implies q_s >= b/g.
                lower=max(float(a),_fdown(_I(b).__truediv__(_I(g)).lo))
            self._single_lower=min(float(self.N),lower)
            # Every historical cut also has only this one nonzero entry.
            # Enforce the strongest such lower bound without another LP.
            for gg,bb in zip(self._G,self._b):
                if gg[s]<0:
                    lower_old=_fdown((_I(_D(bb))/_I(_D(gg[s]))).lo)
                    self._single_lower=max(self._single_lower,min(float(self.N),lower_old))
        lo=self._single_lower
        others=np.arange(self.L)!=s
        if not others.any(): return float(d[s]*self.N)
        other=float(np.max(d[others])); best=max(float(d[s]),other)
        rhs=_I(_D(d[s]))*_I(_D(lo))+(_I(self.N)-_I(_D(lo)))*_I(_D(best))
        return _fup(rhs.hi)

    def support(self,d,tol=1e-5,return_result=False):
        """Certified upper count bound on d'q; d may have either sign.

        tol is an optimization gap target in count-objective units, not a
        coverage or rounding tolerance. Failure to attain it only widens the
        reported interval. return_result exposes proposal/status diagnostics.
        """
        d=np.asarray(d,dtype=float)
        if d.shape!=(self.L,) or not np.isfinite(d).all(): raise ValueError('invalid direction')
        if self.M==0 or not self.active.size or np.ptp(d)==0:
            upper=self._box_bound(d); result=SupportResult(upper,None,True,len(self._G),'deterministic')
        elif self.active.size==1:
            upper=self._single_bound(d)
            result=SupportResult(upper,None,True,len(self._G),'scalar')
        else:
            # If the deterministic box maximizer belongs to the current raw
            # set, the raw-set support is already known. Old cuts may tighten
            # it; the dual recomputation handles that possibility. This quick
            # proposal check affects efficiency only, never validity.
            qbox=self.A.astype(float); qbox[int(np.argmax(d))]+=self.M
            if self.log_ppr(qbox)<=-math.log(self.alpha):
                result=SupportResult(self._dual_bound(d),qbox,True,len(self._G),'box-extreme')
                self.last_result=result
                return result if return_result else result.upper
            # Only the best unseen direction needs remaining population mass.
            idx=list(self.active)
            if len(self.unseen): idx.append(int(self.unseen[np.argmax(d[self.unseen])]))
            idx=np.asarray(idx); k=len(self.active)
            di=d[idx]; spread=max(np.ptp(di),1e-16)
            objective=(di-di.min())/spread
            ai=self.A[self.active].astype(float)
            # A positive start permits previously unobserved destinations.
            mass=self.A[idx]+self.beta[idx]
            y0=mass/mass.sum()
            scale=max(1.,self.u)
            def f(y):
                q=ai+self.M*y[:k]
                return float(sum(gammaln(q-ai+1)-gammaln(q+1)))
            def g(y):
                q=ai+self.M*y[:k]; out=np.zeros(len(idx))
                out[:k]=self.M*(digamma(q-ai+1)-digamma(q+1))
                return out
            opt=minimize(lambda y:-objective@y,y0,jac=lambda y:-objective,
                bounds=[(0,1)]*len(idx),
                constraints=[{'type':'eq','fun':lambda y:y.sum()-1,'jac':lambda y:np.ones(len(idx))},
                    {'type':'ineq','fun':lambda y:(self._threshold_float-f(y))/scale,
                     'jac':lambda y:-g(y)/scale}],method='SLSQP',
                options={'ftol':1e-11,'maxiter':120,'disp':False})
            y=np.maximum(opt.x,0); y=y/y.sum() if y.sum()>0 else y0
            q=self.A.astype(float); q[idx]+=self.M*y
            self._add_cut(q)
            upper=self._dual_bound(d)
            # Reused cuts often make the nonlinear tangent sufficient. If
            # necessary, cut at maximizers of the current outer polytope.
            for _ in range(self.max_cuts):
                if upper-float(d@q)<=tol: break
                G=np.asarray(self._G); b=np.asarray(self._b)
                lp=linprog(-d,A_ub=G,b_ub=b-G@self.A,A_eq=np.ones((1,self.L)),b_eq=[self.M],bounds=(0,None),method='highs')
                if not lp.success: break
                qlp=self.A+np.maximum(lp.x,0)
                self._add_cut(qlp)
                new=self._dual_bound(d)
                if upper-new<=tol*.1: upper=min(upper,new); break
                upper=min(upper,new)
            result=SupportResult(upper,q,bool(opt.success),len(self._G),'convex-dual')
        self.last_result=result
        return result if return_result else result.upper

    def bounds(self,d,tol=1e-5):
        d=np.asarray(d,dtype=float)
        return -self.support(-d,tol=tol),self.support(d,tol=tol)

    def coordinate_bounds(self,tol=1e-5,integer=True):
        """All marginal count ranges; exchangeable unseen maxima computed once.

        Bounds concern integer counts by default, so outward ceil/floor may
        safely tighten the continuous relaxation without enumeration.
        """
        if self._coordinate_cache is not None and self._coordinate_cache[0]==bool(integer):
            return tuple(x.copy() for x in self._coordinate_cache[1])
        lo=self.A.astype(float); hi=self.A.astype(float)+self.M
        e=np.zeros(self.L)
        for s in self.active:
            e[s]=1; lo[s],hi[s]=self.bounds(e,tol=tol); e[s]=0
        if len(self.unseen):
            s=int(self.unseen[0]); e[s]=1
            # Current raw set is symmetric among unseen categories. Historical
            # cuts can only strengthen this set, so use a fresh raw-set oracle
            # when history has nonzero entries in currently unseen directions
            # (this never happens for monotonically increasing A).
            upper=self.support(e,tol=tol); hi[self.unseen]=upper
        lo=np.maximum(lo,self.A); hi=np.minimum(hi,self.A+self.M)
        if integer:
            lo=np.ceil(lo); hi=np.floor(hi)
        # Empty intervals indicate a rejected candidate model/empty CS. The
        # product region must then be reported empty; do not silently reverse.
        self._coordinate_cache=(bool(integer),(lo.copy(),hi.copy()))
        return lo,hi


def support_bound(N,A,beta,alpha,direction,**kwargs):
    """Stateless convenience wrapper; reuse RowPPR for repeated directions."""
    return RowPPR(N,A,beta,alpha).support(direction,**kwargs)


def linear_bounds(N,A,beta,alpha,direction,**kwargs):
    return RowPPR(N,A,beta,alpha).bounds(direction,**kwargs)
