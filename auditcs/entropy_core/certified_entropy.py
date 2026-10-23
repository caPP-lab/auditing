"""Outward-certified propagation from RowPPR oracles to pooled Hill diversity.

The pooled target is P2(C)=exp(H(p(C))), p_s=sum_r C_rs/N. All inputs
are row sizes, working parameters, and observed audits. This module neither
accepts nor accesses a complete ground-truth confusion matrix.

The row confidence region is unchanged. Secant minorants and variational
majorants propagate it using certified linear support bounds. Floating-point
coefficient choices are arbitrary choices of valid affine bounds; their
intercepts, support aggregation, and exponentiation are validated outward.
"""
from decimal import Decimal, localcontext, Context
import math
import numpy as np
from ppr_convex import _I, _D, _DOWN, _UP, _PREC, _fdown, _fup, _isum


def _exp(x):
    """Outward interval exponential, using correctly rounded Decimal exp."""
    x=x if isinstance(x,_I) else _I(x)
    a,b=_DOWN.exp(x.lo),_UP.exp(x.hi)
    ea=Decimal(1).scaleb(a.adjusted()-_PREC+1)
    eb=Decimal(1).scaleb(b.adjusted()-_PREC+1)
    return _I(max(Decimal(0),_DOWN.subtract(a,ea)),_UP.add(b,eb))


def _entropy_term(x):
    x=x if isinstance(x,_I) else _I(x)
    if x.lo==x.hi==0: return _I(0)
    if x.lo<=0: raise ValueError('entropy interval must be strictly positive or exactly zero')
    return -(x*x.log())


def _logsumexp(x):
    """Interval log(sum(exp(x))) for exact binary64 coefficients x."""
    x=np.asarray(x,float)
    shift=_I(_D(max(x)))
    total=_isum(_exp(_I(_D(v))-shift) for v in x)
    return shift+total.log()


def _box_support(direction,lo,hi):
    """Certified support of {p: lo<=p<=hi, sum p=1}.

    For any real nu, d'p <= nu + sum_s max((d_s-nu)lo_s,
    (d_s-nu)hi_s). A floating greedy allocation only chooses nu; the
    displayed bound is recomputed with outward arithmetic.
    """
    d=np.asarray(direction,float)
    rem=1-sum(float(x) for x in lo)
    nu=float(np.min(d))
    for s in np.argsort(-d):
        nu=float(d[s])
        rem-=max(0.,float(hi[s])-float(lo[s]))
        if rem<=0: break
    ni=_I(_D(nu)); result=ni
    for ds,l,u in zip(d,lo,hi):
        result+=(_I(_D(ds))-ni)*_I(u if ds>=nu else l)
    return result.hi


def _row_support_sum(rows,d,N,tol):
    vals=[row.support(d,tol=tol) for row in rows]
    if not np.isfinite(vals).all(): raise ArithmeticError('nonfinite row support bound')
    return _isum(_I(_D(v)) for v in vals)/_I(N)


def _finish(lower_H,upper_H,L):
    # Deterministic entropy bounds [0,log L] provide exact final clipping.
    lower_H=max(Decimal(0),lower_H)
    lower=max(1.,_fdown(_exp(_I(lower_H)).lo))
    upper=min(float(L),_fup(_exp(_I(upper_H)).hi))
    return lower,upper


def certify_pooled(rows,N,coordinate_lower=None,coordinate_upper=None,refs=None,tol=1e-5):
    """Return numerically certified pooled P2 and categorical P1 bounds.

    Parameters
    ----------
    rows : sequence of RowPPR
    N : int
        Total population size; must equal the sum of known row sizes.
    coordinate_lower, coordinate_upper : arrays of shape (R,L), optional
        Already certified row count bounds; if omitted, compute them.
    refs : sequence of positive length-L vectors, optional
        Centers used only to choose affine majorants. No normalization is
        required. Nonpositive entries are replaced by 1e-12 for this choice;
        no species is deleted from the confidence region.

    Returns
    -------
    dict with lo, hi, box_lo, box_hi, p1_lo, p1_hi, p_lo, p_hi,
    count_lower, count_upper, lower_H, upper_H, empty.

    ``box_lo/box_hi`` use the same affine bounds on an enclosing coordinate
    box intersected with the simplex. ``lo/hi`` additionally retain all row
    PPR constraints, and are intersected with these valid box bounds.
    No statistical error probability is spent on choosing coefficients,
    references, or the tightest of these bounds.
    """
    rows=list(rows); N=int(N)
    if not rows or N<=0 or sum(row.N for row in rows)!=N: raise ValueError('inconsistent total N')
    L=rows[0].L; R=len(rows)
    if any(row.L!=L for row in rows): raise ValueError('inconsistent vocabularies')
    if coordinate_lower is None or coordinate_upper is None:
        both=[row.coordinate_bounds() for row in rows]
        coordinate_lower=np.array([x[0] for x in both])
        coordinate_upper=np.array([x[1] for x in both])
    qlo=np.asarray(coordinate_lower,float); qhi=np.asarray(coordinate_upper,float)
    if qlo.shape!=(R,L) or qhi.shape!=(R,L): raise ValueError('invalid coordinate bound shape')
    if not np.isfinite(qlo).all() or not np.isfinite(qhi).all(): raise ValueError('nonfinite coordinate bounds')
    # Integer rounding is exact on represented finite endpoints. No tolerance
    # is used around one: count upper <1 is the absence certificate.
    clo=[]; chi=[]; lo=[]; hi=[]
    observed=np.array([sum(int(row.A[s]) for row in rows) for s in range(L)],dtype=np.int64)
    unaudited=N-sum(row.u for row in rows)
    for s in range(L):
        cl=_isum(_I(_D(qlo[r,s])) for r in range(R)).lo
        cu=_isum(_I(_D(qhi[r,s])) for r in range(R)).hi
        cl=max(int(observed[s]),int(cl.to_integral_value(rounding='ROUND_CEILING')))
        cu=min(N,int(observed[s])+unaudited,int(cu.to_integral_value(rounding='ROUND_FLOOR')))
        clo.append(cl);chi.append(cu)
        lo.append((_I(cl)/_I(N)).lo);hi.append((_I(cu)/_I(N)).hi)
    p1lo=sum(v>=1 for v in clo);p1hi=sum(v>=1 for v in chi)
    empty=any(l>u for l,u in zip(clo,chi)) or sum(clo)>N or sum(chi)<N
    base=dict(p1_lo=p1lo,p1_hi=p1hi,
        p_lo=np.array([_fdown(v) for v in lo]),p_hi=np.array([_fup(v) for v in hi]),
        count_lower=np.array(clo,dtype=np.int64),count_upper=np.array(chi,dtype=np.int64),empty=empty)
    if empty:
        base.update(lo=math.inf,hi=-math.inf,box_lo=math.inf,box_hi=-math.inf,
                    lower_H=math.inf,upper_H=-math.inf)
        return base
    if unaudited==0:
        H=_isum(_entropy_term(_I(int(c))/_I(N)) for c in observed if c)
        lower,upper=_finish(H.lo,H.hi,L)
        base.update(lo=lower,hi=upper,box_lo=lower,box_hi=upper,
                    lower_H=_fdown(H.lo),upper_H=_fup(H.hi))
        return base

    # Choose any float slopes, then validate an affine minorant directly:
    # b_s <= min(h(l_s)-d_s*l_s, h(u_s)-d_s*u_s). Since h-d*x is
    # concave, the resulting line lies below h throughout [l_s,u_s].
    slopes=np.zeros(L); bsum=_I(0)
    for s,(l,u) in enumerate(zip(lo,hi)):
        hl=_entropy_term(_I(l)); hu=_entropy_term(_I(u))
        if u>l:
            with localcontext(Context(prec=80)):
                slope=float(((hu.lo+hu.hi)-(hl.lo+hl.hi))/(2*(u-l)))
            slopes[s]=slope if math.isfinite(slope) else 0.
        ds=_I(_D(slopes[s]))
        intercept=min((hl-ds*_I(l)).lo,(hu-ds*_I(u)).lo)
        bsum+=_I(intercept)
    low_row=(bsum-_row_support_sum(rows,-slopes,N,tol)).lo
    low_box=(bsum-_I(_box_support(-slopes,lo,hi))).lo
    lower_H=max(Decimal(0),low_row,low_box)

    # Entropy variational inequality, valid for every represented vector d:
    # H(p) <= log(sum_s exp(-d_s)) + d'p.
    # Thus coefficient rounding and center normalization introduce no gap
    # in validity. Tiny positive reference entries merely choose directions.
    logL=_I(L).log()
    upper_H=logL.hi; upper_box_H=logL.hi
    if refs is None:
        posterior=sum((row.A+row.M*(row.A+row.beta)/(row.u+row.beta.sum())) for row in rows)/N
        refs=[posterior,(np.array([float(v) for v in lo])+np.array([float(v) for v in hi]))/2]
    for reference in refs:
        v=np.asarray(reference,float)
        if v.shape!=(L,) or not np.isfinite(v).all(): raise ValueError('invalid entropy reference')
        d=-np.log(np.maximum(v,1e-12))
        z=_logsumexp(-d)
        row_upper=(z+_row_support_sum(rows,d,N,tol)).hi
        box_upper=(z+_I(_box_support(d,lo,hi))).hi
        upper_H=min(upper_H,row_upper,box_upper)
        upper_box_H=min(upper_box_H,box_upper)
    lower,upper=_finish(lower_H,upper_H,L)
    box_lower,box_upper=_finish(max(Decimal(0),low_box),upper_box_H,L)
    base.update(lo=lower,hi=upper,box_lo=box_lower,box_hi=box_upper,
                lower_H=_fdown(lower_H),upper_H=_fup(upper_H),
                empty=bool(lower>upper),slopes=slopes)
    return base
