"""Audit-only outer bounds for the manuscript's complete P1, P2 and P3.

Required metadata: disjoint species groups partitioning the complete label
vocabulary, and a symmetric nonnegative taxonomic distance matrix omega.
No grouping or taxonomy is inferred from label names. Empty-group Hill
diversity and all pair ratios with zero denominators are defined as zero.

General grouped bounds use certified column-count boxes; they can be quite
conservative. The separate certified_entropy.certify_pooled helper retains
row dependence for the sharper pooled P2 experiments.
"""
from decimal import Decimal,Context,localcontext
import math
import numpy as np
from ppr_convex import _I,_D,_isum,_fdown,_fup
from certified_entropy import _entropy_term,_box_support,_logsumexp,_finish


def _ratio(num,den):
    return _I(num)/_I(den) if den else _I(0)


def _minimal_support(mass,upper):
    if mass<=0: return 0
    total=0
    for k,u in enumerate(sorted((int(v) for v in upper),reverse=True),1):
        total+=u
        if total>=mass: return k
    return len(upper)+1 # explicitly signals impossible count ranges


def _aggregate(rows,N,qlo,qhi):
    L=rows[0].L; R=len(rows); remaining=N-sum(row.u for row in rows)
    if qlo is None or qhi is None:
        b=[row.coordinate_bounds() for row in rows]
        qlo=np.array([v[0] for v in b]);qhi=np.array([v[1] for v in b])
    qlo=np.asarray(qlo,float);qhi=np.asarray(qhi,float)
    if qlo.shape!=(R,L) or qhi.shape!=(R,L): raise ValueError('coordinate arrays must have shape (R,L)')
    if not np.isfinite(qlo).all() or not np.isfinite(qhi).all(): raise ValueError('nonfinite coordinate bounds')
    lower=[];upper=[]
    for s in range(L):
        observed=sum(int(row.A[s]) for row in rows)
        a=_isum(_I(_D(qlo[r,s])) for r in range(R)).lo
        b=_isum(_I(_D(qhi[r,s])) for r in range(R)).hi
        lower.append(max(observed,int(a.to_integral_value(rounding='ROUND_CEILING'))))
        upper.append(min(N,observed+remaining,int(b.to_integral_value(rounding='ROUND_FLOOR'))))
    return np.array(lower,dtype=np.int64),np.array(upper,dtype=np.int64)


def _hill_box(lo,hi):
    """Entropy secant/variational bounds on a bounded probability simplex."""
    size=len(lo); slopes=np.zeros(size); bsum=_I(0)
    for s,(l,u) in enumerate(zip(lo,hi)):
        hl=_entropy_term(_I(l));hu=_entropy_term(_I(u))
        if u>l:
            with localcontext(Context(prec=80)):
                slopes[s]=float(((hu.lo+hu.hi)-(hl.lo+hl.hi))/(2*(u-l)))
        d=_I(_D(slopes[s]))
        bsum+=_I(min((hl-d*_I(l)).lo,(hu-d*_I(u)).lo))
    lower=(bsum-_I(_box_support(-slopes,lo,hi))).lo
    upper=_I(size).log().hi
    ref=np.maximum([(float(l)+float(u))/2 for l,u in zip(lo,hi)],1e-12)
    d=-np.log(ref)
    upper=min(upper,(_logsumexp(-d)+_I(_box_support(d,lo,hi))).hi)
    return _finish(max(Decimal(0),lower),upper,size)


def _pair_ratio(lower,upper,omega,support_lower,support_upper,mass_lower):
    """Original abundance-weighted pair mean Delta, using count products."""
    possible=np.flatnonzero(upper>0)
    if support_upper<2: return _I(0)
    num=_I(0);den=_I(0); distances=[]
    for pos,i in enumerate(possible):
        for j in possible[pos+1:]:
            product=_I(int(lower[i])*int(lower[j]),int(upper[i])*int(upper[j]))
            num+=_I(_D(omega[i,j]))*product;den+=product
            distances.append(_D(omega[i,j]))
    if not distances: return _I(0)
    dmax=max(distances);dmin=min(distances)
    # Integer counts with S>=s and mass>=m force at least
    # (s-1)m-s(s-1)/2 cross products, attained by one dominant species.
    if support_lower>=2:
        comb=(support_lower-1)*mass_lower-support_lower*(support_lower-1)//2
        den=_I(max(den.lo,_D(max(1,comb))),den.hi)
    if den.hi<=0: return _I(0)
    lower_ratio=(_I(num.lo)/_I(den.hi)).lo
    if support_lower<2: lower_ratio=Decimal(0)
    else: lower_ratio=max(lower_ratio,dmin)
    upper_ratio=dmax if den.lo<=0 else min(dmax,(_I(num.hi)/_I(den.lo)).hi)
    return _I(max(Decimal(0),lower_ratio),max(Decimal(0),upper_ratio))


def certify_pillars(rows,N,groups,omega,coordinate_lower=None,coordinate_upper=None):
    """Certify the unchanged manuscript pillars from currently observed audits.

    groups is a sequence of nonempty index lists partitioning range(L).
    omega has shape (L,L), is exactly symmetric, nonnegative, and zero diagonal.
    The returned Pj=(lower,upper) are outward binary64 enclosures.
    ``empty`` must be checked before an interval is used for stopping.
    """
    rows=list(rows);N=int(N)
    if not rows or N<=0 or sum(row.N for row in rows)!=N: raise ValueError('inconsistent total N')
    L=rows[0].L
    if any(row.L!=L for row in rows): raise ValueError('inconsistent vocabularies')
    groups=[np.asarray(g,dtype=int) for g in groups]
    if not groups or any(g.ndim!=1 or len(g)==0 for g in groups): raise ValueError('nonempty groups required')
    if sorted(np.concatenate(groups).tolist())!=list(range(L)): raise ValueError('groups must partition the full vocabulary')
    omega=np.asarray(omega,float)
    if omega.shape!=(L,L) or not np.isfinite(omega).all() or np.any(omega<0): raise ValueError('invalid omega')
    if not np.array_equal(omega,omega.T) or np.any(np.diag(omega)!=0): raise ValueError('omega must be symmetric with zero diagonal')
    Omega=float(omega.max(initial=0))
    lower,upper=_aggregate(rows,N,coordinate_lower,coordinate_upper)
    empty=bool(np.any(lower>upper) or lower.sum()>N or upper.sum()<N)
    if empty:
        return dict(P1=(math.inf,-math.inf),P2=(math.inf,-math.inf),P3=(math.inf,-math.inf),
            empty=True,count_lower=lower,count_upper=upper,groups=[])
    Slo=max(1,int(np.sum(lower>0)),_minimal_support(N,upper))
    Shi=min(N,int(np.sum(upper>0)))
    details=[];p2total=_I(0)
    for g in groups:
        l=lower[g];u=upper[g];outside=np.ones(L,bool);outside[g]=False
        mlo=max(int(l.sum()),N-int(upper[outside].sum()))
        mhi=min(int(u.sum()),N-int(lower[outside].sum()))
        slo=max(int(np.sum(l>0)),_minimal_support(mlo,u))
        shi=min(mhi,int(np.sum(u>0)))
        if mhi==0:
            Dlo,Dhi=0.,0.;ql=[Decimal(0)]*len(g);qu=ql.copy()
        else:
            ql=[];qu=[]
            for ls,us in zip(l,u):
                ls,us=int(ls),int(us)
                if us==0: ql.append(Decimal(0));qu.append(Decimal(0));continue
                dl=ls+int(u.sum())-us
                du=us+int(l.sum())-ls
                a=max(_ratio(ls,dl).lo,_ratio(ls,mhi).lo)
                b=min(_ratio(us,du).hi,_ratio(us,max(1,mlo)).hi,Decimal(1))
                ql.append(max(Decimal(0),a));qu.append(b)
            Dlo,Dhi=_hill_box(ql,qu)
            # A group possibly absent has an additional zero-contribution branch.
            if mlo==0: Dlo=0.
            Dhi=min(Dhi,float(shi))
        p2total+=_I(_D(Dlo),_D(Dhi))
        delta=_pair_ratio(l,u,omega[np.ix_(g,g)],slo,shi,mlo)
        details.append(dict(indices=g.tolist(),mass_lower=mlo,mass_upper=mhi,
            support_lower=slo,support_upper=shi,P2=(Dlo,Dhi),
            Delta=(_fdown(delta.lo),_fup(delta.hi)),_delta=delta))
    # Presence-weighted global pair mean, the first term of original P3.
    mandatory=np.flatnonzero(lower>0);possible=np.flatnonzero(upper>0)
    numlo=_I(0);numhi=_I(0);distances=[]
    for pos,i in enumerate(possible):
        for j in possible[pos+1:]:
            value=_I(_D(omega[i,j]));numhi+=value;distances.append(_D(omega[i,j]))
            if lower[i]>0 and lower[j]>0: numlo+=value
    if Shi<2: presence=_I(0)
    else:
        pairhi=Shi*(Shi-1)//2;pairlo=Slo*(Slo-1)//2
        pl=(_I(numlo.lo)/_I(pairhi)).lo
        pu=max(distances) if pairlo==0 else min(max(distances),(_I(numhi.hi)/_I(pairlo)).hi)
        if Slo>=2: pl=max(pl,min(distances))
        presence=_I(max(Decimal(0),pl),pu)
    # The second term is a weighted mean of within-group Delta, with weights
    # S_k/S. Bound that simplex explicitly rather than sum loose maxima.
    all_slo=[d['support_lower'] for d in details];all_shi=[d['support_upper'] for d in details]
    wl=[];wu=[];dl=[];du=[]
    for k,d in enumerate(details):
        a=all_slo[k];b=all_shi[k]
        low=max(_ratio(a,a+sum(all_shi)-b).lo,_ratio(a,Shi).lo)
        high=min(_ratio(b,b+sum(all_slo)-a).hi,_ratio(b,Slo).hi,Decimal(1)) if b else Decimal(0)
        wl.append(low);wu.append(high)
        dl.append(_fdown(d['_delta'].lo));du.append(_fup(d['_delta'].hi))
        d['richness_weight']=(_fdown(low),_fup(high));del d['_delta']
    within_low=_box_support(-np.asarray(dl),wl,wu).copy_negate()
    within_high=_box_support(np.asarray(du),wl,wu)
    within=_I(max(Decimal(0),within_low),min(_D(Omega),within_high))
    P3=presence+within
    p1=(Slo,Shi);p2=(max(1.,_fdown(p2total.lo)),min(float(L),float(N),_fup(p2total.hi)))
    p3=(max(0.,_fdown(P3.lo)),min(2*Omega,_fup(P3.hi)))
    return dict(P1=p1,P2=p2,P3=p3,empty=bool(p1[0]>p1[1] or p2[0]>p2[1] or p3[0]>p3[1]),
        count_lower=lower,count_upper=upper,groups=details,
        P3_presence=(_fdown(presence.lo),_fup(presence.hi)),
        P3_within=(_fdown(within.lo),_fup(within.hi)),Omega=Omega,
        method='certified grouped outer propagation; zero pair denominators contribute zero')
