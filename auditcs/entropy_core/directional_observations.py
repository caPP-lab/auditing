"""Outward binary64 arithmetic for adaptive audit observations.

Assumes IEEE-754 round-to-nearest binary64 elementary NumPy operations without
flush-to-zero. Each elementary result is expanded by np.nextafter in the
required direction; every addition in a pairwise sum is expanded separately.
Inputs pred, directions, intercepts and actual selection probabilities are
arbitrary exact represented finite binary64 numbers. No true labels beyond the
single supplied outcome are used. Known integer counts must fit exactly in
binary64. Nonfinite inputs/results are rejected.
"""
from __future__ import annotations
import math
import numpy as np


def _point(x):
    x=np.asarray(x,dtype=float)
    return x,x


def _down(x):return np.nextafter(x,-np.inf)
def _up(x):return np.nextafter(x,np.inf)


def _add(a,b):return _down(a[0]+b[0]),_up(a[1]+b[1])
def _neg(a):return -a[1],-a[0]
def _sub(a,b):return _add(a,_neg(b))


def _mul(a,b):
    p=[a[i]*b[j] for i in range(2) for j in range(2)]
    return _down(np.minimum.reduce(p)),_up(np.maximum.reduce(p))


def _div(a,b):
    if np.any((b[0]<=0)&(b[1]>=0)):
        raise ZeroDivisionError('interval denominator contains zero')
    v=[a[i]/b[j] for i in range(2) for j in range(2)]
    return _down(np.minimum.reduce(v)),_up(np.maximum.reduce(v))


def _sum(a,axis=0):
    """Outward pairwise sum: every individual addition is rounded outward."""
    lo=np.moveaxis(np.asarray(a[0],float),axis,0)
    hi=np.moveaxis(np.asarray(a[1],float),axis,0)
    if len(lo)==0:return _point(np.zeros(lo.shape[1:]))
    while len(lo)>1:
        pairs=len(lo)//2
        ll=_down(lo[:2*pairs:2]+lo[1:2*pairs:2])
        hh=_up(hi[:2*pairs:2]+hi[1:2*pairs:2])
        if len(lo)%2:
            ll=np.concatenate([ll,lo[-1:]],axis=0)
            hh=np.concatenate([hh,hi[-1:]],axis=0)
        lo,hi=ll,hh
    return lo[0],hi[0]


def _finite(*xs):
    if any(not np.isfinite(x).all() for x in xs):
        raise ArithmeticError('nonfinite interval input or output')


def dyadic_probabilities(weights,active=None,bits=53):
    """Quantize desired weights to exact positive dyadic probabilities.

    The returned integer bins sum to 2**bits, and each active row gets at
    least one bin. Desired weights are normalized using their exact binary64
    integer ratios; allocation uses integer division and largest remainders.
    Minimum-one-bin corrections may perturb any probability by at most
    (number of active rows)/2**bits. The returned float probabilities themselves
    are exact dyadic fractions when bits<=53.
    """
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or not 1<=bits<=53:raise ValueError('invalid shape or precision')
    _finite(w)
    active=(w>0) if active is None else np.asarray(active,dtype=bool)
    if active.shape!=w.shape or not active.any():raise ValueError('no active rows')
    if np.any(w<0) or np.any(w[active]<=0):raise ValueError('active weights must be positive')
    indices=np.flatnonzero(active);T=1<<bits
    if len(indices)>T:raise ValueError('not enough positive bins')
    ratios=[float(w[r]).as_integer_ratio() for r in indices]
    power=max(den.bit_length()-1 for _,den in ratios)
    numerators=[num<<(power-(den.bit_length()-1)) for num,den in ratios]
    denominator=sum(numerators)
    divisions=[divmod(T*num,denominator) for num in numerators]
    bins=[max(1,int(whole)) for whole,_ in divisions]
    missing=T-sum(bins)
    if missing>0:
        # Without the positivity adjustment the deficit is less than R;
        # giving zero-floor bins one unit can only decrease that deficit.
        order=sorted(range(len(bins)),key=lambda i:divisions[i][1],reverse=True)
        for i in order[:missing]:bins[i]+=1
    elif missing<0:
        # Remove only from bins larger than one, smallest remainder first.
        order=sorted(range(len(bins)),key=lambda i:divisions[i][1])
        remaining=-missing
        for i in order:
            take=min(remaining,bins[i]-1)
            bins[i]-=take;remaining-=take
            if remaining==0:break
        if remaining:raise ArithmeticError('could not balance positive bins')
    counts=np.zeros(len(w),dtype=np.int64)
    counts[indices]=bins
    if sum(map(int,counts))!=T or np.any(counts[active]<1):
        raise ArithmeticError('invalid dyadic allocation')
    probabilities=counts.astype(float)/float(T)
    return probabilities,counts


def sample_dyadic_row(rng,weights,active=None,bits=53):
    """Return (row, actual_probability_vector, integer_bin_counts)."""
    probabilities,counts=dyadic_probabilities(weights,active,bits)
    draw=int(rng.integers(0,1<<bits))
    row=int(np.searchsorted(np.cumsum(counts,dtype=np.int64),draw,side='right'))
    if counts[row]<=0:raise ArithmeticError('selected inactive row')
    return row,probabilities,counts


def outward_observation(N_r,A,pred,dl,bl,du,bu,pi,r,s,coordinates=True):
    """Return conservative entropy/coordinate outcomes and predictable bounds.

    Keys entropy_ylo/entropy_yhi are lower/upper approximations to the exact
    affine observations. entropy_lower/entropy_upper are predictable bounds
    for those respective one-sided observations. If coordinates=True, keys
    coordinate_ylo/coordinate_yhi, coordinate_lower/coordinate_upper have
    length L. Separate lower and upper observations must be stored by betting.

    The caller must sample using the actual probabilities supplied as pi.
    sample_dyadic_row provides such probabilities with exact normalization.
    """
    nr=np.asarray(N_r);aa=np.asarray(A)
    if nr.ndim!=1 or aa.ndim!=2 or len(nr)!=len(aa):raise ValueError('count shapes')
    if not np.issubdtype(nr.dtype,np.integer) or not np.issubdtype(aa.dtype,np.integer):
        raise ValueError('counts must be integers')
    N=sum(map(int,nr));R,L=aa.shape
    if N<=0 or N>(1<<53) or np.any(nr<0) or np.any(aa<0) or np.any(aa>nr[:,None]):
        raise ValueError('count range')
    u=aa.sum(axis=1);rem=nr-u;active=rem>0
    if np.any(rem<0) or not active.any():raise ValueError('no valid remaining audit')
    pred=np.asarray(pred,float);dl=np.asarray(dl,float);du=np.asarray(du,float);pi=np.asarray(pi,float)
    if pred.shape!=(R,L) or dl.shape!=(L,) or du.shape!=(L,) or pi.shape!=(R,):raise ValueError('coefficient shapes')
    if not(0<=r<R and 0<=s<L) or not active[r]:raise ValueError('invalid observation index')
    _finite(pred,dl,du,pi,bl,bu)
    if np.any(pi[active]<=0) or np.any(pi<0) or np.any(pi[~active]!=0):
        raise ValueError('positive active probabilities and zero inactive probabilities required')
    n=_point(float(N));observed=aa.sum(axis=0).astype(float)
    m=_div(_point(rem.astype(float)),n)
    denominator=np.where(active,pi,1.)
    ratios=_div(m,_point(denominator))
    # Inactive remaining weights are mathematically exactly zero.
    ratios=(np.where(active,ratios[0],0.),np.where(active,ratios[1],0.))
    result={}
    for name,d,b,lower_side in [('lo',dl,bl,True),('hi',du,bu,False)]:
        # Baselines may be arbitrary predictable represented numbers, so this
        # fast approximate dot is selected first and then treated as exact.
        baseline=pred@d
        _finite(baseline)
        known=_div(_sum(_mul(_point(observed),_point(d))),n)
        center=_add(_add(_point(float(b)),known),_sum(_mul(m,_point(baseline))))
        delta=_sub(_point(float(d[s])),_point(float(baseline[r])))
        value=_add(center,_mul((ratios[0][r],ratios[1][r]),delta))
        minimum=_add(center,_mul(ratios,_sub(_point(float(d.min())),_point(baseline))))
        maximum=_add(center,_mul(ratios,_sub(_point(float(d.max())),_point(baseline))))
        lower=float(np.min(minimum[0][active]));upper=float(np.max(maximum[1][active]))
        # Clamping to a predictable outer range keeps a rounded one-sided
        # observation inside its range without sacrificing its direction.
        observed_value=max(float(value[0]),lower) if lower_side else min(float(value[1]),upper)
        result['entropy_y'+name]=observed_value
        result['entropy_'+('lower' if lower_side else 'upper')]=lower if lower_side else upper
        result['entropy_'+name+'_range']=(lower,upper)
    if coordinates:
        center=_add(_div(_point(observed),n),
                    _sum(_mul((m[0][:,None],m[1][:,None]),_point(pred)),axis=0))
        onehot=np.zeros(L);onehot[s]=1.
        value=_add(center,_mul((ratios[0][r],ratios[1][r]),_sub(_point(onehot),_point(pred[r]))))
        ratio_matrix=(ratios[0][:,None],ratios[1][:,None])
        minimum=_add(center,_mul(ratio_matrix,_neg(_point(pred))))
        maximum=_add(center,_mul(ratio_matrix,_sub(_point(1.),_point(pred))))
        lower=np.min(minimum[0][active],axis=0);upper=np.max(maximum[1][active],axis=0)
        result.update(coordinate_ylo=np.maximum(value[0],lower),
                      coordinate_yhi=np.minimum(value[1],upper),
                      coordinate_lower=lower,coordinate_upper=upper)
    for v in result.values():_finite(v)
    return result
