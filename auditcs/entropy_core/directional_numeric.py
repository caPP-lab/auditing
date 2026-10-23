"""Outward lower bounds for monotone capped-bet mixture wealth.

No logarithm or exponential is used in a numerical certificate. The exact
cap is min(k, (9/10)/max(h-b, eps)) on the lower side and the analogous
upper-side expression. k, h, b, eps and restart weights are interpreted as
their represented binary64 values, while 9/10 is the exact rational number.

Assumptions: ordinary IEEE-754 binary64 round-to-nearest elementary NumPy
operations and exact binary frexp/ldexp scaling. Inputs must already enclose
the mathematical observations and predictable ranges. This module cannot
repair an unvalidated computation of those observations.
"""
from dataclasses import dataclass
from decimal import Decimal,Context,localcontext
import math
import numpy as np


_EPS=np.float64(2.**-52)
_INF=np.inf
_MAX=np.finfo(np.float64).max


def _down(x): return np.nextafter(x,-_INF)
def _up(x): return np.nextafter(x,_INF)
def _positive_down(x): return np.maximum(0.,np.nextafter(x,0.))


@dataclass
class ScaledLower:
    """The nonnegative lower bound mantissa * 2**exponent, componentwise."""
    mantissa: np.ndarray
    exponent: np.ndarray

    def to_float(self):
        with np.errstate(over='ignore',under='ignore'):
            x=np.ldexp(self.mantissa,self.exponent)
        # An overflowing represented lower bound still exceeds MAX, so MAX
        # is a valid finite lower bound. Underflow rounds downward to zero.
        x=np.where(np.isposinf(x),_MAX,_positive_down(x))
        return float(x) if np.ndim(x)==0 else x

    def exceeds(self,threshold):
        """Compare to an exact represented threshold without overflow."""
        t=np.asarray(threshold,float)
        if np.any(t<0) or not np.isfinite(t).all():raise ValueError('finite nonnegative threshold required')
        mt,et=np.frexp(t)
        result=(self.mantissa>0)&((self.exponent>et)|((self.exponent==et)&(self.mantissa>mt)))
        result=np.where(t==0,self.mantissa>0,result)
        return bool(result) if np.ndim(result)==0 else result


def _guard(n):
    """Lower bound on reciprocal accumulated relative rounding inflation.

    n operations have relative inflation <=(1+u)^n, u=2^-53.
    For n*u<1, 1-n*2^-52 <=1-n*u <=(1+u)^(-n).
    The subtraction here is exact for the small integer block sizes, and
    nextafter provides an additional outward step.
    """
    return np.nextafter(np.float64(1)-np.float64(n)*_EPS,0.)


def _scaled_product_lower(factors,block_size=128):
    """Products over axis0; positive factors themselves are lower bounds."""
    shape=factors.shape[1:]
    mantissa=np.ones(shape,float);exponent=np.zeros(shape,np.int64)
    for start in range(0,len(factors),block_size):
        block=factors[start:start+block_size]
        m,e=np.frexp(block)
        # m in [1/2,1), hence every intermediate product is normal for
        # block_size<=512. Any binary multiplication ordering has at most
        # block_size elementary roundings; guard(n) covers all of them.
        bm=_positive_down(np.prod(m,axis=0,dtype=np.float64)*_guard(len(block)))
        be=np.sum(e,axis=0,dtype=np.int64)
        bm,shift=np.frexp(bm);be+=shift
        combined=_positive_down(mantissa*bm)
        mantissa,shift=np.frexp(combined);exponent+=be+shift
    # Empty products equal one. Normalize for subsequent scaled summation.
    mantissa,shift=np.frexp(mantissa);exponent+=shift
    return mantissa,exponent


def _scaled_sum_lower(mantissas,exponents):
    """Sum positive scaled bounds across axis0 without overflow."""
    m=np.asarray(mantissas);e=np.asarray(exponents,np.int64)
    emax=np.max(e,axis=0)
    with np.errstate(under='ignore'):
        aligned=_positive_down(np.ldexp(m,e-emax))
    total=_positive_down(np.sum(aligned,axis=0,dtype=np.float64)*_guard(len(m)))
    total,shift=np.frexp(total)
    return ScaledLower(total,emax+shift)


def wealth_lower(h,Ylo_bounds,Yupper_bounds,minlo,maxhi,ks,
                 starts=(0,),weights=(1.,),side='lo',*,
                 denominator_floor=1e-15,block_size=128,return_scaled=False):
    """Certified lower mixture wealth, scalar or a vector of coordinates.

    Ylo_bounds and Yupper_bounds enclose the SAME stream used for this call.
    For side='lo', only Ylo_bounds and minlo enter; for side='hi', only
    Yupper_bounds and maxhi enter. Thus a stream with only the relevant
    one-sided observation bound can be passed twice without losing validity.

    Observation arrays have shape (time,) or (time,coordinates). h is scalar
    or broadcastable to the coordinate shape. ks is a finite nonnegative
    vector. Each restart mixes its k-products uniformly; nonnegative restart
    weights must sum to at most one exactly. A restart at/after current time
    contributes an unborn product of one.

    Returned lower wealth may be compared only to an outward UPPER bound on
    the desired threshold. return_scaled=True permits comparison without
    overflow via ScaledLower.exceeds(threshold).
    """
    if side not in ('lo','hi'):raise ValueError("side must be 'lo' or 'hi'")
    if not isinstance(block_size,int) or not 1<=block_size<=512:raise ValueError('block_size must be in1..512')
    ks=np.asarray(ks,np.float64)
    if ks.ndim!=1 or len(ks)==0 or np.any(ks<0) or not np.isfinite(ks).all():raise ValueError('invalid bets')
    starts=np.asarray(starts,np.int64);weights=np.asarray(weights,np.float64)
    if starts.ndim!=1 or weights.shape!=starts.shape or not len(starts) or np.any(starts<0):raise ValueError('invalid restarts')
    if np.any(weights<0) or not np.isfinite(weights).all():raise ValueError('invalid mixture weights')
    with localcontext(Context(prec=1600)):
        total_weight=sum((Decimal.from_float(float(w)) for w in weights),Decimal(0))
    if total_weight>1:raise ValueError('represented restart weights sum to more than one')
    if not np.isfinite(denominator_floor) or denominator_floor<=0:raise ValueError('positive finite denominator floor required')
    y=np.asarray(Ylo_bounds if side=='lo' else Yupper_bounds,np.float64)
    b=np.asarray(minlo if side=='lo' else maxhi,np.float64)
    h=np.asarray(h,np.float64)
    if y.ndim not in (1,2):raise ValueError('observations need time axis and optional coordinate axis')
    b=np.broadcast_to(b,y.shape)
    for v in (y,b,h):
        # This broad range keeps every factor construction in normal finite
        # arithmetic: cap<=.9/1e-15 and |delta|<=2e200. The wealth products
        # themselves have no such scale restriction, thanks to frexp.
        if not np.isfinite(v).all() or np.any(np.abs(v)>1e200):raise ValueError('input magnitude exceeds supported finite range1e200')
    if denominator_floor<1e-100:raise ValueError('denominator_floor below supported1e-100')
    if side=='lo':
        y=np.maximum(y,b) # both are valid lower bounds for the observation
        delta=_down(y-h);dlo=_down(h-b);dhi=_up(h-b)
    else:
        y=np.minimum(y,b) # both are valid upper bounds for the observation
        delta=_down(h-y);dlo=_down(b-h);dhi=_up(b-h)
    dlo=np.maximum(dlo,denominator_floor);dhi=np.maximum(dhi,denominator_floor)
    c_lo=np.nextafter(np.float64(9)/np.float64(10),0.)
    c_hi=np.nextafter(np.float64(9)/np.float64(10),_INF)
    caplo=_positive_down(c_lo/dhi);caphi=_up(c_hi/dlo)
    kshape=(1,len(ks))+(1,)*(y.ndim-1)
    lamlo=np.minimum(ks.reshape(kshape),caplo[:,None,...])
    lamhi=np.minimum(ks.reshape(kshape),caphi[:,None,...])
    dl=delta[:,None,...]
    chosen=np.where(dl<0,lamhi,lamlo)
    factor=_down(1.+_down(chosen*dl))
    # The exact capped factor is always >=1-9/10=1/10; using this
    # deterministic lower bound is safe even when interval subtraction loses
    # precision. It also guarantees positive normal frexp input.
    factor=np.maximum(factor,np.nextafter(np.float64(1)/np.float64(10),0.))
    if not np.isfinite(factor).all():raise ArithmeticError('factor overflow: reduce input magnitude or bet scales')
    ms=[];es=[]
    for start,weight in zip(starts,weights):
        if weight==0:continue
        m,e=_scaled_product_lower(factor[min(int(start),len(factor)):],block_size)
        wm,we=np.frexp(weight)
        m=_positive_down(m*wm)
        # Uniform mixture over k uses exact division by the integer K.
        m=_positive_down(m/len(ks));m,shift=np.frexp(m);e=e+we+shift
        ms.extend(m);es.extend(e)
    if ms:
        result=_scaled_sum_lower(ms,es)
    else:
        shape=np.broadcast_shapes(y.shape[1:],h.shape)
        result=ScaledLower(np.zeros(shape),np.zeros(shape,np.int64))
    return result if return_scaled else result.to_float()
