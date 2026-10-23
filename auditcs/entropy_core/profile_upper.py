"""Standalone certified upper entropy endpoint; no coordinate localization."""
import numpy as np
from scipy.optimize import brentq
from efficiency_directional import EntropyBet
from directional_numeric import wealth_lower
from ppr_convex import _I,_D,_fdown,_fup
from certified_entropy import _exp

def upper_endpoint(Y,range_upper,L,alpha=.0225,starts=(0,750,2250,5250,11250,23250),weights=(.5,.25,.125,.0625,.03125,.03125)):
    bet=EntropyBet(L,2*alpha,restarts=True)
    bet.Yhi=list(Y);bet.maxhi=list(range_upper)
    bet.starts=np.array(starts);bet.weights=np.array(weights)
    threshold=float(np.log(1/alpha))
    f=lambda h:bet.wealth(h,'hi')-threshold
    h=bet.logL if f(bet.logL)<=0 else (0. if f(0.)>0 else brentq(f,0,bet.logL))
    h=min(bet.logL,h+1e-8)
    if h<bet.logL:
        target=_fup((_I(1)/_I(_D(alpha))).hi)
        okay=bool(wealth_lower(h,Y,Y,np.zeros(len(Y)),range_upper,bet.ks,bet.starts,bet.weights,'hi',return_scaled=True).exceeds(target))
        if not okay:h=bet.logL
    else:okay=True
    return {'entropy_upper':h,'pillar_upper':min(float(L),_fup(_exp(_I(_D(h))).hi)),'numeric_verified':okay}

if __name__=='__main__':
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    z=np.load(root/'output/conservatism_research/metadata_cs_streams_1729.npz');n0=int(z['pilot']);L=int(z['L'])
    rows=[]
    for t in [1000,1500,2000,2500,3000,4000,5000]:
        n=t-n0;a=upper_endpoint(z['Yhi'][:n],z['maxhi'][:n],L);a['t']=t;rows.append(a)
    (root/'output/optimal_revision/profile_upper_development.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows))
