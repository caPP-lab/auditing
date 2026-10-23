"""Practitioner API for observed-label-only, finite-population Pillar 2 audits.

No population truth or replay file is read. Record IDs are positions 0..N-1.
The optional predictor sees revealed IDs/labels and a copied current state;
it returns candidate-independent residual scores for every known record.
"""
from types import SimpleNamespace
import numpy as np
from composition_audit import CompositionAudit,composition_count_bounds,certify_composition_pillars
from efficiency_directional import census_hill,sample_dyadic_row
from directional_observations import _point,_sum,_sub
from convex_profile_replay import positive_observation
from metadata_entropy_audit import control_observation
from smallpilot_research import floor_variance_allocation
from profile_upper import upper_endpoint
from quad_profile_convex import ConvexEntropyProfile
from direct_convex_numeric import certify
from ppr_convex import _I,_D,_fdown,_fup
from certified_entropy import _exp


class ConvexAudit:
    def __init__(self,record_rows,row_to_label,L,*,seed=0,predictor_callback=None,
                 refit_at=(200,500,1000,2000,3000),joint_budget=False,
                 upper_starts=(200,750,2250,5250,11250,23250),eta=1.):
        if not np.isfinite(L) or L!=int(L) or L<2:raise ValueError('integer catalogue size L>=2 required')
        L=int(L)
        rows=np.asarray(record_rows);rowmap=np.asarray(row_to_label)
        if (rows.ndim!=1 or not len(rows) or not np.isfinite(rows).all()
            or np.any(rows!=np.floor(rows)) or np.any(rows<0)):
            raise ValueError('record_rows must contain known nonnegative integer row IDs')
        rows=rows.astype(int);R=int(rows.max())+1
        if (rowmap.shape!=(R,) or not np.isfinite(rowmap).all() or np.any(rowmap!=np.floor(rowmap))
            or np.any(rowmap<0) or np.any(rowmap>=L) or L<2):raise ValueError('invalid row-to-submitted-label map')
        nr=np.bincount(rows,minlength=R)
        if np.any(nr==0):raise ValueError('row IDs must be consecutive and nonempty')
        if not 0<eta<=1:raise ValueError('eta must lie in (0,1]')
        starts=tuple(map(int,upper_starts))
        if len(starts)!=6 or starts[0]<0 or any(a>=b for a,b in zip(starts,starts[1:])):
            raise ValueError('six strictly increasing nonnegative predeclared upper starts required')
        self.record_rows=rows.copy();self.N=len(rows);self.L=int(L);self.R=R
        self.joint_budget=bool(joint_budget);self.alpha_tail=.0125 if joint_budget else .0225
        self.eta=float(eta);self.upper_starts=starts
        self._state=CompositionAudit(nr,rowmap.astype(int),L,alpha_coordinates=.02)
        self._state.update();self._row_rng=np.random.default_rng(seed);self._record_rng=np.random.default_rng(seed+7121)
        self._remaining=[list(np.flatnonzero(rows==r)) for r in range(R)]
        self._callback=predictor_callback;self._refit_at=frozenset(map(int,refit_at))
        if any(t<=0 for t in self._refit_at):raise ValueError('refit counts must be positive')
        self._fit_started=False;self._c=np.zeros(self.N);self._c_total=_point(0.)
        self._seen=[];self._labels=[];self._pending=None
        self._X=[];self._K=[];self._D=[];self._Yhi=[];self._maxhi=[]
        self._lower=1.;self._upper=float(L);self._last_certificate=None;self.last_predictor_error=None

    @property
    def reviews(self):return len(self._seen)

    def _snapshot(self):
        st=self._state
        return {'N':self.N,'L':self.L,'N_r':st.N_r.copy(),'A':st.A.copy(),
                'row_to_label':st.row_to_label.copy(),'du':st.du.copy(),
                'beta':st.beta.copy(),'reviews':self.reviews,'record_rows':self.record_rows.copy()}

    def _fit(self):
        self._fit_started=True
        if self._callback is None:return
        try:
            values=self._callback(np.array(self._seen,dtype=int),np.array(self._labels,dtype=int),self._snapshot())
            values=np.asarray(values,float)
            if values.shape!=(self.N,) or not np.isfinite(values).all():raise ValueError('predictor must return N finite scalar residual scores')
            remaining=np.concatenate([np.array(x,dtype=int) for x in self._remaining])
            total=_sum(_point(values[remaining]))
            self._c=values.copy();self._c_total=total;self.last_predictor_error=None
        except Exception as exc:
            # The observed audit remains recorded. Existing predictions remain
            # valid, so a failed learner cannot corrupt the statistical state.
            self.last_predictor_error=f'{type(exc).__name__}: {exc}'

    def prepare(self):
        """Choose the next row and a uniform remaining record before its reveal.

        Repeated calls return the same pending request. Do not selectively
        skip difficult records: supply a verified eligible label for this ID.
        """
        if self._pending is not None:
            p=self._pending;return {'record_id':p['id'],'row':p['r'],'row_probabilities':p['pi'].copy()}
        if self.reviews==self.N:return None
        st=self._state;u=st.A.sum(1);M=st.N_r-u;desired=st.probabilities()
        if self._fit_started:
            ids=np.array(self._seen,dtype=int);lab=np.array(self._labels,dtype=int);rr=self.record_rows[ids]
            base=(st.A+st.beta)/(u+1)[:,None]@st.du
            residual=st.du[lab]-base[rr]-self._c[ids]
            second=np.bincount(rr,weights=residual**2,minlength=self.R)
            variance=(second+10*st.sd**2)/(u+10)
            desired=floor_variance_allocation(M,M*np.sqrt(np.maximum(variance,1e-12)),epsilon=.5)
        r,pi,_=sample_dyadic_row(self._row_rng,desired,M>0)
        position=int(self._record_rng.integers(len(self._remaining[r])));i=int(self._remaining[r][position])
        cmin=np.array([min(self._c[ids],default=0.) for ids in self._remaining])
        cmax=np.array([max(self._c[ids],default=0.) for ids in self._remaining])
        frozen=SimpleNamespace(N=self.N,L=self.L,N_r=st.N_r.copy(),A=st.A.copy(),beta=st.beta.copy(),
                               dl=st.dl.copy(),bl=st.bl,du=st.du.copy(),bu=st.bu)
        self._pending={'id':i,'r':r,'position':position,'pi':pi,'state':frozen,
                       'c':float(self._c[i]),'total':self._c_total,'cmin':cmin,'cmax':cmax}
        return {'record_id':i,'row':r,'row_probabilities':pi.copy()}

    def observe(self,record_id,verified_label):
        """Record exactly the pending human-verified label; never accepts a full truth table."""
        p=self._pending
        if p is None or record_id!=p['id']:raise ValueError('record_id must match the pending prepare() request')
        if (not np.isfinite(verified_label) or verified_label!=int(verified_label)
            or not 0<=verified_label<self.L):raise ValueError('verified label must be an eligible integer label index')
        s=int(verified_label);r=p['r'];st=p['state']
        xl,_,kl,_,D=positive_observation(st,r,s,p['pi'],p['c'],p['total'],p['cmax'])
        ob=control_observation(st,r,s,p['pi'],p['c'],p['total'],p['cmin'],p['cmax'])
        self._X.append(xl);self._K.append(kl);self._D.append(D)
        self._Yhi.append(ob['entropy_yhi']);self._maxhi.append(ob['entropy_upper'])
        self._state.observation(r,s,p['pi'])
        ids=self._remaining[r];ids[p['position']]=ids[-1];ids.pop()
        self._seen.append(int(record_id));self._labels.append(s)
        self._c_total=_sub(self._c_total,_point(p['c']));self._pending=None
        t=self.reviews
        if t<self.N:
            if (t<=1000 and (t%100==0 or t==750)) or t%500==0:self._state.update()
            if t in self._refit_at:self._fit()
        return {'reviews':t,'record_id':int(record_id),'verified_label':s,'predictor_error':self.last_predictor_error}

    def interval(self,*,entropy_resolution=3e-4):
        """Return the certified P2 CS and clearly separated point estimates."""
        if not np.isfinite(entropy_resolution) or entropy_resolution<=0:raise ValueError('positive resolution required')
        t=self.reviews;cert=None
        if t==self.N:
            lo,hi,_=census_hill(self._state.A.sum(0),self.N);self._lower=lo;self._upper=hi
        elif t:
            X=np.array(self._X);K=np.array(self._K);D=np.array(self._D);observed=self._state.A.sum(0)
            model=ConvexEntropyProfile(X,K,D,self.N,alpha=self.alpha_tail,eta=self.eta)
            ans=model.endpoint(lower=observed/self.N,htol=entropy_resolution)
            h=ans['entropy_lower'];step=entropy_resolution
            point=np.array(ans['best']['p']) if ans['best'] else observed/self.N+(1-t/self.N)/self.L
            while h>0:
                cert=certify(X,K,D,self.N,point,observed,h,alpha=self.alpha_tail,eta=self.eta)
                if cert['certified']:break
                h=max(0.,h-step);step*=2
            lo=_fdown(_exp(_I(_D(h))).lo)
            alpha=float(np.nextafter(self.alpha_tail,0.))
            up=upper_endpoint(self._Yhi,self._maxhi,self.L,alpha=alpha,starts=self.upper_starts)
            self._lower=max(self._lower,lo);self._upper=min(self._upper,up['pillar_upper'])
        self._last_certificate=cert
        lo=self._lower;hi=self._upper;empty=lo>hi
        point=float(2*lo*hi/(lo+hi))
        # Guarantee for the ACTUAL represented report, not an ideal center.
        rel=max((_I(_D(point))/_I(_D(lo))-_I(1)).hi,(_I(1)-_I(_D(point))/_I(_D(hi))).hi)
        width=_fup((_I(_D(hi))-_I(_D(lo))).hi)
        return {'reviews':t,'N':self.N,'lower':lo,'upper':hi,'full_width':width,
                'estimated_corrected_pillar':self._state.estimate(),
                'partially_corrected_pillar':self._state.partial(),
                'harmonic_report':point,'harmonic_relative_error_bound':float('inf') if empty else _fup(rel),
                'numeric_verified':not empty,'empty':empty,'census':t==self.N,
                'confidence_at_least':.955,'joint_other_pillars':self.joint_budget,
                'lower_certificate':cert,'predictor_error':self.last_predictor_error}

    def should_stop(self,relative_tolerance=.05,*,result=None):
        if not np.isfinite(relative_tolerance) or relative_tolerance<=0:raise ValueError('positive tolerance required')
        result=self.interval() if result is None else result
        if result.get('reviews')!=self.reviews:raise ValueError('stale interval report')
        return bool(not result['empty'] and result['numeric_verified']
                    and result['harmonic_relative_error_bound']<=np.nextafter(float(relative_tolerance),0.))

    def pillars(self,groups,omega):
        """All original pillar bounds; real group/taxonomic metadata is required."""
        if not self.joint_budget:raise ValueError('initialize joint_budget=True for joint P1/P2/P3 coverage')
        if self.reviews==self.N:
            counts=self._state.A.sum(0);lo=hi=counts
        else:
            lo,hi,_=composition_count_bounds(self._state.N_r,self._state.A,self._state.coordinates.bounds())
        result=certify_composition_pillars(self.N,lo,hi,groups,omega)
        if len(groups)==1:
            p2=self.interval();result['P2']=(max(result['P2'][0],p2['lower']),min(result['P2'][1],p2['upper']))
            result['empty']=result['empty'] or result['P2'][0]>result['P2'][1]
        return result
