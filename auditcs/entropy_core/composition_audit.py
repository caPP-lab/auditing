"""Auditing population composition without constructing a matrix confidence set.

Only observed counts, catalogue metadata, and actual sampling probabilities
enter the online class. Matrix-valued counts A are bookkeeping for predictable
stratum-specific predictions, not an inferential state space to enumerate.
The P1/P2/P3 entry point accepts certified TOTAL category-count intervals.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import time

import numpy as np

from efficiency_directional import (DirectionalAudit, EntropyBet, CoordinateBet,
    census_hill, _I, _D, _fdown, _fup, _entropy_term, _logsumexp,
    sample_dyadic_row, xlogy)
from certified_pillars import certify_pillars


def composition_count_bounds(N_r, A, extra=None):
    """Outward integer total-count box; no row confidence regions are needed."""
    N = int(sum(N_r))
    A = np.asarray(A)
    observed = A.sum(axis=0)
    remaining = N - int(A.sum())
    lower = observed.astype(np.int64).copy()
    upper = np.minimum(N, observed + remaining).astype(np.int64)
    if extra is not None:
        if np.shape(extra[0]) != observed.shape or np.shape(extra[1]) != observed.shape:
            raise ValueError("Composition bounds must contain one interval per category")
        for s in range(len(observed)):
            lower[s] = max(int(lower[s]), int((_I(_D(extra[0][s])) * _I(N)).lo.to_integral_value(rounding="ROUND_CEILING")))
            upper[s] = min(int(upper[s]), int((_I(_D(extra[1][s])) * _I(N)).hi.to_integral_value(rounding="ROUND_FLOOR")))
    empty = bool(np.any(lower > upper) or int(lower.sum()) > N or int(upper.sum()) < N)
    return lower, upper, empty


def composition_envelopes(N_r, A, beta, extra=None):
    """Affine entropy envelopes derived solely from composition intervals.

    Integer totals obey A_{+s} <= C_{+s} <= A_{+s}+N-t deterministically.
    Optional simultaneous column confidence sequences tighten these bounds.
    Represented slopes and outward intercepts make the lower secants rigorous;
    the upper bound is the entropy variational formula, valid on the simplex.
    """
    N = int(sum(N_r))
    L = A.shape[1]
    lower, upper, empty = composition_count_bounds(N_r, A, extra)
    if empty:
        return np.zeros(L), 0., np.zeros(L), _fup(_I(L).log().hi), True
    ds = np.zeros(L)
    bs = _I(0)
    for s, (cl, cu) in enumerate(zip(lower, upper)):
        l = (_I(int(cl)) / _I(N)).lo
        u = (_I(int(cu)) / _I(N)).hi
        hl = _entropy_term(_I(l))
        hu = _entropy_term(_I(u))
        if u > l:
            ds[s] = float((hu.lo - hl.lo) / (u - l))
        intercept = min((hl - _I(_D(ds[s])) * _I(l)).lo,
                        (hu - _I(_D(ds[s])) * _I(u)).lo)
        bs += _I(intercept)
    rem = np.asarray(N_r) - A.sum(axis=1)
    ph = (A + rem[:, None] * (A + beta) /
          (A.sum(axis=1) + beta.sum(axis=1))[:, None]).sum(axis=0) / N
    au = -np.log(np.maximum(ph, 1e-12))
    zu = _fup(_logsumexp(-au).hi)
    return ds, _fdown(bs.lo), au, zu, False


def certify_composition_pillars(N, count_lower, count_upper, groups, omega):
    """Propagate certified integer composition bounds to all ORIGINAL pillars.

    If these bounds cover the true total counts simultaneously over time with
    probability >=1-alpha, all returned pillar intervals inherit that event.
    This function itself spends no probability budget and enumerates nothing.
    Groups and omega must be provided by the application, not inferred here.
    """
    N = int(N)
    lower = np.asarray(count_lower)
    upper = np.asarray(count_upper)
    if (N <= 0 or N > 2**53 or lower.ndim != 1 or lower.shape != upper.shape
            or not len(lower) or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower != np.floor(lower)) or np.any(upper != np.floor(upper))
            or np.any(lower < 0) or np.any(upper < 0) or np.any(lower > N) or np.any(upper > N)):
        raise ValueError("Expected finite integer total-count bounds in [0,N], with 0<N<=2**53")
    # The existing deterministic propagation accepts a one-row adapter plus
    # already-certified coordinate bounds. Its row.coordinate_bounds branch
    # is never reached, and no RowPPR object is constructed.
    adapter = SimpleNamespace(N=N, L=len(lower), u=0, A=np.zeros(len(lower), dtype=np.int64))
    answer = certify_pillars([adapter], N, groups, omega,
                            coordinate_lower=lower.astype(float)[None, :],
                            coordinate_upper=upper.astype(float)[None, :])
    answer["method"] = "certified composition propagation; no matrix region or enumeration"
    return answer


class CompositionAudit(DirectionalAudit):
    """Pilot-optional composition CS with predictable priority sampling.

    With the defaults, simultaneous composition error is <=.020 and the two
    direct entropy tests together spend <=.025. Thus their joint failure
    probability is <=.045; .005 is intentionally unused for controlled
    comparisons with the previous method's allocation of .050.

    Call update() before the first draw and at predetermined or past-measurable
    refresh times. Before EACH reveal, compute probabilities(), implement the
    actual randomized draw, and pass the actual probabilities to observation().
    Uniform sampling without replacement inside the selected row is required.
    No initial audit sample is required. Audits already available at creation
    may populate A, but only FUTURE predictable observations enter new bets.
    """
    def __init__(self, N_r, row_to_label, L, alpha_coordinates=.020,
                 alpha_bet=.025, explore=.5, restarts=True):
        raw = np.asarray(N_r)
        if (raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all()
                or np.any(raw <= 0) or np.any(raw != np.floor(raw))
                or int(raw.sum()) > 2**53 or L < 2):
            raise ValueError("Positive integer row sizes, total <=2**53, and L>=2 required")
        if not (0 < explore <= 1 and 0 < alpha_coordinates < 1 and 0 < alpha_bet < 1
                and alpha_coordinates + alpha_bet < 1):
            raise ValueError("Invalid exploration fraction or error allocation")
        labels = np.asarray(row_to_label)
        if (labels.shape != raw.shape or np.any(labels != np.floor(labels))
                or np.any(labels < 0) or np.any(labels >= L)):
            raise ValueError("One valid submitted category index is required per row")
        self.N_r = raw.astype(np.int64)
        self.N = int(sum(self.N_r))
        self.R = len(raw)
        self.L = int(L)
        self.row_to_label = labels.astype(int)
        self.A = np.zeros((self.R, self.L), dtype=np.int64)
        self.beta = np.full((self.R, self.L), .5/(self.L-1))
        self.beta[np.arange(self.R), self.row_to_label] = .5
        self.bet = EntropyBet(self.L, alpha_bet, restarts=restarts)
        self.coordinates = CoordinateBet(self.L, alpha_coordinates)
        self.explore = float(explore)
        self.dl = np.zeros(self.L)
        self.bl = 0.
        self.du = np.zeros(self.L)
        self.bu = _fup(_I(self.L).log().hi)
        self.sd = np.ones(self.R)
        self.bad = False

    def update(self):
        extra = self.coordinates.bounds()
        self.dl, self.bl, self.du, self.bu, self.bad = composition_envelopes(
            self.N_r, self.A, self.beta, extra)
        pred = (self.A+self.beta)/(self.A.sum(axis=1)+1)[:, None]
        means = pred @ self.du
        self.sd = np.sqrt(np.maximum(pred @ (self.du**2)-means**2, 1e-12))

    def certify_pillars(self, groups, omega, refresh=True):
        """All three simultaneous pillar bounds, from scalar composition CSs."""
        extra = self.coordinates.bounds() if refresh else (self.coordinates.lo, self.coordinates.hi)
        lower, upper, _ = composition_count_bounds(self.N_r, self.A, extra)
        result = certify_composition_pillars(self.N, lower, upper, groups, omega)
        if len(groups) == 1 and not result['empty']:
            # Both intervals are valid on the joint .020+.025 event. Keeping
            # their intersection cannot spend additional error probability.
            elo, ehi = self.bet.interval()
            result['P2'] = (max(result['P2'][0], elo), min(result['P2'][1], ehi))
            result['empty'] = self.bet.crossed or result['P2'][0] > result['P2'][1]
        return result


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = [100,250,500,748,1000,1500,2000,2500,3000,4000,5000,6000,7500,10000]


def benchmark(seed=1730, repetitions=3):
    """Compare envelope computation on exactly the same frozen audit states.

    The legacy PPR method is called ONLY in this retrospective benchmark, never
    in CompositionAudit or certify_composition_pillars. Wall-clock time includes
    ordinary shared-machine contention and should not be read as human effort.
    """
    from efficiency_directional import envelopes as legacy_envelopes
    pop = np.load(Path(__file__).with_name('replay_population.npz'), allow_pickle=False)
    N_r = pop['N_r']
    L = len(pop['labels'])
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed+7121)
    orders = [rng.permutation(pop['v_label_index'][pop['y_row_index']==r]) for r in range(len(N_r))]
    state = CompositionAudit(N_r,pop['row_to_label_index'],L)
    state.update()
    records = []
    t = 0
    for goal in [748,5000]:
        while t < goal:
            desired = state.probabilities()
            r,pi,_ = sample_dyadic_row(arng,desired,state.N_r-state.A.sum(axis=1)>0)
            u = int(state.A[r].sum())
            state.observation(r,int(orders[r][u]),pi)
            t += 1
            if t%500==0:
                state.update()
        before = time.perf_counter()
        extra = state.coordinates.bounds()
        coordinate_seconds = time.perf_counter()-before
        legacy = []
        composition = []
        for _ in range(repetitions):
            before = time.perf_counter()
            legacy_envelopes(state.N_r,state.A,state.beta,.005,extra)
            legacy.append(time.perf_counter()-before)
            before = time.perf_counter()
            composition_envelopes(state.N_r,state.A,state.beta,extra)
            composition.append(time.perf_counter()-before)
        records.append(dict(t=t,N=state.N,R=state.R,L=state.L,repetitions=repetitions,
            coordinate_cs_seconds=coordinate_seconds,legacy_envelope_seconds=legacy,
            composition_envelope_seconds=composition,
            median_legacy_seconds=float(np.median(legacy)),
            median_composition_seconds=float(np.median(composition)),
            median_ratio=float(np.median(legacy)/np.median(composition))))
    return dict(seed=seed,records=records,
        scope="Same process, same frozen observed audit counts and scalar bounds; envelope computation only, excluding scalar-CS updates and human review time")


def replay(seed, out, explore=.5, checkpoints=CHECKPOINTS, max_audits=10000):
    """Fixed-population retrospective evaluation; truth is isolated here."""
    pop = np.load(Path(__file__).with_name('replay_population.npz'), allow_pickle=False)
    N_r = pop['N_r']
    L = len(pop['labels'])
    N = int(sum(N_r))
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed+7121)
    orders = [rng.permutation(pop['v_label_index'][pop['y_row_index']==r]) for r in range(len(N_r))]
    state = CompositionAudit(N_r, pop['row_to_label_index'], L, explore=explore)
    start = time.perf_counter()
    state.update()
    true_lo, true_hi, true = census_hill(pop['C_star'].sum(axis=0), N)
    ptrue = pop['C_star'].sum(axis=0)/N
    H = -float(xlogy(ptrue,ptrue).sum())
    t = 0
    for goal in sorted(set([v for v in checkpoints if v <= max_audits] + [min(N,max_audits)])):
        while t < goal:
            desired = state.probabilities()
            r, pi, _ = sample_dyadic_row(arng, desired, state.N_r-state.A.sum(axis=1)>0)
            u = int(state.A[r].sum())
            state.observation(r, int(orders[r][u]), pi)
            t += 1
            if t % 500 == 0 and t < N:
                state.update()
        if t == N:
            lo, hi, _ = census_hill(state.A.sum(axis=0),N)
        elif state.bad:
            lo, hi = 1., float(L)
        else:
            lo, hi = state.bet.interval()
        row = dict(seed=seed,t=t,lo=lo,hi=hi,width=hi-lo,estimate=state.estimate(),
                   partial=state.partial(),true=true,coverage=lo<=true_lo and true_hi<=hi,
                   lower_gap=H-state.bl-float(state.dl@ptrue),upper_gap=state.bu+float(state.du@ptrue)-H,
                   seconds=time.perf_counter()-start,bad=state.bad,explore=explore,
                   crossed=state.bet.crossed,
                   numeric_verified=True,matrix_free=True,pilot=0,restarts=True,
                   alpha_coordinates=.020,alpha_entropy=.025)
        out.write(json.dumps(row)+'\n')
        out.flush()
    return time.perf_counter()-start


def summarize(output_directory=None):
    """Aggregate fixed-seed replay paths; uncertainty here is randomization only."""
    directory = ROOT/'output'/'publication' if output_directory is None else Path(output_directory)
    records = [json.loads(line) for path in sorted(directory.glob('composition_zero_pilot*.jsonl'))
               for line in path.read_text().splitlines() if line.strip()]
    seeds = list(range(1730,1750))
    mapping = {(r['seed'],r['t']):r for r in records}
    expected = {(s,t) for s in seeds for t in CHECKPOINTS}
    if set(mapping)!=expected or len(records)!=len(expected):
        raise ValueError(f'Expected twenty complete paths; missing {sorted(expected-set(mapping))}, records={len(records)}')
    truth = records[0]['true']
    result = dict(N=33403,R=46,L=50,seeds=seeds,pilot=0,true_pooled_P2=truth,
                  coordinate_error_budget=.020,entropy_error_budget=.025,
                  total_failure_probability_bound=.045,
                  checkpoints=CHECKPOINTS,checkpoint_coverage_count=sum(r['coverage'] for r in records),
                  checkpoint_count=len(records),path_coverage_count=sum(all(mapping[s,t]['coverage'] for t in CHECKPOINTS) for s in seeds),
                  empty_or_crossed_count=sum(r['bad'] or r.get('crossed',False) for r in records),
                  stopping={},point_estimation={},
                  scope='Fixed catalogue and fixed finite population; twenty replay randomization paths, not external populations')
    for width in [1.,.5]:
        times = []
        for seed in seeds:
            hits = [t for t in CHECKPOINTS if mapping[seed,t]['width']<=width
                    and not mapping[seed,t]['bad'] and not mapping[seed,t].get('crossed',False)]
            times.append(min(hits) if hits else None)
        result['stopping'][str(width)] = dict(full_width=width,midpoint_absolute_error=width/2,
            midpoint_error_percent_truth=100*width/(2*truth),audit_counts=times,
            mean_audits=float(np.mean(times)) if all(v is not None for v in times) else None,
            censored_by_10000=sum(v is None for v in times))
    for error in [.5,.25]:
        times=[]
        for seed in seeds:
            hits=[t for t in CHECKPOINTS if max(mapping[seed,t]['estimate']-mapping[seed,t]['lo'],
                 mapping[seed,t]['hi']-mapping[seed,t]['estimate'])<=error
                 and not mapping[seed,t]['bad'] and not mapping[seed,t].get('crossed',False)]
            times.append(min(hits) if hits else None)
        result['stopping']['plugin_'+str(error)]=dict(estimate_absolute_error=error,audit_counts=times,
            mean_audits=float(np.mean(times)) if all(v is not None for v in times) else None,
            censored_by_10000=sum(v is None for v in times))
    for t in CHECKPOINTS:
        subset=[mapping[s,t] for s in seeds]
        mae=float(np.mean([abs(r['estimate']-truth) for r in subset]))
        mean_width=float(np.mean([r['width'] for r in subset]))
        result['point_estimation'][str(t)]=dict(mean_estimate=float(np.mean([r['estimate'] for r in subset])),
            mean_partial_score=float(np.mean([r['partial'] for r in subset])),
            mean_absolute_error=mae,mean_absolute_error_percent_truth=100*mae/truth,
            mean_width=mean_width,mean_halfwidth_percent_truth=50*mean_width/truth,
            median_seconds=float(np.median([r['seconds'] for r in subset])))
    (directory/'composition_zero_pilot_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11.7,4.5),layout='constrained')
    times=np.asarray(CHECKPOINTS)
    one=[mapping[1730,t] for t in CHECKPOINTS]
    axes[0].fill_between(times,[r['lo'] for r in one],[r['hi'] for r in one],
                         color='#4C78A8',alpha=.20,step='post',label='Anytime interval (one replay)')
    axes[0].plot(times,[r['estimate'] for r in one],color='#1F5D94',lw=2,label='Estimated corrected score')
    axes[0].plot(times,[r['partial'] for r in one],color='#BE7527',lw=1.8,ls=':',label='Partially corrected table')
    axes[0].axhline(truth,color='#292929',ls='--',lw=1.4,label=f'True score = {truth:.5f}')
    axes[0].set(title='Pooled $P_2$: zero-pilot composition auditing',ylabel='Pillar value',ylim=(0.7,13.1))
    axes[0].legend(fontsize=8.5,loc='upper right',frameon=False)
    axes[0].text(.98,.64,'Before auditing: interval [1, 50]',transform=axes[0].transAxes,
                 ha='right',va='center',fontsize=8.5,color='#555555')
    stats=[result['point_estimation'][str(t)] for t in CHECKPOINTS]
    axes[1].plot(times,[r['mean_absolute_error_percent_truth'] for r in stats],
                 color='#1F5D94',lw=2,marker='o',ms=3,label='Mean absolute estimation error')
    axes[1].plot(times,[r['mean_halfwidth_percent_truth'] for r in stats],
                 color='#9B4F7D',lw=2,marker='s',ms=3,label='Mean interval half-width')
    axes[1].axhline(5,color='#777777',ls=':',lw=1,label='5% of true score')
    axes[1].set(title='Accuracy and conservatism across 20 replays',ylabel='Percentage of true pillar value',yscale='log')
    axes[1].legend(fontsize=8.5,loc='upper right',frameon=False)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('Audited records (zero pilot)')
        ax.set_xticks([100,250,500,1000,2500,5000,10000])
        ax.set_xticklabels(['100','250','500','1,000','2,500','5,000','10,000'],rotation=30,ha='right')
        ax.grid(axis='y',alpha=.18)
    fig.savefig(directory/'composition_convergence.png',dpi=220)
    fig.savefig(directory/'composition_convergence.pdf')
    plt.close(fig)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-start', type=int, default=1730)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--max-audits', type=int, default=10000)
    ap.add_argument('--explore', type=float, default=.5)
    ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--output', default='output/publication/composition_zero_pilot.jsonl')
    args = ap.parse_args()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    if args.summarize:
        result=summarize()
        print(json.dumps(result),flush=True)
    elif args.benchmark:
        result = benchmark(args.seed_start)
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    else:
        with open(args.output,'w') as out:
            for seed in range(args.seed_start,args.seed_start+args.seeds):
                elapsed = replay(seed,out,args.explore,max_audits=args.max_audits)
                print(json.dumps(dict(seed=seed,seconds=elapsed)),flush=True)
