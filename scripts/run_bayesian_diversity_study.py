#!/usr/bin/env python3
"""
Bayesian (model-based) audit of pooled diversity P2 = exp H.

Model (hierarchical Dirichlet-multinomial, Kim & Ghahramani 2012):
    theta_r          ~ Dir( kappa_r {beta e_r + (1-beta) gbar_t} + A_{r,t} )
    theta_{r,a} | th ~ Dir( kappa_c theta_r + A_{r,a,t} )
    unaudited labels of stratum (r,a) ~ Multinomial(M_{r,a,t}, theta_{r,a})
    gbar_t = (a_t + 1) / (t + L)          (pooled verified counts, add-one)

Frozen settings (Table 8):
    300 strata (submitted label x station), L = 50, 100 paths per policy,
    seeds 8000..8099, first evaluation 100, then every 25 up to 1000,
    J = 400 posterior draws, 2.25/97.75 percentiles, beta = 0.5,
    kappa_r = 2, kappa_c = 0.5, PAS = 0.2 prop + 0.8 Neyman, no pilot,
    stop when entropy width <= log(21/19).

Usage:
    python scripts/run_bayesian_diversity_study.py --csv data/private_kenya/labels.csv
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- settings
BETA = 0.5
KAPPA_R = 2.0
KAPPA_C = 0.5
J_DRAWS = 400
FIRST_EVAL = 100
EVAL_EVERY = 25
MAX_AUDITS = 1000
Q_LO, Q_HI = 2.25, 97.75            # alpha = 0.045
TARGET_WIDTH = np.log(21 / 19)      # 5% relative accuracy, harmonic centre
W_PROP_PAS, W_NEY_PAS = 0.2, 0.8


# ----------------------------------------------------------------- data
def load_population(csv_path, sub_col="original_label", ver_col="updated_label",
                    strat_col="station_name"):
    d = pd.read_csv(csv_path)
    vocab = sorted(set(d[sub_col]) | set(d[ver_col]))          # L = 50
    lab = {s: i for i, s in enumerate(vocab)}
    y = d[sub_col].map(lab).to_numpy()
    v = d[ver_col].map(lab).to_numpy()
    rows = np.unique(y)                                        # 46 submitted labels
    strata = d.groupby([sub_col, strat_col]).ngroup().to_numpy()  # 300 strata
    G = strata.max() + 1
    strat_row = np.empty(G, dtype=int)                         # vocab index of row
    strat_row[strata] = y
    row_index = {r: k for k, r in enumerate(rows)}
    strat_rowk = np.array([row_index[r] for r in strat_row])   # 0..R-1
    members = [np.flatnonzero(strata == g) for g in range(G)]
    return dict(vocab=vocab, y=y, v=v, rows=rows, G=G, L=len(vocab), N=len(y),
                strat_row=strat_row, strat_rowk=strat_rowk, members=members)


def entropy_from_counts(c, N):
    p = c / N
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p), 0.0)
    return -t.sum(axis=-1)


# ----------------------------------------------------------------- model pieces
class State:
    """Audit counts; everything the model and the policy need."""

    def __init__(self, pop):
        self.pop = pop
        G, R, L = pop["G"], len(pop["rows"]), pop["L"]
        self.A_g = np.zeros((G, L))       # verified counts per stratum
        self.A_r = np.zeros((R, L))       # verified counts per submitted row
        self.a = np.zeros(L)              # pooled verified counts
        self.M = np.array([len(m) for m in pop["members"]], dtype=float)  # unaudited
        self.t = 0

    def record(self, g, s):
        self.A_g[g, s] += 1
        self.A_r[self.pop["strat_rowk"][g], s] += 1
        self.a[s] += 1
        self.M[g] -= 1
        self.t += 1

    def row_prior(self):
        """kappa_r {beta e_r + (1-beta) gbar} + A_r, shape (R, L)."""
        pop = self.pop
        L = pop["L"]
        gbar = (self.a + 1.0) / (self.t + L)
        base = (1 - BETA) * np.tile(gbar, (len(pop["rows"]), 1))
        base[np.arange(len(pop["rows"])), pop["rows"]] += BETA
        return KAPPA_R * base + self.A_r

    def posterior_mean(self):
        """q_g = E[theta_{r,a} | audits], shape (G, L)."""
        alpha_r = self.row_prior()
        q_r = alpha_r / alpha_r.sum(1, keepdims=True)
        q_r_g = q_r[self.pop["strat_rowk"]]
        n_g = self.A_g.sum(1, keepdims=True)
        return (KAPPA_C * q_r_g + self.A_g) / (KAPPA_C + n_g)

    def posterior_H(self, rng, J=J_DRAWS):
        """J draws of H for completed datasets consistent with every audited verdict."""
        pop = self.pop
        alpha_r = self.row_prior()                                   # (R, L)
        th_r = rng.gamma(alpha_r, 1.0, size=(J,) + alpha_r.shape)  # (J, R, L)
        th_r /= th_r.sum(-1, keepdims=True)
        live = np.flatnonzero(self.M > 0)
        alpha_g = KAPPA_C * th_r[:, pop["strat_rowk"][live]] + self.A_g[live][None]
        th_g = rng.gamma(alpha_g, 1.0)                               # (J, G', L)
        s = th_g.sum(-1, keepdims=True)
        th_g = np.where(s > 0, th_g / np.maximum(s, 1e-300), 1.0 / pop["L"])
        n = np.broadcast_to(self.M[live].astype(np.int64), th_g.shape[:2])
        draws = rng.multinomial(n, th_g)                             # (J, G', L)
        counts = self.a[None] + draws.sum(1)                         # column totals
        return entropy_from_counts(counts, pop["N"])


def allocation(state, policy):
    """Stratum-selection probabilities pi_{g,t}."""
    M = state.M
    prop = M / M.sum()
    if policy == "srs":
        return prop
    pop = state.pop
    N, L = pop["N"], pop["L"]
    q = state.posterior_mean()                                       # (G, L)
    p_hat = (state.a + (M[:, None] * q).sum(0)) / N                  # predicted composition
    d = -np.log((N * p_hat + 1.0) / (N + L))                         # entropy direction
    mu = q @ d
    var = (q * (d[None] - mu[:, None]) ** 2).sum(1)
    n_g = state.A_g.sum(1)
    sd = np.sqrt(var * (1.0 + 1.0 / (n_g + KAPPA_C)))
    ney = M * sd
    ney = ney / ney.sum() if ney.sum() > 0 else prop
    pi = W_PROP_PAS * prop + W_NEY_PAS * ney
    pi[M <= 0] = 0.0
    return pi / pi.sum()


# ----------------------------------------------------------------- one path
def run_path(args):
    pop, seed, policy = args
    rng = np.random.default_rng(seed)
    # within-stratum record orders: drawn first, so they are identical across policies
    orders = [rng.permutation(m) for m in pop["members"]]
    ptr = np.zeros(pop["G"], dtype=int)
    H_true = entropy_from_counts(np.bincount(pop["v"], minlength=pop["L"]), pop["N"])

    st = State(pop)
    evals = []
    stop = None
    for t in range(1, MAX_AUDITS + 1):
        pi = allocation(st, policy)
        g = rng.choice(pop["G"], p=pi)
        i = orders[g][ptr[g]]
        ptr[g] += 1
        st.record(g, pop["v"][i])

        if t >= FIRST_EVAL and (t - FIRST_EVAL) % EVAL_EVERY == 0:
            Hs = st.posterior_H(rng)
            lo, hi = np.percentile(Hs, [Q_LO, Q_HI])
            evals.append(dict(t=t, lo=float(lo), hi=float(hi),
                              covers=bool(lo <= H_true <= hi)))
            if hi - lo <= TARGET_WIDTH:
                stop = evals[-1]
                break

    last = stop or evals[-1]
    return dict(seed=seed, policy=policy, stopped=stop is not None,
                t_stop=last["t"], H_lo=last["lo"], H_hi=last["hi"],
                P2_lo=float(np.exp(last["lo"])), P2_hi=float(np.exp(last["hi"])),
                cover_stop=last["covers"],
                cover_path=all(e["covers"] for e in evals),
                miss_side=(None if last["covers"] else
                           ("below" if last["hi"] < H_true else "above")),
                evals=evals)


# ----------------------------------------------------------------- driver
def summarize(res, H_true):
    out = {}
    for pol in ("pas", "srs"):
        r = [x for x in res if x["policy"] == pol]
        if not r:
            continue
        T = np.array([x["t_stop"] for x in r])
        out[pol] = dict(paths=len(r), mean=float(T.mean()), median=float(np.median(T)),
                        censored=sum(not x["stopped"] for x in r),
                        cover_stop=float(np.mean([x["cover_stop"] for x in r])),
                        cover_path=float(np.mean([x["cover_path"] for x in r])),
                        miss_below=sum(x["miss_side"] == "below" for x in r),
                        miss_above=sum(x["miss_side"] == "above" for x in r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/private_kenya/labels.csv")
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=8000)
    ap.add_argument("--policies", default="pas,srs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/kenya_bayesian/results.jsonl")
    a = ap.parse_args()

    pop = load_population(a.csv)
    H_true = float(entropy_from_counts(np.bincount(pop["v"], minlength=pop["L"]), pop["N"]))
    print(f"N={pop['N']} rows={len(pop['rows'])} strata={pop['G']} L={pop['L']} "
          f"H*={H_true:.4f} P2*={np.exp(H_true):.4f}")

    jobs = [(pop, a.seed0 + k, pol) for pol in a.policies.split(",")
            for k in range(a.paths)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(run_path, jobs, chunksize=1)
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")

    s = summarize(res, H_true)
    print(f"\n{'policy':<6} {'paths':>5} {'mean':>7} {'median':>7} {'cens':>5} "
          f"{'cov@stop':>9} {'cov path':>9} {'below':>6} {'above':>6}")
    for pol, x in s.items():
        print(f"{pol:<6} {x['paths']:>5} {x['mean']:>7.1f} {x['median']:>7.0f} "
              f"{x['censored']:>5} {x['cover_stop']:>9.2%} {x['cover_path']:>9.2%} "
              f"{x['miss_below']:>6} {x['miss_above']:>6}")
    print(f"\n{time.time() - t0:.0f}s, per-path results in {a.out}")


if __name__ == "__main__":
    main()
