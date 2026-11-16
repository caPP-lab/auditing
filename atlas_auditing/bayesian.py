import numpy as np
BETA = 0.5
KAPPA_R = 2.0
KAPPA_C = 0.5
J_DRAWS = 400
FIRST_EVAL = 100
EVAL_EVERY = 25
MAX_AUDITS = 1000
Q_LO, Q_HI = (2.25, 97.75)
TARGET_WIDTH = np.log(21 / 19)
W_PROP_PAS, W_NEY_PAS = (0.2, 0.8)

def entropy_from_counts(c, N):
    p = c / N
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(p > 0, p * np.log(p), 0.0)
    return -t.sum(axis=-1)

class State:

    def __init__(self, pop):
        self.pop = pop
        G, R, L = (pop['G'], len(pop['rows']), pop['L'])
        self.A_g = np.zeros((G, L))
        self.A_r = np.zeros((R, L))
        self.a = np.zeros(L)
        self.M = np.array([len(m) for m in pop['members']], dtype=float)
        self.t = 0

    def record(self, g, s):
        self.A_g[g, s] += 1
        self.A_r[self.pop['strat_rowk'][g], s] += 1
        self.a[s] += 1
        self.M[g] -= 1
        self.t += 1

    def row_prior(self):
        pop = self.pop
        L = pop['L']
        gbar = (self.a + 1.0) / (self.t + L)
        base = (1 - BETA) * np.tile(gbar, (len(pop['rows']), 1))
        base[np.arange(len(pop['rows'])), pop['rows']] += BETA
        return KAPPA_R * base + self.A_r

    def posterior_mean(self):
        alpha_r = self.row_prior()
        q_r = alpha_r / alpha_r.sum(1, keepdims=True)
        q_r_g = q_r[self.pop['strat_rowk']]
        n_g = self.A_g.sum(1, keepdims=True)
        return (KAPPA_C * q_r_g + self.A_g) / (KAPPA_C + n_g)

    def posterior_H(self, rng, J=J_DRAWS):
        pop = self.pop
        alpha_r = self.row_prior()
        th_r = rng.gamma(alpha_r, 1.0, size=(J,) + alpha_r.shape)
        th_r /= th_r.sum(-1, keepdims=True)
        live = np.flatnonzero(self.M > 0)
        alpha_g = KAPPA_C * th_r[:, pop['strat_rowk'][live]] + self.A_g[live][None]
        th_g = rng.gamma(alpha_g, 1.0)
        s = th_g.sum(-1, keepdims=True)
        th_g = np.where(s > 0, th_g / np.maximum(s, 1e-300), 1.0 / pop['L'])
        n = np.broadcast_to(self.M[live].astype(np.int64), th_g.shape[:2])
        draws = rng.multinomial(n, th_g)
        counts = self.a[None] + draws.sum(1)
        return entropy_from_counts(counts, pop['N'])

def allocation(state, policy):
    M = state.M
    prop = M / M.sum()
    if policy == 'srs':
        return prop
    pop = state.pop
    N, L = (pop['N'], pop['L'])
    q = state.posterior_mean()
    p_hat = (state.a + (M[:, None] * q).sum(0)) / N
    d = -np.log((N * p_hat + 1.0) / (N + L))
    mu = q @ d
    var = (q * (d[None] - mu[:, None]) ** 2).sum(1)
    n_g = state.A_g.sum(1)
    sd = np.sqrt(var * (1.0 + 1.0 / (n_g + KAPPA_C)))
    ney = M * sd
    ney = ney / ney.sum() if ney.sum() > 0 else prop
    pi = W_PROP_PAS * prop + W_NEY_PAS * ney
    pi[M <= 0] = 0.0
    return pi / pi.sum()

def run_path(args):
    pop, seed, policy = args
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(m) for m in pop['members']]
    ptr = np.zeros(pop['G'], dtype=int)
    H_true = entropy_from_counts(np.bincount(pop['v'], minlength=pop['L']), pop['N'])
    st = State(pop)
    evals = []
    stop = None
    for t in range(1, MAX_AUDITS + 1):
        pi = allocation(st, policy)
        g = rng.choice(pop['G'], p=pi)
        i = orders[g][ptr[g]]
        ptr[g] += 1
        st.record(g, pop['v'][i])
        if t >= FIRST_EVAL and (t - FIRST_EVAL) % EVAL_EVERY == 0:
            Hs = st.posterior_H(rng)
            lo, hi = np.percentile(Hs, [Q_LO, Q_HI])
            evals.append(dict(t=t, lo=float(lo), hi=float(hi), covers=bool(lo <= H_true <= hi)))
            if hi - lo <= TARGET_WIDTH:
                stop = evals[-1]
                break
    last = stop or evals[-1]
    return dict(seed=seed, policy=policy, stopped=stop is not None, t_stop=last['t'], H_lo=last['lo'], H_hi=last['hi'], P2_lo=float(np.exp(last['lo'])), P2_hi=float(np.exp(last['hi'])), cover_stop=last['covers'], cover_path=all((e['covers'] for e in evals)), miss_side=None if last['covers'] else 'below' if last['hi'] < H_true else 'above', evals=evals)
