import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

class ConvexEntropyProfile:

    def __init__(self, X, K, D, N, ks=(0.03, 0.1, 0.3, 1.0), alpha=0.025, eta=0.875):
        self.X = np.asarray(X)
        self.K = np.asarray(K)
        self.D = np.asarray(D)
        self.N = int(N)
        self.L = self.X.shape[1]
        self.eps = 1 / self.N
        self.c = np.log1p(self.L / self.N)
        self.ks = np.array(ks)
        self.alpha = alpha
        self.eta = float(eta)
        self.threshold = np.log(1 / alpha)
        if self.X.min() < -1e-15:
            raise ValueError('Positive HT coefficients required')
        if len(self.K) != len(self.X) or len(self.D) != len(self.X):
            raise ValueError('Stream mismatch')
        self.last = None

    def at(self, h):
        self.h = float(h)
        self.m = h + self.c
        self.lam = np.minimum(self.ks[None, :], self.eta / (self.m + self.D[:, None]))
        self.last = None
        return self

    def calc(self, p):
        if self.last is not None and np.array_equal(self.last[0], p):
            return self.last[1:]
        d = -np.log((p + self.eps) / (1 + self.L * self.eps))
        Y = self.X @ d + self.K
        f = 1 + self.lam * (Y[:, None] - self.m)
        if f.min() <= 0:
            raise ArithmeticError('Invalid factor')
        lw = np.log(f).sum(axis=0)
        val = logsumexp(lw) - np.log(len(lw))
        mix = np.exp(lw - logsumexp(lw))
        allgrad = -(self.X.T @ (self.lam / f)) / (p[:, None] + self.eps)
        grad = allgrad @ mix
        self.last = (p.copy(), float(val), grad, float(f.min()))
        return self.last[1:]

    def value(self, p):
        return self.calc(p)[0]

    def grad(self, p):
        return self.calc(p)[1]

    def solve(self, h, lower=None, start=None, maxiter=500):
        self.at(h)
        L = self.L
        lower = np.zeros(L) if lower is None else np.asarray(lower, float)
        active = np.sum(self.X, axis=0) > 0
        active |= lower > 0
        inds = np.flatnonzero(active)
        if start is None:
            start = lower + (1 - lower.sum()) * self.X.sum(axis=0) / self.X.sum()
        else:
            start = np.maximum(np.asarray(start), lower)
            start = lower + (1 - lower.sum()) * (start - lower) / max(np.sum(start - lower), 1e-100)
        start[~active] = 0.0
        start = lower + (1 - lower.sum()) * (start - lower) / max(np.sum(start - lower), 1e-100)

        def full(v):
            p = np.zeros(L)
            p[inds] = v
            return p
        fit = minimize(lambda v: self.value(full(v)), start[inds], jac=lambda v: self.grad(full(v))[inds], method='SLSQP', bounds=[(lower[j], 1.0) for j in inds], constraints=[{'type': 'eq', 'fun': lambda v: np.sum(v) - 1, 'jac': lambda v: np.ones(len(v))}], options={'ftol': 1e-11, 'maxiter': maxiter})
        p = full(fit.x)
        f, g, mf = self.calc(p)
        affine = f - g @ p + g @ lower + (1 - lower.sum()) * g.min()
        return dict(h=float(h), value=f, affine_lower=float(affine), duality_gap=float(f - affine), p=p.tolist(), iterations=int(fit.nit), success=bool(fit.success), minfactor=mf)

    def endpoint(self, lower=None, start=None, htol=0.0002, maxiter=500):
        l = 0.0
        u = np.log(self.L)
        history = []
        best = None
        while u - l > htol:
            h = 0.5 * (l + u)
            ans = self.solve(h, lower, start, maxiter)
            start = np.array(ans['p'])
            history.append(ans)
            if ans['affine_lower'] > self.threshold:
                l = h
                best = ans
            else:
                u = h
        return dict(entropy_lower=l, pillar_lower=float(np.exp(l)), h_resolution=htol, threshold=self.threshold, alpha=self.alpha, certification='floating supporting-plane bound; needs independent outward certificate', best=best, history=history)
