"""Cost-aware sequential test acquisition (simulated retrospectively).

Key trick: the acquisition ORDER does not depend on the stopping threshold, so we run
each patient's full trajectory once and store the posterior after every step. Any stopping
rule (confidence tau) is then just 'pick the first step that satisfies it'. This gives the
whole cost-vs-accuracy frontier from a single pass.

Policies: 'voi' (ours), 'cheapest' (fixed cost order), 'random', 'given' (fixed list order).
"""
import numpy as np
from posterior import SubsetPosterior, risk_entropy, marginals


class Neighbors:
    """kNN over context rows, used as the generative model for 'what might group g show?'."""
    def __init__(self, Xctx):
        self.med = np.nanmedian(Xctx, 0)
        Xf = np.where(np.isnan(Xctx), self.med, Xctx)
        self.mu, self.sd = Xf.mean(0), Xf.std(0) + 1e-6
        self.Z = (Xf - self.mu) / self.sd

    def query(self, Xq, cols, K, rng):
        if len(cols) == 0:
            return rng.integers(0, len(self.Z), (len(Xq), K))
        Xf = np.where(np.isnan(Xq), self.med, Xq)
        Zq = (Xf - self.mu) / self.sd
        d = ((Zq[:, None, cols] - self.Z[None, :, cols]) ** 2).sum(-1)
        return np.argpartition(d, K, axis=1)[:, :K]


def _cols(key, gcols):
    return sorted(c for g, o in enumerate(key) if o for c in gcols[g])


def posterior_for(post, X, obs, gcols):
    out = np.zeros((len(X), 8))
    keys = [tuple(r) for r in obs]
    for key in set(keys):
        idx = [i for i, k in enumerate(keys) if k == key]
        out[idx] = post.predict(X[idx], _cols(key, gcols))
    return out


def choose_voi(post, nb, X, obs, gcols, costs, paid, cur_H, K, rng):
    """Expected entropy reduction per unit cost, patients bucketed by observed set."""
    choice = np.full(len(X), -1)
    keys = [tuple(r) for r in obs]
    for key in set(keys):
        idx = np.array([i for i, k in enumerate(keys) if k == key])
        remaining = [g for g in paid if not key[g]]
        if not remaining:
            continue
        S = _cols(key, gcols)
        near = nb.query(X[idx], S, K, rng)                                   # (m, K)
        best = np.full(len(idx), -np.inf)
        arg = np.full(len(idx), -1)
        for g in remaining:
            cols = sorted(S + gcols[g])
            Q = np.repeat(X[idx], K, axis=0)
            Q[:, gcols[g]] = nb.Z[near.reshape(-1)][:, gcols[g]] * nb.sd[gcols[g]] + nb.mu[gcols[g]]
            P = post.predict(Q, cols).reshape(len(idx), K, 8)
            gain = (cur_H[idx] - risk_entropy(P).mean(1)) / costs[g]
            upd = gain > best
            best[upd], arg[upd] = gain[upd], g
        choice[idx] = arg
    return choice


def trajectory(post, Xctx, X, gcols, costs, free, policy="voi", K=8, seed=0, order=None):
    rng = np.random.default_rng(seed)
    n, G = len(X), len(gcols)
    paid = [g for g in range(G) if g not in free]
    T = len(paid)
    obs = np.zeros((n, G), bool)
    obs[:, free] = True
    P = np.zeros((n, T + 1, 8)); C = np.zeros((n, T + 1)); seq = np.full((n, T), -1)
    P[:, 0] = posterior_for(post, X, obs, gcols)
    nb = Neighbors(Xctx) if policy == "voi" else None
    rand_orders = np.array([rng.permutation(paid) for _ in range(n)])
    fixed = sorted(paid, key=lambda g: costs[g]) if policy == "cheapest" else order
    for t in range(T):
        if policy == "voi":
            ch = choose_voi(post, nb, X, obs, gcols, costs, paid, risk_entropy(P[:, t]), K, rng)
        elif policy == "random":
            ch = rand_orders[:, t]
        else:
            ch = np.full(n, fixed[t])
        obs[np.arange(n), ch] = True
        seq[:, t] = ch
        C[:, t + 1] = C[:, t] + costs[ch]
        P[:, t + 1] = posterior_for(post, X, obs, gcols)
    return {"P": P, "C": C, "seq": seq}


def stop_index(P, tau):
    """First step where every vessel marginal is tau-confident; else last step.
    Uses tau only (never the conformal quantile) so calibration stays circularity-free."""
    m = marginals(P)                                    # (n, T+1, 3)
    conf = (np.maximum(m, 1 - m) >= tau).all(-1)       # (n, T+1)
    first = np.where(conf.any(1), conf.argmax(1), P.shape[1] - 1)
    return first


def static_greedy_order(Xctx, yctx, gcols, costs, free, seed=0):
    """Strong NON-adaptive baseline: forward-select test groups by CV log-loss gain per cost
    (logistic regression, context rows only). VOI must beat this to justify per-patient adaptivity."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    G = len(gcols)
    bits = [(yctx >> 2) & 1, (yctx >> 1) & 1, yctx & 1]

    def score(groups):
        cols = sorted(c for g in groups for c in gcols[g])
        tot = 0.0
        for y in bits:
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
            p = cross_val_predict(pipe, Xctx[:, cols], y, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                  method="predict_proba")[:, 1]
            tot += log_loss(y, p)
        return tot

    sel, order, remaining = list(free), [], [g for g in range(G) if g not in free]
    cur = score(sel)
    while remaining:
        best_g, best_gain, best_s = None, -np.inf, None
        for g in remaining:
            s = score(sel + [g]); gain = (cur - s) / costs[g]
            if gain > best_gain:
                best_g, best_gain, best_s = g, gain, s
        order.append(best_g); sel.append(best_g); remaining.remove(best_g); cur = best_s
    return order
