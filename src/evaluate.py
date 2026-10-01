"""Per-vessel Mondrian split conformal + metrics + repeated CV experiment runner."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from posterior import SubsetPosterior, marginals
from acquire import trajectory, stop_index, static_greedy_order
from cv import strat_key


# ---------- conformal (class-conditional, per vessel) ----------
def fit_conformal(p_cal, y_cal, alpha):
    q = np.zeros((3, 2))
    for v in range(3):
        for c in (0, 1):
            m = y_cal[:, v] == c
            s = np.sort(p_cal[m, v] if c == 0 else 1 - p_cal[m, v])   # nonconformity of the TRUE class
            n = len(s); k = int(np.ceil((n + 1) * (1 - alpha)))
            q[v, c] = np.inf if (k > n or n == 0) else s[k - 1]
    return q

def pred_sets(p, q):
    """bool (n, 3, 2): is class 0 / class 1 in the prediction set."""
    return np.stack([p <= q[:, 0], (1 - p) <= q[:, 1]], -1)


# ---------- clinical utility ----------
def net_benefit(y, p, thresholds):
    n = len(y); out = []
    for t in thresholds:
        tp = ((p >= t) & (y == 1)).sum(); fp = ((p >= t) & (y == 0)).sum()
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def evaluate(tr_cal, tr_te, ycal_v, yte_v, tau, alpha=0.1):
    def final(tr):
        i = stop_index(tr["P"], tau); r = np.arange(len(i))
        return tr["P"][r, i], tr["C"][r, i]
    Pc, _ = final(tr_cal); Pt, ct = final(tr_te)
    mt = marginals(Pt)
    S = pred_sets(mt, fit_conformal(marginals(Pc), ycal_v, alpha))
    row = {"tau": tau, "cost": ct.mean(), "cad_auc": _auc((yte_v.max(1)), 1 - Pt[:, 0])}
    for v, name in enumerate(["LAD", "LCX", "RCA"]):
        row[f"{name}_auc"] = _auc(yte_v[:, v], mt[:, v])
        row[f"{name}_brier"] = brier_score_loss(yte_v[:, v], mt[:, v])
        row[f"{name}_cov"] = S[np.arange(len(S)), v, yte_v[:, v]].mean()
    both, none = S.all(-1), ~S.any(-1)
    row["ambiguous_rate"] = both.mean(); row["empty_rate"] = none.mean()
    row["singleton_rate"] = (~both & ~none).mean()
    return row


def run(X, y8, Yv, gcols, costs, free, policies=("voi", "static", "cheapest", "random"),
        taus=(0.7, 0.8, 0.85, 0.9, 0.95, 0.99), backend="tabpfn", device="cuda",
        n_splits=5, n_repeats=3, alpha=0.1, K=8, seed=0):
    key = strat_key(y8, n_splits)
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for f, (tr, te) in enumerate(cv.split(X, key)):
        ctx, cal = train_test_split(tr, test_size=0.3, stratify=key[tr], random_state=seed + f)
        post = SubsetPosterior(X[ctx], y8[ctx], backend=backend, device=device, seed=seed)
        for pol in policies:
            kw, pname = {}, pol
            if pol == "static":
                kw = {"order": static_greedy_order(X[ctx], y8[ctx], gcols, costs, free, seed)}; pname = "given"
            tc = trajectory(post, X[ctx], X[cal], gcols, costs, free, pname, K, seed, **kw)
            tt = trajectory(post, X[ctx], X[te], gcols, costs, free, pname, K, seed, **kw)
            for tau in taus:
                r = evaluate(tc, tt, Yv[cal], Yv[te], tau, alpha)
                r.update(fold=f, policy=pol); rows.append(r)
        print(f"fold {f + 1}/{n_splits * n_repeats} done")
    return pd.DataFrame(rows)


def summarise(df):
    return df.groupby(["policy", "tau"]).agg(["mean", "std"]).round(3)
