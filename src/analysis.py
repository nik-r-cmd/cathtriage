"""Offline analysis of saved acquisition trajectories (CPU, seconds). No TabPFN needed.

1. stepwise():  AUC vs mean cumulative cost after t paid tests. Compares ORDERING quality only,
                independent of any stopping rule or conformal choice. This is the fair policy test.
2. frontier():  stopping-rule sweep (rule x tau) with conformal sets, in two calibration modes:
                'split' = the original per-fold calibration split (~25 patients per class: conservative),
                'cross' = cross-conformal: calibrate fold k on out-of-fold final posteriors pooled from the
                          other folds (Vovk 2015). Theory gives >= 1-2*alpha; in practice ~1-alpha, and the
                          sets are far tighter. Say this plainly in the paper.
3. paired_policy_test(): bootstrap over patients on AUC at matched step.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from posterior import marginals
from acquire import stop_index
from evaluate import fit_conformal, pred_sets


def _pool(trajs, policy):
    """Concatenate test-fold trajectories of one policy -> dict with idx, P (n,T+1,8), C (n,T+1), fold."""
    ts = [t for t in trajs if t["policy"] == policy]
    return {"idx": np.concatenate([t["te"] for t in ts]),
            "P": np.concatenate([t["tt"]["P"] for t in ts]),
            "C": np.concatenate([t["tt"]["C"] for t in ts]),
            "fold": np.concatenate([[t["fold"]] * len(t["te"]) for t in ts])}


def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def stepwise(trajs, Yv):
    rows = []
    for pol in sorted({t["policy"] for t in trajs}):
        d = _pool(trajs, pol); Y = Yv[d["idx"]]
        for t in range(d["P"].shape[1]):
            m = marginals(d["P"][:, t])
            rows.append({"policy": pol, "step": t, "cost": d["C"][:, t].mean(),
                         "CAD_auc": _auc(Y.max(1), 1 - d["P"][:, t, 0]),
                         "LAD_auc": _auc(Y[:, 0], m[:, 0]), "LCX_auc": _auc(Y[:, 1], m[:, 1]), "RCA_auc": _auc(Y[:, 2], m[:, 2])})
    return pd.DataFrame(rows).round(4)


def frontier(trajs, Yv, rules=("all", "cad"), taus=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
             alphas=(0.1, 0.2), modes=("split", "cross"), mondrian=(True, False)):
    rows = []
    for pol in sorted({t["policy"] for t in trajs}):
        d = _pool(trajs, pol); Y = Yv[d["idx"]]
        for rule in rules:
            for tau in taus:
                i = stop_index(d["P"], tau, rule); r = np.arange(len(i))
                Pf, cost = d["P"][r, i], d["C"][r, i]; M = marginals(Pf)
                for mode in modes:
                    for mond in mondrian:
                        for alpha in alphas:
                            S = np.zeros((len(M), 3, 2), bool)
                            if mode == "split":
                                for t in [t for t in trajs if t["policy"] == pol]:
                                    sel = np.where(d["fold"] == t["fold"])[0]
                                    ci = stop_index(t["tc"]["P"], tau, rule); cr = np.arange(len(ci))
                                    Mc = marginals(t["tc"]["P"][cr, ci])
                                    S[sel] = pred_sets(M[sel], fit_conformal(Mc, Yv[t["cal"]], alpha, mond))
                            else:
                                for f in np.unique(d["fold"]):
                                    te, ot = d["fold"] == f, d["fold"] != f
                                    S[te] = pred_sets(M[te], fit_conformal(M[ot], Y[ot], alpha, mond))
                            amb, emp = S.all(-1), ~S.any(-1)
                            row = {"policy": pol, "rule": rule, "tau": tau, "mode": mode, "mondrian": mond, "alpha": alpha,
                                   "cost": cost.mean(), "CAD_auc": _auc(Y.max(1), 1 - Pf[:, 0]),
                                   "singleton": (~amb & ~emp).mean(), "empty": emp.mean()}
                            for v, nm in enumerate(["LAD", "LCX", "RCA"]):
                                row[f"{nm}_cov"] = S[np.arange(len(S)), v, Y[:, v]].mean()
                                row[f"{nm}_single"] = (~amb[:, v] & ~emp[:, v]).mean()
                            rows.append(row)
    return pd.DataFrame(rows).round(4)


def paired_policy_test(trajs, Yv, a, b, step, B=1000, seed=0):
    """dAUC(CAD) between policies a and b after `step` paid tests, bootstrapping patients (same folds => paired)."""
    da, db = _pool(trajs, a), _pool(trajs, b)
    ia, ib = np.argsort(da["idx"]), np.argsort(db["idx"])
    y = Yv[da["idx"][ia]].max(1)
    pa, pb = 1 - da["P"][ia, step, 0], 1 - db["P"][ib, step, 0]
    rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        j = rng.integers(0, len(y), len(y))
        if len(np.unique(y[j])) == 2:
            d.append(roc_auc_score(y[j], pa[j]) - roc_auc_score(y[j], pb[j]))
    d = np.array(d)
    return float(d.mean()), (float(np.quantile(d, .025)), float(np.quantile(d, .975))), float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))
