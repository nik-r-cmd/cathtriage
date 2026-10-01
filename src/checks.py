"""Sanity checks that make results believable. Run these BEFORE trusting any number."""
import numpy as np
import pandas as pd
from baselines import benchmark, TARGETS
from metrics import binary_metrics

LEAK = {"lad", "lcx", "rca", "cath", "cad"}


def assert_no_leak(names):
    from data import norm
    bad = [n for n in names if norm(n) in LEAK]
    assert not bad, f"target-derived columns in features: {bad}"


def _mean_auc(preds, Yv):
    T = np.c_[Yv, Yv.max(1)]
    return float(np.mean([binary_metrics(T[:, t], P[0, :, t])["auc"] for P in preds.values() for t in range(4)]))


def permutation_check(X, y8, Yv, seed=0, combo=("lr", "indep")):
    """Shuffle labels jointly. A leak-free harness must return AUC ~ 0.5 (expect 0.5 +/- ~0.05 at n=303)."""
    rng = np.random.default_rng(seed); p = rng.permutation(len(X))
    pr = benchmark(X, y8[p], Yv[p], [combo], n_repeats=1, seed=seed, device="cpu")
    return _mean_auc(pr, Yv[p])


def leakage_canary(X, y8, Yv, seed=0, combo=("lr", "indep")):
    """Append a noisy copy of the target as a fake 'Cath' column. The harness MUST light up (AUC >> real)."""
    rng = np.random.default_rng(seed)
    Xl = np.c_[X, Yv.max(1) + rng.normal(0, 0.1, len(X))]
    pr = benchmark(Xl, y8, Yv, [combo], n_repeats=1, seed=seed, device="cpu")
    return _mean_auc(pr, Yv)


def subgroup_report(P, Yv, masks):
    """P: (n,4) OOF probs (mean over repeats). masks: {name: bool array}. AUC/Brier/ECE + prevalence vs mean pred."""
    T = np.c_[Yv, Yv.max(1)]; rows = []
    for g, m in masks.items():
        for t, nm in enumerate(TARGETS):
            r = binary_metrics(T[m, t], P[m, t])
            rows.append({"group": g, "n": int(m.sum()), "target": nm, "prev": T[m, t].mean(),
                         "mean_pred": P[m, t].mean(), **{k: r[k] for k in ("auc", "brier", "ece")}})
    return pd.DataFrame(rows).round(3)


def learning_curve(X, y8, Yv, combo=("lr", "indep"), fracs=(0.4, 0.6, 0.8, 1.0), seed=0, device="cpu"):
    """Does more data still help? If the curve is flat, model choice isn't the bottleneck."""
    rng = np.random.default_rng(seed); out = []
    for f in fracs:
        i = rng.permutation(len(X))[: int(f * len(X))]
        pr = benchmark(X[i], y8[i], Yv[i], [combo], n_repeats=1, seed=seed, device=device)
        out.append({"frac": f, "n": len(i), "mean_auc": _mean_auc(pr, Yv[i])})
    return pd.DataFrame(out).round(3)
