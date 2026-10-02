"""Full-feature benchmark: models x multi-label strategies, repeated stratified CV.

Targets order in all prediction arrays: [LAD, LCX, RCA, CAD]   (CAD := any vessel >= 50%)
Strategies:  indep  = one binary model per vessel
             joint8 = one 8-class model over joint vessel states (captures multi-vessel structure)
             chain  = classifier chain LAD -> LCX -> RCA
All preprocessing lives INSIDE the estimators, so it is refit per fold (no leakage).
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cv import strat_key
from metrics import binary_metrics, bootstrap_ci, paired_bootstrap
from posterior import STATE_BITS

TARGETS = ["LAD", "LCX", "RCA", "CAD"]


def make(model, n_classes, seed, device):
    if model.endswith("_sig"):      # post-hoc Platt scaling (binary tasks only; joint8 stays uncalibrated)
        base = make(model[:-4], n_classes, seed, device)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3) if n_classes == 2 else base
    if model == "lr":
        clf = (LogisticRegressionCV(Cs=8, cv=3, scoring="neg_log_loss", max_iter=5000)
               if n_classes == 2 else LogisticRegression(C=0.3, max_iter=5000))
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)
    if model == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=seed))
    if model == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=5,
                                  random_seed=seed, verbose=0, allow_writing_files=False)
    if model == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device=device, n_estimators=8, random_state=seed)
    raise ValueError(model)


def _proba(est, Xtr, ytr, Xte, n_classes):
    if len(np.unique(ytr)) < 2:
        out = np.zeros((len(Xte), n_classes)); out[:, int(ytr[0])] = 1; return out
    est.fit(Xtr, ytr)
    out = np.full((len(Xte), n_classes), 1e-4)
    out[:, np.asarray(est.classes_, int)] += est.predict_proba(Xte)
    return out / out.sum(1, keepdims=True)


def _fit_predict(model, strat, Xtr, y8tr, Vtr, Xte, seed, device):
    cad_tr = Vtr.max(1)
    cad = _proba(make(model, 2, seed, device), Xtr, cad_tr, Xte, 2)[:, 1]
    if strat == "joint8":
        P = _proba(make(model, 8, seed, device), Xtr, y8tr, Xte, 8)
        return np.c_[P @ STATE_BITS, 1 - P[:, 0]]
    if strat == "indep":
        v = [_proba(make(model, 2, seed, device), Xtr, Vtr[:, j], Xte, 2)[:, 1] for j in range(3)]
        return np.c_[np.stack(v, 1), cad]
    if strat == "chain":
        ch = ClassifierChain(make(model, 2, seed, device), order=[0, 1, 2], random_state=seed)
        ch.fit(Xtr, Vtr)
        return np.c_[ch.predict_proba(Xte), cad]
    raise ValueError(strat)


def benchmark(X, y8, Yv, combos, n_splits=5, n_repeats=3, seed=0, device="cuda"):
    """combos: list of (model, strategy). Returns {combo: array (n_repeats, n, 4)} of OOF probabilities."""
    key = strat_key(y8, n_splits)
    out = {c: np.zeros((n_repeats, len(X), 4)) for c in combos}
    for r in range(n_repeats):
        for f, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(X, key)):
            for (m, s) in combos:
                out[(m, s)][r, te] = _fit_predict(m, s, X[tr], y8[tr], Yv[tr], X[te], seed + r, device)
        print(f"repeat {r + 1}/{n_repeats} done")
    return out


def report(preds, Yv, B=500):
    T = np.c_[Yv, Yv.max(1)]
    rows = []
    for combo, P in preds.items():
        for t, nm in enumerate(TARGETS):
            ms = [binary_metrics(T[:, t], P[r, :, t]) for r in range(len(P))]
            row = {k: np.nanmean([m[k] for m in ms]) for k in ms[0]}
            lo, hi = bootstrap_ci(T[:, t], P[:, :, t].mean(0), roc_auc_score, B)
            rows.append({"model": combo[0], "strategy": combo[1], "target": nm, **row,
                         "auc_lo": lo, "auc_hi": hi})
    return pd.DataFrame(rows).round(3)


def compare(preds, Yv, a, b, target="CAD"):
    t = TARGETS.index(target); T = np.c_[Yv, Yv.max(1)]
    return paired_bootstrap(T[:, t], preds[a][:, :, t].mean(0), preds[b][:, :, t].mean(0))
