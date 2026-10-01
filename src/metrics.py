"""Metrics with uncertainty: ECE, calibration slope, bootstrap CIs, paired model comparison."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ece(y, p, bins=8):
    """Equal-frequency ECE (n=303 is too small for 15 equal-width bins)."""
    o = np.argsort(p); y, p = y[o], p[o]
    chunks = np.array_split(np.arange(len(p)), bins)
    return float(sum(len(i) / len(p) * abs(y[i].mean() - p[i].mean()) for i in chunks))


def calib_slope(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def binary_metrics(y, p):
    if len(np.unique(y)) < 2:
        return {k: np.nan for k in ("auc", "prauc", "brier", "logloss", "ece", "slope")}
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auc": roc_auc_score(y, p), "prauc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, p),
            "ece": ece(y, p), "slope": calib_slope(y, p)[0]}


def bootstrap_ci(y, p, fn=roc_auc_score, B=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) == 2:
            v.append(fn(y[i], p[i]))
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


def paired_bootstrap(y, p1, p2, fn=roc_auc_score, B=1000, seed=0):
    """Delta = metric(p1) - metric(p2) on the SAME resampled patients. Returns mean, CI, two-sided p."""
    rng = np.random.default_rng(seed); n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) == 2:
            d.append(fn(y[i], p1[i]) - fn(y[i], p2[i]))
    d = np.array(d)
    return float(d.mean()), (float(np.quantile(d, .025)), float(np.quantile(d, .975))), \
        float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
