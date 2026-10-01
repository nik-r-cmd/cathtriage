
"""Any-subset posterior over the 8 joint vessel states (LAD, LCX, RCA).

Conditioning on an observed subset = refitting the in-context learner on only those
columns. TabPFN 'fit' just stores the context, so this is cheap; predict is the cost.
Unobserved columns are NEVER passed to predict, so there is no hidden leakage.
"""
import gc
from collections import OrderedDict
import numpy as np

N_STATES = 8
STATE_BITS = np.array([[(s >> 2) & 1, (s >> 1) & 1, s & 1] for s in range(N_STATES)], float)  # (8,3)

def marginals(P):
    """(..., 8) joint -> (..., 3) per-vessel P(stenosis)."""
    return P @ STATE_BITS

def risk_entropy(P):
    """Sum of binary entropies of the three vessel marginals (decision-relevant uncertainty)."""
    m = np.clip(marginals(P), 1e-6, 1 - 1e-6)
    return -(m * np.log(m) + (1 - m) * np.log(1 - m)).sum(-1)

class SubsetPosterior:
    def __init__(self, X_ctx, y_ctx, backend="tabpfn", device="cuda", n_estimators=8, seed=0, max_cache=None):
        self.X, self.y = X_ctx, y_ctx
        self.backend, self.device, self.n_est, self.seed = backend, device, n_estimators, seed
        counts = np.bincount(y_ctx, minlength=N_STATES) + 1.0
        self.prior = counts / counts.sum()
        # TabPFN estimators may each hold a copy of the weights -> cap the cache to avoid GPU OOM
        self.max_cache = max_cache if max_cache is not None else (12 if backend == "tabpfn" else None)
        self._cache = OrderedDict()

    def _make(self):
        if self.backend == "tabpfn":
            from tabpfn import TabPFNClassifier
            return TabPFNClassifier(device=self.device, n_estimators=self.n_est,
                                    random_state=self.seed)
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.3, max_iter=3000))

    def predict(self, X, cols):
        cols = tuple(sorted(cols))
        if len(cols) == 0:
            return np.tile(self.prior, (len(X), 1))
        if cols in self._cache:
            self._cache.move_to_end(cols)
        else:
            m = self._make()
            m.fit(self.X[:, cols], self.y)
            self._cache[cols] = m
            if self.max_cache and len(self._cache) > self.max_cache:
                self._cache.popitem(last=False); gc.collect()
                try:
                    import torch; torch.cuda.empty_cache()
                except Exception:
                    pass
        m = self._cache[cols]
        proba = m.predict_proba(X[:, list(cols)])
        classes = m.classes_ if hasattr(m, "classes_") else m[-1].classes_
        out = np.full((len(X), N_STATES), 1e-4)
        out[:, classes] += proba
        return out / out.sum(1, keepdims=True)
