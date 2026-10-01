"""Shared CV helpers."""
import numpy as np

def strat_key(y8, n_splits):
    """Stratify on the joint vessel state; fold states with < n_splits members into a big one."""
    key = y8.copy()
    big = np.bincount(y8, minlength=8).argmax()
    for s in range(8):
        if 0 < (key == s).sum() < n_splits:
            key[key == s] = big
    return key
