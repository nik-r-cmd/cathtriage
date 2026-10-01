import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
import data, features, checks
from acquire import stop_index, static_greedy_order
from evaluate import fit_conformal, pred_sets


def _fake_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    names, gcols, costs, gnames, i = [], [], [], [], 0
    for g, (cost, members) in data.GROUPS.items():
        idx = []
        for m in members:
            names.append(m); idx.append(i); i += 1
        gcols.append(idx); costs.append(cost); gnames.append(g)
    X = np.abs(rng.normal(size=(n, i))) + 0.1
    X[:, names.index("Sex")] = rng.integers(0, 2, n)
    X[:, names.index("Region RWMA")] = rng.integers(0, 4, n)
    return X, names, gcols, np.array(costs), gnames, [gnames.index(g) for g in data.FREE]


def test_conformal_marginal_coverage_per_class():
    rng = np.random.default_rng(1)
    def draw(n):
        p = rng.beta(2, 2, (n, 3)); y = (rng.random((n, 3)) < p).astype(int); return p, y
    pc, yc = draw(2000); pt, yt = draw(20000)
    S = pred_sets(pt, fit_conformal(pc, yc, alpha=0.1))
    for v in range(3):
        for c in (0, 1):
            m = yt[:, v] == c
            assert S[m, v, c].mean() >= 0.88          # class-conditional coverage >= 1 - alpha (minus MC noise)


def test_no_leak_assertion_fires():
    with pytest.raises(AssertionError):
        checks.assert_no_leak(["Age", "LAD"])
    checks.assert_no_leak(["Age", "BMI"])


def test_features_respect_group_closure_and_add_columns():
    X, names, gcols, costs, gnames, free = _fake_dataset()
    X2, names2, gcols2, added = features.engineer(X, names, gcols, gnames, free)
    assert X2.shape[1] == X.shape[1] + len(added) and len(added) >= 10
    assert sorted(c for g in gcols2 for c in g) == list(range(X2.shape[1]))


def test_feature_closure_violation_is_caught(monkeypatch):
    X, names, gcols, costs, gnames, free = _fake_dataset()
    bad = ("bad", "ecg", ["LDL"], lambda c: c["ldl"])      # LDL lives in lab_lipid, not ecg/free
    monkeypatch.setattr(features, "SPECS", features.SPECS + [bad])
    with pytest.raises(AssertionError):
        features.engineer(X, names, gcols, gnames, free)


def test_stop_index_monotone_in_tau():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(8), size=(50, 6))
    lo, hi = stop_index(P, 0.6), stop_index(P, 0.95)
    assert (hi >= lo).all()


def test_static_order_is_permutation_of_paid_groups():
    X, names, gcols, costs, gnames, free = _fake_dataset()
    y8 = np.random.default_rng(0).integers(0, 8, len(X))
    order = static_greedy_order(X, y8, gcols, costs, free)
    assert sorted(order) == [g for g in range(len(gcols)) if g not in free]


def test_permutation_check_near_chance():
    X, names, gcols, costs, gnames, free = _fake_dataset(n=250)
    rng = np.random.default_rng(3)
    Yv = (rng.random((250, 3)) < 0.5).astype(int); y8 = Yv[:, 0] * 4 + Yv[:, 1] * 2 + Yv[:, 2]
    assert abs(checks.permutation_check(X, y8, Yv) - 0.5) < 0.08
