"""Clinically-motivated engineered features that respect the acquisition structure.

DESIGN RULE (the thing most feature-engineering code gets wrong here):
every engineered feature is assigned to ONE test group, and ALL of its parent columns must
live in that group or in a FREE group (always observed). Otherwise, when the acquisition
policy has revealed group A but not group B, a feature built from both would silently leak B.
`engineer()` asserts this rule.

Everything is row-wise and deterministic (no target encoding, no fitted statistics), so it
is leakage-safe under CV by construction.
"""
import numpy as np
from data import norm

MALE_CODE = 1          # VERIFY in notebook 00 (print Sex value counts); data.py factorises sorted strings
RWMA_TO_VESSEL = None  # e.g. {1: "LAD", 2: "LCX", 3: "RCA"} ONLY after checking the UCI data dictionary


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def _egfr(cr, age, male):
    """CKD-EPI 2009 (no race term), creatinine in mg/dL."""
    k = np.where(male, 0.9, 0.7); a = np.where(male, -0.411, -0.329)
    r = cr / k
    return 141 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.209 * 0.993 ** age * np.where(male, 1.0, 1.018)


def _g(c, *keys):
    return [c[norm(k)] for k in keys]


# (feature name, test group, parent columns, function of the column dict)
SPECS = [
    ("risk_factor_count", "history", ["DM", "HTN", "Current Smoker", "FH", "DLP", "Obesity"],
     lambda c: sum(_g(c, "DM", "HTN", "Current Smoker", "FH", "DLP", "Obesity"))),
    ("age_x_male", "demographics", ["Age", "Sex"],
     lambda c: c[norm("Age")] * (c[norm("Sex")] == MALE_CODE)),
    ("angina_score", "symptoms", ["Typical Chest Pain", "Exertional CP", "LowTH Ang"],
     lambda c: sum(_g(c, "Typical Chest Pain", "Exertional CP", "LowTH Ang"))),
    ("tg_hdl", "lab_lipid", ["TG", "HDL"], lambda c: _div(*_g(c, "TG", "HDL"))),
    ("ldl_hdl", "lab_lipid", ["LDL", "HDL"], lambda c: _div(*_g(c, "LDL", "HDL"))),
    ("nlr", "lab_cbc_inflam", ["Neut", "Lymph"], lambda c: _div(*_g(c, "Neut", "Lymph"))),
    ("egfr", "lab_glucose_renal", ["CR", "Age", "Sex"],
     lambda c: _egfr(c[norm("CR")], c[norm("Age")], c[norm("Sex")] == MALE_CODE)),
    ("ecg_ischemia_count", "ecg", ["Q Wave", "St Elevation", "St Depression", "Tinversion"],
     lambda c: sum(_g(c, "Q Wave", "St Elevation", "St Depression", "Tinversion"))),
    ("ecg_repol_any", "ecg", ["St Elevation", "St Depression", "Tinversion"],
     lambda c: np.maximum.reduce(_g(c, "St Elevation", "St Depression", "Tinversion"))),
    ("ef_low", "echo", ["EF-TTE"],
     lambda c: np.where(np.isnan(c[norm("EF-TTE")]), np.nan, c[norm("EF-TTE")] < 50)),
    ("rwma_present", "echo", ["Region RWMA"],
     lambda c: np.where(np.isnan(c[norm("Region RWMA")]), np.nan, c[norm("Region RWMA")] > 0)),
    ("dysfunction_burden", "echo", ["EF-TTE", "Region RWMA"],
     lambda c: (100 - c[norm("EF-TTE")]) * (c[norm("Region RWMA")] > 0)),
]


def engineer(X, names, gcols, gnames, free):
    c = {norm(n): X[:, i] for i, n in enumerate(names)}
    col_group = {norm(names[i]): g for g, idx in enumerate(gcols) for i in idx}
    new_cols, new_names = [], []
    gcols2 = [list(i) for i in gcols]
    nxt = X.shape[1]

    def add(name, g, col):
        nonlocal nxt
        new_cols.append(np.asarray(col, float)); new_names.append(name)
        gcols2[g].append(nxt); nxt += 1

    for name, group, parents, fn in SPECS:
        if any(norm(p) not in c for p in parents):
            print(f"[features] skip {name}: missing parent"); continue
        tgt = gnames.index(group)
        allowed = set(free) | {tgt}
        bad = [p for p in parents if col_group[norm(p)] not in allowed]
        assert not bad, f"{name}: parents {bad} outside group '{group}' + free groups (leak risk)"
        add(name, tgt, fn(c))

    # data-driven RWMA region indicators (semantics-free until RWMA_TO_VESSEL is verified)
    k = norm("Region RWMA")
    if k in c:
        echo = gnames.index("echo")
        vals, cnt = np.unique(c[k][~np.isnan(c[k])], return_counts=True)
        for v, n_ in zip(vals, cnt):
            if v > 0 and n_ >= 5:
                add(f"rwma_is_{int(v)}", echo, np.where(np.isnan(c[k]), np.nan, c[k] == v))
        if RWMA_TO_VESSEL:
            for ves in ("LAD", "LCX", "RCA"):
                codes = [code for code, vv in RWMA_TO_VESSEL.items() if vv == ves]
                add(f"rwma_territory_{ves}", echo, np.where(np.isnan(c[k]), np.nan, np.isin(c[k], codes)))

    X2 = np.hstack([X, np.stack(new_cols, 1)]) if new_cols else X
    return X2, list(names) + new_names, gcols2, new_names
