
"""Data loading for the Z-Alizadeh Sani (extension) dataset.

Column names differ slightly between the UCI xlsx and Kaggle csv, so everything is
matched on a normalised name and the loader prints anything it could not match.
VERIFY the printed report on first run, then freeze the names in configs/.
"""
import re
import numpy as np
import pandas as pd

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

# group -> (relative cost, columns). Costs are PLACEHOLDERS (relative units).
# Replace with INR from a public tariff (e.g. CGHS package rates) before the paper.
GROUPS = {
    "demographics": (0.0, ["Age", "Sex", "Weight", "Length", "BMI"]),
    "history":      (0.0, ["DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity", "CRF",
                           "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP"]),
    "symptoms":     (0.5, ["Typical Chest Pain", "Dyspnea", "Function Class", "Atypical",
                           "Nonanginal", "Exertional CP", "LowTH Ang"]),
    "exam_vitals":  (1.0, ["BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales",
                           "Systolic Murmur", "Diastolic Murmur"]),
    "ecg":          (3.0, ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH",
                           "Poor R Progression", "BBB"]),
    "lab_glucose_renal": (4.0, ["FBS", "CR", "BUN", "K", "Na", "HB"]),
    "lab_lipid":    (4.0, ["TG", "LDL", "HDL"]),
    "lab_cbc_inflam": (4.0, ["ESR", "WBC", "Lymph", "Neut", "PLT"]),
    "echo":         (8.0, ["EF-TTE", "Region RWMA", "VHD"]),
}
FREE = ["demographics", "history"]          # collected before any decision is made
TARGETS = ["LAD", "LCX", "RCA"]
LEAK = {"lad", "lcx", "rca", "cath"}         # never allowed in X

NEG = {"normal", "n", "no", "0", "0.0", "false", "none", "nonstenotic"}

def _binarise(s):
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype(int)
    return (~s.astype(str).str.strip().str.lower().isin(NEG)).astype(int)

def load(path):
    df = pd.read_excel(path) if str(path).endswith(("xlsx", "xls")) else pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    lookup = {norm(c): c for c in df.columns}

    # targets
    Yv = np.stack([_binarise(df[lookup[norm(t)]]) for t in TARGETS], 1)       # (n, 3)
    y8 = Yv[:, 0] * 4 + Yv[:, 1] * 2 + Yv[:, 2]                                 # joint state 0..7

    # features, grouped
    cols, gcols, names, costs, gnames = [], [], [], [], []
    for g, (cost, members) in GROUPS.items():
        idx = []
        for m in members:
            real = lookup.get(norm(m))
            if real is None:
                print(f"[data] WARNING not found: {g}/{m}")
                continue
            assert norm(real) not in LEAK, f"leak column in features: {real}"
            s = df[real]
            if not pd.api.types.is_numeric_dtype(s):                            # Y/N, Male/Fmale, ...
                codes, _ = pd.factorize(s.astype(str).str.strip(), sort=True)
                s = pd.Series(np.where(s.isna(), np.nan, codes), index=s.index)
            idx.append(len(cols)); cols.append(s.astype(float).values); names.append(real)
        gcols.append(idx); costs.append(cost); gnames.append(g)

    used = {norm(n) for n in names} | LEAK
    leftover = [c for c in df.columns if norm(c) not in used]
    print(f"[data] n={len(df)}  features={len(names)}  unassigned columns: {leftover}")
    X = np.stack(cols, 1)
    free = [gnames.index(g) for g in FREE]
    return X, y8, Yv, names, gcols, np.array(costs), gnames, free

if __name__ == "__main__":
    import sys
    X, y8, Yv, names, gcols, costs, gnames, free = load(sys.argv[1])
    print("joint state counts:", np.bincount(y8, minlength=8))
    print("vessel prevalence LAD/LCX/RCA:", Yv.mean(0).round(3))
