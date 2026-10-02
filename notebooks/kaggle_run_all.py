# =====================================================================================
# cathtriage - ONE Kaggle notebook, "Save & Run All" ready.   '# %%' = one cell.
# Settings: Accelerator = GPU T4/P100, Internet = ON.
# Kaggle Secrets (Add-ons -> Secrets):  TABPFN_TOKEN (required for TabPFN), GITHUB_TOKEN (only if repo is private)
# Each stage is wrapped so one failure prints a traceback but does NOT kill the later cells.
# Outputs land in /kaggle/working/cathtriage/results/  -> download from the Output tab, commit to repo/results/.
# =====================================================================================

# %% [CELL 1] install dependencies
import os, sys, subprocess
if not os.environ.get("SKIP_INSTALL"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn", "catboost", "shap",
                    "openpyxl", "pytest", "scikit-learn", "matplotlib"], check=False)
print("deps ok")

# %% [CELL 2] clone / update repo, paths, secrets
import os, sys, subprocess, json, time, warnings, contextlib, traceback
warnings.filterwarnings("ignore")
GITHUB_URL = os.environ.get("GITHUB_URL", "https://github.com/<your-username>/cathtriage.git")   # <-- EDIT
REPO    = os.environ.get("REPO", "/kaggle/working/cathtriage")
SRC, RESULTS, DATA_DIR = f"{REPO}/src", f"{REPO}/results", f"{REPO}/data"
QUICK   = os.environ.get("QUICK", "0") == "1"       # QUICK=1 -> 1 repeat / 3 folds smoke mode
REPS    = 1 if QUICK else 5

gh_token = None
try:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    try: os.environ["TABPFN_TOKEN"] = sec.get_secret("TABPFN_TOKEN")
    except Exception: print("no TABPFN_TOKEN secret found")
    try: gh_token = sec.get_secret("GITHUB_TOKEN")
    except Exception: pass
except Exception:
    pass

if not os.environ.get("SKIP_CLONE"):
    url = GITHUB_URL.replace("https://", f"https://{gh_token}@") if gh_token else GITHUB_URL
    if os.path.isdir(f"{REPO}/.git"):
        r = subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], capture_output=True, text=True)
    else:
        r = subprocess.run(["git", "clone", "--depth", "1", url, REPO], capture_output=True, text=True)
    print((r.stdout + r.stderr).replace(gh_token or "@@@", "***")[-400:])
sys.path.insert(0, SRC); os.makedirs(RESULTS, exist_ok=True)

@contextlib.contextmanager
def stage(name):
    t0 = time.time(); print(f"\n{'=' * 12} {name} {'=' * 12}")
    try: yield
    except Exception:
        print(f"!!!!!! {name} FAILED !!!!!!"); traceback.print_exc()
    finally: print(f"[{name}] {(time.time() - t0) / 60:.1f} min")

# %% [CELL 3] copy dataset, load, engineer features, check TabPFN, write run metadata
import shutil, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA_SRC = os.environ.get("DATA_SRC", "/kaggle/input/datasets/nikreddyvempalli/dataset/extention of Z-Alizadeh sani dataset.xlsx")
os.makedirs(DATA_DIR, exist_ok=True); DATA = f"{DATA_DIR}/zalizadeh_sani_ext.xlsx"   # data/ is gitignored: never redistribute
shutil.copy(DATA_SRC, DATA)

from data import load
from features import engineer
from baselines import benchmark, report, compare      # shared by NB01/NB02/NB04
from sklearn.metrics import roc_auc_score
X, y8, Yv, names, gcols, costs, gnames, free = load(DATA)
X2, names2, gcols2, added = engineer(X, names, gcols, gnames, free)
print("raw:", X.shape, "| engineered:", X2.shape, "| added:", added)

try:
    import torch; DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
TABPFN_OK = False
try:
    from tabpfn import TabPFNClassifier
    TabPFNClassifier(device=DEVICE, n_estimators=2).fit(X[:80, :6], Yv[:80, 0]).predict_proba(X[:5, :6])
    TABPFN_OK = True
except Exception as e:
    print("TabPFN UNAVAILABLE ->", type(e).__name__, str(e)[:300],
          "\n   Fix: log in at https://ux.priorlabs.ai, accept the licence, copy your API key, add it as Kaggle secret TABPFN_TOKEN."
          "\n   Falling back to the logreg backend for acquisition (results will NOT be the headline numbers).")
print("DEVICE:", DEVICE, "| TABPFN_OK:", TABPFN_OK)

import platform, sklearn
meta = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(), "sklearn": sklearn.__version__,
        "device": DEVICE, "tabpfn_ok": TABPFN_OK, "quick": QUICK, "n": int(len(X)), "reps": REPS,
        "git": subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}
for pkg in ("catboost", "tabpfn", "torch"):
    try: meta[pkg] = __import__(pkg).__version__
    except Exception: pass
json.dump(meta, open(f"{RESULTS}/run_meta.json", "w"), indent=2); print(meta)

# ---- freshness guard: fail in seconds (not after a 2-hour stage) if GitHub is behind your local files ----
import inspect, importlib
import evaluate as _ev, baselines as _bl, posterior as _po
_problems = []
if "save_traj" not in inspect.signature(_ev.run).parameters: _problems.append("src/evaluate.py (save_traj)")
if "_sig" not in inspect.getsource(_bl.make): _problems.append("src/baselines.py (*_sig models)")
if "max_cache" not in inspect.signature(_po.SubsetPosterior.__init__).parameters: _problems.append("src/posterior.py (max_cache)")
try: importlib.import_module("analysis")
except Exception: _problems.append("src/analysis.py (missing)")
if "objective" not in inspect.signature(importlib.import_module("acquire").trajectory).parameters: _problems.append("src/acquire.py (objective)")
assert not _problems, "STALE REPO - git add/commit/push these, then rerun: " + ", ".join(_problems)
print("code freshness OK")

# %% [NB00] sanity: tests, leak checks, label definition, RWMA eyeball
from checks import assert_no_leak, permutation_check, leakage_canary
with stage("NB00 sanity"):
    print("joint-state counts:", np.bincount(y8, minlength=8), "| prevalence LAD/LCX/RCA:", Yv.mean(0).round(3))
    raw = pd.read_excel(DATA)
    cath_cad = raw["Cath"].astype(str).str.strip().str.lower().eq("cad").astype(int).values
    mism = np.where(cath_cad != Yv.max(1))[0]
    print(f"rows where Cath-CAD != any-vessel>=50%: {mism.tolist()}")
    if len(mism): print(raw.loc[mism, ["Age", "LAD", "LCX", "RCA", "Cath"]])
    print("\nRWMA region vs P(stenosis) per vessel  (EDA ONLY - never used to build features):")
    print(pd.DataFrame({v: pd.Series(Yv[:, i]).groupby(raw["Region RWMA"].values).mean() for i, v in enumerate(["LAD", "LCX", "RCA"])}).round(2))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:warnings", "tests"], cwd=REPO, capture_output=True, text=True)
    print(r.stdout[-300:]); assert r.returncode == 0, "unit tests failed"
    assert_no_leak(names); assert_no_leak(names2)
    perm, canary = permutation_check(X, y8, Yv), leakage_canary(X, y8, Yv)
    print(f"permutation AUC (want ~0.50): {perm:.3f} | leak-canary CAD AUC (want ~1.0): {canary:.3f}")
    assert abs(perm - 0.5) < 0.07 and canary > 0.97, "harness sanity check failed - do NOT trust downstream numbers"

# %% [NB01] full-feature benchmark: models x strategies (+ TabPFN) with bootstrap CIs
from baselines import benchmark, report, compare
with stage("NB01 baselines"):
    combos = [(m, s) for m in ("lr", "rf", "catboost") for s in ("indep", "joint8", "chain")]
    preds = benchmark(X, y8, Yv, combos, n_splits=5, n_repeats=REPS, device="cpu")
    rep = report(preds, Yv); rep.to_csv(f"{RESULTS}/nb01_raw_features.csv", index=False)
    pickle.dump(preds, open(f"{RESULTS}/nb01_preds.pkl", "wb"))
    print(rep[rep.target == "CAD"].sort_values("auc", ascending=False).to_string())
    rows = []
    for m in ("lr", "rf", "catboost"):
        for t in ("LAD", "LCX", "RCA"):
            d, ci, p = compare(preds, Yv, (m, "joint8"), (m, "indep"), t)
            rows.append({"model": m, "vessel": t, "dAUC_joint8_minus_indep": d, "ci_lo": ci[0], "ci_hi": ci[1], "p": p})
    jt = pd.DataFrame(rows).round(3); jt.to_csv(f"{RESULTS}/nb01_joint_vs_indep.csv", index=False); print(jt.to_string())
with stage("NB01 TabPFN baselines"):
    if TABPFN_OK:
        tp = benchmark(X, y8, Yv, [("tabpfn", "indep"), ("tabpfn", "joint8")], n_splits=5, n_repeats=REPS, device=DEVICE)
        rep2 = report(tp, Yv); rep2.to_csv(f"{RESULTS}/nb01_tabpfn.csv", index=False)
        pickle.dump(tp, open(f"{RESULTS}/nb01_tabpfn_preds.pkl", "wb")); print(rep2[rep2.target == "CAD"].to_string())
    else:
        print("skipped (no TabPFN licence token)")

# %% [NB02] feature-engineering ablation: raw vs engineered, then drop-one-family
from sklearn.metrics import roc_auc_score
with stage("NB02 feature ablation"):
    combos = [("lr", "indep"), ("catboost", "indep"), ("catboost", "joint8")]
    base = benchmark(X,  y8, Yv, combos, n_repeats=REPS, device="cpu")
    eng  = benchmark(X2, y8, Yv, combos, n_repeats=REPS, device="cpu")
    pickle.dump({"raw": base, "eng": eng}, open(f"{RESULTS}/nb02_preds.pkl", "wb"))
    rows = []
    for c in combos:
        for t in ("CAD", "LAD", "LCX", "RCA"):
            d, ci, p = compare({("eng",) + c: eng[c], ("raw",) + c: base[c]}, Yv, ("eng",) + c, ("raw",) + c, t)
            rows.append({"model": c[0], "strategy": c[1], "target": t, "dAUC_eng_minus_raw": d, "ci_lo": ci[0], "ci_hi": ci[1], "p": p})
    ab = pd.DataFrame(rows).round(3); ab.to_csv(f"{RESULTS}/nb02_eng_vs_raw.csv", index=False); print(ab.to_string())
    fams = {"risk_count": ["risk_factor_count", "age_x_male"], "lipid_ratio": ["tg_hdl", "ldl_hdl"],
            "renal_inflam": ["egfr", "nlr"], "ecg": ["ecg_ischemia_count", "ecg_repol_any"],
            "echo": ["ef_low", "rwma_present", "dysfunction_burden"] + [a for a in added if a.startswith("rwma_is_")],
            "angina": ["angina_score"]}
    full = roc_auc_score(Yv.max(1), eng[("catboost", "indep")].mean(0)[:, 3]); rows = []
    for fam, cols in fams.items():
        keep = [i for i, n in enumerate(names2) if n not in cols]
        p = benchmark(X2[:, keep], y8, Yv, [("catboost", "indep")], n_repeats=max(2, REPS // 2), device="cpu")
        rows.append({"dropped_family": fam, "CAD_auc_without": roc_auc_score(Yv.max(1), p[("catboost", "indep")].mean(0)[:, 3]), "CAD_auc_full": full})
    fa = pd.DataFrame(rows).round(3); fa.to_csv(f"{RESULTS}/nb02_family_ablation.csv", index=False); print(fa.to_string())

# %% [NB03] THE headline: adaptive acquisition (VOI) vs baselines + conformal sets
from evaluate import run, summarise
BACKEND = "tabpfn" if TABPFN_OK else "logreg"
ACQ_DEV = DEVICE if BACKEND == "tabpfn" else "cpu"
with stage(f"NB03b acquisition ({BACKEND})"):
    TRAJ = f"{RESULTS}/nb03_traj_{BACKEND}.pkl"          # checkpointed after every fold
    df = run(X2, y8, Yv, gcols2, costs, free, backend=BACKEND, device=ACQ_DEV,
             policies=("voi", "voi_cad", "static", "random"), taus=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
             n_splits=3 if QUICK else 5, n_repeats=1, K=8, save_traj=TRAJ)
    df.to_csv(f"{RESULTS}/nb03_acq_{BACKEND}.csv", index=False)
    g = df.groupby(["policy", "tau"]).agg(cost=("cost", "mean"), auc=("cad_auc", "mean"), sd=("cad_auc", "std"),
                                          LAD_cov=("LAD_cov", "mean"), LCX_cov=("LCX_cov", "mean"), RCA_cov=("RCA_cov", "mean"),
                                          ambiguous=("ambiguous_rate", "mean"), singleton=("singleton_rate", "mean")).round(3).reset_index()
    g.to_csv(f"{RESULTS}/nb03_frontier_table.csv", index=False); print(g.to_string())
    plt.figure(figsize=(6, 4))
    for pol, d in g.groupby("policy"): plt.errorbar(d.cost, d.auc, d.sd, marker="o", capsize=2, label=pol)
    plt.xlabel("mean test cost per patient (relative units)"); plt.ylabel("CAD AUC"); plt.legend(); plt.title(f"Cost-accuracy frontier ({BACKEND})")
    plt.tight_layout(); plt.savefig(f"{RESULTS}/frontier_{BACKEND}.png", dpi=200); plt.close()
    target = 0.98 * g.auc.max()            # headline: cheapest operating point reaching 98% of best AUC
    for pol, d in g.groupby("policy"):
        ok = d[d.auc >= target]
        print(f"{pol:9s} cost to reach AUC>={target:.3f}: {ok.cost.min() if len(ok) else float('nan'):.2f}")

# %% [NB03c] offline re-analysis of saved trajectories (CPU, seconds): policy test, stopping rules, cross-conformal
from analysis import stepwise, frontier, paired_policy_test
with stage("NB03c offline analysis"):
    trajs = pickle.load(open(TRAJ, "rb"))
    sw = stepwise(trajs, Yv); sw.to_csv(f"{RESULTS}/nb03c_stepwise.csv", index=False)
    print(sw.pivot(index="step", columns="policy", values="CAD_auc").to_string())
    print(sw.pivot(index="step", columns="policy", values="cost").round(2).to_string())
    plt.figure(figsize=(6, 4))
    for pol, d in sw.groupby("policy"): plt.plot(d.cost, d.CAD_auc, marker="o", label=pol)
    plt.xlabel("mean cumulative cost"); plt.ylabel("CAD AUC (pooled OOF)"); plt.legend(); plt.title("Ordering quality, stopping-rule free")
    plt.tight_layout(); plt.savefig(f"{RESULTS}/stepwise_{BACKEND}.png", dpi=200); plt.close()
    for a_, b_ in [("voi", "static"), ("voi_cad", "static"), ("voi_cad", "voi"), ("voi", "random")]:
        for st in (2, 3, 4, 5):
            d_, ci_, p_ = paired_policy_test(trajs, Yv, a_, b_, st)
            print(f"{a_:8s} - {b_:7s} step {st}: dAUC={d_:+.4f} CI=({ci_[0]:+.4f},{ci_[1]:+.4f}) p={p_:.3f}")
    fr = frontier(trajs, Yv); fr.to_csv(f"{RESULTS}/nb03c_frontier_conformal.csv", index=False)
    key = ["policy", "rule", "mode", "mondrian", "alpha"]
    show = fr[(fr.tau == 0.9)].groupby(key)[["cost", "CAD_auc", "LAD_cov", "LCX_cov", "RCA_cov", "singleton"]].mean().round(3)
    print(show.to_string())

# %% [NB04] audit: calibration fixes, subgroups (sex/age), decision curve, learning curve
from checks import subgroup_report, learning_curve
from evaluate import net_benefit
with stage("NB04a calibration variants"):
    cal = benchmark(X2, y8, Yv, [("catboost", "indep"), ("catboost_sig", "indep"), ("rf", "indep"), ("rf_sig", "indep")],
                    n_repeats=REPS, device="cpu")
    if TABPFN_OK:
        cal.update(benchmark(X2, y8, Yv, [("tabpfn", "indep")], n_repeats=REPS, device=DEVICE))
    rc = report(cal, Yv); rc.to_csv(f"{RESULTS}/nb04_calibration.csv", index=False)
    print(rc[rc.target == "CAD"][["model", "auc", "brier", "ece", "slope"]].to_string())
with stage("NB04b subgroup + decision curve"):
    BEST = ("tabpfn", "indep") if ("tabpfn", "indep") in cal else ("catboost_sig", "indep")
    print("subgroup/decision-curve model:", BEST)
    P = cal[BEST].mean(0)
    sex, age = X2[:, names2.index("Sex")], X2[:, names2.index("Age")]
    masks = {"all": np.ones(len(X2), bool), "female": sex != 1, "male": sex == 1,       # MALE_CODE=1 verified (Male=176)
             "age<50": age < 50, "age50-64": (age >= 50) & (age < 65), "age>=65": age >= 65}
    sub = subgroup_report(P, Yv, masks); sub.to_csv(f"{RESULTS}/nb04_subgroups.csv", index=False)
    print(sub[sub.target == "CAD"].to_string())
    th, y = np.linspace(0.05, 0.9, 40), Yv.max(1)
    plt.figure(figsize=(6, 4)); plt.plot(th, net_benefit(y, P[:, 3], th), label="model")
    plt.plot(th, net_benefit(y, np.ones(len(y)), th), label="cath all"); plt.axhline(0, c="k", label="cath none")
    plt.ylim(-0.1, y.mean() + 0.05); plt.xlabel("risk threshold"); plt.ylabel("net benefit"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{RESULTS}/decision_curve.png", dpi=200); plt.close()
with stage("NB04c learning curve"):
    lc = learning_curve(X2, y8, Yv, combo=("lr", "indep")); lc.to_csv(f"{RESULTS}/nb04_learning_curve.csv", index=False); print(lc.to_string())

# %% [DONE] zip results for easy download
shutil.make_archive("/kaggle/working/results_bundle" if os.path.isdir("/kaggle/working") else f"{REPO}/results_bundle", "zip", RESULTS)
print("files:", sorted(os.listdir(RESULTS)))
