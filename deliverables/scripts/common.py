"""Shared helpers: load per-run JSON, fits. All numbers in the package derive from data/results/*.json and data/*.csv."""
import os, json, glob, math
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
DATA = os.path.join(PKG, "data"); TAB = os.path.join(PKG, "tables"); FIG = os.path.join(PKG, "figures")
FD = os.path.join(DATA, "figure_data"); AUD = os.path.join(PKG, "audit")
for d in (TAB, FIG, FD, AUD): os.makedirs(d, exist_ok=True)

ALIAS = {"google/bert_uncased_L-2_H-128_A-2": "BERT-Tiny", "google/bert_uncased_L-4_H-256_A-4": "BERT-Mini",
         "google/bert_uncased_L-4_H-512_A-8": "BERT-Small", "google/bert_uncased_L-8_H-512_A-8": "BERT-Medium",
         "google/bert_uncased_L-12_H-768_A-12": "BERT-Base"}
ORDER = list(ALIAS)
BATCH, EPOCHS = 128, 3
LN_K = math.log(257)

def load_runs():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "results", "*.json"))):
        b = os.path.basename(f)
        if b in ("selected_lr.json", "smoke_test.json"): continue
        j = json.load(open(f))
        j["file"] = b; j["seed_inferred"] = "seed" not in j
        if "seed" not in j: j["seed"] = 42      # early seed-42 runs carry no seed field (UNVERIFIED provenance; see audit)
        rows.append(j)
    df = pd.DataFrame(rows)
    df["model"] = df["model_name"].map(ALIAS)
    df["steps"] = df["train_instances"].apply(lambda d: math.ceil(d / BATCH) * EPOCHS)
    return df

def agg(df):
    g = df.groupby(["model_name", "train_instances"]).agg(
        model=("model", "first"), N=("non_emb_params", "first"), N_nofc=("non_emb_params_no_fc", "first"),
        total_params=("total_params", "first"), steps=("steps", "first"), n_seeds=("seed", "count"),
        mean=("eval_loss", "mean"), sd=("eval_loss", lambda s: s.std(ddof=1)),
        vmin=("eval_loss", "min"), vmax=("eval_loss", "max"),
        train_loss_mean=("train_loss", "mean"), train_time=("train_time_sec", "mean"),
        eval_time=("eval_time_sec", "mean"), total_time=("total_time_sec", "mean")).reset_index()
    return g

def model_sweep(g):
    s = g[g.train_instances == 10000].copy(); s["o"] = s.model_name.map(ORDER.index); return s.sort_values("o")
def data_sweep(g):
    return g[g.model_name == "google/bert_uncased_L-4_H-512_A-8"].sort_values("train_instances")

def fit2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    lx, ly = np.log(x), np.log(y)
    sl, ic = np.polyfit(lx, ly, 1)
    pr = sl * lx + ic
    r2log = 1 - ((ly - pr) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    yp = np.exp(ic) * x ** sl
    r2lin = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(a=float(np.exp(ic)), alpha=float(-sl), r2_log=float(r2log), r2_lin=float(r2lin))

def f3(x, a, al, c): return a * np.power(x, -al) + c

def fit3(x, y, bounded=True, starts=None):
    """3-param L = a X^-alpha + c. Returns best fit over several starts + convergence diagnostics."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    f2 = fit2(x, y)
    lo, hi = [1e-6, 1e-4, 0.0], [1e4, 5.0, float(y.min())]
    if not bounded: lo, hi = [-np.inf, 1e-6, -np.inf], [np.inf, 20.0, np.inf]
    starts = starts or [[f2["a"] * k, max(f2["alpha"] * m, 0.02), y.min() * cc]
                        for k in (0.1, 0.8, 3.0) for m in (0.5, 1.0, 2.0) for cc in (0.0, 0.5, 0.9)]
    res = []
    for p0 in starts:
        p0 = list(np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)) if bounded else p0
        try:
            p, _ = curve_fit(f3, x, y, p0=p0, bounds=(lo, hi), maxfev=50000)
            sse = float(((y - f3(x, *p)) ** 2).sum()); res.append((sse, p))
        except Exception:
            pass
    if not res: return None
    res.sort(key=lambda t: t[0]); sse, p = res[0]
    n_good = sum(1 for s, _ in res if s < sse * 1.0001 + 1e-12)
    alphas = [q[1] for s, q in res if s < sse * 1.01 + 1e-12]
    r2 = 1 - sse / ((y - y.mean()) ** 2).sum()
    tol = 1e-3
    flags = []
    if bounded:
        if p[1] <= lo[1] + tol or p[1] >= hi[1] - tol: flags.append("alpha at bound")
        if p[2] <= lo[2] + tol or p[2] >= hi[2] - tol: flags.append("c at bound")
        if p[0] >= hi[0] * (1 - tol) or p[0] <= lo[0] + tol: flags.append("a at bound")
    return dict(a=float(p[0]), alpha=float(p[1]), c=float(p[2]), r2=float(r2), sse=sse, n_starts=len(res),
                n_starts_same_optimum=n_good, alpha_range_near_opt=[float(min(alphas)), float(max(alphas))], flags=flags)
