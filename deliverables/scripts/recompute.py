#!/usr/bin/env python3
"""Step 4: independent recomputation from data/results/*.json. Writes tables/*.csv, data/figure_data/*.csv, audit/recompute.json"""
import os, json, math, itertools
import numpy as np, pandas as pd
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

df = load_runs()
R = {}
# --- consistency JSON vs CSV vs SUMMARY csv
csv = pd.read_csv(os.path.join(DATA, "results.csv"))
m = df.merge(csv, on=["model_name", "train_instances", "seed"], suffixes=("", "_csv"))
R["n_json_runs"] = len(df); R["n_csv_rows"] = len(csv); R["n_merged"] = len(m)
R["json_vs_csv_max_abs_diff_eval_loss"] = float((m.eval_loss - m.eval_loss_csv).abs().max())
R["n_seed_inferred_runs"] = int(df.seed_inferred.sum())
g = agg(df); g.to_csv(os.path.join(TAB, "recomputed_config_stats.csv"), index=False)
ss = pd.read_csv(os.path.join(DATA, "results_summary.csv")).merge(g, on=["model_name", "train_instances"])
R["summary_csv_max_abs_diff_mean"] = float((ss.mean_eval_loss - ss["mean"]).abs().max())
R["summary_csv_max_abs_diff_sd"] = float((ss.std_eval_loss - ss["sd"]).abs().max())
R["n_unique_configs"] = len(g)
R["sd_ddof"] = 1
df.to_csv(os.path.join(TAB, "all_runs_from_json.csv"), index=False)

ms, ds = model_sweep(g), data_sweep(g)
def sweep_fits(s, xcol):
    x, y = s[xcol].values, s["mean"].values
    out = {"x": x.tolist(), "y": y.tolist(), "fit2": fit2(x, y)}
    out["fit3_bounded"] = fit3(x, y, True); out["fit3_unbounded"] = fit3(x, y, False)
    # leave-one-out
    loo = []
    for i in range(len(x)):
        k = [j for j in range(len(x)) if j != i]
        f2 = fit2(x[k], y[k]); f3b = fit3(x[k], y[k], True)
        loo.append(dict(dropped=float(x[i]), alpha2=f2["alpha"], r2_2=f2["r2_log"],
                        alpha3=f3b["alpha"] if f3b else None, c3=f3b["c"] if f3b else None, flags3=f3b["flags"] if f3b else None))
    out["loo"] = loo
    return out
R["model"] = sweep_fits(ms, "N"); R["data"] = sweep_fits(ds, "train_instances")
R["model_nofc"] = {"fit2": fit2(ms["N_nofc"], ms["mean"])}
R["model_fit2_all15_points"] = fit2(df[df.train_instances == 10000].sort_values("model").non_emb_params, df[df.train_instances == 10000].sort_values("model").eval_loss)
# per-seed fits
ps = []
for s in (42, 43, 44):
    a = df[(df.seed == s) & (df.train_instances == 10000)].copy(); a["o"] = a.model_name.map(ORDER.index); a = a.sort_values("o")
    b = df[(df.seed == s) & (df.model_name == ORDER[2])].sort_values("train_instances")
    for nm, d, xc in (("model", a, "non_emb_params"), ("data", b, "train_instances")):
        f2 = fit2(d[xc], d.eval_loss); f3b = fit3(d[xc], d.eval_loss, True)
        ps.append(dict(sweep=nm, seed=s, alpha2=f2["alpha"], r2_2=f2["r2_log"], alpha3=f3b["alpha"], c3=f3b["c"], r2_3=f3b["r2"], flags3=";".join(f3b["flags"])))
ps = pd.DataFrame(ps); ps.to_csv(os.path.join(TAB, "per_seed_fits.csv"), index=False); R["per_seed"] = ps.to_dict("records")
# adjacent pairs
def pairs(s, label):
    rows = []
    s = s.reset_index(drop=True)
    for i in range(len(s) - 1):
        a, b = s.loc[i], s.loc[i + 1]
        la = [df[(df.model_name == a.model_name) & (df.train_instances == a.train_instances) & (df.seed == k)].eval_loss.iloc[0] for k in (42, 43, 44)]
        lb = [df[(df.model_name == b.model_name) & (df.train_instances == b.train_instances) & (df.seed == k)].eval_loss.iloc[0] for k in (42, 43, 44)]
        sd_overlap = (a["mean"] - a.sd) <= (b["mean"] + b.sd) and (b["mean"] - b.sd) <= (a["mean"] + a.sd)
        rng_overlap = min(la) <= max(lb) and min(lb) <= max(la)
        paired = [x - y for x, y in zip(la, lb)]
        rows.append(dict(sweep=label, from_cfg=f"{a.model} D={int(a.train_instances)}", to_cfg=f"{b.model} D={int(b.train_instances)}",
                         mean_from=a["mean"], sd_from=a.sd, mean_to=b["mean"], sd_to=b.sd, diff=a["mean"] - b["mean"],
                         pm_sd_overlap=bool(sd_overlap), minmax_overlap=bool(rng_overlap), seeds_where_larger_is_better=int(sum(d > 0 for d in paired))))
    return rows
adj = pd.DataFrame(pairs(ms, "model") + pairs(ds, "data")); adj.to_csv(os.path.join(TAB, "adjacent_pairs.csv"), index=False); R["adjacent"] = adj.to_dict("records")
# extra descriptors
R["ln_K"] = LN_K; R["K"] = 257
g["ppl_equiv"] = np.exp(g["mean"]); g["relsd"] = g.sd / g["mean"]
R["max_relsd"] = float(g.relsd.max()); R["min_relsd"] = float(g.relsd.min())
R["total_train_time_sec"] = float(df.train_time_sec.sum()); R["total_eval_time_sec"] = float(df.eval_time_sec.sum()); R["total_run_time_sec"] = float(df.total_time_sec.sum())
R["best_mean"] = float(g["mean"].min()); R["worst_mean"] = float(g["mean"].max())
sel = json.load(open(os.path.join(DATA, "results", "selected_lr.json"))); R["lr_test"] = sel
st = json.load(open(os.path.join(DATA, "status.json"))); R["peak_vram_mb"] = st["peak_vram_mb"]; R["last_pipeline_elapsed_sec"] = st["elapsed_time_sec"]
# training-loss summary (running mean over run vs eval) - NOT like-for-like
df.sort_values(["model_name", "train_instances", "seed"]).to_csv(os.path.join(TAB, "train_vs_eval_not_comparable.csv"),
    columns=["model", "train_instances", "seed", "steps", "train_loss", "eval_loss"], index=False)
g.to_csv(os.path.join(TAB, "recomputed_config_stats.csv"), index=False)
# compare with SUMMARY.md values
R["summary_md_values"] = dict(alpha=0.299, r2_model=0.986, beta=0.285, r2_data=0.989, alpha3_model=0.440, beta3_data=0.444)
json.dump(R, open(os.path.join(AUD, "recompute.json"), "w"), indent=2, default=float)
print(json.dumps({k: R[k] for k in R if k not in ("per_seed", "adjacent", "model", "data")}, indent=1, default=float))
for k in ("model", "data"):
    print(k, json.dumps({a: R[k][a] for a in ("fit2", "fit3_bounded", "fit3_unbounded")}, default=float))
    print("LOO", [(round(l["alpha2"], 3), None if l["alpha3"] is None else round(l["alpha3"], 3), l["flags3"]) for l in R[k]["loo"]])
print(ps.round(3).to_string()); print(adj.round(4).to_string())
