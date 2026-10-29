#!/usr/bin/env python3
"""Draw F1-F11 (PNG 300 dpi + SVG) and write per-figure source CSVs into data/figure_data/."""
import os, sys, json, glob, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter
from common import *

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
                     "legend.fontsize": 10.5, "xtick.labelsize": 11, "ytick.labelsize": 11, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00", purple="#CC79A7", sky="#56B4E9", grey="#555555", black="#111111")
YL = "Contrastive entropy (NLL in nats, K = 257; lower is better)"
df = load_runs(); g = agg(df); ms, ds = model_sweep(g), data_sweep(g)
R = json.load(open(os.path.join(AUD, "recompute.json"))); DC = json.load(open(os.path.join(AUD, "data_checks.json")))

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIG, name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def Mfmt(n): return f"{n/1e6:.2f}M"

fits_offsets = {'N': [(12, 12), (12, 12), (14, 16), (18, -4), (6, 34)], 'train_instances': [(12, 12)] * 5}

def scaling_plot(s, xcol, fits, name, xlabel, title, labels, fd_name, ticks):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    x, y, sd = s[xcol].values.astype(float), s["mean"].values, s["sd"].values
    sub = df[df.model_name.isin(s.model_name)].copy()
    if xcol == "N": sub = sub[sub.train_instances == 10000]; sub["x"] = sub.non_emb_params.astype(float)
    else: sub = sub[sub.model_name == ORDER[2]]; sub["x"] = sub.train_instances.astype(float)
    for k, (sd_, d) in enumerate(sub.groupby("seed")):
        ax.scatter(d.x * (1 + 0.035 * (k - 1)), d.eval_loss, s=34, color=C["grey"], alpha=.35, zorder=2,
                   label="Individual seeds (42, 43, 44)" if k == 0 else None, marker=["o", "s", "^"][k])
    ax.errorbar(x, y, yerr=sd, fmt="o", ms=9, color=C["blue"], ecolor=C["blue"], elinewidth=1.8, capsize=5, zorder=4,
                label="Mean ± SD over 3 seeds (SD with ddof = 1)")
    xx = np.geomspace(x.min() * .8, x.max() * 1.25, 200)
    f2, f3b = fits["fit2"], fits["fit3_bounded"]
    ax.plot(xx, f2["a"] * xx ** (-f2["alpha"]), color=C["verm"], lw=2.4, zorder=3,
            label=f"2-param (main): L = {f2['a']:.2f}·X^-{f2['alpha']:.3f}   R²(log) = {f2['r2_log']:.3f}")
    ax.plot(xx, f3(xx, f3b["a"], f3b["alpha"], f3b["c"]), color=C["green"], lw=2.2, ls="--", zorder=3,
            label=f"3-param (secondary): L = {f3b['a']:.1f}·X^-{f3b['alpha']:.3f} + {f3b['c']:.3f}   R² = {f3b['r2']:.3f}")
    offs = fits_offsets.get(xcol)
    for xi, yi, lab, v, off in zip(x, y, labels, y, offs):
        ax.annotate(f"{lab}\nmean {v:.4f}", (xi, yi), xytext=off, textcoords="offset points", fontsize=9.6, color=C["black"],
                    bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.8), zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    yt = [0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2]
    ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(FixedLocator(ticks[0])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ticks[1](v))); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(xx.min(), xx.max() * 1.9); ax.set_ylim(0.1, 1.15)
    ax.set_xlabel(xlabel); ax.set_ylabel(YL); ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=True, fontsize=10)
    ax.text(0.01, 0.015, "Log-log axes. Scaled-down experiment, not a replication of the paper.", transform=ax.transAxes, fontsize=9.5, color=C["grey"])
    save(fig, name)
    pd.DataFrame(dict(x=x, label=labels, mean=y, sd=sd, **{f"seed{sd_}": d.sort_values("x").eval_loss.values for sd_, d in sub.groupby("seed")},
                      fit2=f2["a"] * x ** (-f2["alpha"]), fit3=f3(x, f3b["a"], f3b["alpha"], f3b["c"]))).to_csv(os.path.join(FD, fd_name), index=False)

# ---------------- F4 / F5
scaling_plot(ms, "N", R["model"], "F4_model_scaling", "Non-embedding parameters N (1M = 10^6; includes projection layer\nand the unused BERT pooler)",
             "F4. Model-size scaling (D = 10,000 pairs, 237 steps)", [f"{r.model}\n{Mfmt(r.N)}" for r in ms.itertuples()], "F4_model_scaling.csv",
             ([5e5, 1e6, 3e6, 1e7, 3e7, 1e8], lambda v: f"{v/1e6:g}M"))
scaling_plot(ds, "train_instances", R["data"], "F5_data_scaling", "Training pairs D (BERT-Small; steps grow with D: 3 epochs)",
             "F5. Data-size scaling (BERT-Small, 3 epochs)", [f"D={int(r.train_instances):,}\n{int(r.steps)} steps" for r in ds.itertuples()], "F5_data_scaling.csv",
             ([2500, 5000, 10000, 20000, 40000], lambda v: f"{v:,.0f}"))

# ---------------- F6 residuals
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.9))
rows = []
for ax, s, key, xcol, ttl, xl in ((axs[0], ms, "model", "N", "Model-size sweep", "N (non-embedding params)"), (axs[1], ds, "data", "train_instances", "Data-size sweep", "D (training pairs)")):
    x = s[xcol].values.astype(float); y = s["mean"].values; sd = s["sd"].values
    f2, f3b = R[key]["fit2"], R[key]["fit3_bounded"]
    r2_ = y - f2["a"] * x ** (-f2["alpha"]); r3_ = y - f3(x, f3b["a"], f3b["alpha"], f3b["c"])
    ax.axhline(0, color=C["black"], lw=1)
    ax.errorbar(x * 0.93, r2_, yerr=sd, fmt="o", ms=8, color=C["verm"], capsize=4, label="2-param fit (log–log OLS)")
    ax.errorbar(x * 1.07, r3_, yerr=sd, fmt="s", ms=8, color=C["green"], capsize=4, label="3-param fit")
    ax.set_xscale("log"); ax.set_title(ttl, loc="left", fontweight="bold"); ax.set_xlabel(xl); ax.set_ylabel("Residual = observed − fitted (nats)")
    ax.xaxis.set_major_locator(FixedLocator(list(x))); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e5 else f"{v:,.0f}")); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelrotation=30)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.4 * (hi_ - lo_)); ax.legend(loc="upper center", ncol=2, fontsize=10)
    for a_, b_, c_ in zip(x, r2_, r3_): rows.append(dict(sweep=key, x=a_, resid_2param=b_, resid_3param=c_))
fig.suptitle("F6. Residuals of the two fits (error bars = SD of the observed mean, not of the residual)", x=0.01, ha="left", fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .94)); save(fig, "F6_residuals"); pd.DataFrame(rows).to_csv(os.path.join(FD, "F6_residuals.csv"), index=False)

# ---------------- F7 table image
fig, ax = plt.subplots(figsize=(12.5, 4.6)); ax.axis("off")
hdr = ["Config", "Model", "Non-emb. params N", "Pairs D", "Steps", "Seeds", "Mean ± SD (nats)", "Seed values (42, 43, 44)", "Train time (s)"]
body = []
for nm, s in (("Model sweep", ms), ("Data sweep", ds)):
    for r in s.itertuples():
        vals = [df[(df.model_name == r.model_name) & (df.train_instances == r.train_instances) & (df.seed == k)].eval_loss.iloc[0] for k in (42, 43, 44)]
        body.append([nm, r.model, f"{int(r.N):,}", f"{int(r.train_instances):,}", f"{int(r.steps)}", f"{int(r.n_seeds)}", f"{r.mean:.4f} ± {r.sd:.4f}", ", ".join(f"{v:.4f}" for v in vals), f"{r.train_time:.1f}"])
tb = ax.table(cellText=body, colLabels=hdr, loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(10.2); tb.scale(1, 1.75)
tb.auto_set_column_width(list(range(len(hdr))))
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor("#bbbbbb")
    if i == 0: c.set_facecolor("#dfe9f3"); c.set_text_props(fontweight="bold")
ax.set_title("F7. Results table (BERT-Small, D = 10,000 appears in both sweeps; it is one set of 3 runs)", loc="left", fontweight="bold", fontsize=12.5)
save(fig, "F7_results_table"); pd.DataFrame(body, columns=hdr).to_csv(os.path.join(FD, "F7_results_table.csv"), index=False)

# ---------------- F8 training curves
recs = []
for f in sorted(glob.glob(os.path.join(PKG, "..", "checkpoints", "*", "checkpoint-*", "trainer_state.json"))):
    key = f.split(os.sep)[-3]
    if key.startswith(("smoke", "lr_test")): continue
    for h in json.load(open(f))["log_history"]:
        if "loss" in h: recs.append(dict(key=key, step=h["step"], loss=h["loss"]))
tl = pd.DataFrame(recs)
rk = df[["model_key", "model", "train_instances", "seed"]].rename(columns={"model_key": "key"}); tl = tl.merge(rk, on="key")
tl.to_csv(os.path.join(FD, "F8_train_loss_curves.csv"), index=False)
fig, ax = plt.subplots(figsize=(8.8, 5.8))
cols = dict(zip(ORDER, [C["verm"], C["orange"], C["green"], C["blue"], C["purple"]]))
for mn in ORDER:
    t = tl[(tl.train_instances == 10000) & (tl.model == ALIAS[mn])]
    for _, d in t.groupby("seed"): ax.plot(d.step, d.loss, color=cols[mn], alpha=.25, lw=1)
    m_ = t.groupby("step").loss.mean(); ax.plot(m_.index, m_.values, color=cols[mn], lw=2.2, marker="o", ms=4, label=f"{ALIAS[mn]} (train, mean of 3 seeds)")
    ev = g[(g.model_name == mn) & (g.train_instances == 10000)]["mean"].iloc[0]
    ax.scatter([237], [ev], marker="D", s=70, color=cols[mn], edgecolor="black", zorder=5)
ax.scatter([], [], marker="D", s=70, color="white", edgecolor="black", label="Dev evaluation of final model (mean of 3 seeds)")
ax.set_yscale("log"); ax.set_xlabel("Optimizer step (batch size 128; D = 10,000 → 237 steps)"); ax.set_ylabel("Loss (nats)")
ax.set_title("F8. Training loss curves vs. final dev loss (NOT like-for-like)", loc="left", fontweight="bold")
ax.legend(fontsize=9, loc="upper right"); ax.set_xlim(0, 262)
fig.text(0.01, -0.05, "Train curve: mean loss over each 10-step window, dropout on, bf16, on seen pairs, shared random negatives (K = 257).\nDiamonds: eval mode, fp32, held-out dev queries, 100k pool. Same K and formula but different mode, data and averaging → no overfitting claim.", fontsize=9, color=C["grey"])
save(fig, "F8_training_curves")

# ---------------- F9 LOO
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.9)); rows = []
for ax, key, ttl, xl in ((axs[0], "model", "Model-size sweep", [f"{ALIAS[m]}" .replace("BERT-", "") for m in ORDER]), (axs[1], "data", "Data-size sweep", [f"{int(d):,}" for d in ds.train_instances])):
    L = R[key]["loo"]; pos = np.arange(len(L) + 1)
    a2 = [R[key]["fit2"]["alpha"]] + [l["alpha2"] for l in L]; a3 = [R[key]["fit3_bounded"]["alpha"]] + [l["alpha3"] for l in L]
    ax.plot(pos, a2, "o-", color=C["verm"], ms=8, label="2-param exponent (log–log OLS)"); ax.plot(pos, a3, "s--", color=C["green"], ms=8, label="3-param exponent")
    for l, p in zip(L, pos[1:]):
        if l["flags3"]: ax.annotate("c at bound", (p, l["alpha3"]), xytext=(4, -16), textcoords="offset points", fontsize=9)
    ax.set_xticks(pos); ax.set_xticklabels(["none"] + xl); ax.set_xlabel("Point left out (leave-one-out)"); ax.set_ylabel("Fitted exponent"); ax.set_title(ttl, loc="left", fontweight="bold"); ax.legend(fontsize=9.5, loc="best"); ax.set_ylim(0, 0.9)
    for i, (u, v) in enumerate(zip(a2, a3)): rows.append(dict(sweep=key, dropped=(["none"] + xl)[i], exponent_2param=u, exponent_3param=v))
fig.suptitle("F9. Stability of the fitted exponent when one of the 5 points is left out", x=0.01, ha="left", fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .94)); save(fig, "F9_loo_stability"); pd.DataFrame(rows).to_csv(os.path.join(FD, "F9_loo_stability.csv"), index=False)

# ---------------- F10 chance level
fig, ax = plt.subplots(figsize=(9, 5.4))
lab = [f"{r.model}\nD={int(r.train_instances):,}" for r in pd.concat([ms, ds[ds.train_instances != 10000]]).itertuples()]
vals = list(ms["mean"]) + list(ds[ds.train_instances != 10000]["mean"])
ax.bar(range(len(vals)), vals, color=[C["blue"]] * 5 + [C["sky"]] * 4, width=.65)
ax.axhline(LN_K, color=C["verm"], ls="--", lw=2, label=f"Random-scoring loss ln(K) = ln(257) = {LN_K:.3f}")
for i, v in enumerate(vals): ax.text(i, v * 1.12, f"{v:.3f}", ha="center", fontsize=9.5)
ax.set_yscale("log"); ax.set_ylim(0.08, 12); ax.set_xticks(range(len(vals))); ax.set_xticklabels(lab, rotation=40, ha="right", fontsize=9.5)
ax.set_ylabel("Mean eval loss (nats, log scale)"); ax.set_title("F10. Every trained model vs. the loss of random scoring", loc="left", fontweight="bold"); ax.legend(loc="center right")
save(fig, "F10_chance_level"); pd.DataFrame(dict(config=[l.replace("\n", " ") for l in lab], mean_loss=vals, chance_ln_K=LN_K, ratio_to_chance=[v / LN_K for v in vals])).to_csv(os.path.join(FD, "F10_chance_level.csv"), index=False)

# ---------------- F11 training time
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.9))
for ax, s, xcol, ttl, xl in ((axs[0], ms, "N", "Model sweep (D = 10,000, 237 steps)", "Non-embedding parameters N"), (axs[1], ds, "train_instances", "Data sweep (BERT-Small, 3 epochs)", "Training pairs D")):
    sub = df[df.model_name.isin(s.model_name)]
    sub = sub[sub.train_instances == 10000] if xcol == "N" else sub[sub.model_name == ORDER[2]]
    sub = sub.assign(x=sub.non_emb_params if xcol == "N" else sub.train_instances)
    t = sub.groupby("x").train_time_sec.agg(["mean", lambda v: v.std(ddof=1)]).reset_index(); t.columns = ["x", "mean", "sd"]
    ax.errorbar(t.x, t["mean"], yerr=t.sd, fmt="o-", color=C["blue"], capsize=4, ms=8)
    for a_, b_ in zip(t.x, t["mean"]): ax.annotate(f"{b_:.0f}s", (a_, b_), xytext=(6, 8), textcoords="offset points", fontsize=10)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel("Training wall-clock time (s)"); ax.set_title(ttl, loc="left", fontweight="bold")
    ax.xaxis.set_major_locator(FixedLocator(list(t.x))); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e5 else f"{v:,.0f}")); ax.xaxis.set_minor_formatter(NullFormatter()); ax.tick_params(axis="x", labelrotation=30)
    t.to_csv(os.path.join(FD, "F11_train_time_" + ("model" if xcol == "N" else "data") + ".csv"), index=False)
fig.suptitle("F11. Training time (mean ± SD over 3 seeds; one RTX 5880 Ada GPU, bf16)", x=0.01, ha="left", fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .94)); save(fig, "F11_training_time")

# ---------------- diagrams
def box(ax, xy, w, h, text, fc="#eef3f8", ec=C["blue"], fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.6))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)
def arrow(ax, a, b, **k):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.7, color=C["black"], **k))

# F1 workflow
fig, ax = plt.subplots(figsize=(12, 6.3)); ax.set_xlim(0, 12); ax.set_ylim(0, 6.3); ax.axis("off"); ax.grid(False)
ax.text(0.1, 6.05, "F1. Experiment workflow (scaled-down)", fontsize=14, fontweight="bold")
box(ax, (0.2, 4.0), 3.4, 1.6, f"MS MARCO passages\n{DC['corpus_n_passages']:,} passages (file truncated)\ntrain: {DC['n_instances_after_filter_seed42']:,} unique pairs\ndev.small: {DC['n_dev_queries']:,} queries", fc="#fdf3e1", ec=C["orange"], fs=10)
box(ax, (4.2, 4.0), 3.2, 1.6, "Pick D pairs\nseeded shuffle, take first D\nD in {2,500 ... 40,000}\n(nested within a seed)", fc="#fdf3e1", ec=C["orange"])
box(ax, (8.0, 4.0), 3.8, 1.6, "Train DREncoder (BERT + projection)\n3 epochs, lr 1e-4, batch 128, bf16\n256 random negatives / step\n3 seeds: 42, 43, 44", fc="#eef3f8", fs=10)
box(ax, (8.0, 1.9), 3.8, 1.5, f"Evaluate final model\nfirst {DC['n_eval_queries']:,} dev queries by qid\n({DC['n_eval_instances']:,} query-passage pairs)\n{DC['eval_pool_size']:,}-passage pool, 256 negatives,\n5 repeats, fixed seed 1", fc="#e6f4ee", ec=C["green"], fs=10)
box(ax, (4.2, 1.9), 3.2, 1.5, "results/*.json -> results.csv\nmean ± SD over 3 seeds\n(9 distinct configurations)", fc="#e6f4ee", ec=C["green"], fs=10)
box(ax, (0.2, 1.9), 3.4, 1.5, "Fit power laws\n2-param: OLS on log-log (main)\n3-param: curve_fit, bound check\nleave-one-out, per-seed fits", fc="#f8e9e2", ec=C["verm"], fs=10)
box(ax, (0.2, 0.2), 3.4, 1.0, "Figures F4-F11, tables,\nreports, audit", fc="#f1e8f0", ec=C["purple"])
arrow(ax, (3.6, 4.8), (4.2, 4.8)); arrow(ax, (7.4, 4.8), (8.0, 4.8)); arrow(ax, (9.9, 4.0), (9.9, 3.4))
ax.text(10.0, 3.62, "final checkpoint (no early stopping)", fontsize=9, color=C["grey"])
arrow(ax, (8.0, 2.65), (7.4, 2.65)); arrow(ax, (4.2, 2.65), (3.6, 2.65)); arrow(ax, (1.9, 1.9), (1.9, 1.2))
save(fig, "F1_workflow")

# F2 architecture
fig, ax = plt.subplots(figsize=(11.5, 5.6)); ax.set_xlim(0, 11.5); ax.set_ylim(0, 5.6); ax.axis("off"); ax.grid(False)
ax.text(0.1, 5.3, "F2. DREncoder: one shared encoder for queries and passages", fontsize=14, fontweight="bold")
for yb, nm, tx in ((3.4, "query q", "\"what is the capital\nof California\"\n(<= 25 tokens train, 32 eval)"), (0.9, "passage p", "\"The capital of California\nis Sacramento ...\"\n(<= 180 tokens)")):
    box(ax, (0.1, yb), 2.9, 1.4, f"{nm}\n{tx}", fc="#fdf3e1", ec=C["orange"], fs=9.5)
    arrow(ax, (3.0, yb + .7), (3.5, 2.75 if yb > 2 else 2.55))
box(ax, (3.5, 1.7), 2.5, 1.9, "Shared BERT encoder\n(same weights for q and p)\ntiny ... base: 2-12 layers,\nhidden 128-768", fc="#eef3f8", fs=9.5)
arrow(ax, (6.0, 2.65), (6.5, 2.65))
box(ax, (6.5, 1.9), 1.6, 1.5, "Mean pooling\nover real tokens\n(padding masked)", fc="#e6f4ee", ec=C["green"], fs=9.5)
arrow(ax, (8.1, 2.65), (8.6, 2.65))
box(ax, (8.6, 1.9), 1.4, 1.5, "Linear layer\nH -> 768\n(output_fc)", fc="#f8e9e2", ec=C["verm"], fs=9.5)
ax.text(10.1, 2.65, "768-d\nvector\n(no L2\nnorm.)", fontsize=9.5, va="center")
arrow(ax, (9.3, 1.9), (9.3, 1.0))
box(ax, (4.6, 0.3), 6.7, 0.7, "score s(q,p) = <f(q), f(p)>  (dot product; no temperature)", fc="#f1e8f0", ec=C["purple"], fs=10, bold=True)
save(fig, "F2_architecture")

# F3 contrastive loss illustration
pos_s, shown = 9.0, [("negative 1", 3.1), ("negative 2", 4.0), ("negative 3", 2.4)]
rest_n, rest_s = 256 - len(shown), 1.0
sc = np.array([pos_s] + [s for _, s in shown] + [rest_s] * rest_n); p = np.exp(sc - sc.max()); p /= p.sum(); loss = -math.log(p[0])
fig, ax = plt.subplots(figsize=(10.5, 5.8)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 5.8); ax.axis("off"); ax.grid(False)
ax.text(0.1, 5.5, f"F3. Measuring contrastive entropy: 1 correct passage vs {256} random wrong ones (K = {257})", fontsize=13.5, fontweight="bold")
ax.text(0.1, 5.05, "ILLUSTRATIVE numbers (invented scores), computed by the same formula as Eq. 5 of the paper.", fontsize=10, color=C["verm"])
rows3 = [("positive (labelled relevant)", pos_s, p[0], C["green"])] + [(n, s, p[i + 1], C["grey"]) for i, (n, s) in enumerate(shown)] + [(f"… {rest_n} more negatives", rest_s, p[4], C["grey"])]
ax.text(0.2, 4.55, "candidate", fontweight="bold"); ax.text(4.0, 4.55, "score ⟨q,p⟩", fontweight="bold"); ax.text(5.7, 4.55, "softmax prob.", fontweight="bold")
for i, (n, s, pr, c) in enumerate(rows3):
    y = 3.95 - i * .6
    ax.add_patch(plt.Rectangle((0.1, y - .2), 3.7, .45, fc="white", ec=c, lw=1.6)); ax.text(0.2, y, n, va="center", fontsize=10.5)
    ax.text(4.2, y, f"{s:.1f}" + (" each" if "more" in n else ""), va="center", fontsize=10.5)
    ax.add_patch(plt.Rectangle((5.7, y - .15), max(pr, 0.002) * 3.0, .32, fc=c, ec="none"))
    ax.text(5.8 + max(pr, 0.002) * 3.0, y, f"{pr:.4f}" if "more" not in n else f"{pr:.5f} each ({pr*rest_n:.4f} total)", va="center", fontsize=10)
ax.text(0.1, 0.85, f"loss for this pair = −log( p_positive ) = −log({p[0]:.4f}) = {loss:.4f} nats", fontsize=12.5, fontweight="bold")
ax.text(0.1, .42, f"Averaged over all test pairs → the reported number. Random scoring would give ln({257}) = {LN_K:.3f}; a perfect model → 0.", fontsize=11)
ax.text(0.1, .05, "Lower = the model gives the labelled passage a larger share of the probability. Unit: nats (natural log); it is NOT a perplexity (not exponentiated).", fontsize=10, color=C["grey"])
save(fig, "F3_contrastive_loss")
pd.DataFrame(dict(candidate=[r[0] for r in rows3], score=[r[1] for r in rows3], softmax_prob=[r[2] for r in rows3], note="illustrative toy scores")).to_csv(os.path.join(FD, "F3_contrastive_loss.csv"), index=False)
json.dump(dict(F3_loss=loss, F3_p_pos=float(p[0])), open(os.path.join(AUD, "f3.json"), "w"))
pd.DataFrame([dict(step=i, text=t) for i, t in enumerate(["MS MARCO load", "Pick D pairs", "Train (3 seeds)", "Evaluate", "Aggregate", "Fit", "Figures/reports"])]).to_csv(os.path.join(FD, "F1_workflow.csv"), index=False)
pd.DataFrame([dict(stage=s) for s in ["tokenize", "shared BERT", "mean pooling", "Linear H->768", "dot product"]]).to_csv(os.path.join(FD, "F2_architecture.csv"), index=False)
print("done", sorted(os.listdir(FIG)))
