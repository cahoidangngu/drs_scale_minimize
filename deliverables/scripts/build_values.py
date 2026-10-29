"""Compute every number used in the reports/README/matrix as formatted strings (dict V) from CSV/JSON/audit files."""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *

df = load_runs(); g = agg(df); ms, ds = model_sweep(g), data_sweep(g)
R = json.load(open(os.path.join(AUD, "recompute.json"))); DC = json.load(open(os.path.join(AUD, "data_checks.json")))
MC = json.load(open(os.path.join(AUD, "model_checks.json"))); F3 = json.load(open(os.path.join(AUD, "f3.json")))
CFG = pd.read_csv(os.path.join(AUD, "actual_training_config.csv"))
PS = pd.DataFrame(R["per_seed"]); ADJ = pd.DataFrame(R["adjacent"])

# values quoted from the paper (arXiv 2403.18684v2 / SIGIR'24); verified against sources/paper_text.txt by check_numbers.py
PAPER = dict(A_model="3.22", alpha_model="0.53", delta_model="0.04", r2_model="0.991", B_data="3.49", beta_data="1.05", delta_data="0.05", r2_data="0.954",
             A_t2="9.89", alpha_t2="0.53", delta_t2="0.14", r2_t2="0.999", B_t2="6.04", beta_t2="0.50", delta_t2d="0.15", r2_t2d="0.991",
             joint_A="3.6", joint_B="7.1", joint_alpha="0.56", joint_beta="1.31", joint_delta="0.03", steps="10,000", negs="256", pool="100,000",
             size_lo="0.5", size_hi="82", size_hi_intro="87", dollars="20,000", params_b="13")

V = {}
f = lambda x, n=4: f"{x:.{n}f}"
def put(k, v): V[k] = str(v)

# ---- environment / run facts
put("gpu", "NVIDIA RTX 5880 Ada Generation (49,140 MiB; driver 580.178.04, nvidia-smi on 2026-10-29)")
put("cpu", "Intel Core Ultra 9 285K (24 logical CPUs), 93 GiB RAM")
put("torch", MC["torch"]); put("transformers", MC["transformers"])
import scipy, matplotlib
put("numpy", np.__version__); put("scipy", scipy.__version__); put("pandas", pd.__version__); put("mpl", matplotlib.__version__)
put("python", "3.11.7"); put("repo_commit", "81453c3b67486641ad282d58f819392a08561e77")
put("peak_vram", f"{R['peak_vram_mb']:.1f}")
put("n_runs", R["n_json_runs"]); put("n_cfg", R["n_unique_configs"])
put("tot_train_s", f"{R['total_train_time_sec']:.0f}"); put("tot_eval_s", f"{R['total_eval_time_sec']:.0f}"); put("tot_run_s", f"{R['total_run_time_sec']:.0f}")
put("tot_run_min", f"{R['total_run_time_sec']/60:.1f}"); put("last_pipeline_s", f"{R['last_pipeline_elapsed_sec']:.1f}"); put("last_pipeline_min", f"{R['last_pipeline_elapsed_sec']/60:.1f}")
c0 = CFG.iloc[0]
put("cfg_lr", "1e-4"); put("cfg_epochs", int(c0.epochs)); put("cfg_batch", int(c0.batch)); put("cfg_warm", int(c0.warmup_steps)); put("cfg_wd", c0.weight_decay)
put("cfg_clip", c0.max_grad_norm); put("cfg_beta2", c0.adam_beta2); put("cfg_all_same", "yes" if CFG.drop(columns=["run", "seed", "max_steps"]).nunique().max() == 1 else "no")
put("cfg_bf16_all", "yes" if CFG.bf16.all() else "no"); put("n_cfg_runs", len(CFG))
# ---- data checks
for k, v in DC.items():
    if isinstance(v, (int, float)) and not isinstance(v, bool): V["dc_" + k] = f"{v:,}" if isinstance(v, int) else str(v)
put("corpus_n", f"{DC['corpus_n_passages']:,}"); put("corpus_pct", f"{100*DC['corpus_fraction']:.2f}"); put("corpus_miss_pct", f"{100*DC['corpus_missing_fraction']:.2f}")
put("pairs_total", f"{DC['n_train_pairs_total']:,}"); put("pairs_kept", f"{DC['n_train_pairs_in_corpus']:,}"); put("pairs_filt", f"{DC['n_train_pairs_filtered']:,}")
put("pairs_filt_pct", f"{100*DC['train_pairs_filtered_fraction']:.2f}"); put("q_lost", f"{DC['n_train_queries_losing_all_positives']:,}"); put("q_usable", f"{DC['n_train_queries_usable']:,}")
put("q_with_qrels", f"{DC['n_train_queries_with_qrels']:,}"); put("q_file", f"{DC['n_train_queries_file']:,}")
put("dev_q", f"{DC['n_dev_queries']:,}"); put("dev_pairs", f"{DC['n_dev_pairs_total']:,}"); put("eval_q", f"{DC['n_eval_queries']:,}"); put("eval_inst", f"{DC['n_eval_instances']:,}")
put("pool", f"{DC['eval_pool_size']:,}"); put("neg_cand", f"{DC['eval_neg_candidates_total']:,}"); put("draws", f"{DC['n_random_draws_per_model']:,}")
put("K", 257); put("lnK", f"{LN_K:.3f}"); put("lnK4", f"{LN_K:.4f}")
put("qid_ov_all", DC["qid_overlap_trainfile_vs_dev_all"]); put("qid_ov_eval", DC["qid_overlap_trainqrels_vs_eval1000"]); put("txt_ov", DC["exact_query_text_overlap_trainqrels_vs_eval1000"])
put("pos_pass_ov", DC["eval_positive_passages_also_train_positive"])
ov = DC["overlap_pairs_D10000_between_seeds"]; put("ov_lo", min(ov.values())); put("ov_hi", max(ov.values())); put("ov_exp", f"{DC['expected_overlap_if_independent_D10000']:.0f}")
put("pool_sha", DC["eval_pool_sha1"][:12]); put("qlen_lost", f"{DC['train_query_len_words_mean_lost']:.2f}"); put("qlen_kept", f"{DC['train_query_len_words_mean_kept']:.2f}")
es, dr = DC["eval_subset_stats"], DC["dev_rest_stats"]
put("ev_len", f"{es['mean_query_len_words']:.2f}"); put("rest_len", f"{dr['mean_query_len_words']:.2f}"); put("ev_pos", f"{es['mean_positives_per_query']:.3f}"); put("rest_pos", f"{dr['mean_positives_per_query']:.3f}")
put("ev_qid_max", f"{DC['eval_qid_max']:,}"); put("dev_qid_max", f"{DC['dev_qid_max']:,}")
put("dup_lines", "10"); put("pairs_raw_lines", "532,761")
# ---- params
mm = MC["models"]
for m in ORDER:
    a = ALIAS[m].split("-")[1].lower(); d = mm[m]
    for k, kk in (("with_fc", "non_emb_with_fc"), ("nofc", "non_emb_no_fc"), ("fc", "output_fc"), ("pool", "bert_pooler"), ("layers", "transformer_layers_only"), ("tot", "total")):
        put(f"p_{a}_{k}", f"{d[kk]:,}")
    put(f"p_{a}_mi", f"{d['with_fc_in_2pow20_units']:.1f}")
# ---- sweep tables
def lossrow(r): return f"{r.mean:.4f} ± {r.sd:.4f}"
for r in g.itertuples():
    key = ("m_" + ALIAS[r.model_name].split("-")[1].lower()) if r.train_instances == 10000 else ""
for r in ms.itertuples(): V[f"mean_{ALIAS[r.model_name].split('-')[1].lower()}"] = f"{r.mean:.4f}"; V[f"sd_{ALIAS[r.model_name].split('-')[1].lower()}"] = f"{r.sd:.4f}"
for r in ds.itertuples(): V[f"mean_d{int(r.train_instances)}"] = f"{r.mean:.4f}"; V[f"sd_d{int(r.train_instances)}"] = f"{r.sd:.4f}"
put("best_mean", f"{R['best_mean']:.4f}"); put("worst_mean", f"{R['worst_mean']:.4f}")
put("relsd_lo", f"{100*R['min_relsd']:.1f}"); put("relsd_hi", f"{100*R['max_relsd']:.1f}")
put("ratio_best_chance", f"{R['best_mean']/LN_K:.3f}"); put("ratio_worst_chance", f"{R['worst_mean']/LN_K:.3f}")
put("ppl_best", f"{math.exp(R['best_mean']):.3f}"); put("ppl_worst", f"{math.exp(R['worst_mean']):.3f}")
put("tiny_over_base", f"{ms['mean'].iloc[0]/ms['mean'].iloc[-1]:.1f}"); put("d2500_over_d40000", f"{ds['mean'].iloc[0]/ds['mean'].iloc[-1]:.2f}")
# ---- fits
for tag, key in (("m", "model"), ("d", "data")):
    f2, f3b, f3u = R[key]["fit2"], R[key]["fit3_bounded"], R[key]["fit3_unbounded"]
    put(f"{tag}_a2", f"{f2['a']:.2f}"); put(f"{tag}_al2", f"{f2['alpha']:.3f}"); put(f"{tag}_al2_4", f"{f2['alpha']:.4f}"); put(f"{tag}_r2log", f"{f2['r2_log']:.3f}"); put(f"{tag}_r2log4", f"{f2['r2_log']:.4f}"); put(f"{tag}_r2lin", f"{f2['r2_lin']:.4f}")
    put(f"{tag}_a3", f"{f3b['a']:.2f}"); put(f"{tag}_al3", f"{f3b['alpha']:.3f}"); put(f"{tag}_c3", f"{f3b['c']:.3f}"); put(f"{tag}_r23", f"{f3b['r2']:.4f}")
    put(f"{tag}_nstart", f3b["n_starts"]); put(f"{tag}_nsame", f3b["n_starts_same_optimum"]); put(f"{tag}_unb_al", f"{f3u['alpha']:.3f}"); put(f"{tag}_unb_c", f"{f3u['c']:.3f}")
    put(f"{tag}_flags", "none" if not f3b["flags"] else ", ".join(f3b["flags"]))
    A2 = [l["alpha2"] for l in R[key]["loo"]]; A3 = [l["alpha3"] for l in R[key]["loo"]]
    put(f"{tag}_loo2_lo", f"{min(A2):.3f}"); put(f"{tag}_loo2_hi", f"{max(A2):.3f}"); put(f"{tag}_loo3_lo", f"{min(A3):.3f}"); put(f"{tag}_loo3_hi", f"{max(A3):.3f}")
    put(f"{tag}_loo3_nbound", sum(1 for l in R[key]["loo"] if l["flags3"]))
    P = PS[PS.sweep == key]
    put(f"{tag}_ps2_lo", f"{P.alpha2.min():.3f}"); put(f"{tag}_ps2_hi", f"{P.alpha2.max():.3f}"); put(f"{tag}_ps3_lo", f"{P.alpha3.min():.3f}"); put(f"{tag}_ps3_hi", f"{P.alpha3.max():.3f}")
    put(f"{tag}_ps_nbound", int((P.flags3.fillna("") != "").sum()))
    # equivalent paper-style constant A where L=(A/X)^alpha+c  ->  A = a^(1/alpha)
    put(f"{tag}_Aeq", f"{f3b['a'] ** (1 / f3b['alpha']):.3g}")
put("m_a2_nofc_al", f"{R['model_nofc']['fit2']['alpha']:.3f}"); put("m_a2_all15_al", f"{R['model_fit2_all15_points']['alpha']:.3f}"); put("m_a2_all15_r2", f"{R['model_fit2_all15_points']['r2_log']:.3f}")
put("sm_al2", "0.299"); put("sm_r2", "0.986"); put("sm_be2", "0.285"); put("sm_r2d", "0.989"); put("sm_al3", "0.440"); put("sm_be3", "0.444")
put("fit_match", "all six SUMMARY values reproduce to the printed precision" if (V["m_al2"] == "0.299" and V["m_r2log"] == "0.986" and V["d_al2"] == "0.285" and V["d_r2log"] == "0.989" and V["m_al3"] == "0.440" and V["d_al3"] == "0.444") else "MISMATCH (see recompute.json)")
put("json_csv_diff", f"{R['json_vs_csv_max_abs_diff_eval_loss']:.1e}"); put("sumcsv_diff", f"{R['summary_csv_max_abs_diff_mean']:.1e}")
# ---- steps
for d in (2500, 5000, 10000, 20000, 40000): put(f"steps_{d}", math.ceil(d / BATCH) * EPOCHS)
put("f3_loss", f"{F3['F3_loss']:.4f}"); put("f3_p", f"{F3['F3_p_pos']:.4f}")
# ---- adjacency: sentences
def adj(i): r = ADJ.iloc[i]; return r
put("adj_sm_overlap", "overlap" if ADJ.iloc[2].pm_sd_overlap else "do not overlap"); put("adj_d1020_overlap", "overlap" if ADJ.iloc[6].pm_sd_overlap else "do not overlap")
put("n_adj_overlap", int(ADJ.pm_sd_overlap.sum())); put("n_adj", len(ADJ))
put("adj_sm_diff", f"{ADJ.iloc[2]['diff']:.4f}"); put("adj_d1020_diff", f"{ADJ.iloc[6]['diff']:.4f}")
put("lr_a", f"{R['lr_test']['results']['5e-05']:.4f}"); put("lr_b", f"{R['lr_test']['results']['0.0001']:.4f}")
# train-loss vs eval (for transparency)
tr = df[(df.train_instances == 10000)].groupby("model").train_loss.mean()
put("tl_tiny", f"{tr['BERT-Tiny']:.4f}"); put("tl_small", f"{tr['BERT-Small']:.4f}"); put("tl_base", f"{tr['BERT-Base']:.4f}")
tl = pd.read_csv(os.path.join(FD, "F8_train_loss_curves.csv"))
last = tl[(tl.train_instances == 10000)].sort_values("step").groupby(["model", "seed"]).tail(1).groupby("model").loss.mean()
put("tll_small", f"{last['BERT-Small']:.4f}"); put("tll_base", f"{last['BERT-Base']:.4f}"); put("tll_tiny", f"{last['BERT-Tiny']:.4f}")
put("first_step", "10")

# ---- markdown tables (generated, never typed)
def md(df_, cols=None):
    cols = cols or list(df_.columns); out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df_.iterrows(): out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)
rows = []
for nm, s in (("model sweep", ms), ("data sweep", ds)):
    for r in s.itertuples():
        vals = [df[(df.model_name == r.model_name) & (df.train_instances == r.train_instances) & (df.seed == k)].eval_loss.iloc[0] for k in (42, 43, 44)]
        rows.append({"Sweep": nm, "Model": r.model, "N (non-emb., incl. fc)": f"{int(r.N):,}", "D (pairs)": f"{int(r.train_instances):,}", "Steps": int(r.steps), "Seeds": int(r.n_seeds),
                     "Mean ± SD (nats)": lossrow(r), "Seed 42 / 43 / 44": " / ".join(f"{v:.4f}" for v in vals), "exp(mean) (supplementary)": f"{math.exp(r.mean):.3f}", "Train time (s)": f"{r.train_time:.1f}", "Eval time (s)": f"{r.eval_time:.1f}"})
RES = pd.DataFrame(rows); RES.to_csv(os.path.join(TAB, "T1_results_mean_sd.csv"), index=False); open(os.path.join(TAB, "T1_results_mean_sd.md"), "w").write(md(RES) + "\n")
V["T_results"] = md(RES)
fits = []
for lab, key, x in (("Model size N", "model", "N"), ("Data size D", "data", "D")):
    f2, f3b = R[key]["fit2"], R[key]["fit3_bounded"]
    fits.append({"Sweep": lab, "Fit": "2-param  L = a·X^-α (OLS in log–log)", "a": f"{f2['a']:.2f}", "exponent": f"{f2['alpha']:.3f}", "c": "—", "R² (log)": f"{f2['r2_log']:.4f}", "R² (linear)": f"{f2['r2_lin']:.4f}", "notes": "main fit"})
    fits.append({"Sweep": lab, "Fit": "3-param  L = a·X^-α + c (curve_fit)", "a": f"{f3b['a']:.2f}", "exponent": f"{f3b['alpha']:.3f}", "c": f"{f3b['c']:.3f}", "R² (log)": "—", "R² (linear)": f"{f3b['r2']:.4f}", "notes": f"bound flags: {V[('m' if key=='model' else 'd')+'_flags']}; 5 points, 3 parameters"})
FITS = pd.DataFrame(fits); FITS.to_csv(os.path.join(TAB, "T2_fits.csv"), index=False); open(os.path.join(TAB, "T2_fits.md"), "w").write(md(FITS) + "\n"); V["T_fits"] = md(FITS)
loo = []
for lab, key in (("Model size N", "model"), ("Data size D", "data")):
    xs = R[key]["x"]
    for l in R[key]["loo"]:
        x = l["dropped"]; xl = f"{int(x):,}"
        loo.append({"Sweep": lab, "Point left out": xl, "2-param exponent": f"{l['alpha2']:.3f}", "R² (log)": f"{l['r2_2']:.3f}", "3-param exponent": f"{l['alpha3']:.3f}", "3-param c": f"{l['c3']:.3f}", "bound flag": ", ".join(l["flags3"]) if l["flags3"] else "—"})
LOO = pd.DataFrame(loo); LOO.to_csv(os.path.join(TAB, "T3_leave_one_out.csv"), index=False); open(os.path.join(TAB, "T3_leave_one_out.md"), "w").write(md(LOO) + "\n"); V["T_loo"] = md(LOO)
psd = PS.copy()
for c_ in ("alpha2", "r2_2", "alpha3", "c3", "r2_3"): psd[c_] = psd[c_].map(lambda v: f"{v:.3f}")
psd["flags3"] = psd["flags3"].replace("", "—"); psd.columns = ["Sweep", "Seed", "2-param exponent", "R² (log)", "3-param exponent", "3-param c", "R² (3-param)", "bound flag"]
psd.to_csv(os.path.join(TAB, "T4_per_seed_fits.csv"), index=False); open(os.path.join(TAB, "T4_per_seed_fits.md"), "w").write(md(psd) + "\n"); V["T_perseed"] = md(psd)
a2 = ADJ.copy(); a2["mean_from"] = a2.apply(lambda r: f"{r.mean_from:.4f} ± {r.sd_from:.4f}", axis=1); a2["mean_to"] = a2.apply(lambda r: f"{r.mean_to:.4f} ± {r.sd_to:.4f}", axis=1)
a2 = a2[["sweep", "from_cfg", "to_cfg", "mean_from", "mean_to", "diff", "pm_sd_overlap", "minmax_overlap", "seeds_where_larger_is_better"]]
a2["diff"] = a2["diff"].map(lambda v: f"{v:.4f}"); a2["pm_sd_overlap"] = a2.pm_sd_overlap.map({True: "yes", False: "no"}); a2["minmax_overlap"] = a2.minmax_overlap.map({True: "yes", False: "no"})
a2.columns = ["Sweep", "From", "To", "Mean ± SD (from)", "Mean ± SD (to)", "Difference", "±SD intervals overlap", "Seed min–max ranges overlap", "Seeds (of 3) where larger setting is better"]
a2.to_csv(os.path.join(TAB, "T5_adjacent_pairs.csv"), index=False); open(os.path.join(TAB, "T5_adjacent_pairs.md"), "w").write(md(a2) + "\n"); V["T_adj"] = md(a2)
pr = []
for m in ORDER:
    d = mm[m]
    pr.append({"Model": ALIAS[m], "Layers L": d["layers"], "Hidden H": d["hidden"], "Transformer layers only": f"{d['transformer_layers_only']:,}", "+ BERT pooler (unused)": f"{d['bert_pooler']:,}", "= encoder non-emb. (\"no fc\")": f"{d['non_emb_no_fc']:,}",
               "+ output_fc": f"{d['output_fc']:,}", "= non-emb. incl. fc (used on x-axis)": f"{d['non_emb_with_fc']:,}", "same in units of 2^20": f"{d['with_fc_in_2pow20_units']:.1f}", "Total params": f"{d['total']:,}"})
PR = pd.DataFrame(pr); PR.to_csv(os.path.join(TAB, "T6_parameter_counts.csv"), index=False); open(os.path.join(TAB, "T6_parameter_counts.md"), "w").write(md(PR) + "\n"); V["T_params"] = md(PR)
# paper-style comparison
cmp_ = pd.DataFrame([
    {"Quantity": "Model-size exponent α (3-param family; Table 1, Sec. 4.1)", "P": PAPER["alpha_model"], "O": V["m_al3"], "Comparable?": "No: see caveats"},
    {"Quantity": "Model-size constant δ_N / c (Table 1)", "P": PAPER["delta_model"], "O": V["m_c3"], "Comparable?": "No: see caveats"},
    {"Quantity": "Model-size fit R² (Table 1)", "P": PAPER["r2_model"], "O": V["m_r23"], "Comparable?": "Descriptive only (5 points, 3 parameters)"},
    {"Quantity": "Data-size exponent β (3-param family; Table 2, Sec. 4.2)", "P": PAPER["beta_data"], "O": V["d_al3"], "Comparable?": "No: see caveats"},
    {"Quantity": "Data-size constant δ_D / c (Table 2)", "P": PAPER["delta_data"], "O": V["d_c3"], "Comparable?": "No: see caveats"},
    {"Quantity": "Data-size fit R² (Table 2)", "P": PAPER["r2_data"], "O": V["d_r23"], "Comparable?": "Descriptive only"},
    {"Quantity": "2-param exponents (no constant)", "P": "not reported", "O": f"{V['m_al2']} (model), {V['d_al2']} (data)", "Comparable?": "No: the paper has no such fit"}])
cmp_.columns = ["Quantity", "Paper (MS MARCO; Tables 1–2)", "Ours", "Comparable?"]
cmp_.to_csv(os.path.join(TAB, "T7_paper_vs_ours_exponents.csv"), index=False); open(os.path.join(TAB, "T7_paper_vs_ours_exponents.md"), "w").write(md(cmp_) + "\n"); V["T_cmp"] = md(cmp_)
setup = pd.DataFrame([
    ("Task / data", "MS MARCO passage (8.8M passages, 0.5M train queries) and T2Ranking (Sec. 3.3)", f"MS MARCO only; passage file truncated to {V['corpus_n']} passages ({V['corpus_pct']}%)"),
    ("Models", "24 Google BERT checkpoints 0.5M–82M non-emb. params; ERNIE for T2Ranking (Sec. 3.2)", "5 BERT checkpoints: Tiny, Mini, Small, Medium, Base (same family as the paper's red stars, Fig. 3)"),
    ("N definition", "non-embedding parameters (Sec. 3.2, footnote 2)", f"non-embedding incl. projection layer and unused BERT pooler (= the paper's Fig. 6 legend sizes when divided by 2^20)"),
    ("D definition", "number of annotated query–passage pairs (Sec. 3.1, 4.2)", "number of (qid, pid) training pairs"),
    ("Architecture", "shared encoder + projection to 768; CLS or mean pooling (Sec. 3.2, Eq. 2)", "shared BERT + mean pooling + Linear(H→768); no normalisation"),
    ("Training loss", "softmax cross-entropy, random + in-batch negatives (Sec. 3.4, Eq. 3)", "same: 1 positive + 256 random negatives shared by the batch; mean CE; no temperature"),
    ("Optimisation length", f"fixed {PAPER['steps']} steps (Sec. 3.4)", f"3 epochs → {V['steps_2500']}–{V['steps_40000']} steps (depends on D); {V['steps_10000']} steps for the model sweep"),
    ("lr / batch / optimiser", "not stated in the paper (UNVERIFIED)", f"lr 1e-4, batch {V['cfg_batch']}, AdamW, linear decay, warmup {V['cfg_warm']} steps, bf16"),
    ("Metric", "contrastive entropy = −log softmax of the positive among 1+256 candidates (Sec. 3.5, Eq. 5)", "same quantity (mean NLL, nats), K = 257"),
    ("Evaluation set", "dev queries (~7,000, Sec. 3.3); 100,000-passage sub-corpus used for the metric study (Sec. 3.5)", f"first {V['eval_q']} dev.small queries by qid ({V['eval_inst']} pairs); {V['pool']}-passage pool; 5 repeats; fixed eval seed 1"),
    ("Checkpoint reported", "best test loss during training, no early stopping (Sec. 4.1)", "final checkpoint after epoch 3"),
    ("Model sweep", "full training set, many models (Sec. 4.1, Fig. 3)", f"D = 10,000 pairs, 5 models"),
    ("Data sweep", "BERT-Base, up to all ~500K annotations (Sec. 4.2, Fig. 4)", "BERT-Small, D = 2,500 … 40,000"),
    ("Fit form", "L = (A/N)^α + δ (Eq. 6); L = (B/D)^β + δ (Eq. 7); joint Eq. 8", "3-param a·X^-α + c (same family; a = A^α) and a 2-param a·X^-α (extra, main)"),
    ("Seeds", "not stated (UNVERIFIED)", "3 training seeds 42/43/44; evaluation seed fixed"),
    ("Annotation quality, budget allocation", "Sec. 4.3 (ICT, docT5query, ChatGLM3), Sec. 5", "not reproduced"),
], columns=["Aspect", "Paper (section / figure)", "Our experiment"])
setup.to_csv(os.path.join(TAB, "T8_paper_vs_ours_setup.csv"), index=False); open(os.path.join(TAB, "T8_paper_vs_ours_setup.md"), "w").write(md(setup) + "\n"); V["T_setup"] = md(setup)
