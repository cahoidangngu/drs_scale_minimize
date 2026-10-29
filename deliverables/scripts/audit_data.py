#!/usr/bin/env python3
"""Data/seed audit (CPU only, read-only on the raw MS MARCO files and on the experiment code).

Writes deliverables/audit/data_checks.json. Re-run with:
    python3 -I scripts/audit_data.py            (from deliverables/)
Needs the raw MS MARCO files (not shipped in the package); path via MSMARCO_DIR.
"""
import os, sys, json, math, random, hashlib, io, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
EXP = os.path.dirname(PKG)                                  # drscale_exp/
MSM = os.getenv("MSMARCO_DIR", os.path.join(os.path.dirname(EXP), "data", "msmarco"))
sys.path.insert(0, EXP)
os.environ.setdefault("USE_TF", "0")

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import utils as U                                       # experiment copy of utils.py (loaders)

def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)

out = {}

# ---------------- corpus ids (ids only, text is not needed) ----------------
pids = []
with open(os.path.join(MSM, "collection.tsv"), "r", encoding="utf8") as f:
    for line in f:
        pids.append(int(line.split("\t", 1)[0]))
out["corpus_n_passages"] = len(pids)
out["corpus_unique_pids"] = len(set(pids))
out["corpus_min_pid"], out["corpus_max_pid"] = min(pids), max(pids)
out["corpus_pids_contiguous_from_0"] = (out["corpus_unique_pids"] == out["corpus_max_pid"] + 1 and out["corpus_min_pid"] == 0)
FULL = 8841823            # full MS MARCO passage collection size, as given in the task statement (paper Sec 3.3 says "8.8M")
out["full_collection_size_assumed"] = FULL
out["corpus_fraction"] = len(pids) / FULL
out["corpus_missing_fraction"] = 1 - len(pids) / FULL
corpus = dict.fromkeys(pids)                                # same key order / set semantics as the experiment's dict

# ---------------- queries / qrels ----------------
train_q = quiet(U.load_msmarco_queries, "train", os.path.join(MSM, "queries.train.tsv"))
dev_q = quiet(U.load_msmarco_queries, "dev.small", os.path.join(MSM, "queries.dev.small.tsv"))
train_qrels = quiet(U.load_msmarco_qrels, "train", os.path.join(MSM, "qrels.train.tsv"))
dev_qrels = quiet(U.load_msmarco_qrels, "dev.small", os.path.join(MSM, "qrels.dev.small.tsv"))
out["n_train_queries_file"] = len(train_q)
out["n_train_queries_with_qrels"] = len(train_qrels)
out["n_dev_queries"] = len(dev_q)
out["n_train_pairs_total"] = sum(len(v) for v in train_qrels.values())
out["n_train_pairs_in_corpus"] = sum(1 for v in train_qrels.values() for p in v if p in corpus)
out["n_train_pairs_filtered"] = out["n_train_pairs_total"] - out["n_train_pairs_in_corpus"]
out["train_pairs_filtered_fraction"] = out["n_train_pairs_filtered"] / out["n_train_pairs_total"]
out["n_dev_pairs_total"] = sum(len(v) for v in dev_qrels.values())
out["n_dev_pairs_in_corpus"] = sum(1 for v in dev_qrels.values() for p in v if p in corpus)
lost_q = [q for q, v in train_qrels.items() if not any(p in corpus for p in v)]
out["n_train_queries_losing_all_positives"] = len(lost_q)
out["n_train_queries_usable"] = len(train_qrels) - len(lost_q)
filt_pairs = [(q, p) for q, v in train_qrels.items() for p in v if p not in corpus]
out["filtered_pairs_min_pid"] = min(p for _, p in filt_pairs)
out["filtered_pairs_all_beyond_max_pid"] = all(p > out["corpus_max_pid"] for _, p in filt_pairs)
qlen = lambda q: len(train_q[q].split())
lost_set = set(lost_q)
out["train_query_len_words_mean_lost"] = float(np.mean([qlen(q) for q in lost_q]))
out["train_query_len_words_mean_kept"] = float(np.mean([qlen(q) for q in train_qrels if q not in lost_set]))
out["train_qid_mean_lost"] = float(np.mean(lost_q))
out["train_qid_mean_kept"] = float(np.mean([q for q in train_qrels if q not in lost_set]))

# ---------------- evaluation query set (replicates evaluate.contrastive_perplexity) ----------------
valid = sorted(q for q in dev_q if any(p in corpus for p in dev_qrels.get(q, {})))
out["n_dev_valid_queries"] = len(valid)
ev = valid[:1000]
ev_set = set(ev)
out["n_eval_queries"] = len(ev)
out["eval_qid_min"], out["eval_qid_max"] = ev[0], ev[-1]
out["dev_qid_min"], out["dev_qid_max"] = valid[0], valid[-1]
inst = [(q, p) for q in ev for p in dev_qrels[q] if p in corpus]
out["n_eval_instances"] = len(inst)
pos_ids = {p for _, p in inst}
out["n_eval_positive_passages"] = len(pos_ids)
# bias check: first-1000-by-qid vs the full dev.small set
allq = valid
def stats(qs):
    return {"mean_query_len_words": float(np.mean([len(dev_q[q].split()) for q in qs])),
            "mean_positives_per_query": float(np.mean([len(dev_qrels[q]) for q in qs])),
            "frac_queries_with_2plus_positives": float(np.mean([len(dev_qrels[q]) > 1 for q in qs]))}
out["eval_subset_stats"] = stats(ev)
out["dev_all_stats"] = stats(allq)
rest = allq[1000:]
out["dev_rest_stats"] = stats(rest)

# ---------------- leakage: qid / text overlap train vs eval ----------------
out["qid_overlap_trainfile_vs_dev_all"] = len(set(train_q) & set(dev_q))
out["qid_overlap_trainqrels_vs_eval1000"] = len(set(train_qrels) & ev_set)
out["qid_overlap_trainqrels_vs_dev_all"] = len(set(train_qrels) & set(dev_q))
norm = lambda s: " ".join(s.lower().split())
train_txt = {norm(train_q[q]) for q in train_qrels}
out["exact_query_text_overlap_trainqrels_vs_eval1000"] = sum(norm(dev_q[q]) in train_txt for q in ev)
train_pos = {p for v in train_qrels.values() for p in v}
out["eval_positive_passages_also_train_positive"] = len(pos_ids & train_pos)   # passage (not query) overlap: informational

# ---------------- training subset: nesting and seed independence ----------------
def build_instances(seed):
    random.seed(seed)                                       # transformers.set_seed also calls random.seed
    qid2pos = quiet(U.load_msmarco_pos, list(train_q.keys()), corpus, train_qrels)   # shuffles positives per query
    inst_ = [(q, p) for q in qid2pos for p in qid2pos[q]]   # DRDataset.__init__
    random.shuffle(inst_)
    return inst_

SIZES = [2500, 5000, 10000, 20000, 40000]
subs = {}
for s in (42, 43, 44):
    ins = build_instances(s)
    subs[s] = {d: ins[:d] for d in SIZES}
    out[f"n_instances_after_filter_seed{s}"] = len(ins)
out["nested_across_D_within_seed"] = all(
    set(subs[s][SIZES[i]]) <= set(subs[s][SIZES[i + 1]]) for s in subs for i in range(len(SIZES) - 1))
ov = {}
for a, b in ((42, 43), (42, 44), (43, 44)):
    ov[f"{a}-{b}"] = len(set(subs[a][10000]) & set(subs[b][10000]))
out["overlap_pairs_D10000_between_seeds"] = ov
n_inst = out["n_instances_after_filter_seed42"]
out["expected_overlap_if_independent_D10000"] = 10000 * 10000 / n_inst
out["identical_D10000_subset_between_seeds"] = any(subs[a][10000] == subs[b][10000] for a, b in ((42, 43), (42, 44), (43, 44)))
out["train_qids_in_D40000_that_are_dev_qids"] = len({q for q, _ in subs[42][40000]} & set(dev_q))

# ---------------- evaluation negative pool: shared across models? ----------------
def build_pool(seed=1, pool=100000, n_neg=256):
    all_ids = set(corpus.keys())
    neg_ids = list(all_ids - pos_ids)
    random.seed(seed)                                       # evaluate_model_checkpoint -> set_seed(1) right before the call
    sampled = random.sample(neg_ids, pool)
    first_draw = random.sample(range(pool), n_neg)
    return neg_ids, sampled, first_draw
n1, s1, d1 = build_pool()
n2, s2, d2 = build_pool()
out["eval_neg_candidates_total"] = len(n1)
out["eval_pool_size"] = len(s1)
out["eval_pool_identical_on_rebuild"] = (s1 == s2 and d1 == d2)
out["eval_pool_sha1"] = hashlib.sha1(np.array(s1, dtype=np.int64).tobytes()).hexdigest()
out["eval_first_draw_sha1"] = hashlib.sha1(np.array(d1, dtype=np.int64).tobytes()).hexdigest()
out["eval_pool_overlap_with_eval_positives"] = len(set(s1) & pos_ids)
out["eval_pool_overlap_with_train_positives"] = len(set(s1) & train_pos)
out["eval_K_candidates"] = 1 + 256
out["ln_K"] = math.log(1 + 256)
out["n_random_draws_per_model"] = len(inst) * 5

json.dump(out, open(os.path.join(PKG, "audit", "data_checks.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
