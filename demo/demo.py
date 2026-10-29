#!/usr/bin/env python3
"""DRScale demo: use OUR trained checkpoints to retrieve passages for a query.

Sub-commands (run from drscale_exp/demo):
  python3 demo.py prepare            build the small passage pool and encode it with every checkpoint (one-off, ~3 min on GPU)
  python3 demo.py cases              rank the labelled passages for 1,000 dev queries with every model, save good/bad cases
  python3 demo.py ask "your query" [--model base] [--k 5]     retrieve the top passages for any text
  python3 demo.py compare "your query" | --qid 1234           same query through all models side by side
  python3 demo.py scenario NAME      a ready-made demo scenario (see `python3 demo.py scenario list`)

The pool is only ~21k passages (not the 8.3M corpus), so ranks here are far easier than in real search.
Demo metrics (rank, MRR@10) are for illustration; the paper-style number is the contrastive entropy printed by `cases`.
"""
import os, sys, io, json, math, random, argparse, contextlib, textwrap
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
MSM = os.getenv("MSMARCO_DIR", os.path.join(os.path.dirname(EXP), "data", "msmarco"))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
import numpy as np, torch

MODELS = {  # short name -> (checkpoint folder, HF base model, description)
    "tiny":   ("tiny_10000i_3ep", "google/bert_uncased_L-2_H-128_A-2", "BERT-Tiny, D=10,000"),
    "mini":   ("mini_10000i_3ep", "google/bert_uncased_L-4_H-256_A-4", "BERT-Mini, D=10,000"),
    "small":  ("small_10000i_3ep", "google/bert_uncased_L-4_H-512_A-8", "BERT-Small, D=10,000"),
    "medium": ("medium_10000i_3ep", "google/bert_uncased_L-8_H-512_A-8", "BERT-Medium, D=10,000"),
    "base":   ("base_10000i_3ep_seed42", "google/bert_uncased_L-12_H-768_A-12", "BERT-Base, D=10,000"),
    "small2500":  ("small_datasize_2500i_3ep", "google/bert_uncased_L-4_H-512_A-8", "BERT-Small, D=2,500"),
    "small40000": ("small_datasize_40000i_3ep_seed42", "google/bert_uncased_L-4_H-512_A-8", "BERT-Small, D=40,000"),
}
SIZE_ORDER = ["tiny", "mini", "small", "medium", "base"]
POOL_NEG = 20000
EXTRA = [  # hand-written passages so that free-text queries have something sensible to find
    "The capital of California is Sacramento. It is located in the northern part of the state and has been the capital since 1854.",
    "Los Angeles is the largest city in California, known for Hollywood, beaches and a warm Mediterranean climate.",
    "The Golden Gate Bridge is a suspension bridge in San Francisco, California, opened in 1937.",
    "Paris is the capital and most populous city of France. The Eiffel Tower is one of its best known landmarks.",
    "To make pho, simmer beef bones with charred onion, ginger, star anise and cinnamon for several hours, then serve the broth over rice noodles.",
    "A football match lasts 90 minutes and is played by two teams of eleven players. The team that scores more goals wins.",
    "Photosynthesis is the process by which green plants use sunlight, water and carbon dioxide to produce glucose and oxygen.",
    "To reset a forgotten password, open the login page, choose 'Forgot password' and follow the link sent to your email address.",
    "The human heart has four chambers and pumps blood through the body; a normal resting heart rate is 60 to 100 beats per minute.",
    "Python is a popular programming language known for readable syntax and a large collection of libraries.",
]

def log(*a): print(*a, flush=True)

# ------------------------------------------------------------------ data
def load_eval_set():
    """First 1,000 dev.small queries by qid, exactly as in evaluate.py."""
    q = {}
    for l in open(os.path.join(MSM, "queries.dev.small.tsv"), encoding="utf8"):
        a, b = l.rstrip("\r\n").split("\t"); q[int(a)] = b
    qr = {}
    for l in open(os.path.join(MSM, "qrels.dev.small.tsv"), encoding="utf8"):
        p = l.split(); qr.setdefault(int(p[0]), []).append(int(p[2]))
    qids = sorted(q)[:1000]
    return qids, q, qr

def build_pool():
    pf = os.path.join(CACHE, "pool.json")
    if os.path.exists(pf): return json.load(open(pf))
    qids, q, qr = load_eval_set()
    pos = sorted({p for i in qids for p in qr[i]})
    pids = []
    for l in open(os.path.join(MSM, "collection.tsv"), encoding="utf8"): pids.append(int(l.split("\t", 1)[0]))
    neg = list(set(pids) - set(pos)); random.seed(1); sampled = random.sample(neg, 100000)[:POOL_NEG]   # same recipe as the evaluation pool
    need = set(pos) | set(sampled); texts = {}
    for l in open(os.path.join(MSM, "collection.tsv"), encoding="utf8"):
        a, _, b = l.rstrip("\r\n").partition("\t"); a = int(a)
        if a in need: texts[a] = b
    pool = {"pids": pos + sampled, "texts": [texts[p] for p in pos + sampled], "n_pos": len(pos)}
    json.dump(pool, open(pf, "w")); return pool

# ------------------------------------------------------------------ models
_loaded = {}
def load_model(key):
    if key in _loaded: return _loaded[key]
    from transformers import AutoConfig, AutoTokenizer
    from modeling import DREncoder
    ck, base, _ = MODELS[key]
    with contextlib.redirect_stdout(io.StringIO()):
        tok = AutoTokenizer.from_pretrained(base)
        m = DREncoder(config=AutoConfig.from_pretrained(base), init_model=base, vector_dim=768, tokenizer=tok)
    m.load_state_dict(torch.load(os.path.join(EXP, "checkpoints", ck, "pytorch_model.bin"), map_location="cpu"), strict=False)
    m = m.to("cuda" if torch.cuda.is_available() else "cpu").eval(); _loaded[key] = m; return m

@torch.no_grad()
def encode(model, texts, max_len, bs=256):
    dev = next(model.parameters()).device; out = []
    for i in range(0, len(texts), bs):
        f = model.tokenizer(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        out.append(model.encode(f["input_ids"].to(dev), f["attention_mask"].to(dev)).float().cpu())
    return torch.cat(out)

def prepare(args):
    pool = build_pool(); qids, q, qr = load_eval_set(); log(f"pool: {len(pool['pids']):,} passages ({pool['n_pos']} labelled positives + {POOL_NEG:,} random)")
    for k in (args.models or MODELS):
        f = os.path.join(CACHE, f"emb_{k}.pt")
        if os.path.exists(f) and not args.force: log(f"[skip] {k}"); continue
        m = load_model(k); P = encode(m, pool["texts"], 180); Q = encode(m, [q[i] for i in qids], 32)
        torch.save({"P": P.half(), "Q": Q.half(), "qids": qids}, f); log(f"[done] {k}: {MODELS[k][2]}"); del _loaded[k]; torch.cuda.empty_cache()

def get_emb(k):
    f = os.path.join(CACHE, f"emb_{k}.pt")
    if not os.path.exists(f): sys.exit(f"Missing {f}. Run: python3 demo.py prepare")
    d = torch.load(f); return d["P"].float(), d["Q"].float(), d["qids"]

# ------------------------------------------------------------------ analysis
def cases(args):
    pool = build_pool(); qids, q, qr = load_eval_set(); pid_idx = {p: i for i, p in enumerate(pool["pids"])}
    res = {}; rng = np.random.RandomState(1); n_pos = pool["n_pos"]
    neg_idx = np.arange(n_pos, len(pool["pids"]))
    draws = {i: rng.choice(neg_idx, 256, replace=False) for i in qids}   # same 256 random negatives for every model
    summary = []
    for k in MODELS:
        P, Q, _ = get_emb(k); ranks, nll = {}, []
        for j, i in enumerate(qids):
            s = P @ Q[j]; pos = [pid_idx[p] for p in qr[i]]
            best = max(pos, key=lambda x: s[x].item()); ranks[i] = int((s > s[best]).sum().item()) + 1
            for p in pos:
                cand = torch.cat([s[p:p + 1], s[torch.as_tensor(draws[i])]]); nll.append(float(-torch.log_softmax(cand, 0)[0]))
        r = np.array(list(ranks.values())); res[k] = ranks
        summary.append(dict(model=k, desc=MODELS[k][2], nll=float(np.mean(nll)), top1=float((r == 1).mean()), recall10=float((r <= 10).mean()), mrr10=float(np.mean([1 / x if x <= 10 else 0 for x in r])), median_rank=float(np.median(r))))
    log(f"{'model':11} {'description':24} {'NLL(nats)':>9} {'top-1':>7} {'R@10':>7} {'MRR@10':>7} {'median rank':>11}")
    for s in summary: log(f"{s['model']:11} {s['desc']:24} {s['nll']:9.4f} {s['top1']:7.1%} {s['recall10']:7.1%} {s['mrr10']:7.3f} {s['median_rank']:11.0f}")
    log(f"(pool = {len(pool['pids']):,} passages; compare the NLL column with results.csv seed-42 values: tiny 0.7424, mini 0.3759, small 0.2281, medium 0.1977, base 0.1785)")
    # pick demonstration cases
    def R(k, i): return res[k][i]
    all_good = [i for i in qids if all(R(k, i) == 1 for k in SIZE_ORDER)]
    grow = [i for i in qids if R("tiny", i) >= 20 and R("base", i) == 1]
    allbad = [i for i in qids if all(R(k, i) >= 10 for k in SIZE_ORDER)]
    mono = [i for i in qids if R("tiny", i) > R("small", i) > R("base", i) >= 1]
    pick = lambda L, n: sorted(L, key=lambda i: (abs(len(q[i].split()) - 6), i))[:n]   # prefer ordinary 5-7 word queries
    out = dict(summary=summary, ranks={k: {str(i): v for i, v in res[k].items()} for k in res}, all_good=pick(all_good, 8), grow_with_size=pick(grow, 8), all_bad=pick(allbad, 8), staircase=pick(mono, 8),
               counts=dict(all_good=len(all_good), grow_with_size=len(grow), all_bad=len(allbad), staircase=len(mono)))
    json.dump(out, open(os.path.join(CACHE, "cases.json"), "w"), indent=1)
    log("\ncase groups (qid list = ordinary-length queries first):", json.dumps({k: out[k] for k in ("counts", "all_good", "grow_with_size", "all_bad", "staircase")}))

# ------------------------------------------------------------------ retrieval
def snippet(t, n=118): t = " ".join(t.split()); return t if len(t) <= n else t[:n - 1] + "…"

def retrieve(key, text, k=5, qid=None):
    pool = build_pool(); m = load_model(key); P, _, _ = get_emb(key)
    extra = encode(m, EXTRA, 180); texts = pool["texts"] + EXTRA; allP = torch.cat([P, extra])
    qv = encode(m, [text], 32)[0]; s = allP @ qv; top = torch.topk(s, k)
    _, q, qr = load_eval_set(); gold = set(qr.get(qid, [])) if qid else set(); pid_of = lambda i: pool["pids"][i] if i < len(pool["pids"]) else None
    rows = [(r + 1, s[i].item(), texts[i], pid_of(int(i)) in gold if gold else False, "extra" if int(i) >= len(pool["pids"]) else "") for r, i in enumerate(top.indices.tolist())]
    grank = None
    if gold:
        gi = [pool["pids"].index(p) for p in gold]; best = max(gi, key=lambda x: s[x].item()); grank = int((s > s[best]).sum().item()) + 1
    return rows, grank

def show(key, text, k=5, qid=None):
    rows, grank = retrieve(key, text, k, qid)
    log(f"\n[{key}] {MODELS[key][2]}   query: “{text}”" + (f"   (labelled passage rank: {grank})" if qid else ""))
    for r, sc, t, g, tag in rows: log(f"  {r}. score {sc:7.2f} {'✔ LABELLED' if g else '          '} {('['+tag+'] ') if tag else ''}{snippet(t)}")

def dev_query(qid):
    _, q, qr = load_eval_set(); return q[qid]

def ask(args):
    qid = args.qid; text = args.query if args.query else dev_query(qid); show(args.model, text, args.k, qid)

def compare(args):
    qid = args.qid; text = args.query if args.query else dev_query(qid)
    for k in (args.models or SIZE_ORDER): show(k, text, args.k, qid)

SCEN = {
 "free-good": ("Easy free-text question, answer exists in the demo corpus; all models should find the Sacramento passage.", ["What is the capital of California?"], None),
 "free-bad-domain": ("Question outside the training domain (cooking, Vietnamese wording, typo): compare how models behave.", ["how to cook pho", "cach nau pho bo", "whta is photosynthsis"], None),
 "free-bad-negation": ("Meaning flips with a word: dense models often still retrieve the same passage.", ["which city is NOT the capital of California"], None),
}
def scenario(args):
    if args.name == "list":
        for k, v in SCEN.items(): log(f"{k:18} {v[0]}")
        log("also: by-qid scenarios are printed by `cases` (all_good, grow_with_size, all_bad, staircase); run: python3 demo.py compare --qid <qid>"); return
    desc, queries, _ = SCEN[args.name]; log(desc)
    for t in queries:
        for k in (args.models or ["tiny", "small", "base"]): show(k, t, args.k)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("prepare"); a.add_argument("--models", nargs="*"); a.add_argument("--force", action="store_true"); a.set_defaults(f=prepare)
    a = sp.add_parser("cases"); a.set_defaults(f=cases)
    for n, f in (("ask", ask), ("compare", compare)):
        a = sp.add_parser(n); a.add_argument("query", nargs="?"); a.add_argument("--qid", type=int); a.add_argument("--k", type=int, default=5)
        a.add_argument("--model", default="base", choices=list(MODELS)); a.add_argument("--models", nargs="*", choices=list(MODELS)); a.set_defaults(f=f)
    a = sp.add_parser("scenario"); a.add_argument("name", choices=list(SCEN) + ["list"]); a.add_argument("--k", type=int, default=3); a.add_argument("--models", nargs="*", choices=list(MODELS)); a.set_defaults(f=scenario)
    args = ap.parse_args(); args.f(args)
