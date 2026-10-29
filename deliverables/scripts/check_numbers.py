#!/usr/bin/env python3
"""Step 9 quality checks. Independent of common.py: recomputes statistics from data/results/*.json with plain Python/numpy."""
import os, re, sys, json, glob, statistics, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); EXP = os.path.dirname(PKG)
rd = lambda p: open(os.path.join(PKG, p), encoding="utf-8").read()
EN, VI, RM = rd("REPORT_EN.md"), rd("REPORT_VI.md"), rd("README.md") + rd("FIGURES_AND_TABLES_GUIDE.md")
fails, notes = [], []
def check(c, msg):
    (notes if c else fails).append(("PASS " if c else "FAIL ") + msg)

# 1. independent recomputation from JSON
runs = collections.defaultdict(list)
for f in glob.glob(os.path.join(PKG, "data", "results", "*.json")):
    if os.path.basename(f) in ("selected_lr.json", "smoke_test.json"): continue
    j = json.load(open(f)); runs[(j["model_name"], j["train_instances"])].append(j["eval_loss"])
check(len(runs) == 9 and all(len(v) == 3 for v in runs.values()), "9 configurations x 3 runs in JSON")
for (m, d), v in runs.items():
    s = f"{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}"
    for name, txt in (("EN", EN), ("VI", VI)):
        check(s in txt, f"{name}: mean±SD string {s} for D={d} {m.split('_')[-1]}")
def ols(x, y):
    sl, ic = np.polyfit(np.log(x), np.log(y), 1); pr = sl * np.log(x) + ic; ly = np.log(y)
    return -sl, 1 - ((ly - pr) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
order = ["L-2_H-128_A-2", "L-4_H-256_A-4", "L-4_H-512_A-8", "L-8_H-512_A-8", "L-12_H-768_A-12"]
N = {}
for f in glob.glob(os.path.join(PKG, "data", "results", "*.json")):
    if os.path.basename(f) in ("selected_lr.json", "smoke_test.json"): continue
    j = json.load(open(f)); N[j["model_name"].split("bert_uncased_")[1]] = j["non_emb_params"]
xs = [N[o] for o in order]; ys = [statistics.mean(runs[("google/bert_uncased_" + o, 10000)]) for o in order]
a, r2 = ols(xs, ys); check(f"{a:.3f}" in EN and f"{r2:.3f}" in EN, f"model exponent {a:.3f}, R2 {r2:.3f} appear in EN")
ds = [2500, 5000, 10000, 20000, 40000]; yd = [statistics.mean(runs[("google/bert_uncased_L-4_H-512_A-8", d)]) for d in ds]
a, r2 = ols(ds, yd); check(f"{a:.3f}" in EN and f"{r2:.3f}" in EN, f"data exponent {a:.3f}, R2 {r2:.3f} appear in EN")
check(f"{np.log(257):.3f}" in EN and f"{np.log(257):.3f}" in VI and f"{np.log(257):.3f}" in RM, "ln(257) in EN, VI, README")
# results table md equals recomputation
t1 = open(os.path.join(PKG, "tables", "T1_results_mean_sd.md")).read()
check(all(f"{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}" in t1 for v in runs.values()), "tables/T1 matches recomputation")
# 2. EN vs VI numbers
tok = lambda t: collections.Counter(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?", re.sub(r"\*Ghi chú:.*", "", t)))
te, tv = tok(EN), tok(VI); diff = (te - tv) + (tv - te)
check(not diff, f"EN and VI contain identical numeric tokens (differences: {dict(te-tv)} | {dict(tv-te)})")
# 3. paper values exist in paper text
P = open(os.path.join(EXP, "sources", "paper_text.txt"), encoding="utf-8").read()
for lab, v in {"A=3.22": "3.22", "alpha .53": "0.53", "delta .04": "0.04", "R2 .991": "0.991", "B=3.49": "3.49", "beta 1.05": "1.05", "delta .05": "0.05", "R2 .954": "0.954", "joint A 3.6": "3.6 ×", "joint B 7.1": "7.1 ×", "joint alpha .56": "0.56", "joint beta 1.31": "1.31", "10,000 steps": "10,000 steps", "256 negatives": "256", "100,000": "100,000", "82M": "82 million", "87M": "87 million", "$20,000": "20,000", "13 billion": "13 billion", "SIGIR '24": "SIGIR ’24", "pages DOI": "10.1145/3626772.3657743", "bug v2": None}.items():
    if v is None: continue
    check(v in P, f"paper text contains '{v}' ({lab})")
check("fixes a bug in the experiments" in open(os.path.join(EXP, "sources", "arxiv.xml")).read(), "arXiv comment on v2 present in fetched metadata")
check(json.load(open(os.path.join(EXP, "sources", "crossref.json")))["message"]["page"] == "1339-1349", "Crossref pages 1339-1349")
# 4. images
for name, txt in (("EN", EN), ("VI", VI), ("README", RM)):
    for p in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", txt):
        fp = os.path.join(PKG, p); check(os.path.exists(fp) and os.path.getsize(fp) > 1000, f"{name}: image {p} exists and is non-empty")
import fitz
for p in ("REPORT_EN.pdf", "REPORT_VI.pdf"):
    d = fitz.open(os.path.join(PKG, p)); n = sum(len(pg.get_images()) for pg in d); check(n >= 11, f"{p}: {len(d)} pages, {n} embedded images (>=11)")
for f in glob.glob(os.path.join(PKG, "figures", "*.png")):
    from PIL import Image
    im = Image.open(f); dpi = im.info.get("dpi", (0, 0))[0]; check(dpi >= 299, f"{os.path.basename(f)} dpi {dpi:.0f}")
check(len(glob.glob(os.path.join(PKG, "figures", "*.svg"))) == 11, "11 SVG figures")
check(len(glob.glob(os.path.join(PKG, "data", "figure_data", "F*.csv"))) >= 11, "figure_data has a CSV for every figure")
# 5. language
VN = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)
check(not VN.search(EN), "REPORT_EN has no Vietnamese characters")
check(not VN.search(rd("README.md")) and not VN.search(rd("FIGURES_AND_TABLES_GUIDE.md").split("## 12.")[0]), "README (and guide before its Vietnamese summary) has no Vietnamese characters")
check(bool(VN.search(VI)), "REPORT_VI is Vietnamese")
# 6. old label
bad = []
for name, txt in (("EN", EN), ("VI", VI), ("README", RM), ("matrix", rd("AUDIT_COMPLIANCE_MATRIX.md"))):
    for ln in txt.splitlines():
        if re.search(r"contrastive perplexity", ln, re.I) and "[OLD-LABEL]" not in ln and "contrastive_perplexity" not in ln.replace("`", ""): bad.append((name, ln[:90]))
check(not bad, f"old 'Contrastive Perplexity' only appears marked [OLD-LABEL] ({bad})")
for f in glob.glob(os.path.join(PKG, "tables", "*")) + glob.glob(os.path.join(PKG, "data", "figure_data", "*")) + [os.path.join(HERE, "make_figures.py")]:
    check("perplexity" not in open(f, encoding="utf-8", errors="ignore").read().lower().replace("not a perplexity", "").replace("not exponentiated", ""), f"no 'perplexity' label in {os.path.relpath(f, PKG)}") if "perplexity" in open(f, encoding="utf-8", errors="ignore").read().lower() else None
mf = open(os.path.join(HERE, "make_figures.py")).read().replace("NOT a perplexity (not exponentiated)", "")
check("Contrastive Perplexity" not in mf and "Contrastive perplexity" not in mf, "make_figures.py has no wrong label")
# 7. paper claims carry references
for ln in EN.splitlines():
    if ln.startswith("**Paper says"):
        check(bool(re.search(r"(Sec\.|Fig\.|Table|Eq\.|UNVERIFIED)", ln)), f"paper claim has a reference: {ln[:60]}...")
# 8. compliance matrix counts
import csv
rows = list(csv.reader(open(os.path.join(PKG, "AUDIT_COMPLIANCE_MATRIX.csv"), encoding="utf-8")))[1:]
cnt = collections.Counter(r[6] for r in rows); check(f"{cnt['MATCH']} MATCH" in EN and f"{cnt['UNVERIFIED']} UNVERIFIED" in EN, f"matrix counts {dict(cnt)} quoted in report")
print("\n".join(notes[-12:])); print(f"{len(notes)} passed, {len(fails)} failed")
if fails: print("\n".join(fails)); sys.exit(1)
