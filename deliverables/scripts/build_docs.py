#!/usr/bin/env python3
"""Render REPORT_EN/VI, README, compliance matrix and PDFs from templates + computed values."""
import os, sys, re, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, markdown, fitz
from build_values import *
import build_matrix as BM
from report_en import EN
from report_vi import VI
import readme_tpl as RT

V.update(BM.V)
for k in "ABCD": V["verd_" + k] = BM.VERD[k].replace(" (PASS CÓ LƯU Ý)", "")
for k, v in PAPER.items(): V["paper_" + k] = v
f6 = pd.read_csv(os.path.join(FD, "F6_residuals.csv"))
for sw, tag in (("model", "m"), ("data", "d")):
    s = f6[f6.sweep == sw]; V[f"r_{tag}2"] = f"{s.resid_2param.abs().max():.4f}"; V[f"r_{tag}3"] = f"{s.resid_3param.abs().max():.4f}"
tm = ms.set_index(ms.model.str.split("-").str[1].str.lower()); td = ds.set_index("train_instances")
for a in ("tiny", "mini", "small", "medium", "base"): V[f"tt_{a}"] = f"{tm.loc[a, 'train_time']:.1f}"
for d in (2500, 40000): V[f"tt_d{d}"] = f"{td.loc[d, 'train_time']:.1f}"

def render(t):
    miss = set(re.findall(r"<<(\w+)>>", t)) - set(V)
    if miss: raise KeyError(sorted(miss))
    return re.sub(r"<<(\w+)>>", lambda m: str(V[m.group(1)]), t)

TITLES = {"F1": "Experiment workflow", "F2": "DREncoder architecture", "F3": "How the contrastive loss is measured", "F4": "Model-size scaling", "F5": "Data-size scaling", "F6": "Residuals of the fits",
          "F7": "Results table (image)", "F8": "Training curves", "F9": "Leave-one-out stability", "F10": "Comparison with random scoring", "F11": "Training time"}
sec = ["## 5. Figures and tables, one by one\n"]
for fid, img, src, shows, how, key, may, mustnot, cap in RT.FIGS:
    sec.append(f"### {fid}. {TITLES[fid]}\n\n![{fid}]({img})\n")
    sec += [f"- **What it shows:** {shows}", f"- **How to read it:** {how}", f"- **Key numbers:** {key.format(**V)}", f"- **What you MAY claim:** {may}", f"- **What you must NOT claim:** {mustnot.format(**V)}", f"- **Suggested caption:** {cap}", f"- **Source data file:** `data/figure_data/{src}`\n"]
for tid, md_, csv_, shows, key in RT.TABS:
    sec.append(f"### {tid}\n\n- **What it shows:** {shows}\n- **Key numbers:** {key.format(**V)}\n- **Files:** `{md_}` (paste-ready), `{csv_}`\n- **What you MAY claim:** any number in it, with the caveats in Section 6.\n- **What you must NOT claim:** anything the checklist in Section 6 lists as unsafe.\n")
V["gpu_short"] = "NVIDIA RTX 5880 Ada GPU"
DETAIL = render(RT.HEAD).replace("# DRScale scaled-down experiment: package for the report writer", "# Figures and tables guide (detailed)").replace("*Read this file first. ", "*Start with README.md; this file is the detailed reference. ") + "\n".join(sec) + render(RT.TAIL)
open(os.path.join(PKG, "FIGURES_AND_TABLES_GUIDE.md"), "w", encoding="utf-8").write(DETAIL)
from readme_simple_tpl import GUIDE
for dst, pre in ((os.path.join(PKG, "README.md"), ""), (os.path.join(os.path.dirname(PKG), "README.md"), "deliverables/")):
    V["P"] = pre; open(dst, "w", encoding="utf-8").write(render(GUIDE))
V["P"] = ""
open(os.path.join(PKG, "REPORT_EN.md"), "w", encoding="utf-8").write(render(EN))
VI_OVR = {"fit_match": "cả sáu giá trị trong SUMMARY được tính lại khớp đến độ chính xác đã in" if V["fit_match"].startswith("all") else "KHÔNG KHỚP (xem recompute.json)",
          **{"verd_" + k: BM.VERD[k].replace("PASS WITH CAVEATS (PASS CÓ LƯU Ý)", "PASS CÓ LƯU Ý").replace(": definitions match; training length does not", ": định nghĩa khớp, độ dài huấn luyện thì không") for k in "ABCD"}}
_save = {k: V[k] for k in VI_OVR}; V.update(VI_OVR)
open(os.path.join(PKG, "REPORT_VI.md"), "w", encoding="utf-8").write(render(VI) + "\n*Ghi chú: tên cột bảng, danh sách mục kiểm toán và bảng thiết lập giữ nguyên tiếng Anh để khớp với các tệp CSV và ma trận tuân thủ.*\n")
V.update(_save)
BM.write_matrix(PKG)

# ---------------- PDF
CSS = """
@font-face {font-family: dj; src: url(DejaVuSans.ttf);} @font-face {font-family: dj; font-weight: bold; src: url(DejaVuSans-Bold.ttf);}
@font-face {font-family: dj; font-style: italic; src: url(DejaVuSans-Oblique.ttf);} @font-face {font-family: djm; src: url(DejaVuSansMono.ttf);}
body {font-family: dj; font-size: 8.6pt; line-height: 1.32;} h1 {font-size: 15pt;} h2 {font-size: 12pt; margin-top: 12pt; border-bottom: 1px solid #888;} h3 {font-size: 10pt;}
table {border-collapse: collapse; width: 100%; margin: 4pt 0;} th, td {border: 0.5pt solid #999; padding: 1.5pt 2.5pt; font-size: 6.2pt; vertical-align: top;} th {background-color: #dfe9f3;}
code, pre {font-family: djm; font-size: 7.2pt;} pre {background-color: #f2f2f2; padding: 3pt;} img {width: 100%;} p {margin: 3pt 0;} li {margin: 1pt 0;}
"""
def to_pdf(md_name, out_name):
    text = open(os.path.join(PKG, md_name), encoding="utf-8").read()
    html = "<html><body>" + markdown.markdown(text, extensions=["tables", "fenced_code"]).replace(".png\"", ".jpg\"") + "</body></html>"
    import tempfile
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix="pdfimg_"); os.makedirs(os.path.join(tmp, "figures"))
    for f_ in glob.glob(os.path.join(PKG, "figures", "*.png")):
        im = Image.open(f_).convert("RGB"); im.thumbnail((1700, 1700)); im.save(os.path.join(tmp, "figures", os.path.basename(f_)[:-4] + ".jpg"), quality=88)
    arch = fitz.Archive(tmp); arch.add(fitz.Archive("/usr/share/fonts/truetype/dejavu"))
    story = fitz.Story(html=html, user_css=CSS, archive=arch)
    w = fitz.DocumentWriter(os.path.join(PKG, out_name)); mb = fitz.paper_rect("a4"); where = mb + (42, 42, -42, -46)
    more, n = 1, 0
    while more:
        dev = w.begin_page(mb); more, _ = story.place(where); story.draw(dev); w.end_page(); n += 1
    w.close(); return n
for m, p in (("REPORT_EN.md", "REPORT_EN.pdf"), ("REPORT_VI.md", "REPORT_VI.pdf")):
    n = to_pdf(m, p)
    d = fitz.open(os.path.join(PKG, p)); imgs = sum(len(pg.get_images()) for pg in d)
    print(p, n, "pages,", imgs, "images,", os.path.getsize(os.path.join(PKG, p)), "bytes")
# ---------------- CHANGES_AUDIT.md
open(os.path.join(PKG, "CHANGES_AUDIT.md"), "w", encoding="utf-8").write(f"""# CHANGES_AUDIT.md

Everything below was ADDED; no existing result, script or log in `drscale_exp/` or `DRScale_repo/` was modified or deleted.

* New folder `drscale_exp/deliverables/` with the package (reports, README, matrix, figures, tables, data copies, scripts, audit outputs, checksums, zip).
* New folder `drscale_exp/sources/` (paper text, Crossref/arXiv/CORE/CCF fetches); reading only, not shipped.
* `data/` holds COPIES of `results.csv`, `results_summary.csv`, `results/*.json`, `status.json`, `logs/run.log`, `verification_report.txt`.
* Figures were redrawn from `data/results/*.json` with correct axis labels (contrastive entropy, NLL in nats). The original PNGs (`scaling_law_*.png`) and `SUMMARY.md` were left untouched.
* Corrections to statements in `SUMMARY.md` (not edited in place): (1) metric name 'Contrastive Perplexity' → contrastive entropy / NLL in nats; (2) the 1,000,000-passage pool is the repository default, not a statement of the paper (paper Sec. 3.5 mentions a 100,000-passage sub-corpus); (3) 'the standard deviation is relatively small' → SD is {V['relsd_lo']}%–{V['relsd_hi']}% of the mean; (4) the number of training pairs is {V['pairs_total']} unique pairs ({V['pairs_raw_lines']} lines, {V['dup_lines']} duplicated).
* No experiment was re-run. All checks used CPU only and existing files.
""")
