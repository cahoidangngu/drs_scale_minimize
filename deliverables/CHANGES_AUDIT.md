# CHANGES_AUDIT.md

Everything below was ADDED; no existing result, script or log in `drscale_exp/` or `DRScale_repo/` was modified or deleted.

* New folder `drscale_exp/deliverables/` with the package (reports, README, matrix, figures, tables, data copies, scripts, audit outputs, checksums, zip).
* New folder `drscale_exp/sources/` (paper text, Crossref/arXiv/CORE/CCF fetches); reading only, not shipped.
* `data/` holds COPIES of `results.csv`, `results_summary.csv`, `results/*.json`, `status.json`, `logs/run.log`, `verification_report.txt`.
* Figures were redrawn from `data/results/*.json` with correct axis labels (contrastive entropy, NLL in nats). The original PNGs (`scaling_law_*.png`) and `SUMMARY.md` were left untouched.
* Corrections to statements in `SUMMARY.md` (not edited in place): (1) metric name 'Contrastive Perplexity' → contrastive entropy / NLL in nats; (2) the 1,000,000-passage pool is the repository default, not a statement of the paper (paper Sec. 3.5 mentions a 100,000-passage sub-corpus); (3) 'the standard deviation is relatively small' → SD is 2.0%–11.2% of the mean; (4) the number of training pairs is 532,751 unique pairs (532,761 lines, 10 duplicated).
* No experiment was re-run. All checks used CPU only and existing files.
