| Aspect | Paper (section / figure) | Our experiment |
|---|---|---|
| Task / data | MS MARCO passage (8.8M passages, 0.5M train queries) and T2Ranking (Sec. 3.3) | MS MARCO only; passage file truncated to 8,326,970 passages (94.18%) |
| Models | 24 Google BERT checkpoints 0.5M–82M non-emb. params; ERNIE for T2Ranking (Sec. 3.2) | 5 BERT checkpoints: Tiny, Mini, Small, Medium, Base (same family as the paper's red stars, Fig. 3) |
| N definition | non-embedding parameters (Sec. 3.2, footnote 2) | non-embedding incl. projection layer and unused BERT pooler (= the paper's Fig. 6 legend sizes when divided by 2^20) |
| D definition | number of annotated query–passage pairs (Sec. 3.1, 4.2) | number of (qid, pid) training pairs |
| Architecture | shared encoder + projection to 768; CLS or mean pooling (Sec. 3.2, Eq. 2) | shared BERT + mean pooling + Linear(H→768); no normalisation |
| Training loss | softmax cross-entropy, random + in-batch negatives (Sec. 3.4, Eq. 3) | same: 1 positive + 256 random negatives shared by the batch; mean CE; no temperature |
| Optimisation length | fixed 10,000 steps (Sec. 3.4) | 3 epochs → 60–939 steps (depends on D); 237 steps for the model sweep |
| lr / batch / optimiser | not stated in the paper (UNVERIFIED) | lr 1e-4, batch 128, AdamW, linear decay, warmup 10 steps, bf16 |
| Metric | contrastive entropy = −log softmax of the positive among 1+256 candidates (Sec. 3.5, Eq. 5) | same quantity (mean NLL, nats), K = 257 |
| Evaluation set | dev queries (~7,000, Sec. 3.3); 100,000-passage sub-corpus used for the metric study (Sec. 3.5) | first 1,000 dev.small queries by qid (1,086 pairs); 100,000-passage pool; 5 repeats; fixed eval seed 1 |
| Checkpoint reported | best test loss during training, no early stopping (Sec. 4.1) | final checkpoint after epoch 3 |
| Model sweep | full training set, many models (Sec. 4.1, Fig. 3) | D = 10,000 pairs, 5 models |
| Data sweep | BERT-Base, up to all ~500K annotations (Sec. 4.2, Fig. 4) | BERT-Small, D = 2,500 … 40,000 |
| Fit form | L = (A/N)^α + δ (Eq. 6); L = (B/D)^β + δ (Eq. 7); joint Eq. 8 | 3-param a·X^-α + c (same family; a = A^α) and a 2-param a·X^-α (extra, main) |
| Seeds | not stated (UNVERIFIED) | 3 training seeds 42/43/44; evaluation seed fixed |
| Annotation quality, budget allocation | Sec. 4.3 (ICT, docT5query, ChatGLM3), Sec. 5 | not reproduced |
