#!/usr/bin/env python3
"""Model audit (CPU only): parameter counts, what the 'non-embedding' count contains, seed effect on init.
Writes audit/model_checks.json.   Run: python3 -I scripts/audit_model.py   (from deliverables/)"""
import os, sys, json, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); EXP = os.path.dirname(PKG)
sys.path.insert(0, EXP)
os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
import torch
from transformers import AutoConfig, set_seed
from modeling import DREncoder

MODELS = ["google/bert_uncased_L-2_H-128_A-2", "google/bert_uncased_L-4_H-256_A-4", "google/bert_uncased_L-4_H-512_A-8",
          "google/bert_uncased_L-8_H-512_A-8", "google/bert_uncased_L-12_H-768_A-12"]
out = {"models": {}}
for m in MODELS:
    with contextlib.redirect_stdout(io.StringIO()):
        set_seed(42)
        cfg = AutoConfig.from_pretrained(m)
        model = DREncoder(config=cfg, init_model=m, vector_dim=768)
    H, L, I = cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size
    named = list(model.named_parameters())
    with_fc = sum(p.numel() for n, p in named if "embedding" not in n)                     # repo definition (modeling.non_embedding_parameters)
    no_fc = sum(p.numel() for n, p in model.encoder.named_parameters() if "embedding" not in n)
    pooler = sum(p.numel() for n, p in named if "pooler" in n)
    layers = sum(p.numel() for n, p in named if ".encoder.layer." in n)
    fc = sum(p.numel() for p in model.output_fc.parameters())
    total = sum(p.numel() for p in model.parameters())
    pooler_has_grad_path = False   # mean pooling reads last_hidden_state only; BertPooler output is never used
    out["models"][m] = dict(hidden=H, layers=L, heads=cfg.num_attention_heads, intermediate=I,
        non_emb_with_fc=with_fc, non_emb_no_fc=no_fc, bert_pooler=pooler, transformer_layers_only=layers,
        output_fc=fc, total=total, with_fc_in_2pow20_units=with_fc / 2**20,
        analytic_layers=L * (12 * H * H + 13 * H), analytic_pooler=H * H + H)
# seed dependence of the randomly initialised projection layer (same code path as training: set_seed -> build model)
w = {}
for s in (42, 43, 44):
    with contextlib.redirect_stdout(io.StringIO()):
        set_seed(s)
        mm = DREncoder(config=AutoConfig.from_pretrained(MODELS[0]), init_model=MODELS[0], vector_dim=768)
    w[s] = mm.output_fc.weight.detach().clone()
out["output_fc_init_differs_across_seeds"] = {f"{a}-{b}": bool(not torch.equal(w[a], w[b])) for a, b in ((42, 43), (42, 44), (43, 44))}
w2 = {}
for s in (42, 42):
    with contextlib.redirect_stdout(io.StringIO()):
        set_seed(s)
        mm = DREncoder(config=AutoConfig.from_pretrained(MODELS[0]), init_model=MODELS[0], vector_dim=768)
    w2.setdefault(s, []).append(mm.output_fc.weight.detach().clone())
out["output_fc_init_reproducible_same_seed"] = bool(torch.equal(w2[42][0], w2[42][1]))
out["torch"] = torch.__version__
import transformers; out["transformers"] = transformers.__version__
json.dump(out, open(os.path.join(PKG, "audit", "model_checks.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
