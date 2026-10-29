#!/usr/bin/env python3
"""Read the TrainingArguments actually used (training_args.bin saved by HF Trainer next to each checkpoint) for all 27 runs.
Own, trusted files only. Writes audit/actual_training_config.csv"""
import os, sys, glob, json
import torch, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); EXP = os.path.dirname(PKG)
sys.path.insert(0, EXP); os.environ["USE_TF"] = "0"
import training_arguments  # noqa: registers MyTrainingArguments for unpickling
rows = []
for f in sorted(glob.glob(os.path.join(EXP, "checkpoints", "*", "checkpoint-*", "training_args.bin"))):
    key = f.split(os.sep)[-3]
    if key.startswith(("smoke", "lr_test")): continue
    a = torch.load(f, weights_only=False)
    ts = json.load(open(os.path.join(os.path.dirname(f), "trainer_state.json")))
    lr10 = next((h["learning_rate"] for h in ts["log_history"] if h.get("step") == 10), None)
    rows.append(dict(run=key, seed=a.seed, lr=a.learning_rate, epochs=a.num_train_epochs, batch=a.per_device_train_batch_size, grad_accum=a.gradient_accumulation_steps,
        bf16=a.bf16, fp16=a.fp16, warmup_steps=a.warmup_steps, warmup_ratio=a.warmup_ratio, weight_decay=a.weight_decay, optim=str(a.optim), scheduler=str(a.lr_scheduler_type),
        max_grad_norm=a.max_grad_norm, adam_beta2=a.adam_beta2, grad_ckpt=a.gradient_checkpointing, max_steps=ts["max_steps"], lr_logged_step10=lr10))
df = pd.DataFrame(rows); df.to_csv(os.path.join(PKG, "audit", "actual_training_config.csv"), index=False)
print(df.drop(columns=["run"]).drop_duplicates().to_string()); print(len(df), "runs"); print(df[["run","seed"]].to_string())
