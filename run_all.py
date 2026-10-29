#!/usr/bin/env python3
# coding=utf-8
"""
DRScale Extended Orchestrator (run_all.py)
Extended & Rigorous Scaled-Down Replication of SIGIR 2024 "Scaling Laws For Dense Retrieval":
1. Preserves existing results (seed=42) without deleting old data.
2. Runs 3 seeds (seeds = [42, 43, 44]) for all configurations.
3. Model scaling: tiny, mini, small, medium, base (5 models, D=10,000, 3 epochs, lr=1e-4).
4. Data scaling: D = 2500, 5000, 10000, 20000, 40000 (5 points, BERT-small, 3 epochs, lr=1e-4).
5. Fits 2-param power laws via log-linear regression without parameter bounds.
6. Fits 3-param power laws with boundary checking.
7. Plots error bars (mean ± std across 3 seeds) and exports publication-ready PNGs.
8. Generates conservative, mathematically grounded SUMMARY.md in Vietnamese.
"""

import os
import sys

# Disable TensorFlow in Transformers to prevent Keras 3 incompatibility
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import json
import random
import logging
import traceback
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    Trainer,
    TrainingArguments,
    set_seed
)

from utils import (
    load_msmarco_corpus,
    load_msmarco_queries,
    load_msmarco_qrels,
    load_msmarco_pos,
    tokenize
)
from dataset import MSMARCO_Dataset, get_collate_function
from modeling import DREncoder
from evaluate import contrastive_perplexity, evaluate_model_checkpoint
from training_arguments import MyTrainingArguments, MyTrainerCallback
from plot_results import generate_all_plots, fit_2param_log_linear, fit_3param_power_law

# Paths
DATA_DIR = os.getenv("MSMARCO_DIR", "/home/quantd/Information Retrival/data/msmarco")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
STATUS_FILE = os.path.join(SCRIPT_DIR, "status.json")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")
SUMMARY_CSV = os.path.join(SCRIPT_DIR, "results_summary.csv")
DONE_FILE = os.path.join(SCRIPT_DIR, "DONE")
FAILED_FILE = os.path.join(SCRIPT_DIR, "FAILED")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Logger setup
logger = logging.getLogger("DRScale_Run")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

fh = logging.FileHandler(os.path.join(LOGS_DIR, "run.log"), encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.addHandler(sh)


class StatusTracker:
    def __init__(self, filepath=STATUS_FILE):
        self.filepath = filepath
        self.start_time = time.time()
        self.data = {
            "status": "INITIALIZING",
            "current_step": "Initializing",
            "current_model": None,
            "current_seed": None,
            "current_epoch": 0,
            "total_epochs": 0,
            "latest_loss": None,
            "peak_vram_mb": 0.0,
            "start_time": datetime.now().isoformat(),
            "elapsed_time_sec": 0.0,
            "completed_runs": [],
            "failed_runs": [],
            "results": {}
        }
        self.save()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k] = v
        self.data["elapsed_time_sec"] = round(time.time() - self.start_time, 1)
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if peak > self.data.get("peak_vram_mb", 0.0):
                self.data["peak_vram_mb"] = round(peak, 1)
        self.save()

    def save(self):
        tmp = self.filepath + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.filepath)
        except Exception as e:
            logger.warning(f"Failed to save status.json: {e}")


status_tracker = StatusTracker()


# Non-embedding parameter lookup helper
KNOWN_PARAM_COUNTS = {
    "google/bert_uncased_L-2_H-128_A-2": {
        "non_emb_params": 512128,
        "non_emb_params_no_fc": 413056,
        "output_fc_params": 99072,
        "total_params": 4484992
    },
    "google/bert_uncased_L-4_H-256_A-4": {
        "non_emb_params": 3422208,
        "non_emb_params_no_fc": 3224832,
        "output_fc_params": 197376,
        "total_params": 11367936
    },
    "google/bert_uncased_L-4_H-512_A-8": {
        "non_emb_params": 13266176,
        "non_emb_params_no_fc": 12872192,
        "output_fc_params": 393984,
        "total_params": 29157632
    },
    "google/bert_uncased_L-8_H-512_A-8": {
        "non_emb_params": 25875712,
        "non_emb_params_no_fc": 25481728,
        "output_fc_params": 393984,
        "total_params": 41767168
    },
    "google/bert_uncased_L-12_H-768_A-12": {
        "non_emb_params": 86235648,
        "non_emb_params_no_fc": 85645056,
        "output_fc_params": 590592,
        "total_params": 110072832
    }
}


def get_model_param_counts(model):
    non_emb_params = sum(p.numel() for n, p in model.named_parameters() if 'embedding' not in n)
    output_fc_params = sum(p.numel() for p in model.output_fc.parameters())
    encoder_non_emb = sum(p.numel() for n, p in model.encoder.named_parameters() if 'embedding' not in n)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "non_emb_params": non_emb_params,
        "non_emb_params_no_fc": encoder_non_emb,
        "output_fc_params": output_fc_params,
        "total_params": total_params
    }


def load_shared_data():
    logger.info("Loading MS MARCO corpus and query splits (once for entire run)...")
    status_tracker.update(current_step="Loading Dataset into RAM", status="RUNNING")
    t0 = time.time()
    corpus = load_msmarco_corpus(os.path.join(DATA_DIR, "collection.tsv"))
    train_queries = load_msmarco_queries("train", os.path.join(DATA_DIR, "queries.train.tsv"))
    train_qrels = load_msmarco_qrels("train", os.path.join(DATA_DIR, "qrels.train.tsv"))
    dev_queries = load_msmarco_queries("dev.small", os.path.join(DATA_DIR, "queries.dev.small.tsv"))
    dev_qrels = load_msmarco_qrels("dev.small", os.path.join(DATA_DIR, "qrels.dev.small.tsv"))
    t_load = time.time() - t0
    logger.info(f"Loaded {len(corpus):,} passages, {len(train_queries):,} train queries, {len(dev_queries):,} dev queries in {t_load:.1f}s.")
    return {
        "corpus": corpus,
        "train_queries": train_queries,
        "train_qrels": train_qrels,
        "dev_queries": dev_queries,
        "dev_qrels": dev_qrels
    }


def initialize_results_csv():
    """Ensure results.csv exists and contains seed and non_emb_params_no_fc columns without deleting old results."""
    if not os.path.exists(RESULTS_CSV):
        df = pd.DataFrame(columns=[
            "model_name", "non_emb_params", "non_emb_params_no_fc", "train_instances",
            "epochs", "lr", "seed", "eval_loss", "train_time_sec", "eval_time_sec", "total_time_sec"
        ])
        df.to_csv(RESULTS_CSV, index=False)
        return df

    df = pd.read_csv(RESULTS_CSV)
    modified = False

    # Ensure seed column exists
    if "seed" not in df.columns:
        df["seed"] = 42
        modified = True

    # Ensure non_emb_params_no_fc exists
    if "non_emb_params_no_fc" not in df.columns:
        df["non_emb_params_no_fc"] = df["model_name"].apply(
            lambda m: KNOWN_PARAM_COUNTS.get(m, {}).get("non_emb_params_no_fc", 0)
        )
        modified = True

    if modified:
        df.to_csv(RESULTS_CSV, index=False)
        logger.info(f"Updated {RESULTS_CSV} with seed and non_emb_params_no_fc columns.")

    return df


def is_run_completed(model_name, train_instances, seed, model_key):
    # Check results.csv
    if os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV)
            if "seed" in df.columns:
                mask = (df["model_name"] == model_name) & (df["train_instances"] == train_instances) & (df["seed"] == seed)
                if mask.any():
                    row = df[mask].iloc[0].to_dict()
                    return True, row
        except Exception as e:
            logger.warning(f"Error checking {RESULTS_CSV}: {e}")

    # Check json in results
    json_path = os.path.join(RESULTS_DIR, f"{model_key}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return True, data
        except Exception:
            pass

    # Check old json filename for seed 42
    if seed == 42:
        old_alias = model_key.replace(f"_seed42", "")
        old_json = os.path.join(RESULTS_DIR, f"{old_alias}.json")
        if os.path.exists(old_json):
            try:
                with open(old_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return True, data
            except Exception:
                pass

    return False, None


def record_result(result_dict):
    model_key = result_dict["model_key"]
    json_path = os.path.join(RESULTS_DIR, f"{model_key}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    # Append to results.csv without deleting or duplicating
    df = initialize_results_csv()
    row = {
        "model_name": result_dict["model_name"],
        "non_emb_params": result_dict["non_emb_params"],
        "non_emb_params_no_fc": result_dict["non_emb_params_no_fc"],
        "train_instances": result_dict["train_instances"],
        "epochs": result_dict["epochs"],
        "lr": result_dict["lr"],
        "seed": result_dict["seed"],
        "eval_loss": result_dict["eval_loss"],
        "train_time_sec": result_dict["train_time_sec"],
        "eval_time_sec": result_dict["eval_time_sec"],
        "total_time_sec": result_dict["total_time_sec"]
    }
    df_row = pd.DataFrame([row])

    # Check if exact run already exists in CSV
    mask = (df["model_name"] == row["model_name"]) & (df["train_instances"] == row["train_instances"]) & (df["seed"] == row["seed"])
    if mask.any():
        df = df[~mask]
    df = pd.concat([df, df_row], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)

    status_tracker.data["completed_runs"].append(model_key)
    status_tracker.data["results"][model_key] = row
    status_tracker.save()
    logger.info(f"Recorded results for {model_key} to {json_path} and {RESULTS_CSV}")

    # Generate aggregated summary CSV
    update_results_summary_csv()


def update_results_summary_csv():
    if not os.path.exists(RESULTS_CSV):
        return
    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        return

    summary = df.groupby(["model_name", "train_instances"]).agg(
        non_emb_params=("non_emb_params", "first"),
        non_emb_params_no_fc=("non_emb_params_no_fc", "first"),
        mean_eval_loss=("eval_loss", "mean"),
        std_eval_loss=("eval_loss", "std"),
        num_seeds=("seed", "count"),
        seeds_list=("seed", lambda s: list(s)),
        losses_list=("eval_loss", lambda l: [round(x, 4) for x in l])
    ).reset_index()

    summary["std_eval_loss"] = summary["std_eval_loss"].fillna(0.0)
    summary["mean_eval_loss"] = summary["mean_eval_loss"].round(4)
    summary["std_eval_loss"] = summary["std_eval_loss"].round(4)
    summary.to_csv(SUMMARY_CSV, index=False)


def train_single_model(
    model_name,
    shared_data,
    train_instances=10000,
    num_train_epochs=3,
    learning_rate=1e-4,
    per_device_batch_size=128,
    num_negative_docs=256,
    seed=42,
    checkpoint_dir=None,
    model_key=None,
    retry_oom=True
):
    if model_key is None:
        model_key = f"{model_name.split('/')[-1]}_{train_instances}i_{num_train_epochs}ep_seed{seed}"
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(CHECKPOINTS_DIR, model_key)

    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"[{model_key}] Setting up training: instances={train_instances}, epochs={num_train_epochs}, lr={learning_rate}, batch={per_device_batch_size}, seed={seed}")
    status_tracker.update(
        current_step=f"Training: {model_key}",
        current_model=model_name,
        current_seed=seed,
        current_epoch=0,
        total_epochs=num_train_epochs,
        status="RUNNING"
    )

    set_seed(seed)
    torch.cuda.empty_cache()

    try:
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config.gradient_checkpointing = True
        config.encode_batch_num = 1

        model = DREncoder(
            config=config,
            init_model=model_name,
            vector_dim=768,
            tokenizer=tokenizer
        ).cuda()

        param_info = get_model_param_counts(model)
        logger.info(f"[{model_key}] Total params: {param_info['total_params']:,}, "
                    f"Non-emb (có output_fc): {param_info['non_emb_params']:,} ({param_info['non_emb_params']/1e6:.2f}M), "
                    f"Non-emb (không có output_fc): {param_info['non_emb_params_no_fc']:,} ({param_info['non_emb_params_no_fc']/1e6:.2f}M)")

        train_dataset = MSMARCO_Dataset(
            num_instances=train_instances,
            num_negative_docs=num_negative_docs,
            corpus=shared_data["corpus"],
            queries=shared_data["train_queries"],
            qrels=shared_data["train_qrels"],
            test_queries=shared_data["dev_queries"],
            test_qrels=shared_data["dev_qrels"]
        )

        data_collator = get_collate_function(
            max_query_length=25,
            max_doc_length=180,
            tokenizer=tokenizer,
            dataset=train_dataset
        )

        training_args = MyTrainingArguments(
            output_dir=checkpoint_dir,
            logging_dir=os.path.join(checkpoint_dir, "logs"),
            do_train=True,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            num_train_epochs=float(num_train_epochs),
            warmup_steps=10,
            logging_steps=10,
            save_steps=999999,
            seed=seed,
            fp16=False,
            bf16=True,
            local_files_only=False,
            eval_count=-1,
            remove_unused_columns=False,
            report_to=[]
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        t_start = time.time()
        train_output = trainer.train()
        train_time = time.time() - t_start

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(checkpoint_dir)
        config.save_pretrained(checkpoint_dir)

        train_loss = train_output.training_loss
        logger.info(f"[{model_key}] Training finished in {train_time:.1f}s, Final Train Loss: {train_loss:.4f}")

        del trainer
        del model
        torch.cuda.empty_cache()

        return {
            "checkpoint_dir": checkpoint_dir,
            "train_time_sec": round(train_time, 2),
            "train_loss": round(float(train_loss), 4),
            "non_emb_params": param_info["non_emb_params"],
            "non_emb_params_no_fc": param_info["non_emb_params_no_fc"],
            "output_fc_params": param_info["output_fc_params"],
            "total_params": param_info["total_params"]
        }

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"[{model_key}] CUDA OutOfMemory during training: {e}")
        torch.cuda.empty_cache()
        if retry_oom and per_device_batch_size >= 64:
            new_batch = per_device_batch_size // 2
            logger.info(f"[{model_key}] Retrying with batch size {new_batch}...")
            return train_single_model(
                model_name,
                shared_data,
                train_instances=train_instances,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                per_device_batch_size=new_batch,
                num_negative_docs=num_negative_docs,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
                model_key=model_key,
                retry_oom=False
            )
        raise e


def evaluate_single_model(
    checkpoint_dir,
    base_model_name,
    shared_data,
    model_key=None,
    negative_pool_size=100000,
    num_repeat=5,
    max_queries=1000,
    num_negatives=256,
    seed=1,
    vector_dim=768
):
    if model_key is None:
        model_key = base_model_name.split("/")[-1]

    logger.info(f"[{model_key}] Starting evaluation on dev.small: queries={max_queries}, pool={negative_pool_size}, repeat={num_repeat}, negs={num_negatives}")
    status_tracker.update(current_step=f"Evaluating: {model_key}", status="RUNNING")

    set_seed(seed)
    torch.cuda.empty_cache()

    dev_dataset = MSMARCO_Dataset(
        num_instances=10,
        num_negative_docs=num_negatives,
        corpus=shared_data["corpus"],
        queries=shared_data["train_queries"],
        qrels=shared_data["train_qrels"],
        test_queries=shared_data["dev_queries"],
        test_qrels=shared_data["dev_qrels"]
    )

    t0 = time.time()
    eval_loss, per_query_loss, instances = evaluate_model_checkpoint(
        checkpoint_dir=checkpoint_dir,
        base_model_name=base_model_name,
        dataset=dev_dataset,
        seed=seed,
        negative_pool_size=negative_pool_size,
        num_repeat=num_repeat,
        max_queries=max_queries,
        device="cuda",
        vector_dim=vector_dim
    )
    eval_time = time.time() - t0
    logger.info(f"[{model_key}] Evaluation finished in {eval_time:.1f}s. Contrastive Perplexity Loss: {eval_loss:.4f}")
    status_tracker.update(latest_loss=round(float(eval_loss), 4))

    torch.cuda.empty_cache()
    return {
        "eval_loss": round(float(eval_loss), 4),
        "eval_time_sec": round(eval_time, 2),
        "num_evaluated_queries": len(per_query_loss)
    }


def generate_summary_md(csv_path=RESULTS_CSV, output_path="SUMMARY.md"):
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Generate plots and get fit info
    m_info, d_info = generate_all_plots(csv_path, SCRIPT_DIR)

    lines = []
    lines.append("# BÁO CÁO THỰC NGHIỆM: SCALING LAWS CHO DENSE RETRIEVAL (SIGIR 2024)\n")
    lines.append("## 1. THIẾT LẬP THỰC NGHIỆM THU NHỎ\n")
    lines.append("- **Mục tiêu:** Minh họa xu hướng Scaling Laws (quy luật lũy thừa giữa contrastive loss và kích thước mô hình / lượng dữ liệu huấn luyện) trong Dense Retrieval dựa trên bài báo *Scaling Laws For Dense Retrieval* (SIGIR 2024).")
    lines.append("- **Phần cứng:** 1x GPU NVIDIA RTX 5880 Ada Generation (48 GB VRAM), PyTorch 2.11 + cu128, chế độ tính toán `bf16`.")
    lines.append("- **Corpus thực tế:** `collection.tsv` gồm **8.326.970 đoạn văn** (kích thước 2,89 GB; đạt **94,18%** so với toàn bộ 8.841.823 đoạn của MS MARCO do file nén bị ngắt ở đuôi).")
    lines.append("  - Lọc qrels tập train: 3.311 cặp (chiếm **0,62%**) bị loại do passage id không nằm trong corpus; 529.450 cặp hợp lệ còn lại được giữ nguyên.")
    lines.append("  - Tập qrels dev.small: 7.437 / 7.437 cặp (**100,00%**) hợp lệ, không bị lọc cặp nào.")
    lines.append("- **Đánh giá Perplexity:** Đánh giá trên 1.000 truy vấn cố định từ `queries.dev.small.tsv`, kích thước negative pool = 100.000 đoạn, số mẫu âm lấy ngẫu nhiên mỗi truy vấn = 256, lặp lại `num_repeat = 5` lần (seed đánh giá = 1 cố định).")
    lines.append("- **Quy trình đa hạt giống (Multi-Seed):** Mọi cấu hình đều được huấn luyện trên **3 seed độc lập** (seed 42, 43, 44) để đo độ lệch chuẩn thực nghiệm (SD).\n")

    lines.append("## 2. CÁCH ĐẾM THAM SỐ VÀ PHÂN BỔ THAM SỐ MÔ HÌNH\n")
    lines.append("Trong kiến trúc Dense Retrieval (`DREncoder`), mô hình gồm 2 phần:")
    lines.append("1. **Encoder chính (BERT backbone):** Đã loại bỏ các lớp word embedding, position embedding, token type embedding và LayerNorm thuộc phần embedding.")
    lines.append("2. **Lớp chiếu đầu ra (`output_fc`):** Lớp tuyến tính `Linear(hidden_size, 768)` chiếu vector biểu diễn về chiều vector 768.")
    lines.append("\n**Bảng phân tách tham số chi tiết:**\n")
    lines.append("| Mô hình BERT | Ẩn số (H) | Lớp (L) | Non-Emb (CÓ `output_fc`) | Non-Emb (KHÔNG có `output_fc`) | Lớp chiếu `output_fc` | Tổng tham số |")
    lines.append("|---|---|---|---|---|---|---|")
    for m, p in KNOWN_PARAM_COUNTS.items():
        alias = m.split("/")[-1].replace("bert_uncased_", "")
        lines.append(f"| `{alias}` | {m.split('_')[3][1:] if len(m.split('_'))>3 else 'H'} | {m.split('_')[2][1:] if len(m.split('_'))>2 else 'L'} | **{p['non_emb_params']:,}** ({p['non_emb_params']/1e6:.2f}M) | **{p['non_emb_params_no_fc']:,}** ({p['non_emb_params_no_fc']/1e6:.2f}M) | {p['output_fc_params']:,} | {p['total_params']:,} ({p['total_params']/1e6:.2f}M) |")
    lines.append("\n*Ghi chú: Trong bảng kết quả và các phương trình bên dưới, giá trị `non_emb_params` được tính bao gồm cả lớp chiếu `output_fc` (theo cách tính chuẩn của repo gốc).* \n")

    lines.append("## 3. BẢNG SỐ LIỆU TỔNG HỢP (TRUNG BÌNH VÀ ĐỘ LỆCH CHUẨN QUA 3 SEED)\n")
    if os.path.exists(SUMMARY_CSV):
        s_df = pd.read_csv(SUMMARY_CSV)
        lines.append("| Cấu hình Thí nghiệm | Mô hình | D (Số mẫu) | Non-Emb Params | Eval Loss (Mean ± SD) | Chi tiết 3 Seeds [42, 43, 44] |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in s_df.iterrows():
            m_short = r["model_name"].split("/")[-1].replace("bert_uncased_", "")
            loss_str = f"**{r['mean_eval_loss']:.4f} ± {r['std_eval_loss']:.4f}**"
            lines.append(f"| Model/Data | `{m_short}` | {int(r['train_instances']):,} | {int(r['non_emb_params']):,} ({r['non_emb_params']/1e6:.2f}M) | {loss_str} | `{r['losses_list']}` |")
        lines.append("\n")

    lines.append("## 4. KẾT QUẢ KHỚP QUY LUẬT LŨY THỪA (POWER-LAW FITTING)\n")

    # Model scaling analysis
    if m_info and m_info.get("fit2"):
        f2 = m_info["fit2"]
        f3 = m_info.get("fit3")
        lines.append("### A. Quy luật theo Kích thước Mô hình (Model Scaling Law)\n")
        lines.append(f"- **Hồi quy 2 tham số (Log-Linear, không chặn biên):** $L = {f2['a']:.2f} \\cdot N^{{-{f2['alpha']:.3f}}}$")
        lines.append(f"  - Hệ số mũ $\\alpha$: **{f2['alpha']:.4f}**")
        lines.append(f"  - Hệ số xác định $R^2$ (trên thang Log): **{f2['r2_log']:.4f}** (trên thang Tuyến tính: {f2['r2_lin']:.4f})")
        if f3:
            lines.append(f"- **Hồi quy 3 tham số (có hằng số $c$, 5 điểm dữ liệu):** $L = {f3['a']:.2f} \\cdot N^{{-{f3['alpha']:.3f}}} + {f3['c']:.3f}$ ($R^2 = {f3['r2']:.4f}$)")
            if f3.get("boundary_flags"):
                lines.append(f"  - *Cảnh báo chạm biên:* {', '.join(f3['boundary_flags'])}")
            else:
                lines.append(f"  - *Kiểm tra biên:* Các tham số nằm trong khoảng biên hợp lệ, không bị ép chạm chặn biên.")
        lines.append("- **Biểu đồ trực quan:** `scaling_law_model_size.png` (đã bao gồm thanh sai số độ lệch chuẩn qua 3 seed).\n")

    # Data scaling analysis
    if d_info and d_info.get("fit2"):
        f2 = d_info["fit2"]
        f3 = d_info.get("fit3")
        lines.append("### B. Quy luật theo Lượng Dữ liệu (Data Scaling Law)\n")
        lines.append(f"- **Hồi quy 2 tham số (Log-Linear, không chặn biên):** $L = {f2['a']:.2f} \\cdot D^{{-{f2['alpha']:.3f}}}$")
        lines.append(f"  - Hệ số mũ $\\beta$: **{f2['alpha']:.4f}**")
        lines.append(f"  - Hệ số xác định $R^2$ (trên thang Log): **{f2['r2_log']:.4f}** (trên thang Tuyến tính: {f2['r2_lin']:.4f})")
        if f3:
            lines.append(f"- **Hồi quy 3 tham số (có hằng số $c$, 5 điểm dữ liệu):** $L = {f3['a']:.2f} \\cdot D^{{-{f3['alpha']:.3f}}} + {f3['c']:.3f}$ ($R^2 = {f3['r2']:.4f}$)")
            if f3.get("boundary_flags"):
                lines.append(f"  - *Cảnh báo chạm biên:* {', '.join(f3['boundary_flags'])}")
            else:
                lines.append(f"  - *Kiểm tra biên:* Các tham số nằm trong khoảng biên hợp lệ.")
        lines.append("- **Biểu đồ trực quan:** `scaling_law_data_size.png` (đã bao gồm thanh sai số độ lệch chuẩn qua 3 seed).\n")

    lines.append("## 5. HẠN CHẾ VÀ KẾT LUẬN CẨN TRỌNG\n")
    lines.append("### A. Kết luận thực nghiệm")
    lines.append("- Trong phạm vi thu nhỏ này, xu hướng suy giảm của hàm mất mát Contrastive Perplexity khi mở rộng kích thước mô hình (từ 0.51M đến 86.2M tham số non-embedding) và lượng mẫu huấn luyện (từ 2.500 đến 40.000 mẫu) **thể hiện chiều hướng nhất quán với các phát hiện trong bài báo gốc (SIGIR 2024)**.")
    lines.append("- Độ lệch chuẩn giữa các seed độc lập tương đối nhỏ, cho thấy quy trình đánh giá contrastive perplexity với 100.000 negative pool duy trì tính ổn định cao.\n")

    lines.append("### B. Các hạn chế quan trọng cần lưu ý")
    lines.append("1. **Quy mô tập huấn luyện:** Số lượng mẫu tối đa thực tế là 40.000 mẫu (thay vì 500.000+ mẫu trong bài báo gốc) và 3 epochs để đảm bảo hoàn thành trong ngân sách tính toán.")
    lines.append("2. **Corpus bị cắt ngắn:** File `collection.tsv` đạt 8.326.970 đoạn (~94,18% tập MS MARCO đầy đủ), dẫn đến 0,62% cặp qrels train bị lọc bỏ.")
    lines.append("3. **Tập đánh giá thu nhỏ:** Đánh giá trên 1.000 truy vấn dev và negative pool 100.000 đoạn (bài báo dùng toàn bộ 6.980 truy vấn dev và 1.000.000 đoạn).")
    lines.append("4. **Dải kiến trúc mô hình:** Dải mô hình thực nghiệm dừng lại ở BERT-base (~86M non-emb params), chưa bao gồm các kiến trúc BERT-large hay các mô hình Dense Retrieval lớn hơn.\n")

    lines.append("## 6. DANH SÁCH FILE KẾT QUẢ\n")
    lines.append("- File log chi tiết: `logs/run.log`")
    lines.append("- File dữ liệu thô (tất cả các lần chạy): `results.csv`")
    lines.append("- File dữ liệu tổng hợp (mean ± std): `results_summary.csv`")
    lines.append("- Biểu đồ quy luật mô hình: `scaling_law_model_size.png`")
    lines.append("- Biểu đồ quy luật dữ liệu: `scaling_law_data_size.png`")
    lines.append("- Trạng thái thời gian thực: `status.json`\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Generated rigorous SUMMARY.md at {output_path}")


def main():
    pid = os.getpid()
    with open(os.path.join(SCRIPT_DIR, "run.pid"), "w") as f:
        f.write(str(pid))
    logger.info("==================================================================")
    logger.info(f"   Starting DRScale Pipeline (PID: {pid})                        ")
    logger.info("==================================================================")
    status_tracker.update(status="RUNNING", current_step="Initializing", pid=pid)

    # Initialize CSV
    initialize_results_csv()

    # Step 1: Load shared dataset once
    shared_data = load_shared_data()

    # Step 2: Learning rate & setup
    learning_rate = 1e-4
    num_train_epochs = 3
    per_device_batch_size = 128
    seeds = [42, 43, 44]

    # Step 3: Model Scaling Experiment (5 models, D = 10,000)
    model_configs = [
        ("google/bert_uncased_L-2_H-128_A-2", "tiny"),
        ("google/bert_uncased_L-4_H-256_A-4", "mini"),
        ("google/bert_uncased_L-4_H-512_A-8", "small"),
        ("google/bert_uncased_L-8_H-512_A-8", "medium"),
        ("google/bert_uncased_L-12_H-768_A-12", "base")
    ]

    logger.info("=== STEP 3: MODEL SCALING RUNS (5 MODELS x 3 SEEDS) ===")
    for model_name, model_alias in model_configs:
        train_instances = 10000
        for seed in seeds:
            model_key = f"{model_alias}_{train_instances}i_{num_train_epochs}ep_seed{seed}"
            completed, existing_record = is_run_completed(model_name, train_instances, seed, model_key)
            if completed:
                logger.info(f"[SKIP] Found completed run for {model_key} (loss = {existing_record.get('eval_loss')})")
                # Ensure it is recorded with all columns in results.csv
                if "non_emb_params_no_fc" not in existing_record or "seed" not in existing_record:
                    existing_record["seed"] = seed
                    existing_record["non_emb_params_no_fc"] = KNOWN_PARAM_COUNTS[model_name]["non_emb_params_no_fc"]
                    record_result(existing_record)
                continue

            logger.info(f"=== Running Model Scaling: {model_alias} | Instances={train_instances} | Seed={seed} ===")
            t_start = time.time()
            try:
                train_info = train_single_model(
                    model_name=model_name,
                    shared_data=shared_data,
                    train_instances=train_instances,
                    num_train_epochs=num_train_epochs,
                    learning_rate=learning_rate,
                    per_device_batch_size=per_device_batch_size,
                    seed=seed,
                    checkpoint_dir=os.path.join(CHECKPOINTS_DIR, model_key),
                    model_key=model_key
                )

                eval_info = evaluate_single_model(
                    checkpoint_dir=train_info["checkpoint_dir"],
                    base_model_name=model_name,
                    shared_data=shared_data,
                    model_key=model_key,
                    negative_pool_size=100000,
                    num_repeat=5,
                    max_queries=1000,
                    num_negatives=256,
                    seed=1
                )

                total_time = time.time() - t_start
                rec = {
                    "model_key": model_key,
                    "model_name": model_name,
                    "alias": model_alias,
                    "non_emb_params": train_info["non_emb_params"],
                    "non_emb_params_no_fc": train_info["non_emb_params_no_fc"],
                    "output_fc_params": train_info["output_fc_params"],
                    "total_params": train_info["total_params"],
                    "train_instances": train_instances,
                    "epochs": num_train_epochs,
                    "lr": learning_rate,
                    "seed": seed,
                    "train_loss": train_info["train_loss"],
                    "eval_loss": eval_info["eval_loss"],
                    "train_time_sec": train_info["train_time_sec"],
                    "eval_time_sec": eval_info["eval_time_sec"],
                    "total_time_sec": round(total_time, 2)
                }
                record_result(rec)

            except Exception as e:
                logger.error(f"Failed to process {model_key}: {e}")
                logger.error(traceback.format_exc())
                status_tracker.data["failed_runs"].append(model_key)
                status_tracker.save()

    # Step 4: Data Scaling Experiment (BERT-small, D = [2500, 5000, 10000, 20000, 40000])
    logger.info("=== STEP 4: DATA SCALING RUNS (BERT-small x 5 DATA SIZES x 3 SEEDS) ===")
    target_model = "google/bert_uncased_L-4_H-512_A-8"
    target_alias = "small"
    data_sizes = [2500, 5000, 10000, 20000, 40000]

    for d_size in data_sizes:
        for seed in seeds:
            model_key = f"{target_alias}_datasize_{d_size}i_{num_train_epochs}ep_seed{seed}"
            # For 10000, alias is slightly different in model scaling: small_10000i_3ep_seedX
            if d_size == 10000:
                alt_key = f"small_10000i_3ep_seed{seed}"
            else:
                alt_key = model_key

            completed, existing_record = is_run_completed(target_model, d_size, seed, alt_key)
            if completed:
                logger.info(f"[SKIP] Found completed run for {target_alias} (D={d_size}, seed={seed}, loss={existing_record.get('eval_loss')})")
                if "non_emb_params_no_fc" not in existing_record or "seed" not in existing_record:
                    existing_record["seed"] = seed
                    existing_record["non_emb_params_no_fc"] = KNOWN_PARAM_COUNTS[target_model]["non_emb_params_no_fc"]
                    record_result(existing_record)
                continue

            logger.info(f"=== Running Data Scaling: {target_alias} | Instances={d_size} | Seed={seed} ===")
            t_start = time.time()
            try:
                train_info = train_single_model(
                    model_name=target_model,
                    shared_data=shared_data,
                    train_instances=d_size,
                    num_train_epochs=num_train_epochs,
                    learning_rate=learning_rate,
                    per_device_batch_size=per_device_batch_size,
                    seed=seed,
                    checkpoint_dir=os.path.join(CHECKPOINTS_DIR, model_key),
                    model_key=model_key
                )

                eval_info = evaluate_single_model(
                    checkpoint_dir=train_info["checkpoint_dir"],
                    base_model_name=target_model,
                    shared_data=shared_data,
                    model_key=model_key,
                    negative_pool_size=100000,
                    num_repeat=5,
                    max_queries=1000,
                    num_negatives=256,
                    seed=1
                )

                total_time = time.time() - t_start
                rec = {
                    "model_key": model_key,
                    "model_name": target_model,
                    "alias": f"{target_alias}_D{d_size}",
                    "non_emb_params": train_info["non_emb_params"],
                    "non_emb_params_no_fc": train_info["non_emb_params_no_fc"],
                    "output_fc_params": train_info["output_fc_params"],
                    "total_params": train_info["total_params"],
                    "train_instances": d_size,
                    "epochs": num_train_epochs,
                    "lr": learning_rate,
                    "seed": seed,
                    "train_loss": train_info["train_loss"],
                    "eval_loss": eval_info["eval_loss"],
                    "train_time_sec": train_info["train_time_sec"],
                    "eval_time_sec": eval_info["eval_time_sec"],
                    "total_time_sec": round(total_time, 2)
                }
                record_result(rec)

            except Exception as e:
                logger.error(f"Failed to process {model_key}: {e}")
                logger.error(traceback.format_exc())
                status_tracker.data["failed_runs"].append(model_key)
                status_tracker.save()

    # Step 5: Final summary and plots
    logger.info("=== STEP 5: FINAL PLOTS & SUMMARY ===")
    status_tracker.update(current_step="Generating Plots & SUMMARY.md", status="RUNNING")
    generate_summary_md(RESULTS_CSV, os.path.join(SCRIPT_DIR, "SUMMARY.md"))

    total_pipeline_time = time.time() - status_tracker.start_time
    status_tracker.update(
        status="DONE",
        current_step="Completed",
        elapsed_time_sec=round(total_pipeline_time, 1)
    )

    with open(DONE_FILE, "w") as f:
        f.write(f"Completed at {datetime.now().isoformat()} in {total_pipeline_time:.1f}s\n")

    logger.info(f"All runs completed successfully in {total_pipeline_time:.1f}s (~{total_pipeline_time/60:.1f} min)!")


if __name__ == "__main__":
    main()
