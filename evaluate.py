import logging
import pathlib, os
import random
import argparse
import pickle
from dataset import MSMARCO_Dataset
from modeling import DREncoder, AutoTokenizer, AutoConfig
import argparse
import numpy as np
from typing import Dict, List, Tuple

from tqdm import tqdm
import json
import torch

from dataset import tokenize
from transformers.trainer_utils import set_seed


def encode_negatives(model: DREncoder, dataset: MSMARCO_Dataset, negative_ids, device: torch.device):
    dim = getattr(model, "vector_dim", 768)
    all_embeddings = torch.zeros(len(negative_ids), dim, dtype=torch.float32, device=device)
    eval_batch_size = 64
    for start_idx in tqdm(range(0, len(negative_ids), 64), desc="Encoding negatives"):
        end_idx = min(start_idx + eval_batch_size, len(negative_ids))
        batch_neg = [dataset.corpus[pid] for pid in negative_ids[start_idx:end_idx]]
        batch_neg = tokenize(batch_neg, model.tokenizer, 180)

        with torch.no_grad():
            embeds = model.batch_encode(
                batch_neg['input_ids'].to(device),
                batch_neg['attention_mask'].to(device),
                8
            )
            all_embeddings[start_idx:end_idx] = embeds
    return all_embeddings


def contrastive_perplexity(model: DREncoder, dataset: MSMARCO_Dataset, negative_pool_size: int, num_repeat: int, device: torch.device, max_queries: int = 1000):
    num_negatives = dataset.num_negative_docs
    eval_batch_size = 64

    model.eval()

    queries = dataset.test_pairs['human'][0]
    qrels = dataset.test_pairs['human'][1]
    corpus = dataset.corpus
    tokenizer = model.tokenizer

    valid_qids = []
    for qid in queries.keys():
        if any(pid in corpus for pid in qrels.get(qid, {})):
            valid_qids.append(qid)

    # Deterministic sorting and subsampling
    valid_qids = sorted(valid_qids)
    if max_queries > 0 and len(valid_qids) > max_queries:
        valid_qids = valid_qids[:max_queries]

    maybe_positive_doc_ids = set()
    for qid in valid_qids:
        maybe_positive_doc_ids |= set(pid for pid in qrels[qid].keys() if pid in corpus)

    all_doc_ids = set(corpus.keys())
    negative_doc_ids = list(all_doc_ids - maybe_positive_doc_ids)
    # Ensure negative_pool_size does not exceed available negative docs
    pool_size = min(negative_pool_size, len(negative_doc_ids))
    sampled_negative_ids = random.sample(negative_doc_ids, pool_size)
    all_negative_embeddings = encode_negatives(
        model, dataset, sampled_negative_ids, device
    )
    assert len(valid_qids) > 0

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    contrast_entropy_list = []

    instances = [(qid, pid) for qid in valid_qids for pid in qrels[qid] if pid in corpus]

    # print info
    print(f"Total instances: {len(instances)}")
    print(f"Total valid queries: {len(valid_qids)}")
    print(f"Total negative docs: {len(negative_doc_ids)}")
    print(f"Total positive docs: {len(maybe_positive_doc_ids)}")
    print(f"Num sample_negatives: {num_negatives}")

    for start_idx in tqdm(range(0, len(instances), eval_batch_size), desc="Contrastive perplexity"):
        end_idx = min(start_idx + eval_batch_size, len(instances))
        batch_instances = instances[start_idx:end_idx]
        batch_qids = [x[0] for x in batch_instances]
        batch_pids = [x[1] for x in batch_instances]

        batch_queries = [queries[qid] for qid in batch_qids]
        batch_pos = [corpus[pid] for pid in batch_pids]

        batch_queries = tokenize(batch_queries, tokenizer, 32)
        batch_pos = tokenize(batch_pos, tokenizer, 180)

        with torch.no_grad():
            query_embes = model.batch_encode(
                batch_queries['input_ids'].to(device),
                batch_queries['attention_mask'].to(device),
                8
            )
            pos_embes = model.batch_encode(
                batch_pos['input_ids'].to(device),
                batch_pos['attention_mask'].to(device),
                8
            )
            for i in range(len(batch_qids)):
                query_emb = query_embes[i:i+1]  # (1, 768)
                pos_emb = pos_embes[i:i+1]      # (1, 768)

                positive_scores = (query_emb * pos_emb).sum(dim=-1) # (1, )
                sum_loss = 0
                for _ in range(num_repeat):
                    # randomly sample num_negatives rows from all_negative_embeddings
                    neg_embeds = all_negative_embeddings[random.sample(range(negative_pool_size), num_negatives)].to(device)
                    negative_scores = (query_emb * neg_embeds).sum(dim=-1) # (num_neg, )
                    scores = torch.cat([positive_scores, negative_scores], dim=-1).unsqueeze(0) # (1, num_neg+1)
                    labels = torch.zeros(1, dtype=torch.long).to(device) # (1, )
            
                    loss = loss_fn(scores, labels)
                    sum_loss += loss.mean().item()
                contrast_entropy_list.append(sum_loss/num_repeat)

    return np.mean(contrast_entropy_list), contrast_entropy_list, instances


def load_checkpoint_weights(model, model_path, device):
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    safe_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location=device)
    elif os.path.exists(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path, device=str(device))
    else:
        raise FileNotFoundError(f"Neither pytorch_model.bin nor model.safetensors found in {model_path}")
    info = model.load_state_dict(state_dict, strict=False)
    print("Loaded weights with info:", info)
    return info


def evaluate_model_checkpoint(
    checkpoint_dir,
    base_model_name,
    dataset,
    seed=1,
    negative_pool_size=100000,
    num_repeat=5,
    max_queries=1000,
    device="cuda",
    vector_dim=768
):
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    config = AutoConfig.from_pretrained(base_model_name)
    model = DREncoder(
        config=config,
        init_model=base_model_name,
        vector_dim=vector_dim,
        tokenizer=tokenizer
    ).to(device)

    load_checkpoint_weights(model, checkpoint_dir, device)
    set_seed(seed)
    loss, contrast_entropy_list, instances = contrastive_perplexity(
        model, dataset, negative_pool_size, num_repeat, device, max_queries=max_queries
    )
    return loss, contrast_entropy_list, instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_repeat", type=int, default=5)
    parser.add_argument("--negative_pool_size", type=int, default=100000)
    parser.add_argument("--max_queries", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vector_dim", type=int, default=768)
    args = parser.parse_args()

    init_model = args.init_model
    base_model = args.base_model if args.base_model is not None else init_model
    device = torch.device(args.device)

    dataset = MSMARCO_Dataset(
        num_instances=10,
        num_negative_docs=256
    )

    loss, contrast_entropy_list, instances = evaluate_model_checkpoint(
        checkpoint_dir=init_model,
        base_model_name=base_model,
        dataset=dataset,
        seed=args.seed,
        negative_pool_size=args.negative_pool_size,
        num_repeat=args.num_repeat,
        max_queries=args.max_queries,
        device=args.device,
        vector_dim=args.vector_dim
    )
    print("Evaluation loss:", loss)
    
    os.makedirs(pathlib.Path(args.output_path).parent, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({
            "loss": loss,
            "per_query_loss": contrast_entropy_list,
            "instances": instances,
            "init_model": init_model,
            "seed": args.seed
        }, f, indent=2)


if __name__ == "__main__":
    main()
