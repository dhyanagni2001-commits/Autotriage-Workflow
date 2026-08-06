"""
Evaluate retrieval quality (P@k, MRR) on the held-out test set, comparing
the fine-tuned bi-encoder against the off-the-shelf base model.

Usage:
    python retriever/eval_retrieval.py --config config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer, util


def load_kb(path):
    kb = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            kb[row["id"]] = row["text"]
    return kb


def evaluate(model, test_examples, kb, ks):
    kb_ids = list(kb.keys())
    kb_texts = [kb[i] for i in kb_ids]
    kb_embeddings = model.encode(kb_texts, convert_to_numpy=True, normalize_embeddings=True)

    hits_at_k = {k: 0 for k in ks}
    reciprocal_ranks = []

    queries = [ex["ticket_text"] for ex in test_examples]
    query_embeddings = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)

    max_k = max(ks)
    sims = query_embeddings @ kb_embeddings.T
    ranked = np.argsort(-sims, axis=1)

    for i, ex in enumerate(test_examples):
        target_id = ex["positive_id"]
        ranked_ids = [kb_ids[j] for j in ranked[i][:max_k]]
        rank = ranked_ids.index(target_id) + 1 if target_id in ranked_ids else None
        for k in ks:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    n = len(test_examples)
    metrics = {f"P@{k}": hits_at_k[k] / n for k in ks}
    metrics["MRR"] = float(np.mean(reciprocal_ranks))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    r_cfg = cfg["retriever"]
    processed_dir = Path(cfg["data"]["processed_dir"])

    kb = load_kb(processed_dir / "kb_articles.jsonl")
    test_examples = [json.loads(l) for l in open(processed_dir / "retrieval_test.jsonl")]

    print(f"{'model':<28}" + "".join(f"{f'P@{k}':>8}" for k in r_cfg["eval_k"]) + f"{'MRR':>8}")

    baseline = SentenceTransformer(r_cfg["base_model"])
    base_metrics = evaluate(baseline, test_examples, kb, r_cfg["eval_k"])
    row = f"{'baseline (off-the-shelf)':<28}"
    row += "".join(f"{base_metrics[f'P@{k}']:>8.2f}" for k in r_cfg["eval_k"])
    row += f"{base_metrics['MRR']:>8.2f}"
    print(row)

    finetuned = SentenceTransformer(r_cfg["output_dir"])
    ft_metrics = evaluate(finetuned, test_examples, kb, r_cfg["eval_k"])
    row = f"{'fine-tuned':<28}"
    row += "".join(f"{ft_metrics[f'P@{k}']:>8.2f}" for k in r_cfg["eval_k"])
    row += f"{ft_metrics['MRR']:>8.2f}"
    print(row)


if __name__ == "__main__":
    main()
