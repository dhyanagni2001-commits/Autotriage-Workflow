"""
Evaluate retrieval quality (Hit@k, MRR) on the held-out test set, comparing
the fine-tuned bi-encoder against a TF-IDF lexical baseline and the
off-the-shelf base model. With exactly one relevant article per ticket,
Hit@k (target in top k) is the right metric, not precision@k.
Writes results/retrieval.json.

Usage:
    python retriever/eval_retrieval.py --config config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


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
    queries = [ex["ticket_text"] for ex in test_examples]
    if model == "tfidf":
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(kb_texts + queries)
        sims = (vec.transform(queries) @ vec.transform(kb_texts).T).toarray()
        return _score(sims, kb_ids, test_examples, ks)
    kb_embeddings = model.encode(kb_texts, convert_to_numpy=True, normalize_embeddings=True)
    query_embeddings = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    return _score(query_embeddings @ kb_embeddings.T, kb_ids, test_examples, ks)


def _score(sims, kb_ids, test_examples, ks):
    hits_at_k = {k: 0 for k in ks}
    reciprocal_ranks = []
    max_k = len(kb_ids)
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
    metrics = {f"Hit@{k}": hits_at_k[k] / n for k in ks}
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

    results = {"n_test": len(test_examples), "n_articles": len(kb)}
    models = [
        ("tfidf (lexical)", "tfidf"),
        ("minilm (off-the-shelf)", SentenceTransformer(r_cfg["base_model"])),
        ("minilm (fine-tuned)", SentenceTransformer(r_cfg["output_dir"])),
    ]
    print(f"n_test={len(test_examples)}  kb_articles={len(kb)}")
    print(f"{'model':<26}" + "".join(f"{f'Hit@{k}':>8}" for k in r_cfg["eval_k"]) + f"{'MRR':>8}")
    for name, model in models:
        m = evaluate(model, test_examples, kb, r_cfg["eval_k"])
        results[name] = m
        print(f"{name:<26}" + "".join(f"{m[f'Hit@{k}']:>8.3f}" for k in r_cfg["eval_k"]) + f"{m['MRR']:>8.3f}")

    out = Path("results/retrieval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
