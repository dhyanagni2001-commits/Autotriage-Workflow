"""
SageMaker inference entry point for the fine-tuned bi-encoder retriever.

Packaged as the `inference.py` in a SageMaker PyTorch model tarball
alongside: the checkpoint from retriever/train_biencoder.py, the FAISS
index from retriever/build_index.py, and data/processed/kb_articles.jsonl
(for looking up article text by id). See inference_classifier.py's
docstring — same "documented, not deployed" story.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

TOP_K = 3


def model_fn(model_dir):
    model_dir = Path(model_dir)
    model = SentenceTransformer(str(model_dir / "biencoder"))
    index = faiss.read_index(str(model_dir / "kb.index"))
    ids = json.loads((model_dir / "kb.index.ids.json").read_text())

    kb_text_by_id = {}
    kb_path = model_dir / "kb_articles.jsonl"
    if kb_path.exists():
        with open(kb_path) as f:
            for line in f:
                row = json.loads(line)
                kb_text_by_id[row["id"]] = row["text"]

    return {"model": model, "index": index, "ids": ids, "kb_text_by_id": kb_text_by_id}


def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    return json.loads(request_body)["ticket_text"]


def predict_fn(ticket_text, loaded):
    query_emb = loaded["model"].encode(
        [ticket_text], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    k = min(TOP_K, loaded["index"].ntotal)
    if k == 0:
        return []

    scores, indices = loaded["index"].search(query_emb, k)
    articles = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        article_id = loaded["ids"][idx]
        articles.append({
            "id": article_id,
            "text": loaded["kb_text_by_id"].get(article_id, ""),
            "score": float(score),
        })
    return articles


def output_fn(articles, accept="application/json"):
    return json.dumps({"articles": articles}), accept
