"""
Embed all KB articles with the fine-tuned bi-encoder and build a FAISS
index for inference-time retrieval.

Usage:
    python retriever/build_index.py --config config.yaml
"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    r_cfg = cfg["retriever"]
    processed_dir = Path(cfg["data"]["processed_dir"])

    kb_path = processed_dir / "kb_articles.jsonl"
    articles = [json.loads(l) for l in open(kb_path)]
    ids = [a["id"] for a in articles]
    texts = [a["text"] for a in articles]

    model = SentenceTransformer(r_cfg["output_dir"])
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim via normalized inner product
    index.add(embeddings.astype(np.float32))

    index_path = Path(r_cfg["faiss_index_path"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    # Save id mapping alongside the index
    with open(index_path.with_suffix(".ids.json"), "w") as f:
        json.dump(ids, f)

    print(f"Indexed {len(ids)} KB articles -> {index_path}")


if __name__ == "__main__":
    main()
