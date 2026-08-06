"""
Fine-tune a sentence-transformers bi-encoder on (ticket, positive_article,
hard_negatives) triples using MultipleNegativesRankingLoss.

Usage:
    python retriever/train_biencoder.py --config config.yaml
"""
import argparse
import json
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def load_kb(path):
    kb = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            kb[row["id"]] = row["text"]
    return kb


def load_training_examples(retrieval_jsonl, kb):
    """Each row becomes one InputExample per hard negative: (ticket, positive)
    with MultipleNegativesRankingLoss using in-batch negatives, plus the
    explicit hard negatives appended as additional in-batch items."""
    examples = []
    with open(retrieval_jsonl) as f:
        for line in f:
            row = json.loads(line)
            pos_text = kb.get(row["positive_id"])
            if pos_text is None:
                continue
            examples.append(InputExample(texts=[row["ticket_text"], pos_text]))
            # Hard negatives: MultipleNegativesRankingLoss treats every other
            # example's positive in the batch as a negative, but we also
            # explicitly add hard-negative pairs to bias the batch composition.
            for neg_id in row["hard_negative_ids"]:
                neg_text = kb.get(neg_id)
                if neg_text:
                    examples.append(InputExample(texts=[row["ticket_text"], neg_text]))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    r_cfg = cfg["retriever"]
    data_cfg = cfg["data"]
    processed_dir = Path(data_cfg["processed_dir"])

    kb = load_kb(processed_dir / "kb_articles.jsonl")
    train_examples = load_training_examples(processed_dir / "retrieval_train.jsonl", kb)
    print(f"Loaded {len(train_examples)} training pairs")

    model = SentenceTransformer(r_cfg["base_model"])
    model.max_seq_length = r_cfg["max_seq_len"]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=r_cfg["batch_size"])
    train_loss = losses.MultipleNegativesRankingLoss(model)

    output_dir = Path(r_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=r_cfg["epochs"],
        warmup_steps=int(0.1 * len(train_dataloader) * r_cfg["epochs"]),
        optimizer_params={"lr": r_cfg["lr"]},
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    print(f"Saved fine-tuned model -> {output_dir}")


if __name__ == "__main__":
    main()
