"""
Evaluate the fine-tuned DistilBERT classifier against the TF-IDF baseline
on the held-out test set.

Usage:
    python classifier/eval_classifier.py --config config.yaml
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classifier.baseline_tfidf import load_split, train_baseline


def eval_distilbert(model_dir, test_texts, test_labels, labels):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    label2id = {l: i for i, l in enumerate(labels)}
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            batch = test_texts[i:i + 32]
            enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=1).tolist())

    y_true = [label2id[l] for l in test_labels]
    return {
        "accuracy": accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average="macro"),
        "report": classification_report(y_true, preds, target_names=labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    c_cfg = cfg["classifier"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    labels = c_cfg["labels"]

    train_texts, train_labels = load_split(processed_dir / "classifier_train.jsonl")
    test_texts, test_labels = load_split(processed_dir / "classifier_test.jsonl")

    print("Training TF-IDF baseline...")
    baseline = train_baseline(processed_dir / "classifier_train.jsonl")
    base_preds = baseline.predict(test_texts)
    base_acc = accuracy_score(test_labels, base_preds)
    base_f1 = f1_score(test_labels, base_preds, average="macro")

    print("Evaluating fine-tuned DistilBERT...")
    ft_metrics = eval_distilbert(c_cfg["output_dir"], test_texts, test_labels, labels)

    print(f"\n{'model':<28}{'accuracy':>10}{'macro-F1':>10}")
    print(f"{'tfidf + logreg (baseline)':<28}{base_acc:>10.3f}{base_f1:>10.3f}")
    print(f"{'distilbert (fine-tuned)':<28}{ft_metrics['accuracy']:>10.3f}{ft_metrics['macro_f1']:>10.3f}")
    print("\nPer-class report (fine-tuned):")
    print(ft_metrics["report"])


if __name__ == "__main__":
    main()
