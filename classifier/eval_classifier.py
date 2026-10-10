"""
Evaluate the fine-tuned DistilBERT intent classifier against the TF-IDF
baseline on the held-out test set. Reports:
  - 27-way intent accuracy / macro-F1
  - routing-category accuracy (intent mapped to the coarse category the
    agent's rules actually act on)
  - expected calibration error (ECE)
  - selective accuracy / coverage at the agent's confidence threshold
  - single-ticket CPU latency
Writes everything to results/classifier.json.

Usage:
    python classifier/eval_classifier.py --config config.yaml
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Run directly as `python classifier/eval_classifier.py`, this file's own
# directory (not the repo root) ends up as sys.path[0]; resolve siblings by
# path so this also works on Kaggle without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baseline_tfidf import load_split, train_baseline  # noqa: E402
from data.labels import category_for  # noqa: E402


def ece(confidences, correct, n_bins=15):
    confidences, correct = np.asarray(confidences), np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(total)


def summarize(y_true, y_pred, conf, threshold):
    correct = np.array([t == p for t, p in zip(y_true, y_pred)])
    cat_true = [category_for(t) for t in y_true]
    cat_pred = [category_for(p) for p in y_pred]
    conf = np.asarray(conf)
    kept = conf >= threshold
    return {
        "intent_accuracy": accuracy_score(y_true, y_pred),
        "intent_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "category_accuracy": accuracy_score(cat_true, cat_pred),
        "ece": ece(conf, correct),
        "coverage_at_threshold": float(kept.mean()),
        "selective_accuracy_at_threshold": float(correct[kept].mean()) if kept.any() else None,
        "error_caught_by_threshold": float((~kept[~correct]).mean()) if (~correct).any() else None,
    }


def predict_distilbert(model_dir, texts, batch_size=64):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], truncation=True, padding=True, return_tensors="pt").to(device)
            probs.append(torch.softmax(model(**enc).logits, -1).cpu().numpy())
    probs = np.concatenate(probs)
    preds = [model.config.id2label[int(i)] for i in probs.argmax(1)]

    # Latency as the agent sees it: one ticket at a time, CPU.
    model = model.to("cpu")
    lat = []
    with torch.no_grad():
        for t in texts[:200]:
            s = time.perf_counter()
            model(**tokenizer(t, truncation=True, return_tensors="pt"))
            lat.append((time.perf_counter() - s) * 1000)
    return preds, probs.max(1), {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="results/classifier.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    c_cfg = cfg["classifier"]
    threshold = cfg["agent"]["confidence_threshold"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    test_texts, test_labels = load_split(processed_dir / "classifier_test.jsonl")

    print("Training TF-IDF baseline...")
    baseline = train_baseline(processed_dir / "classifier_train.jsonl")
    base_probs = baseline.predict_proba(test_texts)
    base_preds = list(baseline.classes_[base_probs.argmax(1)])
    base = summarize(test_labels, base_preds, base_probs.max(1), threshold)

    print("Evaluating fine-tuned DistilBERT...")
    ft_preds, ft_conf, latency = predict_distilbert(c_cfg["output_dir"], test_texts)
    ft = summarize(test_labels, ft_preds, ft_conf, threshold)
    ft["cpu_latency"] = latency

    print(f"\nn_test={len(test_texts)}  intents={len(set(test_labels))}  threshold={threshold}")
    cols = ["intent_accuracy", "intent_macro_f1", "category_accuracy", "ece",
            "coverage_at_threshold", "selective_accuracy_at_threshold"]
    print(f"{'model':<26}" + "".join(f"{c[:14]:>16}" for c in cols))
    for name, m in [("tfidf + logreg", base), ("distilbert (fine-tuned)", ft)]:
        print(f"{name:<26}" + "".join(f"{m[c]:>16.4f}" for c in cols))
    print(f"distilbert CPU latency: p50={latency['p50_ms']:.1f}ms p95={latency['p95_ms']:.1f}ms")
    print("\nPer-intent report (fine-tuned):")
    print(classification_report(test_labels, ft_preds, digits=3))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n_test": len(test_texts), "threshold": threshold, "tfidf_logreg": base, "distilbert": ft}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
