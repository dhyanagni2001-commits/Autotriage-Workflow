# Classifier — fine-tuned DistilBERT ticket classifier

Predicts one of 27 fine-grained Bitext intents from raw ticket text. The
agent maps the intent to a routing category (`data/labels.py`) and uses
the probability mass summed over that category as its confidence.

## Files

- `train_classifier.py` — fine-tunes `distilbert-base-uncased` for
  multi-class classification using HuggingFace `Trainer`. Uses
  `Trainer`'s built-in per-epoch checkpointing (`save_strategy="epoch"`,
  `save_total_limit` from `config.yaml`) and auto-resumes from the
  latest `checkpoint-*` in `output_dir` if one exists — re-running the
  script after a Kaggle session timeout picks up where it left off
  instead of restarting. A `TRAINING_COMPLETE` marker file makes a
  re-run after a finished training a no-op (delete it to force
  retraining from scratch).
- `eval_classifier.py` — intent accuracy / macro-F1, routing-category
  accuracy, calibration (ECE), coverage and selective accuracy at the
  agent's confidence threshold, and CPU latency, for both the fine-tuned
  model and a TF-IDF + logistic regression baseline. Writes
  `results/classifier.json`.
- `baseline_tfidf.py` — the baseline classifier used for comparison.

## Usage

```bash
python classifier/train_classifier.py --config config.yaml
python classifier/eval_classifier.py --config config.yaml
```

## Running on Kaggle

Training runs on Kaggle (GPU notebook), not locally — see
[`notebooks/kaggle_train_autotriage.ipynb`](../notebooks/kaggle_train_autotriage.ipynb)
for the full pipeline (data prep -> retriever -> classifier -> both
evals) with resume-across-session-timeout built in. Copy the resulting
`classifier/checkpoints/classifier-v1/` back into this repo locally to
wire real classification into the agent (`agent/nodes.py` picks it up
automatically if present).

## Results (2,384 held-out tickets, 27 intents)

```
model                     intent acc  macro-F1  category acc    ECE   coverage@0.75
tfidf + logreg              0.9937     0.9936      0.9983     0.122      0.848
distilbert (fine-tuned)     0.9987     0.9987      0.9996     0.002      0.999
```

DistilBERT's real advantage over the baseline is calibration, not
accuracy: TF-IDF's probabilities are under-confident (ECE 0.12), so at
the agent's 0.75 threshold it would send 15% of correct tickets to a
human; DistilBERT sends 0.1%. CPU latency p50 11 ms / p95 13 ms per
ticket. 3 epochs, ~10 min on an Apple M-series GPU (MPS).

These are in-distribution numbers on a templated dataset. On hand-written
tickets the end-to-end picture is very different — see the top-level
README.
