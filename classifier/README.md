# Classifier — fine-tuned DistilBERT ticket classifier

Predicts ticket category (`billing`, `refund`, `technical`, `account`,
`shipping`, `urgent_escalate`) from raw ticket text. Same pattern as the
DistilBERT usage in OmniSort AI.

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
- `eval_classifier.py` — computes accuracy / macro-F1 on the test set for
  both the fine-tuned model and a TF-IDF + logistic regression baseline.
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

## Expected output of eval_classifier.py

```
model                         accuracy   macro-F1
tfidf + logreg (baseline)     0.XX       0.XX
distilbert (fine-tuned)       0.XX       0.XX
```

Resume bullet target: "fine-tuned DistilBERT for ticket triage,
improving macro-F1 from X (TF-IDF baseline) to Y."

## TODO

- [x] Kaggle training notebook with resume-across-timeout (see above)
- [ ] Train, fill in real numbers above
- [ ] Check per-class F1 — `urgent_escalate` is the safety-critical class,
      worth reporting its recall specifically even if overall accuracy is high
- [ ] Consider adding a confidence-threshold analysis: what % of test
      tickets fall below `agent.confidence_threshold` in config.yaml, and
      how does classifier accuracy differ above/below that threshold
