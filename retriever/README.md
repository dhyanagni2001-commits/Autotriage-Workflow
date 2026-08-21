# Retriever — fine-tuned bi-encoder

Same recipe as QueryLens (contrastive loss + hard negative mining on
USC CARC, A100 nodes) applied to a new domain: ticket text → correct
KB/policy article.

## Files

- `train_biencoder.py` — fine-tunes a `sentence-transformers` bi-encoder
  with `MultipleNegativesRankingLoss` using the hard negatives generated
  in `data/prepare_bitext.py`. Saves a full checkpoint after every epoch
  under `<output_dir>/checkpoints/epoch_N/` plus a `training_state.json`
  tracking completed epochs — re-running the script resumes from the
  last completed epoch instead of restarting from the base model. If
  `training_state.json` already shows all epochs done, it's a no-op
  (delete the file to force retraining from scratch).
- `build_index.py` — embeds all KB articles with the fine-tuned model and
  builds a FAISS index for retrieval at inference time.
- `eval_retrieval.py` — computes P@k and MRR on the held-out retrieval
  test set, for both the fine-tuned model and the off-the-shelf baseline
  (`all-MiniLM-L6-v2`), and prints a side-by-side comparison table.

## Usage

```bash
# 1. fine-tune
python retriever/train_biencoder.py --config config.yaml

# 2. build the FAISS index from the fine-tuned model + KB articles
python retriever/build_index.py --config config.yaml

# 3. evaluate fine-tuned vs. baseline
python retriever/eval_retrieval.py --config config.yaml
```

## Running on Kaggle

Training runs on Kaggle (GPU notebook), not locally or on CARC — see
[`notebooks/kaggle_train_autotriage.ipynb`](../notebooks/kaggle_train_autotriage.ipynb)
for the full pipeline (data prep -> retriever -> classifier -> both
evals) with epoch-level resume across session timeouts. Copy the
resulting `retriever/checkpoints/biencoder-v1/` and
`retriever/faiss_index/` back into this repo locally to wire real
retrieval into the agent (`agent/nodes.py` picks them up automatically
if present).

## Expected output of eval_retrieval.py

```
model                    P@1    P@5    P@10    MRR
all-MiniLM-L6-v2 (base)  0.41   0.68   0.79    0.53
biencoder-v1 (fine-tuned) 0.XX   0.XX   0.XX   0.XX   <- fill in after training
```

This is the number that goes on the resume: "improved retrieval P@10 from
X to Y by fine-tuning a bi-encoder with hard-negative mining, mirroring
the approach used in QueryLens."

## TODO

- [x] Kaggle training notebook with epoch-level resume (see above)
- [ ] Run training, fill in real eval numbers above
- [ ] Decide on final `hard_negatives_per_pos` after checking train loss curve
