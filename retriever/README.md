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
- `eval_retrieval.py` — computes Hit@k and MRR on the held-out retrieval
  test set for a TF-IDF baseline, the off-the-shelf base model
  (`all-MiniLM-L6-v2`), and the fine-tuned model, and prints a side-by-side comparison table.

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

## Results (2,384 held-out tickets, 27-article KB)

```
model                      Hit@1   Hit@3   Hit@5     MRR
tfidf (lexical)            0.490   0.727   0.796   0.630
minilm (off-the-shelf)     0.745   0.941   0.977   0.845
minilm (fine-tuned)        0.997   1.000   1.000   0.999
```

One epoch with MultipleNegativesRankingLoss on (ticket, article,
same-category hard negative) triplets, `NoDuplicatesDataLoader` so a
batch never contains the same article twice. Full numbers in
`results/retrieval.json`.

## TODO

- [x] Kaggle training notebook with epoch-level resume (see above)
- [ ] Run training, fill in real eval numbers above
- [ ] Decide on final `hard_negatives_per_pos` after checking train loss curve
