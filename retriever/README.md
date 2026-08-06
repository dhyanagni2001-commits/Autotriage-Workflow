# Retriever — fine-tuned bi-encoder

Same recipe as QueryLens (contrastive loss + hard negative mining on
USC CARC, A100 nodes) applied to a new domain: ticket text → correct
KB/policy article.

## Files

- `train_biencoder.py` — fine-tunes a `sentence-transformers` bi-encoder
  with `MultipleNegativesRankingLoss` using the hard negatives generated
  in `data/prepare_bitext.py`.
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

## Running on USC CARC

```bash
conda activate querylens   # or a new autotriage env, same Python 3.11 base
sbatch retriever/carc_train.sbatch     # TODO: add sbatch script, copy from QueryLens
```

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

- [ ] Copy/adapt CARC sbatch script from QueryLens repo
- [ ] Run training, fill in real eval numbers above
- [ ] Decide on final `hard_negatives_per_pos` after checking train loss curve
