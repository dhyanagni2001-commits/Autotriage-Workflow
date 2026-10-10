# Data

Source: [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(~27K labeled examples across intents like refund, invoice, cancel order,
delivery, account, etc). Good fit because it already has `instruction`
(the ticket text), `intent` (classification label), and `category`
(coarser grouping) — enough to derive both the classifier labels and
ticket→KB-article pairs for retrieval training.

## What `prepare_bitext.py` does

1. Downloads the dataset via `datasets.load_dataset`.
2. Deduplicates on a *normalized* form of the ticket (lowercased,
   punctuation stripped, `{{placeholders}}` collapsed): 26,872 raw rows ->
   23,826 unique tickets.
3. Keeps the 27 fine-grained `intent` values as classifier labels
   (written to `data/processed/labels.json`). Each intent maps to one of
   6 routing categories (`billing`, `refund`, `technical`, `account`,
   `shipping`, `urgent_escalate`) via `INTENT_MAP` in `data/labels.py`.
4. Builds a 27-article knowledge base: one canonical article per intent,
   taken from the dataset's `response` field. This is what the retriever
   learns to fetch.
5. Builds retrieval triplets: `(ticket_text, correct_kb_article_id,
   hard_negative_kb_article_ids)`, with hard negatives drawn first from
   the same routing category (e.g. `track_refund` vs. `get_refund`).
6. Splits into train/val/test (19,060 / 2,382 / 2,384) per `config.yaml` and writes JSONL to
   `data/processed/`.

## Usage

```bash
python data/prepare_bitext.py --config config.yaml
```

This also runs as the first step of
[`notebooks/kaggle_train_autotriage.ipynb`](../notebooks/kaggle_train_autotriage.ipynb),
which skips it automatically if `data/processed/classifier_train.jsonl`
already exists (data prep is deterministic given the same seed, so
there's nothing to gain from re-running it on every training resume).

Outputs:

```
data/processed/
├── classifier_train.jsonl
├── classifier_val.jsonl
├── classifier_test.jsonl
├── retrieval_train.jsonl
├── retrieval_val.jsonl
├── retrieval_test.jsonl
└── kb_articles.jsonl
```

## TODO

- [ ] Inspect real `intent` value counts and finalize `INTENT_MAP`
- [ ] Decide if synthetic KB articles are good enough or if a real scraped
      help-center (e.g. a public SaaS product's docs) would make the
      retrieval eval more convincing for a resume writeup
- [ ] Add a `--dry-run` flag that prints label distribution without writing files
