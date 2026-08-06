# Data

Source: [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(~27K labeled examples across intents like refund, invoice, cancel order,
delivery, account, etc). Good fit because it already has `instruction`
(the ticket text), `intent` (classification label), and `category`
(coarser grouping) — enough to derive both the classifier labels and
ticket→KB-article pairs for retrieval training.

## What `prepare_bitext.py` does

1. Downloads the dataset via `datasets.load_dataset`.
2. Maps the raw `intent` values down to the 6 labels in `config.yaml`
   (`billing`, `refund`, `technical`, `account`, `shipping`,
   `urgent_escalate`) — see `INTENT_MAP` in the script, edit as needed
   once you've looked at the real label distribution.
3. Builds a small synthetic "knowledge base": one canonical policy/help
   article per intent, generated from the dataset's `response` field
   (deduplicated + cleaned). This is what the retriever learns to fetch.
4. Builds retrieval training pairs: `(ticket_text, correct_kb_article_id,
   hard_negative_kb_article_ids)`.
5. Splits into train/val/test per `config.yaml` and writes JSONL to
   `data/processed/`.

## Usage

```bash
python data/prepare_bitext.py --config config.yaml
```

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
