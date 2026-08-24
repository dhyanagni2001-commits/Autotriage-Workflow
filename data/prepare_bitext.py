"""
Download the Bitext customer-support dataset and derive:
  - classifier train/val/test JSONL (ticket_text -> label)
  - a small synthetic knowledge base (one article per intent)
  - retrieval train/val/test JSONL (ticket_text -> correct KB article id
    + hard negative KB article ids)

Usage:
    python data/prepare_bitext.py --config config.yaml
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset

# Coarse label mapping. Edit after inspecting real intent value counts —
# run this script with --inspect first.
INTENT_MAP = {
    "get_refund": "refund",
    "track_refund": "refund",
    "check_refund_policy": "refund",
    "cancel_order": "billing",
    "change_order": "billing",
    "payment_issue": "billing",
    "check_invoice": "billing",
    "check_payment_methods": "billing",
    "delivery_options": "shipping",
    "delivery_period": "shipping",
    "track_order": "shipping",
    "place_order": "shipping",
    "create_account": "account",
    "delete_account": "account",
    "edit_account": "account",
    "recover_password": "account",
    "switch_account": "account",
    "registration_problems": "account",
    "contact_customer_service": "technical",
    "contact_human_agent": "urgent_escalate",
    "complaint": "urgent_escalate",
    "review": "technical",
    "newsletter_subscription": "technical",
}
DEFAULT_LABEL = "technical"


def load_raw(dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"):
    print(f"Loading {dataset_name} ...")
    return load_dataset(dataset_name, split="train")


def inspect(ds):
    counts = defaultdict(int)
    for row in ds:
        counts[row["intent"]] += 1
    for intent, n in sorted(counts.items(), key=lambda x: -x[1]):
        mapped = INTENT_MAP.get(intent, DEFAULT_LABEL)
        print(f"{intent:30s} n={n:6d}  -> {mapped}")


def build_kb_articles(ds, labels):
    """One canonical article per label, built from a representative
    response for that label. Deduplicate near-identical responses."""
    by_label = defaultdict(list)
    for row in ds:
        label = INTENT_MAP.get(row["intent"], DEFAULT_LABEL)
        by_label[label].append(row["response"])

    articles = []
    for label in labels:
        responses = by_label.get(label, [])
        if not responses:
            continue
        # naive: take the longest response as the "canonical" article text
        text = max(responses, key=len)
        articles.append({"id": f"kb_{label}", "label": label, "text": text})
    return articles


def build_examples(ds):
    """One example per *distinct* instruction string. The raw dataset is
    templated (many rows are exact-duplicate text, sometimes with unfilled
    `{{...}}` placeholders) — without this dedup, identical strings land in
    both train and test after the random split below, and the classifier/
    retriever partly just memorize them instead of generalizing, inflating
    eval numbers."""
    seen = set()
    examples = []
    for row in ds:
        text = row["instruction"]
        if text in seen:
            continue
        seen.add(text)
        label = INTENT_MAP.get(row["intent"], DEFAULT_LABEL)
        examples.append({
            "ticket_text": text,
            "label": label,
            "intent_raw": row["intent"],
        })
    return examples


def split(items, train_frac, val_frac, seed):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows):6d} rows -> {path}")


def build_retrieval_pairs(examples, articles, hard_negatives_per_pos, seed):
    rng = random.Random(seed)
    article_by_label = {a["label"]: a for a in articles}
    all_ids = [a["id"] for a in articles]

    pairs = []
    for ex in examples:
        pos = article_by_label.get(ex["label"])
        if pos is None:
            continue
        negative_pool = [i for i in all_ids if i != pos["id"]]
        k = min(hard_negatives_per_pos, len(negative_pool))
        hard_negs = rng.sample(negative_pool, k) if k > 0 else []
        pairs.append({
            "ticket_text": ex["ticket_text"],
            "positive_id": pos["id"],
            "hard_negative_ids": hard_negs,
        })
    return pairs


def assert_no_split_leakage(splits: dict, key: str = "ticket_text"):
    """Fail fast if the same text ends up in more than one split. This is
    the exact bug that previously inflated eval numbers (classifier hit
    99.9%, even the TF-IDF baseline hit 99.5%) — the raw dataset is
    templated and contains exact-duplicate rows, and a naive random split
    let identical text leak from train into val/test. build_examples()
    dedups up front so this should never trigger again; this assertion
    exists to catch a regression rather than to fix anything itself."""
    seen_in = {}
    for split_name, rows in splits.items():
        for row in rows:
            text = row[key]
            prior = seen_in.get(text)
            if prior is not None and prior != split_name:
                raise AssertionError(
                    f"data leakage: {text!r} appears in both {prior!r} and "
                    f"{split_name!r} splits"
                )
            seen_in[text] = split_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--inspect", action="store_true",
                         help="Print raw intent counts and mapped labels, then exit.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_cfg = cfg["data"]
    labels = cfg["classifier"]["labels"]

    ds = load_raw()

    if args.inspect:
        inspect(ds)
        return

    examples = build_examples(ds)
    articles = build_kb_articles(ds, labels)

    processed_dir = Path(data_cfg["processed_dir"])
    write_jsonl(processed_dir / "kb_articles.jsonl", articles)

    train, val, test = split(
        examples, data_cfg["train_split"], data_cfg["val_split"], data_cfg["seed"]
    )
    assert_no_split_leakage({"train": train, "val": val, "test": test})
    write_jsonl(processed_dir / "classifier_train.jsonl", train)
    write_jsonl(processed_dir / "classifier_val.jsonl", val)
    write_jsonl(processed_dir / "classifier_test.jsonl", test)

    hard_neg_k = cfg["retriever"]["hard_negatives_per_pos"]
    retrieval_splits = {}
    for name, split_examples in [("train", train), ("val", val), ("test", test)]:
        pairs = build_retrieval_pairs(split_examples, articles, hard_neg_k, data_cfg["seed"])
        retrieval_splits[name] = pairs
        write_jsonl(processed_dir / f"retrieval_{name}.jsonl", pairs)
    assert_no_split_leakage(retrieval_splits)


if __name__ == "__main__":
    main()
