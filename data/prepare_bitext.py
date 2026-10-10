"""
Download the Bitext customer-support dataset and derive:
  - classifier train/val/test JSONL (ticket_text -> fine-grained intent label + coarse category)
  - a knowledge base with one article per fine-grained intent (27)
  - retrieval train/val/test JSONL (ticket_text -> correct KB article id
    + hard negative KB article ids)

Usage:
    python data/prepare_bitext.py --config config.yaml
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.labels import category_for  # noqa: E402

_PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    """Canonical form used for dedup: lowercase, placeholders collapsed,
    punctuation stripped, whitespace squashed. Catches near-duplicates that
    differ only in casing/punctuation/placeholder name, which exact-match
    dedup misses."""
    t = _PLACEHOLDER.sub(" slot ", text.lower())
    t = _NON_ALNUM.sub(" ", t)
    return " ".join(t.split())


def load_raw(dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"):
    print(f"Loading {dataset_name} ...")
    return load_dataset(dataset_name, split="train")


def inspect(ds):
    counts = defaultdict(int)
    for row in ds:
        counts[row["intent"]] += 1
    for intent, n in sorted(counts.items(), key=lambda x: -x[1]):
        mapped = category_for(intent)
        print(f"{intent:30s} n={n:6d}  -> {mapped}")


def build_kb_articles(ds):
    """One canonical article per fine-grained intent (27 total), using the
    longest agent response for that intent as the article body."""
    by_intent = defaultdict(list)
    for row in ds:
        by_intent[row["intent"]].append(row["response"])
    return [
        {"id": f"kb_{intent}", "intent": intent, "category": category_for(intent),
         "text": max(responses, key=len)}
        for intent, responses in sorted(by_intent.items())
    ]


def build_examples(ds):
    """One example per distinct *normalized* instruction. The raw dataset is
    templated (exact duplicates, plus near-duplicates differing only in
    case/punctuation/placeholder); without this, the same request lands in
    both train and test and inflates eval numbers."""
    seen = set()
    examples = []
    for row in ds:
        key = normalize(row["instruction"])
        if key in seen:
            continue
        seen.add(key)
        examples.append({
            "ticket_text": row["instruction"],
            "label": row["intent"],
            "category": category_for(row["intent"]),
        })
    print(f"dedup: {len(ds)} raw rows -> {len(examples)} unique normalized tickets")
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
    """Hard negatives are drawn first from articles in the *same coarse
    category* (e.g. track_refund vs. get_refund), which are the confusions
    that actually matter, then topped up from the rest of the KB."""
    rng = random.Random(seed)
    article_by_intent = {a["intent"]: a for a in articles}

    pairs = []
    for ex in examples:
        pos = article_by_intent.get(ex["label"])
        if pos is None:
            continue
        same = [a["id"] for a in articles if a["category"] == pos["category"] and a["id"] != pos["id"]]
        other = [a["id"] for a in articles if a["category"] != pos["category"]]
        rng.shuffle(same)
        rng.shuffle(other)
        pairs.append({
            "ticket_text": ex["ticket_text"],
            "positive_id": pos["id"],
            "hard_negative_ids": (same + other)[:hard_negatives_per_pos],
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

    ds = load_raw()

    if args.inspect:
        inspect(ds)
        return

    examples = build_examples(ds)
    articles = build_kb_articles(ds)

    processed_dir = Path(data_cfg["processed_dir"])
    write_jsonl(processed_dir / "kb_articles.jsonl", articles)
    labels = sorted({ex["label"] for ex in examples})
    (processed_dir / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"{len(labels)} intent labels -> {processed_dir / 'labels.json'}")

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
