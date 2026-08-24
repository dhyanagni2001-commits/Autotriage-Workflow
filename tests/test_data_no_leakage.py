"""
Regression test for the train/val/test leakage bug found after the first
Kaggle run: the raw Bitext dataset is templated and contains exact-duplicate
rows, so a naive random split let identical ticket text appear in both train
and test — the classifier and TF-IDF baseline were partly just memorizing
strings they'd already seen, which is why both hit ~99.9% before the dedup
fix in data/prepare_bitext.py::build_examples().

Requires the real processed data (`python data/prepare_bitext.py --config
config.yaml`, or the artifacts downloaded from the Kaggle training run) —
skips otherwise rather than failing, since this repo is meant to be usable
from a fresh clone with zero setup.
"""
import json
from pathlib import Path

import pytest
import yaml

PROCESSED_DIR = Path(yaml.safe_load(open("config.yaml"))["data"]["processed_dir"])

CLASSIFIER_SPLITS = ["classifier_train.jsonl", "classifier_val.jsonl", "classifier_test.jsonl"]
RETRIEVAL_SPLITS = ["retrieval_train.jsonl", "retrieval_val.jsonl", "retrieval_test.jsonl"]


def _require_processed_files(filenames):
    paths = [PROCESSED_DIR / f for f in filenames]
    if not all(p.exists() for p in paths):
        pytest.skip(f"{PROCESSED_DIR} not populated — run data/prepare_bitext.py first")
    return paths


def _load_texts(path, key="ticket_text"):
    return [json.loads(line)[key] for line in open(path)]


def _assert_disjoint(split_texts: dict):
    seen_in = {}
    leaks = []
    for split_name, texts in split_texts.items():
        for text in texts:
            prior = seen_in.get(text)
            if prior is not None and prior != split_name:
                leaks.append((text, prior, split_name))
            seen_in[text] = split_name
    assert not leaks, f"found {len(leaks)} leaked example(s), e.g. {leaks[:3]}"


def test_classifier_splits_have_no_overlapping_ticket_text():
    train_p, val_p, test_p = _require_processed_files(CLASSIFIER_SPLITS)
    _assert_disjoint({
        "train": _load_texts(train_p),
        "val": _load_texts(val_p),
        "test": _load_texts(test_p),
    })


def test_retrieval_splits_have_no_overlapping_ticket_text():
    train_p, val_p, test_p = _require_processed_files(RETRIEVAL_SPLITS)
    _assert_disjoint({
        "train": _load_texts(train_p),
        "val": _load_texts(val_p),
        "test": _load_texts(test_p),
    })


def test_classifier_splits_have_no_internal_duplicates():
    """A duplicate *within* a single split isn't leakage, but it does mean
    the reported test-set size (and per-class support) is inflated relative
    to the number of genuinely distinct examples being evaluated."""
    train_p, val_p, test_p = _require_processed_files(CLASSIFIER_SPLITS)
    for name, path in [("train", train_p), ("val", val_p), ("test", test_p)]:
        texts = _load_texts(path)
        n_dupes = len(texts) - len(set(texts))
        assert n_dupes == 0, f"{name} split has {n_dupes} internal duplicate ticket_text rows"
