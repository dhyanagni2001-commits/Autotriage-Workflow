"""
TF-IDF + logistic regression baseline classifier, used as the comparison
point for the fine-tuned DistilBERT model.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def load_split(path):
    rows = [json.loads(l) for l in open(path)]
    texts = [r["ticket_text"] for r in rows]
    labels = [r["label"] for r in rows]
    return texts, labels


def train_baseline(train_path: Path):
    texts, labels = load_split(train_path)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(texts, labels)
    return pipeline
