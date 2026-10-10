import json
from pathlib import Path

import pytest

from data.labels import CATEGORIES, INTENT_MAP, category_for
from data.prepare_bitext import normalize


def test_every_mapped_category_is_known():
    assert set(INTENT_MAP.values()) <= set(CATEGORIES)


def test_unknown_intent_falls_back_to_technical():
    assert category_for("something_new") == "technical"


def test_escalation_intents_route_to_urgent():
    assert category_for("complaint") == category_for("contact_human_agent") == "urgent_escalate"


def test_all_dataset_intents_are_mapped():
    labels_path = Path("data/processed/labels.json")
    if not labels_path.exists():
        pytest.skip("run data/prepare_bitext.py first")
    assert set(json.loads(labels_path.read_text())) <= set(INTENT_MAP)


def test_normalize_collapses_near_duplicates():
    a = "I need to cancel order {{Order Number}}!"
    b = "i need to cancel   order {{Order ID}}"
    assert normalize(a) == normalize(b)
