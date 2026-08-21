"""
draft_reply is the pipeline's one LLM call (see agent/tools.py). These
tests only cover the no-API-key fallback path — the real LLM path needs
network + a paid API key and isn't something CI should depend on to pass.
"""
import os

import pytest

from agent import tools


@pytest.fixture(autouse=True)
def _no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tools._llm_cache = None
    yield
    tools._llm_cache = None


def test_draft_reply_falls_back_to_template_without_api_key():
    context = "Refunds are auto-approved under $50."
    reply = tools.draft_reply("I need a refund", context)
    assert reply
    assert context in reply


def test_draft_reply_handles_empty_ticket_text():
    reply = tools.draft_reply("", "some context")
    assert reply


def test_draft_reply_handles_empty_context():
    reply = tools.draft_reply("I need help", "")
    assert reply
