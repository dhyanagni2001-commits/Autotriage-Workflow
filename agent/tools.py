"""
Tools the agent can call.

issue_refund / update_crm / escalate_to_human are deliberately mocked —
swap the bodies for real API calls (Stripe refunds, a real CRM, an
email/ticketing API) once you're ready to integrate with a real system.
The signatures are designed to map directly onto the cloud/lambda/
handlers.

draft_reply is the one real LLM call in the whole pipeline (see
agent/README.md — "decide is the only LLM-backed node"). It uses
langchain-openai, configured from config.yaml (agent.llm_model, etc.)
and OPENAI_API_KEY from the environment. If no key is configured, or the
API call fails for any reason (network error, rate limit, timeout), it
falls back to a deterministic template so the graph never crashes or
blocks on the LLM being unavailable.
"""
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = yaml.safe_load(open(_REPO_ROOT / "config.yaml"))
_AGENT_CFG = _CONFIG["agent"]

_UNAVAILABLE = object()
_llm_cache = None


def _fallback_draft(ticket_text: str, context: str) -> str:
    snippet = context[:200].strip() if context else "our support policy"
    return (
        "Hi,\n\nThanks for reaching out. Based on our records: "
        f"{snippet}\n\nWe'll follow up shortly with next steps.\n\n— AutoTriage"
    )


def _load_llm():
    """Lazily build the ChatOpenAI client. Returns _UNAVAILABLE if no API
    key is configured or the langchain-openai package isn't importable,
    so callers can fall back to the deterministic template without ever
    raising."""
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    if _AGENT_CFG.get("llm_provider") != "openai":
        logger.warning("Unsupported llm_provider=%r in config.yaml; falling back to template replies.",
                        _AGENT_CFG.get("llm_provider"))
        _llm_cache = _UNAVAILABLE
        return _llm_cache

    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set — draft_reply will use a template fallback instead of the LLM.")
        _llm_cache = _UNAVAILABLE
        return _llm_cache

    try:
        from langchain_openai import ChatOpenAI

        _llm_cache = ChatOpenAI(
            model=_AGENT_CFG["llm_model"],
            temperature=_AGENT_CFG.get("llm_temperature", 0.3),
            max_tokens=_AGENT_CFG.get("llm_max_tokens", 300),
            timeout=_AGENT_CFG.get("llm_timeout_s", 15),
        )
    except Exception:
        logger.exception("Failed to initialize ChatOpenAI — falling back to template replies.")
        _llm_cache = _UNAVAILABLE

    return _llm_cache


_SYSTEM_PROMPT = (
    "You are a customer support agent. Write a short, warm, professional "
    "reply to the customer's ticket using ONLY the provided policy/KB "
    "context to justify any claims. Do not invent policies, order "
    "details, or amounts that aren't in the ticket or context. Keep it "
    "under 120 words."
)


def draft_reply(ticket_text: str, context: str) -> str:
    ticket_text = (ticket_text or "").strip()
    if not ticket_text:
        return _fallback_draft(ticket_text, context)

    llm = _load_llm()
    if llm is _UNAVAILABLE:
        return _fallback_draft(ticket_text, context)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Ticket:\n{ticket_text}\n\nPolicy/KB context:\n{context or '(none found)'}"),
        ]
        response = llm.invoke(messages)
        text = (response.content or "").strip()
        return text if text else _fallback_draft(ticket_text, context)
    except Exception:
        logger.exception("LLM draft_reply call failed — falling back to template reply.")
        return _fallback_draft(ticket_text, context)


def issue_refund(order_id: str, amount_usd: float) -> dict:
    # TODO: replace with a real payment provider call (e.g. Stripe refunds API)
    if amount_usd is None or amount_usd < 0:
        return {"status": "error", "reason": f"invalid amount_usd={amount_usd!r}"}
    return {
        "status": "refunded",
        "order_id": order_id or "unknown",
        "amount_usd": amount_usd,
    }


def update_crm(customer_id: str, note: str) -> dict:
    # TODO: replace with a real CRM API call
    return {"status": "updated", "customer_id": customer_id or "unknown", "note": note}


def escalate_to_human(ticket_text: str, reason: str, queue: str = "default") -> dict:
    # TODO: replace with a real ticketing system call (e.g. create a Zendesk ticket)
    return {"status": "escalated", "queue": queue, "reason": reason or "unspecified"}
