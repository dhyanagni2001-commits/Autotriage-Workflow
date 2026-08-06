"""
Node functions for the AutoTriage LangGraph state machine.

retrieve() and classify() currently call mock functions so the graph is
testable without the trained retriever/classifier artifacts present.
Swap MOCK_* for real calls once retriever/build_index.py and
classifier/train_classifier.py have been run.
"""
from pathlib import Path

import yaml

from agent.state import TicketState
from agent.tools import draft_reply, escalate_to_human, issue_refund, update_crm

# Resolve config.yaml relative to the repo root (parent of this file's
# package dir), not the current working directory — otherwise this only
# works when Python happens to be launched from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load(open(_REPO_ROOT / "config.yaml"))
AGENT_CFG = CONFIG["agent"]


def _mock_retrieve(ticket_text: str) -> list:
    # TODO: replace with real FAISS search using retriever/build_index.py output
    return [{"id": "kb_refund", "text": "Duplicate charges are refunded automatically if reported within 30 days and under $50.", "score": 0.91}]


def _mock_classify(ticket_text: str) -> dict:
    # TODO: replace with real inference from classifier/train_classifier.py output
    text = ticket_text.lower()
    if "charged twice" in text or "refund" in text:
        return {"category": "refund", "urgency": "medium", "confidence": 0.93}
    if "angry" in text or "furious" in text or "cancel my account" in text:
        return {"category": "urgent_escalate", "urgency": "high", "confidence": 0.88}
    return {"category": "technical", "urgency": "low", "confidence": 0.6}


def retrieve(state: TicketState) -> TicketState:
    articles = _mock_retrieve(state["ticket_text"])
    return {**state, "retrieved_articles": articles}


def classify(state: TicketState) -> TicketState:
    result = _mock_classify(state["ticket_text"])
    return {**state, **result}


def decide(state: TicketState) -> TicketState:
    trace = []
    category = state["category"]
    confidence = state["confidence"]
    amount = state.get("amount_usd")

    if confidence < AGENT_CFG["confidence_threshold"]:
        trace.append(f"confidence {confidence:.2f} below threshold {AGENT_CFG['confidence_threshold']} -> escalate")
        action = "escalate"
    elif category == "urgent_escalate":
        trace.append("category is urgent_escalate -> always escalate")
        action = "escalate"
    elif category == "refund" and amount is not None and amount <= AGENT_CFG["auto_approve_refund_threshold_usd"]:
        trace.append(f"refund of ${amount} <= threshold ${AGENT_CFG['auto_approve_refund_threshold_usd']} -> auto-resolve")
        action = "auto_resolve"
    elif category in ("billing", "shipping", "account"):
        trace.append(f"category {category} -> draft reply, hold for human review")
        action = "draft_hold"
    else:
        trace.append(f"category {category} did not match any auto-resolve rule -> escalate")
        action = "escalate"

    context = state["retrieved_articles"][0]["text"] if state.get("retrieved_articles") else ""
    reply = draft_reply(state["ticket_text"], context) if action in ("auto_resolve", "draft_hold") else None

    return {**state, "action": action, "reply_draft": reply, "reasoning_trace": trace}


def act(state: TicketState) -> TicketState:
    tool_calls = []
    action = state["action"]

    if action == "auto_resolve" and state["category"] == "refund":
        result = issue_refund(order_id="unknown", amount_usd=state.get("amount_usd", 0.0))
        tool_calls.append({"tool": "issue_refund", "output": result})
        crm_result = update_crm(customer_id="unknown", note="auto-refunded via AutoTriage")
        tool_calls.append({"tool": "update_crm", "output": crm_result})
    elif action == "escalate":
        result = escalate_to_human(
            ticket_text=state["ticket_text"],
            reason="; ".join(state["reasoning_trace"]),
        )
        tool_calls.append({"tool": "escalate_to_human", "output": result})
    # draft_hold: reply is drafted (state["reply_draft"]) but no tool call
    # is made — it waits in a queue for a human to approve/send.

    return {**state, "tool_calls": tool_calls}
