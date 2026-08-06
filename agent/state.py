"""Shared state passed between LangGraph nodes."""
from typing import Optional, TypedDict


class TicketState(TypedDict, total=False):
    ticket_text: str
    amount_usd: Optional[float]

    # Populated by retrieve node
    retrieved_articles: list  # list of {"id": str, "text": str, "score": float}

    # Populated by classify node
    category: str
    urgency: str
    confidence: float

    # Populated by decide node
    action: str  # "auto_resolve" | "draft_hold" | "escalate"
    reply_draft: Optional[str]
    reasoning_trace: list  # list of str, human-readable reasoning steps

    # Populated by act node
    tool_calls: list  # list of {"tool": str, "input": dict, "output": dict}
