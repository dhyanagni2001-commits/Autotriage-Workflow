"""Shared state passed between LangGraph nodes."""
from typing import Optional, TypedDict


class TicketState(TypedDict, total=False):
    ticket_id: Optional[str]
    ticket_text: str
    amount_usd: Optional[float]
    order_id: Optional[str]
    customer_id: Optional[str]

    # Populated by retrieve node
    retrieved_articles: list  # list of {"id": str, "text": str, "score": float}

    # Populated by classify node
    intent: str          # fine-grained Bitext intent (27 classes)
    category: str        # coarse routing category derived from intent
    urgency: str
    confidence: float    # category-level (summed over the category's intents)
    intent_confidence: float

    # Populated by decide node
    action: str  # "auto_resolve" | "draft_hold" | "escalate"
    reply_draft: Optional[str]
    reasoning_trace: list  # list of str, human-readable reasoning steps

    # Populated by act node
    tool_calls: list  # list of {"tool": str, "input": dict, "output": dict}
