"""
Mocked tools the agent can call. Swap the bodies for real API calls
(Stripe refunds, a real CRM, an email/ticketing API) once you're ready to
integrate with a real system — the signatures are designed to map
directly onto Lambda handlers in cloud/lambda/.
"""
from typing import Optional


def issue_refund(order_id: str, amount_usd: float) -> dict:
    # TODO: replace with a real payment provider call (e.g. Stripe refunds API)
    return {
        "status": "refunded",
        "order_id": order_id,
        "amount_usd": amount_usd,
    }


def update_crm(customer_id: str, note: str) -> dict:
    # TODO: replace with a real CRM API call
    return {"status": "updated", "customer_id": customer_id, "note": note}


def draft_reply(ticket_text: str, context: str) -> str:
    # In the real graph this is produced by the LLM node using `context`
    # (retrieved KB text) — this stub is only for testing the graph
    # without an LLM call.
    return f"[draft reply based on: {context[:80]}...]"


def escalate_to_human(ticket_text: str, reason: str, queue: str = "default") -> dict:
    # TODO: replace with a real ticketing system call (e.g. create a Zendesk ticket)
    return {"status": "escalated", "queue": queue, "reason": reason}
