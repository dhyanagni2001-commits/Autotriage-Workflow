"""
Act Lambda: executes the decided action's tool calls (issue_refund,
update_crm, or escalate_to_human — see agent/tools.py) and updates the
ticket's DynamoDB record with the outcome.

On failure, this records an "act_failed" status *before* re-raising, so
the audit trail shows a partially-executed action even though the
exception still routes to EscalateOnFailure via the state machine's
Catch policy — a human reviewing the escalation can see, e.g., whether
issue_refund succeeded before update_crm blew up, instead of a bare
Step Functions error with no ticket-side record. This handler doesn't
have retries in the state machine (Act's Catch has no Retry block)
precisely because a retried refund isn't safe without an idempotency key
on issue_refund — that's a real TODO before this goes near production
traffic, called out in cloud/README.md.
"""
from datetime import datetime, timezone

from agent.nodes import act
from common import safe_put


def handler(event, context):
    try:
        result = act(event)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        safe_put({
            "ticket_id": event.get("ticket_id", "unknown"),
            "status": "act_failed",
            "error": str(exc),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        raise

    safe_put({
        "ticket_id": event.get("ticket_id", "unknown"),
        "status": result.get("action", "unknown"),
        "tool_calls": result.get("tool_calls", []),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    return result
