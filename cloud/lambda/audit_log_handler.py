"""
WriteAuditLog Lambda: the terminal state in the pipeline (see
state_machine.json). Writes the full ticket state — classification,
retrieved context, decision + reasoning trace, and tool call outputs —
to DynamoDB as the audit trail this project is designed around (see
README.md "Logs every decision so the system is inspectable
end-to-end").

This must never be the reason a Step Functions execution fails: it's the
last state, nothing downstream can retry or catch for it, so a
DynamoDB outage here would otherwise turn a fully-processed ticket into
a failed execution. safe_put() already swallows write errors; this
handler additionally never raises, for the same reason.
"""
from datetime import datetime, timezone

from common import safe_put


def handler(event, context):
    item = {
        "ticket_id": event.get("ticket_id", "unknown"),
        "status": "audited",
        "ticket_text": event.get("ticket_text", ""),
        "category": event.get("category"),
        "urgency": event.get("urgency"),
        "confidence": event.get("confidence"),
        "retrieved_articles": event.get("retrieved_articles", []),
        "action": event.get("action"),
        "reply_draft": event.get("reply_draft"),
        "reasoning_trace": event.get("reasoning_trace", []),
        "tool_calls": event.get("tool_calls", []),
        "error": event.get("error"),
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    safe_put(item)
    return event
