"""
EscalateOnFailure Lambda: the state machine's catch-all failure branch
(see the "Catch" block on every other state in state_machine.json). Runs
when Retrieve, Classify, Decide, or Act raises after exhausting its
retries, so a broken model endpoint or a bad LLM response routes a
ticket to a human instead of the execution just failing silently.

The incoming event is whatever state the pipeline had reached, plus the
error Step Functions attached at "$.error" (ResultPath keeps the
original fields rather than replacing them) — ticket_text may be present
or missing depending on which stage failed, so everything here is
defensive about partial state.
"""
import json
from datetime import datetime, timezone

from agent.tools import escalate_to_human
from common import safe_put


def handler(event, context):
    error = event.get("error", {})
    result = escalate_to_human(
        ticket_text=event.get("ticket_text", ""),
        reason=f"pipeline failure: {json.dumps(error)}",
        queue="pipeline_failure",
    )

    safe_put({
        "ticket_id": event.get("ticket_id", "unknown"),
        "status": "escalated_failure",
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    return {**event, "action": "escalate", "tool_calls": [{"tool": "escalate_to_human", "output": result}]}
