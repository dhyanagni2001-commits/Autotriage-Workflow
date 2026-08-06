# Agent — LangGraph decision + tool-calling layer

The agent does **not** retrieve or classify — it consumes the structured
outputs of the fine-tuned retriever and classifier and decides what to do.
This separation is deliberate: it keeps the ML artifacts independently
testable and keeps the agent's job narrow (reason + act), which is easier
to evaluate and debug than one monolithic "do everything" agent.

## Files

- `graph.py` — defines the LangGraph state machine: `retrieve` →
  `classify` → `decide` → `act`. `retrieve` and `classify` call into the
  trained models directly (no LLM call); `decide` is the only LLM-backed
  node.
- `nodes.py` — the individual node functions.
- `tools.py` — mocked tools the agent can call: `issue_refund`,
  `update_crm`, `draft_reply`, `escalate_to_human`.
- `state.py` — the shared `TicketState` TypedDict passed between nodes.

## Decision logic (in `nodes.py::decide`)

```
if classifier.confidence < config.agent.confidence_threshold:
    -> escalate_to_human (low confidence, don't guess)
elif category == "urgent_escalate":
    -> escalate_to_human (never auto-act on this category)
elif category == "refund" and amount <= config.agent.auto_approve_refund_threshold_usd:
    -> issue_refund + draft_reply (auto-resolve)
elif category in ("billing", "shipping", "account"):
    -> draft_reply + hold for review (agent drafts, human sends)
else:
    -> escalate_to_human
```

This is intentionally simple and rule-shaped rather than "ask the LLM to
decide everything" — the LLM's job is to use the retrieved KB context to
generate the actual reply text, not to invent the escalation policy.

## Usage

```python
from agent.graph import build_graph

graph = build_graph()
result = graph.invoke({
    "ticket_text": "I was charged twice, order #48213, please refund $12.99",
    "amount_usd": 12.99,
})
print(result["action"], result["reasoning_trace"])
```

## TODO

- [ ] Wire in the real retriever (`retriever/build_index.py` output) and
      classifier (`classifier/train_classifier.py` output) inside
      `nodes.py` — currently stubbed with mock functions so the graph
      can be tested without trained models.
- [ ] Add unit tests for each branch of the decision logic in `tools.py`
- [ ] Log every state transition to the audit trail format expected by
      `cloud/lambda/ingest_handler.py`
