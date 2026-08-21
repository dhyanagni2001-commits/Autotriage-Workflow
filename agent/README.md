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
- `nodes.py` — the individual node functions. `retrieve`/`classify` lazily
  load the real trained artifacts (FAISS index + bi-encoder,
  fine-tuned DistilBERT) on first call and cache them at module scope;
  if those artifacts aren't present yet — a fresh clone, or training
  still running on Kaggle — they fall back to a small deterministic mock
  instead of raising, and log a warning once. The same fallback triggers
  if a real load or a real inference call raises for any reason (a
  corrupted checkpoint, an OOM), so a bad artifact degrades the pipeline
  instead of taking it down.
- `tools.py` — the tools the agent can call: `issue_refund`, `update_crm`,
  `escalate_to_human` are mocked (swap for a real payment/CRM/ticketing
  API when you're ready). `draft_reply` is real — it calls an OpenAI
  chat model via `langchain-openai`, using `agent.llm_model` from
  `config.yaml` and `OPENAI_API_KEY` from the environment. No key set,
  or the call fails (network error, rate limit, timeout)? It falls back
  to a deterministic template so the graph never blocks on or crashes
  from the LLM being unavailable.
- `state.py` — the shared `TicketState` TypedDict passed between nodes
  (`ticket_id`, `ticket_text`, `amount_usd`, `order_id`, `customer_id` in;
  `retrieved_articles`, `category`/`urgency`/`confidence`,
  `action`/`reply_draft`/`reasoning_trace`, `tool_calls` populated along
  the way).

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

`amount_usd` is coerced through `_coerce_amount()` first: a missing,
non-numeric (e.g. a bad string from a JSON payload), or negative amount
is treated as missing rather than raising — which routes a
would-be-refund to escalation instead of crashing the decide step, since
"can't tell how much this refund is for" is itself a reason not to
auto-act.

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

Set `OPENAI_API_KEY` in the environment to get real LLM-drafted replies;
without it, `draft_reply` uses a template and everything else still
works. Real retrieval/classification requires the trained artifacts from
`retriever/build_index.py` and `classifier/train_classifier.py` to exist
on disk at the paths in `config.yaml` — see the top-level README's
Kaggle training workflow.

## Tests

```bash
pytest tests/test_agent_decide.py tests/test_tools_draft_reply.py -v
```

Covers every branch of `decide()` (confidence threshold, urgent-escalate,
refund threshold — including the boundary and a missing/invalid amount,
the three draft-hold categories, and an unrecognized category), `act()`'s
tool-call dispatch per action, and `draft_reply`'s no-API-key fallback.
Deliberately does not exercise `retrieve()`/`classify()` against real
checkpoints, since those may not exist in CI — their mock-fallback path
is exercised implicitly by `python -m agent.graph`.

## TODO

- [x] Wire in the real retriever and classifier (with mock fallback —
      see `nodes.py`)
- [x] Wire in a real LLM call for `draft_reply` (with template fallback)
- [x] Add unit tests for each branch of the decision logic
- [ ] Log every state transition to the audit trail format expected by
      `cloud/lambda/audit_log_handler.py` (currently only the final
      state is logged, not each intermediate node transition)
