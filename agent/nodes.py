"""
Node functions for the AutoTriage LangGraph state machine.

retrieve() and classify() call the real fine-tuned artifacts
(retriever/build_index.py's FAISS index + bi-encoder, and
classifier/train_classifier.py's DistilBERT checkpoint) when they're
present on disk. Model loading is lazy and cached at module scope, and
the heavy ML imports (sentence_transformers, transformers, faiss) happen
inside the functions rather than at module import time — this keeps
`from agent.nodes import decide` cheap for callers (e.g. the `decide`
Lambda handler) that never need retrieval/classification at all.

If the trained artifacts aren't present yet (fresh clone, training still
running on Kaggle) or fail to load for any reason, each node falls back
to a small deterministic mock so the graph stays runnable end-to-end.
"""
import logging
import re
from pathlib import Path

import yaml

from agent.state import TicketState
from data.labels import category_for
from agent.tools import draft_reply, escalate_to_human, issue_refund, update_crm

logger = logging.getLogger(__name__)

# Resolve config.yaml relative to the repo root (parent of this file's
# package dir), not the current working directory — otherwise this only
# works when Python happens to be launched from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load(open(_REPO_ROOT / "config.yaml"))
AGENT_CFG = CONFIG["agent"]
RETRIEVER_CFG = CONFIG["retriever"]
CLASSIFIER_CFG = CONFIG["classifier"]

# Lazily-populated singletons. `None` means "not loaded yet"; the sentinel
# object `_UNAVAILABLE` means "we tried once and it isn't usable" so we
# don't retry a slow/failing load on every single ticket.
_UNAVAILABLE = object()
_retriever_cache = None
_classifier_cache = None


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def _load_retriever():
    global _retriever_cache
    if _retriever_cache is not None:
        return _retriever_cache

    index_path = _REPO_ROOT / RETRIEVER_CFG["faiss_index_path"]
    ids_path = index_path.with_suffix(".ids.json")
    model_dir = _REPO_ROOT / RETRIEVER_CFG["output_dir"]

    if not (index_path.exists() and ids_path.exists() and model_dir.exists()):
        logger.warning(
            "Fine-tuned retriever artifacts not found at %s / %s — "
            "falling back to mock retrieval. Run retriever/train_biencoder.py "
            "and retriever/build_index.py to enable real retrieval.",
            model_dir, index_path,
        )
        _retriever_cache = _UNAVAILABLE
        return _retriever_cache

    try:
        import json

        import faiss
        from sentence_transformers import SentenceTransformer

        index = faiss.read_index(str(index_path))
        ids = json.loads(ids_path.read_text())
        model = SentenceTransformer(str(model_dir))

        kb_path = _REPO_ROOT / CONFIG["data"]["processed_dir"] / "kb_articles.jsonl"
        kb_text_by_id = {}
        if kb_path.exists():
            with open(kb_path) as f:
                for line in f:
                    row = json.loads(line)
                    kb_text_by_id[row["id"]] = row["text"]

        _retriever_cache = {"model": model, "index": index, "ids": ids, "kb_text_by_id": kb_text_by_id}
    except Exception:
        logger.exception("Failed to load fine-tuned retriever — falling back to mock retrieval.")
        _retriever_cache = _UNAVAILABLE

    return _retriever_cache


def _mock_retrieve(ticket_text: str) -> list:
    return [{
        "id": "kb_refund",
        "text": "Duplicate charges are refunded automatically if reported within 30 days and under $50.",
        "score": 0.91,
    }]


def _real_retrieve(ticket_text: str, retriever: dict, top_k: int) -> list:
    import numpy as np

    query_emb = retriever["model"].encode(
        [ticket_text], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    k = min(top_k, retriever["index"].ntotal)
    if k == 0:
        return []

    scores, indices = retriever["index"].search(query_emb, k)
    articles = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:  # FAISS pads with -1 when fewer than k results exist
            continue
        article_id = retriever["ids"][idx]
        articles.append({
            "id": article_id,
            "text": retriever["kb_text_by_id"].get(article_id, ""),
            "score": float(score),
        })
    return articles


def retrieve(state: TicketState) -> TicketState:
    ticket_text = (state.get("ticket_text") or "").strip()
    if not ticket_text:
        logger.warning("retrieve() called with empty ticket_text; skipping retrieval.")
        return {**state, "retrieved_articles": []}

    top_k = AGENT_CFG.get("retrieval_top_k", 3)
    retriever = _load_retriever()

    if retriever is _UNAVAILABLE:
        articles = _mock_retrieve(ticket_text)
    else:
        try:
            articles = _real_retrieve(ticket_text, retriever, top_k)
        except Exception:
            logger.exception("Real retrieval failed for this ticket — falling back to mock retrieval.")
            articles = _mock_retrieve(ticket_text)

    return {**state, "retrieved_articles": articles}


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def _load_classifier():
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache

    model_dir = _REPO_ROOT / CLASSIFIER_CFG["output_dir"]
    if not model_dir.exists():
        logger.warning(
            "Fine-tuned classifier not found at %s — falling back to mock "
            "classification. Run classifier/train_classifier.py to enable "
            "real classification.",
            model_dir,
        )
        _classifier_cache = _UNAVAILABLE
        return _classifier_cache

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        _classifier_cache = {"tokenizer": tokenizer, "model": model}
    except Exception:
        logger.exception("Failed to load fine-tuned classifier — falling back to mock classification.")
        _classifier_cache = _UNAVAILABLE

    return _classifier_cache


# Categories heavy/urgent enough that a low-confidence mock guess should
# never silently resolve a ticket — used only by the mock fallback below.
_URGENT_KEYWORDS = ("angry", "furious", "cancel my account", "lawsuit", "legal action")


def _mock_classify(ticket_text: str) -> dict:
    text = ticket_text.lower()
    if any(kw in text for kw in _URGENT_KEYWORDS):
        return {"category": "urgent_escalate", "urgency": "high", "confidence": 0.88}
    if "charged twice" in text or "refund" in text:
        return {"category": "refund", "urgency": "medium", "confidence": 0.93}
    return {"category": "technical", "urgency": "low", "confidence": 0.6}


def _urgency_from_category(category: str) -> str:
    if category == "urgent_escalate":
        return "high"
    if category in ("refund", "billing"):
        return "medium"
    return "low"


def _real_classify(ticket_text: str, classifier: dict, max_seq_len: int) -> dict:
    import torch
    import torch.nn.functional as F

    tokenizer = classifier["tokenizer"]
    model = classifier["model"]

    enc = tokenizer(
        ticket_text, truncation=True, max_length=max_seq_len, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)[0].tolist()
    id2label = model.config.id2label

    # Routing acts on the coarse category, so confidence is marginalized
    # over all intents in that category: a ticket split 0.45/0.40 between
    # get_refund and track_refund is a confident *refund*, not an uncertain
    # one. The reported intent is the top intent inside the chosen category.
    category_conf = {}
    for i, p in enumerate(probs):
        cat = category_for(id2label[i])
        category_conf[cat] = category_conf.get(cat, 0.0) + p
    category = max(category_conf, key=category_conf.get)
    intent_id = max((i for i in range(len(probs)) if category_for(id2label[i]) == category),
                    key=lambda i: probs[i])

    return {
        "intent": id2label[intent_id],
        "intent_confidence": probs[intent_id],
        "category": category,
        "urgency": _urgency_from_category(category),
        "confidence": category_conf[category],
    }


def classify(state: TicketState) -> TicketState:
    ticket_text = (state.get("ticket_text") or "").strip()
    if not ticket_text:
        logger.warning("classify() called with empty ticket_text; forcing low-confidence escalate.")
        return {**state, "category": "unknown", "urgency": "low", "confidence": 0.0}

    classifier = _load_classifier()

    if classifier is _UNAVAILABLE:
        result = _mock_classify(ticket_text)
    else:
        try:
            result = _real_classify(ticket_text, classifier, CLASSIFIER_CFG["max_seq_len"])
        except Exception:
            logger.exception("Real classification failed for this ticket — falling back to mock classification.")
            result = _mock_classify(ticket_text)

    return {**state, **result}


# --------------------------------------------------------------------------
# Decide / Act
# --------------------------------------------------------------------------

def _coerce_amount(amount) -> float:
    """Return a non-negative float, or None if amount is missing/invalid.

    Ticket amounts can arrive as strings from API Gateway JSON, be
    negative due to upstream bugs, or be missing entirely — none of
    those should ever raise inside the decision logic.
    """
    if amount is None:
        return None
    try:
        value = float(amount)
    except (TypeError, ValueError):
        logger.warning("Could not parse amount_usd=%r as a number; treating as missing.", amount)
        return None
    if value < 0:
        logger.warning("amount_usd=%r is negative; treating as missing.", value)
        return None
    return value


def _risk_flags(ticket_text: str) -> list:
    """Config-driven patterns (legal threats, chargebacks, anger, prompt
    injection) that must never be auto-actioned regardless of what the
    classifier says. The classifier is trained on polite templated tickets
    and confidently labels "refund $45 now or I sue" as get_refund."""
    text = (ticket_text or "").lower()
    return [p for p in AGENT_CFG.get("risk_patterns", []) if re.search(p, text)]


def decide(state: TicketState) -> TicketState:
    trace = []
    category = state.get("category", "unknown")
    confidence = state.get("confidence", 0.0)
    amount = _coerce_amount(state.get("amount_usd"))
    flags = _risk_flags(state.get("ticket_text", ""))

    if flags:
        trace.append(f"risk guardrail matched {flags} -> escalate")
        action = "escalate"
    elif confidence < AGENT_CFG["confidence_threshold"]:
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
    reply = None
    if action in ("auto_resolve", "draft_hold"):
        reply = draft_reply(state.get("ticket_text", ""), context)

    return {**state, "action": action, "reply_draft": reply, "reasoning_trace": trace}


def act(state: TicketState) -> TicketState:
    tool_calls = []
    action = state.get("action")
    order_id = state.get("order_id") or "unknown"
    customer_id = state.get("customer_id") or "unknown"

    if action == "auto_resolve" and state.get("category") == "refund":
        amount = _coerce_amount(state.get("amount_usd")) or 0.0
        result = issue_refund(order_id=order_id, amount_usd=amount)
        tool_calls.append({"tool": "issue_refund", "output": result})
        crm_result = update_crm(customer_id=customer_id, note="auto-refunded via AutoTriage")
        tool_calls.append({"tool": "update_crm", "output": crm_result})
    elif action == "escalate":
        result = escalate_to_human(
            ticket_text=state.get("ticket_text", ""),
            reason="; ".join(state.get("reasoning_trace", [])),
        )
        tool_calls.append({"tool": "escalate_to_human", "output": result})
    # draft_hold: reply is drafted (state["reply_draft"]) but no tool call
    # is made — it waits in a queue for a human to approve/send.

    return {**state, "tool_calls": tool_calls}
