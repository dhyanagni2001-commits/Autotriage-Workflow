"""
SageMaker inference entry point for the fine-tuned DistilBERT classifier.

Packaged as the `inference.py` in a SageMaker HuggingFace/PyTorch model
tarball alongside the checkpoint produced by
classifier/train_classifier.py. Implements the four functions the
SageMaker PyTorch inference toolkit looks for by name.

Not invoked by anything in this repo directly — classify_handler.py
calls the *endpoint* this becomes (via sagemaker-runtime.invoke_endpoint)
once it's deployed with deploy_endpoints.py. Kept here, uploaded, and
documented rather than actually deployed, since a real-time SageMaker
endpoint bills per hour it's up — see cloud/README.md.
"""
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    return payload["ticket_text"]


def predict_fn(ticket_text, loaded):
    tokenizer, model = loaded["tokenizer"], loaded["model"]
    enc = tokenizer(ticket_text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    intent = model.config.id2label[pred_id]
    # train_classifier.py stores the intent -> routing-category map in the
    # model config so this tarball doesn't need the repo's data/labels.py.
    intent_to_category = getattr(model.config, "intent_to_category", {}) or {}
    return {
        "intent": intent,
        "category": intent_to_category.get(intent, "technical"),
        "confidence": float(probs[pred_id].item()),
    }


def output_fn(prediction, accept="application/json"):
    urgency = "high" if prediction["category"] == "urgent_escalate" else (
        "medium" if prediction["category"] in ("refund", "billing") else "low"
    )
    return json.dumps({**prediction, "urgency": urgency}), accept
