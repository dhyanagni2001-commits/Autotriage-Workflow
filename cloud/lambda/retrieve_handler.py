"""
Retrieve Lambda: looks up KB articles relevant to a ticket.

Two modes, chosen by whether RETRIEVER_ENDPOINT is set:

- **SageMaker endpoint** (production): invokes a real-time SageMaker
  endpoint hosting the fine-tuned bi-encoder + FAISS index. Keeps
  torch/sentence-transformers/faiss out of this Lambda's package, which
  matters because Lambda has a 250MB unzipped size limit.
- **In-process fallback** (local testing / no endpoint deployed yet):
  imports agent.nodes.retrieve, which loads the FAISS index and
  bi-encoder directly. Heavier, but lets the full pipeline run without
  any AWS infra — this is the path used for local dev and for the
  resume-project "code complete, not deployed" story.

Step Functions retries this on failure (see state_machine.json) and
routes to EscalateOnFailure if all retries are exhausted, so this
handler intentionally lets exceptions propagate rather than swallowing
them.
"""
import json
import os

import boto3

_runtime = None


def _sagemaker_retrieve(ticket_text: str, endpoint_name: str) -> list:
    global _runtime
    if _runtime is None:
        _runtime = boto3.client("sagemaker-runtime")
    response = _runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"ticket_text": ticket_text}),
    )
    payload = json.loads(response["Body"].read())
    return payload["articles"]


def _local_retrieve(ticket_text: str) -> list:
    from agent.nodes import retrieve
    return retrieve({"ticket_text": ticket_text}).get("retrieved_articles", [])


def handler(event, context):
    ticket_text = event.get("ticket_text", "")
    endpoint_name = os.environ.get("RETRIEVER_ENDPOINT")

    if endpoint_name:
        articles = _sagemaker_retrieve(ticket_text, endpoint_name)
    else:
        articles = _local_retrieve(ticket_text)

    return {**event, "retrieved_articles": articles}
