"""
Classify Lambda: predicts ticket category/urgency/confidence.

Same SageMaker-endpoint-or-local-fallback pattern as retrieve_handler.py
— see that file's docstring for the reasoning. Lets exceptions propagate
so Step Functions' Retry/Catch policy on the Classify state handles
failures (retry up to 3x, then EscalateOnFailure).
"""
import json
import os

import boto3

_runtime = None


def _sagemaker_classify(ticket_text: str, endpoint_name: str) -> dict:
    global _runtime
    if _runtime is None:
        _runtime = boto3.client("sagemaker-runtime")
    response = _runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"ticket_text": ticket_text}),
    )
    return json.loads(response["Body"].read())


def _local_classify(ticket_text: str) -> dict:
    from agent.nodes import classify
    result = classify({"ticket_text": ticket_text})
    return {
        "category": result.get("category"),
        "urgency": result.get("urgency"),
        "confidence": result.get("confidence"),
    }


def handler(event, context):
    ticket_text = event.get("ticket_text", "")
    endpoint_name = os.environ.get("CLASSIFIER_ENDPOINT")

    if endpoint_name:
        result = _sagemaker_classify(ticket_text, endpoint_name)
    else:
        result = _local_classify(ticket_text)

    return {**event, **result}
