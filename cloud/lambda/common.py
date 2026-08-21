"""
Shared helpers for the AutoTriage Lambda handlers. Deployed alongside
each handler file (flat, at the root of the Lambda package — "lambda" is
a Python keyword so `cloud/lambda/` can never be a dotted import path;
each function's zip bundles the contents of this directory plus `agent/`
and `config.yaml` from the repo root, not this package structure).
"""
import logging
import os
from decimal import Decimal
from typing import Any

import boto3

logger = logging.getLogger(__name__)

_dynamodb_resource = None


def get_table():
    global _dynamodb_resource
    if _dynamodb_resource is None:
        _dynamodb_resource = boto3.resource("dynamodb")
    table_name = os.environ.get("DYNAMODB_TABLE", "autotriage-tickets")
    return _dynamodb_resource.Table(table_name)


def to_dynamo_safe(obj: Any) -> Any:
    """DynamoDB's boto3 resource API rejects native Python floats —
    convert recursively via str() -> Decimal so refund amounts and
    confidence scores round-trip without float/Decimal type errors."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: to_dynamo_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_dynamo_safe(v) for v in obj]
    return obj


def safe_put(item: dict) -> bool:
    """Best-effort DynamoDB write. Never raises — a missing table, a
    throttled write, or a bad IAM policy shouldn't be the reason a
    ticket fails to process, especially in the audit-log step, which
    runs last and has nothing downstream to catch it."""
    try:
        get_table().put_item(Item=to_dynamo_safe(item))
        return True
    except Exception:
        logger.exception("DynamoDB put_item failed for ticket_id=%r", item.get("ticket_id"))
        return False
