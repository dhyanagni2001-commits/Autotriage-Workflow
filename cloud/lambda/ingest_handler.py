"""
Entry-point Lambda. Triggered by API Gateway (webhook) or an S3 event
(dropped file). Parses the incoming ticket, writes an initial record to
DynamoDB, and starts a Step Functions execution for the pipeline.

Deploy this behind API Gateway at e.g. POST /tickets.
"""
import json
import os
import uuid
from datetime import datetime, timezone

import boto3

dynamodb = boto3.resource("dynamodb")
sfn = boto3.client("stepfunctions")

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "autotriage-tickets")
STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN")


def parse_event(event: dict) -> dict:
    """Support both API Gateway proxy events and direct S3-triggered events."""
    if "body" in event:  # API Gateway
        body = json.loads(event["body"])
    else:  # direct invoke / S3 event already normalized upstream
        body = event
    return {
        "ticket_text": body["ticket_text"],
        "customer_id": body.get("customer_id", "unknown"),
        "order_id": body.get("order_id"),
        "amount_usd": body.get("amount_usd"),
    }


def handler(event, context):
    parsed = parse_event(event)
    ticket_id = str(uuid.uuid4())

    table = dynamodb.Table(TABLE_NAME)
    table.put_item(Item={
        "ticket_id": ticket_id,
        "status": "received",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **parsed,
    })

    sfn.start_execution(
        stateMachineArn=STATE_MACHINE_ARN,
        name=ticket_id,
        input=json.dumps({"ticket_id": ticket_id, **parsed}),
    )

    return {
        "statusCode": 202,
        "body": json.dumps({"ticket_id": ticket_id, "status": "processing"}),
    }
