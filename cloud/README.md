# Cloud deployment

Maps the local `retrieve -> classify -> decide -> act` pipeline onto AWS
Step Functions, with each stage as a Lambda. This is the "run RPA on the
cloud" piece — Step Functions gives retries, error branches, and a visual
execution history for free, which is what real RPA orchestration tools
provide.

```
API Gateway ─▶ Lambda: ingest_handler ─▶ Step Functions: autotriage-pipeline
                                            ├─ Retrieve   (Lambda)
                                            ├─ Classify   (Lambda)
                                            ├─ Decide     (Lambda, agent step)
                                            └─ Act        (Lambda, tool calls)
                                                 │
                                                 ▼
                                           DynamoDB: autotriage-tickets
                                           (ticket state + full audit trail)
```

## Files

- `step_functions/state_machine.json` — Amazon States Language definition
  for the pipeline, including a retry policy on the Classify/Retrieve
  steps and a `Catch` branch that routes failures to `escalate_to_human`
  instead of silently dropping the ticket.
- `lambda/ingest_handler.py` — entry point Lambda: parses the incoming
  ticket (from API Gateway or an S3 event), writes an initial DynamoDB
  record, and starts a Step Functions execution.
- `lambda/requirements.txt` — Lambda-specific deps (kept separate from
  the training-time `requirements.txt` since Lambda has package size
  limits — no torch/transformers here, inference should hit a SageMaker
  endpoint or a lightweight ONNX-exported model instead).

## Deploy (outline — fill in once you're ready to actually deploy)

```bash
# 1. Package and deploy the Lambdas (e.g. via AWS SAM or the CDK)
# 2. Create the DynamoDB table
aws dynamodb create-table \
  --table-name autotriage-tickets \
  --attribute-definitions AttributeName=ticket_id,AttributeType=S \
  --key-schema AttributeName=ticket_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# 3. Create the state machine
aws stepfunctions create-state-machine \
  --name autotriage-pipeline \
  --definition file://cloud/step_functions/state_machine.json \
  --role-arn <YOUR_STEP_FUNCTIONS_EXECUTION_ROLE_ARN>
```

## Open design question: where does the model inference run?

The retriever and classifier are too large/slow for a plain Lambda cold
start. Two options, pick one and document the tradeoff in the final
writeup:

1. **SageMaker real-time endpoint** — deploy both models behind a
   SageMaker endpoint, Lambda just calls it. Cleanest, costs money while
   idle.
2. **Lambda + container image with ONNX-exported models** — quantize/export
   both models to ONNX, bundle in a Lambda container image. Cheaper at
   low volume, more setup work.

## TODO

- [ ] Decide SageMaker vs. Lambda-container for inference (see above)
- [ ] Write the Retrieve/Classify/Decide/Act Lambda handlers (currently
      only `ingest_handler.py` exists)
- [ ] Add a CloudWatch dashboard: tickets processed, auto-resolve %,
      escalation rate, average latency per stage
- [ ] Write the SAM/CDK template instead of manual `aws` CLI deploy steps
