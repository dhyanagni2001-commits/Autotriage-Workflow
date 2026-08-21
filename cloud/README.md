# Cloud deployment

Maps the local `retrieve -> classify -> decide -> act` pipeline onto AWS
Step Functions, with each stage as a Lambda. This is the "run RPA on the
cloud" piece — Step Functions gives retries, error branches, and a visual
execution history for free, which is what real RPA orchestration tools
provide.

```
API Gateway ─▶ Lambda: ingest_handler ─▶ Step Functions: autotriage-pipeline
                                            ├─ Retrieve   (Lambda -> SageMaker endpoint)
                                            ├─ Classify   (Lambda -> SageMaker endpoint)
                                            ├─ Decide     (Lambda, agent step + 1 LLM call)
                                            ├─ Act        (Lambda, tool calls)
                                            └─ EscalateOnFailure (Lambda, Catch branch)
                                                 │
                                                 ▼
                                           WriteAuditLog (Lambda)
                                                 │
                                                 ▼
                                           DynamoDB: autotriage-tickets
                                           (ticket state + full audit trail)
```

**Status: code-complete, intentionally not deployed.** This is a
portfolio project, not a service that needs to actually be up — running
`sam deploy` would start real hourly billing (SageMaker endpoints most of
all) for something with no traffic. Everything below is written and
internally consistent (the Lambda handlers, the state machine, the SAM
template) but has never been applied against a real AWS account. If you
do want to deploy it, read this whole file first — the cost note at the
bottom especially.

## Files

- `step_functions/state_machine.json` — Amazon States Language definition
  for the pipeline: retries on Retrieve/Classify (3x) and Decide (2x, an
  LLM timeout is less likely to be transient than a network blip), a
  `Catch` branch on every state that routes to `EscalateOnFailure`
  instead of silently dropping the ticket, and a terminal `WriteAuditLog`
  state so both the happy path and the failure path get logged. Uses
  `${FunctionNameArn}` placeholders — see `template.yaml` below for how
  they're filled in.
- `template.yaml` — AWS SAM template: the DynamoDB table, all seven
  Lambdas (`ingest`, `retrieve`, `classify`, `decide`, `act`, `escalate`,
  `audit-log`), the API Gateway route, the Step Functions state machine
  (substituting real Lambda ARNs into `state_machine.json`), and the IAM
  roles each piece needs. This is the thing `sam deploy` would consume.
- `lambda/ingest_handler.py` — entry point: parses the incoming ticket
  (API Gateway proxy event or a normalized S3 event), writes an initial
  DynamoDB record, and starts a Step Functions execution.
- `lambda/retrieve_handler.py`, `lambda/classify_handler.py` — call a
  SageMaker endpoint if `RETRIEVER_ENDPOINT` / `CLASSIFIER_ENDPOINT` is
  set; otherwise fall back to running `agent.nodes.retrieve` /
  `agent.nodes.classify` in-process (loads the FAISS index / DistilBERT
  checkpoint directly). The fallback is for local testing only — it's
  what pulls torch/transformers/faiss back in, which is exactly what the
  SageMaker split is meant to avoid in a real deployment.
- `lambda/decide_handler.py` — thin wrapper around `agent.nodes.decide`,
  which makes the pipeline's one LLM call (drafting a reply — see
  `agent/README.md`).
- `lambda/act_handler.py` — runs the decided tool calls and records the
  outcome (including partial failures) to DynamoDB before re-raising on
  error, so a human looking at an escalated-by-failure ticket can see
  what actually happened.
- `lambda/escalate_handler.py` — the `Catch` target for every other
  state; defensive about partial state since it can be reached after
  Retrieve, Classify, Decide, *or* Act fails.
- `lambda/audit_log_handler.py` — terminal state, writes the full ticket
  state to DynamoDB. Never raises — it's the last state, nothing catches
  for it.
- `lambda/common.py` — shared helpers (`safe_put` for
  never-crash-the-pipeline DynamoDB writes with float->Decimal
  conversion, since DynamoDB's boto3 API rejects native floats).
- `lambda/requirements.txt` — Lambda-specific deps, kept deliberately
  separate from the training-time root `requirements.txt` (no
  torch/transformers/sentence-transformers/faiss — Lambda has a 250MB
  unzipped package limit).
- `sagemaker/inference_classifier.py`, `sagemaker/inference_retriever.py`
  — SageMaker inference entry points (`model_fn`/`input_fn`/`predict_fn`/
  `output_fn`) that get packaged alongside the trained checkpoints into
  the model tarballs `deploy_endpoints.py` would upload.
- `sagemaker/deploy_endpoints.py` — reference script for packaging +
  deploying both endpoints. **Not run** — see the cost note below.
- `cloudwatch/dashboard.json` — dashboard definition: Step Functions
  execution counts, per-stage Lambda duration/errors, and an
  auto-resolve/draft-hold/escalate breakdown (the last one is a custom
  EMF metric under the `AutoTriage` namespace that the Lambdas would
  need to emit — not wired up, since nothing is deployed to emit it).

## If you actually want to deploy this

```bash
# Validate the template without touching AWS
sam validate --template cloud/template.yaml

# Build + deploy (creates real, billed resources)
sam build --template cloud/template.yaml
sam deploy --guided \
  --parameter-overrides OpenAIApiKeySecretArn=<your-secrets-manager-arn>

# Optionally point Retrieve/Classify at real SageMaker endpoints
# (after training on Kaggle and copying checkpoints back locally):
python cloud/sagemaker/deploy_endpoints.py --config config.yaml --which classifier --role-arn <sagemaker-execution-role-arn>
python cloud/sagemaker/deploy_endpoints.py --config config.yaml --which retriever  --role-arn <sagemaker-execution-role-arn>
# then set RetrieverEndpointName / ClassifierEndpointName params and redeploy
```

Without SageMaker endpoint params set, `retrieve`/`classify` fall back to
loading the models in-process inside the Lambda — which will blow past
the 250MB package limit and cold-start slowly. That fallback exists for
local testing (`python -m agent.graph`), not for a real deployment; treat
"no endpoint configured" in production as a config bug, not a feature.

## Design decision: where does model inference run?

**Chosen: SageMaker real-time endpoints**, not a Lambda container image
with ONNX-exported models. Reasoning: this project's fine-tuned models
(a sentence-transformers bi-encoder and DistilBERT) are standard
HuggingFace/PyTorch artifacts with first-class SageMaker deployment
support (`sagemaker.pytorch.PyTorchModel`) — no export/quantization step
that could silently change eval numbers between what
`eval_retrieval.py`/`eval_classifier.py` report and what's actually
served. The ONNX-container path is cheaper at low, spiky volume, but
that tradeoff only matters for a service that's actually running; it
isn't for this project.

## Cost note

Nothing in `cloud/` has been deployed. A deployed real-time SageMaker
endpoint bills per hour regardless of traffic (the two cheapest CPU
instance options are still tens of dollars/month if left running); API
Gateway + Lambda + Step Functions + DynamoDB are all effectively free at
portfolio-project traffic levels on their own. If you deploy this,
tear down the SageMaker endpoints (`aws sagemaker delete-endpoint`) the
moment you're done demoing it — everything else in the stack costs
approximately nothing to leave running, the endpoints are the one thing
that doesn't.

## TODO

- [x] Decide SageMaker vs. Lambda-container for inference (see above)
- [x] Write the Retrieve/Classify/Decide/Act Lambda handlers, plus
      Escalate and WriteAuditLog
- [x] Add a CloudWatch dashboard: tickets processed, auto-resolve %,
      escalation rate, average latency per stage
- [x] Write the SAM template instead of manual `aws` CLI deploy steps
- [ ] If you actually deploy: wire the `AutoTriage` custom EMF metrics
      the dashboard's pie chart expects (`act_handler.py` /
      `escalate_handler.py` would emit these)
- [ ] If you actually deploy: add an idempotency key to `issue_refund`
      before Act gets any retry policy in the state machine (see
      `lambda/act_handler.py`'s docstring)
