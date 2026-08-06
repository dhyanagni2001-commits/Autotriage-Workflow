# AutoTriage

Agentic support-ticket resolution system. Two fine-tuned models (a
bi-encoder retriever and a DistilBERT classifier) feed a LangGraph agent
that decides whether to auto-resolve a ticket, draft a reply for human
review, or escalate — then executes that decision and logs why. Designed
to run locally and deploy as a serverless pipeline on AWS.

> Built to demonstrate three things together — **workflow automation
> (RPA)**, **RAG**, and **cloud deployment of an agentic pipeline** —
> combined in one system rather than as three separate demos, per my
> advisor's guidance on where the field is heading.

## Table of contents

- [What it does](#what-it-does)
- [Why two trained models, not just an LLM agent](#why-two-trained-models-not-just-an-llm-agent)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Repo layout](#repo-layout)
- [Quickstart](#quickstart)
- [Example](#example)
- [Training the models](#training-the-models)
- [Evaluation](#evaluation)
- [Cloud deployment](#cloud-deployment)
- [Status / roadmap](#status--roadmap)
- [Design notes](#design-notes)

## What it does

A support ticket comes in (email, webhook, or a dropped file). The
pipeline:

1. **Retrieves** relevant knowledge-base / policy articles using a
   bi-encoder fine-tuned on ticket → KB-article pairs with contrastive
   loss and hard negative mining.
2. **Classifies** the ticket into a category and urgency level using a
   fine-tuned DistilBERT classifier — a real trained, benchmarked model,
   not a zero-shot LLM prompt.
3. **Reasons** over the retrieved context + classification with a
   LangGraph agent, which decides: auto-resolve, draft-and-hold for a
   human, or escalate outright.
4. **Acts** on that decision: calls a (mocked) refund/CRM tool, drafts a
   reply, or opens an escalation with the full reasoning trace attached.
5. **Logs** every decision so the system is inspectable end-to-end, not a
   black box.

## Why two trained models, not just an LLM agent

An agent that just prompts an LLM to "classify this ticket" produces no
benchmarkable ML artifact. Here, both the retriever and the classifier
are trained models with real held-out metrics:

| Component  | Fine-tuned              | Baseline                         | Metric        |
|------------|--------------------------|-----------------------------------|----------------|
| Retriever  | bi-encoder (contrastive + hard negatives) | `all-MiniLM-L6-v2` off-the-shelf | P@k, MRR |
| Classifier | `distilbert-base-uncased` | TF-IDF + logistic regression      | accuracy, macro-F1 |

*(numbers filled in under [Evaluation](#evaluation) once training on CARC is done)*

The LangGraph agent deliberately does **not** do retrieval or
classification itself — it only consumes structured outputs from the two
trained models and focuses on the decision + tool-calling step. This
mirrors how production RPA systems separate "understand the input" (ML)
from "decide what to do" (orchestration/agent logic), and keeps each
piece independently testable.

## Architecture

**Local / logical flow:**

```
Ticket ──▶ Fine-tuned retriever ──▶ Fine-tuned classifier ──▶ LangGraph agent ──▶ Action
            (FAISS + bi-encoder)      (DistilBERT)              (reasoning +         (refund API /
                                                                   tool calls)          CRM / draft /
                                                                                         escalate)
                                                                        │
                                                                        ▼
                                                                 Audit trail
```

**AWS deployment:**

```
API Gateway ──▶ Lambda (ingest/parse) ──▶ Step Functions
                                             ├─ Retrieve   (Lambda → FAISS / SageMaker endpoint)
                                             ├─ Classify   (Lambda → classifier endpoint)
                                             ├─ Decide     (Lambda → LangGraph agent step)
                                             └─ Act        (Lambda → tool calls)
                                                  │
                                                  ▼
                                            DynamoDB (ticket state + audit log)
```

Full state machine definition and deployment notes: [`cloud/README.md`](cloud/README.md).

## Tech stack

- **Retrieval**: `sentence-transformers`, FAISS, contrastive loss (`MultipleNegativesRankingLoss`), hard negative mining
- **Classification**: `transformers` (DistilBERT), HuggingFace `Trainer`, scikit-learn (TF-IDF baseline)
- **Agent orchestration**: LangGraph
- **Training infra**: USC CARC (A100 GPUs)
- **Cloud**: AWS Lambda, Step Functions, API Gateway, DynamoDB
- **Data**: [Bitext customer-support LLM training dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

## Repo layout

```
autotriage/
├── data/            # dataset download + prep (Bitext customer-support corpus)
├── retriever/       # bi-encoder fine-tuning (contrastive loss, hard negatives) + FAISS index + eval
├── classifier/      # DistilBERT ticket classifier fine-tuning + eval vs. TF-IDF baseline
├── agent/           # LangGraph agent: state, nodes, tools, graph definition
├── cloud/           # Step Functions state machine + Lambda handlers
├── eval/            # end-to-end pipeline evaluation (auto-resolve rate, escalation accuracy)
├── notebooks/       # scratch / exploration
├── config.yaml      # single source of truth for model names, thresholds, AWS settings
└── setup.py         # makes the repo pip-installable so imports work from anywhere
```

Each subfolder has its own `README.md` with usage details and a TODO list
specific to that piece.

## Quickstart

```bash
git clone https://github.com/dhyanagni2001-commits/autotriage-workflow.git
cd autotriage-workflow

conda create -n autotriage python=3.11 -y
conda activate autotriage
pip install -r requirements.txt
pip install -e .        # installs the repo as a package so agent.*, classifier.*
                         # imports resolve correctly regardless of cwd
```

Sanity-check the agent graph (runs with mocked retrieve/classify until the
real trained models are wired in — see [`agent/README.md`](agent/README.md)):

```bash
python -m agent.graph
python -m eval.run_eval_suite --test-file eval/labeled_tickets.jsonl
```

## Example

Input ticket:

> "Hi, I was charged twice for my subscription this month, can you refund
> one of the charges? Order #48213." — amount: $12.99

Pipeline trace:

```
1. Retrieve  → top KB match: "Duplicate charge refund policy: auto-approve under $50"
2. Classify  → category: refund, urgency: medium, confidence: 0.93
3. Decide    → refund $12.99 <= $50 threshold → auto_resolve
4. Act       → issue_refund() + update_crm() + draft confirmation reply
```

Contrast case — same category, larger amount ($340): the decide step
routes to `draft_hold` / `escalate` instead of calling the refund tool
directly, because it exceeds the auto-approval threshold in
`config.yaml`. That branching behavior — knowing when *not* to act — is
the core design decision this project is built around, not just "can an
LLM answer a support ticket."

## Training the models

```bash
# 1. Prepare data (classifier splits + retrieval pairs + synthetic KB)
python data/prepare_bitext.py --config config.yaml

# 2. Fine-tune the retriever (bi-encoder, contrastive + hard negatives)
python retriever/train_biencoder.py --config config.yaml
python retriever/build_index.py --config config.yaml

# 3. Fine-tune the classifier
python classifier/train_classifier.py --config config.yaml
```

Training is meant to run on USC CARC (A100 nodes), same setup as my
QueryLens project. See [`retriever/README.md`](retriever/README.md) and
[`classifier/README.md`](classifier/README.md) for CARC job notes.

## Evaluation

```bash
# Retrieval: fine-tuned vs. off-the-shelf baseline (P@k, MRR)
python retriever/eval_retrieval.py --config config.yaml

# Classification: fine-tuned vs. TF-IDF baseline (accuracy, macro-F1)
python classifier/eval_classifier.py --config config.yaml

# End-to-end: auto-resolve rate, escalation rate, decision accuracy
python -m eval.run_eval_suite --test-file eval/labeled_tickets.jsonl
```

Results table (to fill in after training runs complete):

```
retriever                 P@1    P@5    P@10   MRR
baseline (off-the-shelf)  —      —      —      —
fine-tuned                —      —      —      —

classifier                 accuracy  macro-F1
tfidf + logreg (baseline)  —         —
distilbert (fine-tuned)    —         —
```

## Cloud deployment

Deployment target is AWS: API Gateway → Lambda (ingest) → Step Functions
(orchestrates retrieve/classify/decide/act with retries and a
failure-escalation branch) → DynamoDB (state + audit trail). See
[`cloud/README.md`](cloud/README.md) for the full state machine
definition, Lambda handler notes, and the open question on where model
inference runs (SageMaker endpoint vs. Lambda container image).

## Status / roadmap

- [x] Repo scaffold: data prep, retriever, classifier, agent, cloud, eval
- [x] Agent decision logic + local end-to-end test (mocked retrieve/classify)
- [ ] Data prep run against real Bitext dataset, finalize intent → label mapping
- [ ] Bi-encoder fine-tuning + retrieval eval (P@k vs. baseline)
- [ ] DistilBERT classifier fine-tuning + eval (accuracy / F1 vs. baseline)
- [ ] Wire real trained models into `agent/nodes.py` (replace mocks)
- [ ] Step Functions / Lambda deployment on AWS
- [ ] Audit-trail dashboard (CloudWatch or a small dashboard view)
- [ ] Final eval report with real numbers in this README

## Design notes

- The classifier and retriever are kept as separate, independently
  evaluable artifacts. The agent only ever sees the classifier's
  structured output (category, urgency, confidence) — never its
  internals — so each piece can be swapped or re-benchmarked on its own.
- Escalation logic (knowing when *not* to auto-act) is the actual
  interesting design decision in this project, not the retrieval or
  classification accuracy alone — see the decision table in
  [`agent/README.md`](agent/README.md).
- Retrieval fine-tuning reuses the contrastive loss + hard-negative
  mining approach from my [QueryLens](https://github.com/dhyanagni2001-commits/ShopMind)
  project, applied to a new domain, to show the technique transfers
  rather than being a one-off.

---
