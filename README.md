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

See [Evaluation](#evaluation) for the actual numbers from training on Kaggle.

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
                                             ├─ Retrieve    (Lambda → SageMaker retriever endpoint)
                                             ├─ Classify    (Lambda → SageMaker classifier endpoint)
                                             ├─ Decide      (Lambda → agent decision logic + 1 LLM call)
                                             ├─ Act         (Lambda → tool calls)
                                             └─ on failure: EscalateOnFailure (Lambda)
                                                  │
                                                  ▼
                                            WriteAuditLog (Lambda)
                                                  │
                                                  ▼
                                            DynamoDB (ticket state + audit log)
```

Full state machine definition and deployment notes: [`cloud/README.md`](cloud/README.md).

## Tech stack

- **Retrieval**: `sentence-transformers`, FAISS, contrastive loss (`MultipleNegativesRankingLoss`), hard negative mining
- **Classification**: `transformers` (DistilBERT), HuggingFace `Trainer`, scikit-learn (TF-IDF baseline)
- **Agent orchestration**: LangGraph, `langchain-openai` for the one LLM-backed node
- **Training infra**: Kaggle GPU notebooks (see [`notebooks/kaggle_train_autotriage.ipynb`](notebooks/kaggle_train_autotriage.ipynb))
- **Cloud**: AWS Lambda, Step Functions, API Gateway, DynamoDB, SageMaker (inference), CloudWatch, SAM
- **Data**: [Bitext customer-support LLM training dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

## Repo layout

```
autotriage/
├── data/            # dataset download + prep (Bitext customer-support corpus)
├── retriever/       # bi-encoder fine-tuning (contrastive loss, hard negatives) + FAISS index + eval
├── classifier/      # DistilBERT ticket classifier fine-tuning + eval vs. TF-IDF baseline
├── agent/           # LangGraph agent: state, nodes, tools, graph definition
├── cloud/           # Step Functions state machine, Lambda handlers, SAM template,
│                    #   SageMaker inference entry points, CloudWatch dashboard
├── eval/            # end-to-end pipeline evaluation (auto-resolve rate, escalation accuracy)
├── tests/           # unit tests for the agent's decision logic
├── notebooks/       # kaggle_train_autotriage.ipynb — the training pipeline
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

Sanity-check the agent graph. `retrieve`/`classify` automatically use the
real fine-tuned artifacts once they exist on disk (see
[Training the models](#training-the-models)); until then they fall back
to a small deterministic mock so the graph is runnable from a fresh
clone with zero setup — see [`agent/README.md`](agent/README.md):

```bash
python -m agent.graph
python -m eval.run_eval_suite --test-file eval/labeled_tickets.jsonl
pytest tests/ -v
```

Optional: set `OPENAI_API_KEY` in the environment to get real LLM-drafted
replies out of the `decide` node. Without it, `draft_reply` falls back to
a deterministic template and everything else still works.

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

Training runs on **Kaggle** (free GPU notebooks), not locally — the
whole pipeline (data prep → bi-encoder fine-tuning → FAISS index →
retrieval eval → DistilBERT fine-tuning → classification eval) is one
notebook:
[`notebooks/kaggle_train_autotriage.ipynb`](notebooks/kaggle_train_autotriage.ipynb).

1. Upload the notebook to Kaggle (or push this repo to GitHub and let the
   notebook `git clone` it — see the notebook's first cell), turn on a
   GPU accelerator and Internet in the notebook settings, and run all
   cells.
2. Both training scripts checkpoint every epoch and **auto-resume** on
   re-run instead of starting over — see
   [`retriever/README.md`](retriever/README.md) and
   [`classifier/README.md`](classifier/README.md) for how, and the
   notebook's own "Resuming after a timeout / disconnect" section for
   carrying that state across Kaggle sessions via **Save Version** +
   re-attaching the output as an input dataset.
3. The last cell zips the trained checkpoints, FAISS index, processed
   data, and an `eval_results.txt` into `autotriage_artifacts.zip` under
   the notebook's Output tab. Download it and unzip into this repo at the
   matching paths (`retriever/checkpoints/...`,
   `retriever/faiss_index/...`, `classifier/checkpoints/...`,
   `data/processed/...`).
4. Re-run `python -m agent.graph` locally — `agent/nodes.py` picks up
   the real artifacts automatically (see [`agent/README.md`](agent/README.md)),
   no code changes needed.

Equivalent standalone commands, if you'd rather run each step by hand
(e.g. locally on a GPU box) instead of via the notebook:

```bash
# 1. Prepare data (classifier splits + retrieval pairs + synthetic KB)
python data/prepare_bitext.py --config config.yaml

# 2. Fine-tune the retriever (bi-encoder, contrastive + hard negatives)
python retriever/train_biencoder.py --config config.yaml
python retriever/build_index.py --config config.yaml

# 3. Fine-tune the classifier
python classifier/train_classifier.py --config config.yaml
```

## Evaluation

```bash
# Retrieval: fine-tuned vs. off-the-shelf baseline (P@k, MRR)
python retriever/eval_retrieval.py --config config.yaml

# Classification: fine-tuned vs. TF-IDF baseline (accuracy, macro-F1)
python classifier/eval_classifier.py --config config.yaml

# End-to-end: auto-resolve rate, escalation rate, decision accuracy
python -m eval.run_eval_suite --test-file eval/labeled_tickets.jsonl
```

Results (Kaggle T4 x2, Bitext dataset, deduplicated splits — see
[Design notes](#design-notes) for why dedup mattered):

```
retriever                 P@1    P@5    P@10   MRR
baseline (off-the-shelf)  0.54   0.96   1.00   0.68
fine-tuned                0.97   1.00   1.00   0.99

classifier                 accuracy  macro-F1
tfidf + logreg (baseline)  0.992     0.992
distilbert (fine-tuned)    0.997     0.997
```

Both retrieval and classification score near-ceiling on both the
fine-tuned model *and* the dumb baseline, which is a real result, not a
bug — worth reading honestly rather than at face value:

- **Retrieval** only has 6 possible targets (one canonical KB article per
  label), so even the off-the-shelf baseline has a small candidate pool
  to search over. The fine-tuned bi-encoder's real win is going from
  P@1 0.54 → 0.97, i.e. actually learning to pick the *right* one of the
  6 on the first try, not the ceiling values themselves.
- **Classification** is easy because the 6 coarse categories
  (`billing`/`refund`/`technical`/`account`/`shipping`/`urgent_escalate`)
  are lexically well-separated — a linear TF-IDF model gets 99.2% on its
  own. DistilBERT's real advantage here is a small, consistent margin
  (0.997 vs. 0.992), not a dramatic one. A harder, more convincing
  benchmark would classify against the dataset's original ~27 fine-grained
  intents instead of the coarse 6-label mapping in `config.yaml`.
- These numbers were confirmed *not* to be train/test leakage: an initial
  run (before the dedup fix below) hit 99.9%/100% on the classifier
  because the raw Bitext dataset contains exact-duplicate rows (it's
  templated, e.g. unfilled `{{Order Number}}` placeholders repeat
  verbatim) that a naive random split let leak across train/test.
  `data/prepare_bitext.py::build_examples` now dedupes by exact ticket
  text before splitting, `assert_no_split_leakage` fails the whole data
  prep step if that ever regresses, and `tests/test_data_no_leakage.py`
  re-verifies it against the real processed data.

## Cloud deployment

Deployment target is AWS: API Gateway → Lambda (ingest) → Step Functions
(orchestrates retrieve/classify/decide/act, each a separate Lambda, with
retries and a failure-escalation branch) → DynamoDB (state + audit
trail). The full architecture — all seven Lambda handlers, the state
machine, a SAM template, a CloudWatch dashboard, and SageMaker inference
entry points for both trained models — is written and internally
consistent, but **intentionally not deployed** (a real-time SageMaker
endpoint bills per hour; this is a portfolio project, not a running
service). See [`cloud/README.md`](cloud/README.md) for the full
breakdown, the SageMaker-vs-Lambda-container design decision, and exact
`sam deploy` steps if you want to actually stand it up.

## Status / roadmap

- [x] Repo scaffold: data prep, retriever, classifier, agent, cloud, eval
- [x] Agent decision logic + local end-to-end test (mocked retrieve/classify)
- [x] Wire real trained models into `agent/nodes.py`, with mock fallback
      when artifacts aren't present (see [`agent/README.md`](agent/README.md))
- [x] Wire a real LLM call into the `decide` node's `draft_reply`, with a
      template fallback when no API key is configured
- [x] Kaggle training notebook (data prep → retriever → classifier →
      both evals) with epoch-level resume across session timeouts
- [x] Step Functions / Lambda code + SAM template for AWS deployment
      (written, not deployed — see [`cloud/README.md`](cloud/README.md))
- [x] CloudWatch dashboard definition
- [x] Unit tests for the agent's decision logic (`pytest tests/`)
- [x] Data prep run against real Bitext dataset, finalize intent → label mapping
- [x] Bi-encoder fine-tuning + retrieval eval (P@k vs. baseline) — run on Kaggle
- [x] DistilBERT classifier fine-tuning + eval (accuracy / F1 vs. baseline) — run on Kaggle
- [x] Final eval report with real numbers in this README (see [Evaluation](#evaluation))
- [x] Found and fixed train/test leakage from duplicate rows in the raw
      dataset (`data/prepare_bitext.py::build_examples` dedup +
      `assert_no_split_leakage` guard + `tests/test_data_no_leakage.py`)
      and reproducibility gaps in both training scripts (unseeded shuffling
      / classifier head init — see `_set_seed` in
      [`retriever/train_biencoder.py`](retriever/train_biencoder.py) and
      `set_seed` in [`classifier/train_classifier.py`](classifier/train_classifier.py))
- [ ] Download `autotriage_artifacts.zip` from the Kaggle run into this
      repo's local paths and confirm `python -m agent.graph` picks up the
      real trained models instead of the mock fallback

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
