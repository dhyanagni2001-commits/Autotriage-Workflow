"""
Fine-tune distilbert-base-uncased for ticket category classification.

Saves a HuggingFace `Trainer` checkpoint every epoch under `output_dir`
(checkpoint-<step>/) and a `TRAINING_COMPLETE` marker once the final
model has been saved. Re-running the script resumes from the latest
checkpoint instead of starting over, and does nothing if training
already finished — this matters on Kaggle, where a session can time out
or disconnect mid-run and you don't want to lose GPU hours.

Usage:
    python classifier/train_classifier.py --config config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.labels import category_for  # noqa: E402


def load_split(path, label2id):
    rows = [json.loads(l) for l in open(path)]
    return Dataset.from_dict({
        "text": [r["ticket_text"] for r in rows],
        "label": [label2id[r["label"]] for r in rows],
    })


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    c_cfg = cfg["classifier"]
    seed = cfg["data"]["seed"]
    processed_dir = Path(cfg["data"]["processed_dir"])

    labels = json.loads((processed_dir / "labels.json").read_text())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    output_dir = Path(c_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = output_dir / "TRAINING_COMPLETE"

    if complete_marker.exists():
        print(f"{complete_marker} already exists — training already finished. "
              f"Delete it to force retraining from scratch.")
        return

    # Must run before from_pretrained() below — the classification head on
    # top of distilbert is randomly initialized at that call, and
    # TrainingArguments.seed (applied later, inside Trainer.__init__) is too
    # late to control it. Without this, resuming vs. a fresh run — or two
    # fresh runs — start from different head weights even with the same
    # config.
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(c_cfg["base_model"])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=c_cfg["max_seq_len"])

    train_ds = load_split(processed_dir / "classifier_train.jsonl", label2id).map(tokenize, batched=True)
    val_ds = load_split(processed_dir / "classifier_val.jsonl", label2id).map(tokenize, batched=True)

    resume_checkpoint = get_last_checkpoint(str(output_dir))
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        model = AutoModelForSequenceClassification.from_pretrained(
            resume_checkpoint, num_labels=len(labels), id2label=id2label, label2id=label2id,
        )
    else:
        print(f"Starting fresh from base model {c_cfg['base_model']}")
        model = AutoModelForSequenceClassification.from_pretrained(
            c_cfg["base_model"], num_labels=len(labels), id2label=id2label, label2id=label2id,
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=c_cfg["batch_size"],
        per_device_eval_batch_size=c_cfg["batch_size"],
        num_train_epochs=c_cfg["epochs"],
        learning_rate=c_cfg["lr"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=c_cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        seed=seed,
        data_seed=seed,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    try:
        # transformers >= 4.49 renamed this kwarg; older pins still need
        # `tokenizer=` since they don't accept `processing_class` at all.
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    # Ship the intent -> routing-category map inside config.json so serving
    # code (SageMaker tarball) can route without importing data/labels.py.
    trainer.model.config.intent_to_category = {l: category_for(l) for l in labels}
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    complete_marker.write_text("ok")
    print(f"Saved fine-tuned classifier -> {output_dir}")


if __name__ == "__main__":
    main()
