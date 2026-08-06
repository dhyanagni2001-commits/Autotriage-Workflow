"""
Fine-tune distilbert-base-uncased for ticket category classification.

Usage:
    python classifier/train_classifier.py --config config.yaml
"""
import argparse
import json
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
)


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
    processed_dir = Path(cfg["data"]["processed_dir"])

    labels = c_cfg["labels"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(c_cfg["base_model"])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=c_cfg["max_seq_len"])

    train_ds = load_split(processed_dir / "classifier_train.jsonl", label2id).map(tokenize, batched=True)
    val_ds = load_split(processed_dir / "classifier_val.jsonl", label2id).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        c_cfg["base_model"],
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = Path(c_cfg["output_dir"])
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=c_cfg["batch_size"],
        per_device_eval_batch_size=c_cfg["batch_size"],
        num_train_epochs=c_cfg["epochs"],
        learning_rate=c_cfg["lr"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved fine-tuned classifier -> {output_dir}")


if __name__ == "__main__":
    main()
