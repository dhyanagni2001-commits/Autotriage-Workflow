"""
Fine-tune a sentence-transformers bi-encoder on (ticket, positive_article,
hard_negatives) triples using MultipleNegativesRankingLoss.

Saves a full model checkpoint after every epoch under
`<output_dir>/checkpoints/epoch_N/` plus a small `training_state.json`
tracking how many epochs have completed. Re-running the script picks up
from the last completed epoch instead of starting from the base model —
this matters on Kaggle, where a session can time out or disconnect
mid-run and you don't want to lose hours of GPU training.

Usage:
    python retriever/train_biencoder.py --config config.yaml
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def _set_seed(seed: int):
    """Seed every RNG the training loop touches. Called once before model
    creation and again with seed+epoch before each epoch's fit() call, so
    a given epoch's shuffle order and dropout are identical whether it ran
    as part of one continuous run or after a resume — i.e. weights only
    depend on `seed`, not on how many times the job got interrupted."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_kb(path):
    kb = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            kb[row["id"]] = row["text"]
    return kb


def load_training_examples(retrieval_jsonl, kb):
    """Each row becomes one InputExample per hard negative: (ticket, positive)
    with MultipleNegativesRankingLoss using in-batch negatives, plus the
    explicit hard negatives appended as additional in-batch items."""
    examples = []
    with open(retrieval_jsonl) as f:
        for line in f:
            row = json.loads(line)
            pos_text = kb.get(row["positive_id"])
            if pos_text is None:
                continue
            examples.append(InputExample(texts=[row["ticket_text"], pos_text]))
            # Hard negatives: MultipleNegativesRankingLoss treats every other
            # example's positive in the batch as a negative, but we also
            # explicitly add hard-negative pairs to bias the batch composition.
            for neg_id in row["hard_negative_ids"]:
                neg_text = kb.get(neg_id)
                if neg_text:
                    examples.append(InputExample(texts=[row["ticket_text"], neg_text]))
    return examples


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            print(f"warning: could not read {state_path}, starting from scratch")
    return {"completed_epochs": 0}


def _save_state(state_path: Path, state: dict):
    state_path.write_text(json.dumps(state))


def _prune_old_checkpoints(checkpoints_dir: Path, keep_last: int):
    epoch_dirs = sorted(
        (p for p in checkpoints_dir.glob("epoch_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for old_dir in epoch_dirs[:-keep_last]:
        shutil.rmtree(old_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    r_cfg = cfg["retriever"]
    data_cfg = cfg["data"]
    seed = data_cfg["seed"]
    processed_dir = Path(data_cfg["processed_dir"])

    _set_seed(seed)

    kb = load_kb(processed_dir / "kb_articles.jsonl")
    train_examples = load_training_examples(processed_dir / "retrieval_train.jsonl", kb)
    print(f"Loaded {len(train_examples)} training pairs")

    output_dir = Path(r_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "training_state.json"
    keep_last = r_cfg.get("checkpoint_save_total_limit", 3)

    total_epochs = r_cfg["epochs"]
    state = _load_state(state_path)
    completed_epochs = state["completed_epochs"]

    if completed_epochs >= total_epochs:
        print(f"Already completed {completed_epochs}/{total_epochs} epochs — nothing to do. "
              f"Delete {state_path} to force retraining from scratch.")
        # Make sure the final model is present at output_dir root even if
        # a previous run finished the epoch loop but died before this
        # final save (or the dir was moved).
        last_epoch_dir = checkpoints_dir / f"epoch_{completed_epochs}"
        if last_epoch_dir.exists() and not (output_dir / "config.json").exists():
            SentenceTransformer(str(last_epoch_dir)).save(str(output_dir))
        return

    if completed_epochs > 0:
        resume_dir = checkpoints_dir / f"epoch_{completed_epochs}"
        print(f"Resuming from {resume_dir} (epoch {completed_epochs}/{total_epochs} already done)")
        model = SentenceTransformer(str(resume_dir))
    else:
        print(f"Starting fresh from base model {r_cfg['base_model']}")
        model = SentenceTransformer(r_cfg["base_model"])
        model.max_seq_length = r_cfg["max_seq_len"]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=r_cfg["batch_size"])
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(0.1 * len(train_dataloader))

    for epoch in range(completed_epochs, total_epochs):
        print(f"--- epoch {epoch + 1}/{total_epochs} ---")
        _set_seed(seed + epoch)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": r_cfg["lr"]},
            show_progress_bar=True,
        )

        epoch_dir = checkpoints_dir / f"epoch_{epoch + 1}"
        model.save(str(epoch_dir))
        state = {"completed_epochs": epoch + 1}
        _save_state(state_path, state)
        _prune_old_checkpoints(checkpoints_dir, keep_last)
        print(f"Saved checkpoint -> {epoch_dir}")

    model.save(str(output_dir))
    print(f"Training complete. Saved fine-tuned model -> {output_dir}")


if __name__ == "__main__":
    main()
