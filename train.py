"""
train.py
────────
Fine-tunes MuRIL (or IndicBERT / XLM-RoBERTa) on HiNER for Hindi NER.

Why MuRIL over mBERT?
  MuRIL was pre-trained specifically on 17 Indian languages using transliterated
  and bilingual text. It has richer Hindi subword vocabulary than mBERT.
  On HiNER, MuRIL achieves ~88% F1 vs mBERT's ~83% F1 out of the box.

Why not IndicBERT?
  IndicBERT is smaller (12M vs 236M params) and faster, but MuRIL gives
  consistently higher entity-level F1 on Hindi NER tasks. Use IndicBERT
  if you need to run inference on CPU in production; use MuRIL for maximum
  accuracy.

Training decisions explained:
  - Learning rate 2e-5: standard for BERT fine-tuning on NER; higher rates
    cause catastrophic forgetting of the pre-trained representations.
  - Warmup ratio 0.1: avoids large gradient steps in early epochs before
    the classification head has learned reasonable weights.
  - Weight decay 0.01: mild L2 regularisation on non-bias parameters.
  - gradient_checkpointing: halves GPU memory at ~20% speed cost — critical
    for running on a single consumer GPU.
  - label_all_subwords=False: only the first subword of each word contributes
    to the loss. This gives sharper entity boundary detection.

Usage:
    # With GPU (recommended):
    python3 train.py --model google/muril-base-cased --epochs 5

    # CPU only (slow but works, ~6h for full HiNER):
    python3 train.py --model google/muril-base-cased --epochs 3 --batch_size 8

    # Fast experiment with IndicBERT:
    python3 train.py --model ai4bharat/indic-bert --epochs 5 --batch_size 32
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# seqeval is the standard NER evaluation library
try:
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False
    print("Warning: seqeval not installed. Install with: pip install seqeval")


# ── Label schema ──────────────────────────────────────────────────────────────
HINER_LABELS = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-LOCATION", "I-LOCATION",
    "B-ORGANIZATION", "I-ORGANIZATION",
    "B-DATE", "I-DATE",
    "B-TIME", "I-TIME",
    "B-NUMBER", "I-NUMBER",
    "B-OTHER", "I-OTHER",
]
LABEL2ID = {l: i for i, l in enumerate(HINER_LABELS)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(HINER_LABELS)


# ── Dataset ───────────────────────────────────────────────────────────────────
class HiNERDataset(Dataset):
    """
    Loads pre-tokenised data from .jsonl files produced by prepare_dataset.py.
    Each line is a JSON dict with keys: input_ids, attention_mask, labels.
    """

    def __init__(self, jsonl_path: str, max_examples: Optional[int] = None):
        self.examples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                self.examples.append(json.loads(line))
        print(f"  Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "input_ids":      torch.tensor(item["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"],  dtype=torch.long),
            "labels":         torch.tensor(item["labels"],          dtype=torch.long),
        }


# ── Evaluation ────────────────────────────────────────────────────────────────
def decode_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> tuple:
    """
    Convert model output logits + label tensors to two lists of label
    sequences (one per sentence), ignoring -100 (special tokens / padding).

    Returns (true_labels, pred_labels) where each is List[List[str]].
    seqeval expects this exact format.
    """
    pred_ids  = predictions.argmax(dim=-1)
    true_seqs = []
    pred_seqs = []

    for pred_seq, label_seq in zip(pred_ids, labels):
        true_row = []
        pred_row = []
        for p, l in zip(pred_seq, label_seq):
            if l.item() == -100:
                continue
            true_row.append(ID2LABEL[l.item()])
            pred_row.append(ID2LABEL[p.item()])
        true_seqs.append(true_row)
        pred_seqs.append(pred_row)

    return true_seqs, pred_seqs


def evaluate(model, dataloader, device) -> Dict:
    """Run evaluation on a dataloader, return seqeval metrics."""
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            true_seqs, pred_seqs = decode_predictions(outputs.logits.cpu(), labels.cpu())
            all_true.extend(true_seqs)
            all_pred.extend(pred_seqs)

    if SEQEVAL_AVAILABLE:
        return {
            "f1":        f1_score(all_true, all_pred),
            "precision": precision_score(all_true, all_pred),
            "recall":    recall_score(all_true, all_pred),
            "report":    classification_report(all_true, all_pred, digits=4),
        }
    else:
        # Fallback: compute token-level accuracy (not entity-level)
        correct = sum(
            1 for ts, ps in zip(all_true, all_pred)
            for t, p in zip(ts, ps) if t == p
        )
        total = sum(len(ts) for ts in all_true)
        return {"accuracy": correct / total if total else 0}


# ── Training loop ──────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cpu":
        print("WARNING: Training on CPU will be very slow (~6-8 hours for full HiNER).")
        print("         Consider using Google Colab (free GPU) or a cloud VM.")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # classification head is random-init
    )

    # Gradient checkpointing halves memory; worth ~20% speed cost
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable:,} trainable")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir = args.data_dir
    train_ds = HiNERDataset(f"{data_dir}/train.jsonl",      args.max_train_examples)
    val_ds   = HiNERDataset(f"{data_dir}/validation.jsonl", args.max_val_examples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=2)

    # ── Optimiser ────────────────────────────────────────────────────────────
    # Separate weight decay: apply only to weight matrices, not biases/layernorm
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimiser = AdamW(param_groups, lr=args.lr)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimiser, warmup_steps, total_steps
    )

    print(f"\nTraining config:")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Total steps:    {total_steps}")
    print(f"  Warmup steps:   {warmup_steps}")
    print(f"  Train examples: {len(train_ds)}")
    print(f"  Val examples:   {len(val_ds)}")

    # ── Mixed precision ───────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1      = 0.0
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path     = output_dir / "training_log.jsonl"

    print("\n" + "=" * 60)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss    = 0.0
        epoch_start   = time.time()
        steps_in_epoch = len(train_loader)

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimiser.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            scheduler.step()
            epoch_loss += loss.item()

            if step % 100 == 0 or step == steps_in_epoch:
                avg = epoch_loss / step
                elapsed = time.time() - epoch_start
                eta = (elapsed / step) * (steps_in_epoch - step)
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch}/{args.epochs} | "
                      f"Step {step}/{steps_in_epoch} | "
                      f"Loss {avg:.4f} | "
                      f"LR {lr_now:.2e} | "
                      f"ETA {eta/60:.1f}m")

        # ── Validation ────────────────────────────────────────────────────────
        print(f"\n  Running validation (epoch {epoch})...")
        metrics = evaluate(model, val_loader, device)

        f1        = metrics.get("f1", metrics.get("accuracy", 0))
        precision = metrics.get("precision", 0)
        recall    = metrics.get("recall", 0)
        avg_loss  = epoch_loss / steps_in_epoch

        print(f"  Val F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  "
              f"TrainLoss={avg_loss:.4f}  "
              f"Time={(time.time()-epoch_start)/60:.1f}m")

        if "report" in metrics:
            print(f"\n{metrics['report']}")

        # Log to file
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_f1": f1,
            "val_precision": precision,
            "val_recall": recall,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Save best checkpoint
        if f1 > best_f1:
            best_f1 = f1
            best_dir = output_dir / "best"
            model.save_pretrained(best_dir)
            AutoTokenizer.from_pretrained(args.model).save_pretrained(best_dir)
            print(f"  *** New best model saved to {best_dir} (F1={f1:.4f}) ***")

        # Always save latest
        latest_dir = output_dir / "latest"
        model.save_pretrained(latest_dir)
        print("=" * 60)

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Best model saved to: {output_dir}/best")
    print(f"Next step: python3 evaluate.py --model_dir {output_dir}/best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/muril-base-cased",
                        choices=["google/muril-base-cased",
                                 "ai4bharat/indic-bert",
                                 "xlm-roberta-base",
                                 "xlm-roberta-large"],
                        help="Base pre-trained model to fine-tune")
    parser.add_argument("--data_dir", default="data/processed",
                        help="Directory with train/validation/test .jsonl files")
    parser.add_argument("--output_dir", default="models/muril-hiner",
                        help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device batch size. Reduce to 8 if OOM on GPU.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Peak learning rate. 2e-5 is optimal for MuRIL on NER.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--max_train_examples", type=int, default=None,
                        help="Limit training data (for quick experiments)")
    parser.add_argument("--max_val_examples", type=int, default=None)
    args = parser.parse_args()

    train(args)
