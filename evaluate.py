"""
evaluate.py
───────────
Evaluates a fine-tuned model on the HiNER test set with full entity-level
metrics broken down by entity type.

Distinction between token-level and entity-level accuracy:
  Token-level: "what % of individual tokens were labelled correctly?"
  Entity-level: "what % of complete entity spans were correctly identified?"

Token-level accuracy is inflated because most tokens are O (non-entity),
and those are trivially correct. Entity-level F1 is the true measure.

seqeval uses exact span matching: a PERSON entity "नरेंद्र मोदी" is only
correct if BOTH tokens are labelled correctly AND the span boundaries match.
A system that gets "नरेंद्र" right but misses "मोदी" gets 0 credit.

Usage:
    python3 evaluate.py --model_dir models/muril-hiner/best
    python3 evaluate.py --model_dir models/muril-hiner/best --error_analysis
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

try:
    from seqeval.metrics import classification_report, f1_score
    SEQEVAL = True
except ImportError:
    SEQEVAL = False

from train import HiNERDataset, ID2LABEL, decode_predictions


def load_model(model_dir: str, device: torch.device):
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_test(model_dir: str, data_dir: str, batch_size: int = 32,
                  error_analysis: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_dir, device)

    test_ds     = HiNERDataset(f"{data_dir}/test.jsonl")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs        = model(input_ids=input_ids,
                                   attention_mask=attention_mask)
            true_seqs, pred_seqs = decode_predictions(outputs.logits.cpu(), labels.cpu())
            all_true.extend(true_seqs)
            all_pred.extend(pred_seqs)

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS — HiNER Test Set")
    print("=" * 65)

    if SEQEVAL:
        print(classification_report(all_true, all_pred, digits=4))
        overall_f1 = f1_score(all_true, all_pred)
        print(f"Overall entity-level F1: {overall_f1:.4f} ({overall_f1*100:.2f}%)")
    else:
        # Manual token-level stats
        correct = total = 0
        for ts, ps in zip(all_true, all_pred):
            for t, p in zip(ts, ps):
                if t == p:
                    correct += 1
                total += 1
        print(f"Token-level accuracy: {correct/total:.4f} (install seqeval for entity-level F1)")

    if error_analysis:
        print("\n" + "=" * 65)
        print("ERROR ANALYSIS — Most Common Confusion Types")
        print("=" * 65)
        confusion = defaultdict(int)
        for ts, ps in zip(all_true, all_pred):
            for t, p in zip(ts, ps):
                if t != p:
                    confusion[(t, p)] += 1
        sorted_conf = sorted(confusion.items(), key=lambda x: -x[1])
        for (true, pred), count in sorted_conf[:20]:
            print(f"  True={true:<18} Pred={pred:<18} Count={count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/muril-hiner/best")
    parser.add_argument("--data_dir",  default="data/processed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--error_analysis", action="store_true")
    args = parser.parse_args()
    evaluate_test(args.model_dir, args.data_dir, args.batch_size, args.error_analysis)
