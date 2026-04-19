"""
evaluate_hybrid.py
──────────────────
Evaluate the full hybrid NER pipeline (MuRIL model + rules layer + voter)
on the HiNER test split and print entity-level metrics.

This script evaluates the exact hybrid decode path used in inference_updated.py:
    neural logits -> active rules layer -> ConfidenceVoter -> postposition trim -> tags

Usage:
  python evaluate_hybrid.py
  python evaluate_hybrid.py --model_dir models/muril-hiner/best --batch_size 32
  python evaluate_hybrid.py --test_json data/test.json --limit 2000

Notes:
  - By default it loads HiNER test from HuggingFace raw JSON URL.
  - Ground-truth labels are mapped from HiNER's original 23 labels to this
    project's 15-label schema (same mapping as data/prepare_dataset.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download

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

from inference_updated import HindiNERInference, ID2LABEL


DEFAULT_HINER_TEST_JSON = (
    "https://huggingface.co/datasets/cfilt/HiNER-original/resolve/main/data/test.json"
)

# Fallback order from HiNER's official data/labels_list.txt
# (Do not reorder. Raw ner_tags ids index into this sequence.)
FALLBACK_ORIGINAL_HINER_LABELS = [
    "B-FESTIVAL",
    "B-GAME",
    "B-LANGUAGE",
    "B-LITERATURE",
    "B-LOCATION",
    "B-MISC",
    "B-NUMEX",
    "B-ORGANIZATION",
    "B-PERSON",
    "B-RELIGION",
    "B-TIMEX",
    "I-FESTIVAL",
    "I-GAME",
    "I-LANGUAGE",
    "I-LITERATURE",
    "I-LOCATION",
    "I-MISC",
    "I-NUMEX",
    "I-ORGANIZATION",
    "I-PERSON",
    "I-RELIGION",
    "I-TIMEX",
    "O",
]

# Must match mapping used in data/prepare_dataset.py
ORIGINAL_TO_PROJECT_LABEL = {
    "B-TIMEX": "B-TIME", "I-TIMEX": "I-TIME",
    "B-NUMEX": "B-NUMBER", "I-NUMEX": "I-NUMBER",
    "B-FESTIVAL": "B-OTHER", "I-FESTIVAL": "I-OTHER",
    "B-GAME": "B-OTHER", "I-GAME": "I-OTHER",
    "B-LANGUAGE": "B-OTHER", "I-LANGUAGE": "I-OTHER",
    "B-LITERATURE": "B-OTHER", "I-LITERATURE": "I-OTHER",
    "B-MISC": "B-OTHER", "I-MISC": "I-OTHER",
    "B-RELIGION": "B-OTHER", "I-RELIGION": "I-OTHER",
}

# Must also match LABEL_MERGE in data/prepare_dataset.py
# The model was trained/evaluated with NUMBER merged into OTHER.
LABEL_MERGE = {
    "B-NUMBER": "B-OTHER",
    "I-NUMBER": "I-OTHER",
}


def _load_original_label_names() -> List[str]:
    """
    Load the authoritative raw-label id order from the HiNER dataset repo.
    Falls back to a baked-in copy if network/hub access fails.
    """
    try:
        labels_path = hf_hub_download(
            repo_id="cfilt/HiNER-original",
            repo_type="dataset",
            filename="data/labels_list.txt",
        )
        with open(labels_path, encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if names and "O" in names:
            return names
    except Exception:
        pass

    return FALLBACK_ORIGINAL_HINER_LABELS


def _map_ground_truth(tag_ids: List[int], original_label_names: List[str]) -> List[str]:
    mapped: List[str] = []
    for tid in tag_ids:
        if 0 <= tid < len(original_label_names):
            raw = original_label_names[tid]
        else:
            raw = "O"
        lbl = ORIGINAL_TO_PROJECT_LABEL.get(raw, raw)
        lbl = LABEL_MERGE.get(lbl, lbl)
        mapped.append(lbl)
    return mapped


def _to_builtin(obj: Any) -> Any:
    """Recursively convert numpy / tensor scalar types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(v) for v in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def _load_test_split(test_json: str, limit: Optional[int]):
    ds = load_dataset("json", data_files=test_json, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _decode_neural_batch(
    ner: HindiNERInference,
    token_lists: List[List[str]],
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Decode neural predictions for a batch of tokenized sentences.
    Mirrors the first-subword logic in inference_updated.py.
    """
    encodings = ner.tokenizer(
        token_lists,
        is_split_into_words=True,
        truncation=True,
        max_length=ner.max_length,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = ner.model(
            input_ids=encodings["input_ids"].to(ner.device),
            attention_mask=encodings["attention_mask"].to(ner.device),
        )

    logits_batch = outputs.logits
    probs_batch = torch.softmax(logits_batch, dim=-1)

    all_tags: List[List[str]] = []
    all_confs: List[List[float]] = []

    for batch_idx, tokens in enumerate(token_lists):
        word_ids = encodings.word_ids(batch_index=batch_idx)
        pred_ids = logits_batch[batch_idx].argmax(dim=-1).cpu().tolist()
        pred_probs = probs_batch[batch_idx].max(dim=-1).values.cpu().tolist()

        tags: List[str] = []
        confs: List[float] = []
        seen = set()

        for sub_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            if word_id < len(tokens):
                tags.append(ID2LABEL[pred_ids[sub_idx]])
                confs.append(round(pred_probs[sub_idx], 4))

        all_tags.append(tags)
        all_confs.append(confs)

    return all_tags, all_confs


def evaluate_hybrid(
    model_dir: str,
    test_json: str,
    batch_size: int,
    max_length: int,
    limit: Optional[int],
    device: Optional[str],
    hybrid_mode: str,
    hybrid_gate_conf: float,
) -> Dict:
    ds = _load_test_split(test_json, limit)
    original_label_names = _load_original_label_names()

    print(f"Loading hybrid pipeline from {model_dir} ...")
    ner = HindiNERInference(
        model_dir=model_dir,
        device=device,
        max_length=max_length,
        hybrid_mode=hybrid_mode,
        hybrid_gate_conf=hybrid_gate_conf,
    )

    if ner.rule_engine is None or ner.voter is None:
        raise RuntimeError(
            "Enhanced rule engine is not available. "
            "Cannot run model+rule hybrid evaluation."
        )

    total_rows = len(ds)
    print(f"Evaluating {total_rows} test sentences...")
    print(f"Raw tag mapping source size: {len(original_label_names)} labels")

    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []

    processed = 0
    truncated = 0
    started = time.time()

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch = ds[start:end]

        token_lists: List[List[str]] = batch["tokens"]
        gt_tag_ids_lists: List[List[int]] = batch["ner_tags"]
        gt_label_lists: List[List[str]] = [
            _map_ground_truth(ids, original_label_names) for ids in gt_tag_ids_lists
        ]

        neural_tags_batch, neural_confs_batch = _decode_neural_batch(ner, token_lists)

        for i, tokens in enumerate(token_lists):
            gt_labels = gt_label_lists[i]
            neural_tags = neural_tags_batch[i]
            neural_confs = neural_confs_batch[i]

            n = min(len(tokens), len(gt_labels), len(neural_tags), len(neural_confs))
            if n == 0:
                continue
            if n < len(tokens) or n < len(gt_labels):
                truncated += 1

            tokens_n = tokens[:n]
            gt_n = gt_labels[:n]
            neural_tags_n = neural_tags[:n]
            neural_confs_n = neural_confs[:n]

            pred_tags, _, _ = ner._hybrid_decode(
                tokens=tokens_n,
                neural_tags=neural_tags_n,
                neural_confs=neural_confs_n,
                debug=False,
            )

            all_true.append(gt_n)
            all_pred.append(pred_tags)
            processed += 1

        if processed % max(1000, batch_size * 20) == 0:
            elapsed = time.time() - started
            rate = processed / max(elapsed, 1e-6)
            print(f"  Processed {processed}/{total_rows} sentences ({rate:.1f} sent/s)")

    elapsed = time.time() - started

    # Token-level accuracy (for reference only)
    token_total = 0
    token_correct = 0
    for ts, ps in zip(all_true, all_pred):
        for t, p in zip(ts, ps):
            token_total += 1
            if t == p:
                token_correct += 1
    token_accuracy = (token_correct / token_total) if token_total else 0.0

    metrics: Dict = {
        "model_dir": model_dir,
        "test_json": test_json,
        "sentences_total": total_rows,
        "sentences_evaluated": processed,
        "sentences_truncated": truncated,
        "max_length": max_length,
        "batch_size": batch_size,
        "hybrid_mode": hybrid_mode,
        "hybrid_gate_conf": hybrid_gate_conf,
        "elapsed_seconds": round(elapsed, 3),
        "sent_per_sec": round(processed / max(elapsed, 1e-6), 3),
        "token_accuracy": token_accuracy,
    }

    if SEQEVAL_AVAILABLE:
        metrics.update({
            "entity_precision": precision_score(all_true, all_pred, zero_division=0),
            "entity_recall": recall_score(all_true, all_pred, zero_division=0),
            "entity_f1": f1_score(all_true, all_pred, zero_division=0),
            "classification_report": classification_report(
                all_true, all_pred, digits=4, output_dict=True, zero_division=0
            ),
            "classification_report_text": classification_report(
                all_true, all_pred, digits=4, zero_division=0
            ),
        })

    return metrics


def print_metrics(metrics: Dict) -> None:
    print("\n" + "=" * 72)
    print("HYBRID EVALUATION (MODEL + RULE LAYER)")
    print("=" * 72)
    print(f"Model dir:            {metrics['model_dir']}")
    print(f"Test source:          {metrics['test_json']}")
    print(f"Sentences total:      {metrics['sentences_total']}")
    print(f"Sentences evaluated:  {metrics['sentences_evaluated']}")
    print(f"Truncated sentences:  {metrics['sentences_truncated']} (max_length={metrics['max_length']})")
    print(f"Batch size:           {metrics['batch_size']}")
    print(f"Hybrid mode:          {metrics.get('hybrid_mode', 'default')}")
    print(f"Hybrid gate conf:     {metrics.get('hybrid_gate_conf', 0.80):.2f}")
    print(f"Runtime:              {metrics['elapsed_seconds']:.2f}s")
    print(f"Throughput:           {metrics['sent_per_sec']:.2f} sent/s")
    print(f"Token accuracy:       {metrics['token_accuracy']:.4f} ({metrics['token_accuracy']*100:.2f}%)")

    if "entity_f1" in metrics:
        p = metrics["entity_precision"]
        r = metrics["entity_recall"]
        f1 = metrics["entity_f1"]
        print("\nEntity-level (seqeval, exact span match):")
        print(f"  Precision:          {p:.4f} ({p*100:.2f}%)")
        print(f"  Recall:             {r:.4f} ({r*100:.2f}%)")
        print(f"  F1:                 {f1:.4f} ({f1*100:.2f}%)")
        print("\nPer-class report:")
        print(metrics["classification_report_text"])
    else:
        print("\nseqeval not installed, so entity-level metrics were skipped.")
        print("Install with: pip install seqeval")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/muril-hiner/best")
    parser.add_argument("--test_json", default=DEFAULT_HINER_TEST_JSON,
                        help="Path or URL to raw HiNER test.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N sentences")
    parser.add_argument("--device", default=None,
                        help="cuda | cpu (default: auto)")
    parser.add_argument("--hybrid_mode", default="default",
                        choices=["default", "conservative"],
                        help="default: full rules; conservative: strong rules + neural confidence gate")
    parser.add_argument("--hybrid_gate_conf", type=float, default=0.80,
                        help="In conservative mode, keep neural tag when confidence >= this threshold")
    parser.add_argument("--metrics_out", default="hybrid_test_metrics.json")
    args = parser.parse_args()

    metrics = evaluate_hybrid(
        model_dir=args.model_dir,
        test_json=args.test_json,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
        device=args.device,
        hybrid_mode=args.hybrid_mode,
        hybrid_gate_conf=args.hybrid_gate_conf,
    )

    print_metrics(metrics)

    out_path = Path(args.metrics_out)
    out_path.write_text(
        json.dumps(_to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
