"""
Run batch inference on a CSV file using HindiNERInference from inference_updated.py.

Default usage:
    python run_csv_inference_updated.py

Custom usage:
    python run_csv_inference_updated.py \
        --input_csv hindi_100_sentences.csv \
        --output_csv hindi_100_sentences_predictions.csv \
        --model_dir models/muril-hiner/best \
        --hybrid_mode conservative
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from inference_updated import HindiNERInference, NERResult


def _safe_add_column(existing: List[str], desired: str) -> str:
    """Return a unique column name by appending suffixes if needed."""
    if desired not in existing:
        return desired
    idx = 2
    while f"{desired}_{idx}" in existing:
        idx += 1
    return f"{desired}_{idx}"


def _read_csv_rows(input_csv: Path, sentence_col: str, encoding: str) -> tuple[List[str], List[Dict[str, str]]]:
    with input_csv.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if sentence_col not in fieldnames:
            raise ValueError(
                f"Column '{sentence_col}' was not found in {input_csv}. "
                f"Available columns: {fieldnames}"
            )

        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})

    return fieldnames, rows


def _entity_payload(result: NERResult) -> List[Dict[str, object]]:
    return [
        {
            "text": e.text,
            "label": e.label,
            "confidence": round(float(e.confidence), 4),
            "source": e.source,
            "start_token": e.start_token,
            "end_token": e.end_token,
        }
        for e in result.entities
    ]


def _entity_compact(result: NERResult) -> str:
    if not result.entities:
        return ""
    return " ; ".join(
        f"{e.text}<{e.label}:{e.confidence:.2f}>" for e in result.entities
    )


def run(args: argparse.Namespace) -> Path:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"Reading input CSV: {input_csv}")
    fieldnames, rows = _read_csv_rows(input_csv, args.sentence_col, args.encoding)

    sentences: List[str] = []
    kept_rows: List[Dict[str, str]] = []
    for row in rows:
        sentence = row.get(args.sentence_col, "").strip()
        if not sentence and args.skip_empty:
            continue
        sentences.append(sentence)
        kept_rows.append(row)

    if args.limit is not None:
        sentences = sentences[: args.limit]
        kept_rows = kept_rows[: args.limit]

    print(f"Loaded {len(kept_rows)} row(s) for inference")

    if not kept_rows:
        raise ValueError("No rows available for inference after filtering.")

    model = HindiNERInference(
        model_dir=args.model_dir,
        hybrid_mode=args.hybrid_mode,
        hybrid_gate_conf=args.hybrid_gate_conf,
    )

    print("Running batch inference...")
    results = model.tag_batch(sentences)

    if len(results) != len(kept_rows):
        raise RuntimeError(
            "Internal mismatch: number of results does not match number of input rows."
        )

    output_fieldnames = list(fieldnames)
    entities_col = _safe_add_column(output_fieldnames, "predicted_entities_json")
    compact_col = _safe_add_column(output_fieldnames, "predicted_entities")
    output_fieldnames.extend([entities_col, compact_col])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding=args.encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for row, result in zip(kept_rows, results):
            enriched = dict(row)
            enriched[entities_col] = json.dumps(_entity_payload(result), ensure_ascii=False)
            enriched[compact_col] = _entity_compact(result)
            writer.writerow(enriched)

    print(f"Saved predictions to: {output_csv}")
    print("Preview:")
    preview_count = min(args.preview, len(kept_rows))
    for i in range(preview_count):
        sent = kept_rows[i].get(args.sentence_col, "")
        ents = _entity_compact(results[i]) or "(no entities)"
        print(f"  {i + 1}. {sent}")
        print(f"     -> {ents}")

    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Hindi NER inference_updated.py over a CSV and write predictions."
    )
    parser.add_argument("--input_csv", default="testpls.csv")
    parser.add_argument("--output_csv", default="test_predictions2.csv")
    parser.add_argument("--sentence_col", default="sentence")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--model_dir", default="models/muril-hiner/best")
    parser.add_argument(
        "--hybrid_mode",
        default="default",
        choices=["default", "conservative"],
    )
    parser.add_argument("--hybrid_gate_conf", type=float, default=0.80)
    parser.add_argument("--skip_empty", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview", type=int, default=5)
    return parser


if __name__ == "__main__":
    cli_parser = build_parser()
    cli_args = cli_parser.parse_args()
    run(cli_args)