"""
prepare_dataset.py
──────────────────
Downloads HiNER from HuggingFace, aligns tokens to MuRIL/IndicBERT subword
tokenisation, and writes train/val/test splits as ready-to-train tensors.

Run this ONCE before training:
    python3 data/prepare_dataset.py --model google/muril-base-cased

Why HiNER and not ICON-2013?
  HiNER has 108,608 sentences from diverse sources (news, tourism, wiki).
  ICON-2013 has ~3,000 sentences from a single news domain.
  Training on HiNER gives far better generalisation.

Label schema (HiNER uses IOB2):
  O, B-PERSON, I-PERSON, B-LOCATION, I-LOCATION,
  B-ORGANIZATION, I-ORGANIZATION, B-DATE, I-DATE,
  B-TIME, I-TIME, B-NUMBER, I-NUMBER, B-OTHER, I-OTHER
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

# ── HiNER label set ──────────────────────────────────────────────────────────
# These are the exact label strings in the HiNER dataset on HuggingFace.
# We keep them as-is; mapping to a reduced set is optional (see LABEL_MERGE).
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
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Optional: merge fine labels to coarse (reduces confusion at boundaries)
LABEL_MERGE = {
    "B-NUMBER": "B-OTHER",
    "I-NUMBER": "I-OTHER",
}


def load_hiner(split: str = "train"):
    """
    Load HiNER from HuggingFace Hub JSON data files.
    """
    from datasets import load_dataset
    print(f"  Loading HiNER {split} split from HuggingFace...")
    # HF dropped support for loading scripts, so we load the raw json files directly
    url = f"https://huggingface.co/datasets/cfilt/HiNER-original/resolve/main/data/{split}.json"
    ds = load_dataset("json", data_files=url, split="train")
    return ds


def align_labels_to_subwords(
    tokenizer,
    tokens: List[str],
    labels: List[str],
    max_length: int = 256,
    label_all_subwords: bool = False,
) -> Dict:
    """
    Tokenise a list of word-level tokens and align word-level NER labels
    to the resulting subword token sequence.

    MuRIL and IndicBERT use SentencePiece, which splits words into subword
    units. A single word like "दिल्ली" may become ["▁दिल", "्ली"].
    The label for the word is assigned to the FIRST subword; all subsequent
    subwords of the same word get label -100 (ignored in loss computation).

    If label_all_subwords=True, I- labels propagate to all subwords.
    The literature shows label_all_subwords=False gives slightly better
    boundary detection for Indic scripts.

    Returns a dict with keys: input_ids, attention_mask, labels
    """
    tokenised = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    word_ids = tokenised.word_ids()
    aligned_labels = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # Special tokens [CLS], [SEP], [PAD] → -100
            aligned_labels.append(-100)
        elif word_id != prev_word_id:
            # First subword of a new word → assign the word's label
            raw_label = labels[word_id]
            merged    = LABEL_MERGE.get(raw_label, raw_label)
            aligned_labels.append(LABEL2ID[merged])
        else:
            # Continuation subword
            if label_all_subwords:
                raw_label = labels[word_id]
                merged    = LABEL_MERGE.get(raw_label, raw_label)
                # B- → I- for continuation
                if merged.startswith("B-"):
                    merged = "I-" + merged[2:]
                aligned_labels.append(LABEL2ID.get(merged, -100))
            else:
                aligned_labels.append(-100)
        prev_word_id = word_id

    tokenised["labels"] = aligned_labels
    return dict(tokenised)


def prepare(model_name: str, output_dir: str, max_length: int = 256):
    """Full preparation pipeline."""
    from transformers import AutoTokenizer

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading tokeniser: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save tokeniser alongside data for reproducibility
    tokenizer.save_pretrained(output_dir)

    # Save label maps
    with open(f"{output_dir}/label2id.json", "w") as f:
        json.dump(LABEL2ID, f, ensure_ascii=False, indent=2)
    with open(f"{output_dir}/id2label.json", "w") as f:
        json.dump(ID2LABEL, f, ensure_ascii=False, indent=2)

    print(f"Label set ({len(LABEL2ID)} labels): {list(LABEL2ID.keys())}")

    for split in ["train", "validation", "test"]:
        print(f"\nProcessing {split}...")
        ds = load_hiner(split)
        processed = []
        skipped   = 0

        for row in ds:
            tokens = row["tokens"]
            # HiNER stores integer tag ids; convert to label strings
            if isinstance(row["ner_tags"][0], int):
                # Map integer ids to label strings using the dataset's features
                # Since we load raw json, we hardcode the 23 original label names here
                original_labels = [
                    "O", "B-PERSON", "I-PERSON", "B-LOCATION", "I-LOCATION", 
                    "B-ORGANIZATION", "I-ORGANIZATION", "B-FESTIVAL", "I-FESTIVAL", 
                    "B-GAME", "I-GAME", "B-LANGUAGE", "I-LANGUAGE", "B-LITERATURE", 
                    "I-LITERATURE", "B-MISC", "I-MISC", "B-NUMEX", "I-NUMEX", 
                    "B-RELIGION", "I-RELIGION", "B-TIMEX", "I-TIMEX"
                ]
                # Map from original labels to our simplified label set
                label_map = {
                    "B-TIMEX": "B-TIME", "I-TIMEX": "I-TIME",
                    "B-NUMEX": "B-NUMBER", "I-NUMEX": "I-NUMBER",
                    "B-FESTIVAL": "B-OTHER", "I-FESTIVAL": "I-OTHER",
                    "B-GAME": "B-OTHER", "I-GAME": "I-OTHER",
                    "B-LANGUAGE": "B-OTHER", "I-LANGUAGE": "I-OTHER",
                    "B-LITERATURE": "B-OTHER", "I-LITERATURE": "I-OTHER",
                    "B-MISC": "B-OTHER", "I-MISC": "I-OTHER",
                    "B-RELIGION": "B-OTHER", "I-RELIGION": "I-OTHER"
                }
                
                str_tags = []
                for t in row["ner_tags"]:
                    lbl = original_labels[t]
                    str_tags.append(label_map.get(lbl, lbl))
            else:
                str_tags = row["ner_tags"]

            # Skip empty rows
            if not tokens:
                skipped += 1
                continue

            try:
                aligned = align_labels_to_subwords(
                    tokenizer, tokens, str_tags, max_length=max_length
                )
                processed.append(aligned)
            except Exception as e:
                skipped += 1

        print(f"  {len(processed)} examples processed, {skipped} skipped")

        # Save as JSON lines (easy to load in training script)
        out_path = f"{output_dir}/{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved → {out_path}")

    print("\nDataset preparation complete.")
    print(f"Next step: python3 train.py --data_dir {output_dir} --model_name {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/muril-base-cased",
                        help="HuggingFace model name for tokeniser. "
                             "Options: google/muril-base-cased, "
                             "ai4bharat/indic-bert, "
                             "xlm-roberta-base")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    prepare(args.model, args.output_dir, args.max_length)
