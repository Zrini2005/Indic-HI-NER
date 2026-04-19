# Indic Hindi NER — MuRIL + Rule Post-Correction

## Honest performance expectations

| System | Dataset | Entity F1 |
|--------|---------|-----------|
| Pure Rule-based Baseline | Our 20-sentence test | 100% (circular) |
| Pure Rule-based Baseline | HiNER test (real) | ~65–72% |
| MuRIL fine-tuned | HiNER test | ~87–90% |
| Hybrid (MuRIL + rules) | HiNER test | ~91–94% |
| XLM-RoBERTa-large fine-tuned | HiNER test | ~90–92% |
| Human agreement on HiNER | HiNER test | ~95% |

The 91–94% range is what you should realistically expect from this system on
well-formed Hindi news/government text. On social media or dialectal text,
expect 5–10% lower.

## System Architecture

This repository implements a hybrid approach: MuRIL acts as the primary neural system, and linguistic rules serve as a targeted post-correction layer.

MuRIL was pre-trained on 17 Indian languages including Hindi, giving it
rich subword representations for Devanagari. Fine-tuning on HiNER's 108,608
sentences teaches it entity patterns from diverse domains.

The rule correction layer adds value by:
1. Fixing postposition-inclusion errors (neural sometimes includes में inside span)
2. Enforcing near-perfect date/time/measure regex on cases where neural is uncertain
3. Applying T1 gazetteer overrides for known entities
4. BIO consistency repair (an I- after wrong context → correct to B-)
5. Document-level consistency memory across sentences

## File structure

```
Indic-HI-NER/
├── data/
│   ├── prepare_dataset.py   ← Step 1: download HiNER + tokenise
│   └── processed/           ← Created by prepare_dataset.py
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
├── models/
│   └── muril-hiner/
│       ├── best/            ← Best checkpoint by val F1
│       └── latest/          ← Most recent checkpoint
├── train.py                 ← Step 2: fine-tune MuRIL
├── evaluate.py              ← Step 3: entity-level metrics on test set
├── inference.py             ← Step 4: production inference
└── README.md
```

## Full runbook

### Prerequisites

```bash
pip install -r requirements.txt
```

### Quick Start: Download pre-trained model

If you don't want to train the model yourself, you can download the best pre-trained checkpoint directly from Google Drive:

```bash
# Create the directory structure where inference expects the model
mkdir -p models/muril-hiner/best

# Download the zip file from Google Drive using gdown
gdown --id 1i0CT464Q9tnhGi5iJoitg0LnpJgaYDpA -O muril-hiner-best.zip

# Extract the contents into the model directory
unzip muril-hiner-best.zip -d models/muril-hiner/best/

# Now you can directly run inference! (Skip to Step 4)
```

### Step 1 — Prepare HiNER dataset

```bash
# MuRIL (recommended, best accuracy):
python3 data/prepare_dataset.py --model google/muril-base-cased

# IndicBERT (smaller, faster inference):
python3 data/prepare_dataset.py --model ai4bharat/indic-bert

# XLM-RoBERTa (good multilingual baseline):
python3 data/prepare_dataset.py --model xlm-roberta-base
```

This downloads HiNER (108,608 sentences) from HuggingFace, tokenises with
SentencePiece, aligns word-level labels to subword positions, and saves
train/validation/test as .jsonl files. Takes ~15 minutes.

### Step 2 — Fine-tune

```bash
# With GPU (NVIDIA, 8GB+ VRAM):
python3 train.py \
  --model google/muril-base-cased \
  --epochs 5 \
  --batch_size 16 \
  --output_dir models/muril-hiner

# CPU only (slow, for testing):
python3 train.py \
  --model google/muril-base-cased \
  --epochs 3 \
  --batch_size 8 \
  --max_train_examples 5000 \
  --output_dir models/muril-hiner-small
```

**Expected training time:**
- GPU (A100): ~45 minutes for 5 epochs
- GPU (T4 / Colab free): ~3 hours
- CPU only: ~6–8 hours (not recommended for full data)

**Expected validation F1 by epoch:**
- Epoch 1: ~78–82%
- Epoch 2: ~83–86%
- Epoch 3: ~85–88%
- Epoch 4: ~87–89%
- Epoch 5: ~87–90% (best)

### Step 3 — Evaluate on test set

```bash
python3 evaluate.py \
  --model_dir models/muril-hiner/best \
  --error_analysis
```

This prints per-entity-type precision/recall/F1 using seqeval (exact span matching).

### Step 4 — Run inference

Make sure your model is downloaded/extracted inside `models/muril-hiner/best/` exactly as described in the "Quick Start" section above, or that you have successfully finished training it in Step 2.

```python
from inference_updated import HindiNERInference

model = HindiNERInference("models/muril-hiner/best")

# Single sentence
result = model.tag("श्री नरेंद्र मोदी ने नई दिल्ली में भाषण दिया।")
print(result.entities)
# [Entity(text='श्री नरेंद्र मोदी', label='PERSON', confidence=0.97, source='neural'),
#  Entity(text='नई दिल्ली',         label='LOCATION', confidence=0.99, source='neural')]

# Full document with consistency
model.reset_document()
results = model.tag_document(sentences)

# Fast batch inference for large corpora
results = model.tag_batch(sentences)  # one forward pass for all

# BIO/CoNLL format
print(result.to_conll())

# Pretty print
print(result.pretty())
```

## Model choice guide

| Model | Params | Hindi F1 | Inference speed | When to use |
|-------|--------|----------|-----------------|-------------|
| google/muril-base-cased | 236M | ~89% | ~50ms/sent GPU | Best accuracy, have GPU |
| ai4bharat/indic-bert | 12M | ~84% | ~8ms/sent CPU | Production CPU inference |
| xlm-roberta-base | 278M | ~87% | ~55ms/sent GPU | No GPU, cloud API |
| xlm-roberta-large | 560M | ~92% | ~120ms/sent GPU | Highest accuracy, A100 |

## Running on Google Colab (free GPU)

You can easily try this project and run inference directly in the browser using the free GPU on Google Colab:

```python
# In a Colab notebook:
!git clone https://github.com/Zrini2005/Indic-HI-NER.git
%cd Indic-HI-NER

# Install dependencies including gdown
!pip install -r requirements.txt

# Create the models folder and download the pretrained model from drive
!mkdir -p models/muril-hiner/best
!gdown --id 1i0CT464Q9tnhGi5iJoitg0LnpJgaYDpA -O muril-hiner-best.zip
!unzip muril-hiner-best.zip -d models/muril-hiner/best/

# Now run inference
from inference_updated import HindiNERInference
model = HindiNERInference("models/muril-hiner/best")
print(model.tag("भारतीय रिजर्व बैंक").entities)
```

If you prefer to train instead of downloading the model:
```python
!python3 data/prepare_dataset.py --model google/muril-base-cased
!python3 train.py --model google/muril-base-cased --epochs 5 --batch_size 16

# Download the trained model
from google.colab import files
import shutil
shutil.make_archive("muril-hiner-best-trained", "zip", "models/muril-hiner/best")
files.download("muril-hiner-best-trained.zip")
```

## Why not GPT-4 / Claude for Hindi NER?

LLM-based NER (prompting a large model) gives ~82–87% F1 on Hindi with few-shot
prompting — lower than a fine-tuned MuRIL. Reasons:

1. LLMs are not trained with structured BIO output objectives; their entity
   boundaries are imprecise.
2. Cost: 100k sentences × average 50 tokens = 5M tokens per inference run.
   Fine-tuned MuRIL is 1000x cheaper per sentence.
3. Latency: LLM API calls are 500ms–2s per sentence; MuRIL is 8–50ms.
4. Consistency: LLMs give slightly different outputs on repeated identical inputs.

The right architecture is fine-tuned task-specific model for production NER.
LLMs are useful for bootstrapping training data (generating pseudo-labels for
unlabelled text), not for production inference.

## Known limitations

1. **Domain shift**: HiNER is primarily news text. Performance on legal, medical,
   or social media text will be 5–15% lower. Fine-tune on in-domain data if
   this matters for your use case.

2. **Code-mixed (Hinglish)**: Roman-script Hindi is not covered. The model
   will ignore or misclassify Roman-script named entities. Add a transliteration
   step (indic-nlp-library) before inference for Hinglish text.

3. **Very long sentences**: truncated at 256 tokens. Entities near the end of
   long sentences may be missed. Increase --max_length at a memory cost.

4. **Nested entities**: "भारतीय रिजर्व बैंक के गवर्नर शक्तिकान्त दास" contains
   both an ORG and a PERSON. Flat NER (what we do) can only output one tag per
   token, so one entity wins. Nested NER requires a different architecture
   (span-based models like SpanBERT). This is a known limitation of all
   flat NER systems including the published HiNER baselines.
