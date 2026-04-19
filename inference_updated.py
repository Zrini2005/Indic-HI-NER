"""
inference.py  (v2 — enhanced rule layer)
─────────────────────────────────────────
Production inference engine stacking MuRIL NER on top of the enhanced
rule-based post-correction layer.

Architecture
────────────
  MuRIL (primary)
      ↓ (per-token BIO tags + softmax confidence)
  EnhancedHindiRuleEngine   ← fires 75 linguistic rules
      ↓ (list of RuleMatch objects with confidence + evidence_type + priority)
  ConfidenceVoter           ← log-odds weighted combination
      ↓ (final BIO tags + blended confidence)
  Document memory           ← cross-sentence consistency boost
      ↓
  Final Entity list

Why log-odds voting?
  Each evidence source (neural softmax, rule match) is independently
  converted to log-odds = log(p/(1-p)).  Summing log-odds under conditional
  independence is equivalent to Naive Bayes — a principled way to combine
  heterogeneous signals without hand-tuned thresholds.  Source multipliers
  (regex=1.6, keyword=1.4, neural=1.0, postposition=0.7) encode our prior
  belief about each source's precision; priority multipliers (1.7/1.3/1.0)
  encode rule-specific confidence.

Usage
─────
  from inference import HindiNERInference
  model = HindiNERInference("models/muril-hiner/best")
  result = model.tag("श्री नरेंद्र मोदी ने नई दिल्ली में भाषण दिया।")
  print(result.pretty())

  # Debug mode — shows per-token voting evidence
  result = model.tag("...", debug=True)
  for dbg in result.vote_debug:
      print(dbg)
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# ── Enhanced rule engine (same directory or adjacent) ─────────────────────────
try:
    from rules_layerv2 import EnhancedHindiRuleEngine, ConfidenceVoter, RuleMatch, VotingDebug
    ENHANCED_RULES_AVAILABLE = True
except ImportError:
    ENHANCED_RULES_AVAILABLE = False
    print("Warning: enhanced_rules.py not found — rule post-correction disabled.")

# ── Legacy rule pipeline (v1 — optional, for gazetteer only) ─────────────────
RULE_PIPELINE_DIR = Path(__file__).parent.parent / "hindi_ner"
if RULE_PIPELINE_DIR.exists():
    sys.path.insert(0, str(RULE_PIPELINE_DIR))

try:
    from normalizer import Normalizer
    from morphology import MorphAnalyser
    from gazetteer  import Gazetteer
    LEGACY_RULES_AVAILABLE = True
except ImportError:
    LEGACY_RULES_AVAILABLE = False


# ── Label constants ────────────────────────────────────────────────────────────

HINER_LABELS = [
    "O",
    "B-PERSON",       "I-PERSON",
    "B-LOCATION",     "I-LOCATION",
    "B-ORGANIZATION", "I-ORGANIZATION",
    "B-DATE",         "I-DATE",
    "B-TIME",         "I-TIME",
    "B-NUMBER",       "I-NUMBER",
    "B-OTHER",        "I-OTHER",
]
LABEL2ID = {l: i for i, l in enumerate(HINER_LABELS)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}

# Map HiNER entity types to legacy rule engine types and back
HINER_TO_RULE = {
    "LOCATION": "LOC", "PERSON": "PERSON", "ORGANIZATION": "ORG",
    "DATE": "DATE",    "TIME": "TIME",     "NUMBER": "MEASURE", "OTHER": "OTHER",
}
RULE_TO_HINER = {v: k for k, v in HINER_TO_RULE.items()}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Entity:
    text:        str
    label:       str            # HiNER style: PERSON, LOCATION, …
    start_token: int
    end_token:   int            # exclusive
    confidence:  float
    source:      str            # 'neural' | 'rule' | 'neural+rule'
    tokens:      List[str] = field(default_factory=list)


@dataclass
class NERResult:
    text:        str
    tokens:      List[str]
    bio_tags:    List[str]
    confidences: List[float]
    entities:    List[Entity]
    vote_debug:  Optional[List["VotingDebug"]] = None   # populated when debug=True

    def pretty(self) -> str:
        lines = [f"Text: {self.text}", "Entities:"]
        for e in self.entities:
            lines.append(
                f"  [{e.label:<12}] \"{e.text}\"  conf={e.confidence:.2f}  src={e.source}"
            )
        return "\n".join(lines)

    def to_conll(self) -> str:
        return "\n".join(
            f"{tok}\t{tag}" for tok, tag in zip(self.tokens, self.bio_tags)
        )


# ── Inference engine ───────────────────────────────────────────────────────────

class HindiNERInference:
    """
    Full Hindi NER inference engine.

    Primary  : fine-tuned MuRIL (token classification)
    Secondary: EnhancedHindiRuleEngine (75 linguistic rules)
    Fusion   : ConfidenceVoter (log-odds weighted combination)
    Optional : legacy v1 Gazetteer (T1 exact-match override on top of voter)
    """

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        max_length: int = 256,
        hybrid_mode: str = "default",
        hybrid_gate_conf: float = 0.80,
    ) -> None:

        # ── Device ────────────────────────────────────────────────────────
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Neural model ──────────────────────────────────────────────────
        print(f"Loading MuRIL NER from {model_dir} on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # ── Enhanced rule engine ──────────────────────────────────────────
        if ENHANCED_RULES_AVAILABLE:
            self.rule_engine = EnhancedHindiRuleEngine()
            self.voter       = ConfidenceVoter()
            print("  Enhanced rule engine (75 rules) + log-odds voter: active")
        else:
            self.rule_engine = None
            self.voter       = None
            print("  Rule post-corrector: NOT available")

        # ── Legacy v1 components (gazetteer only) ─────────────────────────
        if LEGACY_RULES_AVAILABLE:
            self.normaliser = Normalizer()
            self.morph      = MorphAnalyser()
            self.gazetteer  = Gazetteer()
            print("  Legacy gazetteer (T1 exact-match): active")
        else:
            self.normaliser = None
            self.gazetteer  = None

        if hybrid_mode not in ("default", "conservative"):
            raise ValueError("hybrid_mode must be 'default' or 'conservative'")
        self.hybrid_mode = hybrid_mode
        self.hybrid_gate_conf = float(hybrid_gate_conf)

        self.max_length  = max_length
        self._doc_memory: Dict[str, str] = {}

        if self.hybrid_mode == "conservative":
            print(
                f"  Hybrid mode: conservative "
                f"(priority>=3, non-keyword, neural gate>={self.hybrid_gate_conf:.2f})"
            )

    def reset_document(self) -> None:
        self._doc_memory.clear()

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        if LEGACY_RULES_AVAILABLE and self.normaliser:
            return self.normaliser.tokenize(self.normaliser.normalize(text))
        return text.split()

    # ── Neural pass ───────────────────────────────────────────────────────────

    def _neural_tag(self, tokens: List[str]) -> Tuple[List[str], List[float]]:
        """
        Run MuRIL, return (bio_tags, confidences) aligned to input tokens.
        First-subword prediction strategy for multi-piece words.
        """
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids      = encoding["input_ids"].to(self.device),
                attention_mask = encoding["attention_mask"].to(self.device),
            )

        logits     = outputs.logits[0]
        probs      = torch.softmax(logits, dim=-1)
        pred_ids   = logits.argmax(dim=-1).cpu().tolist()
        pred_probs = probs.max(dim=-1).values.cpu().tolist()
        word_ids   = encoding.word_ids(batch_index=0)

        bio_tags:    List[str]   = []
        confidences: List[float] = []
        seen: set = set()

        for sub_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            bio_tags.append(ID2LABEL[pred_ids[sub_idx]])
            confidences.append(round(pred_probs[sub_idx], 4))

        return bio_tags, confidences

    # ── Gazetteer post-correction (legacy v1) ─────────────────────────────────

    def _apply_gazetteer(
        self,
        tokens: List[str],
        tags:   List[str],
        confs:  List[float],
    ) -> Tuple[List[str], List[float]]:
        """
        Apply T1 (tier-1, exact-match, conf=1.0) gazetteer overrides.
        Only fires when neural confidence is below 0.80 OR neural says O.
        This runs AFTER the ConfidenceVoter — it is the last resort.
        """
        if not (LEGACY_RULES_AVAILABLE and self.gazetteer and self.normaliser):
            return tags, confs

        norm_tokens = [self.normaliser.normalize(t) for t in tokens]
        morph_feats = self.morph.tokenize_with_features(norm_tokens)
        roots       = [f["root"] for f in morph_feats]
        gaz_matches = self.gazetteer.scan(norm_tokens, roots)

        for m in gaz_matches:
            if m.tier == 1 and m.confidence == 1.0:
                for i in range(m.start_token, min(m.end_token, len(tokens))):
                    if confs[i] < 0.80 or tags[i] == "O":
                        etype  = RULE_TO_HINER.get(m.entity_type, m.entity_type)
                        prefix = "B" if i == m.start_token else "I"
                        tags[i]  = f"{prefix}-{etype}"
                        confs[i] = m.confidence

        return tags, confs

    # ── Postposition trimming ────────────────────────────────────────────────-

    _POSTPOSITIONS = frozenset({
        "में", "से", "तक", "पर", "को", "का", "के", "की", "ने",
        "द्वारा", "पे", "तले", "हेतु", "साथ",
    })

    _LOCATION_TAIL_TOKENS = frozenset({
        "आश्रम", "मंदिर", "मस्जिद", "दरगाह", "गुरुद्वारा", "चर्च",
        "अड्डा", "स्टेशन", "बंदरगाह", "तट", "घाट", "किला", "महल",
    })

    def _trim_postpositions(
        self,
        tokens: List[str],
        tags:   List[str],
        confs:  List[float],
    ) -> Tuple[List[str], List[float]]:
        """
        If a pure postposition is tagged as any entity token (B- or I-),
        force it to O. This is a hard structural rule that is always applied,
        regardless of neural confidence.
        """
        for i, tok in enumerate(tokens):
            norm_tok = tok.strip().rstrip("।,!?:;")
            if norm_tok in self._POSTPOSITIONS and tags[i] != "O":
                tags[i]  = "O"
                confs[i] = 1.0
        return tags, confs

    # ── Entity extraction from BIO ────────────────────────────────────────────

    @staticmethod
    def _bio_to_entities(
        tokens: List[str],
        tags:   List[str],
        confs:  List[float],
    ) -> List[Entity]:
        entities: List[Entity] = []
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith("B-"):
                etype = tag[2:]
                start = i
                span_tokens = [tokens[i]]
                span_confs  = [confs[i]]
                i += 1
                while i < len(tags) and tags[i] == f"I-{etype}":
                    span_tokens.append(tokens[i])
                    span_confs.append(confs[i])
                    i += 1
                entity_text = " ".join(span_tokens).strip(" ,;:!?।()[]{}'\"")
                entities.append(Entity(
                    text        = entity_text,
                    label       = etype,
                    start_token = start,
                    end_token   = i,
                    confidence  = round(min(span_confs), 4),  # conservative
                    source      = "neural",
                    tokens      = span_tokens,
                ))
            else:
                i += 1
        return entities

    @classmethod
    def _merge_adjacent_location_tails(cls, entities: List[Entity]) -> List[Entity]:
        """
        Merge adjacent LOCATION spans when the right span is a location tail
        noun (e.g., "साबरमती" + "आश्रम" -> "साबरमती आश्रम").
        """
        if not entities:
            return entities

        merged: List[Entity] = [entities[0]]
        for cur in entities[1:]:
            prev = merged[-1]
            cur_tail = cur.text.strip().split()[-1].rstrip(" ,;:!?।()[]{}'\"")

            if (
                prev.label == "LOCATION"
                and cur.label == "LOCATION"
                and prev.end_token == cur.start_token
                and cur_tail in cls._LOCATION_TAIL_TOKENS
            ):
                prev.text = f"{prev.text} {cur.text}".strip()
                prev.end_token = cur.end_token
                prev.confidence = round(min(prev.confidence, cur.confidence), 4)
                prev.tokens = list(prev.tokens) + list(cur.tokens)
                if prev.source != cur.source:
                    prev.source = "neural+rule"
                continue

            merged.append(cur)

        return merged

    # ── Document memory ───────────────────────────────────────────────────────

    def _apply_doc_memory(self, entities: List[Entity]) -> List[Entity]:
        for e in entities:
            key = e.text.strip().lower()
            if key in self._doc_memory:
                known_type = self._doc_memory[key]
                if e.label != known_type and e.confidence < 0.75:
                    e.label      = known_type
                    e.confidence = min(e.confidence + 0.12, 0.88)
                    e.source     = "neural+rule"
        return entities

    def _update_doc_memory(self, entities: List[Entity]) -> None:
        for e in entities:
            if e.confidence >= 0.82:
                self._doc_memory[e.text.strip().lower()] = e.label

    # ── Full hybrid decode (neural + rules + voter) ───────────────────────────

    def _hybrid_decode(
        self,
        tokens:       List[str],
        neural_tags:  List[str],
        neural_confs: List[float],
        debug: bool = False,
    ) -> Tuple[List[str], List[float], Optional[List["VotingDebug"]]]:
        """
        Core fusion step.

        1. Apply 75-rule engine → get RuleMatch list
        2. Merge with neural predictions via ConfidenceVoter (log-odds vote)
        3. Hard-trim postpositions from inside entity spans
        4. Apply T1 gazetteer override (if legacy v1 available)

        Returns (final_tags, final_confs, debug_info).
        """
        if not (ENHANCED_RULES_AVAILABLE and self.rule_engine and self.voter):
            # Fall back: identity (neural only) with BIO repair
            return self._bio_repair(neural_tags), neural_confs, None

        # Step 1 – fire rules
        rule_matches: List[RuleMatch] = self.rule_engine.apply(tokens)

        # Optional conservative mode:
        # keep only strongest rules and suppress broad keyword-driven flips.
        if self.hybrid_mode == "conservative":
            rule_matches = [
                m for m in rule_matches
                if m.priority >= 3 and m.evidence_type != "keyword"
            ]

        # Step 2 – log-odds vote
        final_tags, final_confs, dbg = self.voter.vote(
            tokens, neural_tags, neural_confs, rule_matches, debug=debug
        )

        # Step 3 – hard postposition trim (structural rule, always wins)
        final_tags, final_confs = self._trim_postpositions(tokens, final_tags, final_confs)

        # Step 4 – legacy T1 gazetteer (last resort)
        final_tags, final_confs = self._apply_gazetteer(tokens, final_tags, final_confs)

        # Optional conservative safety gate:
        # if neural confidence is already high, keep neural prediction.
        if self.hybrid_mode == "conservative":
            for i in range(min(len(final_tags), len(neural_tags), len(neural_confs))):
                if neural_confs[i] >= self.hybrid_gate_conf:
                    final_tags[i] = neural_tags[i]
                    final_confs[i] = neural_confs[i]

        final_tags = self._bio_repair(final_tags)
        return final_tags, final_confs, dbg

    @staticmethod
    def _bio_repair(tags: List[str]) -> List[str]:
        """
        Repair BIO sequence consistency.

        1. If I-X follows O or a different type, convert it to B-X.
        2. If B-X directly follows B-X/I-X, convert it to I-X to avoid
           fragmenting contiguous same-type spans.
        """
        tags = list(tags)
        for i in range(1, len(tags)):
            cur = tags[i]
            prev = tags[i - 1]

            if cur.startswith("I-"):
                etype = cur[2:]
                if prev not in (f"B-{etype}", f"I-{etype}"):
                    tags[i] = f"B-{etype}"
                    cur = tags[i]

            if cur.startswith("B-"):
                etype = cur[2:]
                if etype in ("PERSON", "ORGANIZATION") and prev in (
                    f"B-{etype}",
                    f"I-{etype}",
                ):
                    tags[i] = f"I-{etype}"
        return tags

    # ── Public API ────────────────────────────────────────────────────────────

    def tag(self, text: str, debug: bool = False) -> NERResult:
        """
        Tag a single sentence.

        Parameters
        ----------
        text  : raw Hindi sentence
        debug : if True, VotingDebug objects are attached to NERResult
        """
        tokens = self._tokenize(text)
        if not tokens:
            return NERResult(text=text, tokens=[], bio_tags=[],
                             confidences=[], entities=[])

        # Neural pass
        neural_tags, neural_confs = self._neural_tag(tokens)

        # Align lengths (truncation)
        n            = min(len(tokens), len(neural_tags))
        tokens       = tokens[:n]
        neural_tags  = neural_tags[:n]
        neural_confs = neural_confs[:n]

        # Hybrid decode
        final_tags, final_confs, vote_dbg = self._hybrid_decode(
            tokens, neural_tags, neural_confs, debug=debug
        )

        # Entity extraction
        entities = self._bio_to_entities(tokens, final_tags, final_confs)
        entities = self._merge_adjacent_location_tails(entities)

        # Document memory
        entities = self._apply_doc_memory(entities)
        self._update_doc_memory(entities)

        return NERResult(
            text        = text,
            tokens      = tokens,
            bio_tags    = final_tags,
            confidences = final_confs,
            entities    = entities,
            vote_debug  = vote_dbg,
        )

    def tag_document(self, sentences: List[str], debug: bool = False) -> List[NERResult]:
        """Tag a list of sentences as a coherent document (shared memory)."""
        self.reset_document()
        return [self.tag(s, debug=debug) for s in sentences]

    def tag_batch(self, sentences: List[str]) -> List[NERResult]:
        """
        Faster batch inference — one forward pass for all sentences,
        then per-sentence rule correction + voting.
        """
        token_lists: List[List[str]] = [self._tokenize(s) for s in sentences]

        encodings = self.tokenizer(
            token_lists,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids      = encodings["input_ids"].to(self.device),
                attention_mask = encodings["attention_mask"].to(self.device),
            )

        logits_batch = outputs.logits
        probs_batch  = torch.softmax(logits_batch, dim=-1)
        results: List[NERResult] = []

        for batch_idx, tokens in enumerate(token_lists):
            word_ids   = encodings.word_ids(batch_index=batch_idx)
            pred_ids   = logits_batch[batch_idx].argmax(dim=-1).cpu().tolist()
            pred_probs = probs_batch[batch_idx].max(dim=-1).values.cpu().tolist()

            tags, confs = [], []
            seen: set = set()
            for sub_idx, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                if word_id < len(tokens):
                    tags.append(ID2LABEL[pred_ids[sub_idx]])
                    confs.append(round(pred_probs[sub_idx], 4))

            n      = min(len(tokens), len(tags))
            tokens = tokens[:n]
            tags   = tags[:n]
            confs  = confs[:n]

            final_tags, final_confs, _ = self._hybrid_decode(tokens, tags, confs)
            entities = self._bio_to_entities(tokens, final_tags, final_confs)
            entities = self._merge_adjacent_location_tails(entities)

            results.append(NERResult(
                text        = sentences[batch_idx],
                tokens      = tokens,
                bio_tags    = final_tags,
                confidences = final_confs,
                entities    = entities,
            ))

        return results


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/muril-hiner/best")
    parser.add_argument("--hybrid_mode", default="default", choices=["default", "conservative"])
    parser.add_argument("--hybrid_gate_conf", type=float, default=0.80)
    parser.add_argument("--text",      default=None)
    parser.add_argument("--debug",     action="store_true",
                        help="Print per-token voting evidence")
    args = parser.parse_args()

    model = HindiNERInference(
        args.model_dir,
        hybrid_mode=args.hybrid_mode,
        hybrid_gate_conf=args.hybrid_gate_conf,
    )

    demo_sentences = [
    "डॉ भीमराव आंबेडकर अंतरराष्ट्रीय हवाई अड्डा दिल्ली के पास स्थित है जहाँ प्रोफेसर राजेश कुमार ने भाषण दिया।",
    "इंदिरा गांधी राष्ट्रीय मुक्त विश्वविद्यालय के छात्र राहुल गांधी ने अमेठी में कार्यक्रम आयोजित किया।",
    "अटल बिहारी वाजपेयी भारतीय प्रौद्योगिकी संस्थान कानपुर के प्रोफेसर अटल बिहारी ने भोपाल में व्याख्यान दिया।",
    "चंद्रशेखर आज़ाद कृषि एवं प्रौद्योगिकी विश्वविद्यालय कानपुर के छात्र आज़ाद ने इलाहाबाद विश्वविद्यालय में प्रवेश लिया।",
    "नेहरू मेमोरियल म्यूज़ियम एंड लाइब्रेरी नई दिल्ली में स्थित है जहाँ डॉक्टर राकेश शर्मा ने व्याख्यान दिया।",
    "डॉ ए पी जे अब्दुल कलाम तकनीकी विश्वविद्यालय लखनऊ के वैज्ञानिक डॉक्टर अनिल कुमार ने शोध प्रस्तुत किया।",
    "भारतीय रिजर्व बैंक के पूर्व गवर्नर रघुराम राजन ने शिकागो विश्वविद्यालय में प्रोफेसर के रूप में कार्य किया।",
    "राजीव गांधी अंतरराष्ट्रीय हवाई अड्डा हैदराबाद में स्थित है जहाँ इंजीनियर गुप्ता ने प्रस्तुति दी।",
    "टाटा इंस्टिट्यूट ऑफ फंडामेंटल रिसर्च मुंबई में स्थित है जहाँ वैज्ञानिक सीमा वर्मा ने प्रस्तुति दी।",
    "सरदार वल्लभभाई पटेल राष्ट्रीय पुलिस अकादमी हैदराबाद के अधिकारी पटेल ने गुजरात में प्रशिक्षण दिया।",
    "डॉ राम मनोहर लोहिया राष्ट्रीय विधि विश्वविद्यालय लखनऊ के प्रोफेसर राम मनोहर ने दिल्ली विश्वविद्यालय में व्याख्यान दिया।",
    "मौलाना आज़ाद राष्ट्रीय प्रौद्योगिकी संस्थान भोपाल के प्रोफेसर आज़ाद ने दिल्ली के जामिया मिल्लिया इस्लामिया विश्वविद्यालय में व्याख्यान दिया।"
    ] if not args.text else [args.text]

    print("\n" + "=" * 70)
    for sent in demo_sentences:
        result = model.tag(sent, debug=args.debug)
        print(result.pretty())
        if args.debug and result.vote_debug:
            print("  [Vote detail]")
            for d in result.vote_debug:
                if d.final_label != "O":
                    src_str = " | ".join(d.evidence_sources[:4])
                    print(f"    {d.token:<16} → {d.final_label:<18} conf={d.final_conf:.2f}  [{src_str}]")
        print()