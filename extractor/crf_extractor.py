"""CRF-based sequence labeling for journal-front-matter fields.

Labels per token (BIO-style):
  TITLE, AUTHOR, AFFIL, JOURNAL, ABSTRACT, KEYWORD, OTHER

Features include word shape, layout cues (font size, position),
and surrounding context. We train a small model from labeled examples
and fall back gracefully (heuristic-only) if the model file is missing.
"""
from __future__ import annotations

import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

try:
    import sklearn_crfsuite  # type: ignore
    HAS_CRF = True
except Exception:
    HAS_CRF = False

from .pdf_reader import PdfDocument, TextLine

LABELS = ["TITLE", "AUTHOR", "AFFIL", "JOURNAL", "ABSTRACT", "KEYWORD", "OTHER"]


def _word_shape(w: str) -> str:
    s = []
    for ch in w:
        if ch.isupper():
            s.append("X")
        elif ch.islower():
            s.append("x")
        elif ch.isdigit():
            s.append("d")
        else:
            s.append(ch)
    # collapse runs
    out = []
    prev = None
    for c in s:
        if c != prev:
            out.append(c)
        prev = c
    return "".join(out)


def line_to_tokens(line: TextLine) -> List[Tuple[str, Dict[str, float]]]:
    """Tokenize a line and attach per-token layout features."""
    tokens = re.findall(r"\S+", line.text)
    feats_common = {
        "size": line.size,
        "bold": 1.0 if line.is_bold else 0.0,
        "page": float(line.page),
        "y0": float(line.bbox[1]) if line.bbox else 0.0,
    }
    return [(t, feats_common) for t in tokens]


def token_features(
    tokens: List[Tuple[str, Dict[str, float]]],
    i: int,
    max_size: float,
) -> Dict[str, object]:
    word, layout = tokens[i]
    rel_size = layout["size"] / max_size if max_size else 0.0
    feats = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.has_at": "@" in word,
        "word.has_digit": any(c.isdigit() for c in word),
        "word.shape": _word_shape(word),
        "word.len": len(word),
        "layout.bold": layout["bold"],
        "layout.rel_size": rel_size,
        "layout.size": layout["size"],
        "layout.page": layout["page"],
        "layout.y0_norm": min(layout["y0"] / 1000.0, 1.0),
        "is_first_in_doc": i == 0,
    }
    if i > 0:
        prev_w = tokens[i - 1][0]
        feats.update({
            "-1:word.lower": prev_w.lower(),
            "-1:word.istitle": prev_w.istitle(),
            "-1:word.isupper": prev_w.isupper(),
        })
    else:
        feats["BOS"] = True
    if i < len(tokens) - 1:
        nxt = tokens[i + 1][0]
        feats.update({
            "+1:word.lower": nxt.lower(),
            "+1:word.istitle": nxt.istitle(),
            "+1:word.isupper": nxt.isupper(),
        })
    else:
        feats["EOS"] = True
    return feats


def doc_to_token_seq(doc: PdfDocument, max_lines: int = 60) -> List[Tuple[str, Dict[str, float]]]:
    """Flatten the first N lines of the document into a token sequence."""
    lines = [ln for ln in doc.lines if ln.page <= 1][:max_lines]
    tokens: List[Tuple[str, Dict[str, float]]] = []
    for ln in lines:
        tokens.extend(line_to_tokens(ln))
    return tokens


def featurize_sequence(tokens: List[Tuple[str, Dict[str, float]]]) -> List[Dict[str, object]]:
    if not tokens:
        return []
    max_size = max((t[1]["size"] for t in tokens), default=1.0) or 1.0
    return [token_features(tokens, i, max_size) for i in range(len(tokens))]


class CRFExtractor:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception:
                self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, doc: PdfDocument) -> Dict[str, List[str]]:
        """Return token spans grouped by label."""
        tokens = doc_to_token_seq(doc)
        if not tokens:
            return {label: [] for label in LABELS}

        if not self.is_loaded():
            return self._heuristic_predict(doc, tokens)

        features = featurize_sequence(tokens)
        try:
            preds = self.model.predict_single(features)
        except Exception:
            return self._heuristic_predict(doc, tokens)
        return self._group_predictions(tokens, preds)

    @staticmethod
    def _group_predictions(
        tokens: List[Tuple[str, Dict[str, float]]],
        preds: List[str],
    ) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {label: [] for label in LABELS}
        cur_label: Optional[str] = None
        buf: List[str] = []
        for (tok, _), label in zip(tokens, preds):
            if label != cur_label:
                if cur_label and cur_label != "OTHER" and buf:
                    groups[cur_label].append(" ".join(buf))
                buf = [tok]
                cur_label = label
            else:
                buf.append(tok)
        if cur_label and cur_label != "OTHER" and buf:
            groups[cur_label].append(" ".join(buf))
        return groups

    def _heuristic_predict(
        self,
        doc: PdfDocument,
        tokens: List[Tuple[str, Dict[str, float]]],
    ) -> Dict[str, List[str]]:
        """Layout-only fallback: largest-font line on page 0 = title;
        next bold/title-cased lines = authors; lines containing
        'University|Departement|Institute' = affiliations; line starting
        with a journal-ish header on top of page 0 = journal title.
        """
        groups: Dict[str, List[str]] = {label: [] for label in LABELS}
        page0 = [ln for ln in doc.lines if ln.page == 0]
        if not page0:
            return groups
        # journal title: top-most line, often italic/bold and title-cased
        top_line = min(page0, key=lambda l: l.bbox[1] if l.bbox else 0)
        if top_line.text and len(top_line.text) < 120:
            groups["JOURNAL"].append(top_line.text)

        # title: largest font in the upper half of page 0
        upper = [ln for ln in page0 if (ln.bbox[1] if ln.bbox else 0) < 500]
        if upper:
            biggest = max(upper, key=lambda l: l.size)
            title_lines = [l for l in upper if abs(l.size - biggest.size) < 0.5]
            title_text = " ".join(l.text for l in title_lines)
            if title_text:
                groups["TITLE"].append(title_text.strip())

            # authors: title-cased lines just below biggest
            biggest_y = biggest.bbox[1] if biggest.bbox else 0
            below = [l for l in upper if (l.bbox[1] if l.bbox else 0) > biggest_y]
            for l in below[:8]:
                t = l.text.strip(" ,;.")
                if not t:
                    continue
                if any(k in t.lower() for k in ["university", "universitas",
                                                "department", "departement",
                                                "institute", "institut",
                                                "faculty", "fakultas"]):
                    groups["AFFIL"].append(t)
                elif _looks_like_author(t):
                    groups["AUTHOR"].append(t)
        return groups


def _looks_like_author(line: str) -> bool:
    if not line or len(line) > 120:
        return False
    # remove superscript markers
    cleaned = re.sub(r"[\d\*†‡§¶,;]+", " ", line).strip()
    parts = [p for p in cleaned.split() if p]
    if len(parts) < 2 or len(parts) > 6:
        return False
    if not all(p[0].isupper() for p in parts if p[0].isalpha()):
        return False
    return True
