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
        """Layout-only fallback for journals (incl. Indonesian).

        Strategy:
          1. Drop journal-header lines (ISSN / Vol / Tahun / page-number-only).
          2. Title = consecutive lines sharing the largest non-header font.
          3. Journal title = lines above ISSN/Vol header, joined.
          4. After title:
               - lines that look like names (with optional superscript) → AUTHOR
               - first line matching Fakultas/Universitas/Department/Jl./...
                 starts the AFFIL block; AFFIL continues until email/abstract.
        """
        groups: Dict[str, List[str]] = {label: [] for label in LABELS}
        page0 = [ln for ln in doc.lines if ln.page == 0]
        if not page0:
            return groups

        page0 = sorted(page0, key=lambda l: (l.bbox[1] if l.bbox else 0))

        header_re = re.compile(
            r"\b(?:ISSN|P[\-\s]?ISSN|E[\-\s]?ISSN|Vol(?:ume)?\.?\s*\d|"
            r"No\.?\s*\d|Tahun\s*\d{4}|Year\s*\d{4})\b",
            re.IGNORECASE,
        )
        affil_re = re.compile(
            r"\b(Fakultas|Universitas|Department|Departement|University|"
            r"Institute|Institut|Faculty|Sekolah\s+Tinggi|Politeknik|Akademi|"
            r"Jl\.|Jalan\s|Kampus)\b",
            re.IGNORECASE,
        )
        abstract_re = re.compile(r"^\s*(abstract|abstrak)\b", re.IGNORECASE)
        keyword_re = re.compile(r"^\s*(keywords?|kata\s+kunci)\b", re.IGNORECASE)

        def is_header(text: str) -> bool:
            t = text.strip()
            if not t:
                return True
            if t.isdigit() and len(t) <= 4:
                return True
            return bool(header_re.search(t))

        body = [ln for ln in page0 if not is_header(ln.text)]
        if not body:
            return groups

        # Journal title: lines that appear ABOVE the first header line
        first_header_y = None
        for ln in page0:
            if header_re.search(ln.text):
                first_header_y = ln.bbox[1] if ln.bbox else 0
                break
        if first_header_y is not None:
            above = [ln for ln in body if (ln.bbox[1] or 0) < first_header_y]
            if above:
                jt = " ".join(l.text for l in above).strip()
                jt = re.sub(r"\s+", " ", jt)
                if 3 < len(jt) < 200:
                    groups["JOURNAL"].append(jt)

        # Title: consecutive lines sharing the largest font in body
        max_size = max(ln.size for ln in body)
        title_lines = [ln for ln in body if abs(ln.size - max_size) < 0.4]
        # keep only the first contiguous run of same-size lines (by y order)
        if title_lines:
            ordered = sorted(title_lines, key=lambda l: l.bbox[1] or 0)
            run = [ordered[0]]
            for prev, cur in zip(ordered, ordered[1:]):
                gap = (cur.bbox[1] or 0) - (prev.bbox[3] or prev.bbox[1] or 0)
                if gap < (cur.size or 12) * 2.5:
                    run.append(cur)
                else:
                    break
            title_text = " ".join(l.text for l in run).strip()
            title_text = re.sub(r"\s+", " ", title_text)
            if title_text:
                groups["TITLE"].append(title_text)
            title_end_y = max((l.bbox[3] or l.bbox[1] or 0) for l in run)
        else:
            title_end_y = 0

        # Below the title: authors → affiliations → abstract
        below = [ln for ln in body if (ln.bbox[1] or 0) > title_end_y]
        author_lines: List[str] = []
        affil_lines: List[str] = []
        in_affil = False
        for ln in below:
            t = ln.text.strip()
            if not t:
                continue
            if abstract_re.match(t) or keyword_re.match(t):
                break
            if "@" in t:  # email block — stop both
                break
            if affil_re.search(t):
                in_affil = True
            if in_affil:
                affil_lines.append(t)
                continue
            if _looks_like_author(t):
                author_lines.append(t)
            else:
                # if we already saw authors and this is unusual, stop
                if author_lines:
                    break

        if author_lines:
            groups["AUTHOR"].extend(author_lines)
        if affil_lines:
            groups["AFFIL"].append(" ".join(affil_lines))

        return groups


def _looks_like_author(line: str) -> bool:
    """Heuristic: line is one or more title-cased name groups separated by
    commas / 'and' / '&', possibly with superscript digit/symbol markers.
    """
    if not line or len(line) > 240:
        return False
    if any(kw in line.lower() for kw in [
        "abstract", "abstrak", "keyword", "kata kunci",
        "introduction", "pendahuluan", "fakultas", "universitas",
        "university", "department", "departement", "faculty",
        "institute", "institut", "jl.", "jalan ", "@",
    ]):
        return False
    # Strip superscript markers and split into name groups
    stripped = re.sub(r"[\d\*†‡§¶]+", " ", line)
    groups = re.split(r",|\band\b|;|&", stripped, flags=re.IGNORECASE)
    valid = 0
    for g in groups:
        parts = [p for p in g.strip().split() if p]
        if 2 <= len(parts) <= 5 and all(
            p[0].isupper() for p in parts if p[0].isalpha()
        ):
            valid += 1
    return valid >= 1
