"""Thin wrapper over spaCy for NER-driven enrichment.

The model is loaded lazily so the rest of the pipeline still runs
even when no spaCy model is installed locally.
"""
from __future__ import annotations

from typing import Dict, List

_NLP = None
_LOAD_FAILED = False


def _load():
    global _NLP, _LOAD_FAILED
    if _NLP is not None or _LOAD_FAILED:
        return _NLP
    try:
        import spacy  # type: ignore
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            # try blank model — at least tokenization works
            _NLP = spacy.blank("en")
    except Exception:
        _LOAD_FAILED = True
        _NLP = None
    return _NLP


def extract_entities(text: str, max_chars: int = 6000) -> Dict[str, List[str]]:
    """Return {PERSON: [...], ORG: [...], GPE: [...]} for a chunk of text."""
    out: Dict[str, List[str]] = {"PERSON": [], "ORG": [], "GPE": []}
    nlp = _load()
    if nlp is None:
        return out
    snippet = text[:max_chars]
    try:
        doc = nlp(snippet)
    except Exception:
        return out
    if not doc.has_annotation("ENT_IOB"):
        return out
    seen = {k: set() for k in out}
    for ent in doc.ents:
        label = ent.label_
        if label in out:
            val = ent.text.strip()
            if val and val.lower() not in seen[label]:
                seen[label].add(val.lower())
                out[label].append(val)
    return out


def is_available() -> bool:
    return _load() is not None
