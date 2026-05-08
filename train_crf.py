"""Train a small CRF model from labeled samples and pickle it.

Usage:
    python train_crf.py

Reads data/training/samples.json (each sample has aligned tokens, labels,
and a layout block), builds features, fits sklearn-crfsuite, and writes
models/crf_model.pkl.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List

import sklearn_crfsuite

from extractor.crf_extractor import featurize_sequence

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES_PATH = os.path.join(ROOT, "data", "training", "samples.json")
MODEL_PATH = os.path.join(ROOT, "models", "crf_model.pkl")


def sample_to_tokens(sample: Dict[str, Any]):
    tokens = sample["tokens"]
    sizes = sample.get("layout", {}).get("size_per_token") or [12.0] * len(tokens)
    pairs = []
    for tok, sz in zip(tokens, sizes):
        layout = {"size": float(sz), "bold": 0.0, "page": 0.0, "y0": 100.0}
        pairs.append((tok, layout))
    return pairs


def main():
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    X: List[List[Dict[str, Any]]] = []
    y: List[List[str]] = []
    for s in samples:
        token_pairs = sample_to_tokens(s)
        X.append(featurize_sequence(token_pairs))
        y.append(s["labels"])

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=200,
        all_possible_transitions=True,
    )
    crf.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(crf, f)
    print(f"Saved CRF model to {MODEL_PATH}")
    print(f"Trained on {len(samples)} samples, labels: {sorted(crf.classes_)}")


if __name__ == "__main__":
    main()
