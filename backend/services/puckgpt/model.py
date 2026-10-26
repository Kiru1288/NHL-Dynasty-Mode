"""PuckLM intent model: multinomial naive Bayes over word and character n-grams.

Pure Python, trains on the generated corpus in about a second, and is built
lazily on the first PuckGPT question (never during the sim). The trained
tables are cached on disk keyed by a hash of the corpus.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import threading
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from . import corpus

_ALPHA = 0.35
_TEMPERATURE = 0.22
_LOCK = threading.Lock()
_MODEL: Optional["IntentModel"] = None
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def features(text: str) -> List[str]:
    words = text.split()
    feats = [f"w:{w}" for w in words]
    feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
    for w in words:
        if w.isupper():  # placeholders count as whole tokens only
            continue
        padded = f"<{w}>"
        for n in (3, 4):
            feats += [f"c:{padded[i:i + n]}" for i in range(len(padded) - n + 1)]
    return feats


class IntentModel:
    def __init__(self) -> None:
        self.priors: Dict[str, float] = {}
        self.loglik: Dict[str, Dict[str, float]] = {}
        self.unseen: Dict[str, float] = {}
        self.vocab: set = set()

    def fit(self, rows: List[Tuple[str, str]]) -> "IntentModel":
        counts: Dict[str, Counter] = defaultdict(Counter)
        docs: Counter = Counter()
        for text, label in rows:
            docs[label] += 1
            counts[label].update(features(text))
        self.vocab = set()
        for c in counts.values():
            self.vocab.update(c)
        v = len(self.vocab)
        total_docs = sum(docs.values())
        for label, c in counts.items():
            total = sum(c.values()) + _ALPHA * v
            self.priors[label] = math.log(docs[label] / total_docs)
            self.loglik[label] = {f: math.log((n + _ALPHA) / total) for f, n in c.items()}
            self.unseen[label] = math.log(_ALPHA / total)
        return self

    def predict(self, text: str, top: int = 4) -> Tuple[List[Tuple[str, float]], float]:
        """Ranked (intent, probability) and the share of features the model knows."""
        feats = [f for f in features(text) if f in self.vocab]
        known = len(feats) / max(1, len(features(text)))
        if not feats:
            return [("help", 0.0)], 0.0
        scores = {}
        for label, table in self.loglik.items():
            miss = self.unseen[label]
            scores[label] = self.priors[label] + sum(table.get(f, miss) for f in feats)
        # Per-feature average keeps long and short questions on one confidence scale.
        n = float(len(feats))
        best = max(scores.values())
        exp = {k: math.exp((s - best) / n / _TEMPERATURE) for k, s in scores.items()}
        z = sum(exp.values())
        ranked = sorted(((k, e / z) for k, e in exp.items()), key=lambda kv: -kv[1])
        return ranked[:top], known


def _corpus_key() -> str:
    raw = repr(sorted(corpus.INTENTS.items())) + repr(sorted(corpus.SWAPS.items())) + f"{_ALPHA}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def get_model() -> IntentModel:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        path = os.path.join(_CACHE_DIR, f"pucklm-{_corpus_key()}.pkl")
        model = None
        try:
            with open(path, "rb") as fh:
                model = pickle.load(fh)
        except Exception:
            model = None
        if not isinstance(model, IntentModel):
            model = IntentModel().fit(corpus.training_rows())
            try:
                os.makedirs(_CACHE_DIR, exist_ok=True)
                with open(path, "wb") as fh:
                    pickle.dump(model, fh)
            except Exception:
                pass
        _MODEL = model
        return model
