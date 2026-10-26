"""Shared text canonicalisation for training templates and live questions."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

from .lexicon import ABBREVIATIONS, POSITION_WORDS, STAT_WORDS

_PUNCT = re.compile(r"[^\w\s$%+/.#-]", re.UNICODE)


def fold(text: str) -> str:
    """Lowercase and strip Latin accents. Cyrillic letters are kept."""
    out = []
    for ch in unicodedata.normalize("NFKD", str(text or "").lower()):
        if unicodedata.combining(ch):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _clean_token(t: str) -> List[str]:
    if t.startswith("#") or t.startswith("$") or t in ABBREVIATIONS:
        return [t]  # "top6", "sv%", "#97" are shorthand, not leetspeak
    letters = sum(ch.isalpha() for ch in t)
    if letters >= 2 and re.search(r"[013457@]", t) and not re.fullmatch(r"\d+(st|nd|rd|th|m|k|yo)?", t):
        if re.fullmatch(r"[a-z]+\d+", t) and len(re.sub(r"[a-z]", "", t)) >= 2:
            return [re.sub(r"\d+$", "", t), re.sub(r"^[a-z]+", "", t)]  # "under25" -> under 25
        t = t.translate(_LEET)
    t = re.sub(r"([a-z])\1{2,}", r"\1\1", t)  # "longgggest" -> "longgest"
    return [t] if t else []


def tokens(text: str) -> List[str]:
    text = re.sub(r"[{}<>\[\]]", " ", fold(text))
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)  # "stay-at-home" -> "stay at home"
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"(?<=\D)\.|\.(?=\D)|\.$", " ", text)
    out: List[str] = []
    for t in text.split():
        out.extend(_clean_token(t))
    return [t for t in out if t]


def expand_abbreviations(words: List[str]) -> List[str]:
    out: List[str] = []
    for w in words:
        rep = ABBREVIATIONS.get(w)
        if rep is None:
            out.append(w)
        elif rep:
            out.extend(rep.split())
    return out


def _phrase_table(table: Dict[str, str]) -> List[Tuple[List[str], str]]:
    return sorted(((k.split(), v) for k, v in table.items()), key=lambda kv: -len(kv[0]))


_POS_PHRASES = _phrase_table(POSITION_WORDS)
_STAT_PHRASES = _phrase_table(STAT_WORDS)


def match_phrases(words: List[str], phrases: List[Tuple[List[str], str]]) -> List[Tuple[int, int, str]]:
    """Longest-first, non-overlapping phrase matches as (start, end, value)."""
    taken = [False] * len(words)
    found: List[Tuple[int, int, str]] = []
    for parts, value in phrases:
        n = len(parts)
        for i in range(len(words) - n + 1):
            if any(taken[i:i + n]):
                continue
            if words[i:i + n] == parts:
                found.append((i, i + n, value))
                for j in range(i, i + n):
                    taken[j] = True
    return sorted(found)


def replace_spans(words: List[str], spans: List[Tuple[int, int, str]], label: str) -> List[str]:
    out: List[str] = []
    i = 0
    for start, end, _value in spans:
        out.extend(words[i:start])
        out.append(label)
        i = end
    out.extend(words[i:])
    return out


def placeholderize(words: List[str]) -> List[str]:
    """Positions -> POS, stat words -> STAT, bare numbers -> NUM (placeholders stay)."""
    words = replace_spans(words, match_phrases(words, _POS_PHRASES), "POS")
    words = replace_spans(words, match_phrases(words, _STAT_PHRASES), "STAT")
    return ["NUM" if re.fullmatch(r"\$?\d+(\.\d+)?(m|k|%)?", w) else w for w in words]


def canonical_template(text: str) -> str:
    """Training-side canonicalisation (templates already use placeholders)."""
    words = []
    for t in text.split():
        words.append(t if t.isupper() else t.lower())
    return " ".join(placeholderize(expand_abbreviations(words)))
