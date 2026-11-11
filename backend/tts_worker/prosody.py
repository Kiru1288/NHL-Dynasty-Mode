"""Commentary prosody planning for the live-game TTS (stdlib only).

Shared by the FastAPI-side client (services/live_tts.py) and the worker
(tts_worker/worker.py) so that both compute the same speed for a line (the
speed is part of the clip cache key).

  speed_for(profile, hint, speaker)  -> float     profile/hint -> engine speed
  priority_for(profile, hint, text, lang, speaker) -> int   queue priority
  clean_text(text)                    -> str       TTS-safe text
  plan_segments(text, lang, profile, hint, speaker) -> [Segment]
  apply_pronunciations(text, lang)    -> [(kind, str)]  kind "text" | "ipa"
  pron_version()                      -> str       hash of the override table
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# speed

# profile -> (base, lo, hi). Kokoro speed > 1 = faster; Piper uses 1/speed.
PROFILE_SPEED: Dict[str, Tuple[float, float, float]] = {
    "hype": (1.20, 1.15, 1.25),
    "build": (1.10, 1.06, 1.14),
    "calm": (1.05, 1.02, 1.08),
    "somber": (0.95, 0.92, 0.98),
}
_DEFAULT_PROFILE = "calm"


def norm_profile(profile: Any) -> str:
    p = str(profile or "").strip().lower()
    return p if p in PROFILE_SPEED else _DEFAULT_PROFILE


def _hint(hint: Any) -> Tuple[float, str, int]:
    """(energy 0..1, pace slow|normal|fast, pause_before_ms) from a hint dict (or None)."""
    energy, pace, pause = 0.5, "normal", 0
    if isinstance(hint, dict):
        try:
            energy = float(hint.get("energy", 0.5))
        except (TypeError, ValueError):
            energy = 0.5
        pace = str(hint.get("pace") or "normal").lower()
        try:
            pause = int(hint.get("pause_before_ms") or 0)
        except (TypeError, ValueError):
            pause = 0
    energy = min(1.0, max(0.0, energy))
    if pace not in ("slow", "normal", "fast"):
        pace = "normal"
    return energy, pace, max(0, min(pause, 3000))


def speed_for(profile: Any, hint: Any = None, speaker: str = "pbp") -> float:
    """Engine speed for a line. Rounded to 0.01 so the cache key is stable."""
    p = norm_profile(profile)
    base, lo, hi = PROFILE_SPEED[p]
    energy, pace, _ = _hint(hint)
    s = base + (energy - 0.5) * 0.10          # +-0.05 from energy
    if pace == "fast":
        s += 0.03
    elif pace == "slow":
        s -= 0.04
    s = min(hi, max(lo, s))                   # stay inside the profile's band
    if speaker == "color":                    # the analyst talks a touch slower
        s -= 0.03
    return round(min(1.30, max(0.88, s)), 2)


def priority_for(profile: Any, hint: Any = None, text: str = "", lang: str = "en",
                 speaker: str = "pbp") -> int:
    """Higher = synthesised first, dropped last. Goal calls > hype > build > somber > calm > colour."""
    p = norm_profile(profile)
    energy, _, _ = _hint(hint)
    base = {"hype": 80, "build": 60, "somber": 50, "calm": 30}[p]
    if p == "hype" and detect_call(text, lang)[0]:
        base = 100
    if speaker == "color":
        base -= 5
    return int(base + round(energy * 10))


# --------------------------------------------------------------------------
# text cleanup

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF️‍]+")
_WS_RE = re.compile(r"\s+")


def clean_text(text: Any, limit: int = 600) -> str:
    s = unicodedata.normalize("NFC", str(text or ""))
    s = _EMOJI_RE.sub(" ", s)
    s = (s.replace("—", ", ").replace("–", ", ").replace(" - ", ", ")
          .replace("“", "").replace("”", "").replace("„", "")
          .replace("«", "").replace("»", "").replace('"', "")
          .replace("’", "'").replace("‘", "'").replace("…", "..."))
    s = re.sub(r"!{2,}", "!", s)
    s = re.sub(r"\?{2,}", "?", s)
    s = re.sub(r"\.{4,}", "...", s)
    s = re.sub(r"\s+([,;:])", r"\1", s)
    s = re.sub(r"([!?.])\s*,", r"\1", s)
    s = _WS_RE.sub(" ", s).strip()
    # Shouty words (GOAL, SCORES) read fine lower-cased; keep short acronyms (PP, NHL).
    s = re.sub(r"\b([A-ZÄÖÅÜÉÈ]{5,})\b", lambda m: m.group(1).capitalize(), s)
    return s[:limit]


# --------------------------------------------------------------------------
# goal-call detection and segment planning

CALL_WORDS: Dict[str, str] = {
    "en": r"(?:(?:he|she)\s+)?scores|goal|score|what a goal|it'?s in|in the net|he shoots,? he scores",
    "fr": r"but|il marque|elle marque|et le but|quel but",
    "es": r"g[oó]+l|anota|qu[eé] gol|golazo",
    "de": r"to+r|was f[üu]r ein tor",
    "sv": r"m[åa]+l|vilket m[åa]l",
    "it": r"go+l|rete",
    "pt": r"go+l|gola[cç]o",
}
_LEAD = r"^\W*(?:(?:and|oh|y|et|und|och|e)\W+)?"


def _split_sentences(text: str) -> List[str]:
    # split after . ! ? (also French " !" spacing); keep the punctuation
    parts = re.split(r"(?<=[.!?])\s+(?=[^\s])", text)
    return [p.strip() for p in parts if p.strip()]


def detect_call(text: str, lang: str = "en") -> Tuple[bool, str, str]:
    """(is_call, call_part, rest). A goal call is a short leading exclamation
    ('Scores!', 'He scores!', 'But !', '¡Gol!', 'Tor!', 'Mål!')."""
    t = (text or "").strip()
    sents = _split_sentences(t)
    if not sents:
        return False, "", t
    first = sents[0]
    if not first.endswith("!") or len(first.split()) > 4:
        return False, "", t
    pat = CALL_WORDS.get((lang or "en")[:2], CALL_WORDS["en"])
    if re.match(_LEAD + r"(?:" + pat + r")\W*$", first, flags=re.IGNORECASE):
        # absorb directly repeated calls ("Scores! Scores!")
        call = [first]
        for s in sents[1:]:
            if s.endswith("!") and len(s.split()) <= 3 and re.match(_LEAD + r"(?:" + pat + r")\W*$", s, re.I):
                call.append(s)
            else:
                break
        rest = " ".join(sents[len(call):])
        return True, " ".join(call), rest
    return False, "", t


@dataclass
class Segment:
    text: str
    kind: str              # "call" | "sentence" | "clause"
    speed: float           # engine speed for this segment
    gain_db: float = 0.0   # relative gain applied before loudness normalisation
    pitch_st: float = 0.0  # semitone lift (rendered by resampling; adds brightness)
    pause_after_ms: int = 0


def _pause_ms(kind: str, profile: str, pace: str, seed: int) -> int:
    base = {"call": 180, "sentence": 270, "clause": 150}[kind]
    mul = {"fast": 0.80, "normal": 1.0, "slow": 1.25}[pace]
    mul *= {"hype": 0.85, "build": 0.95, "calm": 1.0, "somber": 1.25}[profile]
    jitter = 1.0 + (((seed % 21) - 10) / 100.0)            # deterministic +-10 %
    return int(min(350, max(120, base * mul * jitter)))


def plan_segments(text: str, lang: str = "en", profile: Any = "calm", hint: Any = None,
                  speaker: str = "pbp", speed: Optional[float] = None) -> List[Segment]:
    """Split a line into synthesis segments with pauses, speed and gain.

    Sentences are synthesised separately (natural 120-350 ms gaps between them);
    commas inside a sentence are left to the model, except in long sentences
    (> 16 words) which are split at , ; : as well. A leading goal call is its own
    segment: faster, louder, slightly brighter.
    """
    prof = norm_profile(profile)
    energy, pace, _ = _hint(hint)
    s0 = float(speed) if speed else speed_for(prof, hint, speaker)
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    out: List[Segment] = []

    is_call, call, rest = detect_call(text, lang) if speaker == "pbp" else (False, "", text)
    if is_call and prof in ("hype", "build"):
        out.append(Segment(call, "call", speed=round(min(1.35, s0 + 0.08), 2),
                           gain_db=2.0 + 2.0 * energy, pitch_st=1.0 + 1.0 * energy,
                           pause_after_ms=_pause_ms("call", prof, pace, seed)))
    else:
        rest = text

    sentences = _split_sentences(rest)
    for i, sent in enumerate(sentences):
        if prof == "hype":
            sp = max(PROFILE_SPEED["build"][0], s0 - 0.02 * i)     # excitement decays
        elif prof == "build":
            sp = min(PROFILE_SPEED["build"][2] + 0.03, s0 + 0.015 * i)  # tension rises
        else:
            sp = s0
        clauses = [sent]
        if len(sent.split()) > 16:
            clauses = [c.strip() for c in re.split(r"(?<=[,;:])\s+", sent) if c.strip()]
        for j, cl in enumerate(clauses):
            last_clause = j == len(clauses) - 1
            kind = "sentence" if last_clause else "clause"
            out.append(Segment(cl, kind, speed=round(sp, 2),
                               pause_after_ms=_pause_ms(kind, prof, pace, seed + i * 7 + j)))
    if out:
        out[-1].pause_after_ms = 0
    return out


# --------------------------------------------------------------------------
# pronunciation overrides (C10)

_HERE = os.path.dirname(os.path.abspath(__file__))
PRON_FILE = os.environ.get("NHL_TTS_PRONUNCIATIONS") or os.path.join(_HERE, "pronunciations.json")
_pron_lock = threading.Lock()
_pron: Dict[str, Any] = {"mtime": None, "table": {}, "version": "0"}

# Character fallbacks for *English* commentary when a name has letters the English
# phonemiser guesses badly. Only applied inside words that contain such letters.
_EN_CHAR_FALLBACK = [
    ("ö", "er"), ("ø", "er"), ("ä", "eh"), ("æ", "eh"), ("å", "oh"), ("ü", "ew"),
    ("č", "ch"), ("ć", "ch"), ("š", "sh"), ("ś", "sh"), ("ž", "zh"), ("ż", "zh"), ("ř", "rzh"),
    ("ň", "n"), ("ń", "n"), ("ľ", "l"), ("ł", "w"), ("ď", "d"), ("ť", "t"), ("ě", "yeh"),
    ("é", "ay"), ("è", "eh"), ("ê", "eh"), ("ë", "eh"), ("á", "ah"), ("à", "ah"), ("â", "ah"),
    ("í", "ee"), ("ï", "ee"), ("î", "ee"), ("ó", "oh"), ("ô", "oh"), ("ú", "oo"), ("ů", "oo"),
    ("ý", "ee"), ("ç", "s"),
]
_DIACRITIC_WORD = re.compile(r"\b\w*[öøäæåüčćšśžżřňńľłďťěéèêëáàâíïîóôúůýç]\w*\b", re.IGNORECASE)


def _load_pron() -> Dict[str, Any]:
    with _pron_lock:
        try:
            mtime = os.path.getmtime(PRON_FILE)
        except OSError:
            mtime = None
        if mtime != _pron["mtime"]:
            table: Dict[str, Any] = {}
            raw = b""
            if mtime is not None:
                try:
                    with open(PRON_FILE, "rb") as fh:
                        raw = fh.read()
                    data = json.loads(raw.decode("utf-8"))
                    table = {k: v for k, v in data.items() if not k.startswith("_")}
                except Exception:
                    table = {}
            _pron.update(mtime=mtime, table=table,
                         version=hashlib.sha1(raw).hexdigest()[:8] if raw else "0")
        return _pron


def pron_version() -> str:
    return _load_pron()["version"]


def _entry_for(entry: Any, lang: str) -> Tuple[Optional[str], Optional[str]]:
    """(say, ipa) for one override entry; entries may be a string (English respelling),
    or {"say": .., "ipa": .., "langs": [..]} or {"en": {...}, "fr": {...}}."""
    if isinstance(entry, str):
        return (entry, None) if lang == "en" else (None, None)
    if not isinstance(entry, dict):
        return None, None
    if lang in entry and isinstance(entry[lang], (dict, str)):
        return _entry_for(entry[lang], lang) if isinstance(entry[lang], dict) else (entry[lang], None)
    langs = entry.get("langs") or ["en"]
    if lang not in langs:
        return None, None
    return entry.get("say"), entry.get("ipa")


def apply_pronunciations(text: str, lang: str = "en", allow_ipa: bool = True) -> List[Tuple[str, str]]:
    """Apply the override table. Returns parts [("text", s) | ("ipa", s)].
    Engines that cannot take phonemes get the "say" respelling instead of IPA."""
    lang = (lang or "en")[:2]
    table = _load_pron()["table"]
    parts: List[Tuple[str, str]] = [("text", text)]
    if table:
        names = sorted(table.keys(), key=len, reverse=True)
        rx = re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in names) + r")(?:'s)?(?!\w)")
        new_parts: List[Tuple[str, str]] = []
        pos = 0
        for m in rx.finditer(text):
            name = m.group(1)
            say, ipa = _entry_for(table.get(name), lang)
            if not say and not ipa:
                continue
            new_parts.append(("text", text[pos:m.start()]))
            poss = m.group(0)[len(name):]               # "'s"
            if ipa and allow_ipa:
                new_parts.append(("ipa", ipa + ("z" if poss else "")))
            else:
                new_parts.append(("text", (say or name) + poss))
            pos = m.end()
        new_parts.append(("text", text[pos:]))
        parts = new_parts
    if lang == "en":
        def fb(m: "re.Match[str]") -> str:
            w = m.group(0)
            out = w
            for a, b in _EN_CHAR_FALLBACK:
                out = out.replace(a, b).replace(a.upper(), b.capitalize())
            return out
        parts = [(k, _DIACRITIC_WORD.sub(fb, s) if k == "text" else s) for k, s in parts]
    # merge adjacent text parts and drop empties
    merged: List[Tuple[str, str]] = []
    for k, s in parts:
        if not s:
            continue
        if merged and merged[-1][0] == k == "text":
            merged[-1] = ("text", merged[-1][1] + s)
        else:
            merged.append((k, s))
    return merged or [("text", text)]


def parts_to_text(parts: List[Tuple[str, str]]) -> str:
    return "".join(s for _, s in parts)
