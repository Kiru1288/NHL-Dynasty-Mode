"""PuckLM: turn a messy question into a structured query.

Pipeline (each step is cheap and deterministic):
  1. fold + tokenise
  2. detect language, translate words to English
  3. expand hockey shorthand and slang
  4. fix typos against the game's vocabulary (including player and team names)
  5. extract entities and slots (players, teams, positions, stats, numbers, scope)
  6. resolve references from the conversation ("his", "which one", "what about X")
  7. score intents with the model and assign a confidence band
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import corpus
from .canon import expand_abbreviations, fold, match_phrases, placeholderize, tokens, _phrase_table
from .lexicon import (
    COMMON_WORDS,
    ABBREVIATIONS, DEPTH_WORDS, FOREIGN, LANG_MARKERS, POSITION_WORDS, PRONOUNS_LIST, PRONOUNS_ONE,
    ROLE_WORDS, SELF_TEAM, STAT_WORDS, TEAM_NICKNAMES,
)
from .model import get_model

HIGH = 0.55
MEDIUM = 0.30

_FOREIGN = {lang: {fold(k): v for k, v in table.items()} for lang, table in FOREIGN.items()}
_ROLE_PHRASES = _phrase_table(ROLE_WORDS)
_DEPTH_PHRASES = _phrase_table(DEPTH_WORDS)
_POS_PHRASES = _phrase_table(POSITION_WORDS)
_STAT_PHRASES = _phrase_table(STAT_WORDS)

_CAREER = {"ever", "career", "franchise", "history", "all", "record", "records", "alltime", "greatest"}
_LAST_SEASON = ("last season", "last year", "previous season")


# --------------------------------------------------------------------------- world index
@dataclass
class Person:
    id: str
    name: str
    team_id: str
    pos: str
    number: Optional[int]
    first: str
    last: str


@dataclass
class World:
    """Names the matcher knows. Built from the live save by ``data.world_for``."""
    people: List[Person] = field(default_factory=list)
    teams: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # tid -> {abbr, name, city}
    user_team_id: str = ""
    by_full: Dict[str, List[Person]] = field(default_factory=dict)
    by_last: Dict[str, List[Person]] = field(default_factory=dict)
    by_first: Dict[str, List[Person]] = field(default_factory=dict)
    by_joined: Dict[str, List[Person]] = field(default_factory=dict)
    team_alias: Dict[str, str] = field(default_factory=dict)
    vocab: set = field(default_factory=set)
    by_skel: Dict[str, List[Person]] = field(default_factory=dict)

    def index(self) -> "World":
        for p in self.people:
            full = " ".join(tokens(p.name))
            parts = full.split()
            if not parts:
                continue
            p.first, p.last = parts[0], parts[-1]
            self.by_full.setdefault(full, []).append(p)
            self.by_last.setdefault(p.last, []).append(p)
            self.by_first.setdefault(p.first, []).append(p)
            self.by_joined.setdefault("".join(parts), []).append(p)
            self.by_joined.setdefault("".join(parts[1:]), []).append(p)
            if len(p.last) >= 5:
                self.by_skel.setdefault(skeleton(p.last), []).append(p)
            self.vocab.update(parts)
        for tid, t in self.teams.items():
            names = {t.get("abbr", ""), t.get("name", ""), t.get("city", ""), t.get("nickname", "")}
            names.update(TEAM_NICKNAMES.get(str(t.get("abbr", "")).upper(), []))
            for n in names:
                key = " ".join(tokens(n))
                if key:
                    self.team_alias[key] = tid
                    self.vocab.update(key.split())
                    if key.endswith("s") and len(key) >= 4 and " " not in key:
                        self.team_alias.setdefault(key[:-1], tid)  # "leaf", "hab", "oiler"
                        self.vocab.add(key[:-1])
        return self


def skeleton(word: str) -> str:
    """Consonants only: texting drops vowels ("mtthws", "mcdvd")."""
    return re.sub(r"[aeiouy]", "", word)


# --------------------------------------------------------------------------- language
def detect_language(words: List[str]) -> str:
    scores = {}
    for lang, table in _FOREIGN.items():
        hits = sum(1 for w in words if w in table)
        markers = sum(2 for w in words if w in LANG_MARKERS.get(lang, ()))
        scores[lang] = hits + markers
    if any(re.search("[а-я]", w) for w in words):
        return "ru"
    lang, score = max(scores.items(), key=lambda kv: kv[1])
    english = sum(1 for w in words if w in _english_vocab())
    if score >= 3 and score > english:
        return lang
    return "en"


def translate(words: List[str], lang: str) -> List[str]:
    if lang == "en":
        return words
    table = _FOREIGN.get(lang, {})
    out: List[str] = []
    i = 0
    while i < len(words):
        pair = " ".join(words[i:i + 2])
        if pair in table:
            out.extend(table[pair].split())
            i += 2
            continue
        rep = table.get(words[i])
        if rep is None:
            out.append(words[i])
        elif rep:
            out.extend(rep.split())
        i += 1
    return out


# --------------------------------------------------------------------------- typo correction
_EN_VOCAB: Optional[set] = None


def _english_vocab() -> set:
    global _EN_VOCAB
    if _EN_VOCAB is None:
        v = set(corpus.vocabulary())
        for table in (ABBREVIATIONS, POSITION_WORDS, ROLE_WORDS, DEPTH_WORDS, STAT_WORDS):
            for k, val in table.items():
                v.update(k.split())
                v.update(str(val).lower().split())
        v.update(SELF_TEAM)
        v.update(COMMON_WORDS)
        try:
            from .metrics import PHRASES

            for ph in PHRASES:
                v.update(w for w in ph.split() if w.isalpha())
        except Exception:
            pass
        v.update({"stats", "stat", "goat", "hot", "cold", "selling", "average", "league", "bottom", "who", "the"})
        v.update({"under", "over", "older", "younger", "than", "since", "ago", "years", "old", "million",
                  "first", "second", "third", "fourth", "round", "pick", "overall", "potential", "career",
                  "ever", "record", "worst", "least", "fewest", "defensively", "offensively", "about",
                  "what", "and", "or", "the", "a", "an", "in", "on", "of", "for", "to", "with", "is", "are",
                  "his", "him", "he", "them", "they", "which", "one", "those", "these", "why", "how"})
        _EN_VOCAB = v
    return _EN_VOCAB


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Damerau-Levenshtein (adjacent swaps), early exit above ``cap``."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2 = None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


class Speller:
    """Symmetric-delete lookup: candidates share a 1-deletion form, then edit distance ranks."""

    def __init__(self, vocab: Iterable[str], names: Iterable[str] = (), priority: Iterable[str] = ()) -> None:
        self.vocab = {w for w in vocab if w and len(w) >= 2}
        self.priority = set(priority)
        # Name-only words: a plain English word must not be "corrected" into a surname.
        self.names = set(names)
        self.deletes: Dict[str, set] = {}
        for w in self.vocab:
            for d in self._dels(w):
                self.deletes.setdefault(d, set()).add(w)

    @staticmethod
    def _dels(w: str) -> set:
        out = {w}
        for i in range(len(w)):
            d = w[:i] + w[i + 1:]
            out.add(d)
            if len(w) >= 6:
                for j in range(len(d)):
                    out.add(d[:j] + d[j + 1:])
        return out

    def correct(self, word: str) -> str:
        if word in self.vocab or len(word) < 3 or not word.isalpha():
            return word
        cap = 1 if len(word) <= 5 else 2
        cands = set()
        for d in self._dels(word):
            cands |= self.deletes.get(d, set())
        best, best_d = word, cap + 1
        for c in sorted(cands):
            dist = _edit_distance(word, c, cap)
            # Real typos keep the first letter (or swap the first two: "hwo").
            # Anything else is more likely a real word the vocabulary lacks.
            first_ok = c[0] == word[0] or (len(word) >= 2 and len(c) >= 2 and c[0] == word[1] and c[1] == word[0])
            if not first_ok:
                continue
            if dist == 2 and len(word) < 7:
                continue
            if c in self.names and dist > (2 if len(word) >= 8 else 1):
                continue
            if len(word) < 4 and c in self.names:
                continue  # "dos" is not Das
            def _tie_rank(x: str) -> tuple:
                # Swapped letters, then a missing letter, are the commonest typos.
                return (sorted(x) == sorted(word), len(x) > len(word), x in self.priority and len(word) >= 4)

            better_tie = dist == best_d and best != word and _tie_rank(c) > _tie_rank(best)
            better_tie = better_tie or dist == best_d and best == word and (
                (len(word) >= 4 and c in self.priority and best not in self.priority)
                or (len(c) == len(word) and len(best) != len(word) and not (best in self.priority and c not in self.priority))
            )
            if dist < best_d or better_tie:
                best, best_d = c, dist
        return best if best_d <= cap else word


# --------------------------------------------------------------------------- interpretation
@dataclass
class Interpretation:
    raw: str
    language: str
    rewritten: str
    intent: str
    confidence: float
    band: str
    slots: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    assumptions: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent, "confidence": round(self.confidence, 3), "band": self.band,
            "language": self.language, "rewritten": self.rewritten, "slots": self.slots,
            "alternatives": self.alternatives, "assumptions": self.assumptions,
        }


def _match_people(words: List[str], world: World, original_case: List[str]) -> List[Tuple[int, int, List[Person]]]:
    """Find player mentions. Common words only count as a name next to another name part."""
    en = _english_vocab()
    found: List[Tuple[int, int, List[Person]]] = []
    i = 0
    while i < len(words):
        hit = None
        for n in (3, 2):
            span = words[i:i + n]
            if len(span) < n:
                continue
            key = " ".join(span)
            if key in world.by_full:
                hit = (i, i + n, world.by_full[key])
                break
            joined = "".join(span)
            if joined in world.by_joined and not all(w in en for w in span):
                hit = (i, i + n, world.by_joined[joined])
                break
            # "c mcd" / "c mcdavid": initial + last-name prefix
            if n == 2 and len(span[0]) == 1 and len(span[1]) >= 3:
                cands = [p for last, ps in world.by_last.items() if last.startswith(span[1]) for p in ps
                         if p.first.startswith(span[0])]
                if cands:
                    hit = (i, i + 2, cands)
                    break
        if hit is None:
            w = words[i]
            common = w in en or w in SELF_TEAM
            if w in world.by_last and (not common or (original_case[i:i + 1] and original_case[i][:1].isupper())):
                hit = (i, i + 1, world.by_last[w])
            elif w in world.by_joined and not common and len(w) >= 5:
                hit = (i, i + 1, world.by_joined[w])
            elif w in world.by_first and not common and len(world.by_first[w]) <= 3 and len(w) >= 4:
                hit = (i, i + 1, world.by_first[w])
            elif not common and len(w) >= 4 and w == skeleton(w) and w in world.by_skel:
                hit = (i, i + 1, world.by_skel[w])
            elif not common and len(w) >= 5 and skeleton(w) in world.by_skel and len(skeleton(w)) >= 4 \
                    and all(p.last[0] == w[0] for p in world.by_skel[skeleton(w)]):
                hit = (i, i + 1, world.by_skel[skeleton(w)])
            elif w.startswith("#") and w[1:].isdigit():
                num = int(w[1:])
                cands = [p for p in world.people if p.number == num]
                if cands:
                    hit = (i, i + 1, cands)
        if hit:
            found.append(hit)
            i = hit[1]
        else:
            i += 1
    return found


def _match_teams(words: List[str], world: World) -> List[Tuple[int, int, str]]:
    found: List[Tuple[int, int, str]] = []
    i = 0
    en = _english_vocab()
    while i < len(words):
        hit = None
        for n in (3, 2, 1):
            key = " ".join(words[i:i + n])
            if len(words[i:i + n]) < n:
                continue
            tid = world.team_alias.get(key) or world.team_alias.get(key + "s")
            if tid and not (n == 1 and key in en and key not in ("wild", "kings", "stars", "jets", "blues")):
                hit = (i, i + n, tid)
                break
        if hit:
            found.append(hit)
            i = hit[1]
        else:
            i += 1
    return found


def _numbers(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m = re.search(r"\b(under|younger than|below|less than|max)\s+(\d{2})\b(?!\s*(overall|ovr|million|m\b))", text)
    if m:
        out["age_max"] = int(m.group(2)) - (1 if m.group(1) in ("under", "younger than", "below", "less than") else 0)
    m = re.search(r"\b(over|older than|above)\s+(\d{2})\s*(years|yrs|yo)\b", text)
    if m:
        out["age_min"] = int(m.group(2)) + 1
    m = re.search(r"\b(over|above|at least|min)\s+(\d{2})\s*(overall)?\b(?!\s*(years|yrs))", text)
    if m and "age_min" not in out:
        # Ratings live in the 60s-90s; "over 30" with no "overall" is an age.
        if int(m.group(2)) < 50 and not m.group(3):
            out["age_min"] = int(m.group(2)) + (1 if m.group(1) in ("over", "above") else 0)
        else:
            out["ovr_min"] = int(m.group(2))
    if re.search(r"\bleft[\s-]?(shot|shooting|handed)\b|\blefty\b|\bshoots left\b", text):
        out["shoots"] = "L"
    elif re.search(r"\bright[\s-]?(shot|shooting|handed)\b|\brighty\b|\bshoots right\b", text):
        out["shoots"] = "R"
    m = re.search(r"\b(\d{2})\s*(\+|or older|and up|and older|plus)", text) or re.search(r"\bolder than\s+(\d{2})\b", text)
    if m:
        out["age_min"] = int(m.group(1)) + (1 if "older than" in m.group(0) else 0)
    m = re.search(r"\b(\d{2})\s*(or younger|and under|and younger)", text)
    if m:
        out["age_max"] = int(m.group(1))
    m = re.search(r"\b(\d{2})\s*(year old|years old|yo|yr old)\b", text) or re.search(r"\bexactly\s+(\d{2})\b", text)
    if m:
        out["age_min"] = out["age_max"] = int(m.group(1))
    m = re.search(r"\$?(\d+(?:\.\d+)?)\s*(m|mil|million)\b", text)
    if m:
        out["cap_max"] = float(m.group(1))
    m = re.search(r"\b(\d{1,2})\s+(years|yrs|seasons)\s+ago\b", text)
    if m:
        out["years_ago"] = int(m.group(1))
    m = re.search(r"\b(?:year|season)\s+(\d{1,2})\b", text)
    if m:
        out["save_year"] = int(m.group(1))
    m = re.search(r"\b(20\d{2})\b", text)
    if m:
        out["season"] = int(m.group(1))
    m = re.search(r"\btop\s+(\d{1,2})\b", text)
    if m:
        out["limit"] = int(m.group(1))
    m = re.search(r"\bpair(?:ing)?\s+(\d)\b|\b(\d)\s+pair", text)
    if m:
        out["depth"] = f"pair{m.group(1) or m.group(2)}"
    return out


def extract(words: List[str], world: World, original_case: List[str],
            protected: Optional[List[List[Person]]] = None) -> Tuple[List[str], Dict[str, Any]]:
    slots: Dict[str, Any] = {}
    text = " ".join(words)
    people = _match_people(words, world, original_case)
    for i, w in enumerate(words):
        # Names protected before translation and spelling ("c mcd", "#97").
        m = re.fullmatch(r"zzp(\d+)", w)
        if m and protected is not None:
            people.append((i, i + 1, protected[int(m.group(1))]))
    people.sort(key=lambda h: h[0])
    if people:
        slots["players"] = [[p.id for p in ps[:5]] for _s, _e, ps in people]
    canon = list(words)
    for start, end, _ps in reversed(people):
        canon[start:end] = ["PLAYER"]
    teams = _match_teams(canon, world)
    if teams:
        slots["teams"] = [tid for _s, _e, tid in teams]
        for start, end, _tid in reversed(teams):
            canon[start:end] = ["TEAM"]
    if any(w in SELF_TEAM for w in words):
        slots["self"] = True
    roles = match_phrases(canon, _ROLE_PHRASES)
    if roles:
        slots["role"] = roles[0][2]
    depth = match_phrases(canon, _DEPTH_PHRASES)
    if depth:
        slots["depth"] = depth[0][2]
    pos = match_phrases(canon, _POS_PHRASES)
    if pos:
        slots["position"] = pos[0][2]
    stat = match_phrases(canon, _STAT_PHRASES)
    if stat:
        slots["stat"] = stat[0][2]
    slots.update({k: v for k, v in _numbers(text).items() if k not in slots or k == "depth"})
    if "pair" in words and "position" not in slots:
        slots["position"] = "D"
    if any(w in _CAREER for w in words) or "all time" in text:
        slots["scope"] = "career"
    elif any(p in text for p in _LAST_SEASON):
        slots["scope"] = "last_season"
    if any(w in words for w in ("cheapest", "cheaper")):
        slots["sort"] = "cap_asc"
    elif "expensive" in words or "priciest" in words or ("highest" in words and "paid" in words):
        slots["sort"] = "cap_desc"
    if any(w in words for w in ("average", "middle")) or "mid" in words:
        slots["direction"] = "mid"
    if "younger" in words or "older" in words:
        slots["angle"] = "age"
    if "upside" in words or ("more" in words and "potential" in words):
        slots["angle"] = "upside"
    if re.search(r"\b(scores?|scored|scoring) more\b|\bmore (goals|points)\b|\bbetter scorer\b", text):
        slots["angle"] = "offense"
    if "penalty kill" in text or ("kill" in words and "penalties" in words):
        slots["focus"] = "pk"
    elif "power play" in text:
        slots["focus"] = "pp"
    elif "goalie" in words and any(w in words for w in ("bad", "sieve", "losing", "awful")):
        slots["focus"] = "goalie"
    elif "shots" in words and any(w in words for w in ("give", "allow", "allowing", "too")):
        slots["focus"] = "defense"
    low_words = ("worst", "least", "fewest", "lowest", "weakest", "bad", "overpaid", "cold", "declining",
                 "decline", "worse", "regressed", "dropped", "fallers")
    if "bottom" in words and not any(w in words for w in ("six", "pair", "pairing", "line")):
        words = words + ["lowest"]
    if any(w in words for w in low_words):
        slots["direction"] = "low"
        if any(w in words for w in ("best", "top", "highest", "strongest")):
            slots["both"] = True  # "best vs worst"
    if "losing" in words and "streak" in words or "losses" in words:
        slots["streak_kind"] = "loss"
    if "defensively" in words or "defense" in words and "compare" in text:
        slots["angle"] = "defense"
    elif "offensively" in words:
        slots["angle"] = "offense"
    for k in ("free", "ufa", "rfa", "prospect", "rookie", "young", "veteran", "available"):
        if k in words:
            slots.setdefault("filters", []).append(k)
    if "defend" in words or "defense" in words and "losing" not in words:
        slots.setdefault("focus", "defense")
    if any(w in words for w in ("score", "scoring", "offense", "goals")):
        slots.setdefault("focus", "offense")
    if "goalie" in words:
        slots.setdefault("focus_goalie", True)
    return placeholderize(canon), slots


def _is_reference(words: List[str]) -> Tuple[bool, bool]:
    text = " " + " ".join(words) + " "
    one = any(f" {p} " in text for p in PRONOUNS_ONE)
    many = any(f" {p} " in text for p in PRONOUNS_LIST)
    return one, many


def interpret(raw: str, world: World, context: Optional[Dict[str, Any]] = None) -> Interpretation:
    context = context or {}
    original = [t for t in re.sub(r"[^\w\s#$.%+-]", " ", str(raw or ""), flags=re.UNICODE).split()]
    words = tokens(raw)
    if len(words) == 1 and re.fullmatch(r"\d{1,2}", words[0]):
        words = ["#" + words[0]]  # a lone number is a sweater number
    # Strong name forms are pulled out first so shorthand and spelling never touch them.
    protected: List[List[Person]] = []
    pre = [h for h in _match_people(words, world, original if len(original) == len(words) else [""] * len(words))
           if h[1] - h[0] >= 2 or words[h[0]].startswith("#")]
    for start, end, ps in reversed(pre):
        words[start:end] = [f"zzp{len(protected)}"]
        protected.append(ps)
    lang = detect_language(words)
    words = translate(words, lang)
    words = expand_abbreviations(words)
    speller = _speller_for(world)
    split: List[str] = []
    for w in words:
        pieces = _segment(w, speller)
        split.extend(expand_abbreviations(pieces) if len(pieces) > 1 else pieces)
    words = split
    fixed: List[str] = []
    n_corrected = 0
    for w in words:
        c = speller.correct(w)
        n_corrected += int(c != w)
        # A correction can land on shorthand ("ovr"); expand only what changed.
        fixed.extend(expand_abbreviations([c]) if c != w else [c])
    case_hint = original if len(original) == len(fixed) else [""] * len(fixed)
    canon, slots = extract(fixed, world, case_hint, protected)
    rewritten = " ".join(protected[int(w[3:])][0].name if re.fullmatch(r"zzp\d+", w) else w for w in fixed)
    assumptions: List[str] = []

    # "the second one" / "3rd one" points into the last list.
    ordinals = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4}
    if context.get("list") and not slots.get("players") and len(fixed) <= 4 and ("one" in fixed or "guy" in fixed or "player" in fixed):
        for w in fixed:
            if w in ordinals and ordinals[w] < len(context["list"]):
                slots["players"] = [[context["list"][ordinals[w]]]]
                slots["ordinal"] = True
                assumptions.append(f"#{ordinals[w] + 1} from your last list")
                break

    # "their team" is the team the conversation was last about.
    if "their" in fixed and context.get("team") and not slots.get("teams"):
        slots["teams"] = [context["team"]]
    # References back into the conversation.
    one, many = _is_reference(fixed)
    if one and not slots.get("players") and context.get("players"):
        slots["players"] = [[context["players"][0]]]
        assumptions.append("his = the player from your last question")
    if many and not slots.get("players") and context.get("list"):
        slots["from_list"] = list(context["list"])
        assumptions.append("using the list from your last answer")
    if not slots.get("players") and slots.get("angle") and context.get("list") and "one" in fixed and len(fixed) <= 6:
        slots["players"] = [[pid] for pid in context["list"][:4]]
    if "compare" not in fixed and context.get("intent") == "compare" and not slots.get("players") \
            and any(w in fixed for w in ("better", "defensively", "offensively", "potential", "younger", "older", "cheaper")) \
            and len(fixed) <= 6:
        slots["players"] = [[pid] for pid in context.get("compare", [])]

    model = get_model()
    ranked, known = model.predict(" ".join(canon))
    intent, prob = ranked[0]
    margin = prob - (ranked[1][1] if len(ranked) > 1 else 0.0)
    conf = prob * (0.55 + 0.45 * known)
    if margin > 0.35:
        conf = max(conf, min(0.95, prob))
    # Words nobody taught the model ("asdf qwerty") cap the confidence.
    en = _english_vocab()
    content = [w for w in canon if not w.isupper()]
    if content:
        recognised = (sum(1 for w in content if w in en or w in world.vocab) - 0.5 * n_corrected) / len(content)
        if recognised < 0.5:
            conf = min(conf, 0.25 + 0.3 * recognised)

    # Rules that sharpen the model where the slots are unambiguous.
    n_players = len(slots.get("players") or [])
    if n_players >= 2 and intent not in ("compare", "fit_partner"):
        if any(w in fixed for w in ("or", "versus", "vs", "better", "compare", "and")):
            intent, conf = "compare", max(conf, 0.7)
    if n_players == 1 and len(canon) <= 2 and intent not in ("player_contract", "injuries"):
        intent, conf = "player_card", max(conf, 0.75)
    if "what about" in rewritten and context.get("intent") and n_players and \
            not any(w in fixed for w in ("contract", "cap", "paid", "salary", "trade", "injury")):
        intent, conf = context["intent"], max(conf, 0.65)
        assumptions.append(f"same question as before, about the new player")
    if slots.get("ordinal"):
        intent, conf = "player_card", max(conf, 0.8)
    # "and the worst?" / "what about the best" after a ranked answer: same question, other end.
    if len(fixed) <= 5 and context.get("intent") and not n_players and (fixed[:1] == ["and"] or "what about" in rewritten) \
            and slots.get("direction") in ("low", None) and any(w in fixed for w in ("worst", "best", "lowest", "top")):
        intent, conf = context["intent"], max(conf, 0.7)
        if "worst" in fixed or "lowest" in fixed:
            slots["direction"] = "low"
        assumptions.append("same question as before, other end of the list")
    if slots.get("angle") == "age" and len(slots.get("players") or []) >= 2:
        intent = "compare"
    if slots.get("age_min") and slots.get("age_min") == slots.get("age_max") and not n_players:
        intent, conf = "find_players", max(conf, 0.65)
    if "hat" in fixed and "trick" in fixed:
        intent, conf = "leaders_season", max(conf, 0.7)
    if "will" in fixed and intent in ("champions", "standings", "streak") and any(w in fixed for w in ("win", "make", "finish")):
        intent, slots["prediction"] = "unsupported", True
    if slots.get("stat") and not n_players and intent in ("best_on_team", "find_players") \
            and any(w in fixed for w in ("best", "most", "top", "leader", "leads")):
        intent = "leaders_season"
    if slots.get("sort") and not n_players and "contract" not in fixed and "contracts" not in fixed:
        intent, conf = "find_players", max(conf, 0.7)
    if slots.get("direction") == "mid" and not n_players:
        intent = "standings" if ("team" in fixed or "teams" in fixed or "we" in fixed) else ("find_players" if intent not in ("standings",) else intent)
    if "need" in fixed and (slots.get("role") or slots.get("position")) and intent not in ("find_players", "trade_target", "fit_partner"):
        intent = "fit_partner"
    if intent == "player_card" and not n_players and not slots.get("ordinal"):
        conf = min(conf, 0.28)  # "how", "that": nothing to look up
    ranks = any(w in fixed for w in ("best", "worst", "top", "lowest", "weakest", "strongest", "first", "last"))
    if ranks and not n_players and not slots.get("position") and ("team" in fixed or "teams" in fixed) \
            and not slots.get("self") and not slots.get("teams"):
        intent, conf = "standings", max(conf, 0.75)  # "worst team", "best teams in the league"
    social_words = ("social", "tweet", "tweets", "post", "posts", "fans", "media", "saying", "said", "mad", "happy", "drama")
    if intent == "social" and not any(w in fixed for w in social_words):
        intent = "find_players" if ("league" in fixed or not slots.get("self")) else "best_on_team"
        conf = min(conf, 0.5)
        assumptions.append("ranked by overall")
    if slots.get("both") and intent in ("best_on_team", "find_players", "contracts_value") and "league" in fixed:
        intent = "find_players"
    if intent in ("player_contract", "player_card", "trade_advice") and not n_players and slots.get("filters"):
        intent, conf = "find_players", max(conf, 0.6)
    if intent == "team_card" and slots.get("focus_goalie"):
        intent = "goalie_choice"
    if len(slots.get("teams") or []) >= 2 and not n_players:
        intent, conf = "compare", max(conf, 0.7)
    if intent == "compare" and n_players < 2 and len(context.get("compare") or []) >= 2:
        slots["players"] = [[pid] for pid in context["compare"]]
    if slots.get("from_list") and intent in ("top_prospects", "most_improved", "compare", "player_card"):
        conf = max(conf, 0.6)

    # "best player" / "best ld" with no scope could mean several things.
    if intent in ("best_on_team", "find_players") and not slots.get("self") and not slots.get("filters")             and not slots.get("teams") and len(canon) <= 3 and any(w in fixed for w in ("best", "top"))             and not any(k in slots for k in ("age_max", "age_min", "ovr_min", "cap_max", "role")):
        intent = "best_on_team"
        conf = min(conf, 0.5)
        assumptions.append("no scope given, so your team first")
        ranked = [("best_on_team", conf), ("find_players", 0.3), ("top_prospects", 0.2), ("trade_target", 0.15)]
    band = "high" if conf >= HIGH else ("medium" if conf >= MEDIUM else "low")
    alts = [{"intent": k, "confidence": round(p, 3)} for k, p in ranked[1:4] if p >= 0.05]
    return Interpretation(raw=str(raw or ""), language=lang, rewritten=rewritten, intent=intent,
                          confidence=float(conf), band=band, slots=slots, alternatives=alts,
                          assumptions=assumptions)


def _segment(word: str, speller: "Speller") -> List[str]:
    """Split smashed words ("whoisourbestplayer") into known words; else keep it."""
    if len(word) < 7 or word in speller.vocab or not word.isalpha():
        return [word]
    n = len(word)
    best: List[Optional[List[str]]] = [None] * (n + 1)
    best[0] = []
    for i in range(1, n + 1):
        for j in range(max(0, i - 14), i):
            piece = word[j:i]
            if best[j] is None or (len(piece) < 2 and piece not in ("i", "a", "d", "g", "c")):
                continue
            if piece in speller.vocab or piece in ABBREVIATIONS:
                cand = best[j] + [piece]
                if best[i] is None or len(cand) < len(best[i]):
                    best[i] = cand
    return best[n] if best[n] and len(best[n]) >= 2 else [word]


_HOCKEY: Optional[set] = None


def _hockey_vocab() -> set:
    """Words that win spelling ties: stats, positions, roles and rating names."""
    global _HOCKEY
    if _HOCKEY is None:
        v: set = set()
        for table in (STAT_WORDS, POSITION_WORDS, ROLE_WORDS, DEPTH_WORDS):
            for k, val in table.items():
                v.update(k.split())
                v.update(str(val).lower().split())
        v.update(w for val in ABBREVIATIONS.values() for w in str(val).split())
        try:
            from .metrics import PHRASES

            for ph in PHRASES:
                v.update(w for w in ph.split() if w.isalpha() and len(w) >= 4)
        except Exception:
            pass
        v.update({"goals", "assists", "points", "trade", "draft", "prospect", "contract", "streak", "playoffs", "cup",
                  "goalie", "defenseman", "center", "potential", "overall", "season", "roster", "injury"})
        _HOCKEY = v
    return _HOCKEY


_SPELLER_CACHE: Dict[int, Speller] = {}


def _speller_for(world: World) -> Speller:
    key = id(world)
    sp = _SPELLER_CACHE.get(key)
    if sp is None:
        _SPELLER_CACHE.clear()
        sp = Speller(_english_vocab() | world.vocab, world.vocab - _english_vocab(), _hockey_vocab())
        _SPELLER_CACHE[key] = sp
    return sp
