"""Clean slang, texting shorthand and common typos before anything reads the question.

Whole-word replacements only, so names and stat codes are untouched. Applied once at
the top of ask(), so the intent classifier, the name matcher and every answer layer
all see the same cleaned text ("shud i resign giroux" no longer reads "shud" as Shun).
"""

from __future__ import annotations

import re

_WORDS = {
    "wat": "what", "wut": "what", "wot": "what", "whut": "what", "whats": "what is", "wats": "what is", "wuts": "what is",
    "whos": "who is", "hows": "how is", "wheres": "where is", "whens": "when is", "whys": "why is",
    "r": "are", "u": "you", "ur": "your", "yall": "you all", "n": "and", "&": "and", "w/": "with", "w/o": "without",
    "b/c": "because", "bc": "because", "cuz": "because", "coz": "because", "tho": "though", "thru": "through",
    "yr": "year", "yrs": "years", "szn": "season", "rn": "right now", "atm": "right now", "tmrw": "tomorrow", "lately": "lately",
    "smth": "something", "sth": "something", "smthn": "something", "anything": "anything", "any1": "anyone", "every1": "everyone", "some1": "someone",
    "shud": "should", "shoud": "should", "shld": "should", "wud": "would", "cud": "could", "cn": "can", "gonna": "going to", "gotta": "have to", "wanna": "want to",
    "dont": "don't", "doesnt": "doesn't", "didnt": "didn't", "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't", "cant": "can't", "wont": "won't",
    "im": "i am", "ive": "i have", "theyre": "they are", 
    "losin": "losing", "winnin": "winning", "playin": "playing", "doin": "doing", "goin": "going", "scorin": "scoring", "tryin": "trying",
    "actualy": "actually", "acually": "actually", "happend": "happened", "happned": "happened", "probly": "probably", "realy": "really",
    "leage": "league", "legue": "league", "leauge": "league", "compar": "compare", "comapre": "compare", "compre": "compare",
    "bets": "best", "worst": "worst", "plyer": "player", "playr": "player", "goaile": "goalie", "golie": "goalie", "goalei": "goalie",
    "defence": "defense", "defenceman": "defenseman", "defencemen": "defensemen", "dman": "defenseman", "dmen": "defensemen", "d-man": "defenseman",
    "fwd": "forward", "fwds": "forwards", "pts": "points", "pt": "point", "gls": "goals", "asts": "assists", "gp": "games played",
    "gd": "goal differential", "diff": "differential", "pp": "power play", "pk": "penalty kill", "ot": "overtime",
    "dub": "win", "dubs": "wins", 
    "resign": "re-sign", "resigning": "re-signing", "lol": "", "lmao": "", "bro": "", "bruh": "", "fr": "", "ngl": "", "tbh": "", "idk": "", "pls": "", "plz": "", "yo": "",
    "nah": "no", "yea": "yes", "yeah": "yes", "nope": "no",
}
_PHRASES = [
    (r"\b(best|top|worst)\b([^?]*?)\bsince (?:like )?(january|february|march|april|october|november|december|jan|feb|mar|apr|oct|nov|dec)\b", r"\1\2this season"),
    (r"\bwhos been\b", "who has been"), (r"\bwho's been\b", "who has been"),
    (r"\bd[- ]?man\b", "defenseman"), (r"\bd[- ]?men\b", "defensemen"), (r"\b1[- ]goal\b", "one goal"), (r"\b2[- ]goal\b", "two goal"),
    (r"\bno cap\b", ""), (r"\bold af\b", "old"), (r"\bat home\b", "at home"), (r"\bthis yr\b", "this year"), (r"\bgoals?\b([^?]{0,30}?)\bon the (pp|power ?play)\b", r"power play goals\1"), (r"\bon the pp\b", "on the power play"),
    (r"\bmore then\b", "more than"), (r"\bbetter then\b", "better than"), (r"\bworse then\b", "worse than"), (r"\bthe d\b", "the defense"),
    (r"\bour d\b", "our defense"), (r"\bsince like\b", "since"), (r"\bor nah\b", "or not"), (r"\bkinda\b", "kind of"), (r"\bsorta\b", "sort of"),
]
_TOKEN = re.compile(r"[A-Za-z0-9&/'-]+|[^\sA-Za-z0-9]")


def normalize(q: str) -> str:
    text = " ".join(str(q or "").replace("’", "'").split())
    low = text.lower()
    for pat, rep in _PHRASES:
        low = re.sub(pat, rep, low)
    out = []
    for tok in _TOKEN.findall(low):
        rep = _WORDS.get(tok)
        out.append(tok if rep is None else rep)
    cleaned = " ".join(w for w in out if w)
    cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)
    return " ".join(cleaned.split())
