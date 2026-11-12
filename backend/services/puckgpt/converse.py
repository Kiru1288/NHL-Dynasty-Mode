"""PuckGPT conversation layer: makes the desk talk like an assistant instead of a lookup table.

It sits around the intent engine and adds what a rule router alone can't do:
  * small talk (hello, thanks, who are you) answered as small talk, not a help dump
  * natural paraphrases rewritten to wordings the router already understands
  * several questions in one message, each answered in turn
  * "why?", "tell me more", "are you sure?" answered from the previous reply
  * judgment questions ("are we good enough to win the Cup", "what do we need",
    "what do you think of my team") answered from the roster, even before a game is played
  * dead ends ("No games played yet") rescued with an on-paper read
  * replies written as a short paragraph, with the supporting detail kept underneath
  * a useful reply when the question wasn't understood: say what was caught, offer next steps

Everything here reads the live save. Nothing is invented and nothing is written back.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

Answer = Dict[str, Any]

_FWD = ("C", "LW", "RW", "F", "W")
_DEF = ("D", "LD", "RD")
_GROUPS = (("top6", "top-six forwards", 0.30), ("d", "top-four defense", 0.30),
           ("g", "goaltending", 0.25), ("depth", "forward depth", 0.15))
_GROUP_ASK = {"top6": "Best forwards available", "depth": "Best forwards available",
              "d": "Best defensemen available", "g": "Best goalies available"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'#$%.+-]", " ", str(text or "").lower())).strip(" .")


def _ord(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _sentence(text: str) -> str:
    text = str(text or "").strip()
    return text if not text or text[-1] in ".!?…”\"" else text + "."


# --------------------------------------------------------------------------- small talk

_GREET = re.compile(r"^(?:hi+|hey+|hello+|yo+|sup|what'?s up|whats up|wassup|howdy|hiya|good (?:morning|afternoon|evening)|bonjour|hola|hallo)(?: (?:there|pucky|puckgpt|buddy|bud|man|boss|coach))?$")
_THANKS = re.compile(r"^(?:thanks?(?: you)?|thx|ty|tysm|cheers|appreciate it|much appreciated|merci|gracias|danke)(?: (?:a lot|so much|man|bud|buddy|pucky|puckgpt))?$")
_BYE = re.compile(r"^(?:bye+|goodbye|see (?:ya|you)|later|cya|gn|good ?night|that'?s all|im done|i'?m done)(?: (?:pucky|puckgpt|for now))?$")
_PRAISE = re.compile(r"^(?:nice|great|awesome|cool|sweet|perfect|good (?:job|stuff|one|call)|love it|beauty|amazing|sick|ok(?:ay)?|k|kk|got it|makes sense|fair(?: enough)?|lol|lmao|haha+|wow)$")
_INSULT = re.compile(r"^(?:you(?:'?re| are)? (?:suck|dumb|stupid|useless|wrong|bad|trash|terrible)|that'?s (?:wrong|not right|dumb|useless|not what i asked)|wrong|no that'?s wrong|useless|not helpful)$")
_WHO = re.compile(r"^(?:who are you|what are you|what is puckgpt|are you (?:an? )?(?:ai|bot|robot|real|human|chatgpt|claude)|what'?s your name|who made you)$")
_HOWRU = re.compile(r"^(?:how are you|how'?s it going|hows it going|how you doing|how are things|you good)$")
_WHY = re.compile(r"^(?:why|why is that|why so|how come|why do you (?:say|think) (?:that|so)|explain(?: that| why)?|what makes you say that|based on what|what'?s the reasoning|reasoning)$")
_MORE = re.compile(r"^(?:more|tell me more|go on|elaborate|more details?|details|keep going|and|what else|anything else|say more|expand(?: on that)?)$")
_SURE = re.compile(r"^(?:are you sure|you sure|really|seriously|for real|is that right|is that true|u sure)$")


def _team_line(view: Any) -> str:
    """One line on where the user's club stands, for hello/how-are-you replies."""
    tid = str(view.user_tid)
    name = view.team_meta.get(tid, {}).get("nickname") or view.team_meta.get(tid, {}).get("name") or "your club"
    try:
        row = next((r for r in view.standings() if str(r.get("team_id")) == tid), None)
    except Exception:
        row = None
    if row and int(row.get("gp") or 0):
        rec = f"{int(row.get('w') or 0)}-{int(row.get('l') or 0)}-{int(row.get('otl') or 0)}"
        rank = int(row.get("league_rank") or row.get("rank") or 0)
        return f"The {name} are {rec}" + (f", {_ord(rank)} in the league." if rank else ".")
    return f"The {name} haven't played a game yet this season."


def small_talk(raw: str, view: Any, ctx: Dict[str, Any]) -> Optional[Tuple[str, Answer]]:
    """A reply for chatter, or for a follow-up that leans on the last answer. None for real questions."""
    q = _clean(raw)
    if not q or len(q) > 60:
        return None
    last = ctx.get("last") or {}
    starters = ["How is my team doing?", "What do we need?", "Who's my best player?"]
    if _GREET.match(q):
        return "chat:greet", {"answer": f"Hey boss. {_team_line(view)} What do you want to dig into?", "analysis": [], "followups": starters}
    if _HOWRU.match(q):
        return "chat:greet", {"answer": f"All good in the film room. {_team_line(view)} What's on your mind?", "analysis": [], "followups": starters}
    if _THANKS.match(q):
        return "chat:thanks", {"answer": "Anytime, boss. Ask whenever something else comes up.", "analysis": [],
                               "followups": list(last.get("followups") or [])[:3]}
    if _BYE.match(q):
        return "chat:bye", {"answer": "See you at the rink, boss.", "analysis": [], "followups": []}
    if _PRAISE.match(q):
        return "chat:ack", {"answer": "Glad that helped. Want to keep going?", "analysis": [],
                            "followups": list(last.get("followups") or starters)[:3]}
    if _WHO.match(q):
        return "chat:about", {
            "answer": "I'm PuckGPT, your front-office assistant. I read this save directly (rosters, contracts, stats, standings, "
                      "trades, drafts and history) and answer from what's actually in it.",
            "analysis": ["I'm built into the game, so I work offline and I never make numbers up.",
                         "Ask the way you'd ask a scout: typos, slang, follow-ups and two questions at once are all fine."],
            "followups": starters}
    if _INSULT.match(q):
        tip = f"Last time I read your question as “{last.get('q')}”. " if last.get("q") else ""
        return "chat:repair", {"answer": "Fair. Let me take another run at it. " + tip + "Tell me what I missed, or put it another way.",
                               "analysis": ["Naming the player, team or stat you mean usually fixes it."],
                               "followups": list(last.get("followups") or starters)[:3]}
    if last and (_WHY.match(q) or _MORE.match(q) or _SURE.match(q)):
        detail = [str(a) for a in last.get("detail") or [] if a]
        if _SURE.match(q):
            head = f"Yes. That comes straight from the save: {last.get('answer')}"
            if not detail:
                return "chat:confirm", {"answer": head, "analysis": [], "followups": list(last.get("followups") or [])[:3]}
            return "chat:confirm", {"answer": head, "analysis": detail[:6], "followups": list(last.get("followups") or [])[:3]}
        if not detail:
            return "chat:why", {"answer": "That one is a straight read of the save, with no judgment call behind it: " + str(last.get("answer") or ""),
                                "analysis": [], "followups": list(last.get("followups") or [])[:3]}
        head = "Here's what that's based on." if _WHY.match(q) else "Here's the rest of what I have on that."
        out = {"answer": head, "analysis": detail[:8], "followups": list(last.get("followups") or [])[:3]}
        if last.get("table"):
            out["table"] = last["table"]
        return "chat:why", out
    return None


# --------------------------------------------------------------------------- paraphrases

_NOT_A_NAME = r"(?!(?:my|our|the|this|that|a|an|we|us|it|there)\b)"
_REWRITES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(?:so |ok(?:ay)? |hey |yo |pucky |puckgpt |please |pls |can you |could you |would you |i want to know |i wanna know |i was wondering |tell me |do you know )+(?=\w)", re.I), ""),
    (re.compile(r"^how (?:did|have|do|does|are) (we|they|the [\w ]+?) (?:do|done|doing|fare|fared|faring|play|played|playing|stack up|match up) (?:against|vs\.?|versus) (.+)$", re.I), r"what is our record against \2"),
    (re.compile(r"^(?:what do you (?:think|make) (?:of|about)|thoughts on|(?:what'?s |whats )?your (?:take|read|opinion) on|talk to me about|give me (?:the|a) (?:rundown|breakdown|scoop|lowdown) on|break down|what'?s the deal with|whats the deal with) " + _NOT_A_NAME + r"(.+)$", re.I), r"tell me about \1"),
    (re.compile(r"^(?:is|how good is) " + _NOT_A_NAME + r"(.+?)(?: (?:any good|good|legit|for real|the real deal|a stud|elite))?$", re.I), r"tell me about \1"),
    (re.compile(r"\b(?:get rid of|ship out|move on from|offload|unload|part ways with|sell off|cut bait on|dump)\b", re.I), "trade"),
]
_GO_AFTER = re.compile(r"^who (?:should|could|can|do) (?:i|we) (?:go after|target|pick up|bring in|acquire|pursue|trade for|go get|look at)\b(.*)$", re.I)
_POS_WORD = (("goalie", r"goalies?|goaltenders?|netminders?|in net"), ("center", r"cent(?:er|re)s?|down the middle|c"),
             ("left wing", r"left wing(?:er)?s?|lw"), ("right wing", r"right wing(?:er)?s?|rw"), ("winger", r"wing(?:er)?s?|on the wing"),
             ("defenseman", r"defen[cs]e(?:m[ae]n)?|defenders?|d-?m[ae]n|blue ?line|d|ld|rd"), ("forward", r"forwards?|up front"))


def _pos_in(text: str) -> str:
    for word, pat in _POS_WORD:
        if re.search(r"\b(?:" + pat + r")\b", text, re.I):
            return word
    return ""


def rewrite(raw: str) -> str:
    """Turn common natural phrasings into a wording the intent router already handles."""
    q = str(raw or "").strip()
    stripped = q.rstrip(" ?!.")
    m = _GO_AFTER.match(re.sub(r"^(?:so |ok(?:ay)? |hey |yo )+", "", stripped, flags=re.I))
    if m:
        pos = _pos_in(m.group(1))
        return f"best {pos} trade targets" if pos else "best trade targets"
    m = re.match(r"^(?:is|are) (.+?) (?:overpaid|underpaid|worth (?:his|the|that) (?:contract|money|deal|cap hit|salary)|a bad contract|a good contract|on a (?:good|bad) (?:deal|contract))$", stripped, re.I)
    if m and not re.match(r"^(?:my|our|the) (?:team|roster)", m.group(1), re.I):
        return f"{m.group(1)} contract"
    if re.search(r"\b(?:trade idea|trade ideas|good trade|trade suggestion|suggest a trade|propose a trade|trade i (?:could|can|should) make|make a trade)\b", stripped, re.I):
        pos = _pos_in(re.sub(r"\btrade\b", " ", stripped, flags=re.I))
        return f"best {pos} trade targets" if pos else "who are the top trade targets"
    if re.search(r"\bufas?\b", stripped, re.I) and not re.search(r"\b(?:my|our|we|i)\b", stripped, re.I):
        stripped = re.sub(r"\bufas?\b", "free agent", stripped, flags=re.I)
    for i, (pat, rep) in enumerate(_REWRITES):
        new = pat.sub(rep, stripped, count=1).strip()
        if new and new != stripped:
            # The "is X good" rule must not eat real questions like "is he worth it" or "is X available".
            if i == 3 and (len(stripped.split()) > 6 or re.search(r"\b(worth|available|injured|hurt|better|worse|happy|signed|a (?:ufa|rfa)|ufa|rfa|overpaid|underpaid|ready|tradeable|or)\b", stripped, re.I)):
                continue
            stripped = new
    return stripped or q


_TOP_TWO = re.compile(r"\b(?:my|our) (?:top|best) (?:two|2) (\w[\w -]*?)(?=$| and | to | on | by | in |\?|,)", re.I)


def expand(raw: str, view: Any) -> str:
    """Resolve "my top two centers" into the two names, so compare questions work without typing them."""
    q = str(raw or "")
    m = _TOP_TWO.search(q)
    if not m or not re.search(r"\b(?:compare|vs|versus|better|between)\b", q, re.I):
        return q
    pos = _pos_in(m.group(1))
    rows = view.team_rows(str(view.user_tid))
    want = {"goalie": ("G",), "center": ("C",), "left wing": ("LW",), "right wing": ("RW",), "winger": ("LW", "RW", "W"),
            "defenseman": _DEF, "forward": _FWD}.get(pos)
    pool = sorted((r for r in rows if not want or r.get("pos") in want), key=lambda r: -float(r.get("ovr") or 0))[:2]
    if len(pool) < 2:
        return q
    return f"compare {pool[0]['name']} and {pool[1]['name']}"


# --------------------------------------------------------------------------- several questions at once

_STARTER = r"(?:who|whos|who's|what|whats|what's|which|why|how|should|is|are|can|could|do|does|did|will|would|where|when|show|compare|tell|give|list|rank|find)"
_SPLIT = re.compile(r"\s*(?:\?+|;|\.(?=\s))\s*|\s*,?\s+(?:and|also|plus|then|and also|and then)\s+(?=" + _STARTER + r"\b)", re.I)


def split_questions(raw: str) -> List[str]:
    """Split "who should I sign and who should I trade?" into its questions (three at most)."""
    q = str(raw or "").strip()
    if len(q) < 16 or re.match(r"^\s*(?:compare|between)\b", q, re.I):
        return [q]
    parts = [p.strip(" ,") for p in _SPLIT.split(q) if p and p.strip(" ,")]
    parts = [p for p in parts if len(p.split()) >= 2]
    if len(parts) < 2 or len(parts) > 3:
        return [q]
    # A part must look like a question on its own; otherwise the split was a false cut.
    if not all(re.match(r"^" + _STARTER + r"\b", p, re.I) or len(p.split()) >= 3 for p in parts):
        return [q]
    return parts


# --------------------------------------------------------------------------- roster read (works with zero games played)

def _mean(rows: List[Dict[str, Any]]) -> float:
    return sum(float(r.get("ovr") or 0) for r in rows) / len(rows) if rows else 0.0


def roster_read(view: Any, tid: str) -> Dict[str, Any]:
    """How one club's roster stacks up against the league by position group, from ratings alone."""
    cache = view.__dict__.setdefault("_converse_roster", {})
    if "all" not in cache:
        table: Dict[str, Dict[str, Any]] = {}
        for t in view.teams:
            rows = [r for r in view.team_rows(t) if r]
            fw = sorted((r for r in rows if r.get("pos") in _FWD), key=lambda r: -float(r.get("ovr") or 0))
            de = sorted((r for r in rows if r.get("pos") in _DEF), key=lambda r: -float(r.get("ovr") or 0))
            gl = sorted((r for r in rows if r.get("pos") == "G"), key=lambda r: -float(r.get("ovr") or 0))
            table[str(t)] = {"top6": _mean(fw[:6]), "depth": _mean(fw[6:12]), "d": _mean(de[:4]), "g": _mean(gl[:1]),
                             "fw": fw, "de": de, "gl": gl}
        cache["all"] = table
    table = cache["all"]
    me = table.get(str(tid))
    n = max(1, len(table))
    if not me:
        return {}
    ranks, median = {}, {}
    for key, _label, _w in _GROUPS:
        vals = sorted((v[key] for v in table.values()), reverse=True)
        ranks[key] = 1 + sum(1 for v in vals if v > me[key] + 1e-9)
        median[key] = vals[len(vals) // 2]
    score = sum(ranks[k] * w for k, _l, w in _GROUPS) / n
    ordered = sorted(_GROUPS, key=lambda g: ranks[g[0]])
    return {"n": n, "ranks": ranks, "median": median, "vals": {k: me[k] for k, _l, _w in _GROUPS}, "score": score,
            "best": ordered[0], "worst": ordered[-1], "fw": me["fw"], "de": me["de"], "gl": me["gl"]}


def _who(view: Any, tid: str) -> Tuple[str, str, str]:
    """(subject, verb, possessive) so replies read "You are…" for the user and "The Bruins are…" otherwise."""
    if str(tid) == str(view.user_tid):
        return "You", "are", "your"
    name = view.team_meta.get(str(tid), {}).get("nickname") or view.team_meta.get(str(tid), {}).get("name") or "They"
    return f"The {name}", "are", "their"


def _group_lines(read: Dict[str, Any]) -> List[str]:
    lines = []
    for key, label, _w in sorted(_GROUPS, key=lambda g: read["ranks"][g[0]]):
        lines.append(f"{label.capitalize()}: {read['vals'][key]:.1f} overall on average, {_ord(read['ranks'][key])} of {read['n']} "
                     f"(league median {read['median'][key]:.1f}).")
    return lines


def _tier(score: float) -> str:
    if score <= 0.22:
        return "one of the strongest rosters in the league on paper"
    if score <= 0.42:
        return "a playoff-calibre roster on paper"
    if score <= 0.66:
        return "a middle-of-the-pack roster on paper"
    return "a roster with more holes than strengths right now"


def _standing_row(view: Any, tid: str) -> Optional[Dict[str, Any]]:
    try:
        return next((r for r in view.standings() if str(r.get("team_id")) == str(tid) and int(r.get("gp") or 0)), None)
    except Exception:
        return None


def _cards(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"id": r["id"], "label": r["name"], "sub": f"{r.get('pos')} · {r.get('ovr'):.0f} OVR"} for r in rows if r.get("id")]


def team_opinion(view: Any, tid: str) -> Answer:
    read = roster_read(view, tid)
    if not read:
        return {"answer": "I couldn't read that roster.", "analysis": []}
    subj, verb, poss = _who(view, tid)
    best, worst = read["best"], read["worst"]
    head = f"{subj} have {_tier(read['score'])}."
    head += f" The strength is {poss} {best[1]} ({_ord(read['ranks'][best[0]])} of {read['n']})"
    head += f", and the soft spot is {worst[1]} ({_ord(read['ranks'][worst[0]])})." if worst[0] != best[0] else "."
    lines = _group_lines(read)
    row = _standing_row(view, tid)
    if row:
        rank = int(row.get("league_rank") or row.get("rank") or 0)
        lines.insert(0, f"On the ice so far: {int(row.get('w') or 0)}-{int(row.get('l') or 0)}-{int(row.get('otl') or 0)}"
                        + (f", {_ord(rank)} in the league." if rank else "."))
    else:
        lines.append("No games played yet, so this is ratings only. Ask again once the season is rolling and I'll use results too.")
    stars = (read["fw"][:2] + read["de"][:1] + read["gl"][:1])
    return {"answer": head, "analysis": lines, "cards": _cards(stars),
            "followups": ["What do we need?", "Are we good enough to win the Cup?", _GROUP_ASK[worst[0]]],
            "focus_ids": [r["id"] for r in stars[:1] if r.get("id")]}


def team_needs(view: Any, tid: str) -> Answer:
    read = roster_read(view, tid)
    if not read:
        return {"answer": "I couldn't read that roster.", "analysis": []}
    subj, _verb, poss = _who(view, tid)
    order = sorted(_GROUPS, key=lambda g: -read["ranks"][g[0]])
    worst = order[0]
    gap = read["median"][worst[0]] - read["vals"][worst[0]]
    if read["ranks"][worst[0]] <= max(3, read["n"] // 3):
        head = (f"Honestly, not much. {poss.capitalize()} weakest group is {worst[1]}, and it's still {_ord(read['ranks'][worst[0]])} of {read['n']}. "
                "Any move from here is a luxury upgrade, not a fix.")
    else:
        head = f"{poss.capitalize()} biggest need is {worst[1]}: {_ord(read['ranks'][worst[0]])} of {read['n']} in the league"
        head += f", about {gap:.1f} overall points under the league median." if gap > 0.2 else "."
    lines = []
    pool = {"top6": read["fw"][:6], "depth": read["fw"][6:12], "d": read["de"][:4], "g": read["gl"][:1]}[worst[0]]
    if pool:
        weakest = min(pool, key=lambda r: float(r.get("ovr") or 0))
        lines.append(f"The weakest piece in that group is {weakest['name']} ({weakest.get('pos')}, {weakest.get('ovr'):.0f} overall).")
    if len(order) > 1 and read["ranks"][order[1][0]] > read["n"] // 2:
        lines.append(f"Second on the list: {order[1][1]} ({_ord(read['ranks'][order[1][0]])}).")
    lines += _group_lines(read)
    return {"answer": head, "analysis": lines, "cards": _cards(pool[:4]),
            "followups": [_GROUP_ASK[worst[0]], "Best trade targets", "Who should I trade?"]}


def contender(view: Any, tid: str, goalie_only: bool = False) -> Answer:
    read = roster_read(view, tid)
    if not read:
        return {"answer": "I couldn't read that roster.", "analysis": []}
    subj, _verb, poss = _who(view, tid)
    n = read["n"]
    if goalie_only:
        g = read["gl"][0] if read["gl"] else None
        if not g:
            return {"answer": f"{subj} don't have a goalie on the NHL roster, so no.", "analysis": [], "followups": ["Best goalies available"]}
        rank = read["ranks"]["g"]
        if rank <= max(4, n // 5):
            verdict = f"Yes. {g['name']} ({g['ovr']:.0f} overall) is the {_ord(rank)}-best starter in the league, which is good enough to win with."
        elif rank <= n // 2:
            verdict = (f"He can, but he won't steal a series. {g['name']} ({g['ovr']:.0f} overall) is the {_ord(rank)}-best starter of {n}, "
                       "so the team in front of him has to carry the load.")
        else:
            verdict = (f"Probably not as things stand. {g['name']} ({g['ovr']:.0f} overall) ranks {_ord(rank)} of {n} starters, "
                       f"{max(0.0, read['median']['g'] - g['ovr']):.1f} points under the league median.")
        lines = []
        if g.get("sv_pct") and g.get("gp"):
            lines.append(f"This season: {g['sv_pct']:.3f} save percentage in {g['gp']} games.")
        if len(read["gl"]) > 1:
            b = read["gl"][1]
            lines.append(f"Behind him: {b['name']} ({b['ovr']:.0f} overall).")
        lines.append("Cup winners usually have a top-10 starter or a top-5 team in front of an average one.")
        return {"answer": verdict, "analysis": lines, "cards": _cards(read["gl"][:2]), "focus_ids": [g["id"]],
                "followups": ["Best goalies available", "Which goalie should start?", "What do we need?"]}
    row = _standing_row(view, tid)
    rank = int((row or {}).get("league_rank") or (row or {}).get("rank") or 0)
    worst = read["worst"]
    hole = read["ranks"][worst[0]] > n // 2
    top = read["score"] <= 0.25 and (not rank or rank <= max(6, n // 4))
    if top and not hole:
        verdict = f"Yes. {subj} are a real contender, with no group ranked in the bottom half of the league."
    elif read["score"] <= 0.42 or (rank and rank <= n // 3):
        verdict = (f"{subj} are close, but not a favourite." if not hole else
                   f"{subj} are a playoff team with one real hole: {worst[1]} ({_ord(read['ranks'][worst[0]])} of {n}). Fix that and the answer changes.")
    elif read["score"] <= 0.66:
        verdict = f"Not this year, as built. {subj} are a bubble team, and {worst[1]} ({_ord(read['ranks'][worst[0]])} of {n}) is the biggest gap."
    else:
        verdict = f"No. {subj} are a long way off, with {worst[1]} {_ord(read['ranks'][worst[0]])} of {n}. This is a build, not a push."
    lines = []
    if row:
        lines.append(f"Results so far: {int(row.get('w') or 0)}-{int(row.get('l') or 0)}-{int(row.get('otl') or 0)}" + (f", {_ord(rank)} overall." if rank else "."))
    else:
        lines.append("No games played yet, so this is the roster on paper.")
    lines += _group_lines(read)
    return {"answer": verdict, "analysis": lines,
            "followups": ["What do we need?", _GROUP_ASK[worst[0]], "Best trade targets"]}


def sell_candidates(view: Any, tid: str) -> Answer:
    """Who the club could move: age, cap hit and term weighed against what the player still gives."""
    rows = [r for r in view.team_rows(tid) if r]
    if not rows:
        return {"answer": "I couldn't read that roster.", "analysis": []}
    core = {r["id"] for r in sorted(rows, key=lambda r: -float(r.get("ovr") or 0))[:3]}
    scored = []
    for r in rows:
        c = r.get("contract") or {}
        cap_m = float(c.get("cap_hit_m") or 0)
        years = int(c.get("years_left") or 0)
        ovr, pot, age = float(r.get("ovr") or 0), float(r.get("pot") or 0), int(r.get("age") or 0)
        if r["id"] in core or (age <= 24 and pot - ovr >= 3):
            continue  # stars and kids who are still growing stay
        score = cap_m * min(max(years, 1), 4) / max(4.0, ovr - 68) + max(0, age - 29) * 0.6
        why = [f"{age} years old", f"{ovr:.0f} overall"]
        if cap_m:
            why.append(f"${cap_m:.2f}M" + (f" for {years} more year{'s' if years != 1 else ''}" if years else ""))
        if years == 1:
            score += 0.8
            why.append("expiring, so he's a rental chip")
        scored.append((score, r, ", ".join(why)))
    scored.sort(key=lambda t: -t[0])
    top = scored[:5]
    if not top:
        return {"answer": "Nobody stands out as a player to move. The roster is mostly core pieces and kids who are still improving.", "analysis": []}
    subj, _v, _p = _who(view, tid)
    first = top[0]
    head = f"If {subj.lower() if subj == 'You' else subj} move someone, start with {first[1]['name']} ({first[1].get('pos')}): {first[2]}."
    lines = [f"{r['name']} ({r.get('pos')}): {why}." for _s, r, why in top[1:]]
    lines.append("This weighs age, cap hit and remaining term against rating. Your three best players and prospects who are still growing are left off.")
    return {"answer": head, "analysis": lines, "cards": _cards([r for _s, r, _w in top]), "focus_ids": [first[1]["id"]],
            "list_ids": [r["id"] for _s, r, _w in top],
            "followups": [f"Should I trade {first[1]['name']}?", "Worst contracts on my team", "Best trade targets"]}


# --------------------------------------------------------------------------- judgment questions and forced routes

_TEAM_WORD = r"(?:team|roster|squad|lineup|club|group|guys|boys)"
_CUP = re.compile(r"\b(?:good enough|can (?:we|i|they)|could (?:we|i|they)|are (?:we|they)|do (?:we|i|they) have|is (?:my|our|this|the) " + _TEAM_WORD + r"|will (?:we|they)|chance|shot at|ready)\b.*\b(?:cup|contend\w*|championship|title|win it all|deep run|playoffs?)\b"
                  r"|\b(?:are|r) (?:we|they) (?:a |real |legit |true |actual )*(?:contenders?|pretenders?)\b|\bwin it all\b|\bcup (?:contenders?|window)\b", re.I)
_GOALIE = re.compile(r"\b(?:goalie|goalies|goaltend\w*|netminder|tender|in net|crease|starter)\b", re.I)
_OPINION = re.compile(r"\b(?:what do you (?:think|make) (?:of|about)|thoughts on|your (?:take|read|opinion) on|rate|grade|evaluate|assess|break ?down|how (?:good|bad|strong|deep|solid) (?:is|are)|how(?:'?s| is| are| does| do)) (?:my|our|the|this|we) ?" + _TEAM_WORD + r"?\b(?: look\w*| doing| shaping up| stack up)?"
                      r"|\b(?:are|r) we (?:any )?(?:good|bad|decent|legit|for real|trash|ok(?:ay)?)\b|\bhow (?:good|bad) are we\b|\bhow (?:are|r) we looking\b|\bhow do we look\b|\b(?:rate|grade) (?:us|my " + _TEAM_WORD + r")\b", re.I)
_NEEDS = re.compile(r"\bwhat (?:do|does|should) (?:we|i|my " + _TEAM_WORD + r"|our " + _TEAM_WORD + r"|they) (?:still )?(?:need|lack|miss|missing|fix|address|upgrade|improve|work on)\b"
                    r"|\b(?:biggest|main|top|greatest|glaring) (?:need|needs|hole|holes|weakness|weaknesses|gap|gaps)\b|\bwhere (?:are|r) (?:we|they) (?:weak\w*|thin\w*|lacking|short)\b"
                    r"|\bwhat (?:are|is) (?:my|our) (?:needs?|weak(?:ness(?:es)?| spots?| points?)|holes?)\b|\bweak(?:est)? (?:spot|link|position|area)\b|\bwhat(?:'?s| is) (?:my|our) weakness\b|\bwhere (?:should|do) (?:i|we) upgrade\b", re.I)
_PROBLEM = re.compile(r"\b(?:biggest|main|real|number one|#1) (?:problem|issue)\b|\bwhat(?:'?s| is) (?:wrong|the matter|going wrong|the problem|the issue) with (?:us|my " + _TEAM_WORD + r"|our " + _TEAM_WORD + r"|the " + _TEAM_WORD + r"|this " + _TEAM_WORD + r")\b"
                      r"|\b(?:holding|keeping|stopping|preventing) us\b|\bwhy (?:do|are|r) we (?:suck\w*|so bad|bad|struggl\w*|keep losing|always los\w*|chok\w*|slump\w*)\b|\bwhat(?:'?s| is) going wrong\b|\bwhy can'?t we win\b", re.I)
_WHY_UNIT = re.compile(r"\bwhy\b.*\b(power ?play|pp|penalty ?kill|pk|goaltending|goalies?|goalie)\b", re.I)


_SELL = re.compile(r"\bwho (?:should|could|can|do|would) (?:i|we|you) (?:trade|sell|move|shop|deal|get rid of|dump|ship out|offload|unload|part ways with|cut|let go)(?: away| off| on from)?(?: first| now| this year| at the deadline)?$"
                   r"|\b(?:which|what) players? (?:are|is) (?:expendable|tradeable|trade bait|on the block|dead ?weight)\b|\bwho(?:'?s| is| are) (?:expendable|trade bait|dead ?weight|on the block)\b"
                   r"|\b(?:sell|trade) candidates\b|\bwho(?:m)? (?:can|could|should) (?:i|we) sell\b", re.I)


def _focus_team(view: Any, slots: Dict[str, Any], q: str) -> str:
    teams = [str(t) for t in (slots.get("teams") or []) if str(t) in view.teams]
    if teams and not re.search(r"\b(?:we|us|our|my|i)\b", q, re.I):
        return teams[0]
    return str(view.user_tid)


def judgment(q: str, view: Any, slots: Dict[str, Any]) -> Optional[Tuple[str, Answer]]:
    """Opinion-style team questions the stat handlers can't answer, built from the roster read."""
    ql = _clean(q)
    if slots.get("players") and not _GOALIE.search(ql):
        return None  # a named player is a player question
    if re.match(r"^(?:who|whos|who's|which team|what team)\b", ql) and not re.search(r"\b(?:we|us|our|my|i)\b", ql) and not _SELL.search(ql):
        return None  # "who's the favourite to win it all" is a league question, not a verdict on the user's club
    tid = _focus_team(view, slots, ql)
    if _SELL.search(ql) and not slots.get("players"):
        return "judge:sell", sell_candidates(view, tid)
    if _CUP.search(ql):
        return "judge:contender", contender(view, tid, goalie_only=bool(_GOALIE.search(ql)))
    if _NEEDS.search(ql):
        return "judge:needs", team_needs(view, tid)
    if _OPINION.search(ql) and not re.search(r"\b(?:record|standings?|power ?play|penalty ?kill|against|vs)\b", ql):
        return "judge:team", team_opinion(view, tid)
    return None


def directive(q: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Questions that must go to one handler whatever the layers above would guess."""
    ql = _clean(q)
    m = _WHY_UNIT.search(ql)
    if m:
        unit = m.group(1).replace(" ", "")
        focus = "pp" if unit in ("powerplay", "pp") else "pk" if unit in ("penaltykill", "pk") else "goalie"
        return "diagnose", {"focus": focus}
    if _PROBLEM.search(ql):
        return "diagnose", {}
    return None


def rescue(out: Answer, intent: str, q: str, view: Any, slots: Dict[str, Any]) -> Answer:
    """A handler that had nothing to say about the user's club gets the on-paper read instead of a dead end."""
    text = str(out.get("answer") or "")
    base = str(intent or "").split(":")[0]
    dead = text.startswith(("No games", "No results", "Nothing to diagnose")) or "nothing to diagnose" in text
    if not dead or base not in ("diagnose", "team_card", "team_plan", "teamq", "carried_us", "standings"):
        return out
    tid = _focus_team(view, slots, q)
    extra = team_needs(view, tid) if base == "diagnose" else team_opinion(view, tid)
    lead = "No games have been played yet, so there are no results to read. Here's the roster on paper instead. "
    return {**extra, "answer": lead + str(extra.get("answer") or "")}


# --------------------------------------------------------------------------- "I didn't get that", done properly

def not_understood(out: Answer, view: Any, slots: Dict[str, Any]) -> Answer:
    """Replace the bare "not sure what you're after" with what was caught and where to go next."""
    names = []
    for cands in slots.get("players") or []:
        if cands and cands[0] in view.players:
            names.append(view.row(cands[0])["name"])
    teams = [view.team_meta[str(t)]["name"] for t in slots.get("teams") or [] if str(t) in view.team_meta]
    if names:
        who = names[0]
        out["answer"] = f"I caught {who} in there, but not what you want to know about him. Here's what I can pull up:"
        out["clarify"] = [{"label": "Profile", "query": f"Tell me about {who}"}, {"label": "Contract", "query": f"{who} contract"},
                          {"label": "Trade advice", "query": f"Should I trade {who}?"}] + \
                         ([{"label": "Compare", "query": f"Compare {who} and {names[1]}"}] if len(names) > 1 else [])
        out["analysis"] = []
    elif teams:
        team = teams[0]
        out["answer"] = f"I caught the {team}, but not the question about them. Pick one, or put it another way:"
        out["clarify"] = [{"label": "Team profile", "query": f"How are the {team} doing?"}, {"label": "Best player", "query": f"Best player on the {team}"},
                          {"label": "Record against them", "query": f"What is our record against the {team}?"}]
        out["analysis"] = []
    else:
        out["answer"] = "That one got past me. Try naming the player, team or stat you mean, or start from one of these:"
        out["analysis"] = []
        if not out.get("clarify"):
            out["clarify"] = [{"label": "Team check", "query": "What do you think of my team?"}, {"label": "Needs", "query": "What do we need?"},
                              {"label": "Best player", "query": "Who's my best player?"}, {"label": "Market", "query": "Best trade targets"}]
    return out


# --------------------------------------------------------------------------- voice

_SKIP_LINE = re.compile(r"^(?:Read as|Compris comme|Entendido como|Verstanden als|Tolkat som|Tulkittu|Понято как|Did you mean|Sorted by)\b|:$|[“”]")


def voice(out: Answer, intent: str) -> Answer:
    """Write the reply as a short paragraph: the answer plus its strongest one or two reasons.

    The remaining lines stay in `analysis` as supporting detail, so nothing is lost.
    """
    base = str(intent or "").split(":")[0]
    answer = str(out.get("answer") or "").strip()
    if not answer or base in ("help", "chat", "games_on_date") or out.get("clarify") and not out.get("table") and not out.get("analysis"):
        return out
    lines = [str(a) for a in out.get("analysis") or []]
    lead, rest, used = [], [], 0
    for line in lines:
        fits = len(lead) < 2 and not _SKIP_LINE.search(line.strip()) and used + len(line) <= 240 and len(line) <= 170
        if fits and not rest_blocks(rest):
            lead.append(_sentence(line))
            used += len(line)
        else:
            rest.append(line)
    out["say"] = " ".join([_sentence(answer)] + lead)
    out["analysis"] = rest
    return out


def rest_blocks(rest: List[str]) -> bool:
    """Once a real detail line has been left behind, keep the order: don't pull later lines above it."""
    return any(not _SKIP_LINE.search(r.strip()) for r in rest)


def remember(ctx: Dict[str, Any], q: str, out: Answer) -> None:
    """Keep the last reply so "why?", "tell me more" and "are you sure?" have something to stand on."""
    if not out.get("answer"):
        return
    say = str(out.get("say") or "")
    detail = [str(a) for a in out.get("analysis") or [] if a and not str(a).startswith("Read as")]
    if say and say != _sentence(str(out.get("answer"))):
        detail = [say[len(_sentence(str(out.get("answer")))):].strip()] + detail
    ctx["last"] = {"q": q, "answer": out.get("answer"), "detail": [d for d in detail if d],
                   "followups": list(out.get("followups") or [])[:3], "table": out.get("table")}


def merge(parts: List[Tuple[str, Answer]]) -> Answer:
    """One reply for a message that held several questions."""
    first = parts[0][1]
    followups: List[str] = []
    for _q, o in parts:
        for f in o.get("followups") or []:
            if f not in followups:
                followups.append(f)
    sections = [{"q": q, "say": o.get("say") or o.get("answer"), "answer": o.get("answer"), "analysis": o.get("analysis") or [],
                 "table": o.get("table"), "cards": o.get("cards") or [], "clarify": o.get("clarify") or []} for q, o in parts]
    count = {2: "two", 3: "three"}.get(len(parts), str(len(parts)))
    interp = dict(first.get("interpretation") or {})
    interp["intent"] = "multi:" + "+".join(str((o.get("interpretation") or {}).get("intent") or "?") for _q, o in parts)
    interp["rewritten"] = " · ".join(q for q, _o in parts)
    return {"answer": f"You asked {count} things, so here they are in order.", "say": f"You asked {count} things, so here they are in order.",
            "analysis": [], "sections": sections, "followups": followups[:4], "interpretation": interp,
            "elapsed_ms": round(sum(float(o.get("elapsed_ms") or 0) for _q, o in parts), 1)}
