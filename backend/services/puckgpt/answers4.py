"""PuckGPT, detail layer: questions that name specific attributes, several of them, or a window of games.

Checked after the NLU reads the question (it needs the resolved players) and
before the general handlers, which only ever see one metric:
  * a player's attribute profile: "McKenna's weakest attributes", "Rielly's defensive attributes"
  * comparisons on the named attributes: "who is more physical, Rielly or OEL"
  * rankings on two or more attributes at once: "highest ceiling and best development speed"
  * a player's last N games: "Stolarz's save % over his last 5 starts"

``caveats`` names any part of a question no layer can honour, so an answer
never quietly drops it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .answers import POS_NAMES, SINGULAR, _answer, _card, _pos_ok, _resolve_players, _team_label
from .data import LiveView
from .metrics import METRICS, find_metrics, fmt, metric_value

Answer = Dict[str, Any]

_PROFILE = re.compile(r"\b(attributes?|ratings? breakdown|breakdown|profile|strengths?|weakness(es)?|weakest|strongest|"
                      r"best (traits|skills)|worst (traits|skills)|skill set|tools)\b")
_COMPARE = re.compile(r"\bcompare\b|\bvs\.?\b|\bversus\b|\bwho('?s| is) (more|better|faster|stronger|smarter|tougher)\b|\bbetter\b.*\bor\b|\bmore\b.*\bor\b")
_LAST_N = re.compile(r"\b(?:last|past|previous)\s+(\d{1,2}|three|four|five|six|seven|eight|nine|ten)\s+(games|starts|outings|contests)\b")
_WORDS = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# Chapter words that point a profile at one chapter.
_CHAPTER_WORDS = {
    "offence": ("offensive", "offence", "offense", "scoring"), "defence": ("defensive", "defence", "defense"),
    "transition": ("transition", "skating"), "mental": ("mental",), "physical": ("physical",),
    "character": ("character", "intangibles"), "potential": ("potential", "development"),
    "glove": ("glove",), "blocker": ("blocker",), "stick": ("stick",),
}

# Splits no layer can make, and what the answer falls back to.
CAVEATS: List[Tuple[str, str]] = [
    (r"\b(at home|on the road|home games|road games)\b", "home/road splits for players"),
    (r"\b(scored|points|goals|played|doing|numbers|stats) against the [a-z]+\b", "player stats against one opponent"),
    (r"\bper 60\b|\bper sixty\b", "per-60 rates"),
    (r"\b(first|second|third) period\b|\bin overtime\b", "per-period stats"),
    (r"\bsince the (trade|deadline|break)\b|\bafter the trade\b", "stats split at a date"),
    (r"\b(in|during) the playoffs\b|\bplayoff (stats|goals|points)\b", "playoff-only stats this season"),
    (r"\b(5 on 5|five on five|even strength)\b", "even-strength-only stats"),
]


def _skater_chapters(goalie: bool):
    try:
        from app.sim_engine.entities.chapter_attributes import GOALIE_CHAPTERS, SKATER_CHAPTERS
    except Exception:
        return ()
    return GOALIE_CHAPTERS if goalie else SKATER_CHAPTERS


def _percentile(view: LiveView, pid: str, metric: str, pool: List[str]) -> Optional[float]:
    v = metric_value(view, pid, metric)
    if v is None:
        return None
    vals = [x for x in (metric_value(view, p, metric) for p in pool) if x is not None]
    if not vals:
        return None
    higher = METRICS.get(metric, {}).get("higher", True)
    below = sum(1 for x in vals if (x < v if higher else x > v))
    return 100.0 * below / len(vals)


# =========================================================================== profile
def profile(view: LiveView, row: Dict[str, Any], raw: str) -> Answer:
    p = view.players.get(row["id"])
    goalie = row["pos"] == "G"
    chapters = _skater_chapters(goalie)
    focus = [cid for cid, words in _CHAPTER_WORDS.items() if any(re.search(rf"\b{w}\b", raw) for w in words)]
    pairs: List[Tuple[str, str, float]] = []  # (chapter label, attribute label, value)
    chapter_scores: List[str] = []
    for ch in chapters:
        if focus and ch.id not in focus:
            continue
        cv = metric_value(view, row["id"], f"ch:{ch.id}")
        if cv is not None:
            chapter_scores.append(f"{ch.label} {cv:.0f}")
        for hc in ch.hidden:
            v = metric_value(view, row["id"], f"hd:{hc.id}")
            if v is not None:
                pairs.append((ch.label, hc.label, v))
    if not pairs:
        return _answer(f"{row['name']} has no attribute profile stored in this save.", focus_ids=[row["id"]])
    weak = bool(re.search(r"\b(weak|weakest|weakness|weaknesses|worst)\b", raw))
    strong = bool(re.search(r"\b(strong|strongest|strength|strengths|best)\b", raw))
    pairs.sort(key=lambda x: x[2], reverse=not weak)
    shown = pairs[:12] if (weak or strong) else sorted(pairs, key=lambda x: (x[0], -x[2]))
    label = (" / ".join(c.title() for c in focus) + " ") if focus else ""
    if weak:
        head = f"{row['name']}'s weakest {label}attributes: " + ", ".join(f"{a} {v:.0f}" for _c, a, v in pairs[:3]) + "."
    elif strong:
        head = f"{row['name']}'s strongest {label}attributes: " + ", ".join(f"{a} {v:.0f}" for _c, a, v in pairs[:3]) + "."
    else:
        head = f"{row['name']}'s {label}attributes ({len(pairs)} ratings)."
    analysis = [f"{row['pos']}, {row['age']}, {row['ovr']:.0f} overall."]
    if chapter_scores:
        analysis.append("Chapter scores: " + ", ".join(chapter_scores) + ".")
    return _answer(head, analysis=analysis, table={"cols": ["Chapter", "Attribute", "Rating"],
                                                    "rows": [[c, a, round(v)] for c, a, v in shown]},
                   cards=[_card(row)], focus_ids=[row["id"]],
                   followups=[f"{row['name']}'s strongest attributes", f"{row['name']}'s weakest attributes"])


# =========================================================================== compare on named measures
def compare_on(view: LiveView, rows: List[Dict[str, Any]], metrics: List[str]) -> Answer:
    rows = rows[:4]
    cols = ["Measure"] + [r["name"] for r in rows]
    table, wins = [], {r["id"]: 0 for r in rows}
    for m in metrics:
        vals = [metric_value(view, r["id"], m) for r in rows]
        if all(v is None for v in vals):
            continue
        higher = METRICS[m].get("higher", True)
        known = [(v, r) for v, r in zip(vals, rows) if v is not None]
        best = (max if higher else min)(known, key=lambda vr: vr[0])[1]
        wins[best["id"]] += 1
        table.append([METRICS[m]["label"]] + [fmt(m, v) for v in vals])
    if not table:
        names = ", ".join(METRICS[m]["label"].lower() for m in metrics)
        return _answer(f"Neither player has {names} on record.", focus_ids=[r["id"] for r in rows])
    leader = max(rows, key=lambda r: wins[r["id"]])
    tie = sorted(wins.values(), reverse=True)
    what = " and ".join(METRICS[m]["label"].lower() for m in metrics if METRICS[m]["label"] in [t[0] for t in table])
    head = (f"Even on {what}." if len(tie) > 1 and tie[0] == tie[1]
            else f"{leader['name']} comes out ahead on {what}.")
    return _answer(head, analysis=[f"Won {wins[leader['id']]} of {len(table)} measures."],
                   table={"cols": cols, "rows": table}, cards=[_card(r) for r in rows],
                   focus_ids=[r["id"] for r in rows])


# =========================================================================== several measures at once
def multi_rank(view: LiveView, ids: List[str], metrics: List[str], where: str, slots: Dict[str, Any], low: bool) -> Answer:
    pos = slots.get("position")
    pool = [p for p in ids if view.row(p) and _pos_ok(view.row(p), pos)]
    if slots.get("age_max"):
        pool = [p for p in pool if 0 < view.row(p)["age"] <= slots["age_max"]]
    if slots.get("age_min"):
        pool = [p for p in pool if view.row(p)["age"] >= slots["age_min"]]
    if slots.get("shoots"):
        pool = [p for p in pool if view.row(p).get("shoots") == slots["shoots"]]
    scored = []
    for pid in pool:
        pct = [_percentile(view, pid, m, pool) for m in metrics]
        if any(x is None for x in pct):
            continue
        scored.append((sum(pct) / len(pct), pid))
    scored.sort(reverse=not low)
    labels = [METRICS[m]["label"].lower() for m in metrics]
    if not scored:
        return _answer(f"No one {where} has {' and '.join(labels)} on record.")
    top = view.row(scored[0][1])
    shown = scored[:10]
    table = {"cols": ["Player", "Pos", "Team", "Age"] + [METRICS[m]["label"] for m in metrics] + ["Blend"],
             "rows": [[view.row(p)["name"], view.row(p)["pos"], view.row(p)["team"], view.row(p)["age"]]
                      + [fmt(m, metric_value(view, p, m)) for m in metrics] + [f"{s:.0f}"] for s, p in shown]}
    return _answer(f"{'Weakest' if low else 'Best'} on {' and '.join(labels)} {where}: {top['name']}.",
                   analysis=[f"Each measure turned into a percentile among the {len(pool)} players in scope, then averaged (Blend)."],
                   table=table, cards=[_card(view.row(p)) for _s, p in shown[:6]], list_ids=[p for _s, p in shown],
                   focus_ids=[top["id"]])


# =========================================================================== last N games
def last_n(view: LiveView, row: Dict[str, Any], n: int, metrics: List[str]) -> Answer:
    games = list((view.game_log(row["id"]).get("games") or []))
    games = [g for g in games if str(g.get("scope") or "regular_season") == "regular_season"]
    if row["pos"] == "G":
        games = [g for g in games if g.get("goalie") and int(g.get("toi_sec") or 0) >= 1800]
    if not games:
        return _answer(f"{row['name']} has no games in this season's box-score book yet.", focus_ids=[row["id"]])
    recent = games[-n:]
    if row["pos"] == "G":
        sa = sum(int(g.get("sa") or 0) for g in recent)
        sv = sum(int(g.get("sv") or 0) for g in recent)
        ga = sum(int(g.get("ga") or 0) for g in recent)
        rec = [g.get("decision") for g in recent]
        head = f"{row['name']} over his last {len(recent)} starts: {sv / sa:.3f} save %, {ga} goals against." if sa else \
            f"{row['name']} has faced no shots in his last {len(recent)} starts."
        analysis = [f"Record {rec.count('W')}-{rec.count('L')}-{rec.count('OTL')}.",
                    f"Season: {row['sv_pct']:.3f}." if row.get("sv_pct") is not None else ""]
        cols = ["Date", "Opponent", "Result", "SA", "SV", "SV%"]
        trows = [[g.get("date"), g.get("opp_name"), g.get("result"), g.get("sa"), g.get("sv"), g.get("sv_pct")] for g in recent[::-1]]
    else:
        g_, a_ = sum(int(g.get("g") or 0) for g in recent), sum(int(g.get("a") or 0) for g in recent)
        extra = ""
        for m in metrics:
            if m in ("sog", "hit", "blk", "pim", "ppg"):
                extra += f", {sum(int(g.get(m) or 0) for g in recent)} {METRICS[m]['label'].lower()}"
        head = f"{row['name']} over his last {len(recent)} games: {g_} G, {a_} A, {g_ + a_} PTS{extra}."
        analysis = [f"Season: {row['g']} G, {row['a']} A, {row['pts']} PTS in {row['gp']} GP.",
                    f"Plus/minus {sum(int(g.get('pm') or 0) for g in recent):+d} over that stretch."]
        cols = ["Date", "Opponent", "Result", "G", "A", "PTS", "SOG", "TOI"]
        trows = [[g.get("date"), g.get("opp_name"), g.get("result"), g.get("g"), g.get("a"), g.get("pts"), g.get("sog"),
                  f"{int(g.get('toi_sec') or 0) // 60}:{int(g.get('toi_sec') or 0) % 60:02d}"] for g in recent[::-1]]
    return _answer(head, analysis=analysis, table={"cols": cols, "rows": trows}, cards=[_card(row)], focus_ids=[row["id"]])


# =========================================================================== entry points
def detail(view: LiveView, raw: str, it: Any, conv: Dict[str, Any]) -> Optional[Tuple[str, Answer]]:
    """(intent name, answer) when the question is one of this layer's, else None."""
    from .answers2 import _pool  # local: answers2 imports this module's neighbours

    text = it.rewritten.lower()
    words = text.split()
    slots = it.slots
    rows = _resolve_players(view, slots)
    metrics = [m for m in find_metrics(words) if m not in ("age", "cap", "value", "morale")]

    m = _LAST_N.search(raw)
    if m and rows:
        n = int(_WORDS.get(m.group(1), m.group(1)))
        return "last_n_games", last_n(view, rows[0], max(1, min(n, 30)), metrics)
    if rows and len(rows) == 1 and _PROFILE.search(raw) and not re.search(r"\bcontract\b", raw):
        return "attribute_profile", profile(view, rows[0], raw)
    if len(rows) >= 2 and metrics and (metrics != ["ovr"]) and (_COMPARE.search(raw) or " or " in f" {raw} "):
        return "compare_on", compare_on(view, rows, metrics)
    if not rows and len(metrics) >= 2 and re.search(r"\b(and|plus|with|while|but)\b", raw):
        ids, where = _pool(view, slots, conv, text)
        low = bool(re.search(r"\b(worst|lowest|weakest|least)\b", raw))
        return "multi_rank", multi_rank(view, ids, metrics, where, slots, low)
    return None


def caveats(raw: str, intent: str) -> List[str]:
    """Parts of the question that the answer could not honour."""
    if intent.startswith(("teamq", "history:")) or intent in ("last_n_games",):
        return []
    out = []
    for pattern, what in CAVEATS:
        if re.search(pattern, raw):
            out.append(f"I can't split by {what}; this answer uses the full regular season.")
    return out
