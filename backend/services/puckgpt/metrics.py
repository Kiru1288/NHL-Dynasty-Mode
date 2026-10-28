"""Every number PuckGPT can rank players by, and the words people use for each.

Metrics come from three places:
  * the dossier chapters (offence, defence, transition, ... glove, blocker, stick)
    and their ~100 sub-ratings (zone exits, gap control, leadership, ...),
  * the season ledger (goals, points, hits, save %, ...),
  * contract and profile facts (cap hit, value for money, upside, age, morale).

The phrase table is built from the chapter definitions themselves, so a new
sub-rating in the engine is askable without touching this file.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

Metric = Dict[str, Any]

# name -> label, kind (chapter | hidden | stat | fact), higher_is_better, goalie flag
METRICS: Dict[str, Metric] = {}
PHRASES: Dict[str, str] = {}  # phrase -> metric name


def _add(name: str, label: str, kind: str, phrases: List[str], *, higher: bool = True,
         goalie: Optional[bool] = None, min_gp: int = 0) -> None:
    prev = METRICS.get(name)
    if prev is not None and prev.get("goalie") != goalie:
        goalie = None  # defined for skaters and goalies alike (the potential chapter)
    METRICS[name] = {"name": name, "label": label, "kind": kind, "higher": higher, "goalie": goalie, "min_gp": min_gp}
    for ph in phrases + [label.lower()]:
        PHRASES.setdefault(" ".join(ph.lower().replace("-", " ").replace("/", " ").split()), name)


# Stats and facts. Phrases are in the normalised vocabulary (slang already expanded).
_add("ovr", "Overall", "fact", ["overall", "rating", "rated", "best player", "ovr"])
_add("pot", "Potential", "fact", ["potential", "ceiling", "highest ceiling"])
_add("upside", "Upside (potential minus overall)", "fact", ["upside", "room to grow", "growth room", "late bloomer"])
_add("age", "Age", "fact", ["age", "oldest", "youngest"], higher=False)
_add("cap", "Cap hit", "fact", ["cap hit", "salary", "paid", "expensive", "cost", "money"], higher=False)
_add("value", "Value for money", "fact", ["value", "for the money", "bang for buck", "bargain", "underpaid", "best contract",
                                           "nobody talks about", "underrated", "hidden gem"])
_add("morale", "Morale", "fact", ["morale", "happy", "happiest", "unhappy", "mood", "hates being here", "wants out"])
_add("g", "Goals", "stat", ["goals", "goal", "scorer", "sniper", "who can score", "finisher", "lit the lamp"], goalie=False)
_add("a", "Assists", "stat", ["assists", "assist", "playmaker", "setup man"], goalie=False)
_add("ppg", "Power-play goals", "stat", ["power play goals", "pp goals", "power play goal", "ppg"], goalie=False)
_add("ppp", "Power-play points", "stat", ["power play points", "pp points", "ppp"], goalie=False)
_add("shg", "Shorthanded goals", "stat", ["shorthanded goals", "short handed goals", "shg", "shorties"], goalie=False)
_add("gwg", "Game-winning goals", "stat", ["game winning goals", "game winners", "gwg", "clutch goals"], goalie=False)
_add("pts", "Points", "stat", ["points", "point", "production", "carries the mail", "offensive production", "scoring",
                                   "top scorer", "leading scorer", "best scorer", "scoring leader"], goalie=False)
_add("ppg_rate", "Points per game", "stat", ["points per game", "ppg rate", "per game"], min_gp=5, goalie=False)
_add("plus_minus", "Plus/minus", "stat", ["plus minus", "plus/minus", "+/-"], goalie=False)
_add("hit", "Hits", "stat", ["hits", "hitter", "physical play", "hits the most"], goalie=False)
_add("blk", "Blocked shots", "stat", ["blocks", "blocked shots", "shot blocker", "blocks the most shots"], goalie=False)
_add("pim", "Penalty minutes", "stat", ["penalty minutes", "pim", "goon", "enforcer"], goalie=False)
_add("sog", "Shots", "stat", ["shots", "shooter", "shoots the most"], goalie=False)
_add("toi_pg", "Ice time per game", "stat", ["ice time", "minutes", "toi", "most minutes"], goalie=False)
_add("fo_pct", "Faceoff %", "stat", ["faceoffs", "faceoff", "faceoff %", "draws", "wins the most faceoffs", "faceoff percentage"], min_gp=5, goalie=False)
_add("cf_pct", "Possession (shot attempt share)", "stat", ["drives play", "possession", "corsi", "shot share", "controls play"], min_gp=5)
_add("w", "Wins", "stat", ["wins"], goalie=True)
_add("sv_pct", "Save %", "stat", ["save percentage", "save %", "steals games", "hot goalie", "best goaltending"], goalie=True, min_gp=3)
_add("so", "Shutouts", "stat", ["shutout", "shutouts"], goalie=True)


def _register_chapters() -> None:
    try:
        from app.sim_engine.entities.chapter_attributes import GOALIE_CHAPTERS, SKATER_CHAPTERS
    except Exception:
        return
    extra = {
        "offence": ["offence", "offense", "offensive", "offensively", "offensive rating", "offensive ability"],
        "defence": ["defence", "defense", "defensive", "defensively", "defensive rating", "stay at home", "shutdown",
                    "defensive defenseman", "best defender"],
        "transition": ["transition", "transition rating", "skating", "skater", "puck mover", "puck moving", "rush", "breakout"],
        "mental": ["mental", "hockey iq", "smart", "smartest", "iq", "composure"],
        "physical": ["physical", "physicality", "toughness", "tough", "grit", "gritty", "jam", "strength", "strongest"],
        "character": ["character", "beauty", "good guy", "leader", "leadership guy", "locker room guy", "glue guy"],
        "glove": ["glove", "glove hand"], "blocker": ["blocker"], "stick": ["stick", "puckhandling goalie"],
    }
    for chapters in (SKATER_CHAPTERS, GOALIE_CHAPTERS):
        for ch in chapters:
            _add(f"ch:{ch.id}", f"{ch.label} rating", "chapter", extra.get(ch.id, []) + [ch.id, ch.label.lower()],
                 goalie=True if ch.goalie_only else (False if ch.skater_only or chapters is SKATER_CHAPTERS else None))
            for hc in ch.hidden:
                phrases = [hc.id.replace("_", " "), hc.label.lower()]
                if hc.id == "locker_room_presence":
                    phrases += ["carries the locker room", "locker room", "locker room presence"]
                if hc.id == "leadership":
                    phrases += ["leadership", "captain material", "best leader"]
                if hc.id == "top_speed":
                    phrases += ["speed", "fastest", "quickest", "speedster"]
                if hc.id == "net_front":
                    phrases += ["net front", "net front presence", "crease"]
                if hc.id == "shooting_accuracy":
                    phrases += ["accuracy", "shooting"]
                if hc.id == "passing":
                    phrases += ["passer", "passing"]
                if hc.id == "pk_ability":
                    phrases += ["penalty killer", "penalty kill", "pk guy", "pk specialist", "shorthanded"]
                if hc.id == "pp_offence":
                    phrases += ["power play quarterback", "pp quarterback", "pp qb", "power play specialist", "pp specialist"]
                _add(f"hd:{hc.id}", hc.label, "hidden", phrases,
                     goalie=True if ch.goalie_only else (False if chapters is SKATER_CHAPTERS else None))
                # A sub-rating's own name beats a chapter synonym ("hockey iq" is the
                # hockey_iq rating, not the whole mental chapter).
                for own in (hc.id.replace("_", " "), hc.label.lower()):
                    PHRASES[" ".join(own.replace("-", " ").replace("/", " ").split())] = f"hd:{hc.id}"


_register_chapters()
_PHRASE_LIST: List[Tuple[List[str], str]] = sorted(((k.split(), v) for k, v in PHRASES.items()), key=lambda kv: -len(kv[0]))


def find_metric(words: List[str]) -> Optional[str]:
    """Longest phrase in the question that names a metric.

    Overall is the fallback: "worst transition rating" normalises to
    "transition overall", and the transition chapter is what was meant.
    """
    found: List[str] = []
    for parts, name in _PHRASE_LIST:
        n = len(parts)
        for i in range(len(words) - n + 1):
            if words[i:i + n] == parts:
                found.append(name)
                break
    specific = [f for f in found if f != "ovr"]
    if specific:
        return specific[0]
    return found[0] if found else None


def find_metrics(words: List[str]) -> List[str]:
    """Every metric the question names, longest phrases first, without overlapping words.

    "highest ceiling and best development speed" -> [ceiling, development speed].
    """
    taken = [False] * len(words)
    out: List[str] = []
    for parts, name in _PHRASE_LIST:
        n = len(parts)
        for i in range(len(words) - n + 1):
            if words[i:i + n] == parts and not any(taken[i:i + n]):
                for j in range(i, i + n):
                    taken[j] = True
                if name not in out:
                    out.append(name)
    order = {w: k for k, w in enumerate(out)}
    specific = [m for m in out if m != "ovr"]
    return sorted(specific or out, key=lambda m: order[m])


def metric_value(view: Any, pid: str, name: str) -> Optional[float]:
    """The metric for one player, or None when it does not apply or is unknown."""
    row = view.row(pid)
    if not row:
        return None
    m = METRICS.get(name) or {}
    if m.get("goalie") is True and row.get("pos") != "G":
        return None
    if m.get("goalie") is False and row.get("pos") == "G":
        return None
    if (row.get("gp") or 0) < int(m.get("min_gp") or 0) and m.get("kind") == "stat":
        return None
    if name.startswith("ch:") or name.startswith("hd:"):
        p = view.players.get(pid)
        if p is None:
            return None
        if name.startswith("ch:"):
            try:
                from app.sim_engine.entities.chapter_attributes import get_player_chapters

                v = get_player_chapters(p).get(name[3:])
            except Exception:
                v = None
        else:
            prof = getattr(p, "attribute_profile", None)
            v = ((prof or {}).get("hidden") or {}).get(name[3:]) if isinstance(prof, dict) else None
            if v is None:
                # Potential sub-ratings (breakout/bust probability...) live on the career seeds.
                v = getattr(getattr(p, "career", None), name[3:], None)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v * 99.0 if 0 < v <= 1.5 else v
    if name == "upside":
        return round((row.get("pot") or 0) - (row.get("ovr") or 0), 1)
    if name == "cap":
        return (row.get("contract") or {}).get("cap_hit_m") or None
    if name == "value":
        cap = (row.get("contract") or {}).get("cap_hit_m") or 0
        if cap <= 0:
            return None
        try:
            from services.contract_economy import compute_market_value

            return round(float(compute_market_value(view.players[pid], view.league)) - cap, 2)
        except Exception:
            return None
    if name == "morale":
        p = view.players.get(pid)
        psych = getattr(p, "psych", None)
        v = getattr(psych, "morale", None)
        w = getattr(p, "_world_morale_100", None)
        if w is not None:
            return float(w)
        return round(float(v) * 100, 1) if v is not None else None
    if name == "ppg_rate":
        gp = row.get("gp") or 0
        return round((row.get("pts") or 0) / gp, 2) if gp else None
    if name == "fo_pct" and row.get("fo_pct") is None:
        # The season ledger doesn't carry faceoffs; the box-score book does. Only
        # look for players who take draws, so league rankings stay cheap.
        if row.get("pos") not in ("C", "LW", "RW", "F", "W"):
            return None
        tot = ((view.game_log(pid).get("totals") or {}).get("regular_season") or {})
        taken = int(tot.get("fow") or 0) + int(tot.get("fol") or 0)
        return round(tot["fow"] / taken, 3) if taken >= 20 else None
    if name == "cf_pct":
        st = view.stats.get(pid) or {}
        cf, ca = float(st.get("cf") or 0), float(st.get("ca") or 0)
        return round(cf / (cf + ca) * 100, 1) if cf + ca > 20 else None
    v = row.get(name)
    return float(v) if isinstance(v, (int, float)) else None


def fmt(name: str, v: Optional[float]) -> str:
    if v is None:
        return "—"
    if name == "cap":
        return f"${v:.2f}M"
    if name == "value":
        return f"{v:+.2f}M"
    if name == "sv_pct":
        return f"{v:.3f}"
    if name in ("cf_pct", "fo_pct"):
        return f"{v:.1f}%" if v > 1 else f"{v * 100:.1f}%"
    if name in ("ppg_rate", "toi_pg", "upside"):
        return f"{v:.2f}" if name == "ppg_rate" else f"{v:.1f}"
    return f"{v:.0f}"
