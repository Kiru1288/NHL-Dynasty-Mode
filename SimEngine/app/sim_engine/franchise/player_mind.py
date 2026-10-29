"""Player inner life: emotions, mood, beliefs, goals, cause log, journal, event ledger.

Layers on top of the Universe player entity (``session.universe_players[pid]``):

* emotions  — short-lived feelings (pride, anger, hurt...) that fade over days
* mood      — slow drift fed by emotions; nudges ``state.morale`` a little each day
* beliefs   — what the player THINKS (the GM backs me, I'm being shopped...). They
              can be wrong: they move toward the truth slowly, so a misread lingers
* goals     — one or two things he is after; storylines fire when one is met or blocked
* cause_log — every change made here is stored with what caused it
* journal   — first-person lines written from what he felt
* ledger    — ``session.event_ledger``: ordered, seeded record of every mind event

Every state change goes through ``change`` so it is tagged with a cause.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# kind -> (valence -1..1, half-life in days)
EMOTIONS: Dict[str, Tuple[float, float]] = {
    "pride": (0.8, 4.0),
    "relief": (0.6, 2.5),
    "gratitude": (0.7, 5.0),
    "excitement": (0.7, 3.0),
    "anger": (-0.9, 3.0),
    "hurt": (-0.7, 6.0),
    "embarrassment": (-0.6, 2.0),
    "anxiety": (-0.6, 5.0),
    "frustration": (-0.7, 4.0),
    "jealousy": (-0.5, 5.0),
}

# belief -> neutral starting value (0..100)
BELIEFS: Dict[str, float] = {
    "gm_backs_me": 55.0,       # the GM is in my corner
    "being_shopped": 15.0,     # the club is trying to move me
    "role_secure": 55.0,       # my spot in the lineup is safe
    "coach_rates_me": 55.0,    # the coach thinks I'm good
    "club_wants_me_long": 50.0,  # they want me here beyond this deal
}

LEDGER_CAP = 3000
CAUSE_LOG_CAP = 40
JOURNAL_CAP = 20
MOOD_TO_MORALE = 0.35  # morale points per day at full mood


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))


def _day(session: Any) -> Tuple[int, str, int]:
    from app.sim_engine.franchise.storyline_engine import _u_current_meta  # noqa: WPS433

    return _u_current_meta(session)


def mind_rng(session: Any, *parts: Any) -> random.Random:
    """Named, seeded stream: the same save + parts always gives the same rolls (replayable)."""
    from app.sim_engine.franchise.storyline_engine import _u_seed  # noqa: WPS433

    return random.Random(_u_seed("mind", getattr(session, "user_team_id", ""), *parts))


def ensure_mind(entity: Dict[str, Any]) -> Dict[str, Any]:
    mind = entity.get("mind")
    if not isinstance(mind, dict):
        mind = {}
        entity["mind"] = mind
    mind.setdefault("emotions", [])
    mind.setdefault("mood", 0.0)
    beliefs = mind.setdefault("beliefs", {})
    state = entity.get("state") or {}
    for key, base in BELIEFS.items():
        if key not in beliefs:
            # Seed from what is true today so a new save starts without misreads.
            if key == "gm_backs_me":
                base = float(state.get("gm_trust", base) or base)
            elif key == "coach_rates_me":
                base = float(state.get("coach_trust", base) or base)
            elif key == "role_secure":
                base = float(state.get("role_satisfaction", base) or base)
            beliefs[key] = round(_clip(base), 1)
    mind.setdefault("goals", [])
    mind.setdefault("journal", [])
    mind.setdefault("cause_log", [])
    return mind


def entity_for(session: Any, player_id: str) -> Optional[Dict[str, Any]]:
    ent = (getattr(session, "universe_players", None) or {}).get(str(player_id or ""))
    return ent if isinstance(ent, dict) else None


# ---------------------------------------------------------------- ledger / causes

def ledger_append(session: Any, kind: str, player_id: str, cause: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ledger = getattr(session, "event_ledger", None)
    if not isinstance(ledger, list):
        ledger = []
        session.event_ledger = ledger
    seq = int(getattr(session, "event_ledger_seq", 0) or 0) + 1
    session.event_ledger_seq = seq
    day, iso, season = _day(session)
    row = {
        "seq": seq, "day": int(day), "iso": iso, "season": int(season),
        "kind": kind, "player_id": str(player_id or ""), "cause": str(cause or ""),
        "data": dict(data or {}),
    }
    ledger.append(row)
    if len(ledger) > LEDGER_CAP:
        del ledger[: len(ledger) - LEDGER_CAP]
    return row


def change(session: Any, entity: Dict[str, Any], field: str, delta: float, cause: str) -> Dict[str, Any]:
    """Apply a profile delta and record why it happened."""
    from app.sim_engine.franchise.storyline_engine import _u_apply_profile_delta  # noqa: WPS433

    receipt = _u_apply_profile_delta(entity, field, float(delta))
    if abs(float(receipt.get("delta") or 0)) < 0.01:
        return receipt
    mind = ensure_mind(entity)
    day, _, _ = _day(session)
    log = mind["cause_log"]
    # Fold daily mood drift into one row per day so it doesn't bury real causes.
    if log and log[-1].get("field") == field and log[-1].get("cause") == cause and log[-1].get("day") == day:
        log[-1]["delta"] = round(float(log[-1]["delta"]) + float(receipt["delta"]), 2)
    else:
        log.append({"day": int(day), "field": field, "delta": float(receipt["delta"]), "cause": cause})
    if len(log) > CAUSE_LOG_CAP:
        del log[: len(log) - CAUSE_LOG_CAP]
    if not cause.startswith("mood"):
        ledger_append(session, "change", str(entity.get("player_id") or ""), cause,
                      {"field": field, "delta": receipt["delta"]})
    return receipt


def why(entity: Dict[str, Any], field: str = "state.morale", limit: int = 3) -> List[Dict[str, Any]]:
    """The biggest recent causes behind a field, largest first ('his morale fell because of X')."""
    log = [r for r in ensure_mind(entity)["cause_log"] if r.get("field") == field]
    return sorted(log[-15:], key=lambda r: -abs(float(r.get("delta") or 0)))[:limit]


# ---------------------------------------------------------------- emotions / journal

_JOURNAL_LINES: Dict[str, List[str]] = {
    "pride": ["{cause}. Felt good to hear that.", "{cause}. That one stays with me.", "{cause}. Nice to be noticed."],
    "relief": ["{cause}. Weight off my shoulders.", "{cause}. Finally."],
    "gratitude": ["{cause}. Not every GM does that.", "{cause}. I owe them one."],
    "excitement": ["{cause}. Can't wait.", "{cause}. This is the chance."],
    "anger": ["{cause}. Still steaming.", "{cause}. Not how you treat people."],
    "hurt": ["{cause}. Thought they had my back.", "{cause}. That stung more than I'll say."],
    "embarrassment": ["{cause}. Everyone read it.", "{cause}. Rough day to look at my phone."],
    "anxiety": ["{cause}. Can't stop thinking about it.", "{cause}. Keeping my bags half packed."],
    "frustration": ["{cause}. Doing everything they ask and nothing changes.", "{cause}. Something has to give."],
    "jealousy": ["{cause}. What about the rest of us?", "{cause}. Guess we know who the favourite is."],
}


def write_journal(session: Any, entity: Dict[str, Any], kind: str, cause_text: str) -> str:
    mind = ensure_mind(entity)
    day, iso, _ = _day(session)
    pool = _JOURNAL_LINES.get(kind) or ["{cause}."]
    rng = mind_rng(session, "journal", entity.get("player_id"), day, kind, cause_text)
    cause_clean = cause_text.rstrip(".")
    line = rng.choice(pool)
    if cause_clean.endswith(("\u201d", '"', "!", "?")):
        line = line.replace("{cause}.", "{cause}")
    text = line.format(cause=cause_clean)
    mind["journal"].append({"day": int(day), "iso": iso, "emotion": kind, "text": text})
    if len(mind["journal"]) > JOURNAL_CAP:
        del mind["journal"][: len(mind["journal"]) - JOURNAL_CAP]
    return text


def feel(session: Any, entity: Dict[str, Any], kind: str, intensity: float, cause: str, *, journal: bool = True) -> Dict[str, Any]:
    """A short-lived emotion. It fades by its half-life and feeds mood while it lasts."""
    if kind not in EMOTIONS:
        return {}
    mind = ensure_mind(entity)
    day, _, _ = _day(session)
    # Personality: volatile players feel harder, composed ones less.
    vol = float((entity.get("personality") or {}).get("volatility", 50) or 50)
    scaled = _clip(float(intensity) * (0.75 + vol / 200.0), 0, 100)
    for e in mind["emotions"]:
        if e.get("kind") == kind and e.get("cause") == cause:
            e["intensity"] = round(_clip(float(e["intensity"]) + scaled * 0.5), 1)
            return e
    row = {"kind": kind, "intensity": round(scaled, 1), "cause": cause, "day": int(day)}
    mind["emotions"].append(row)
    mind["emotions"] = sorted(mind["emotions"], key=lambda e: -float(e["intensity"]))[:6]
    ledger_append(session, "emotion", str(entity.get("player_id") or ""), cause, {"kind": kind, "intensity": row["intensity"]})
    if journal and scaled >= 18:
        write_journal(session, entity, kind, cause)
    return row


def dominant_emotion(entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    em = [e for e in ensure_mind(entity)["emotions"] if float(e.get("intensity") or 0) >= 8]
    return max(em, key=lambda e: float(e["intensity"])) if em else None


def shift_belief(session: Any, entity: Dict[str, Any], key: str, delta: float, cause: str) -> float:
    mind = ensure_mind(entity)
    before = float(mind["beliefs"].get(key, BELIEFS.get(key, 50.0)))
    after = round(_clip(before + float(delta)), 1)
    mind["beliefs"][key] = after
    if abs(after - before) >= 0.5:
        ledger_append(session, "belief", str(entity.get("player_id") or ""), cause, {"belief": key, "before": before, "after": after})
    return after


# ---------------------------------------------------------------- goals

GOAL_LABELS = {
    "get_paid": "Land a big new contract",
    "stay_home": "Sign long term and stay here",
    "earn_bigger_role": "Earn a bigger role",
    "win_cup": "Win a Cup before it's over",
    "make_the_jump": "Make the jump to a full-time NHL job",
}


def _contract_years_left(player: Any) -> int:
    c = getattr(player, "contract", None)
    if isinstance(c, dict):
        try:
            return int(c.get("years_remaining") or 0)
        except (TypeError, ValueError):
            return 0
    return int(getattr(c, "years_remaining", 0) or 0) if c is not None else 0


def _has_pending_extension(player: Any) -> bool:
    c = getattr(player, "contract", None)
    return bool(isinstance(c, dict) and (c.get("pending_extension") or c.get("extension_aav_m")))


def assign_goals(session: Any, entity: Dict[str, Any], player: Any) -> List[Dict[str, Any]]:
    """Pick 1-2 goals from age, contract, role and personality. Kept until met or dropped."""
    from app.sim_engine.franchise.storyline_engine import _player_age, _player_ovr99  # noqa: WPS433

    mind = ensure_mind(entity)
    live = [g for g in mind["goals"] if g.get("status") == "active"]
    if len(live) >= 2:
        return live
    day0, _, season0 = _day(session)
    have = {g["kind"] for g in live} | {
        g["kind"] for g in mind["goals"] if g.get("status") != "active" and int(g.get("season") or 0) == int(season0)
    }
    age = _player_age(player)
    ovr = _player_ovr99(player)
    pers = entity.get("personality") or {}
    state = entity.get("state") or {}
    life = entity.get("life") or {}
    years = _contract_years_left(player)
    wants: List[Tuple[float, str]] = []
    if years <= 1 and not _has_pending_extension(player):
        loyal = float(pers.get("loyalty", 50) or 50) + float(life.get("city_attachment", 50) or 50) * 0.5
        money = float(pers.get("money_focus", 45) or 45)
        wants.append((money + 20, "get_paid") if money >= loyal * 0.66 else (loyal * 0.66 + 10, "stay_home"))
    if age >= 31 and ovr >= 78:
        wants.append((40 + (age - 30) * 6 + float(pers.get("competitiveness", 50) or 50) * 0.3, "win_cup"))
    if float(state.get("role_satisfaction", 60) or 60) < 52 and age <= 31:
        wants.append((70 - float(state.get("role_satisfaction", 60) or 60) + 20, "earn_bigger_role"))
    team_now = _team(session, str(entity.get("team_id") or ""))
    on_nhl_now = any(p is player for p in list(getattr(team_now, "roster", None) or []))
    if age <= 22 and not on_nhl_now:
        wants.append((45, "make_the_jump"))
    day, _, season = _day(session)
    for _score, kind in sorted(wants, reverse=True):
        if kind in have or len(live) >= 2:
            continue
        goal = {"kind": kind, "label": GOAL_LABELS[kind], "status": "active", "progress": 0.0,
                "set_day": int(day), "season": int(season), "last_check_day": int(day)}
        mind["goals"].append(goal)
        live.append(goal)
        ledger_append(session, "goal_set", str(entity.get("player_id") or ""), kind, {})
    mind["goals"] = mind["goals"][-8:]
    return live


def active_goal(entity: Optional[Dict[str, Any]], kind: str) -> bool:
    if not entity:
        return False
    return any(g.get("kind") == kind and g.get("status") == "active" for g in ensure_mind(entity)["goals"])


def _goal_storyline(session: Any, entity: Dict[str, Any], team_id: str, goal: Dict[str, Any], outcome: str, detail: str) -> None:
    if str(team_id) != str(getattr(session, "user_team_id", "") or ""):
        return
    from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433

    name = str(entity.get("player_name") or "He")
    if outcome == "met":
        headline = f"{name} reaches a personal goal: {goal['label'].lower()}"
        tone, prio = "positive", "MEDIUM"
    else:
        headline = f"{name} frustrated: {goal['label'].lower()} is slipping away"
        tone, prio = "negative", "HIGH"
    _record_storyline(session, {
        "headline": headline, "summary": detail, "team_id": team_id,
        "player_id": entity.get("player_id"), "player_name": name,
        "category": "player_goal", "type": "player_goal", "cause_type": f"GOAL_{outcome.upper()}",
        "priority": prio, "heat": 58 if outcome == "blocked" else 45, "tone": tone,
    })


def _check_goal(session: Any, entity: Dict[str, Any], player: Any, team_id: str, goal: Dict[str, Any], rank: int) -> None:
    state = entity.get("state") or {}
    kind = goal["kind"]
    name = str(entity.get("player_name") or "He")
    before = float(goal.get("progress") or 0)
    met = blocked = False
    detail = ""
    if kind in ("get_paid", "stay_home"):
        if _has_pending_extension(player) or _contract_years_left(player) >= 2:
            met, detail = True, f"{name} has his next contract."
        else:
            goal["progress"] = round(min(0.95, before + 0.05), 2)
            phase = str(getattr(session, "phase", "") or "")
            blocked = phase == "playoffs" and kind == "stay_home" and before >= 0.5
            detail = "Season nearly over and no extension on the table."
    elif kind == "earn_bigger_role":
        sat = float(state.get("role_satisfaction", 50) or 50)
        goal["progress"] = round(_clip(sat, 0, 100) / 100.0, 2)
        met = sat >= 66
        blocked = sat <= 35 and before > 0
        detail = f"{name} still isn't getting the minutes he wants." if blocked else f"{name} has the role he wanted."
    elif kind == "win_cup":
        goal["progress"] = round(max(0.0, (33 - rank) / 32.0), 2)
        gp = _team_games_played(session, team_id)
        blocked = gp >= 30 and rank >= 24 and str(getattr(session, "phase", "")) == "regular"
        detail = f"{name}'s club sits #{rank}. Time is running out on his window."
    elif kind == "make_the_jump":
        on_nhl = any(p is player for p in list(getattr(_team(session, team_id), "roster", None) or []))
        goal["progress"] = 1.0 if on_nhl else round(min(0.9, before + 0.03), 2)
        met = on_nhl
        detail = f"{name} is an NHL regular now."
    cause = f"goal:{kind}"
    if met:
        goal["status"] = "met"
        feel(session, entity, "pride" if kind != "get_paid" else "relief", 40, f"{goal['label']} — done")
        change(session, entity, "state.morale", 4, cause)
        ledger_append(session, "goal_met", str(entity.get("player_id") or ""), kind, {})
        _goal_storyline(session, entity, team_id, goal, "met", detail)
    elif blocked and not goal.get("blocked_flagged"):
        goal["blocked_flagged"] = True
        feel(session, entity, "frustration", 45, f"{goal['label']} is slipping away")
        change(session, entity, "state.morale", -4, cause)
        if kind in ("earn_bigger_role", "win_cup"):
            shift_belief(session, entity, "club_wants_me_long", -6, cause)
        ledger_append(session, "goal_blocked", str(entity.get("player_id") or ""), kind, {})
        _goal_storyline(session, entity, team_id, goal, "blocked", detail)


def _team_games_played(session: Any, team_id: str) -> int:
    from app.sim_engine.franchise.storyline_engine import _team_record  # noqa: WPS433

    try:
        w, l, o, _rec = _team_record(session, str(team_id))
        return int(w + l + o)
    except Exception:
        return 0


def _team(session: Any, team_id: str) -> Any:
    return (getattr(session, "team_by_id", None) or {}).get(str(team_id))


# ---------------------------------------------------------------- daily tick

def tick_minds(session: Any, days: int = 1) -> Dict[str, int]:
    """Fade emotions, drift mood into morale, settle beliefs toward the truth, check goals."""
    from app.sim_engine.franchise.storyline_engine import (  # noqa: WPS433
        _build_standings_rank_map,
        _u_all_players,
    )

    days = max(1, min(60, int(days)))
    ents = getattr(session, "universe_players", None) or {}
    day, _, _ = _day(session)
    weekly = int(day) - int(getattr(session, "_mind_goal_check_day", -99) or -99) >= 7
    if weekly:
        session._mind_goal_check_day = int(day)
    ranks = _build_standings_rank_map(session) if weekly else {}
    counts = {"emotions_faded": 0, "goals_checked": 0}
    for team_id, player in _u_all_players(session):
        ent = ents.get(str(getattr(player, "id", "") or ""))
        if not isinstance(ent, dict):
            continue
        mind = ensure_mind(ent)
        valence = 0.0
        keep = []
        for e in mind["emotions"]:
            _val, half = EMOTIONS.get(e.get("kind"), (0.0, 3.0))
            e["intensity"] = round(float(e["intensity"]) * (0.5 ** (days / half)), 1)
            if e["intensity"] >= 3:
                keep.append(e)
                valence += _val * float(e["intensity"]) / 100.0
            else:
                counts["emotions_faded"] += 1
        mind["emotions"] = keep
        # Mood: slow average of what he's been feeling (weeks, not days).
        k = 1 - 0.88 ** days
        mind["mood"] = round(max(-1.0, min(1.0, float(mind["mood"]) * (1 - k) + max(-1.0, min(1.0, valence * 2.0)) * k)), 3)
        if abs(mind["mood"]) >= 0.05:
            cause = "mood:" + (dominant_emotion(ent) or {}).get("cause", "how he's been feeling")
            change(session, ent, "state.morale", mind["mood"] * MOOD_TO_MORALE * days, cause)
        # Beliefs drift to the truth slowly: a misread can last a month.
        state = ent.get("state") or {}
        truth = {
            "gm_backs_me": float(state.get("gm_trust", 55) or 55),
            "coach_rates_me": float(state.get("coach_trust", 55) or 55),
            "role_secure": float(state.get("role_satisfaction", 55) or 55),
        }
        settle = 1 - 0.97 ** days
        for key, target in truth.items():
            cur = float(mind["beliefs"].get(key, target))
            mind["beliefs"][key] = round(cur + (target - cur) * settle, 1)
        shop = float(mind["beliefs"].get("being_shopped", 15))
        mind["beliefs"]["being_shopped"] = round(shop + (15 - shop) * (1 - 0.96 ** days), 1)
        # Believing you're being shopped erodes trust in the GM, true or not.
        if shop >= 55:
            change(session, ent, "state.gm_trust", -0.25 * days, "belief:thinks he is being shopped")
        if weekly:
            assign_goals(session, ent, player)
            for g in [g for g in mind["goals"] if g.get("status") == "active"]:
                _check_goal(session, ent, player, str(team_id), g, int(ranks.get(str(team_id)) or 16))
                counts["goals_checked"] += 1
    return counts


# ---------------------------------------------------------------- willingness to talk

def willingness(entity: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Does he want to sit down with the GM? willing / curt / agent / refuse."""
    if not entity:
        return {"mode": "willing", "reason": ""}
    mind = ensure_mind(entity)
    backs = float(mind["beliefs"].get("gm_backs_me", 55))
    anger = sum(float(e["intensity"]) for e in mind["emotions"] if e.get("kind") in ("anger", "hurt"))
    shopped = float(mind["beliefs"].get("being_shopped", 15))
    top = dominant_emotion(entity) or {}
    because = str(top.get("cause") or "")
    if anger >= 70 or (backs <= 22 and anger >= 30):
        return {"mode": "refuse", "reason": f"He isn't ready to talk yet. Still upset: {because}." if because else "He isn't ready to talk yet."}
    if shopped >= 65 or backs <= 28:
        return {"mode": "agent", "reason": "He'd rather his agent handle it — he thinks the club is moving him." if shopped >= 65 else "He sent his agent instead. He doesn't think you're in his corner."}
    if anger >= 35 or backs <= 40:
        return {"mode": "curt", "reason": f"He came in, but he's short with you{(' — ' + because) if because else ''}."}
    return {"mode": "willing", "reason": ""}


def extension_reluctance(entity: Optional[Dict[str, Any]]) -> float:
    """Extra chance he won't extend mid-season: a 'get paid' goal waits for the market."""
    if not entity:
        return 0.0
    extra = 0.0
    if active_goal(entity, "get_paid"):
        extra += 0.15
    if active_goal(entity, "stay_home"):
        extra -= 0.12
    mind = ensure_mind(entity)
    if float(mind["beliefs"].get("being_shopped", 15)) >= 55:
        extra += 0.10
    if float(mind["beliefs"].get("club_wants_me_long", 50)) >= 65:
        extra -= 0.05
    return extra


def snapshot(entity: Dict[str, Any]) -> Dict[str, Any]:
    """UI / PuckGPT view of one player's inner state."""
    mind = ensure_mind(entity)
    return {
        "mood": mind["mood"],
        "emotions": list(mind["emotions"]),
        "beliefs": dict(mind["beliefs"]),
        "goals": [g for g in mind["goals"] if g.get("status") == "active"],
        "journal": list(mind["journal"][-6:]),
        "morale_causes": why(entity, "state.morale"),
        "willingness": willingness(entity),
    }
