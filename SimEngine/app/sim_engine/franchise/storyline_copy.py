"""Adaptive headline/body templates for data-driven franchise storylines."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

GOALIE_HEATER_SV_FLOOR = 0.880
GOALIE_MELTDOWN_SV_CEILING = 0.940

# window_days, max league public stories in that window
_HIGH_VOLUME_CAPS: Dict[str, Tuple[int, int]] = {
    "GOALIE_HEATER": (1, 2),
    "SHUTOUT": (1, 1),
    "SUPERSTAR_CARRY": (1, 2),
    "COMMUNITY_LIFT": (1, 1),
    "REPORTER_CONFRONT": (3, 1),
}

HEADLINES: Dict[str, List[str]] = {
    "star_underperforming": [
        "{name} enters witness protection program on the scoresheet",
        "Local ${cap:.1f}M forward reportedly still searching for the net",
        "{name}'s scoring lamp remains cold",
        "Top-line {role} producing at the pace of a decorative traffic cone",
        "The ${cap:.1f}M man has gone missing",
        "{name}'s drought becoming impossible to ignore",
        "Scoring drought becoming impossible to ignore",
    ],
    "rookie_breakout": [
        "{name} refuses to wait politely for his development timeline",
        "Rookie {name} is forcing the coaching staff to rewrite the depth chart",
        "{team} rookie {name} outpacing every internal projection",
    ],
    "superstar_carrying": [
        "{name} accidentally becomes the entire offense",
        "{name} is the only reason {team}'s record isn't worse",
        "Star {name} carrying {team} like it's a second job",
    ],
    "contract_pressure": [
        "Cap hit remains elite, production has filed a missing persons report",
        "${cap:.1f}M AAV vs {pts} points — the math isn't mathing",
        "{name}'s contract is louder than his box score",
    ],
    "goalie_meltdown": [
        "Goalie has apparently decided the puck deserves freedom",
        "{name} letting in goals like the net is a suggestion",
        "{team} crease in crisis — {name} at {sv:.3f} SV%",
    ],
    "goalie_heater": [
        "Netminder commits grand larceny, steals another two points",
        "{name} rolling at elite save percentage over {gp} starts",
        "{team} goalie {name} is stealing games nightly",
    ],
    "backup_taking_net": [
        "Backup goalie has politely stolen the starter's office keys",
        "{name} forcing a crease conversation in {team}",
        "Starter seat getting warm — {name} pushing for more starts",
    ],
    "surprise_team": [
        "{team} keeps winning despite the league's best attempts to explain it away",
        "Surprise contender alert: {team} sitting {record}",
        "{team} outperforming every preseason projection",
    ],
    "contender_collapse": [
        "{team} begins annual tradition of making everyone nervous",
        "Paper contender {team} stuck at {record}",
        "{team} collapse has the room playing tight",
    ],
    "playoff_race": [
        "Every point now feels like it comes with a stress tax",
        "{team} bubble watch — {record} and no margin for error",
        "Wild-card math getting cruel for {team}",
    ],
    "losing_skid": [
        "{team} skid has the room playing like the clock is guilty",
        "{team} losing streak turning the dressing room tense",
        "Slide continues — {team} at {record}",
    ],
    "win_streak": [
        "{team} rolling like they found a cheat code for effort",
        "{team} on a heater at {record}",
        "Win streak energy building around {team}",
    ],
}

BODIES: Dict[str, List[str]] = {
    "star_underperforming": [
        "{name} ({ovr:.0f} OVR) has {pts} points in {gp} games ({ppg:.2f} P/GP) vs ~{exp_pts:.0f} expected for role.",
        "Production pace {ppg:.2f} P/GP is well below expected {exp_ppg:.2f} for a {ovr:.0f}-overall {role}.",
    ],
    "rookie_breakout": [
        "{name} is outpacing development expectations with {pts} points in {gp} games.",
        "Rookie scoring pace {ppg:.2f} P/GP exceeds expected {exp_ppg:.2f}.",
    ],
    "superstar_carrying": [
        "{name} is producing like a star on a team sitting {record}.",
        "Elite individual scoring on a sub-.500 club.",
    ],
    "contract_pressure": [
        "${cap:.2f}M AAV vs {pts} points in {gp} games.",
        "High cap hit with well-below-expected production.",
    ],
    "goalie_meltdown": [
        "{name} at {sv:.3f} SV% and {gaa:.2f} GAA over {gp} starts.",
        "Save percentage well below expected baseline for {ovr:.0f} OVR.",
    ],
    "goalie_heater": [
        "{name} rolling at {sv:.3f} SV% over {gp} starts.",
        "Save percentage materially above expected baseline.",
    ],
    "backup_taking_net": [
        "{name} is pushing for a larger share of starts in {team}.",
        "Backup performance is forcing a starter conversation.",
    ],
    "surprise_team": [
        "{team} is outperforming expectations at {record}.",
        "Strong run reflected in {record} record.",
    ],
    "contender_collapse": [
        "Strong on-paper club is {record} through {gp} games.",
        "High expectations meet a losing record.",
    ],
    "playoff_race": [
        "{team} sits around the wild-card line at {record}.",
        "Late-season standings place the club near the cut line.",
    ],
    "losing_skid": [
        "Team record {record} with mounting losses.",
        "Multiple losses in current standings snapshot.",
    ],
    "win_streak": [
        "Strong run reflected in {record} record.",
        "Standings show sustained winning.",
    ],
}


def normalize_save_pct(*values: Any) -> Optional[float]:
    """Return a 0–1 save percentage, or None if missing/invalid."""
    for raw in values:
        if raw is None or raw == "":
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 1.5:
            val = val / 100.0
        if 0.0 < val <= 1.0:
            return round(val, 4)
    return None


def format_sv_pct(value: Any) -> str:
    """NHL wire style: .912"""
    sv = normalize_save_pct(value)
    if sv is None:
        return ""
    return f"{sv:.3f}".replace("0.", ".", 1) if sv < 1 else "1.000"


def valid_goalie_heater_sv(value: Any) -> bool:
    sv = normalize_save_pct(value)
    return sv is not None and sv >= GOALIE_HEATER_SV_FLOOR


def valid_goalie_meltdown_sv(value: Any) -> bool:
    sv = normalize_save_pct(value)
    return sv is not None and 0.50 <= sv <= GOALIE_MELTDOWN_SV_CEILING


def story_volume_slot(event: Optional[Dict[str, Any]] = None, **hints: Any) -> str:
    ev = dict(event or {})
    cause = str(ev.get("cause_type") or hints.get("cause_type") or "").upper()
    stype = str(ev.get("type") or ev.get("stype") or ev.get("kind") or hints.get("stype") or "").lower()
    headline = str(ev.get("headline") or ev.get("title") or hints.get("headline") or "").lower()
    if cause in {"GOALIE_HEATER"} or stype == "goalie_heater":
        return "GOALIE_HEATER"
    if cause == "SHUTOUT" or stype == "shutout":
        return "SHUTOUT"
    if cause in {"SUPERSTAR_CARRY"} or stype == "superstar_carrying":
        return "SUPERSTAR_CARRY"
    if stype == "community_lift" or cause == "COMMUNITY_MOMENT" or "spends a day on" in headline:
        return "COMMUNITY_LIFT"
    if cause in {"PLAYER_REPORTER_CONFRONTATION", "PLAYER_REPORTER_ALTERCATION"} or stype in (
        "reporter_confrontation",
        "reporter_altercation",
    ):
        return "REPORTER_CONFRONT"
    return ""


def headline_fingerprint(headline: str) -> str:
    text = str(headline or "").lower()
    text = re.sub(r"\.[0-9]{3}", "#", text)
    text = re.sub(r"\d+(\.\d+)?", "#", text)
    text = re.sub(r"[^a-z#\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120]


def claim_league_story_slot(session: Any, event: Dict[str, Any], *, user_club: bool) -> bool:
    """Cap high-volume league copy. User-club stories always publish."""
    if user_club:
        return True
    slot = story_volume_slot(event)
    headline = str(event.get("headline") or event.get("title") or "")
    fp = headline_fingerprint(headline)
    day = int(event.get("calendar_day") or getattr(session, "calendar_cursor", 0) or 0)

    fp_days = getattr(session, "_story_fp_days", None)
    if not isinstance(fp_days, dict):
        fp_days = {}
        try:
            setattr(session, "_story_fp_days", fp_days)
        except Exception:
            fp_days = {}
    if slot and fp and fp_days.get(f"{day}|{fp}"):
        return False

    if slot:
        window, max_n = _HIGH_VOLUME_CAPS.get(slot, (1, 1))
        store = getattr(session, "_story_slot_days", None)
        if not isinstance(store, dict):
            store = {}
            try:
                setattr(session, "_story_slot_days", store)
            except Exception:
                return True
        recent = [int(d) for d in (store.get(slot) or []) if day - int(d) < window]
        if len(recent) >= max_n:
            return False
        recent.append(day)
        store[slot] = recent[-12:]

    if slot and fp:
        fp_days[f"{day}|{fp}"] = True
        if len(fp_days) > 400:
            try:
                session._story_fp_days = dict(list(fp_days.items())[-200:])
            except Exception:
                pass
    return True


def select_daily_data_stories(
    generated: List[Dict[str, Any]],
    user_team_id: str,
    *,
    league_cap: int = 7,
) -> List[Dict[str, Any]]:
    """Keep every user-club data story; take a small, unique league slice."""
    uid = str(user_team_id or "")
    user: List[Dict[str, Any]] = []
    league: List[Dict[str, Any]] = []
    for row in generated or []:
        if not isinstance(row, dict):
            continue
        if not str(row.get("headline") or "").strip():
            continue
        if uid and str(row.get("team_id") or "") == uid:
            user.append(row)
        else:
            league.append(row)
    league.sort(key=lambda s: -int(s.get("heat") or 0))
    kept: List[Dict[str, Any]] = []
    seen_slots: Dict[str, int] = {}
    seen_fp: set = set()
    for row in league:
        slot = story_volume_slot(row)
        fp = headline_fingerprint(str(row.get("headline") or ""))
        if fp and fp in seen_fp:
            continue
        if slot:
            _window, max_n = _HIGH_VOLUME_CAPS.get(slot, (1, 1))
            if seen_slots.get(slot, 0) >= max_n:
                continue
        kept.append(row)
        if fp:
            seen_fp.add(fp)
        if slot:
            seen_slots[slot] = seen_slots.get(slot, 0) + 1
        if len(kept) >= league_cap:
            break
    return user + kept


def note_goalie_shutout(session: Any, player_id: str, day: int) -> int:
    """Record a shutout and return how many this goalie has in the last 7 days, including today."""
    log = getattr(session, "_goalie_shutout_log", None)
    if not isinstance(log, dict):
        log = {}
        try:
            setattr(session, "_goalie_shutout_log", log)
        except Exception:
            return 1
    pid = str(player_id or "")
    if not pid:
        return 1
    days = [int(d) for d in (log.get(pid) or []) if day - int(d) <= 7]
    days.append(int(day))
    log[pid] = days[-12:]
    return len(days)


def story_ctx(**kwargs: Any) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        ctx[key] = value
    if "name" not in ctx and "player_name" in ctx:
        ctx["name"] = ctx["player_name"]
    if "team" not in ctx and "team_name" in ctx:
        ctx["team"] = ctx["team_name"]
    sv = normalize_save_pct(ctx.get("save_pct"), ctx.get("sv"), ctx.get("sv_pct"))
    if sv is not None:
        ctx["save_pct"] = sv
        ctx["sv"] = sv
        ctx["sv_fmt"] = format_sv_pct(sv)
    exp = normalize_save_pct(ctx.get("expected_save_pct"), ctx.get("exp_sv"))
    if exp is not None:
        ctx["expected_save_pct"] = exp
        ctx["exp_sv"] = exp
        ctx["exp_sv_fmt"] = format_sv_pct(exp)
    return ctx


def _format_line(template: str, ctx: Dict[str, Any]) -> str:
    safe = defaultdict(str)
    for key, value in (ctx or {}).items():
        safe[key] = value
    try:
        return str(template).format_map(safe).strip()
    except (KeyError, ValueError, TypeError):
        return str(template).strip()


def pick_line(rng: random.Random, stype: str, ctx: Dict[str, Any], body: bool = False) -> str:
    from app.sim_engine.franchise.storyline_procedural import compose_data_story_copy  # noqa: WPS433

    return compose_data_story_copy(str(stype), dict(ctx or {}), rng, body=body)


def classify_story_lane(
    *,
    cause_type: str = "",
    category: str = "",
    stype: str = "",
    severity: str = "",
    priority: str = "",
    heat: int = 0,
    knowledge_type: str = "",
    public_level: str = "",
    legal_status: str = "",
    incident_family: str = "",
    **_: Any,
) -> str:
    kt = str(knowledge_type or "").lower()
    pl = str(public_level or "").lower()
    ct = str(cause_type or "").upper()
    st = str(stype or "").lower()

    if kt in {"claim", "rumor", "rumour"} or pl in {"rumour", "rumor", "unconfirmed", "whisper"}:
        return "rumor"
    if kt in {"fact", "confirmed"} or pl in {"reported", "confirmed", "verified"}:
        return "recap"
    if "TRADE" in ct or "RUMOR" in ct or "trade" in st:
        return "rumor"
    if "CONDUCT" in ct or legal_status or incident_family:
        return "recap"
    if int(heat or 0) >= 68 and str(priority or "").upper() in {"HIGH", "CRITICAL"}:
        return "recap"
    if str(severity or "").lower() == "major":
        return "recap"
    return "default"


def lane_flags(lane: str) -> Dict[str, Any]:
    lane_key = str(lane or "default")
    return {
        "story_lane": lane_key,
        "is_rumor": lane_key == "rumor",
        "is_recap": lane_key == "recap",
        "is_fact_lane": lane_key == "recap",
    }
