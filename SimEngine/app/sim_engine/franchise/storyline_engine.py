"""
Data-driven franchise storyline engine.

Reads real franchise session data (stats ledger, standings, injuries, draft ranks)
and generates NHL-style storylines with evidence and small sim effects.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.franchise.storyline_copy import (
    classify_story_lane,
    lane_flags,
    pick_line,
    story_ctx,
)

_log = logging.getLogger("uvicorn.error")
_DEV = os.environ.get("NODE_ENV", "") == "development" or os.environ.get("NHL_FRANCHISE_DEBUG", "0") == "1"

COOLDOWN_MINOR_DAYS = 7
COOLDOWN_MAJOR_DAYS = 14
COOLDOWN_CRISIS_DAYS = 10

SKATER_GP_MINOR = 6
SKATER_GP_MAJOR = 10
GOALIE_GP_MINOR = 3
GOALIE_GP_MAJOR = 6
TEAM_GP_MIN = 8


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _stat_int(row: Dict[str, Any], *keys: str, default: int = 0) -> int:
    for k in keys:
        if k in row and row[k] is not None and row[k] != "":
            try:
                return int(row[k])
            except (TypeError, ValueError):
                pass
    return default


def _pos_bucket(pos: str) -> str:
    p = str(pos or "F").upper()
    if p in ("G", "GK", "GOALIE"):
        return "G"
    if p in ("D", "LD", "RD", "DEF", "DEFENSE"):
        return "D"
    return "F"


def _expected_points_per_game(ovr: float, pos: str, age: int, is_rookie: bool) -> float:
    """Expected scoring pace from overall + role bucket (not fake stats)."""
    bucket = _pos_bucket(pos)
    if bucket == "G":
        return 0.0
    base = max(0.08, (ovr - 62.0) / 38.0)
    if bucket == "D":
        base *= 0.52
    elif bucket == "F":
        base *= 0.88
    if is_rookie:
        base *= 0.72
    if age >= 33:
        base *= 0.92
    return round(max(0.12, min(1.35, base * 0.95)), 3)


def _expected_save_pct(ovr: float) -> float:
    return round(0.870 + (ovr - 70.0) * 0.0018, 3)


def _storyline_id(stable_key: str) -> str:
    digest = hashlib.sha1(stable_key.encode("utf-8", "ignore")).hexdigest()[:14]
    return f"sl_{digest}"


def _headline_pick(rng: random.Random, pool: List[str]) -> str:
    return str(rng.choice(pool)) if pool else "Story developing"


def _effect_summary(effects: Dict[str, Any]) -> str:
    parts: List[str] = []
    label = {
        "player_confidence": "Player confidence",
        "player_morale": "Player morale",
        "team_morale": "Team morale",
        "media_pressure": "Media pressure",
        "fan_confidence": "Fan confidence",
        "trade_market_heat": "Trade market heat",
        "trade_value": "Trade value",
        "room_tension": "Room tension",
        "development_confidence": "Development confidence",
        "draft_stock": "Draft stock",
        "scouting_uncertainty": "Scouting uncertainty",
        "coach_trust": "Coach trust",
        "lineup_pressure": "Lineup pressure",
        "goalie_confidence": "Goalie confidence",
        "owner_patience": "Owner patience",
        "coach_security": "Coach security",
        "depth_pressure": "Depth pressure",
    }
    for k, v in sorted(effects.items()):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if abs(fv) < 0.01:
            continue
        sign = "+" if fv > 0 else ""
        parts.append(f"{label.get(k, k.replace('_', ' ').title())} {sign}{fv:.0f}")
    return " · ".join(parts) if parts else "Minor narrative ripple"


def _cooldown_state(session: Any) -> Dict[str, Dict[str, Any]]:
    st = getattr(session, "_storyline_engine_cooldowns", None)
    if not isinstance(st, dict):
        st = {}
        setattr(session, "_storyline_engine_cooldowns", st)
    return st


def _can_fire(session: Any, stable_key: str, cur_day: int, severity: str) -> Tuple[bool, int]:
    """Return (allowed, repeat_count). Escalation bypasses duplicate block if severity worsened."""
    cd = _cooldown_state(session)
    prev = cd.get(stable_key)
    if not prev:
        return True, 0
    last = int(prev.get("last_day", -999))
    rep = int(prev.get("repeat_count", 0))
    prev_sev = str(prev.get("severity") or "minor")
    window = COOLDOWN_MINOR_DAYS if severity == "minor" else COOLDOWN_MAJOR_DAYS
    if severity == "crisis":
        window = COOLDOWN_CRISIS_DAYS
    if cur_day - last < window and prev_sev == severity:
        return False, rep
    return True, rep


def _mark_fired(session: Any, stable_key: str, cur_day: int, severity: str, repeat_count: int) -> None:
    cd = _cooldown_state(session)
    cd[stable_key] = {
        "last_day": int(cur_day),
        "severity": str(severity),
        "repeat_count": int(repeat_count) + 1,
    }


def _player_from_roster(session: Any, player_id: str) -> Optional[Any]:
    pid = str(player_id or "")
    if not pid:
        return None
    idx = _player_index_by_id(session)
    return idx.get(pid)


def _player_index_by_id(session: Any) -> Dict[str, Any]:
    """One roster walk per stats revision — O(1) player lookup for storyline passes."""
    rev = int(getattr(session, "_stats_revision", 0) or 0)
    cached_rev = int(getattr(session, "_player_index_rev", -1) or -1)
    cached = getattr(session, "_player_index_by_id", None)
    if isinstance(cached, dict) and cached_rev == rev:
        return cached
    out: Dict[str, Any] = {}
    for tm in (getattr(session, "team_by_id", None) or {}).values():
        for bucket in ("roster", "ahl_roster", "injured_reserve", "scratches", "echl_roster"):
            for p in getattr(tm, bucket, None) or []:
                pid = str(getattr(p, "id", "") or "")
                if pid:
                    out[pid] = p
    session._player_index_by_id = out
    session._player_index_rev = rev
    return out


def _build_standings_rank_map(session: Any) -> Dict[str, int]:
    """Sort league standings once; reuse ranks across all storyline triggers."""
    cached_rev = int(getattr(session, "_standings_rank_rev", -1) or -1)
    rev = int(getattr(session, "_stats_revision", 0) or 0)
    cached = getattr(session, "_standings_rank_by_team", None)
    if isinstance(cached, dict) and cached_rev == rev:
        return cached
    st = getattr(session, "standings", None)
    if st is None:
        return {}
    recs = getattr(st, "records", None) or {}
    rows: List[Tuple[int, str]] = []
    if isinstance(recs, dict):
        iter_rows = recs.items()
    elif isinstance(recs, list):
        iter_rows = ((getattr(rr, "team_id", None) or getattr(rr, "id", i), rr) for i, rr in enumerate(recs))
    else:
        return {}
    for tid, rr in iter_rows:
        pts = int(getattr(rr, "points", 0) or 0)
        rows.append((pts, str(tid)))
    rows.sort(key=lambda x: (-x[0], x[1]))
    out = {tid: i + 1 for i, (_, tid) in enumerate(rows)}
    session._standings_rank_by_team = out
    session._standings_rank_rev = rev
    return out


def _league_points_rank(session: Any, team_id: str) -> int:
    ranks = _build_standings_rank_map(session)
    if not ranks:
        return 16
    return int(ranks.get(str(team_id), len(ranks) or 16))


def _player_ovr99(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    try:
        v = float(fn() if callable(fn) else fn or 0)
    except Exception:
        return 0.0
    return v * 99.0 if v <= 1.5 else v


def _player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    try:
        return int(getattr(ident, "age", 26) or 26) if ident is not None else 26
    except Exception:
        return 26


def _cap_hit_m(player: Any) -> float:
    c = getattr(player, "contract", None)
    if c is None:
        return 0.0
    for attr in ("cap_hit_m", "aav_m", "salary_m"):
        try:
            v = float(getattr(c, attr, 0) or 0)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return 0.0


def _apply_storyline_effects(session: Any, team_id: str, player_id: str, effects: Dict[str, Any]) -> None:
    """Apply small narrative effects to player psych / team state when available."""
    scale = 0.01  # effects use integer-ish deltas; psych is 0..1
    player = _player_from_roster(session, player_id) if player_id else None
    if player is not None:
        psych = getattr(player, "psych", None)
        traits = getattr(player, "traits", None)
        if psych is not None:
            for key, attr, mult in (
                ("player_morale", "morale", 1.0),
                ("player_confidence", "confidence_level", 1.0),
                ("goalie_confidence", "confidence_level", 1.0),
                ("development_confidence", "internal_motivation", 0.8),
                ("media_pressure", "media_stress", 0.9),
            ):
                if key in effects:
                    try:
                        dv = float(effects[key]) * scale * mult
                        cur = float(getattr(psych, attr, 0.5))
                        setattr(psych, attr, _clamp(cur + dv))
                    except (TypeError, ValueError):
                        pass
            if hasattr(psych, "clamp_all"):
                psych.clamp_all()
        if traits is not None:
            if "player_confidence" in effects:
                try:
                    traits.confidence = _clamp(float(getattr(traits, "confidence", 0.5)) + float(effects["player_confidence"]) * scale)
                except (TypeError, ValueError):
                    pass
            if hasattr(traits, "clamp_all"):
                traits.clamp_all()

    tm = (getattr(session, "team_by_id", None) or {}).get(str(team_id))
    if tm is not None:
        st = getattr(tm, "state", None)
        if st is not None:
            if "team_morale" in effects:
                try:
                    st.team_morale = _clamp(float(getattr(st, "team_morale", 0.5)) + float(effects["team_morale"]) * scale)
                except (TypeError, ValueError):
                    pass
            if "media_pressure" in effects or "fan_confidence" in effects:
                try:
                    bump = float(effects.get("media_pressure", 0)) - float(effects.get("fan_confidence", 0)) * 0.35
                    st.organizational_pressure = _clamp(float(getattr(st, "organizational_pressure", 0.5)) + bump * scale)
                except (TypeError, ValueError):
                    pass
            if hasattr(st, "clamp"):
                st.clamp()


REDDIT_ENGAGEMENT_WEIGHT = 2.5
UNIVERSE_MAX_REDDIT_THREADS = 150

REDDIT_FAN_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "diehard_optimist": {"tone": "hopeful", "upvote_bias": 1.15},
    "diehard_doomer": {"tone": "critical", "upvote_bias": 1.0},
    "stats_nerd": {"tone": "analytical", "upvote_bias": 1.3},
    "old_guard": {"tone": "nostalgic", "upvote_bias": 0.9},
    "rival_troll": {"tone": "hostile", "upvote_bias": 0.8, "cross_team": True},
}
REDDIT_FLAIRS = ["Rumor", "Confirmed", "Discussion", "Shitpost", "Analysis", "Highlight"]


def apply_fan_engagement_delta(session: Any, team_id: str, delta: float, source: str = "reddit_thread") -> None:
    """Mutate organizational pressure + fan profile from social sentiment."""
    tid = str(team_id or getattr(session, "user_team_id") or "")
    if not tid:
        return
    tm = (getattr(session, "team_by_id", None) or {}).get(tid)
    st = getattr(tm, "state", None) if tm is not None else None
    if st is not None and hasattr(st, "organizational_pressure"):
        try:
            st.organizational_pressure = _clamp(float(getattr(st, "organizational_pressure", 0.5)) - float(delta) * 0.04)
        except (TypeError, ValueError):
            pass
        if hasattr(st, "clamp"):
            try:
                st.clamp()
            except Exception:
                pass
    try:
        from services.franchise_sim import _ensure_team_fan_profile  # noqa: WPS433

        profile = _ensure_team_fan_profile(session, tid)
        profile["fan_confidence"] = max(0.0, min(100.0, float(profile.get("fan_confidence", 55)) + float(delta) * 8.0))
        pulse = list(getattr(session, "reddit_engagement_pulse", None) or [])
        pulse.append({"team_id": tid, "delta": round(float(delta), 4), "source": source})
        session.reddit_engagement_pulse = pulse[-40:]
    except Exception:
        pulse = list(getattr(session, "reddit_engagement_pulse", None) or [])
        pulse.append({"team_id": tid, "delta": round(float(delta), 4), "source": source})
        session.reddit_engagement_pulse = pulse[-40:]


def _build_storyline(
    *,
    rng: random.Random,
    session: Any,
    stable_key: str,
    stype: str,
    category: str,
    severity: str,
    priority: str,
    tone: str,
    headline: str,
    description: str,
    short_summary: str,
    cause: str,
    team_id: str,
    team_name: str,
    player_id: str = "",
    player_name: str = "",
    player_position: str = "",
    player_overall: Optional[float] = None,
    evidence: Dict[str, Any],
    effects: Dict[str, Any],
    requires_action: bool = False,
    action_options: Optional[List[Dict[str, Any]]] = None,
    calendar_iso: str = "",
    cur_day: int = 0,
    heat: int = 50,
    repeat_count: int = 0,
    escalated_from: str = "",
) -> Dict[str, Any]:
    sid = _storyline_id(stable_key)
    fx = {k: float(v) for k, v in effects.items()}
    cause_type_map = {
        "star_underperforming": "PLAYER_LOW_PRODUCTION",
        "cold_streak": "PLAYER_REALDATA_DROP",
        "goalie_meltdown": "GOALIE_BAD_FORM",
        "contract_pressure": "CONTRACT_DISPUTE",
        "losing_skid": "LOSING_STREAK",
        "cold_streak_team": "LOSING_STREAK",
        "win_streak": "WINNING_STREAK",
        "hot_streak_team": "WINNING_STREAK",
        "injury_ripple": "PLAYER_INJURED",
        "goalie_heater": "GOALIE_HEATER",
        "goal_drought": "GOAL_DROUGHT",
        "veteran_fade": "VETERAN_FADE",
        "backup_taking_net": "BACKUP_TAKING_NET",
        "rookie_breakout": "ROOKIE_BREAKOUT",
        "superstar_carrying": "SUPERSTAR_CARRY",
        "surprise_team": "WINNING_STREAK",
        "contender_collapse": "LOSING_STREAK",
        "playoff_race": "WINNING_CONCERN",
    }
    cause_type = cause_type_map.get(str(stype), "")
    lane = classify_story_lane(
        cause_type=cause_type,
        category=category,
        stype=str(stype),
        severity=severity,
        priority=priority,
        heat=heat,
    )
    flags = lane_flags(lane)
    return {
        "id": sid,
        "storyline_id": sid,
        "stable_key": stable_key,
        "type": stype,
        "category": category,
        **flags,
        "knowledge_type": "claim" if lane == "rumor" else ("fact" if lane == "recap" else ""),
        "public_knowledge_level": "rumour" if lane == "rumor" else ("reported" if lane == "recap" else ""),
        "priority": priority,
        "severity": severity,
        "tone": tone,
        "headline": headline,
        "title": headline,
        "description": description,
        "short_summary": short_summary,
        "summary": short_summary,
        "cause": cause,
        "cause_type": cause_type,
        "cause_event_id": stable_key if cause_type else "",
        "effect_summary": _effect_summary(fx),
        "calendar_iso": calendar_iso,
        "date": calendar_iso or cur_day,
        "calendar_day": cur_day,
        "created_at": calendar_iso,
        "expires_on": "",
        "team_id": team_id,
        "team": team_id,
        "team_name": team_name,
        "player_id": player_id,
        "player_name": player_name,
        "player_position": player_position,
        "player_overall": player_overall,
        "players": [player_name] if player_name else [],
        "related_team_ids": [team_id] if team_id else [],
        "related_player_ids": [player_id] if player_id else [],
        "evidence": evidence,
        "effects": fx,
        "requires_action": bool(requires_action),
        "action_options": list(action_options or []),
        "status": "escalating" if repeat_count > 0 else "active",
        "source": "data_storyline_engine",
        "heat": int(heat),
        "credibility": 92 if evidence else 60,
        "repeat_count": int(repeat_count),
        "escalated_from": escalated_from,
    }


def _team_record(session: Any, team_id: str) -> Tuple[int, int, int, str]:
    st = getattr(session, "standings", None)
    if st is None:
        return 0, 0, 0, "0-0-0"
    rec = getattr(st, "records", None) or {}
    rr = None
    if isinstance(rec, dict):
        rr = rec.get(team_id) or rec.get(str(team_id))
    elif isinstance(rec, list):
        for row in rec:
            tid = str(getattr(row, "team_id", None) or getattr(row, "id", "") or "")
            if tid == str(team_id):
                rr = row
                break
    if rr is None:
        return 0, 0, 0, "0-0-0"
    w = int(getattr(rr, "wins", 0) or 0)
    l = int(getattr(rr, "losses", 0) or 0)
    o = int(getattr(rr, "otl", 0) or 0)
    return w, l, o, f"{w}-{l}-{o}"


def _team_games_played(session: Any, team_id: str) -> int:
    w, l, o, _ = _team_record(session, team_id)
    return w + l + o


def _underperform_actions() -> List[Dict[str, Any]]:
    return [
        {"id": "back_publicly", "label": "Publicly back the player", "effects": {"player_confidence": 3, "media_pressure": -1, "fan_confidence": -1}, "effect_summary": "Confidence +3 · Media -1"},
        {"id": "challenge_media", "label": "Challenge him through the media", "effects": {"player_confidence": -2, "media_pressure": 2}, "effect_summary": "Confidence -2 · Media +2"},
        {"id": "reduce_toi", "label": "Reduce ice time", "effects": {"player_morale": -3, "lineup_pressure": 3, "room_tension": 1}, "effect_summary": "Morale -3 · Lineup pressure +3"},
        {"id": "explore_trade", "label": "Explore trade market", "effects": {"trade_market_heat": 4, "room_tension": 2}, "effect_summary": "Trade heat +4"},
    ]


def _rookie_breakout_actions() -> List[Dict[str, Any]]:
    return [
        {"id": "promote_role", "label": "Promote him", "effects": {"development_confidence": 4, "lineup_pressure": 2}, "effect_summary": "Dev confidence +4"},
        {"id": "steady_plan", "label": "Keep development plan steady", "effects": {"development_confidence": 2, "fan_confidence": -1}, "effect_summary": "Dev confidence +2"},
        {"id": "shelter_minutes", "label": "Shelter minutes", "effects": {"player_confidence": 2, "media_pressure": 1}, "effect_summary": "Stability focus"},
    ]


def _goalie_meltdown_actions() -> List[Dict[str, Any]]:
    return [
        {"id": "start_backup", "label": "Start the backup", "effects": {"goalie_confidence": -2, "fan_confidence": 1}, "effect_summary": "Starter confidence -2 · Fans +1"},
        {"id": "another_chance", "label": "Give starter another chance", "effects": {"goalie_confidence": 2, "media_pressure": 2}, "effect_summary": "Confidence +2 · Media +2"},
        {"id": "call_up_ahl", "label": "Call up AHL goalie", "effects": {"depth_pressure": 1, "room_tension": 1}, "effect_summary": "Depth pressure +1"},
    ]


def run_data_storyline_pass(
    session: Any,
    *,
    calendar_idx: int,
    day_meta: Dict[str, Any],
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Examine franchise ledger data and emit storylines + apply effects.
    Returns debug stats for dev logging.
    """
    r = rng or random.Random()
    iso = str(day_meta.get("iso") or "")
    cur_day = int(calendar_idx)
    season = int(getattr(session, "season_calendar_year", 2025) or 2025)
    uid = str(getattr(session, "user_team_id", "") or "")
    stats = dict(getattr(session, "player_season_stats", None) or {})
    generated: List[Dict[str, Any]] = []
    skipped_cd = 0
    skipped_sample = 0

    # Build OVR / cap / name lookup from rosters (single walk)
    ovr_by_id: Dict[str, float] = {}
    age_by_id: Dict[str, int] = {}
    name_by_id: Dict[str, str] = {}
    cap_hit_by_id: Dict[str, float] = {}
    team_name_by_id: Dict[str, str] = {}
    player_by_id = _player_index_by_id(session)
    rank_by_team = _build_standings_rank_map(session)
    for tid, tm in (getattr(session, "team_by_id", None) or {}).items():
        team_name_by_id[str(tid)] = str(getattr(tm, "name", "") or getattr(tm, "city", "") or tid)
        for bucket in ("roster", "ahl_roster", "injured_reserve", "scratches", "echl_roster"):
            for p in getattr(tm, bucket, None) or []:
                pid = str(getattr(p, "id", "") or "")
                if not pid:
                    continue
                player_by_id.setdefault(pid, p)
                ovr_by_id[pid] = _player_ovr99(p)
                age_by_id[pid] = _player_age(p)
                name_by_id[pid] = str(getattr(p, "name", "") or "Player")
                cap_hit_by_id[pid] = _cap_hit_m(p)

    def emit(raw: Dict[str, Any], fx_team: str, fx_player: str = "") -> None:
        generated.append(raw)
        _apply_storyline_effects(session, fx_team, fx_player, dict(raw.get("effects") or {}))
        if raw.get("requires_action") and raw.get("action_options"):
            _enqueue_decision(session, raw)

    def try_emit(**kwargs: Any) -> None:
        nonlocal skipped_cd, skipped_sample
        stable_key = str(kwargs.pop("stable_key"))
        severity = str(kwargs.get("severity") or "minor")
        ok, rep = _can_fire(session, stable_key, cur_day, severity)
        if not ok:
            skipped_cd += 1
            return
        esc = ""
        if rep > 0:
            esc = "Earlier beat on same issue"
            kwargs["repeat_count"] = rep
            kwargs["escalated_from"] = stable_key
        sl = _build_storyline(rng=r, session=session, stable_key=stable_key, cur_day=cur_day, calendar_iso=iso, **kwargs)
        _mark_fired(session, stable_key, cur_day, severity, rep)
        emit(sl, str(kwargs.get("team_id") or ""), str(kwargs.get("player_id") or ""))

    # --- Skater triggers ---
    for pid, row in stats.items():
        if not isinstance(row, dict):
            continue
        gp = _stat_int(row, "gp", "games_played")
        if gp <= 0:
            continue
        pos = str(row.get("position") or row.get("pos") or "F")
        if _pos_bucket(pos) == "G":
            continue

        tid = str(row.get("team_id") or "")
        pname = str(row.get("name") or name_by_id.get(str(pid), "Player"))
        ovr = float(ovr_by_id.get(str(pid), 0) or row.get("overall") or 0)
        if ovr <= 0:
            player = player_by_id.get(str(pid))
            if player:
                ovr = _player_ovr99(player)
        age = int(age_by_id.get(str(pid), 27))
        is_rookie = age <= 23 and gp <= 40
        g = _stat_int(row, "g", "goals")
        a = _stat_int(row, "a", "assists")
        pts = g + a
        ppg = pts / max(1, gp)
        exp_ppg = _expected_points_per_game(ovr, pos, age, is_rookie)
        exp_pts = round(exp_ppg * gp, 1)
        delta = pts - exp_pts
        tw, tl, to, trec = _team_record(session, tid)
        cap = float(cap_hit_by_id.get(str(pid), 0) or 0)
        role_word = "defenseman" if _pos_bucket(pos) == "D" else "forward"
        ctx = story_ctx(
            name=pname,
            team=team_name_by_id.get(tid, tid),
            role=role_word,
            ovr=ovr,
            gp=gp,
            pts=pts,
            goals=g,
            ppg=ppg,
            exp_pts=exp_pts,
            cap=cap,
            record=trec,
            age=age,
        )

        # Star underperforming
        if ovr >= 84 and gp >= SKATER_GP_MINOR:
            major = gp >= SKATER_GP_MAJOR
            threshold = -2.2 if major else -1.4
            if ppg < exp_ppg * (0.45 if major else 0.62):
                sev = "major" if major and delta <= threshold * 2 else "minor"
                pri = "HIGH" if major else "MEDIUM"
                stype = "star_underperforming"
                hl = pick_line(r, "star_underperforming", ctx)
                body = pick_line(r, "star_underperforming", ctx, body=True)
                try_emit(
                    stable_key=f"{stype}|{pid}|{season}",
                    stype=stype,
                    category="performance",
                    severity=sev,
                    priority=pri,
                    tone="negative",
                    headline=hl,
                    description=body,
                    short_summary=body,
                    cause=f"Production pace {ppg:.2f} P/GP is well below expected {exp_ppg:.2f} for a {round(ovr)}-overall {_pos_bucket(pos)}.",
                    team_id=tid,
                    team_name=team_name_by_id.get(tid, tid),
                    player_id=str(pid),
                    player_name=pname,
                    player_position=pos,
                    player_overall=round(ovr, 1),
                    evidence={"games_played": gp, "goals": g, "assists": a, "points": pts, "expected_points": exp_pts, "overall": round(ovr, 1), "cap_hit": round(cap, 2), "team_record": trec, "points_per_game": round(ppg, 3)},
                    effects={"player_confidence": -6 if major else -3, "media_pressure": 5 if major else 3, "team_morale": -3 if major else -1, "trade_value": -2 if major else -1},
                    requires_action=major and tid == uid,
                    action_options=_underperform_actions() if major and tid == uid else None,
                    heat=78 if major else 55,
                )
            elif gp >= SKATER_GP_MAJOR:
                skipped_sample += 1

        # Rookie / young breakout
        if is_rookie and ovr <= 83 and gp >= SKATER_GP_MINOR and ppg >= max(0.55, exp_ppg * 1.45):
            try_emit(
                stable_key=f"rookie_breakout|{pid}|{season}",
                stype="rookie_breakout",
                category="development",
                severity="minor",
                priority="MEDIUM",
                tone="positive",
                headline=pick_line(r, "rookie_breakout", ctx),
                description=pick_line(r, "rookie_breakout", ctx, body=True),
                short_summary=pick_line(r, "rookie_breakout", ctx, body=True),
                cause=f"Rookie scoring pace {ppg:.2f} P/GP exceeds expected {exp_ppg:.2f}.",
                team_id=tid,
                team_name=team_name_by_id.get(tid, tid),
                player_id=str(pid),
                player_name=pname,
                player_position=pos,
                player_overall=round(ovr, 1),
                evidence={"games_played": gp, "points": pts, "expected_points": exp_pts, "overall": round(ovr, 1), "age": age},
                effects={"fan_confidence": 3, "team_morale": 2, "development_confidence": 4, "lineup_pressure": 2},
                requires_action=tid == uid,
                action_options=_rookie_breakout_actions() if tid == uid else None,
                heat=62,
            )

        # Superstar carrying bad team
        if ovr >= 86 and gp >= SKATER_GP_MAJOR and pts >= 14:
            rank = int(rank_by_team.get(tid, len(rank_by_team) or 16))
            if rank >= 20 and ppg >= 0.85:
                try_emit(
                    stable_key=f"superstar_carry|{pid}|{season}",
                    stype="superstar_carrying",
                    category="team",
                    severity="minor",
                    priority="MEDIUM",
                    tone="mixed",
                    headline=pick_line(r, "superstar_carrying", ctx),
                    description=pick_line(r, "superstar_carrying", ctx, body=True),
                    short_summary=pick_line(r, "superstar_carrying", ctx, body=True),
                    cause="Elite individual scoring on a sub-.500 team.",
                    team_id=tid,
                    team_name=team_name_by_id.get(tid, tid),
                    player_id=str(pid),
                    player_name=pname,
                    player_position=pos,
                    player_overall=round(ovr, 1),
                    evidence={"points": pts, "games_played": gp, "team_record": trec, "league_rank": rank},
                    effects={"player_morale": -1, "fan_confidence": 2, "media_pressure": 3, "trade_market_heat": 2},
                    heat=58,
                )

        # Expensive contract pressure
        if cap >= 6.5 and gp >= SKATER_GP_MAJOR and pts < exp_pts * 0.55:
            try_emit(
                stable_key=f"contract_pressure|{pid}|{season}",
                stype="contract_pressure",
                category="business",
                severity="minor",
                priority="MEDIUM",
                tone="negative",
                headline=pick_line(r, "contract_pressure", ctx),
                description=pick_line(r, "contract_pressure", ctx, body=True),
                short_summary=pick_line(r, "contract_pressure", ctx, body=True),
                cause="High cap hit with well-below-expected production.",
                team_id=tid,
                team_name=team_name_by_id.get(tid, tid),
                player_id=str(pid),
                player_name=pname,
                player_position=pos,
                player_overall=round(ovr, 1),
                evidence={"cap_hit": round(cap, 2), "points": pts, "expected_points": exp_pts, "games_played": gp},
                effects={"fan_confidence": -2, "media_pressure": 4, "trade_value": -2, "owner_patience": -1},
                heat=64,
            )

        # Rolling last-10 hot/cold is published by storyline_coverage, not season PPG.

    # --- Goalie triggers + backup net share (single stats pass) ---
    goalies_by_team: Dict[str, List[Dict[str, Any]]] = {}
    for pid, row in stats.items():
        if not isinstance(row, dict):
            continue
        gp = _stat_int(row, "gp", "games_played")
        if gp < GOALIE_GP_MINOR:
            continue
        pos = str(row.get("position") or "G")
        is_g = _pos_bucket(pos) == "G" or _stat_int(row, "shots_against", "sa") > 0
        if not is_g:
            continue
        tid = str(row.get("team_id") or "")
        pname = str(row.get("name") or name_by_id.get(str(pid), "Goalie"))
        ovr = float(ovr_by_id.get(str(pid), 78))
        sa = _stat_int(row, "shots_against", "sa")
        ga = _stat_int(row, "ga", "goals_against")
        sv_pct = float(row.get("save_pct") or ((sa - ga) / max(1, sa)))
        gaa = float(row.get("gaa") or (ga / max(1, gp)))
        exp_sv = _expected_save_pct(ovr)
        major = gp >= GOALIE_GP_MAJOR
        gctx = story_ctx(
            name=pname,
            team=team_name_by_id.get(tid, tid),
            role="goaltender",
            ovr=ovr,
            gp=gp,
            sv=sv_pct,
            gaa=gaa,
            exp_sv=exp_sv,
        )

        if sv_pct < exp_sv - 0.018 and major:
            try_emit(
                stable_key=f"goalie_meltdown|{pid}|{season}",
                stype="goalie_meltdown",
                category="performance",
                severity="minor",
                priority="HIGH" if sv_pct < exp_sv - 0.028 else "MEDIUM",
                tone="negative",
                headline=pick_line(r, "goalie_meltdown", gctx),
                description=pick_line(r, "goalie_meltdown", gctx, body=True),
                short_summary=pick_line(r, "goalie_meltdown", gctx, body=True),
                cause=f"Save percentage {sv_pct:.3f} well below {exp_sv:.3f} expected for {round(ovr)} OVR.",
                team_id=tid,
                team_name=team_name_by_id.get(tid, tid),
                player_id=str(pid),
                player_name=pname,
                player_position="G",
                player_overall=round(ovr, 1),
                evidence={"games_played": gp, "save_pct": round(sv_pct, 3), "gaa": round(gaa, 2), "expected_save_pct": exp_sv},
                effects={"goalie_confidence": -8 if major else -4, "team_morale": -2, "media_pressure": 3},
                requires_action=tid == uid,
                action_options=_goalie_meltdown_actions() if tid == uid else None,
                heat=72,
            )
        elif sv_pct >= exp_sv + 0.012 and gp >= GOALIE_GP_MINOR:
            try_emit(
                stable_key=f"goalie_heater|{pid}|{season}",
                stype="goalie_heater",
                category="performance",
                severity="minor",
                priority="MEDIUM",
                tone="positive",
                headline=pick_line(r, "goalie_heater", gctx),
                description=pick_line(r, "goalie_heater", gctx, body=True),
                short_summary=pick_line(r, "goalie_heater", gctx, body=True),
                cause="Save percentage materially above expected baseline.",
                team_id=tid,
                team_name=team_name_by_id.get(tid, tid),
                player_id=str(pid),
                player_name=pname,
                player_position="G",
                player_overall=round(ovr, 1),
                evidence={"save_pct": round(sv_pct, 3), "games_played": gp, "gaa": round(gaa, 2)},
                effects={"goalie_confidence": 5, "team_morale": 3, "fan_confidence": 4},
                heat=60,
            )

        goalies_by_team.setdefault(tid, []).append(
            {
                "pid": str(pid),
                "gp": gp,
                "sv": sv_pct,
                "name": pname,
                "ovr": ovr,
            }
        )

    for tid, group in goalies_by_team.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda x: -x["gp"])
        starter, backup = group[0], group[1]
        if backup["gp"] >= GOALIE_GP_MINOR and backup["sv"] >= starter["sv"] + 0.015 and starter["gp"] >= GOALIE_GP_MAJOR:
            bctx = story_ctx(
                name=backup["name"],
                team=team_name_by_id.get(tid, tid),
                role="goaltender",
                ovr=backup["ovr"],
                gp=backup["gp"],
                sv=backup["sv"],
            )
            try_emit(
                stable_key=f"backup_net|{backup['pid']}|{season}",
                stype="backup_taking_net",
                category="performance",
                severity="minor",
                priority="MEDIUM",
                tone="mixed",
                headline=pick_line(r, "backup_taking_net", bctx),
                description=pick_line(r, "backup_taking_net", bctx, body=True),
                short_summary=pick_line(r, "backup_taking_net", bctx, body=True),
                cause="Backup save percentage materially ahead of the starter's workload.",
                team_id=tid,
                team_name=team_name_by_id.get(tid, tid),
                player_id=str(backup["pid"]),
                player_name=backup["name"],
                player_position="G",
                player_overall=round(backup["ovr"], 1),
                evidence={
                    "backup_sv": round(backup["sv"], 3),
                    "starter_sv": round(starter["sv"], 3),
                    "backup_gp": backup["gp"],
                    "starter_gp": starter["gp"],
                },
                effects={"goalie_confidence": 3, "lineup_pressure": 2, "media_pressure": 2},
                heat=57,
            )

    # --- Team triggers ---
    for tid, tm in (getattr(session, "team_by_id", None) or {}).items():
        tid = str(tid)
        gp = _team_games_played(session, tid)
        if gp < TEAM_GP_MIN:
            continue
        w, l, o, trec = _team_record(session, tid)
        win_pct = w / max(1, gp)
        rank = int(rank_by_team.get(tid, len(rank_by_team) or 16))
        strength = float((getattr(session, "strength_map", None) or {}).get(tid, 0.5) or 0.5)
        tname = team_name_by_id.get(tid, tid)
        tctx = story_ctx(name=tname, team=tname, record=trec, gp=gp)

        # Surprise team
        if strength < 0.46 and win_pct >= 0.58 and gp >= TEAM_GP_MIN:
            try_emit(
                stable_key=f"surprise_team|{tid}|{season}",
                stype="surprise_team",
                category="league",
                severity="minor",
                priority="MEDIUM",
                tone="positive",
                headline=pick_line(r, "surprise_team", tctx),
                description=f"Expected also-ran is {trec} with a {win_pct:.3f} win rate.",
                short_summary=pick_line(r, "surprise_team", tctx),
                cause="Low pre-season strength rating with strong actual record.",
                team_id=tid,
                team_name=tname,
                evidence={"team_record": trec, "win_pct": round(win_pct, 3), "strength_rating": round(strength, 3)},
                effects={"fan_confidence": 4, "media_pressure": 2},
                heat=56,
            )

        # Contender collapse
        if strength >= 0.58 and win_pct <= 0.42 and gp >= TEAM_GP_MIN:
            try_emit(
                stable_key=f"contender_collapse|{tid}|{season}",
                stype="contender_collapse",
                category="league",
                severity="minor",
                priority="HIGH",
                tone="negative",
                headline=pick_line(r, "contender_collapse", tctx),
                description=f"Strong on paper club is {trec} through {gp} games.",
                short_summary=pick_line(r, "contender_collapse", tctx),
                cause="High team strength rating with losing record.",
                team_id=tid,
                team_name=tname,
                evidence={"team_record": trec, "win_pct": round(win_pct, 3), "strength_rating": round(strength, 3)},
                effects={"media_pressure": 5, "coach_security": -3, "trade_market_heat": 4, "team_morale": -3},
                heat=80,
            )

        # Playoff bubble (user team late season)
        ui_phase = str(day_meta.get("ui_phase") or "")
        if tid == uid and gp >= 55 and rank in (7, 8, 9, 10, 11):
            try_emit(
                stable_key=f"playoff_race|{tid}|{season}",
                stype="playoff_race",
                category="team",
                severity="minor",
                priority="HIGH" if tid == uid else "MEDIUM",
                tone="mixed",
                headline=pick_line(r, "playoff_race", tctx),
                description=f"{tname} sits around the wild-card line at {trec} (rank ~{rank}).",
                short_summary=f"Bubble team · rank {rank}",
                cause="Late-season standings place team near playoff cut line.",
                team_id=tid,
                team_name=tname,
                evidence={"team_record": trec, "league_rank": rank, "phase": ui_phase},
                effects={"media_pressure": 3, "team_morale": 1 if win_pct >= 0.5 else -2},
                heat=70 if tid == uid else 50,
            )

        # Losing streak proxy
        if l >= 4 and w <= 2 and gp >= TEAM_GP_MIN:
            try_emit(
                stable_key=f"losing_skid|{tid}|{season}",
                stype="cold_streak_team",
                category="team",
                severity="minor",
                priority="MEDIUM",
                tone="negative",
                headline=pick_line(r, "losing_skid", tctx),
                description=f"Team record {trec} with mounting losses.",
                short_summary=f"Skid · {trec}",
                cause="Multiple losses in current standings snapshot.",
                team_id=tid,
                team_name=tname,
                evidence={"team_record": trec, "losses": l},
                effects={"team_morale": -2, "media_pressure": 3, "room_tension": 2},
                heat=52,
            )

        # Win streak proxy
        if w >= 6 and gp >= TEAM_GP_MIN:
            try_emit(
                stable_key=f"win_streak|{tid}|{season}",
                stype="hot_streak_team",
                category="team",
                severity="minor",
                priority="LOW",
                tone="positive",
                headline=pick_line(r, "win_streak", tctx),
                description=f"Strong run reflected in {trec} record.",
                short_summary=f"Heating up · {trec}",
                cause="Standings show sustained winning.",
                team_id=tid,
                team_name=tname,
                evidence={"team_record": trec, "wins": w},
                effects={"team_morale": 2, "fan_confidence": 3},
                heat=46,
            )

    # --- Draft prospect stock ---
    ranks = dict(getattr(session, "draft_rank_prev", None) or {})
    prev = dict(getattr(session, "draft_preseason_rank", None) or ranks)
    if ranks and prev:
        for key, rank in list(ranks.items())[:80]:
            old = int(prev.get(key, rank))
            cur = int(rank)
            delta = old - cur  # positive = rose (lower rank number is better)
            if abs(delta) < 4:
                continue
            pname = str(key).split("|")[-1][:40] or "Prospect"
            if delta >= 8:
                try_emit(
                    stable_key=f"prospect_riser|{key}|{season}",
                    stype="prospect_rising",
                    category="draft",
                    severity="minor",
                    priority="MEDIUM",
                    tone="positive",
                    headline="Anonymous draft hopeful has rudely entered the first-round conversation",
                    description=f"{pname} climbed from rank ~{old} to ~{cur} on internal boards.",
                    short_summary=f"Stock +{delta} spots",
                    cause="Draft rank improved materially vs preseason baseline.",
                    team_id=uid,
                    team_name=team_name_by_id.get(uid, "League"),
                    evidence={"previous_rank": old, "current_rank": cur, "delta": delta},
                    effects={"draft_stock": min(12, delta), "scout_attention": 2},
                    heat=50,
                )
            elif delta <= -8:
                try_emit(
                    stable_key=f"prospect_faller|{key}|{season}",
                    stype="prospect_falling",
                    category="draft",
                    severity="minor",
                    priority="MEDIUM",
                    tone="negative",
                    headline="Top prospect's draft stock now sliding like a Zamboni with bad brakes",
                    description=f"{pname} dropped from ~{old} to ~{cur} on the board.",
                    short_summary=f"Stock -{abs(delta)} spots",
                    cause="Draft rank fell materially vs preseason baseline.",
                    team_id=uid,
                    team_name=team_name_by_id.get(uid, "League"),
                    evidence={"previous_rank": old, "current_rank": cur, "delta": delta},
                    effects={"draft_stock": -min(10, abs(delta)), "scouting_uncertainty": 2},
                    heat=48,
                )

    # --- Injury ripple (recent major injuries) ---
    for inj in list(getattr(session, "injury_log_major", None) or [])[-12:]:
        if not isinstance(inj, dict):
            continue
        try:
            inj_day = int(inj.get("calendar_day", -999))
        except (TypeError, ValueError):
            inj_day = -999
        if cur_day - inj_day > 2:
            continue
        tid = str(inj.get("team_id") or "")
        pid = str(inj.get("player_id") or "")
        pname = str(inj.get("player_name") or inj.get("player") or "Player")
        games = int(inj.get("games") or inj.get("games_remaining") or 0)
        try_emit(
            stable_key=f"injury_ripple|{pid}|{season}",
            stype="injury_ripple",
            category="injury",
            severity="major" if games >= 4 else "minor",
            priority="HIGH" if tid == uid else "MEDIUM",
            tone="negative",
            headline=f"{pname} injury forces {team_name_by_id.get(tid, tid)} to reshuffle the deck",
            description=f"{pname} expected out {games} games — depth chart under stress.",
            short_summary=f"Injury · {games}g out",
            cause="Major injury logged in franchise injury engine.",
            team_id=tid,
            team_name=team_name_by_id.get(tid, tid),
            player_id=pid,
            player_name=pname,
            evidence={"games_out": games, "injury_type": str(inj.get("tier") or inj.get("injury_type") or "")},
            effects={"team_morale": -2, "depth_pressure": 4, "lineup_pressure": 3},
            heat=75 if tid == uid else 55,
        )

    # Cap storylines per day (league desks + rolling form live elsewhere)
    cap = 16
    generated.sort(key=lambda s: (-int(s.get("heat") or 0), str(s.get("priority") or "")))
    trimmed = generated[:cap]

    out = {
        "generated": len(trimmed),
        "skipped_cooldown": skipped_cd,
        "skipped_sample": skipped_sample,
        "storylines": trimmed,
    }
    if _DEV:
        _log.info(
            "[storyline_engine] day=%s generated=%s skipped_cd=%s",
            cur_day,
            len(trimmed),
            skipped_cd,
        )
    return out


def _enqueue_decision(session: Any, storyline: Dict[str, Any]) -> None:
    """Push GM decision linked to storyline."""
    sid = str(storyline.get("storyline_id") or storyline.get("id") or "")
    if not sid:
        return
    pending = list(getattr(session, "pending_decisions", None) or [])
    for d in pending:
        meta = dict(d.get("meta") or {})
        if str(meta.get("storyline_id") or "") == sid:
            return
    opts = list(storyline.get("action_options") or [])
    if not opts:
        return
    pending.append(
        {
            "id": f"dec_{sid}",
            "storyline_id": sid,
            "kind": "data_storyline_decision",
            "priority": str(storyline.get("priority") or "MEDIUM"),
            "title": str(storyline.get("headline") or "Storyline decision"),
            "description": str(storyline.get("description") or storyline.get("short_summary") or ""),
            "options": opts,
            "meta": {
                "storyline_id": sid,
                "team_id": str(storyline.get("team_id") or ""),
                "player_id": str(storyline.get("player_id") or ""),
                "player_name": str(storyline.get("player_name") or ""),
                "cause": str(storyline.get("cause") or ""),
            },
        }
    )
    session.pending_decisions = pending


def franchise_record_data_storylines(
    session: Any,
    calendar_idx: int,
    day_meta: Dict[str, Any],
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Run engine and append storylines to session via _record_storyline callback."""
    from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433

    result = run_data_storyline_pass(session, calendar_idx=calendar_idx, day_meta=day_meta, rng=rng)
    utid = str(getattr(session, "user_team_id", "") or "")
    for raw in result.get("storylines") or []:
        if not isinstance(raw, dict) or not raw.get("headline"):
            continue
        tid = str(raw.get("team_id") or raw.get("team") or "")
        if tid == utid and str(raw.get("cause_type") or ""):
            _emit_cause_storyline(session, raw)
        else:
            _record_storyline(session, raw)
    return result


# ---------------------------------------------------------------------------
# Cause-and-effect storyline system (franchise mode)
# ---------------------------------------------------------------------------

STORYLINE_CAUSE_TYPES = frozenset(
    {
        "TRADE_ATTEMPTED_BY_USER",
        "TRADE_REJECTED",
        "PLAYER_REPEATEDLY_SHOPPED",
        "PLAYER_TRADED",
        "CULPRIT_TRADED",
        "PLAYER_SCRATCHED_BY_USER",
        "PLAYER_DEMOTED",
        "PLAYER_PROMOTED",
        "PLAYER_ROLE_REDUCED",
        "PLAYER_LOW_PRODUCTION",
        "PLAYER_REALDATA_DROP",
        "GOALIE_BAD_FORM",
        "CONTRACT_DISPUTE",
        "EXTENSION_REJECTED",
        "LOW_CHARACTER_CONFLICT",
        "LOSING_STREAK",
        "WINNING_STREAK",
        "CAPTAIN_TRADED",
        "CAPTAINCY_CHANGED",
        "TEAM_CHEMISTRY_LOW",
        "COACH_TRUST_LOW",
        "PROSPECT_BLOCKED",
        "STAR_UNHAPPY",
        "MEDIA_PRESSURE_HIGH",
        "CONDUCT_RESOLVED",
        "PLAYER_INJURED",
        "TRADE_DEMAND",
        "PLAYER_TRADE_DEMAND",
    }
)

_FAKE_STORYLINE_TEXT_MARKERS = (
    "scratched",
    "healthy scratch",
    "trade rumor",
    "trade rumours",
    "shopped",
    "conduct issue",
    "conduct violation",
    "morale collapsed",
    "overall dropped",
    "away from team",
    "league opens investigation",
    "domestic violence",
    "dui",
    "betting on games",
)

_SOURCE_LABEL_BY_CAUSE: Dict[str, str] = {
    "TRADE_REJECTED": "TradeHub Fallout",
    "TRADE_ATTEMPTED_BY_USER": "TradeHub Fallout",
    "PLAYER_REPEATEDLY_SHOPPED": "TradeHub Fallout",
    "PLAYER_LOW_PRODUCTION": "Team Report",
    "PLAYER_REALDATA_DROP": "Team Report",
    "GOALIE_BAD_FORM": "Net Report",
    "LOW_CHARACTER_CONFLICT": "Locker Room Pulse",
    "LOSING_STREAK": "Team Report",
    "WINNING_STREAK": "Team Report",
    "CULPRIT_TRADED": "Locker Room Pulse",
    "PLAYER_TRADED": "League Insider",
    "PLAYER_SCRATCHED_BY_USER": "Team Report",
    "PLAYER_DEMOTED": "Team Report",
    "PLAYER_ROLE_REDUCED": "Team Report",
    "PLAYER_PROMOTED": "Team Report",
    "CONTRACT_DISPUTE": "Team Report",
    "CAPTAINCY_CHANGED": "Locker Room Pulse",
}

_LINEUP_FORWARD_RANK = {"f1": 1, "f2": 2, "f3": 3, "f4": 4}
_LINEUP_DEFENSE_RANK = {"d1": 1, "d2": 2, "d3": 3}
_LINEUP_GOALIE_SLOT_RANK = {"Starter": 1, "Backup": 2, "Third": 3}

_LOCKER_ROOM_TRIGGER_EVENTS = frozenset(
    {
        "PLAYER_SCRATCHED_BY_USER",
        "PLAYER_DEMOTED",
        "PLAYER_ROLE_REDUCED",
        "TRADE_REJECTED",
        "PLAYER_REPEATEDLY_SHOPPED",
        "LOSING_STREAK",
        "CONTRACT_DISPUTE",
        "CAPTAINCY_CHANGED",
    }
)

_CPU_EVENT_CAUSE_MAP: Dict[str, str] = {
    "locker_room_issue": "LOW_CHARACTER_CONFLICT",
    "team_conflict": "LOW_CHARACTER_CONFLICT",
    "legal_trouble": "LOW_CHARACTER_CONFLICT",
    "scandal": "LOW_CHARACTER_CONFLICT",
    "breakout": "PLAYER_LOW_PRODUCTION",
    "emergence": "PLAYER_LOW_PRODUCTION",
    "clutch_run": "WINNING_STREAK",
    "leader_emergence": "CAPTAINCY_CHANGED",
    "goalie_meltdown": "GOALIE_BAD_FORM",
}


def migrate_session_storyline_state(session: Any) -> None:
    """Default new fields for old saves."""
    if getattr(session, "decision_event_log", None) is None:
        session.decision_event_log = []
    if getattr(session, "active_cause_storylines", None) is None:
        session.active_cause_storylines = []
    if getattr(session, "story_arcs", None) is None:
        session.story_arcs = []
    if getattr(session, "social_posts", None) is None:
        session.social_posts = []
    if getattr(session, "player_narrative_memory", None) is None:
        session.player_narrative_memory = {}
    if getattr(session, "knowledge_graph", None) is None:
        session.knowledge_graph = []
    if getattr(session, "press_conference_queue", None) is None:
        session.press_conference_queue = []
    if getattr(session, "reddit_threads", None) is None:
        session.reddit_threads = []
    if getattr(session, "reddit_engagement_pulse", None) is None:
        session.reddit_engagement_pulse = []
    if getattr(session, "gm_burner_account", None) is None:
        session.gm_burner_account = {
            "handle": "",
            "created_day": 0,
            "posts": [],
            "suspicion_score": 0.0,
            "exposed": False,
        }
    if getattr(session, "gm_burner_investigation", None) is None:
        session.gm_burner_investigation = {}
    if getattr(session, "narrative_archive", None) is None:
        session.narrative_archive = []
    if getattr(session, "narrative_eras", None) is None:
        session.narrative_eras = []
    if getattr(session, "prospect_social_profiles", None) is None:
        session.prospect_social_profiles = {}
    if getattr(session, "agent_relationships", None) is None:
        session.agent_relationships = {}
    if getattr(session, "_narrative_sealed_seasons", None) is None:
        session._narrative_sealed_seasons = []
    if getattr(session, "_storyline_blocked_log", None) is None:
        session._storyline_blocked_log = []
    pending = list(getattr(session, "pending_decisions", None) or [])
    if pending:
        session.pending_decisions = [
            d
            for d in pending
            if isinstance(d, dict) and str(d.get("id") or "").strip()
        ]
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is not None:
        setattr(league, "_franchise_user_team_id", str(getattr(session, "user_team_id", "") or ""))
    stats_rev = int(getattr(session, "_stats_revision", 0) or 0)
    if int(getattr(session, "_storyline_migrate_stats_rev", -1) or -1) != stats_rev:
        for tm in (getattr(session, "team_by_id", None) or {}).values():
            for pl in getattr(tm, "roster", None) or []:
                _ensure_player_storyline_state(pl)
        session._storyline_migrate_stats_rev = stats_rev


def _ensure_player_storyline_state(player: Any) -> Dict[str, Any]:
    st = getattr(player, "_franchise_storyline_state", None)
    if not isinstance(st, dict):
        st = {
            "trade_attempt_count": 0,
            "was_recently_shopped": False,
            "trade_rumor_heat": 0,
            "last_trade_rumor_day": -1,
            "last_trade_rumor_week": -1,
            "last_trade_rumor_season": -1,
            "last_trade_rumor_context": "",
            "last_trade_rumor_decay_week": -1,
            "gm_trust": 0.72,
            "morale_note": 0.0,
        }
        setattr(player, "_franchise_storyline_state", st)
    for k, default in (
        ("trade_attempt_count", 0),
        ("was_recently_shopped", False),
        ("trade_rumor_heat", 0),
        ("last_trade_rumor_day", -1),
        ("last_trade_rumor_week", -1),
        ("last_trade_rumor_season", -1),
        ("last_trade_rumor_context", ""),
        ("last_trade_rumor_decay_week", -1),
        ("gm_trust", 0.72),
    ):
        if k not in st:
            st[k] = default
    return st


def _ordinal(n: int) -> str:
    v = int(n or 0)
    mod100 = v % 100
    if 11 <= mod100 <= 13:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(v % 10, "th")
    return f"{v}{suf}"


def _trade_rumor_context_key(partner_id: str, verdict: str) -> str:
    return f"{str(partner_id or '')}:{str(verdict or '')}"


def _is_trade_rumor_cooldown_active(pst: Dict[str, Any], *, cur_day: int, season: int) -> bool:
    last_season = int(pst.get("last_trade_rumor_season") or -1)
    if last_season != season:
        return False
    if int(pst.get("last_trade_rumor_day") or -1) == cur_day:
        return True
    week = cur_day // 7
    return int(pst.get("last_trade_rumor_week") or -1) == week


def _classify_trade_rumor_verdict(evaluation: Dict[str, Any]) -> str:
    verdict = str(evaluation.get("verdict") or "").lower()
    reasons = " ".join(str(r or "").lower() for r in (evaluation.get("rejection_reasons") or []))
    if verdict in ("ntc_nmc_conflict", "player_unavailable", "asset_not_owned"):
        return "technical_no_fallout"
    if verdict in ("cap_illegal", "roster_illegal", "blocked"):
        if "slot" in reasons or "contract slot" in reasons:
            return "soft_blocked"
        return "soft_blocked"
    if verdict in ("rejected", "trade_value_too_low"):
        return "rejected"
    if "cap" in reasons or "roster" in reasons or "slot" in reasons or "clause" in reasons:
        return "soft_blocked"
    if "not found" in reasons or "does not own" in reasons or "unavailable" in reasons or "registry" in reasons:
        return "technical_no_fallout"
    return "rejected"


def _player_character_0_100(player: Any) -> int:
    c = getattr(player, "character", None)
    if c is not None:
        try:
            ci = int(c)
            if 20 <= ci <= 90:
                return ci
        except (TypeError, ValueError):
            pass
    tr = getattr(player, "traits", None)
    if tr is None:
        return 50
    blend = (
        0.22 * float(getattr(tr, "coachability", 0.5))
        + 0.20 * float(getattr(tr, "mental_toughness", 0.5))
        + 0.18 * float(getattr(tr, "work_ethic", 0.5))
        + 0.16 * float(getattr(tr, "leadership", 0.5))
        + 0.14 * float(getattr(tr, "competitiveness", 0.5))
        + 0.10 * (1.0 - float(getattr(tr, "volatility", 0.5)))
    )
    return int(round(_clamp(blend, 0.0, 1.0) * 100.0))


def record_decision_event(session: Any, event: Dict[str, Any]) -> str:
    migrate_session_storyline_state(session)
    eid = str(event.get("event_id") or event.get("id") or f"evt_{uuid.uuid4().hex[:12]}")
    row = dict(event)
    row["event_id"] = eid
    row.setdefault("date", str(getattr(session, "calendar_cursor", 0) or 0))
    log = list(getattr(session, "decision_event_log", None) or [])
    log.append(row)
    session.decision_event_log = log[-400:]
    return eid


def find_decision_event(session: Any, event_id: str) -> Optional[Dict[str, Any]]:
    eid = str(event_id or "")
    if not eid:
        return None
    for ev in reversed(list(getattr(session, "decision_event_log", None) or [])):
        if str(ev.get("event_id") or ev.get("id") or "") == eid:
            return ev
    return None


def _team_display(session: Any, tid: str) -> str:
    tm = (getattr(session, "team_by_id", None) or {}).get(str(tid))
    if tm is None:
        return str(tid)
    city = str(getattr(tm, "city", "") or "").strip()
    name = str(getattr(tm, "name", "") or "").strip()
    return f"{city} {name}".strip() or str(tid)


def _outgoing_user_players(session: Any, assets_by_team: Dict[str, Any]) -> List[Tuple[str, str, Any]]:
    utid = str(getattr(session, "user_team_id", "") or "")
    out: List[Tuple[str, str, Any]] = []
    for acq_tid, assets in (assets_by_team or {}).items():
        for raw in assets or []:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("type") or "").lower() != "player":
                continue
            src = str(raw.get("team") or "")
            if src != utid:
                continue
            pid = str(raw.get("id") or "")
            pl = _player_from_roster(session, pid)
            if pl is not None:
                pname = str(getattr(pl, "name", "") or "Player")
                out.append((pid, pname, pl))
    return out


def record_trade_hub_evaluation(
    session: Any,
    evaluation: Dict[str, Any],
    assets_by_team: Dict[str, List[Dict[str, Any]]],
    *,
    proposal_submitted: bool = False,
) -> List[Dict[str, Any]]:
    """Log TradeHub proposal; emit fallout storylines when user shops own players."""
    # Trade rumor fallout only applies to real rejected proposals.
    # Evaluation previews must never damage player OVR.
    if not proposal_submitted:
        return []
    migrate_session_storyline_state(session)
    utid = str(getattr(session, "user_team_id", "") or "")
    if not utid:
        return []
    accepted = bool(evaluation.get("accepted"))
    if accepted:
        return []
    rejection_kind = _classify_trade_rumor_verdict(evaluation)
    partner_id = ""
    for tid in (evaluation.get("participating_teams") or evaluation.get("team_ids") or []):
        if str(tid) != utid:
            partner_id = str(tid)
            break
    if not partner_id:
        for tid in (assets_by_team or {}).keys():
            if str(tid) != utid:
                partner_id = str(tid)
                break

    generated: List[Dict[str, Any]] = []
    iso = ""
    cal = getattr(session, "nhl_calendar", None) or []
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    if 0 <= cur < len(cal):
        iso = str(cal[cur].get("iso") or "")

    for pid, pname, pl in _outgoing_user_players(session, assets_by_team):
        pst = _ensure_player_storyline_state(pl)
        season = int(getattr(session, "season_calendar_year", 2025) or 2025)
        cooldown_active = _is_trade_rumor_cooldown_active(pst, cur_day=cur, season=season)
        context_key = _trade_rumor_context_key(partner_id, str(evaluation.get("verdict") or ""))
        pst["last_trade_rumor_context"] = context_key
        pst["was_recently_shopped"] = True

        if rejection_kind == "technical_no_fallout":
            continue

        if rejection_kind == "soft_blocked":
            # Cap / roster / slots blocks should not create full fallout.
            if not cooldown_active:
                pst["trade_rumor_heat"] = min(100, int(pst.get("trade_rumor_heat") or 0) + 1)
                pst["last_trade_rumor_day"] = cur
                pst["last_trade_rumor_week"] = cur // 7
                pst["last_trade_rumor_season"] = season
            continue

        if cooldown_active:
            continue

        pst["trade_attempt_count"] = int(pst.get("trade_attempt_count") or 0) + 1
        pst["trade_rumor_heat"] = min(100, int(pst.get("trade_rumor_heat") or 0) + 12)
        pst["last_trade_rumor_day"] = cur
        pst["last_trade_rumor_week"] = cur // 7
        pst["last_trade_rumor_season"] = season
        attempt_n = int(pst["trade_attempt_count"])

        evt_type = "TRADE_REJECTED"
        if attempt_n >= 2:
            evt_type = "PLAYER_REPEATEDLY_SHOPPED"

        eid = record_decision_event(
            session,
            {
                "event_type": evt_type,
                "date": iso or cur,
                "calendar_day": cur,
                "team_id": utid,
                "player_ids": [pid],
                "player_id": pid,
                "player_name": pname,
                "target_team_id": partner_id,
                "trade_id": f"trade_{cur}_{pid[:8]}",
                "severity": "medium" if attempt_n < 3 else "high",
                "accepted": accepted,
                "attempt_count": attempt_n,
            },
        )

        sl = _build_trade_rejected_storyline(
            session,
            player=pl,
            player_id=pid,
            player_name=pname,
            team_id=utid,
            partner_team_id=partner_id,
            attempt_count=attempt_n,
            cause_event_id=eid,
            calendar_iso=iso,
            cur_day=cur,
        )
        if sl:
            generated.append(sl)
            _emit_cause_storyline(session, sl)
        try:
            from app.sim_engine.franchise.trade_stability_engine import apply_trade_hub_exposure  # noqa: WPS433

            apply_trade_hub_exposure(
                session,
                pl,
                attempt_n=attempt_n,
                rejection_kind=rejection_kind,
            )
        except Exception:
            pass
    return generated


def _trade_fallout_magnitude(player: Any, attempt_count: int) -> Tuple[int, int, int, str]:
    char = _player_character_0_100(player)
    pst = _ensure_player_storyline_state(player)
    heat = int(pst.get("trade_rumor_heat") or 0)
    gm_trust_drop = 4 + min(10, attempt_count * 2)
    morale_drop = 3 + min(9, attempt_count * 2)
    base_penalty = 1
    if attempt_count >= 2:
        base_penalty = 2
    if attempt_count >= 4:
        base_penalty = 3
    if heat >= 70:
        base_penalty += 1
    if char < 35:
        base_penalty += 1
        gm_trust_drop += 2
        morale_drop += 2
    if char >= 70:
        gm_trust_drop = max(2, gm_trust_drop - 2)
        morale_drop = max(2, morale_drop - 1)
    major_drama = attempt_count >= 6 or heat >= 90
    max_penalty = 5 if major_drama else 4
    ovr_mod = max(1, min(max_penalty, base_penalty))
    severity = "minor" if attempt_count <= 1 else "major" if major_drama or attempt_count >= 5 else "mid"
    pst["gm_trust"] = _clamp(float(pst.get("gm_trust", 0.72)) - gm_trust_drop * 0.01)
    return morale_drop, gm_trust_drop, ovr_mod, severity


def _build_trade_rejected_storyline(
    session: Any,
    *,
    player: Any,
    player_id: str,
    player_name: str,
    team_id: str,
    partner_team_id: str,
    attempt_count: int,
    cause_event_id: str,
    calendar_iso: str,
    cur_day: int,
) -> Optional[Dict[str, Any]]:
    tname = _team_display(session, team_id)
    partner = _team_display(session, partner_team_id)
    morale_drop, gm_trust_drop, ovr_mod, severity = _trade_fallout_magnitude(player, attempt_count)

    if attempt_count >= 4:
        headline = f"{player_name} camp reportedly pushing for clarity after repeated trade talks"
        body = (
            f"After reports surfaced that {tname} attempted to move {player_name} in trade talks "
            f"for the {attempt_count}{'th' if attempt_count != 3 else 'rd'} time, "
            f"{player_name}'s representatives are said to be frustrated. "
            f"The latest deal with {partner} fell apart, but teammates are beginning to notice."
        )
        cause_type = "PLAYER_REPEATEDLY_SHOPPED"
    elif attempt_count >= 2:
        headline = f"Agent contacts {tname} after another failed {player_name} trade proposal"
        body = (
            f"{tname}'s attempt to trade {player_name} to {partner} was rejected. "
            f"This is the {_ordinal(attempt_count)} time management has shopped him this season."
        )
        cause_type = "PLAYER_REPEATEDLY_SHOPPED"
    else:
        headline = f"Trade talks involving {player_name} create internal ripple"
        body = (
            f"After reports surfaced that {tname} attempted to move {player_name} in trade talks, "
            f"{player_name}'s camp is said to be frustrated. The deal with {partner} fell apart, "
            f"but the damage inside the room may already be done."
        )
        cause_type = "TRADE_REJECTED"

    stable_key = f"{cause_type}|{player_id}|{cause_event_id}"
    effects = {
        "player_morale": -float(morale_drop),
        "gm_trust": -float(gm_trust_drop),
        "room_tension": float(3 + min(6, attempt_count)),
        "trade_market_heat": float(2 + attempt_count),
    }
    if attempt_count >= 3:
        effects["media_pressure"] = 5.0

    recovery = [
        "Win games together",
        "Restore top-line role",
        "Publicly commit to player",
        "Trade player if relationship is broken",
        "Let time pass",
    ]
    if _player_character_0_100(player) >= 65:
        recovery.append("Team leadership may stabilize the room")

    return {
        "id": _storyline_id(stable_key),
        "storyline_id": _storyline_id(stable_key),
        "stable_key": stable_key,
        "type": "trade_fallout",
        "category": "trade",
        "cause_type": cause_type,
        "cause_event_id": cause_event_id,
        "culprit_player_id": player_id,
        "player_id": player_id,
        "player_name": player_name,
        "affected_player_ids": [player_id],
        "team_id": team_id,
        "team_name": tname,
        "severity": severity,
        "priority": "HIGH" if attempt_count >= 3 else "MEDIUM",
        "tone": "negative",
        "headline": headline,
        "title": headline,
        "description": body,
        "short_summary": body[:160],
        "summary": body,
        "cause": f"User proposed a TradeHub deal involving {player_name}; {partner} rejected the package.",
        "user_visible_explanation": body,
        "calendar_iso": calendar_iso,
        "calendar_day": cur_day,
        "date": calendar_iso or cur_day,
        "source": _SOURCE_LABEL_BY_CAUSE.get(cause_type, "TradeHub Fallout"),
        "source_label": _SOURCE_LABEL_BY_CAUSE.get(cause_type, "TradeHub Fallout"),
        "effects": effects,
        "ovr_modifier": -int(ovr_mod),
        "ovr_modifier_games": 8 + attempt_count * 2,
        "recovery_conditions": recovery,
        "resolution_condition": "culprit_traded_or_attempts_reset",
        "status": "active",
        "resolved": False,
        "requires_action": attempt_count >= 3,
        "action_options": (
            [
                {"id": "meet_player", "label": "Meet with player privately", "effects": {"player_morale": 4, "gm_trust": 3}},
                {"id": "open_tradehub", "label": "Revisit trade market", "effects": {"trade_market_heat": 2}},
                {"id": "restore_role", "label": "Restore prominent role", "effects": {"player_morale": 3, "room_tension": -2}},
            ]
            if attempt_count >= 3
            else []
        ),
    }


def should_block_random_storyline_for_user(
    row: Dict[str, Any],
    session: Any,
    *,
    user_team_id: str,
) -> bool:
    """Return True if a random engine storyline must not apply to the user team.

    Equal rules for legal/conduct: real legal_crime / legal_trouble rows are allowed
    so GM decisions and the conduct state machine can fire for the user team.
    Still block fake text markers and uncaused trade-rumour spam.
    """
    tid = str(row.get("team_id") or "")
    if not user_team_id or tid != user_team_id:
        return False
    if str(row.get("cause_type") or row.get("cause_event_id") or ""):
        return False
    pool = str(row.get("pool") or "").lower()
    et = str(row.get("event_type") or "").lower()
    # Allow legal / conduct generation for the user team (equal rules).
    if pool == "legal_crime" or et in ("legal_trouble",):
        return False
    text = str(row.get("storyline_text") or row.get("storyline") or "").lower()
    # Caused trade demands / registered arcs are allowed through.
    if str(row.get("cause_type") or "") in (
        "TRADE_DEMAND",
        "PLAYER_TRADE_DEMAND",
        "LOW_CHARACTER_CONFLICT",
        "LOSING_STREAK",
        "PLAYER_LOW_PRODUCTION",
    ):
        return False
    # Still block uncaused locker/scandal spam and fake markers.
    if et in ("scandal", "locker_room_issue", "team_conflict"):
        return True
    for marker in _FAKE_STORYLINE_TEXT_MARKERS:
        if marker in text:
            return True
    # Block only uncaused rumor-style text; real TRADE_DEMAND rows carry cause_type.
    if any(k in text for k in ("trade rumor", "shopped")) and not str(row.get("cause_type") or ""):
        return True
    return False


def log_blocked_storyline(session: Any, row: Dict[str, Any], reason: str) -> None:
    migrate_session_storyline_state(session)
    pname = str(row.get("player_name") or "Player")
    et = str(row.get("event_type") or "storyline")
    msg = f"Blocked storyline: {et} for user player {pname} because {reason}."
    log = list(getattr(session, "_storyline_blocked_log", None) or [])
    log.append({"message": msg, "player": pname, "event_type": et, "reason": reason})
    session._storyline_blocked_log = log[-80:]
    if _DEV:
        _log.warning(msg)


def validate_storyline_before_effects(session: Any, storyline: Dict[str, Any]) -> bool:
    """Backend safeguard — negative user-team effects require a registered cause."""
    migrate_session_storyline_state(session)
    utid = str(getattr(session, "user_team_id", "") or "")
    tid = str(storyline.get("team_id") or "")
    cause_type = str(storyline.get("cause_type") or "")
    cause_event_id = str(storyline.get("cause_event_id") or "")
    tone = str(storyline.get("tone") or "").lower()
    is_negative = tone == "negative" or float((storyline.get("effects") or {}).get("player_morale", 0) or 0) < 0

    if tid == utid and is_negative:
        if not cause_type or cause_type not in STORYLINE_CAUSE_TYPES:
            log_blocked_storyline(session, storyline, f"invalid or missing cause_type ({cause_type or 'none'})")
            return False
        if not cause_event_id and cause_type not in (
            "PLAYER_LOW_PRODUCTION",
            "PLAYER_REALDATA_DROP",
            "GOALIE_BAD_FORM",
            "LOSING_STREAK",
            "WINNING_STREAK",
        ):
            log_blocked_storyline(session, storyline, "no cause_event_id")
            return False
        if cause_event_id and find_decision_event(session, cause_event_id) is None:
            if not str(storyline.get("stable_key") or "").startswith(
                ("star_underperforming|", "goalie_meltdown|", "cold_streak|", "losing_skid|")
            ):
                log_blocked_storyline(session, storyline, f"cause_event_id {cause_event_id} not in decision log")
                return False
    pid = str(storyline.get("player_id") or "")
    if pid and _player_from_roster(session, pid) is None and tid == utid:
        log_blocked_storyline(session, storyline, "player not on roster")
        return False
    return True


def _emit_cause_storyline(session: Any, sl: Dict[str, Any]) -> None:
    from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433
    from app.sim_engine.franchise.storyline_conduct import (  # noqa: WPS433
        apply_storyline_ovr_nudge,
        apply_temporary_ovr_modifier,
        build_impact_storyline_fields,
    )

    if not validate_storyline_before_effects(session, sl):
        return

    tid = str(sl.get("team_id") or "")
    pid = str(sl.get("player_id") or "")
    pl = _player_from_roster(session, pid) if pid else None
    sid = str(sl.get("storyline_id") or sl.get("id") or "")

    conduct_fields: Dict[str, Any] = {}
    if pl is not None and int(sl.get("ovr_modifier") or 0) != 0:
        ctype = str(sl.get("cause_type") or "")
        amt = int(sl.get("ovr_modifier") or 0)
        reason = str(sl.get("cause") or sl.get("user_visible_explanation") or "")[:120]
        if amt < 0:
            meta = apply_temporary_ovr_modifier(
                pl,
                source=ctype or "CAUSE_STORYLINE",
                amount=amt,
                reason=reason,
                duration_games=12,
                storyline_id=sid,
                cause_type=ctype,
                cause_event_id=str(sl.get("cause_event_id") or ""),
                modifier_type="storyline_readiness",
            )
        else:
            meta = apply_storyline_ovr_nudge(
                pl,
                amount=amt,
                storyline_id=sid,
                cause_type=ctype,
                cause_event_id=str(sl.get("cause_event_id") or ""),
                reason=reason,
            )
        conduct_fields = build_impact_storyline_fields(meta)
        sl.update(conduct_fields)

    _apply_storyline_effects(session, tid, pid, dict(sl.get("effects") or {}))
    _record_storyline(session, sl)

    active = list(getattr(session, "active_cause_storylines", None) or [])
    active.append(
        {
            "storyline_id": sid,
            "cause_type": sl.get("cause_type"),
            "cause_event_id": sl.get("cause_event_id"),
            "culprit_player_id": sl.get("culprit_player_id") or pid,
            "affected_player_ids": list(sl.get("affected_player_ids") or []),
            "team_id": tid,
            "active_effects": dict(sl.get("effects") or {}),
            "resolution_condition": sl.get("resolution_condition"),
            "resolved": False,
        }
    )
    session.active_cause_storylines = active[-60:]

    _ensure_session_event_lists(session)
    cur = int(sl.get("calendar_day") or getattr(session, "calendar_cursor", 0) or 0)
    iso = str(sl.get("calendar_iso") or "")
    pname = str(sl.get("player_name") or "Player")
    extra = dict(conduct_fields)
    extra.update(
        {
            "cause_type": sl.get("cause_type"),
            "cause_event_id": sl.get("cause_event_id"),
            "culprit_player_id": sl.get("culprit_player_id"),
            "source_label": sl.get("source_label"),
        }
    )
    tail = ""
    if conduct_fields.get("overall_delta"):
        tail = f" · {conduct_fields.get('effect_summary') or ''}"
    session.notifications.append(
        {
            "id": f"notif:{sid}",
            "type": "storyline",
            "priority": str(sl.get("priority") or "MEDIUM"),
            "title": str(sl.get("headline") or "")[:100],
            "text": f"{pname}: {sl.get('short_summary') or sl.get('description') or ''}{tail}",
            "date": iso or cur,
            "calendar_day": cur,
            "calendar_iso": iso,
            "team_id": tid,
            "player_id": pid,
            "source": str(sl.get("source") or "cause_storyline_engine"),
            **{k: v for k, v in extra.items() if k not in ("id",)},
        }
    )
    if tid == str(getattr(session, "user_team_id", "") or ""):
        session.pending_ui_popups.append(
            {
                "id": sid,
                "kind": "storyline",
                "storyline_id": sid,
                "title": str(sl.get("source_label") or "Team Report"),
                "headline": str(sl.get("headline") or ""),
                "summary": str(sl.get("description") or ""),
                "description": str(sl.get("description") or ""),
                "cause": str(sl.get("cause") or ""),
                "cause_type": sl.get("cause_type"),
                "culprit_player_id": sl.get("culprit_player_id"),
                "culprit_player_name": pname,
                "affected_players": sl.get("affected_player_ids") or [],
                "recovery_conditions": sl.get("recovery_conditions") or [],
                "team_id": tid,
                "player_id": pid,
                "player_name": pname,
                "calendar_day": cur,
                "calendar_iso": iso,
                "source_label": sl.get("source_label"),
                "presentation_type": "trade_fallout" if "trade" in str(sl.get("category") or "") else "team_story",
                "theme": "warning",
                "requires_decision": bool(sl.get("requires_action")),
                "choices": sl.get("action_options") or [],
                **conduct_fields,
            }
        )

    if sl.get("requires_action") and sl.get("action_options"):
        _enqueue_decision(session, sl)


def _line_rank_from_slot(group: str, line_id: str, slot: str) -> int:
    if group == "forwards":
        return int(_LINEUP_FORWARD_RANK.get(str(line_id), 4))
    if group == "defense":
        return int(_LINEUP_DEFENSE_RANK.get(str(line_id), 3))
    if group == "goalies":
        return int(_LINEUP_GOALIE_SLOT_RANK.get(str(slot), 3))
    return 5


def _parse_lineup_role_ranks(lines: Any) -> Dict[str, int]:
    """Map player_id -> lineup rank (1 = top line / starter)."""
    out: Dict[str, int] = {}
    if not isinstance(lines, dict):
        return out
    for group in ("forwards", "defense", "goalies"):
        for line in lines.get(group) or []:
            if not isinstance(line, dict):
                continue
            line_id = str(line.get("id") or "")
            for slot, pid in (line.get("slots") or {}).items():
                spid = str(pid or "")
                if spid:
                    out[spid] = _line_rank_from_slot(group, line_id, str(slot))
    return out


def _lineup_calendar_meta(session: Any) -> Tuple[int, str]:
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    iso = ""
    cal = getattr(session, "nhl_calendar", None) or []
    if 0 <= cur < len(cal):
        iso = str(cal[cur].get("iso") or "")
    return cur, iso


def _active_arc_for_event(session: Any, cause_event_id: str) -> bool:
    eid = str(cause_event_id or "")
    if not eid:
        return False
    for arc in getattr(session, "active_cause_storylines", None) or []:
        if str(arc.get("cause_event_id") or "") == eid and not arc.get("resolved"):
            return True
    return False


def _build_lineup_fallout_storyline(
    session: Any,
    *,
    player: Any,
    player_id: str,
    player_name: str,
    team_id: str,
    cause_type: str,
    cause_event_id: str,
    prev_rank: Optional[int],
    new_rank: Optional[int],
    calendar_iso: str,
    cur_day: int,
) -> Optional[Dict[str, Any]]:
    tname = _team_display(session, team_id)
    char = _player_character_0_100(player)
    stable_key = f"{cause_type}|{player_id}|{cause_event_id}"

    if cause_type == "PLAYER_SCRATCHED_BY_USER":
        headline = f"{player_name} scratched — room watching GM's message"
        body = (
            f"{tname} left {player_name} out of the active lineup. "
            f"Teammates are reading into whether this is accountability or punishment."
        )
        effects = {"player_morale": -6 if char < 50 else -3, "room_tension": 3, "lineup_pressure": 4}
        ovr_mod = 2 if char < 45 else 1
        severity = "major" if char < 40 else "minor"
    elif cause_type == "PLAYER_DEMOTED":
        headline = f"{player_name} demoted in lineup shuffle"
        body = (
            f"{player_name} dropped from a top role (line {prev_rank}) to line {new_rank}. "
            f"The camp is watching how {player_name} handles reduced responsibility."
        )
        effects = {"player_morale": -8 if char < 50 else -4, "room_tension": 2, "coach_trust": -2}
        ovr_mod = 3 if char < 45 else 2
        severity = "major" if char < 40 else "mid"
    elif cause_type == "PLAYER_ROLE_REDUCED":
        headline = f"{player_name}'s role trimmed on {tname} depth chart"
        body = (
            f"{player_name} moved down one line (from {prev_rank} to {new_rank}). "
            f"Not a benching, but the usage change is being noticed."
        )
        effects = {"player_morale": -4 if char < 55 else -2, "lineup_pressure": 2}
        ovr_mod = 1
        severity = "minor"
    elif cause_type == "PLAYER_PROMOTED":
        headline = f"{player_name} earns bigger role in lineup"
        body = f"{player_name} promoted from line {prev_rank} to line {new_rank} — confidence building."
        effects = {"player_morale": 4, "player_confidence": 3, "development_confidence": 2}
        ovr_mod = 0
        severity = "minor"
    else:
        return None

    tone = "positive" if cause_type == "PLAYER_PROMOTED" else "negative"
    return {
        "id": _storyline_id(stable_key),
        "storyline_id": _storyline_id(stable_key),
        "stable_key": stable_key,
        "type": "lineup_fallout",
        "category": "lineup",
        "cause_type": cause_type,
        "cause_event_id": cause_event_id,
        "culprit_player_id": player_id if tone == "negative" else "",
        "player_id": player_id,
        "player_name": player_name,
        "affected_player_ids": [player_id],
        "team_id": team_id,
        "team_name": tname,
        "severity": severity,
        "priority": "MEDIUM" if severity == "minor" else "HIGH",
        "tone": tone,
        "headline": headline,
        "title": headline,
        "description": body,
        "short_summary": body[:160],
        "summary": body,
        "cause": f"User saved lineup changes affecting {player_name} ({cause_type.replace('_', ' ').lower()}).",
        "user_visible_explanation": body,
        "calendar_iso": calendar_iso,
        "calendar_day": cur_day,
        "date": calendar_iso or cur_day,
        "source": _SOURCE_LABEL_BY_CAUSE.get(cause_type, "Team Report"),
        "source_label": _SOURCE_LABEL_BY_CAUSE.get(cause_type, "Team Report"),
        "effects": effects,
        "ovr_modifier": -int(ovr_mod) if ovr_mod else 0,
        "ovr_modifier_games": 6 if ovr_mod else 0,
        "recovery_conditions": ["Restore prior role", "Win games together", "Meet with player privately"],
        "resolution_condition": "role_restored_or_time",
        "status": "active",
        "resolved": False,
        "requires_action": cause_type in ("PLAYER_SCRATCHED_BY_USER", "PLAYER_DEMOTED") and char < 45,
        "action_options": (
            [
                {"id": "restore_role", "label": "Restore prior role", "effects": {"player_morale": 4, "room_tension": -2}},
                {"id": "meet_player", "label": "Meet with player privately", "effects": {"player_morale": 3, "gm_trust": 2}},
            ]
            if cause_type in ("PLAYER_SCRATCHED_BY_USER", "PLAYER_DEMOTED")
            else []
        ),
    }


def record_lineup_save_decisions(
    session: Any,
    *,
    new_lines: Any,
    unit_type: str = "even_strength",
    previous_lines: Any = None,
) -> List[Dict[str, Any]]:
    """Compare saved lineup to prior state; log decision events and emit fallout storylines."""
    migrate_session_storyline_state(session)
    utid = str(getattr(session, "user_team_id", "") or "")
    if not utid or unit_type != "even_strength":
        return []
    if not isinstance(new_lines, dict):
        return []

    prev_ranks = _parse_lineup_role_ranks(previous_lines) if previous_lines else {}
    new_ranks = _parse_lineup_role_ranks(new_lines)
    # First save: establish a baseline from roster depth so scratches/demotions can fire.
    if not prev_ranks:
        user_team = (getattr(session, "team_by_id", None) or {}).get(utid)
        roster = [p for p in (getattr(user_team, "roster", None) or []) if not getattr(p, "retired", False)]
        fw = []
        defs = []
        gl = []
        for p in roster:
            pos = str(getattr(getattr(getattr(p, "identity", None), "position", None), "value", None) or getattr(p, "position", "") or "").upper()
            pid = str(getattr(p, "id", "") or "")
            if not pid:
                continue
            if pos == "G":
                gl.append(pid)
            elif pos in ("D", "LD", "RD"):
                defs.append(pid)
            else:
                fw.append(pid)
        for i, pid in enumerate(fw[:12]):
            prev_ranks[pid] = (i // 3) + 1
        for i, pid in enumerate(defs[:6]):
            prev_ranks[pid] = (i // 2) + 1
        for i, pid in enumerate(gl[:3]):
            prev_ranks[pid] = i + 1
        if not prev_ranks:
            return []

    cur_day, iso = _lineup_calendar_meta(session)
    generated: List[Dict[str, Any]] = []
    user_team = (getattr(session, "team_by_id", None) or {}).get(utid)
    roster_ids = {
        str(getattr(p, "id", "") or "")
        for p in (getattr(user_team, "roster", None) or [])
        if str(getattr(p, "id", "") or "")
    }

    for pid in roster_ids:
        if pid not in prev_ranks and pid not in new_ranks:
            continue
        prev_rank = prev_ranks.get(pid)
        new_rank = new_ranks.get(pid)
        pl = _player_from_roster(session, pid)
        if pl is None:
            continue
        pname = str(getattr(pl, "name", "") or "Player")

        event_type = ""
        if prev_rank is not None and new_rank is None:
            event_type = "PLAYER_SCRATCHED_BY_USER"
            try:
                setattr(pl, "_recently_scratched", True)
            except Exception:
                pass
        elif prev_rank is not None and new_rank is not None:
            delta = int(new_rank) - int(prev_rank)
            if delta >= 2:
                event_type = "PLAYER_DEMOTED"
            elif delta == 1:
                event_type = "PLAYER_ROLE_REDUCED"
            elif delta <= -1:
                event_type = "PLAYER_PROMOTED"
            if new_rank is not None:
                try:
                    setattr(pl, "_recently_scratched", False)
                    setattr(pl, "_deployed_line_rank", int(new_rank) - 1)
                except Exception:
                    pass

        if not event_type:
            continue

        eid = record_decision_event(
            session,
            {
                "event_type": event_type,
                "date": iso or cur_day,
                "calendar_day": cur_day,
                "team_id": utid,
                "player_ids": [pid],
                "player_id": pid,
                "player_name": pname,
                "prev_line_rank": prev_rank,
                "new_line_rank": new_rank,
                "unit_type": unit_type,
                "severity": "medium" if event_type in ("PLAYER_DEMOTED", "PLAYER_SCRATCHED_BY_USER") else "low",
            },
        )

        if _active_arc_for_event(session, eid):
            continue

        sl = _build_lineup_fallout_storyline(
            session,
            player=pl,
            player_id=pid,
            player_name=pname,
            team_id=utid,
            cause_type=event_type,
            cause_event_id=eid,
            prev_rank=prev_rank,
            new_rank=new_rank,
            calendar_iso=iso,
            cur_day=cur_day,
        )
        if sl:
            generated.append(sl)
            _emit_cause_storyline(session, sl)

    return generated


def _build_locker_room_conflict_storyline(
    session: Any,
    *,
    culprit: Any,
    culprit_id: str,
    culprit_name: str,
    team_id: str,
    trigger_event: Dict[str, Any],
    cause_event_id: str,
    calendar_iso: str,
    cur_day: int,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    char = _player_character_0_100(culprit)
    if char >= 55:
        return None
    trigger_type = str(trigger_event.get("event_type") or "")
    tname = _team_display(session, team_id)
    stable_key = f"LOW_CHARACTER_CONFLICT|{culprit_id}|{cause_event_id}"

    trigger_labels = {
        "PLAYER_SCRATCHED_BY_USER": f"being scratched from the lineup",
        "PLAYER_DEMOTED": f"a lineup demotion",
        "PLAYER_ROLE_REDUCED": f"reduced ice-time responsibility",
        "TRADE_REJECTED": f"failed trade talks involving {culprit_name}",
        "PLAYER_REPEATEDLY_SHOPPED": f"repeated trade rumors around {culprit_name}",
        "LOSING_STREAK": f"the team's losing stretch",
        "CONTRACT_DISPUTE": f"contract frustration",
        "CAPTAINCY_CHANGED": f"leadership changes in the room",
    }
    trigger_text = trigger_labels.get(trigger_type, "recent team friction")

    headline = f"Locker room tension rises around {culprit_name}"
    body = (
        f"Sources around {tname} say {culprit_name} has become a distraction after {trigger_text}. "
        f"Teammates are growing tired of the negative energy, and coaches are monitoring the situation."
    )
    morale_drop = 6 + min(8, (55 - char) // 4)
    effects = {
        "player_morale": -float(morale_drop),
        "room_tension": float(4 + min(6, (55 - char) // 5)),
        "team_morale": -3.0,
        "coach_trust": -2.0,
    }
    if trigger_type in ("TRADE_REJECTED", "PLAYER_REPEATEDLY_SHOPPED"):
        effects["gm_trust"] = -float(4 + min(6, morale_drop // 2))

    return {
        "id": _storyline_id(stable_key),
        "storyline_id": _storyline_id(stable_key),
        "stable_key": stable_key,
        "type": "locker_room_conflict",
        "category": "team",
        "cause_type": "LOW_CHARACTER_CONFLICT",
        "cause_event_id": cause_event_id,
        "culprit_player_id": culprit_id,
        "player_id": culprit_id,
        "player_name": culprit_name,
        "affected_player_ids": [culprit_id],
        "team_id": team_id,
        "team_name": tname,
        "severity": "major" if char < 35 else "mid",
        "priority": "HIGH" if char < 35 else "MEDIUM",
        "tone": "negative",
        "headline": headline,
        "title": headline,
        "description": body,
        "short_summary": body[:160],
        "summary": body,
        "cause": f"Low-character player reacting to {trigger_text}.",
        "user_visible_explanation": body,
        "calendar_iso": calendar_iso,
        "calendar_day": cur_day,
        "date": calendar_iso or cur_day,
        "source": "Locker Room Pulse",
        "source_label": "Locker Room Pulse",
        "effects": effects,
        "ovr_modifier": -int(2 + min(4, (55 - char) // 8)),
        "ovr_modifier_games": 10,
        "recovery_conditions": [
            "Trade culprit",
            "Restore role",
            "Win games together",
            "Team leadership intervention",
        ],
        "resolution_condition": "culprit_traded_or_attempts_reset",
        "status": "active",
        "resolved": False,
        "requires_action": char < 40,
        "action_options": [
            {"id": "meet_player", "label": "Meet with player privately", "effects": {"player_morale": 3}},
            {"id": "open_tradehub", "label": "Explore trade market", "effects": {"trade_market_heat": 2}},
            {"id": "restore_role", "label": "Restore prominent role", "effects": {"player_morale": 4, "room_tension": -3}},
        ],
    }


def _maybe_record_losing_streak_event(session: Any, team_id: str, cur_day: int, iso: str) -> None:
    if not team_id:
        return
    w, l, o, _ = _team_record(session, team_id)
    gp = w + l + o
    if gp < TEAM_GP_MIN or l < 4 or w > 2:
        return
    stable = f"losing_skid_evt|{team_id}|{cur_day // 7}"
    for ev in reversed(list(getattr(session, "decision_event_log", None) or [])):
        if str(ev.get("event_type") or "") == "LOSING_STREAK" and str(ev.get("team_id") or "") == team_id:
            try:
                if int(ev.get("calendar_day") or 0) >= cur_day - 5:
                    return
            except (TypeError, ValueError):
                pass
    record_decision_event(
        session,
        {
            "event_type": "LOSING_STREAK",
            "event_id": stable,
            "date": iso or cur_day,
            "calendar_day": cur_day,
            "team_id": team_id,
            "severity": "medium",
            "losses": l,
            "wins": w,
        },
    )


def _process_locker_room_triggers(
    session: Any,
    rng: random.Random,
    cur_day: int,
    iso: str,
    utid: str,
) -> int:
    """Turn recent decision events into low-character locker room conflicts."""
    if not utid:
        return 0
    spawned = 0
    recent = list(getattr(session, "decision_event_log", None) or [])[-40:]
    for ev in reversed(recent):
        evt = str(ev.get("event_type") or "")
        if evt not in _LOCKER_ROOM_TRIGGER_EVENTS:
            continue
        tid = str(ev.get("team_id") or utid)
        if tid != utid:
            continue
        try:
            ev_day = int(ev.get("calendar_day") or ev.get("date") or 0)
        except (TypeError, ValueError):
            ev_day = 0
        if cur_day - ev_day > 3:
            continue
        eid = str(ev.get("event_id") or ev.get("id") or "")
        if not eid or _active_arc_for_event(session, eid):
            continue
        if ev.get("_locker_room_spawned"):
            continue

        pid = str(ev.get("player_id") or "")
        if not pid:
            pids = ev.get("player_ids") or []
            pid = str(pids[0]) if pids else ""
        pl = _player_from_roster(session, pid) if pid else None
        if pl is None and evt != "LOSING_STREAK":
            continue

        if evt == "LOSING_STREAK":
            tm = (getattr(session, "team_by_id", None) or {}).get(utid)
            candidates = []
            for p in (getattr(tm, "roster", None) or []):
                if getattr(p, "retired", False):
                    continue
                cid = str(getattr(p, "id", "") or "")
                if cid and _player_character_0_100(p) < 50:
                    candidates.append(p)
            if not candidates:
                continue
            pl = candidates[rng.randint(0, len(candidates) - 1)]
            pid = str(getattr(pl, "id", "") or "")

        char = _player_character_0_100(pl)
        if char >= 55:
            continue
        roll = 0.55 if char < 40 else 0.35 if char < 50 else 0.18
        if rng.random() > roll:
            continue

        pname = str(getattr(pl, "name", "") or "Player")
        sl = _build_locker_room_conflict_storyline(
            session,
            culprit=pl,
            culprit_id=pid,
            culprit_name=pname,
            team_id=utid,
            trigger_event=ev,
            cause_event_id=eid,
            calendar_iso=iso,
            cur_day=cur_day,
            rng=rng,
        )
        if not sl:
            continue
        ev["_locker_room_spawned"] = True
        _emit_cause_storyline(session, sl)
        spawned += 1
        if spawned >= 1:
            break
    return spawned


def ensure_cpu_storyline_cause(session: Any, row: Dict[str, Any], team_id: str) -> Dict[str, Any]:
    """Attach cause metadata and log a decision event for CPU negative storylines."""
    migrate_session_storyline_state(session)
    out = dict(row)
    if str(out.get("cause_type") or ""):
        return out
    et = str(out.get("event_type") or "").lower()
    pol = str(out.get("storyline_polarity") or "negative").lower()
    if pol == "positive":
        return out
    cause_type = _CPU_EVENT_CAUSE_MAP.get(et, "")
    if not cause_type:
        return out
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    pid = str(out.get("player_id") or "")
    pname = str(out.get("player_name") or "Player")
    eid = record_decision_event(
        session,
        {
            "event_type": cause_type,
            "team_id": str(team_id),
            "player_id": pid,
            "player_ids": [pid] if pid else [],
            "player_name": pname,
            "cpu_narrative": True,
            "source_event_type": et,
            "calendar_day": cur,
            "severity": str(out.get("arc_tier") or "minor"),
        },
    )
    out["cause_type"] = cause_type
    out["cause_event_id"] = eid
    out["culprit_player_id"] = pid
    return out


def _ensure_session_event_lists(session: Any) -> None:
    if getattr(session, "notifications", None) is None:
        session.notifications = []
    if getattr(session, "pending_ui_popups", None) is None:
        session.pending_ui_popups = []


def resolve_culprit_traded_storylines(session: Any, moved_players: List[Dict[str, Any]]) -> None:
    """When a conflict culprit is traded away, resolve storyline and heal the room."""
    migrate_session_storyline_state(session)
    utid = str(getattr(session, "user_team_id", "") or "")
    traded_ids = {str(m.get("asset_id") or m.get("player_id") or "") for m in moved_players if str(m.get("asset_id") or m.get("player_id") or "")}
    traded_names = {str(m.get("player_name") or "") for m in moved_players if m.get("player_name")}
    active = list(getattr(session, "active_cause_storylines", None) or [])

    from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433
    from app.sim_engine.franchise.storyline_conduct import (  # noqa: WPS433
        clear_trade_fallout_modifiers,
        resolve_modifiers_for_storyline,
    )

    for pid in traded_ids:
        pl = _player_from_roster(session, pid)
        if pl is None:
            continue
        clear_trade_fallout_modifiers(pl)
        pst = _ensure_player_storyline_state(pl)
        pst["was_recently_shopped"] = False
        pst["trade_rumor_heat"] = 0
        pst["trade_attempt_count"] = 0
        pst["last_trade_rumor_day"] = -1
        pst["last_trade_rumor_week"] = -1
        pst["last_trade_rumor_context"] = ""

    if not active:
        return

    remaining: List[Dict[str, Any]] = []
    for arc in active:
        culprit = str(arc.get("culprit_player_id") or "")
        cause_type = str(arc.get("cause_type") or "")
        if culprit not in traded_ids:
            remaining.append(arc)
            continue
        sid = str(arc.get("storyline_id") or "")
        pl = _player_from_roster(session, culprit)
        pname = str(getattr(pl, "name", "") if pl else "") or next(iter(traded_names), "Player")
        if pl is not None:
            resolve_modifiers_for_storyline(pl, sid)
            clear_trade_fallout_modifiers(pl)
        if cause_type in ("TRADE_REJECTED", "PLAYER_REPEATEDLY_SHOPPED", "TRADE_ATTEMPTED_BY_USER"):
            arc["resolved"] = True
            continue
        tname = _team_display(session, utid)
        headline = f"Room feels lighter after {pname} departure"
        body = (
            f"{tname}'s room feels lighter after moving {pname}. Several players privately felt "
            f"the situation had become a distraction, and the team's leadership group appears relieved."
        )
        sl = {
            "storyline_id": f"{sid}:resolved",
            "type": "locker_room_recovery",
            "category": "team",
            "cause_type": "CULPRIT_TRADED",
            "cause_event_id": str(arc.get("cause_event_id") or ""),
            "culprit_player_id": culprit,
            "team_id": utid,
            "headline": headline,
            "description": body,
            "cause": f"{pname} was traded away, resolving an active locker-room storyline.",
            "tone": "positive",
            "priority": "MEDIUM",
            "effects": {"team_morale": 6.0, "room_tension": -8.0, "coach_trust": 4.0, "media_pressure": -3.0},
            "status": "resolved",
            "resolved": True,
            "resolution_reason": "culprit_traded",
            "source_label": "Locker Room Pulse",
        }
        _apply_storyline_effects(session, utid, "", dict(sl.get("effects") or {}))
        _record_storyline(session, sl)
        arc["resolved"] = True
    session.active_cause_storylines = remaining


def tick_franchise_storyline_modifiers(session: Any) -> None:
    from app.sim_engine.franchise.storyline_conduct import tick_player_ovr_modifiers  # noqa: WPS433

    for tm in (getattr(session, "team_by_id", None) or {}).values():
        for pl in getattr(tm, "roster", None) or []:
            tick_player_ovr_modifiers(pl)


def _decay_trade_rumor_state(session: Any, calendar_idx: int) -> None:
    season = int(getattr(session, "season_calendar_year", 2025) or 2025)
    week = int(calendar_idx // 7)
    for tm in (getattr(session, "team_by_id", None) or {}).values():
        for pl in getattr(tm, "roster", None) or []:
            pst = _ensure_player_storyline_state(pl)
            prev_season = int(pst.get("last_trade_rumor_season") or -1)
            if prev_season >= 0 and prev_season != season:
                pst["trade_attempt_count"] = 0
                pst["trade_rumor_heat"] = 0
                pst["was_recently_shopped"] = False
                pst["last_trade_rumor_day"] = -1
                pst["last_trade_rumor_week"] = -1
                pst["last_trade_rumor_context"] = ""
            last_decay_week = int(pst.get("last_trade_rumor_decay_week") or -1)
            if last_decay_week == week:
                continue
            heat = max(0, int(pst.get("trade_rumor_heat") or 0) - 10)
            pst["trade_rumor_heat"] = heat
            if heat <= 0:
                pst["was_recently_shopped"] = False
                pst["trade_attempt_count"] = max(0, int(pst.get("trade_attempt_count") or 0) - 1)
            pst["last_trade_rumor_decay_week"] = week
            pst["last_trade_rumor_season"] = season


def franchise_cause_storyline_daily_pass(
    session: Any,
    calendar_idx: int,
    day_meta: Dict[str, Any],
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Process pending cause events, modifier ticks, and culprit resolution checks."""
    migrate_session_storyline_state(session)
    tick_franchise_storyline_modifiers(session)
    _decay_trade_rumor_state(session, int(calendar_idx))
    r = rng or random.Random()
    utid = str(getattr(session, "user_team_id", "") or "")
    iso = str(day_meta.get("iso") or "")
    cur = int(calendar_idx)
    _maybe_record_losing_streak_event(session, utid, cur, iso)
    locker_spawned = _process_locker_room_triggers(session, r, cur, iso, utid)
    nu = narrative_universe_daily_pass(session, cur, day_meta, r)
    return {"processed": True, "day": calendar_idx, "locker_room_spawned": locker_spawned, **nu}


def build_storyline_debug_payload(session: Any) -> Dict[str, Any]:
    """Dev-only debug view of cause-linked storylines."""
    migrate_session_storyline_state(session)
    active = list(getattr(session, "active_cause_storylines", None) or [])
    blocked = list(getattr(session, "_storyline_blocked_log", None) or [])[-20:]
    events = list(getattr(session, "decision_event_log", None) or [])[-15:]
    modifiers_sample: List[Dict[str, Any]] = []
    utid = str(getattr(session, "user_team_id", "") or "")
    tm = (getattr(session, "team_by_id", None) or {}).get(utid)
    if tm is not None:
        from app.sim_engine.franchise.storyline_conduct import serialize_ovr_modifiers_for_ui  # noqa: WPS433

        for pl in (getattr(tm, "roster", None) or [])[:8]:
            mods = serialize_ovr_modifiers_for_ui(pl)
            if mods:
                modifiers_sample.append(
                    {
                        "player_id": str(getattr(pl, "id", "") or ""),
                        "player_name": str(getattr(pl, "name", "") or ""),
                        "modifiers": mods,
                    }
                )
    return {
        "active_storylines": active,
        "recent_decision_events": events,
        "blocked_storylines": blocked,
        "active_modifiers": modifiers_sample,
    }


# ---------------------------------------------------------------------------
# Narrative Universe — World Event Ledger, Story Arcs, Media, Social
# Facts vs claims, reporter entities, player narrative memory.
# ---------------------------------------------------------------------------

MEDIA_REPORTERS: List[Dict[str, Any]] = [
    {"id": "ellison", "name": "Mark Ellison", "outlet": "NorthStar Hockey", "role": "national_insider", "credibility_base": 84, "specialty": "trades"},
    {"id": "morin", "name": "Rachel Morin", "outlet": "Team Ledger", "role": "beat_reporter", "credibility_base": 76, "specialty": "local"},
    {"id": "knox", "name": "Derek Knox", "outlet": "NBN", "role": "analyst", "credibility_base": 72, "specialty": "performance"},
    {"id": "reid", "name": "Mason Reid", "outlet": "PuckFinance", "role": "cap_specialist", "credibility_base": 78, "specialty": "contracts"},
    {"id": "petrov", "name": "Alex Petrov", "outlet": "Future Ice", "role": "prospect_analyst", "credibility_base": 70, "specialty": "draft"},
    {"id": "lee", "name": "Jenna Lee", "outlet": "National Sports Desk", "role": "investigative", "credibility_base": 88, "specialty": "conduct"},
    {"id": "hart", "name": "Chris Hart", "outlet": "Hot Take TV", "role": "hot_take", "credibility_base": 48, "specialty": "controversy"},
    {"id": "okada", "name": "Kenji Okada", "outlet": "Pacific Desk", "role": "beat_reporter", "credibility_base": 74, "specialty": "west"},
    {"id": "beaumont", "name": "Claire Beaumont", "outlet": "Atlantic Wire", "role": "beat_reporter", "credibility_base": 74, "specialty": "east"},
    {"id": "howe", "name": "Sam Howe", "outlet": "Crease Report", "role": "analyst", "credibility_base": 80, "specialty": "goalies"},
    {"id": "dops", "name": "League Desk", "outlet": "Department of Player Safety", "role": "officials", "credibility_base": 90, "specialty": "discipline"},
]

_REPORTER_BY_ID = {r["id"]: r for r in MEDIA_REPORTERS}

_CLAIM_CAUSE_TYPES = frozenset({
    "TRADE_DEMAND", "TRADE_RUMOR", "TRADE_HUB_REJECT", "TRADE_HUB_SOFT_BLOCK",
    "LOCKER_ROOM_PULSE", "AGENT_DISSATISFACTION",
    "CONTRACT_YEAR_HEAT", "PLAYER_ROLE_FRUSTRATION", "ROOM_BELONGING",
    "WINNING_CONCERN", "COACH_HOT_SEAT", "GM_JOB_SECURITY",
})

_FACT_CAUSE_TYPES = frozenset({
    "PLAYER_TRADED", "LINEUP_SCRATCH", "LINEUP_PROMOTION", "INJURY", "PLAYER_LOW_PRODUCTION",
    "GOALIE_BAD_FORM", "LOSING_STREAK", "WINNING_STREAK",
    "HAT_TRICK", "SHUTOUT", "OT_WINNER", "FIRST_GOAL", "ON_ICE_ALTERCATION",
    "AHL_CALLUP", "AHL_SENDDOWN", "ROLLING_HOT", "ROLLING_COLD", "CAPTAINCY_PULSE",
})


def _pick_reporter_for_storyline(sl: Dict[str, Any], session: Any) -> Dict[str, Any]:
    cat = str(sl.get("category") or sl.get("type") or "").lower()
    ctype = str(sl.get("cause_type") or "").upper()
    if "draft" in cat or "prospect" in cat or ctype == "DRAFT_STOCK":
        return _REPORTER_BY_ID["petrov"]
    if "contract" in cat or "cap" in cat or ctype.startswith("CONTRACT"):
        return _REPORTER_BY_ID["reid"]
    if sl.get("information_status") or sl.get("legal_status") or "legal" in cat or "conduct" in cat:
        return _REPORTER_BY_ID["lee"]
    if "trade" in cat or "rumor" in cat or ctype.startswith("TRADE"):
        return _REPORTER_BY_ID["ellison"]
    if "injury" in cat:
        return _REPORTER_BY_ID.get("knox") or MEDIA_REPORTERS[2]
    if "goalie" in cat or ctype == "SHUTOUT":
        return _REPORTER_BY_ID.get("howe") or _REPORTER_BY_ID.get("knox") or MEDIA_REPORTERS[2]
    if ctype in ("ON_ICE_ALTERCATION",) or "discipline" in cat:
        return _REPORTER_BY_ID.get("dops") or _REPORTER_BY_ID.get("lee") or MEDIA_REPORTERS[5]
    if ctype in ("COACH_HOT_SEAT", "GM_JOB_SECURITY", "AHL_CALLUP", "AHL_SENDDOWN"):
        return _REPORTER_BY_ID.get("ellison") or MEDIA_REPORTERS[0]
    if "locker" in cat or ctype in ("ROOM_BELONGING", "PLAYER_ROLE_FRUSTRATION", "CAPTAINCY_PULSE", "LOCKER_ROOM_PULSE"):
        return _REPORTER_BY_ID.get("morin") or MEDIA_REPORTERS[1]
    utid = str(getattr(session, "user_team_id") or "")
    if str(sl.get("team_id") or "") == utid:
        return _REPORTER_BY_ID["morin"]
    if int(sl.get("heat") or 0) >= 70 and sl.get("priority") != "CRITICAL":
        return _REPORTER_BY_ID["hart"]
    return _REPORTER_BY_ID["knox"]


def _knowledge_type_for_storyline(sl: Dict[str, Any]) -> str:
    if sl.get("knowledge_type"):
        return str(sl.get("knowledge_type"))
    ctype = str(sl.get("cause_type") or "").upper()
    cat = str(sl.get("category") or sl.get("type") or "").lower()
    if sl.get("information_status") or sl.get("legal_status"):
        return "claim"
    if ctype in _CLAIM_CAUSE_TYPES or "rumor" in cat:
        cred = int(sl.get("credibility") or 0)
        if cred >= 85:
            return "corroborated_claim"
        return "claim"
    if ctype in _FACT_CAUSE_TYPES or sl.get("execution"):
        return "fact"
    if int(sl.get("credibility") or 0) < 40:
        return "speculation"
    return "report"


def _narrative_angle(sl: Dict[str, Any]) -> str:
    ctype = str(sl.get("cause_type") or "").upper()
    cat = str(sl.get("category") or "").lower()
    if "trade" in cat or "TRADE" in ctype:
        return "trade_market"
    if "injury" in cat:
        return "injury_watch"
    if "goalie" in cat:
        return "goaltending"
    if "draft" in cat or "prospect" in cat:
        return "draft_board"
    if "contract" in cat:
        return "contract_battle"
    if sl.get("information_status"):
        return "conduct_desk"
    if "performance" in cat or "underperform" in cat:
        return "slump_watch"
    return "league_wire"


def _arc_phase_from_beat_index(beat_index: int, heat: int = 0) -> str:
    if beat_index <= 0:
        return "curiosity"
    if beat_index == 1:
        return "concern"
    if beat_index <= 3:
        return "pressure"
    if beat_index <= 5:
        return "conflict" if heat >= 55 else "developing"
    return "resolution" if heat < 30 else "escalating"


def narrative_director_score(session: Any, sl: Dict[str, Any]) -> int:
    score = 18.0
    ovr = int(sl.get("player_overall") or 0)
    if ovr >= 92:
        score += 38
    elif ovr >= 86:
        score += 24
    elif ovr >= 80:
        score += 12
    priority = str(sl.get("priority") or "MEDIUM").upper()
    if priority == "CRITICAL":
        score += 42
    elif priority == "HIGH":
        score += 22
    if sl.get("requires_action"):
        score += 18
    if str(sl.get("team_id") or "") == str(getattr(session, "user_team_id") or ""):
        score += 10
    cat = str(sl.get("category") or "").lower()
    if "trade" in cat or "rumor" in cat:
        score += 14
    return int(_clamp(score, 5, 100))


def _default_credibility(sl: Dict[str, Any], reporter: Dict[str, Any]) -> int:
    base = int(reporter.get("credibility_base") or 70)
    ktype = _knowledge_type_for_storyline(sl)
    if ktype == "fact":
        return min(95, base + 8)
    if ktype == "speculation":
        return max(18, base - 28)
    if ktype == "corroborated_claim":
        return min(92, base + 4)
    return max(25, base - 10)


def _arc_key_for_storyline(sl: Dict[str, Any]) -> str:
    explicit = str(sl.get("arc_id") or sl.get("storyline_id") or "").strip()
    if explicit and not explicit.startswith("story_"):
        return explicit
    sid = str(sl.get("storyline_id") or "").strip()
    if sid:
        return sid
    pid = str(sl.get("player_id") or "")
    ctype = str(sl.get("cause_type") or sl.get("category") or "general")
    if pid:
        return f"arc:{pid}:{ctype}"
    stable = str(sl.get("stable_key") or "")
    if stable:
        return f"arc:{stable.split('|')[0]}"
    return f"arc:{uuid.uuid4().hex[:12]}"


def _append_story_arc_beat(session: Any, sl: Dict[str, Any]) -> Dict[str, Any]:
    arcs = list(getattr(session, "story_arcs", None) or [])
    arc_id = _arc_key_for_storyline(sl)
    beat_id = f"beat_{uuid.uuid4().hex[:10]}"
    heat = int(sl.get("heat") or 0)
    beat = {
        "beat_id": beat_id,
        "headline": str(sl.get("headline") or ""),
        "summary": str(sl.get("summary") or sl.get("short_summary") or ""),
        "calendar_iso": str(sl.get("calendar_iso") or sl.get("date") or ""),
        "calendar_day": sl.get("calendar_day"),
        "knowledge_type": sl.get("knowledge_type"),
        "reporter_id": sl.get("reporter_id"),
        "world_event_id": sl.get("world_event_id"),
        "heat": heat,
        "credibility": sl.get("credibility"),
    }
    existing = next((a for a in arcs if str(a.get("arc_id") or "") == arc_id), None)
    if existing:
        beats = list(existing.get("beats") or [])
        beat_index = len(beats)
        beats.append(beat)
        existing["beats"] = beats
        existing["headline"] = beat["headline"] or existing.get("headline")
        existing["heat"] = max(int(existing.get("heat") or 0), heat)
        existing["updated_iso"] = beat["calendar_iso"]
        existing["phase"] = _arc_phase_from_beat_index(beat_index, heat)
        existing["beat_count"] = len(beats)
        if str(sl.get("arc_status") or sl.get("status") or "").lower() == "resolved":
            existing["status"] = "resolved"
    else:
        beat_index = 0
        arcs.append(
            {
                "arc_id": arc_id,
                "headline": beat["headline"],
                "player_id": sl.get("player_id"),
                "player_name": sl.get("player_name"),
                "team_id": sl.get("team_id"),
                "team_name": sl.get("team_name"),
                "category": sl.get("category") or sl.get("type"),
                "cause_type": sl.get("cause_type"),
                "narrative_angle": sl.get("narrative_angle"),
                "status": str(sl.get("arc_status") or sl.get("status") or "active"),
                "phase": _arc_phase_from_beat_index(0, heat),
                "heat": heat,
                "beats": [beat],
                "beat_count": 1,
                "started_iso": beat["calendar_iso"],
                "updated_iso": beat["calendar_iso"],
            }
        )
    session.story_arcs = arcs[-80:]
    sl["arc_id"] = arc_id
    sl["beat_id"] = beat_id
    sl["beat_index"] = beat_index
    sl["arc_phase"] = _arc_phase_from_beat_index(beat_index, heat)
    sl["repeat_count"] = max(int(sl.get("repeat_count") or 0), beat_index)
    return sl


def _spawn_social_posts(session: Any, sl: Dict[str, Any]) -> None:
    heat = int(sl.get("heat") or 0)
    if heat < 12 and str(sl.get("priority") or "") not in ("CRITICAL", "HIGH"):
        return
    from app.sim_engine.franchise.social_copy_engine import (  # noqa: WPS433
        build_evidence_context,
        compose_ambient_fan_post,
        compose_reporter_post,
    )

    posts = list(getattr(session, "social_posts", None) or [])
    reporter = _REPORTER_BY_ID.get(str(sl.get("reporter_id") or "")) or _pick_reporter_for_storyline(sl, session)
    iso = str(sl.get("calendar_iso") or sl.get("date") or "")
    story_id = str(sl.get("storyline_id") or sl.get("id") or "")
    rng = random.Random(_u_seed(story_id, reporter.get("id"), heat))

    posts.append(
        {
            "id": f"soc_{uuid.uuid4().hex[:10]}",
            "arc_id": sl.get("arc_id"),
            "beat_id": sl.get("beat_id"),
            "storyline_id": story_id,
            "author_type": "reporter",
            "author_id": reporter["id"],
            "author_name": reporter["name"],
            "handle": f"@{reporter['name'].replace(' ', '')}",
            "outlet": reporter["outlet"],
            "verified": True,
            "text": compose_reporter_post(sl, reporter, rng, session)[:280],
            "related_headline": str(sl.get("headline") or ""),
            "calendar_iso": iso,
            "heat": heat,
            "knowledge_type": sl.get("knowledge_type"),
            "platform": "twitter_style_feed",
            "likes": int(heat * 140 + random.randint(80, 900)),
            "reposts": int(heat * 35 + random.randint(10, 200)),
            "replies": int(heat * 18 + random.randint(5, 120)),
        }
    )

    pname = str(sl.get("player_name") or "")
    if pname and heat >= 28:
        ctx = build_evidence_context(sl, session)
        sentiment = "outrage" if heat >= 55 else "concern" if heat >= 40 else "hype"
        posts.append(
            {
                "id": f"soc_{uuid.uuid4().hex[:10]}",
                "arc_id": sl.get("arc_id"),
                "storyline_id": story_id,
                "author_type": "fan",
                "author_name": f"{pname.split()[-1]} Sicko",
                "handle": f"@Fan{abs(hash(pname)) % 9000 + 1000}",
                "verified": False,
                "text": compose_ambient_fan_post(sentiment, ctx, rng, sl, reporter),
                "related_headline": str(sl.get("headline") or ""),
                "calendar_iso": iso,
                "heat": max(10, heat - 15),
                "platform": "twitter_style_feed",
                "likes": int(heat * 45 + random.randint(20, 400)),
                "reposts": int(heat * 8),
                "replies": int(heat * 5),
            }
        )

    session.social_posts = posts[-200:]
    _u_add_reddit_thread(session, sl, rng)


def _update_player_narrative_memory(session: Any, sl: Dict[str, Any]) -> None:
    pid = str(sl.get("player_id") or "")
    if not pid:
        return
    mem = dict(getattr(session, "player_narrative_memory", None) or {})
    row = dict(mem.get(pid) or {})
    tags = list(row.get("reputation_tags") or [])
    angle = str(sl.get("narrative_angle") or "")
    tag_map = {
        "trade_market": "Trade speculation magnet",
        "injury_watch": "Injury narrative",
        "goaltending": "Goaltending storyline",
        "draft_board": "Draft buzz",
        "contract_battle": "Contract storyline",
        "conduct_desk": "Off-ice scrutiny",
        "slump_watch": "Performance storyline",
    }
    tag = tag_map.get(angle)
    if tag and tag not in tags:
        tags.append(tag)
        tags = tags[-8:]
    headlines = list(row.get("headlines") or [])
    hl = str(sl.get("headline") or "")
    if hl:
        headlines.append({"headline": hl, "iso": sl.get("calendar_iso"), "arc_id": sl.get("arc_id")})
        headlines = headlines[-24:]
    row.update(
        {
            "player_id": pid,
            "player_name": sl.get("player_name") or row.get("player_name"),
            "reputation_tags": tags,
            "headlines": headlines,
            "active_arc_id": sl.get("arc_id"),
            "media_heat": max(int(row.get("media_heat") or 0), int(sl.get("heat") or 0)),
            "last_updated_iso": sl.get("calendar_iso"),
        }
    )
    mem[pid] = row
    session.player_narrative_memory = mem


def enrich_storyline_for_narrative_universe(session: Any, event: Dict[str, Any]) -> Dict[str, Any]:
    """Attach arc beats, reporter, knowledge typing, social posts, and player memory."""
    migrate_session_storyline_state(session)
    sl = dict(event or {})
    if not str(sl.get("headline") or sl.get("title") or "").strip():
        return sl

    if not sl.get("storyline_id"):
        sl["storyline_id"] = str(sl.get("id") or f"story_{uuid.uuid4().hex[:12]}")

    cur = int(sl.get("calendar_day") or getattr(session, "calendar_cursor", 0) or 0)
    cal = getattr(session, "nhl_calendar", None) or []
    if not sl.get("calendar_iso") and 0 <= cur < len(cal):
        sl["calendar_iso"] = str(cal[cur].get("iso") or "")

    if not sl.get("cause_event_id") and str(sl.get("cause_type") or ""):
        eid = record_decision_event(
            session,
            {
                "event_type": str(sl["cause_type"]),
                "team_id": sl.get("team_id"),
                "player_id": sl.get("player_id"),
                "player_name": sl.get("player_name"),
                "headline": sl.get("headline"),
                "knowledge_type": "fact",
                "calendar_iso": sl.get("calendar_iso"),
                "calendar_day": cur,
            },
        )
        sl["world_event_id"] = eid
        sl.setdefault("cause_event_id", eid)
    elif sl.get("cause_event_id"):
        sl["world_event_id"] = str(sl["cause_event_id"])

    reporter = _pick_reporter_for_storyline(sl, session)
    sl["reporter_id"] = reporter["id"]
    sl["reporter_name"] = reporter["name"]
    sl["outlet_id"] = reporter["id"]
    sl["outlet_name"] = reporter["outlet"]
    sl.setdefault("source_label", f"{reporter['name']} · {reporter['outlet']}")

    sl.setdefault("heat", narrative_director_score(session, sl))
    sl.setdefault("credibility", _default_credibility(sl, reporter))
    sl["knowledge_type"] = _knowledge_type_for_storyline(sl)
    sl["narrative_angle"] = _narrative_angle(sl)

    sl = _assign_knowledge_layers(session, sl)
    sl = _apply_market_media_tone(session, sl)
    sl = _append_story_arc_beat(session, sl)
    _maybe_agent_leak(session, sl)
    _spawn_social_posts(session, sl)
    _update_player_narrative_memory(session, sl)
    _archive_storyline_beat(session, sl)
    _maybe_queue_press_conference(session, sl)
    signal = _breaking_news_signal(sl)
    if signal:
        sl["breaking_level"] = signal
    return sl


def build_narrative_universe_payload(session: Any) -> Dict[str, Any]:
    """API payload for Storylines UI + player dossier media tabs."""
    migrate_session_storyline_state(session)
    mem = getattr(session, "player_narrative_memory", None) or {}
    eras = build_narrative_eras(session)
    utid = str(getattr(session, "user_team_id") or "")
    user_market = _market_profile_for_team(session, utid) if utid else MARKET_MEDIA_PROFILES["default"]
    recent_breaking = [
        s for s in (getattr(session, "storyline_events", None) or [])[-30:]
        if isinstance(s, dict) and s.get("breaking_level") in ("breaking", "league_defining")
    ][-3:]
    return {
        "reporters": list(MEDIA_REPORTERS),
        "agents": list(PLAYER_AGENTS),
        "story_arcs": list(getattr(session, "story_arcs", None) or [])[-40:],
        "social_posts": list(getattr(session, "social_posts", None) or [])[-120:],
        "reddit_threads": list(getattr(session, "reddit_threads", None) or [])[-100:],
        "reddit_engagement_pulse": list(getattr(session, "reddit_engagement_pulse", None) or [])[-20:],
        "world_events": list(getattr(session, "decision_event_log", None) or [])[-30:],
        "knowledge_graph": list(getattr(session, "knowledge_graph", None) or [])[-80:],
        "player_narrative_memory": dict(mem) if isinstance(mem, dict) else {},
        "press_conference_queue": list(getattr(session, "press_conference_queue", None) or []),
        "narrative_archive": list(getattr(session, "narrative_archive", None) or [])[-120:],
        "narrative_eras": eras,
        "prospect_social_profiles": dict(getattr(session, "prospect_social_profiles", None) or {}),
        "agent_relationships": dict(getattr(session, "agent_relationships", None) or {}),
        "user_market_profile": user_market,
        "market_profiles": MARKET_MEDIA_PROFILES,
        "active_arc_count": len(getattr(session, "story_arcs", None) or []),
        "breaking_alerts": [
            {
                "headline": s.get("headline"),
                "level": s.get("breaking_level"),
                "storyline_id": s.get("storyline_id") or s.get("id"),
                "calendar_iso": s.get("calendar_iso"),
            }
            for s in recent_breaking
        ],
    }


def player_narrative_profile(session: Any, player_id: str) -> Dict[str, Any]:
    """Single-player slice for roster/draft dossier media tab."""
    migrate_session_storyline_state(session)
    pid = str(player_id or "")
    mem = dict(getattr(session, "player_narrative_memory", None) or {})
    profile = dict(mem.get(pid) or {})
    arcs = [
        a for a in (getattr(session, "story_arcs", None) or [])
        if str(a.get("player_id") or "") == pid
    ]
    posts = [
        p for p in (getattr(session, "social_posts", None) or [])
        if pid and pid in str(p.get("related_headline") or "") + str(p.get("text") or "")
    ][-12:]
    return {
        "player_id": pid,
        "reputation_tags": list(profile.get("reputation_tags") or []),
        "headlines": list(profile.get("headlines") or []),
        "active_arc_id": profile.get("active_arc_id"),
        "media_heat": profile.get("media_heat"),
        "story_arcs": arcs[-6:],
        "social_posts": posts,
    }


# ---------------------------------------------------------------------------
# Extended Narrative Universe — knowledge graph, agents, press, markets,
# historical archive, prospect social, breaking-news signals.
# ---------------------------------------------------------------------------

PLAYER_AGENTS: List[Dict[str, Any]] = [
    {"id": "carter", "name": "Allan Carter", "agency": "Carter Hockey Group", "style": "leaker", "leak_tendency": 0.74, "negotiation": "aggressive"},
    {"id": "walsh", "name": "Patricia Walsh", "agency": "Walsh Sports", "style": "discreet", "leak_tendency": 0.09, "negotiation": "patient"},
    {"id": "kim", "name": "Daniel Kim", "agency": "Kim & Partners", "style": "leverage", "leak_tendency": 0.52, "negotiation": "competitive"},
    {"id": "rossi", "name": "Marco Rossi", "agency": "Northline Athletes", "style": "media_savvy", "leak_tendency": 0.38, "negotiation": "stable"},
    {"id": "blake", "name": "Jordan Blake", "agency": "Blake Advisory", "style": "disruptor", "leak_tendency": 0.61, "negotiation": "demanding"},
]

_AGENT_BY_ID = {a["id"]: a for a in PLAYER_AGENTS}

# Market intensity shapes heat amplification and media tone copy.
MARKET_MEDIA_PROFILES: Dict[str, Dict[str, Any]] = {
    "montreal": {"label": "Montreal", "pressure_mult": 1.48, "tone": "intense", "descriptor": "Fishbowl scrutiny"},
    "toronto": {"label": "Toronto", "pressure_mult": 1.42, "tone": "relentless", "descriptor": "National microscope"},
    "vancouver": {"label": "Vancouver", "pressure_mult": 1.28, "tone": "volatile", "descriptor": "Passionate and unforgiving"},
    "ottawa": {"label": "Ottawa", "pressure_mult": 1.18, "tone": "focused", "descriptor": "Capital-city pressure"},
    "edmonton": {"label": "Edmonton", "pressure_mult": 1.22, "tone": "hungry", "descriptor": "Star-driven expectations"},
    "calgary": {"label": "Calgary", "pressure_mult": 1.15, "tone": "traditional", "descriptor": "Old-school hockey town"},
    "winnipeg": {"label": "Winnipeg", "pressure_mult": 1.12, "tone": "loyal", "descriptor": "Small market, loud arena"},
    "boston": {"label": "Boston", "pressure_mult": 1.25, "tone": "demanding", "descriptor": "Championship standard"},
    "new_york": {"label": "New York", "pressure_mult": 1.35, "tone": "tabloid", "descriptor": "Back-page obsession"},
    "philadelphia": {"label": "Philadelphia", "pressure_mult": 1.20, "tone": "hostile", "descriptor": "Unfiltered fan base"},
    "chicago": {"label": "Chicago", "pressure_mult": 1.14, "tone": "proud", "descriptor": "Original Six weight"},
    "detroit": {"label": "Detroit", "pressure_mult": 1.10, "tone": "steady", "descriptor": "Hockeytown expectations"},
    "tampa": {"label": "Tampa Bay", "pressure_mult": 0.88, "tone": "calm", "descriptor": "Winner's patience"},
    "florida": {"label": "Florida", "pressure_mult": 0.78, "tone": "relaxed", "descriptor": "Sunbelt spotlight"},
    "arizona": {"label": "Arizona", "pressure_mult": 0.72, "tone": "quiet", "descriptor": "Low daily volume"},
    "default": {"label": "League", "pressure_mult": 1.0, "tone": "standard", "descriptor": "Standard NHL coverage"},
}

_CITY_TO_MARKET_KEY = {
    "montreal": "montreal", "montréal": "montreal", "toronto": "toronto", "vancouver": "vancouver",
    "ottawa": "ottawa", "edmonton": "edmonton", "calgary": "calgary", "winnipeg": "winnipeg",
    "boston": "boston", "buffalo": "new_york", "new york": "new_york", "brooklyn": "new_york",
    "philadelphia": "philadelphia", "chicago": "chicago", "detroit": "detroit",
    "tampa": "tampa", "tampa bay": "tampa", "miami": "florida", "sunrise": "florida",
    "fort lauderdale": "florida", "phoenix": "arizona", "utah": "default",
}


def _market_key_for_team(session: Any, team_id: str) -> str:
    tm = (getattr(session, "team_by_id", None) or {}).get(str(team_id or ""))
    if tm is None:
        return "default"
    city = str(getattr(tm, "city", "") or "").strip().lower()
    for fragment, key in _CITY_TO_MARKET_KEY.items():
        if fragment in city:
            return key
    name = str(getattr(tm, "name", "") or "").lower()
    for fragment, key in _CITY_TO_MARKET_KEY.items():
        if fragment in name:
            return key
    return "default"


def _market_profile_for_team(session: Any, team_id: str) -> Dict[str, Any]:
    key = _market_key_for_team(session, team_id)
    prof = dict(MARKET_MEDIA_PROFILES.get(key) or MARKET_MEDIA_PROFILES["default"])
    prof["market_key"] = key
    return prof


def _assign_knowledge_layers(session: Any, sl: Dict[str, Any]) -> Dict[str, Any]:
    """Separate facts from claims and track who knows what."""
    ktype = str(sl.get("knowledge_type") or "report")
    utid = str(getattr(session, "user_team_id") or "")
    tid = str(sl.get("team_id") or "")
    is_user = tid == utid and bool(tid)
    layers = {
        "world_fact": bool(sl.get("world_event_id")),
        "gm_private": is_user and ktype in ("fact", "report"),
        "team_internal": is_user,
        "league_office": bool(sl.get("legal_status") or sl.get("league_status")),
        "public": ktype in ("fact", "report", "corroborated_claim"),
        "media_claim": ktype in ("claim", "speculation", "corroborated_claim"),
    }
    if ktype == "fact":
        public_level = "confirmed"
    elif ktype == "corroborated_claim":
        public_level = "widely_reported"
    elif ktype == "claim":
        public_level = "rumour"
    elif ktype == "speculation":
        public_level = "chatter"
    else:
        public_level = "reported"
    gm_knows_more = is_user and ktype == "fact" and public_level != "confirmed"
    sl["knowledge_layers"] = layers
    sl["public_knowledge_level"] = sl.get("public_knowledge_level") or public_level
    sl["gm_knows_more"] = gm_knows_more
    sl["visibility"] = "public" if layers["public"] else ("team_only" if is_user else "private")
    graph = list(getattr(session, "knowledge_graph", None) or [])
    graph.append(
        {
            "world_event_id": sl.get("world_event_id"),
            "storyline_id": sl.get("storyline_id"),
            "team_id": tid,
            "player_id": sl.get("player_id"),
            "player_name": sl.get("player_name"),
            "headline": sl.get("headline"),
            "summary": sl.get("summary") or sl.get("description") or sl.get("short_summary"),
            "knowledge_type": ktype,
            "public_knowledge_level": sl.get("public_knowledge_level") or public_level,
            "source_label": sl.get("source_label") or sl.get("reporter_name") or "",
            "reporter_name": sl.get("reporter_name") or "",
            "outlet_name": sl.get("outlet_name") or "",
            "specialty": sl.get("narrative_angle") or sl.get("category"),
            "heat": sl.get("heat"),
            "category": sl.get("category"),
            "gm_knows_more": gm_knows_more,
            "calendar_iso": sl.get("calendar_iso"),
        }
    )
    session.knowledge_graph = graph[-250:]
    return sl


def _apply_market_media_tone(session: Any, sl: Dict[str, Any]) -> Dict[str, Any]:
    tid = str(sl.get("team_id") or getattr(session, "user_team_id") or "")
    prof = _market_profile_for_team(session, tid)
    mult = float(prof.get("pressure_mult") or 1.0)
    base_heat = int(sl.get("heat") or 0)
    sl["market_key"] = prof.get("market_key")
    sl["market_tone"] = prof.get("tone")
    sl["market_descriptor"] = prof.get("descriptor")
    sl["heat"] = int(_clamp(base_heat * mult, 5, 100))
    if mult >= 1.25 and not sl.get("short_summary"):
        sl["short_summary"] = str(sl.get("summary") or sl.get("headline") or "")
    return sl


def _agent_for_player(session: Any, player_id: str, rng: Optional[random.Random] = None) -> Dict[str, Any]:
    rel = dict(getattr(session, "agent_relationships", None) or {})
    pid = str(player_id or "")
    if pid and pid in rel:
        aid = str(rel[pid].get("agent_id") or "")
        if aid in _AGENT_BY_ID:
            return _AGENT_BY_ID[aid]
    r = rng or random.Random()
    agent = r.choice(PLAYER_AGENTS)
    if pid:
        rel[pid] = {"agent_id": agent["id"], "trust": 0.55, "gm_trust": 0.5}
        session.agent_relationships = rel
    return agent


def _maybe_agent_leak(session: Any, sl: Dict[str, Any], rng: Optional[random.Random] = None) -> None:
    """Agents occasionally leak partial truths — separate fact from public claim."""
    r = rng or random.Random()
    pid = str(sl.get("player_id") or "")
    if not pid:
        return
    agent = _agent_for_player(session, pid, r)
    leak = float(agent.get("leak_tendency") or 0.3)
    ctype = str(sl.get("cause_type") or "").upper()
    if ctype not in _CLAIM_CAUSE_TYPES and "trade" not in str(sl.get("category") or "").lower() and ctype not in (
        "CONTRACT_YEAR_HEAT",
        "PLAYER_ROLE_FRUSTRATION",
        "COACH_HOT_SEAT",
        "GM_JOB_SECURITY",
        "WINNING_CONCERN",
    ):
        return
    if r.random() > leak * 0.35:
        return
    posts = list(getattr(session, "social_posts", None) or [])
    reporter = _REPORTER_BY_ID.get("ellison", MEDIA_REPORTERS[0])
    posts.append(
        {
            "id": f"soc_leak_{uuid.uuid4().hex[:10]}",
            "arc_id": sl.get("arc_id"),
            "storyline_id": sl.get("storyline_id"),
            "author_type": "agent",
            "author_id": agent["id"],
            "author_name": agent["name"],
            "handle": f"@{agent['agency'].replace(' ', '')}",
            "verified": False,
            "text": "My client wants to stay — but we will explore every option.",
            "related_headline": str(sl.get("headline") or ""),
            "calendar_iso": sl.get("calendar_iso"),
            "heat": max(20, int(sl.get("heat") or 0) - 10),
            "knowledge_type": "claim",
            "leak_chain": [agent["id"], reporter["id"]],
            "likes": int(r.randint(400, 2400)),
            "reposts": int(r.randint(80, 600)),
            "replies": int(r.randint(120, 900)),
        }
    )
    session.social_posts = posts[-200:]


def _build_press_questions(sl: Dict[str, Any], session: Any) -> List[Dict[str, Any]]:
    headline = str(sl.get("headline") or "the situation")
    pname = str(sl.get("player_name") or "the player")
    reporter = _pick_reporter_for_storyline(sl, session)
    return [
        {
            "id": "q_coach",
            "reporter_id": reporter["id"],
            "reporter_name": reporter["name"],
            "outlet": reporter["outlet"],
            "question": f"What's your message to fans concerned about {headline.lower()}?",
            "responses": [
                {"id": "deflect", "label": "Deflect", "description": "Keep internal matters internal.", "tone": "neutral"},
                {"id": "support_staff", "label": "Back the staff", "description": "Reaffirm confidence in coaching decisions.", "tone": "firm"},
                {"id": "support_player", "label": "Back the player", "description": "Publicly support {0}.".format(pname), "tone": "diplomatic"},
                {"id": "no_comment", "label": "Decline comment", "description": "Silence — media will interpret.", "tone": "cold"},
            ],
        },
        {
            "id": "q_trade",
            "reporter_id": "ellison",
            "reporter_name": "Mark Ellison",
            "outlet": "NorthStar Hockey",
            "question": f"Can you deny trade rumours swirling around {pname}?",
            "responses": [
                {"id": "deny", "label": "Deny actively shopping", "description": "State he is not being shopped.", "tone": "firm"},
                {"id": "neither_confirm", "label": "Neither confirm nor deny", "description": "Classic GM dodge — heat may rise.", "tone": "neutral"},
                {"id": "listen", "label": "Acknowledge calls", "description": "Admit teams have inquired.", "tone": "honest"},
            ],
        },
    ]


def _maybe_queue_press_conference(session: Any, sl: Dict[str, Any]) -> None:
    utid = str(getattr(session, "user_team_id") or "")
    if str(sl.get("team_id") or "") != utid:
        return
    heat = int(sl.get("heat") or 0)
    priority = str(sl.get("priority") or "")
    if heat < 52 and priority not in ("CRITICAL", "HIGH"):
        return
    queue = list(getattr(session, "press_conference_queue", None) or [])
    sid = str(sl.get("storyline_id") or "")
    if any(str(p.get("storyline_id") or "") == sid for p in queue):
        return
    if len([p for p in queue if str(p.get("status") or "") == "pending"]) >= 3:
        return
    press_id = f"press_{uuid.uuid4().hex[:10]}"
    queue.append(
        {
            "id": press_id,
            "storyline_id": sid,
            "arc_id": sl.get("arc_id"),
            "headline": sl.get("headline"),
            "summary": sl.get("summary") or sl.get("short_summary"),
            "heat": heat,
            "player_id": sl.get("player_id"),
            "player_name": sl.get("player_name"),
            "calendar_iso": sl.get("calendar_iso"),
            "status": "pending",
            "questions": _build_press_questions(sl, session),
            "requires_action": True,
        }
    )
    session.press_conference_queue = queue[-12:]
    sl["press_conference_id"] = press_id
    sl["requires_action"] = True
    opts: List[Dict[str, Any]] = []
    for q in queue[-1]["questions"]:
        for resp in q.get("responses") or []:
            opts.append(
                {
                    "id": f"{q['id']}:{resp['id']}",
                    "label": f"{q['reporter_name']}: {resp['label']}",
                    "effect_summary": resp.get("description"),
                    "effects": {"press_tone": resp.get("tone"), "question_id": q["id"], "response_id": resp["id"]},
                }
            )
    if opts:
        sl["action_options"] = opts[:6]


def apply_press_conference_response(session: Any, press_id: str, question_id: str, response_id: str) -> Dict[str, Any]:
    """Resolve a press conference answer; returns headline for follow-up coverage."""
    migrate_session_storyline_state(session)
    pid = str(press_id or "")
    qid = str(question_id or "")
    rid = str(response_id or "")
    queue = list(getattr(session, "press_conference_queue", None) or [])
    entry = next((p for p in queue if str(p.get("id") or "") == pid), None)
    if entry is None:
        raise ValueError(f"Press conference not found: {pid}")
    question = next((q for q in (entry.get("questions") or []) if str(q.get("id") or "") == qid), None)
    if question is None:
        raise ValueError(f"Question not found: {qid}")
    response = next((r for r in (question.get("responses") or []) if str(r.get("id") or "") == rid), None)
    if response is None:
        raise ValueError(f"Response not found: {rid}")
    entry["status"] = "answered"
    entry["answered"] = {"question_id": qid, "response_id": rid, "tone": response.get("tone")}
    session.press_conference_queue = queue
    tone = str(response.get("tone") or "neutral")
    headline_map = {
        "deflect": "GM deflects questions amid growing media pressure",
        "support_staff": "GM strongly backs coaching staff",
        "support_player": f"GM publicly backs {entry.get('player_name') or 'player'}",
        "no_comment": "GM declines comment — story gains traction",
        "deny": "GM denies active trade talks",
        "neither_confirm": "GM refuses to deny trade speculation",
        "listen": "GM acknowledges trade interest",
    }
    headline = headline_map.get(rid, headline_map.get(tone, "GM addresses media"))
    from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433

    _record_storyline(
        session,
        {
            "headline": headline,
            "summary": str(response.get("description") or ""),
            "team_id": getattr(session, "user_team_id", ""),
            "player_id": entry.get("player_id"),
            "player_name": entry.get("player_name"),
            "storyline_id": entry.get("storyline_id"),
            "category": "decision",
            "type": "press_conference",
            "cause_type": "GM_PRESS_RESPONSE",
            "priority": "HIGH",
            "heat": max(40, int(entry.get("heat") or 0) - 5),
            "calendar_iso": entry.get("calendar_iso"),
        },
    )
    rel = dict(getattr(session, "agent_relationships", None) or {})
    plid = str(entry.get("player_id") or "")
    if plid and plid in rel:
        if tone in ("support_player", "diplomatic"):
            rel[plid]["gm_trust"] = min(1.0, float(rel[plid].get("gm_trust") or 0.5) + 0.08)
        elif tone in ("cold", "firm") and rid == "no_comment":
            rel[plid]["gm_trust"] = max(0.0, float(rel[plid].get("gm_trust") or 0.5) - 0.05)
        session.agent_relationships = rel
    return {"headline": headline, "press_id": pid, "tone": tone}


def _archive_storyline_beat(session: Any, sl: Dict[str, Any]) -> None:
    archive = list(getattr(session, "narrative_archive", None) or [])
    season = int(getattr(session, "season_calendar_year", 2025) or 2025)
    archive.append(
        {
            "season": season,
            "season_label": f"{season}-{season + 1}",
            "headline": sl.get("headline"),
            "summary": sl.get("summary") or sl.get("short_summary"),
            "arc_id": sl.get("arc_id"),
            "storyline_id": sl.get("storyline_id"),
            "player_id": sl.get("player_id"),
            "player_name": sl.get("player_name"),
            "team_id": sl.get("team_id"),
            "heat": sl.get("heat"),
            "category": sl.get("category"),
            "calendar_iso": sl.get("calendar_iso"),
            "reporter_name": sl.get("reporter_name"),
        }
    )
    session.narrative_archive = archive[-600:]


def build_narrative_eras(session: Any) -> List[Dict[str, Any]]:
    archive = list(getattr(session, "narrative_archive", None) or [])
    by_season: Dict[int, List[Dict[str, Any]]] = {}
    for item in archive:
        s = int(item.get("season") or 0)
        if s <= 0:
            continue
        by_season.setdefault(s, []).append(item)
    eras: List[Dict[str, Any]] = []
    for season in sorted(by_season.keys()):
        items = by_season[season]
        top = sorted(items, key=lambda x: int(x.get("heat") or 0), reverse=True)[:8]
        themes: List[str] = []
        cats = [str(x.get("category") or "") for x in items]
        if any("trade" in c for c in cats):
            themes.append("Trade deadline chaos")
        if any("injury" in c for c in cats):
            themes.append("Injury cloud")
        if any("draft" in c for c in cats):
            themes.append("Draft obsession")
        if any("legal" in c or "conduct" in c for c in cats):
            themes.append("Off-ice storm")
        eras.append(
            {
                "season": season,
                "label": f"{season}-{season + 1} Era",
                "story_count": len(items),
                "top_stories": top,
                "themes": themes[:4] or ["Season coverage"],
            }
        )
    session.narrative_eras = eras[-20:]
    return list(session.narrative_eras)


def seal_narrative_season(session: Any, season: Optional[int] = None) -> None:
    """Snapshot a completed season into the historical archive."""
    migrate_session_storyline_state(session)
    sy = int(season or getattr(session, "season_calendar_year", 2025) or 2025)
    sealed = list(getattr(session, "_narrative_sealed_seasons", None) or [])
    if sy in sealed:
        return
    build_narrative_eras(session)
    sealed.append(sy)
    session._narrative_sealed_seasons = sealed[-30:]


def _ensure_prospect_social_profile(session: Any, prospect_key: str, prospect_name: str, rng: random.Random) -> Dict[str, Any]:
    profiles = dict(getattr(session, "prospect_social_profiles", None) or {})
    if prospect_key in profiles:
        return profiles[prospect_key]
    handle_base = prospect_name.replace(" ", "").lower()[:12] or "prospect"
    personality = rng.choice(["polished", "awkward", "extremely_online", "private", "hype_beast"])
    profile = {
        "prospect_key": prospect_key,
        "prospect_name": prospect_name,
        "handle": f"@{handle_base}{rng.randint(10, 99)}",
        "personality": personality,
        "followers": rng.randint(1200, 85000) if personality != "private" else rng.randint(200, 3000),
        "following": rng.randint(80, 400),
        "bio": rng.choice([
            "Chasing the dream.",
            "Junior hockey · Draft eligible",
            "God first · Hockey second",
            "OHL · 🇨🇦",
            "",
        ]),
        "posts": [],
    }
    profiles[prospect_key] = profile
    session.prospect_social_profiles = profiles
    return profile


def generate_prospect_social_posts(session: Any, rng: Optional[random.Random] = None) -> int:
    """Draft-season prospect Twitter — first-class social entities."""
    r = rng or random.Random()
    phase = str(getattr(session, "phase", "") or "").lower()
    month = 0
    cal = getattr(session, "nhl_calendar", None) or []
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    if 0 <= cur < len(cal):
        iso = str(cal[cur].get("iso") or "")
        if len(iso) >= 7:
            try:
                month = int(iso[5:7])
            except ValueError:
                month = 0
    if phase not in ("regular", "offseason", "preseason") and month not in (5, 6, 7, 8, 9, 10, 11):
        return 0
    ranks = dict(getattr(session, "draft_rank_prev", None) or {})
    if not ranks:
        return 0
    created = 0
    posts = list(getattr(session, "social_posts", None) or [])
    for key in list(ranks.keys())[:40]:
        if r.random() > 0.12:
            continue
        name = str(key).replace("_", " ").title()
        prof = _ensure_prospect_social_profile(session, str(key), name, r)
        templates = [
            f"Proud of the guys tonight. Still work to do.",
            f"Grateful for another opportunity to play in front of scouts.",
            f"One shift at a time.",
            f"Junior teammates know what's up 👀",
        ]
        if prof.get("personality") == "hype_beast":
            templates.append("Draft day can't come soon enough.")
        if prof.get("personality") == "extremely_online":
            templates.append("Why follow 20 NHL teams before the draft? Because hockey Twitter is undefeated.")
        text = r.choice(templates)
        posts.append(
            {
                "id": f"soc_prosp_{uuid.uuid4().hex[:10]}",
                "author_type": "prospect",
                "author_id": str(key),
                "author_name": name,
                "handle": prof.get("handle"),
                "verified": False,
                "text": text,
                "related_headline": f"Prospect watch: {name}",
                "calendar_iso": cal[cur].get("iso") if 0 <= cur < len(cal) else "",
                "heat": r.randint(8, 35),
                "knowledge_type": "social",
                "prospect_key": str(key),
                "likes": r.randint(200, 12000),
                "reposts": r.randint(20, 800),
                "replies": r.randint(10, 400),
            }
        )
        prof_posts = list(prof.get("posts") or [])
        prof_posts.append({"text": text, "iso": cal[cur].get("iso") if 0 <= cur < len(cal) else ""})
        prof["posts"] = prof_posts[-20:]
        created += 1
    session.social_posts = posts[-200:]
    return created


def _breaking_news_signal(sl: Dict[str, Any]) -> Optional[str]:
    priority = str(sl.get("priority") or "").upper()
    heat = int(sl.get("heat") or 0)
    if priority == "CRITICAL" or heat >= 85:
        return "league_defining"
    if priority == "HIGH" or heat >= 72:
        return "breaking"
    if heat >= 55:
        return "developing"
    return None


def narrative_universe_daily_pass(
    session: Any,
    calendar_idx: int,
    day_meta: Dict[str, Any],
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Daily tick: prospect social, season seal check, refresh eras."""
    migrate_session_storyline_state(session)
    r = rng or random.Random()
    prospect_posts = generate_prospect_social_posts(session, r)
    phase = str(getattr(session, "phase", "") or "").lower()
    if phase == "offseason":
        seal_narrative_season(session)
    eras = build_narrative_eras(session)
    pending_press = len([p for p in (getattr(session, "press_conference_queue", None) or []) if str(p.get("status") or "") == "pending"])
    return {
        "prospect_social_created": prospect_posts,
        "narrative_eras": len(eras),
        "pending_press_conferences": pending_press,
    }


# ===========================================================================
# STORYLINE UNIVERSE V2
# Paste this complete section at the bottom of the existing storyline module.
#
# It is deliberately dictionary/session based so old franchise saves migrate
# without a schema migration and so every public payload remains JSON-safe.
# Existing entry points are wrapped at the bottom of this section.
# ===========================================================================

UNIVERSE_ENGINE_VERSION = 2
UNIVERSE_MAX_MEMORIES = 40
UNIVERSE_MAX_EVENTS = 300
UNIVERSE_MAX_INTERACTIONS = 120
UNIVERSE_MAX_SOCIAL_POSTS = 500
UNIVERSE_PERMANENT_ATTRIBUTE_CAP = 3.0

_UNIVERSE_LEGACY_MIGRATE = migrate_session_storyline_state
_UNIVERSE_LEGACY_DAILY_PASS = narrative_universe_daily_pass
_UNIVERSE_LEGACY_PAYLOAD = build_narrative_universe_payload
_UNIVERSE_LEGACY_PLAYER_PROFILE = player_narrative_profile
_UNIVERSE_LEGACY_ENRICH = enrich_storyline_for_narrative_universe

_UNIVERSE_CAUSE_TYPES = frozenset(
    {
        "PLAYER_INTERACTION",
        "PLAYER_REPORTER_CONFRONTATION",
        "PLAYER_REPORTER_ALTERCATION",
        "TEAMMATE_CONFLICT",
        "TEAMMATE_FIGHT",
        "LOCKER_ROOM_LEADERSHIP",
        "HIGH_CHARACTER_IMPACT",
        "LOW_CHARACTER_GAME_IMPACT",
        "PERSONAL_CONCERN",
        "PROMISE_KEPT",
        "PROMISE_BROKEN",
        "COMMUNITY_MOMENT",
        "HAT_TRICK",
        "SHUTOUT",
        "OT_WINNER",
        "FIRST_GOAL",
        "ON_ICE_ALTERCATION",
        "ROLLING_HOT",
        "ROLLING_COLD",
        "PLAYER_ROLE_FRUSTRATION",
        "CONTRACT_YEAR_HEAT",
        "ROOM_BELONGING",
        "WINNING_CONCERN",
        "LOCKER_ROOM_PULSE",
        "COACH_HOT_SEAT",
        "GM_JOB_SECURITY",
        "AHL_CALLUP",
        "AHL_SENDDOWN",
        "CAPTAINCY_PULSE",
    }
)
STORYLINE_CAUSE_TYPES = frozenset(set(STORYLINE_CAUSE_TYPES) | set(_UNIVERSE_CAUSE_TYPES))


NICHE_ABILITY_CATALOG: Dict[str, Dict[str, Any]] = {
    "glue_guy": {
        "label": "Glue Guy",
        "description": "Keeps teammates connected and limits morale spirals.",
        "room": {"unity": 3.0, "belonging": 4.0, "tension": -2.0},
        "game": {"team_composure": 0.20},
    },
    "mentor": {
        "label": "Mentor",
        "description": "Accelerates young teammates' habits and recovery from mistakes.",
        "room": {"development": 4.0, "accountability": 2.0},
        "game": {"rookie_composure": 0.35},
    },
    "peacemaker": {
        "label": "Peacemaker",
        "description": "Can defuse arguments before they split the room.",
        "room": {"tension": -4.0, "unity": 2.0},
        "game": {"discipline": 0.15},
    },
    "accountability_driver": {
        "label": "Accountability Driver",
        "description": "Raises practice standards even without star-level talent.",
        "room": {"accountability": 4.0, "work_ethic": 3.0},
        "game": {"effort": 0.25},
    },
    "culture_carrier": {
        "label": "Culture Carrier",
        "description": "Protects team identity during slumps and roster turnover.",
        "room": {"identity": 4.0, "unity": 2.0},
        "game": {"team_resilience": 0.25},
    },
    "media_shield": {
        "label": "Media Shield",
        "description": "Takes hard questions and keeps attention off vulnerable teammates.",
        "room": {"media_stress": -4.0, "leadership": 2.0},
        "game": {"pressure_resistance": 0.20},
    },
    "spark_plug": {
        "label": "Spark Plug",
        "description": "Can lift the bench with energy, contact, or a momentum shift.",
        "room": {"energy": 3.0},
        "game": {"momentum": 0.35},
    },
    "clutch_composure": {
        "label": "Clutch Composure",
        "description": "Stays calm in one-goal games and late high-pressure moments.",
        "room": {"confidence": 1.0},
        "game": {"late_game": 0.45, "composure": 0.25},
    },
    "penalty_kill_voice": {
        "label": "Penalty-Kill Voice",
        "description": "Organizes teammates and improves communication while shorthanded.",
        "room": {"accountability": 1.0},
        "game": {"penalty_kill": 0.40, "defensive_awareness": 0.20},
    },
    "faceoff_specialist": {
        "label": "Faceoff Specialist",
        "description": "Creates narrow situational value on important draws.",
        "room": {},
        "game": {"faceoffs": 0.55},
    },
    "power_play_quarterback": {
        "label": "Power-Play Quarterback",
        "description": "Reads pressure and organizes the top of a power play.",
        "room": {},
        "game": {"power_play": 0.45, "passing": 0.20},
    },
    "net_front_menace": {
        "label": "Net-Front Menace",
        "description": "Creates screens, rebounds, and frustration around the crease.",
        "room": {"energy": 1.0},
        "game": {"net_front": 0.50},
    },
    "agitator": {
        "label": "Agitator",
        "description": "Can draw opponents off their game, but may cross the line.",
        "room": {"tension": 1.0},
        "game": {"opponent_composure": -0.30, "penalty_risk": 0.20},
    },
    "streak_rider": {
        "label": "Streak Rider",
        "description": "Confidence strongly amplifies both hot and cold runs.",
        "room": {},
        "game": {"confidence_variance": 0.45},
    },
    "quiet_professional": {
        "label": "Quiet Professional",
        "description": "Provides a dependable example without needing status or attention.",
        "room": {"work_ethic": 2.0, "stability": 2.0},
        "game": {"effort_floor": 0.25},
    },
    "prankster": {
        "label": "Prankster",
        "description": "Keeps the room loose, provided the joke lands well.",
        "room": {"belonging": 2.0, "tension": -1.0},
        "game": {},
    },
    "volatile_competitor": {
        "label": "Volatile Competitor",
        "description": "Competes fiercely but can turn frustration against teammates or media.",
        "room": {"tension": 3.0},
        "game": {"effort": 0.20, "penalty_risk": 0.35},
    },
    "self_first": {
        "label": "Self-First Mentality",
        "description": "Personal status can override structure, trust, and team needs.",
        "room": {"unity": -3.0, "tension": 3.0},
        "game": {"passing": -0.20, "defensive_effort": -0.25},
    },
}

_UNIVERSE_ATTRIBUTE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "offensive_awareness": ("offensive_awareness", "off_awareness", "offense_awareness", "oaw"),
    "defensive_awareness": ("defensive_awareness", "def_awareness", "defense_awareness", "daw"),
    "passing": ("passing", "pass_accuracy", "playmaking"),
    "puck_control": ("puck_control", "puckcontrol", "hands"),
    "shot_accuracy": ("shot_accuracy", "wrist_accuracy", "shooting_accuracy"),
    "faceoffs": ("faceoffs", "faceoff", "draws"),
    "discipline": ("discipline", "poise"),
    "checking": ("checking", "body_checking", "physicality"),
    "shot_blocking": ("shot_blocking", "blocking", "shot_blocks"),
    "stick_checking": ("stick_checking", "stickchecking", "defensive_stick"),
    "speed": ("speed", "skating_speed"),
    "stamina": ("stamina", "endurance"),
    "agility": ("agility",),
    "rebound_control": ("rebound_control", "reboundcontrol"),
    "positioning": ("positioning", "goalie_positioning"),
}


def _u_seed(*parts: Any) -> int:
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:16], 16)


def _u_clip(value: Any, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = lo
    return round(lo if number < lo else hi if number > hi else number, 2)


def _u_get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _u_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    return str(getattr(player, "name", "") or _u_get(ident, "name", "") or "Player")


def _u_position(player: Any) -> str:
    ident = getattr(player, "identity", None)
    return str(getattr(player, "position", "") or _u_get(ident, "position", "") or "F").upper()


def _u_team_players(session: Any, team_id: str) -> List[Any]:
    team = (getattr(session, "team_by_id", None) or {}).get(str(team_id or ""))
    return list(getattr(team, "roster", None) or []) if team is not None else []


def _u_all_players(session: Any) -> List[Tuple[str, Any]]:
    rows: List[Tuple[str, Any]] = []
    seen = set()
    for team_id, team in (getattr(session, "team_by_id", None) or {}).items():
        for bucket in ("roster", "ahl_roster"):
            for player in getattr(team, bucket, None) or []:
                pid = str(getattr(player, "id", "") or "")
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                rows.append((str(team_id), player))
    return rows


def _u_current_meta(session: Any, calendar_idx: Optional[int] = None) -> Tuple[int, str, int]:
    day = int(calendar_idx if calendar_idx is not None else getattr(session, "calendar_cursor", 0) or 0)
    iso = ""
    calendar = getattr(session, "nhl_calendar", None) or []
    if 0 <= day < len(calendar):
        iso = str((calendar[day] or {}).get("iso") or "")
    season = int(getattr(session, "season_calendar_year", 2025) or 2025)
    return day, iso, season


def _u_read_numeric(player: Any, names: Tuple[str, ...], default: float = 50.0) -> float:
    containers = [player, getattr(player, "attributes", None), getattr(player, "ratings", None), getattr(player, "skills", None), getattr(player, "traits", None)]
    for container in containers:
        for name in names:
            value = _u_get(container, name, None)
            if value is None or value == "":
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            return number * 100.0 if -0.01 <= number <= 1.5 else number
    return float(default)


def _u_character(player: Any) -> float:
    try:
        return float(_player_character_0_100(player))
    except Exception:
        return _u_read_numeric(player, ("character", "professionalism", "sportsmanship"), 55.0)


def _u_psych_value(player: Any, names: Tuple[str, ...], default: float) -> float:
    psych = getattr(player, "psych", None)
    for name in names:
        value = _u_get(psych, name, None)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        return number * 100.0 if number <= 1.5 else number
    return default


def _u_contract_years(player: Any, season: int) -> int:
    contract = getattr(player, "contract", None)
    for name in ("years_remaining", "term_remaining", "remaining_years"):
        try:
            value = int(_u_get(contract, name, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    try:
        expiry = int(_u_get(contract, "expiry_year", 0) or _u_get(contract, "expires", 0) or 0)
    except (TypeError, ValueError):
        expiry = 0
    return max(0, expiry - season) if expiry else 2


def _u_social_handle(name: str, player_id: str) -> str:
    base = "".join(ch for ch in name if ch.isalnum())[:15] or "Player"
    return f"@{base}{_u_seed(player_id, 'handle') % 90 + 10}"


def _u_trait_100(player: Any, *names: str, default: float = 50.0) -> float:
    """Read PersonalityTraits (0-1) or PsychologyState fields onto a 0-100 axis."""
    traits = getattr(player, "traits", None)
    psych = getattr(player, "psych", None)
    for name in names:
        value = None
        if traits is not None:
            value = getattr(traits, name, None)
        if value is None and psych is not None:
            value = getattr(psych, name, None)
        if value is None:
            value = _u_get(player, name, None)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        return number * 100.0 if number <= 1.5 else number
    return default


def _u_personality(player: Any, player_id: str) -> Dict[str, float]:
    """Wire Player.traits / Player.psych directly. Do not invent a second personality from a hash."""
    introversion = _u_trait_100(player, "introversion", default=50.0)
    loyalty = _u_trait_100(player, "loyalty", default=55.0)
    work_ethic = _u_trait_100(player, "work_ethic", "patience", default=55.0)
    character = _u_clip(work_ethic * 0.45 + loyalty * 0.55)
    if getattr(player, "traits", None) is None and getattr(player, "psych", None) is None:
        character = _u_clip(_u_character(player))
    return {
        "character": character,
        "professionalism": _u_clip(work_ethic),
        "empathy": _u_clip(_u_trait_100(player, "family_priority", default=50.0) * 0.6 + (100.0 - _u_trait_100(player, "ego", default=50.0)) * 0.4),
        "accountability": _u_clip(_u_trait_100(player, "work_ethic", "coachability", default=55.0)),
        "loyalty": _u_clip(loyalty),
        "ego": _u_clip(_u_trait_100(player, "ego", default=50.0)),
        "volatility": _u_clip(_u_trait_100(player, "volatility", default=50.0)),
        "competitiveness": _u_clip(_u_trait_100(player, "competitiveness", default=55.0)),
        "resilience": _u_clip(_u_trait_100(player, "mental_toughness", "patience", default=55.0)),
        "leadership": _u_clip(_u_trait_100(player, "leadership", default=50.0)),
        "sociability": _u_clip(100.0 - introversion),
        "media_savvy": _u_clip(_u_trait_100(player, "media_comfort", default=50.0)),
        "ambition": _u_clip(_u_trait_100(player, "ambition", default=55.0)),
        "family_orientation": _u_clip(_u_trait_100(player, "family_priority", default=50.0)),
        "coachability": _u_clip(_u_trait_100(player, "coachability", default=55.0)),
        "money_focus": _u_clip(_u_trait_100(player, "money_focus", default=45.0)),
        "clutch": _u_clip(_u_trait_100(player, "clutch_tendency", default=50.0)),
    }


def _u_pick_niches(player: Any, personality: Dict[str, float], age: int, position: str) -> List[Dict[str, Any]]:
    ovr = _player_ovr99(player)
    candidates: List[Tuple[float, str]] = []
    candidates.append((personality["empathy"] + personality["sociability"] * 0.35, "glue_guy"))
    candidates.append((personality["professionalism"] + max(0, age - 27) * 3.0, "mentor"))
    candidates.append((personality["empathy"] + (100.0 - personality["volatility"]) * 0.45, "peacemaker"))
    candidates.append((personality["accountability"] + personality["competitiveness"] * 0.35, "accountability_driver"))
    candidates.append((personality["loyalty"] + personality["professionalism"] * 0.35, "culture_carrier"))
    candidates.append((personality["media_savvy"] + personality["leadership"] * 0.30, "media_shield"))
    candidates.append((personality["competitiveness"] + personality["sociability"] * 0.25, "spark_plug"))
    candidates.append((personality["resilience"] + personality["competitiveness"] * 0.35, "clutch_composure"))
    candidates.append((_u_read_numeric(player, ("defensive_awareness", "shot_blocking"), 50.0) + personality["leadership"] * 0.25, "penalty_kill_voice"))
    candidates.append((_u_read_numeric(player, ("faceoffs", "faceoff"), 45.0) + (8 if position == "C" else 0), "faceoff_specialist"))
    candidates.append((_u_read_numeric(player, ("passing", "offensive_awareness"), 50.0) + (8 if _pos_bucket(position) == "D" else 0), "power_play_quarterback"))
    candidates.append((_u_read_numeric(player, ("strength", "hand_eye", "checking"), 50.0), "net_front_menace"))
    candidates.append((personality["competitiveness"] + personality["volatility"] * 0.45, "agitator"))
    candidates.append((personality["ego"] + personality["resilience"] * 0.25, "streak_rider"))
    candidates.append((personality["professionalism"] + (100.0 - personality["ego"]) * 0.35, "quiet_professional"))
    candidates.append((personality["sociability"] + (100.0 - personality["volatility"]) * 0.20, "prankster"))
    if personality["volatility"] >= 62 and personality["competitiveness"] >= 62:
        candidates.append((personality["volatility"] + personality["competitiveness"] * 0.5, "volatile_competitor"))
    if personality["character"] <= 45 and personality["ego"] >= 62:
        candidates.append((personality["ego"] + (100.0 - personality["character"]), "self_first"))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen: List[Dict[str, Any]] = []
    for score, ability_id in candidates:
        if ability_id in {row["id"] for row in chosen}:
            continue
        if score < 74.0 and len(chosen) >= 2:
            continue
        tier = 3 if score >= 125 else 2 if score >= 98 else 1
        meta = NICHE_ABILITY_CATALOG[ability_id]
        chosen.append(
            {
                "id": ability_id,
                "label": meta["label"],
                "tier": tier,
                "description": meta["description"],
                "room_effects": dict(meta.get("room") or {}),
                "game_effects": dict(meta.get("game") or {}),
            }
        )
        if len(chosen) >= (5 if ovr < 79 and personality["character"] >= 72 else 4):
            break
    return chosen


def _u_initial_concerns(player: Any, personality: Dict[str, float], age: int, season: int) -> Dict[str, Dict[str, Any]]:
    years = _u_contract_years(player, season)
    ovr = _player_ovr99(player)
    return {
        "role": {"label": "Role and ice time", "importance": _u_clip(45 + personality["ambition"] * 0.38), "satisfaction": 58.0, "trend": 0.0},
        "respect": {"label": "Respect from management", "importance": _u_clip(35 + personality["ego"] * 0.42), "satisfaction": 62.0, "trend": 0.0},
        "winning": {"label": "Competing for a winner", "importance": _u_clip(38 + personality["competitiveness"] * 0.48), "satisfaction": 55.0, "trend": 0.0},
        "contract": {"label": "Contract security", "importance": _u_clip(78 - years * 12 + personality["ambition"] * 0.18), "satisfaction": _u_clip(35 + years * 18), "trend": 0.0},
        "development": {"label": "Personal development", "importance": _u_clip(92 - max(0, age - 20) * 5), "satisfaction": 60.0 if ovr >= 76 else 48.0, "trend": 0.0},
        "home_life": {"label": "Home-life stability", "importance": _u_clip(35 + personality["family_orientation"] * 0.52), "satisfaction": 68.0, "trend": 0.0},
        "team_belonging": {"label": "Belonging in the room", "importance": _u_clip(40 + personality["sociability"] * 0.42), "satisfaction": 60.0, "trend": 0.0},
    }


def _u_create_player_entity(session: Any, team_id: str, player: Any) -> Dict[str, Any]:
    player_id = str(getattr(player, "id", "") or "")
    day, iso, season = _u_current_meta(session)
    age = _player_age(player)
    position = _u_position(player)
    personality = _u_personality(player, player_id)
    rng = random.Random(_u_seed("life", player_id))
    relationship_status = rng.choices(["single", "partnered", "family_household"], weights=[35, 38, 27])[0]
    dependents = 0 if relationship_status != "family_household" else rng.choice([1, 1, 2, 2, 3])
    state = {
        "morale": _u_psych_value(player, ("morale",), 58.0),
        "confidence": _u_psych_value(player, ("confidence_level", "confidence"), 56.0),
        "coach_trust": _u_psych_value(player, ("coach_trust",), 60.0),
        "gm_trust": 65.0,
        "belonging": 60.0,
        "energy": 72.0,
        "focus": 66.0,
        "media_stress": _u_psych_value(player, ("media_stress",), 28.0),
        "personal_stress": 28.0 + rng.uniform(-8, 12),
        "role_satisfaction": 58.0,
    }
    return {
        "player_id": player_id,
        "player_name": _u_name(player),
        "team_id": str(team_id),
        "position": position,
        "age": age,
        "overall": round(_player_ovr99(player), 1),
        "created_season": season,
        "created_day": day,
        "last_tick_day": day,
        "personality": personality,
        "identity": {
            "name": _u_name(player),
            "age": age,
            "birth_city": str(getattr(getattr(player, "identity", None), "birth_city", "") or ""),
            "birth_country": str(getattr(getattr(player, "identity", None), "birth_country", "") or ""),
            "draft_year": int(getattr(getattr(player, "identity", None), "draft_year", 0) or 0),
            "draft_round": int(getattr(getattr(player, "identity", None), "draft_round", 0) or 0),
            "draft_pick": int(getattr(getattr(player, "identity", None), "draft_pick", 0) or 0),
            "position": position,
            "overall": round(_player_ovr99(player), 1),
        },
        "trusts": {
            "coach": round(_u_psych_value(player, ("coach_trust", "coach_relationship"), 55.0), 1),
            "gm": round(_u_psych_value(player, ("trust_in_management",), 55.0), 1),
            "teammates": round(_u_psych_value(player, ("trust_in_teammates",), 55.0), 1),
            "room": round(_u_psych_value(player, ("locker_room_fit",), 55.0), 1),
        },
        "personality_tags": [],
        "niche_abilities": _u_pick_niches(player, personality, age, position),
        "concerns": _u_initial_concerns(player, personality, age, season),
        "state": state,
        "life": {
            "relationship_status": relationship_status,
            "dependents": dependents,
            "home_stability": _u_clip(65 + rng.uniform(-12, 20)),
            "relocation_strain": _u_clip(rng.uniform(4, 32)),
            "community_connection": _u_clip(rng.uniform(18, 72)),
            "financial_stress": _u_clip(rng.uniform(2, 28)),
            "sleep_quality": _u_clip(rng.uniform(55, 86)),
            "privacy_preference": rng.choice(["private", "balanced", "public"]),
            "current_note": "Settling into the season.",
        },
        "social": {
            "handle": _u_social_handle(_u_name(player), player_id),
            "style": rng.choice(["quiet", "polished", "team_first", "playful", "online"]),
            "followers": int(max(700, _player_ovr99(player) ** 3 * rng.uniform(0.7, 3.5))),
            "fan_sentiment": 55.0,
            "last_post_day": -99,
        },
        "room_role": "member",
        "room_value": 50.0,
        "disruption_risk": 10.0,
        "reputation_tags": [],
        "memories": [],
        "attribute_ledger": [],
        "season_permanent_attribute_delta": {},
        "private": True,
        "last_updated_iso": iso,
    }


def _u_personality_tags(entity: Dict[str, Any]) -> List[str]:
    p = entity.get("personality") or {}
    tags: List[str] = []
    tests = [
        ("Team-first", p.get("character", 0) >= 72 and p.get("ego", 100) <= 52),
        ("Demanding", p.get("ambition", 0) >= 72),
        ("Emotionally volatile", p.get("volatility", 0) >= 72),
        ("Steady presence", p.get("resilience", 0) >= 72 and p.get("professionalism", 0) >= 68),
        ("Natural leader", p.get("leadership", 0) >= 72),
        ("Media savvy", p.get("media_savvy", 0) >= 72),
        ("Private personality", p.get("sociability", 100) <= 34),
        ("Highly coachable", p.get("coachability", 0) >= 74),
        ("Self-interested", p.get("character", 100) <= 42 and p.get("ego", 0) >= 65),
    ]
    for label, applies in tests:
        if applies:
            tags.append(label)
    return tags[:5] or ["Balanced personality"]


def _u_migrate_v2(session: Any) -> None:
    defaults = {
        "universe_engine_version": UNIVERSE_ENGINE_VERSION,
        "universe_players": {},
        "universe_locker_rooms": {},
        "universe_interactions": [],
        "universe_interaction_queue": [],
        "universe_event_log": [],
        "universe_reporter_relationships": {},
        "universe_attribute_modifiers": {},
        "universe_promises": [],
        "universe_daily_snapshots": [],
        "universe_game_contexts": {},
        "_universe_last_daily_tick": -1,
    }
    for key, default in defaults.items():
        if getattr(session, key, None) is None:
            setattr(session, key, default.copy() if isinstance(default, dict) else list(default) if isinstance(default, list) else default)
    session.universe_engine_version = UNIVERSE_ENGINE_VERSION


def _u_sync_player_entities(session: Any) -> Dict[str, Dict[str, Any]]:
    _u_migrate_v2(session)
    entities = dict(getattr(session, "universe_players", None) or {})
    active_ids: List[str] = []
    for team_id, player in _u_all_players(session):
        player_id = str(getattr(player, "id", "") or "")
        active_ids.append(player_id)
        entity = dict(entities.get(player_id) or {})
        if not entity:
            entity = _u_create_player_entity(session, team_id, player)
        entity["player_name"] = _u_name(player)
        entity["team_id"] = str(team_id)
        entity["position"] = _u_position(player)
        entity["age"] = _player_age(player)
        entity["overall"] = round(_player_ovr99(player), 1)
        entity["personality"] = _u_personality(player, player_id)
        ident = getattr(player, "identity", None)
        entity["identity"] = {
            "name": _u_name(player),
            "age": entity["age"],
            "birth_city": str(getattr(ident, "birth_city", "") or ""),
            "birth_country": str(getattr(ident, "birth_country", "") or ""),
            "draft_year": int(getattr(ident, "draft_year", 0) or 0),
            "draft_round": int(getattr(ident, "draft_round", 0) or 0),
            "draft_pick": int(getattr(ident, "draft_pick", 0) or 0),
            "position": entity["position"],
            "overall": entity["overall"],
        }
        entity["trusts"] = {
            "coach": round(_u_psych_value(player, ("coach_trust", "coach_relationship"), 55.0), 1),
            "gm": round(_u_psych_value(player, ("trust_in_management",), 55.0), 1),
            "teammates": round(_u_psych_value(player, ("trust_in_teammates",), 55.0), 1),
            "room": round(_u_psych_value(player, ("locker_room_fit",), 55.0), 1),
        }
        entity["personality_tags"] = _u_personality_tags(entity)
        entity.setdefault("memories", [])
        entity.setdefault("attribute_ledger", [])
        entity.setdefault("reputation_tags", [])
        entity.setdefault("season_permanent_attribute_delta", {})
        entity.setdefault("concerns", {})
        entity.setdefault("state", {})
        entity.setdefault("life", {})
        entity.setdefault("social", {})
        entities[player_id] = entity
    for player_id, entity in entities.items():
        entity["active_roster"] = player_id in active_ids
    session.universe_players = entities
    return entities


def _u_pair_key(a: str, b: str) -> str:
    return "|".join(sorted((str(a), str(b))))


def _u_relationship(session: Any, team_id: str, a: str, b: str) -> Dict[str, Any]:
    rooms = getattr(session, "universe_locker_rooms", None) or {}
    room = rooms.setdefault(str(team_id), {})
    rels = room.setdefault("relationships", {})
    key = _u_pair_key(a, b)
    if key not in rels:
        entities = getattr(session, "universe_players", None) or {}
        pa = (entities.get(str(a)) or {}).get("personality") or {}
        pb = (entities.get(str(b)) or {}).get("personality") or {}
        rng = random.Random(_u_seed("relationship", key))
        compatibility = 54.0
        compatibility += (float(pa.get("empathy", 50)) + float(pb.get("empathy", 50)) - 100) * 0.10
        compatibility -= abs(float(pa.get("ego", 50)) - float(pb.get("ego", 50))) * 0.08
        compatibility -= (float(pa.get("volatility", 50)) + float(pb.get("volatility", 50)) - 100) * 0.06
        compatibility += rng.uniform(-14, 14)
        rels[key] = {
            "player_ids": [str(a), str(b)],
            "chemistry": _u_clip(compatibility),
            "trust": _u_clip(compatibility + rng.uniform(-8, 8)),
            "respect": _u_clip(compatibility + rng.uniform(-6, 12)),
            "tension": _u_clip(100 - compatibility + rng.uniform(-18, 4)),
            "familiarity": _u_clip(rng.uniform(18, 48)),
            "history": [],
        }
    rooms[str(team_id)] = room
    session.universe_locker_rooms = rooms
    return rels[key]


def _u_niche_ids(entity: Dict[str, Any]) -> List[str]:
    return [str(row.get("id") or "") for row in (entity.get("niche_abilities") or [])]


def _u_room_value(entity: Dict[str, Any]) -> Tuple[float, float]:
    p = entity.get("personality") or {}
    state = entity.get("state") or {}
    niches = _u_niche_ids(entity)
    positive = (
        float(p.get("character", 50)) * 0.22
        + float(p.get("professionalism", 50)) * 0.18
        + float(p.get("empathy", 50)) * 0.17
        + float(p.get("accountability", 50)) * 0.16
        + float(p.get("leadership", 50)) * 0.13
        + float(state.get("belonging", 50)) * 0.08
    )
    positive += sum(4.5 for n in niches if n in ("glue_guy", "mentor", "peacemaker", "culture_carrier", "accountability_driver", "quiet_professional"))
    disruption = max(0.0, 50.0 - float(p.get("character", 50))) * 0.72
    disruption += max(0.0, float(p.get("ego", 50)) - 62.0) * 0.35
    disruption += max(0.0, float(p.get("volatility", 50)) - 62.0) * 0.36
    disruption += max(0.0, 42.0 - float(state.get("role_satisfaction", 55))) * 0.35
    if "self_first" in niches:
        disruption += 12
    return _u_clip(positive), _u_clip(disruption)


def _u_compute_factions(player_ids: List[str], relationships: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    unseen = set(player_ids)
    factions: List[Dict[str, Any]] = []
    while unseen:
        seed = sorted(unseen)[0]
        stack = [seed]
        members: List[str] = []
        unseen.remove(seed)
        while stack:
            current = stack.pop()
            members.append(current)
            for candidate in list(unseen):
                rel = relationships.get(_u_pair_key(current, candidate)) or {}
                if float(rel.get("chemistry", 0)) >= 67 and float(rel.get("trust", 0)) >= 60:
                    unseen.remove(candidate)
                    stack.append(candidate)
        if len(members) >= 2:
            factions.append({"id": f"faction_{_u_seed(*members) % 100000}", "player_ids": sorted(members), "strength": min(100, 48 + len(members) * 9)})
    return sorted(factions, key=lambda row: (-len(row["player_ids"]), row["id"]))[:6]


def _u_rebuild_locker_room(session: Any, team_id: str) -> Dict[str, Any]:
    entities = getattr(session, "universe_players", None) or {}
    players = _u_team_players(session, team_id)
    player_ids = [str(getattr(p, "id", "") or "") for p in players if str(getattr(p, "id", "") or "")]
    rooms = getattr(session, "universe_locker_rooms", None) or {}
    old_room = dict(rooms.get(str(team_id)) or {})
    relationships = dict(old_room.get("relationships") or {})
    for i, first in enumerate(player_ids):
        for second in player_ids[i + 1:]:
            _u_relationship(session, team_id, first, second)
    old_room = dict((getattr(session, "universe_locker_rooms", None) or {}).get(str(team_id)) or {})
    relationships = dict(old_room.get("relationships") or {})
    active_rels = [relationships.get(_u_pair_key(a, b)) or {} for i, a in enumerate(player_ids) for b in player_ids[i + 1:]]
    rows: List[Dict[str, Any]] = []
    for player_id in player_ids:
        entity = entities.get(player_id) or {}
        value, disruption = _u_room_value(entity)
        entity["room_value"] = value
        entity["disruption_risk"] = disruption
        rows.append(entity)
    leaders = sorted(rows, key=lambda row: (-float(row.get("room_value", 0)), -float((row.get("personality") or {}).get("leadership", 0))))
    for entity in rows:
        niches = _u_niche_ids(entity)
        if entity in leaders[:2] and float(entity.get("room_value", 0)) >= 67:
            entity["room_role"] = "leadership_group"
        elif "glue_guy" in niches or "peacemaker" in niches:
            entity["room_role"] = "connector"
        elif "mentor" in niches:
            entity["room_role"] = "mentor"
        elif float(entity.get("disruption_risk", 0)) >= 50:
            entity["room_role"] = "disruptor_risk"
        else:
            entity["room_role"] = "member"
    avg_chem = sum(float(row.get("chemistry", 50)) for row in active_rels) / max(1, len(active_rels))
    avg_tension = sum(float(row.get("tension", 30)) for row in active_rels) / max(1, len(active_rels))
    avg_morale = sum(float((row.get("state") or {}).get("morale", 55)) for row in rows) / max(1, len(rows))
    avg_conf = sum(float((row.get("state") or {}).get("confidence", 55)) for row in rows) / max(1, len(rows))
    avg_prof = sum(float((row.get("personality") or {}).get("professionalism", 55)) for row in rows) / max(1, len(rows))
    avg_account = sum(float((row.get("personality") or {}).get("accountability", 55)) for row in rows) / max(1, len(rows))
    positive_niches = sum(1 for row in rows for n in _u_niche_ids(row) if n in ("glue_guy", "peacemaker", "culture_carrier", "quiet_professional"))
    disruptors = [row for row in rows if float(row.get("disruption_risk", 0)) >= 48]
    high_character_depth = [row for row in rows if float(row.get("overall", 99)) < 80 and float(row.get("room_value", 0)) >= 70]
    unity = _u_clip(avg_chem * 0.48 + avg_morale * 0.28 + 12 + positive_niches * 1.1 - len(disruptors) * 1.7)
    tension = _u_clip(avg_tension * 0.55 + len(disruptors) * 7 - positive_niches * 1.4)
    room = {
        **old_room,
        "team_id": str(team_id),
        "culture": {
            "unity": unity,
            "tension": tension,
            "confidence": _u_clip(avg_conf),
            "accountability": _u_clip(avg_account + len([r for r in rows if "accountability_driver" in _u_niche_ids(r)]) * 2),
            "work_ethic": _u_clip(avg_prof),
            "leadership": _u_clip(sum(float((r.get("personality") or {}).get("leadership", 50)) for r in leaders[:5]) / max(1, min(5, len(leaders)))),
            "belonging": _u_clip(sum(float((r.get("state") or {}).get("belonging", 55)) for r in rows) / max(1, len(rows))),
        },
        "relationships": relationships,
        "leadership_group": [row.get("player_id") for row in leaders[:5]],
        "high_character_depth_players": [row.get("player_id") for row in high_character_depth],
        "disruptor_risks": [row.get("player_id") for row in disruptors],
        "factions": _u_compute_factions(player_ids, relationships),
        "last_updated_day": _u_current_meta(session)[0],
    }
    rooms[str(team_id)] = room
    session.universe_locker_rooms = rooms
    return room


def _u_add_memory(entity: Dict[str, Any], *, kind: str, summary: str, day: int, iso: str, emotional_delta: float = 0.0, related_ids: Optional[List[str]] = None, public: bool = False) -> None:
    memories = list(entity.get("memories") or [])
    memories.append(
        {
            "id": f"mem_{uuid.uuid4().hex[:10]}",
            "kind": kind,
            "summary": summary,
            "calendar_day": day,
            "calendar_iso": iso,
            "emotional_delta": float(emotional_delta),
            "related_player_ids": list(related_ids or []),
            "public": bool(public),
        }
    )
    entity["memories"] = memories[-UNIVERSE_MAX_MEMORIES:]


def _u_apply_profile_delta(entity: Dict[str, Any], dotted_key: str, delta: float) -> Dict[str, Any]:
    pieces = dotted_key.split(".")
    node = entity
    for piece in pieces[:-1]:
        child = node.get(piece)
        if not isinstance(child, dict):
            child = {}
            node[piece] = child
        node = child
    leaf = pieces[-1]
    before = float(node.get(leaf, 50.0) or 0.0)
    after = _u_clip(before + float(delta))
    node[leaf] = after
    return {"field": dotted_key, "before": before, "after": after, "delta": round(after - before, 2)}


def _u_write_actual_attribute(player: Any, attribute: str, delta: float) -> Tuple[Optional[float], Optional[float], str]:
    aliases = _UNIVERSE_ATTRIBUTE_ALIASES.get(attribute, (attribute,))
    containers: List[Tuple[str, Any]] = [
        ("attributes", getattr(player, "attributes", None)),
        ("ratings", getattr(player, "ratings", None)),
        ("skills", getattr(player, "skills", None)),
        ("player", player),
    ]
    for container_name, container in containers:
        if container is None:
            continue
        for alias in aliases:
            exists = alias in container if isinstance(container, dict) else hasattr(container, alias)
            if not exists:
                continue
            raw = _u_get(container, alias, None)
            try:
                before = float(raw)
            except (TypeError, ValueError):
                continue
            scale = 0.01 if -0.01 <= before <= 1.5 else 1.0
            lo, hi = (0.0, 1.0) if scale == 0.01 else (1.0, 99.0)
            after = max(lo, min(hi, before + float(delta) * scale))
            if isinstance(container, dict):
                container[alias] = after
            else:
                try:
                    setattr(container, alias, after)
                except Exception:
                    continue
            return before, after, f"{container_name}.{alias}"
    return None, None, "ledger_only"


def _u_apply_attribute_change(session: Any, player_id: str, change: Dict[str, Any], source_id: str) -> Dict[str, Any]:
    entities = getattr(session, "universe_players", None) or {}
    entity = entities.get(str(player_id)) or {}
    attribute = str(change.get("attribute") or "").strip()
    requested = float(change.get("delta") or 0.0)
    permanent = bool(change.get("permanent"))
    duration_games = int(change.get("duration_games") or 0)
    season = _u_current_meta(session)[2]
    if permanent:
        caps = entity.setdefault("season_permanent_attribute_delta", {})
        cap_key = f"{season}:{attribute}"
        used = float(caps.get(cap_key, 0.0) or 0.0)
        if requested > 0:
            actual_delta = min(requested, max(0.0, UNIVERSE_PERMANENT_ATTRIBUTE_CAP - used))
        else:
            actual_delta = max(requested, min(0.0, -UNIVERSE_PERMANENT_ATTRIBUTE_CAP - used))
        caps[cap_key] = round(used + actual_delta, 2)
    else:
        actual_delta = requested
    player = _player_from_roster(session, str(player_id))
    before: Optional[float] = None
    after: Optional[float] = None
    location = "ledger_only"
    if player is not None and permanent and attribute and actual_delta:
        before, after, location = _u_write_actual_attribute(player, attribute, actual_delta)
    receipt = {
        "id": f"attr_{uuid.uuid4().hex[:10]}",
        "source_id": source_id,
        "player_id": str(player_id),
        "attribute": attribute,
        "requested_delta": requested,
        "applied_delta": actual_delta,
        "permanent": permanent,
        "duration_games": duration_games,
        "before": before,
        "after": after,
        "location": location,
        "season": season,
    }
    ledger = list(entity.get("attribute_ledger") or [])
    ledger.append(receipt)
    entity["attribute_ledger"] = ledger[-60:]
    if not permanent and actual_delta and duration_games > 0:
        modifiers = dict(getattr(session, "universe_attribute_modifiers", None) or {})
        rows = list(modifiers.get(str(player_id)) or [])
        rows.append(
            {
                "id": receipt["id"],
                "source_id": source_id,
                "attribute": attribute,
                "delta": actual_delta,
                "games_remaining": duration_games,
                "reason": str(change.get("reason") or "Narrative interaction"),
            }
        )
        modifiers[str(player_id)] = rows[-16:]
        session.universe_attribute_modifiers = modifiers
    return receipt


def _u_change_relationship(session: Any, team_id: str, a: str, b: str, changes: Dict[str, Any], summary: str = "") -> List[Dict[str, Any]]:
    if not a or not b or a == b:
        return []
    rel = _u_relationship(session, team_id, a, b)
    receipts: List[Dict[str, Any]] = []
    for field, delta in changes.items():
        before = float(rel.get(field, 50.0) or 0.0)
        after = _u_clip(before + float(delta))
        rel[field] = after
        receipts.append({"field": field, "before": before, "after": after, "delta": round(after - before, 2)})
    if summary:
        history = list(rel.get("history") or [])
        history.append({"summary": summary, "calendar_day": _u_current_meta(session)[0]})
        rel["history"] = history[-16:]
    return receipts


def _u_reporter_relationship(session: Any, reporter_id: str, player_id: str) -> Dict[str, Any]:
    store = dict(getattr(session, "universe_reporter_relationships", None) or {})
    key = f"{reporter_id}|{player_id}"
    if key not in store:
        rng = random.Random(_u_seed("reporter_relationship", key))
        store[key] = {
            "reporter_id": reporter_id,
            "player_id": player_id,
            "access": _u_clip(48 + rng.uniform(-12, 14)),
            "trust": _u_clip(52 + rng.uniform(-12, 12)),
            "friction": _u_clip(24 + rng.uniform(-10, 16)),
            "fairness_view": _u_clip(55 + rng.uniform(-12, 12)),
            "interview_count": 0,
            "history": [],
        }
    session.universe_reporter_relationships = store
    return store[key]


def _u_append_event(session: Any, event: Dict[str, Any]) -> None:
    rows = list(getattr(session, "universe_event_log", None) or [])
    rows.append(event)
    session.universe_event_log = rows[-UNIVERSE_MAX_EVENTS:]


def _u_record_storyline(session: Any, *, event: Dict[str, Any], headline: str, summary: str, cause_type: str, category: str, heat: int, public: bool = True) -> Optional[Dict[str, Any]]:
    if not public:
        return None
    day, iso, _ = _u_current_meta(session)
    participants = list(event.get("participants") or [])
    player_id = str(participants[0] if participants else event.get("player_id") or "")
    entities = getattr(session, "universe_players", None) or {}
    entity = entities.get(player_id) or {}
    team_id = str(event.get("team_id") or entity.get("team_id") or "")
    event_id = str(event.get("id") or f"uve_{uuid.uuid4().hex[:10]}")
    try:
        cause_event_id = record_decision_event(
            session,
            {
                "event_type": cause_type,
                "team_id": team_id,
                "player_id": player_id,
                "player_name": entity.get("player_name"),
                "calendar_day": day,
                "calendar_iso": iso,
                "universe_event_id": event_id,
            },
        )
    except Exception:
        cause_event_id = event_id
    row = {
        "id": f"story_{uuid.uuid4().hex[:12]}",
        "storyline_id": f"story_{uuid.uuid4().hex[:12]}",
        "type": str(event.get("kind") or category),
        "category": category,
        "cause_type": cause_type,
        "cause_event_id": cause_event_id,
        "universe_event_id": event_id,
        "team_id": team_id,
        "team_name": _team_display(session, team_id) if team_id else "League",
        "player_id": player_id,
        "player_name": entity.get("player_name") or event.get("player_name") or "",
        "related_player_ids": participants,
        "headline": headline,
        "title": headline,
        "summary": summary,
        "short_summary": summary[:180],
        "description": summary,
        "tone": "negative" if cause_type in ("TEAMMATE_CONFLICT", "TEAMMATE_FIGHT", "PLAYER_REPORTER_ALTERCATION", "PROMISE_BROKEN", "LOW_CHARACTER_GAME_IMPACT") else "positive" if cause_type in ("HIGH_CHARACTER_IMPACT", "PROMISE_KEPT", "COMMUNITY_MOMENT") else "neutral",
        "priority": "HIGH" if heat >= 70 else "MEDIUM" if heat >= 38 else "LOW",
        "severity": "major" if heat >= 70 else "minor",
        "heat": int(_u_clip(heat, 5, 100)),
        "credibility": 90,
        "calendar_day": day,
        "calendar_iso": iso,
        "date": iso or day,
        "source": "storyline_universe_v2",
        "effects": dict(event.get("effects") or {}),
        "stable_key": str(event.get("stable_key") or ""),
        "knowledge_type": event.get("knowledge_type"),
        "public_knowledge_level": event.get("public_knowledge_level"),
        "source_label": event.get("source_label") or "",
        "reporter_name": event.get("reporter_name") or "",
        "outlet_name": event.get("outlet_name") or "",
        "evidence": dict(event.get("evidence") or {}),
        "personality_tags": list(entity.get("personality_tags") or []),
        "top_concerns": list(entity.get("top_concerns") or []),
        "trusts": dict(entity.get("trusts") or {}),
    }
    row["storyline_id"] = row["id"]
    try:
        from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433
        _record_storyline(session, row)
    except (ImportError, ModuleNotFoundError):
        enriched = _UNIVERSE_LEGACY_ENRICH(session, row)
        existing = list(getattr(session, "storyline_events", None) or [])
        existing.append(enriched)
        session.storyline_events = existing[-300:]
    return row


def _u_player_post_text(entity: Dict[str, Any], kind: str) -> str:
    from app.sim_engine.franchise.social_copy_engine import compose_player_post  # noqa: WPS433

    mood_map = {
        "support": "grateful",
        "community": "grateful",
        "win": "win",
        "loss": "loss",
        "frustrated": "frustrated",
    }
    rng = random.Random(_u_seed(entity.get("player_id"), kind, _u_current_meta_placeholder(entity)))
    return compose_player_post(entity, mood_map.get(kind, "win"), rng)


def _flair_from_knowledge_type(ktype: str) -> str:
    k = str(ktype or "report").lower()
    if k == "fact":
        return "Confirmed"
    if k in ("claim", "corroborated_claim"):
        return "Rumor"
    if k == "speculation":
        return "Discussion"
    return "Analysis"


def _team_subreddit_name(session: Any, team_id: str) -> str:
    tm = (getattr(session, "team_by_id", None) or {}).get(str(team_id or ""))
    if tm is None:
        return "hockey"
    name = str(getattr(tm, "name", "") or getattr(tm, "city", "") or "Team")
    slug = re.sub(r"[^A-Za-z0-9]+", "", name.split()[-1] if name else "Team")
    return slug or "hockey"


def _resolve_subreddit(session: Any, sl: Dict[str, Any]) -> str:
    utid = str(getattr(session, "user_team_id") or "")
    tid = str(sl.get("team_id") or utid)
    cat = str(sl.get("category") or "").lower()
    if not tid or "league" in cat or int(sl.get("heat") or 0) >= 90 and not sl.get("player_id"):
        return "r/hockey"
    return f"r/{_team_subreddit_name(session, tid)}"


def _team_from_subreddit(subreddit: str) -> str:
    slug = str(subreddit or "").replace("r/", "").strip()
    return slug


def _reddit_thread_title(sl: Dict[str, Any], rng: random.Random) -> str:
    headline = str(sl.get("headline") or "Thread")
    ktype = str(sl.get("knowledge_type") or "")
    if ktype in ("claim", "speculation"):
        prefixes = ["[Rumor]", "Hearing that", "Is it just me or", "Real talk:"]
        return f"{rng.choice(prefixes)} {headline}"[:120]
    if ktype == "fact":
        return headline[:120]
    return f"Discussion: {headline}"[:120]


def _reddit_op_body(sl: Dict[str, Any], archetype: str, rng: random.Random) -> str:
    from app.sim_engine.franchise.social_copy_engine import build_evidence_context, compose_ambient_fan_post  # noqa: WPS433

    ctx = build_evidence_context(sl)
    tone_map = {
        "diehard_optimist": "hype",
        "diehard_doomer": "outrage",
        "stats_nerd": "concern",
        "old_guard": "concern",
        "rival_troll": "outrage",
    }
    base = compose_ambient_fan_post(tone_map.get(archetype, "concern"), ctx, rng)
    if archetype == "stats_nerd":
        return f"{base} [OC] PPG {ctx.get('ppg')} through {ctx.get('games_played')} GP."
    if archetype == "old_guard":
        return f"{base} Back in my day we didn't need headlines for this."
    return base


def _reddit_top_comments(session: Any, sl: Dict[str, Any], subreddit: str, rng: random.Random) -> Tuple[List[Dict[str, Any]], float]:
    from app.sim_engine.franchise.social_copy_engine import build_evidence_context, compose_ambient_fan_post  # noqa: WPS433

    ctx = build_evidence_context(sl)
    utid = str(getattr(session, "user_team_id") or "")
    is_user_team = subreddit != "r/hockey" and _team_subreddit_name(session, utid).lower() in subreddit.lower()
    comments: List[Dict[str, Any]] = []
    sentiments: List[float] = []
    archetypes = list(REDDIT_FAN_ARCHETYPES.keys())
    for i in range(rng.randint(2, 5)):
        if is_user_team and rng.random() < 0.35:
            arch = "rival_troll"
            rival_ids = [tid for tid in (getattr(session, "team_by_id", None) or {}) if str(tid) != utid]
            rival = rng.choice(rival_ids) if rival_ids else utid
            author = f"u/{_team_subreddit_name(session, rival)}Fan{rng.randint(10, 999)}"
            is_rival = True
            sent = -0.6
            tone = "outrage"
        else:
            arch = rng.choice([a for a in archetypes if a != "rival_troll"] or archetypes)
            author = f"u/{rng.choice(['PuckWatch', 'GlassSeats', 'ThirdLine', 'CapFriendly', 'OldTimer'])}{rng.randint(10, 9999)}"
            is_rival = False
            meta = REDDIT_FAN_ARCHETYPES.get(arch, {})
            tone = {"hopeful": "hype", "critical": "outrage", "analytical": "concern", "nostalgic": "concern", "hostile": "outrage"}.get(
                str(meta.get("tone")), "concern"
            )
            sent = {"hype": 0.7, "outrage": -0.65, "concern": -0.15, "meme": 0.2}.get(tone, 0.0)
        text = compose_ambient_fan_post(tone if tone != "meme" else "meme", ctx, rng)
        if i == 0 and str(sl.get("reporter_name") or ""):
            text = f"{text} ({sl.get('reporter_name')} had something on this too.)"
        up = int(rng.randint(12, 400) * float(REDDIT_FAN_ARCHETYPES.get(arch, {}).get("upvote_bias", 1.0)))
        comments.append({"author": author, "archetype": arch, "text": text[:280], "upvotes": up, "is_rival": is_rival})
        sentiments.append(sent)
    avg_sent = sum(sentiments) / max(1, len(sentiments))
    return comments, avg_sent


def _apply_reddit_sentiment_to_engagement(session: Any, thread: Dict[str, Any]) -> None:
    confidence = float(thread.get("upvote_ratio") or 0.75)
    weight = REDDIT_ENGAGEMENT_WEIGHT * confidence
    delta = float(thread.get("sentiment_score") or 0) * weight
    sub = str(thread.get("subreddit") or "r/hockey")
    utid = str(getattr(session, "user_team_id") or "")
    if sub == "r/hockey":
        team_id = utid
        scale = 0.3
    else:
        team_id = utid
        for tid in (getattr(session, "team_by_id", None) or {}):
            if _team_subreddit_name(session, str(tid)).lower() in sub.lower():
                team_id = str(tid)
                break
        scale = 1.0
    apply_fan_engagement_delta(session, team_id, delta * scale, source="reddit_thread")


def _u_add_reddit_thread(session: Any, sl: Dict[str, Any], rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
    heat = int(sl.get("heat") or 0)
    if heat < 15:
        return None
    r = rng or random.Random(_u_seed(sl.get("storyline_id"), "reddit", heat))
    subreddit = _resolve_subreddit(session, sl)
    ktype = str(sl.get("knowledge_type") or "report")
    flair = _flair_from_knowledge_type(ktype)
    if str(sl.get("reporter_id") or "") == "hart":
        flair = "Discussion"
    arch_key = r.choice(list(REDDIT_FAN_ARCHETYPES.keys()))
    if arch_key == "rival_troll" and subreddit == "r/hockey":
        arch_key = "stats_nerd"
    op_author = f"u/{r.choice(['ThrowawayGM', 'GlassSeats', 'ThirdLineTruth', 'CapWatcher', 'RinkRat'])}{r.randint(100, 9999)}"
    day, iso, _ = _u_current_meta(session)
    comments, comment_sent = _reddit_top_comments(session, sl, subreddit, r)
    op_sent = {"diehard_optimist": 0.5, "diehard_doomer": -0.55, "stats_nerd": 0.1, "old_guard": -0.1, "rival_troll": -0.7}.get(arch_key, 0.0)
    sentiment_score = (op_sent + comment_sent) / 2.0
    if ktype == "fact":
        upvote_ratio = round(r.uniform(0.88, 0.97), 2)
    elif ktype in ("claim", "speculation"):
        upvote_ratio = round(r.uniform(0.55, 0.75), 2)
    else:
        upvote_ratio = round(r.uniform(0.72, 0.88), 2)
    thread = {
        "thread_id": f"ih_{uuid.uuid4().hex[:10]}",
        "subreddit": subreddit,
        "title": _reddit_thread_title(sl, r),
        "op_author": op_author,
        "op_archetype": arch_key,
        "flair": flair,
        "body": _reddit_op_body(sl, arch_key, r),
        "upvotes": int(heat * 18 + r.randint(40, 800)),
        "upvote_ratio": upvote_ratio,
        "comment_count": len(comments) + r.randint(3, 40),
        "top_comments": comments,
        "storyline_id": str(sl.get("storyline_id") or sl.get("id") or ""),
        "knowledge_type": ktype,
        "sentiment_score": round(sentiment_score, 3),
        "created_at": iso,
        "calendar_day": day,
        "heat": heat,
        "player_id": sl.get("player_id"),
        "player_name": sl.get("player_name"),
    }
    threads = list(getattr(session, "reddit_threads", None) or [])
    threads.append(thread)
    session.reddit_threads = threads[-UNIVERSE_MAX_REDDIT_THREADS:]
    _apply_reddit_sentiment_to_engagement(session, thread)
    return thread


def _u_current_meta_placeholder(entity: Dict[str, Any]) -> Any:
    return entity.get("last_tick_day", 0)


def _u_add_social_post(session: Any, post: Dict[str, Any]) -> Dict[str, Any]:
    day, iso, _ = _u_current_meta(session)
    row = {
        "id": str(post.get("id") or f"soc_u_{uuid.uuid4().hex[:10]}"),
        "calendar_day": day,
        "calendar_iso": iso,
        "author_type": "fan",
        "author_name": "Hockey Fan",
        "handle": f"@PuckFan{_u_seed(day, uuid.uuid4().hex) % 9000 + 1000}",
        "verified": False,
        "text": "Hockey night.",
        "likes": 0,
        "reposts": 0,
        "replies": 0,
        "platform": "twitter_style_feed",
        **post,
    }
    posts = list(getattr(session, "social_posts", None) or [])
    posts.append(row)
    session.social_posts = posts[-UNIVERSE_MAX_SOCIAL_POSTS:]
    return row


def _u_social_burst(session: Any, storyline: Dict[str, Any], event: Dict[str, Any], rng: random.Random) -> None:
    if not storyline:
        return
    from app.sim_engine.franchise.social_copy_engine import (  # noqa: WPS433
        build_evidence_context,
        compose_ambient_fan_post,
        compose_player_post,
        compose_reporter_post,
    )

    heat = int(storyline.get("heat") or 35)
    participants = list(event.get("participants") or [])
    entities = getattr(session, "universe_players", None) or {}
    reporter = _pick_reporter_for_storyline(storyline, session)
    root = _u_add_social_post(
        session,
        {
            "author_type": "reporter",
            "author_id": reporter["id"],
            "author_name": reporter["name"],
            "handle": f"@{reporter['name'].replace(' ', '')}",
            "verified": True,
            "outlet": reporter["outlet"],
            "text": compose_reporter_post(storyline, reporter, rng, session)[:280],
            "related_headline": storyline.get("headline"),
            "storyline_id": storyline.get("storyline_id"),
            "universe_event_id": event.get("id"),
            "sentiment": "reporting",
            "likes": rng.randint(180, 900) + heat * 35,
            "reposts": rng.randint(40, 220) + heat * 11,
            "replies": rng.randint(30, 180) + heat * 7,
        },
    )
    ctx = build_evidence_context(storyline, session)
    fan_sentiments = ["hype", "concern", "meme", "outrage"] if heat >= 60 else ["hype", "concern", "meme"]
    for index in range(2 + (1 if heat >= 60 else 0)):
        sent = rng.choice(fan_sentiments)
        _u_add_social_post(
            session,
            {
                "author_name": rng.choice(["Rink Rat", "Cap Space Enjoyer", "Fourth Line Truthers", "PuckWatch", "Hockey After Dark"]),
                "handle": f"@FanVoice{rng.randint(100, 9999)}",
                "text": compose_ambient_fan_post(sent, ctx, rng, storyline, reporter),
                "reply_to_id": root["id"] if index > 0 else None,
                "storyline_id": storyline.get("storyline_id"),
                "universe_event_id": event.get("id"),
                "sentiment": rng.choice(["supportive", "skeptical", "joking", "concerned"]),
                "likes": rng.randint(15, 900),
                "reposts": rng.randint(2, 120),
                "replies": rng.randint(1, 90),
            },
        )
    if participants and heat >= 42:
        entity = entities.get(str(participants[0])) or {}
        style = str((entity.get("social") or {}).get("style") or "quiet")
        if style != "quiet" and rng.random() < 0.48:
            mood = "frustrated" if heat >= 65 else "win"
            _u_add_social_post(
                session,
                {
                    "author_type": "player",
                    "author_id": entity.get("player_id"),
                    "author_name": entity.get("player_name"),
                    "handle": (entity.get("social") or {}).get("handle"),
                    "verified": True,
                    "text": compose_player_post(entity, mood, rng),
                    "reply_to_id": root["id"],
                    "storyline_id": storyline.get("storyline_id"),
                    "universe_event_id": event.get("id"),
                    "sentiment": "personal",
                    "likes": rng.randint(800, 18000),
                    "reposts": rng.randint(100, 2500),
                    "replies": rng.randint(120, 3200),
                },
            )
    _u_add_reddit_thread(session, storyline, rng)


def _u_choice(choice_id: str, label: str, description: str, outcome: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": choice_id, "label": label, "description": description, "outcome": outcome}


def _u_make_interaction(session: Any, team_id: str, kind: str, actor_id: str, target_id: str, rng: random.Random, score: float) -> Dict[str, Any]:
    entities = getattr(session, "universe_players", None) or {}
    actor = entities.get(actor_id) or {}
    target = entities.get(target_id) or {}
    actor_name = str(actor.get("player_name") or "Player")
    target_name = str(target.get("player_name") or "a teammate")
    day, iso, _ = _u_current_meta(session)
    interaction_id = f"int_{uuid.uuid4().hex[:12]}"
    base = {
        "id": interaction_id,
        "kind": kind,
        "team_id": str(team_id),
        "participants": [pid for pid in (actor_id, target_id) if pid],
        "actor_id": actor_id,
        "target_id": target_id,
        "calendar_day": day,
        "calendar_iso": iso,
        "expires_day": day + 3,
        "status": "pending",
        "score": round(score, 2),
        "requires_action": str(team_id) == str(getattr(session, "user_team_id", "") or ""),
        "private": False,
        "stakes": "medium",
        "dialogue": [],
        "choices": [],
        "default_choice_id": "observe",
    }
    if kind == "mentor_session":
        base.update(
            {
                "title": f"{actor_name} takes {target_name} under his wing",
                "summary": f"A veteran-led film and practice session gives {target_name} a clearer path through recent mistakes.",
                "private": True,
                "stakes": "low",
                "dialogue": [
                    {"speaker": actor_name, "text": "You don't need to solve the league in one shift. Read the first option and trust it."},
                    {"speaker": target_name, "text": "I’ve been forcing it. Show me what you’re seeing."},
                ],
                "choices": [
                    _u_choice("recognize_mentor", "Recognize the mentorship", "Give the veteran an explicit development role.", {"profile_changes": {"actor": {"state.belonging": 3, "state.morale": 2}, "target": {"state.confidence": 4}}, "relationship": {"trust": 6, "respect": 5, "tension": -3}, "attributes": [{"who": "target", "attribute": "offensive_awareness", "delta": 1, "permanent": False, "duration_games": 8, "reason": "Veteran mentorship"}], "public": False}),
                    _u_choice("let_players_lead", "Let the players own it", "The bond grows organically without management stepping in.", {"profile_changes": {"actor": {"state.belonging": 2}, "target": {"state.confidence": 3}}, "relationship": {"trust": 7, "respect": 4, "tension": -2}, "attributes": [{"who": "target", "attribute": "passing", "delta": 0.5, "permanent": False, "duration_games": 6, "reason": "Peer film work"}], "public": False}),
                ],
                "default_choice_id": "let_players_lead",
            }
        )
    elif kind == "glue_intervention":
        base.update(
            {
                "title": f"{actor_name} steadies the room",
                "summary": f"With tension building, {actor_name} pulls teammates together and keeps frustration from becoming a split.",
                "private": True,
                "stakes": "medium",
                "dialogue": [{"speaker": actor_name, "text": "We can be angry about the result without turning on each other. Say it now, then we move."}],
                "choices": [
                    _u_choice("empower", "Empower the leadership group", "Let the room solve the issue internally.", {"profile_changes": {"actor": {"state.belonging": 4, "state.coach_trust": 3}, "target": {"state.morale": 2}}, "relationship": {"trust": 5, "tension": -7}, "team_changes": {"unity": 4, "tension": -6}, "public": False}),
                    _u_choice("join_meeting", "Join the meeting", "Management reinforces the message but risks crowding the room.", {"profile_changes": {"actor": {"state.gm_trust": 3}, "target": {"state.gm_trust": 2}}, "relationship": {"trust": 3, "tension": -4}, "team_changes": {"accountability": 3, "tension": -3}, "public": False}),
                ],
                "default_choice_id": "empower",
            }
        )
    elif kind == "role_frustration":
        base.update(
            {
                "title": f"{actor_name} wants clarity on his role",
                "summary": f"{actor_name} asks for a direct conversation after feeling his role and status have slipped.",
                "private": True,
                "stakes": "high",
                "dialogue": [
                    {"speaker": actor_name, "text": "I can handle an honest answer. I can't handle not knowing what you expect from me."},
                    {"speaker": "GM", "text": "This is your chance to set expectations."},
                ],
                "choices": [
                    _u_choice("honest_role", "Give an honest role assessment", "Trust improves even if the answer is difficult.", {"profile_changes": {"actor": {"state.gm_trust": 5, "state.role_satisfaction": -1, "state.focus": 3}}, "public": False}),
                    _u_choice("promise_opportunity", "Promise a larger opportunity", "Morale rises now; failing to deliver will carry a larger cost.", {"profile_changes": {"actor": {"state.morale": 5, "state.gm_trust": 3}}, "promise": {"type": "role_opportunity", "player_id": actor_id, "due_games": 6, "description": "Give the player a meaningful lineup opportunity within six games."}, "public": False}),
                    _u_choice("challenge_player", "Challenge him to earn it", "A competitive player may respond, but trust can fall.", {"profile_changes": {"actor": {"state.gm_trust": -4, "state.focus": 4, "state.morale": -2}}, "attributes": [{"who": "actor", "attribute": "stamina", "delta": 0.5, "permanent": False, "duration_games": 4, "reason": "Role challenge"}], "public": False}),
                ],
                "default_choice_id": "honest_role",
            }
        )
    elif kind == "blame_game":
        base.update(
            {
                "title": f"{actor_name} and {target_name} trade blame",
                "summary": f"A video-session disagreement spills into the room as {actor_name} publicly blames {target_name} for a breakdown.",
                "stakes": "high",
                "dialogue": [
                    {"speaker": actor_name, "text": "I can't cover two assignments because somebody decides structure is optional."},
                    {"speaker": target_name, "text": "Say my name if you're going to make this about me."},
                ],
                "choices": [
                    _u_choice("private_mediation", "Mediate in private", "Address the behavior without creating public winners and losers.", {"profile_changes": {"actor": {"state.coach_trust": -1, "state.morale": -1}, "target": {"state.coach_trust": 2}}, "relationship": {"trust": -2, "respect": 2, "tension": -6}, "team_changes": {"tension": -4, "accountability": 2}, "public": False}),
                    _u_choice("hold_actor_accountable", f"Hold {actor_name} accountable", "Back team standards and risk alienating the instigator.", {"profile_changes": {"actor": {"state.morale": -5, "state.coach_trust": -3}, "target": {"state.morale": 3}}, "relationship": {"trust": -4, "respect": 1, "tension": 2}, "team_changes": {"accountability": 5, "tension": -1}, "public": True, "cause_type": "TEAMMATE_CONFLICT", "heat": 48}),
                    _u_choice("let_room_handle", "Let the room handle it", "Leadership may contain it, or the disagreement may deepen.", {"profile_changes": {"actor": {"state.gm_trust": -1}, "target": {"state.gm_trust": -1}}, "relationship": {"trust": -5, "tension": 7}, "team_changes": {"tension": 5}, "public": rng.random() < 0.35, "cause_type": "TEAMMATE_CONFLICT", "heat": 55}),
                ],
                "default_choice_id": "private_mediation",
            }
        )
    elif kind == "teammate_fight":
        base.update(
            {
                "title": f"Practice fight: {actor_name} and {target_name} separated by teammates",
                "summary": f"A high-intensity practice ends with punches exchanged between {actor_name} and {target_name}. The incident exposes a deeper split in the room.",
                "stakes": "critical",
                "dialogue": [
                    {"speaker": "Coach", "text": "Competing is one thing. Turning on your own teammate is another."},
                    {"speaker": actor_name, "text": "It had been building. I'm not pretending it hadn't."},
                ],
                "choices": [
                    _u_choice("discipline_both", "Discipline both players", "Set a clear standard, with a short-term morale cost.", {"profile_changes": {"actor": {"state.morale": -5, "state.coach_trust": -3}, "target": {"state.morale": -5, "state.coach_trust": -3}}, "relationship": {"trust": -8, "respect": -4, "tension": -5}, "team_changes": {"accountability": 6, "tension": -2}, "attributes": [{"who": "actor", "attribute": "discipline", "delta": -1, "permanent": False, "duration_games": 4, "reason": "Practice altercation"}, {"who": "target", "attribute": "discipline", "delta": -1, "permanent": False, "duration_games": 4, "reason": "Practice altercation"}], "public": True, "cause_type": "TEAMMATE_FIGHT", "heat": 78}),
                    _u_choice("leadership_repair", "Order a leadership-led repair", "Create a path back, but only if both players engage honestly.", {"profile_changes": {"actor": {"state.belonging": -2}, "target": {"state.belonging": -2}}, "relationship": {"trust": -4, "respect": 1, "tension": -10}, "team_changes": {"unity": -2, "tension": -6}, "public": True, "cause_type": "TEAMMATE_FIGHT", "heat": 66}),
                    _u_choice("separate_players", "Separate the players", "Stop immediate escalation without solving the underlying issue.", {"profile_changes": {"actor": {"state.morale": -2}, "target": {"state.morale": -2}}, "relationship": {"trust": -7, "tension": 4}, "team_changes": {"unity": -5, "tension": 5}, "public": True, "cause_type": "TEAMMATE_FIGHT", "heat": 82}),
                ],
                "default_choice_id": "discipline_both",
            }
        )
    elif kind in ("reporter_confrontation", "reporter_altercation"):
        reporter = _REPORTER_BY_ID.get("hart", MEDIA_REPORTERS[-1])
        physical = kind == "reporter_altercation"
        base.update(
            {
                "reporter_id": reporter["id"],
                "title": f"{actor_name} confronts {reporter['name']}" if not physical else f"Media hallway altercation involving {actor_name}",
                "summary": f"{actor_name} challenges {reporter['name']} over repeated coverage." if not physical else f"A heated exchange between {actor_name} and {reporter['name']} turns into a brief shoving incident before security intervenes.",
                "stakes": "critical" if physical else "high",
                "dialogue": [
                    {"speaker": reporter["name"], "text": "If the reporting is wrong, tell me exactly what is wrong."},
                    {"speaker": actor_name, "text": "You know what you're doing. You're turning every answer into a crisis."},
                ],
                "choices": [
                    _u_choice("back_player_privately", "Back the player privately", "Support him while making public conduct expectations clear.", {"profile_changes": {"actor": {"state.gm_trust": 5, "state.media_stress": -4, "state.coach_trust": -1}}, "reporter_changes": {"friction": 6, "access": -8, "trust": -3}, "public": True, "cause_type": "PLAYER_REPORTER_ALTERCATION" if physical else "PLAYER_REPORTER_CONFRONTATION", "heat": 82 if physical else 62}),
                    _u_choice("public_accountability", "Hold the player accountable publicly", "Reduce institutional heat while risking player trust.", {"profile_changes": {"actor": {"state.gm_trust": -7, "state.morale": -4, "state.media_stress": 3}}, "reporter_changes": {"friction": -4, "access": 5, "trust": 4}, "team_changes": {"accountability": 4}, "attributes": [{"who": "actor", "attribute": "discipline", "delta": -1, "permanent": False, "duration_games": 5, "reason": "Media incident"}], "public": True, "cause_type": "PLAYER_REPORTER_ALTERCATION" if physical else "PLAYER_REPORTER_CONFRONTATION", "heat": 86 if physical else 68}),
                    _u_choice("joint_statement", "Arrange a joint statement", "Lower the temperature without declaring either side the winner.", {"profile_changes": {"actor": {"state.media_stress": -2, "state.gm_trust": 2}}, "reporter_changes": {"friction": -8, "access": 2, "trust": 1}, "team_changes": {"tension": -1}, "public": True, "cause_type": "PLAYER_REPORTER_CONFRONTATION", "heat": 48}),
                ],
                "default_choice_id": "joint_statement",
            }
        )
    elif kind == "personal_check_in":
        base.update(
            {
                "title": f"Private check-in with {actor_name}",
                "summary": f"Travel, home responsibilities, and hockey pressure are beginning to drain {actor_name}'s focus.",
                "private": True,
                "stakes": "medium",
                "dialogue": [{"speaker": actor_name, "text": "I'm not asking for special treatment. I just need a little room to get everything settled."}],
                "choices": [
                    _u_choice("support_day", "Offer a personal day", "Energy and trust improve at a small short-term readiness cost.", {"profile_changes": {"actor": {"state.personal_stress": -10, "state.energy": 7, "state.gm_trust": 5}}, "attributes": [{"who": "actor", "attribute": "stamina", "delta": -0.5, "permanent": False, "duration_games": 1, "reason": "Personal day"}], "public": False}),
                    _u_choice("support_resources", "Offer team support resources", "A balanced response that protects privacy.", {"profile_changes": {"actor": {"state.personal_stress": -6, "state.gm_trust": 4, "state.focus": 3}}, "public": False}),
                    _u_choice("hockey_first", "Tell him to compartmentalize", "No schedule cost, but the player may feel unseen.", {"profile_changes": {"actor": {"state.gm_trust": -6, "state.focus": 1, "state.personal_stress": 4}}, "public": False}),
                ],
                "default_choice_id": "support_resources",
            }
        )
    elif kind == "unheralded_leader":
        base.update(
            {
                "title": f"The value behind {actor_name}'s stat line",
                "summary": f"Coaches credit {actor_name} with stabilizing teammates, raising practice habits, and keeping the bench connected despite limited scoring production.",
                "private": False,
                "stakes": "low",
                "dialogue": [{"speaker": "Assistant Coach", "text": f"You won't find all of {actor_name}'s value in goals and assists. Take him out of this room and everybody feels it."}],
                "choices": [
                    _u_choice("recognize_publicly", "Recognize him publicly", "Raise belonging and status, with a small media boost.", {"profile_changes": {"actor": {"state.morale": 5, "state.belonging": 6, "state.gm_trust": 4}}, "team_changes": {"unity": 3}, "public": True, "cause_type": "HIGH_CHARACTER_IMPACT", "heat": 34}),
                    _u_choice("recognize_privately", "Recognize him privately", "Deepen trust without changing public expectations.", {"profile_changes": {"actor": {"state.morale": 4, "state.belonging": 4, "state.gm_trust": 6}}, "team_changes": {"unity": 2}, "public": False}),
                ],
                "default_choice_id": "recognize_privately",
            }
        )
    else:
        base.update(
            {
                "title": f"Locker-room conversation involving {actor_name}",
                "summary": f"A team interaction puts {actor_name}'s role and relationships in focus.",
                "choices": [_u_choice("observe", "Observe", "Let the interaction resolve naturally.", {"public": False})],
            }
        )
    return base


def _u_interaction_candidates(session: Any, team_id: str, rng: random.Random) -> List[Tuple[float, str, str, str]]:
    entities = getattr(session, "universe_players", None) or {}
    room = _u_rebuild_locker_room(session, team_id)
    player_ids = [str(getattr(p, "id", "") or "") for p in _u_team_players(session, team_id)]
    candidates: List[Tuple[float, str, str, str]] = []
    for player_id in player_ids:
        entity = entities.get(player_id) or {}
        state = entity.get("state") or {}
        personality = entity.get("personality") or {}
        niches = _u_niche_ids(entity)
        if float(state.get("role_satisfaction", 60)) < 42:
            candidates.append((78 - float(state.get("role_satisfaction", 60)) + float(personality.get("ambition", 50)) * 0.25, "role_frustration", player_id, ""))
        if float(state.get("personal_stress", 25)) > 65:
            candidates.append((float(state.get("personal_stress", 25)) + 8, "personal_check_in", player_id, ""))
        reporter_rel = _u_reporter_relationship(session, "hart", player_id)
        media_score = float(state.get("media_stress", 25)) + float(reporter_rel.get("friction", 20)) * 0.55 + float(personality.get("volatility", 40)) * 0.25
        if media_score > 76:
            kind = "reporter_altercation" if media_score > 118 and float(personality.get("volatility", 0)) > 76 and rng.random() < 0.16 else "reporter_confrontation"
            candidates.append((media_score, kind, player_id, ""))
        if float(entity.get("overall", 99)) < 80 and float(entity.get("room_value", 0)) >= 72:
            candidates.append((float(entity.get("room_value", 0)) - 8, "unheralded_leader", player_id, ""))
        if "mentor" in niches or "glue_guy" in niches or "peacemaker" in niches:
            possible_targets = [pid for pid in player_ids if pid != player_id]
            if possible_targets:
                target_id = min(possible_targets, key=lambda pid: float((entities.get(pid) or {}).get("age", 27)) + float(((entities.get(pid) or {}).get("state") or {}).get("confidence", 55)) * 0.08)
                target = entities.get(target_id) or {}
                if "mentor" in niches and int(target.get("age", 30)) <= 24:
                    candidates.append((58 + max(0, 52 - float((target.get("state") or {}).get("confidence", 55))), "mentor_session", player_id, target_id))
                if ("glue_guy" in niches or "peacemaker" in niches) and float((room.get("culture") or {}).get("tension", 30)) >= 44:
                    candidates.append((62 + float((room.get("culture") or {}).get("tension", 30)) * 0.35, "glue_intervention", player_id, target_id))
    relationships = room.get("relationships") or {}
    for rel in relationships.values():
        ids = list(rel.get("player_ids") or [])
        if len(ids) != 2 or any(pid not in entities for pid in ids):
            continue
        a = entities[ids[0]]
        b = entities[ids[1]]
        tension = float(rel.get("tension", 0))
        volatility = (float((a.get("personality") or {}).get("volatility", 40)) + float((b.get("personality") or {}).get("volatility", 40))) / 2
        character_floor = min(float((a.get("personality") or {}).get("character", 55)), float((b.get("personality") or {}).get("character", 55)))
        if tension >= 62 and character_floor < 52:
            kind = "teammate_fight" if tension >= 82 and volatility >= 72 and rng.random() < 0.24 else "blame_game"
            candidates.append((tension + volatility * 0.35, kind, ids[0], ids[1]))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    return candidates


def _u_apply_outcome(session: Any, interaction: Dict[str, Any], choice: Dict[str, Any]) -> Dict[str, Any]:
    outcome = dict(choice.get("outcome") or {})
    entities = getattr(session, "universe_players", None) or {}
    actor_id = str(interaction.get("actor_id") or "")
    target_id = str(interaction.get("target_id") or "")
    team_id = str(interaction.get("team_id") or "")
    day, iso, _ = _u_current_meta(session)
    receipts: Dict[str, Any] = {"profiles": [], "relationships": [], "attributes": [], "reporter": [], "team": []}
    role_ids = {"actor": actor_id, "target": target_id}
    for role, changes in (outcome.get("profile_changes") or {}).items():
        player_id = role_ids.get(str(role), str(role))
        entity = entities.get(player_id)
        if not entity:
            continue
        for field, delta in (changes or {}).items():
            receipt = _u_apply_profile_delta(entity, str(field), float(delta))
            receipt["player_id"] = player_id
            receipts["profiles"].append(receipt)
        _u_add_memory(entity, kind=str(interaction.get("kind") or "interaction"), summary=str(interaction.get("summary") or "Team interaction"), day=day, iso=iso, emotional_delta=sum(float(v) for v in (changes or {}).values()), related_ids=[pid for pid in (actor_id, target_id) if pid and pid != player_id], public=bool(outcome.get("public")))
    if actor_id and target_id and outcome.get("relationship"):
        receipts["relationships"] = _u_change_relationship(session, team_id, actor_id, target_id, dict(outcome["relationship"]), str(interaction.get("title") or "Interaction"))
    room = _u_rebuild_locker_room(session, team_id) if team_id else {}
    culture = room.get("culture") or {}
    for field, delta in (outcome.get("team_changes") or {}).items():
        before = float(culture.get(field, 50.0) or 0.0)
        after = _u_clip(before + float(delta))
        culture[field] = after
        receipts["team"].append({"field": field, "before": before, "after": after, "delta": round(after - before, 2)})
    for change in outcome.get("attributes") or []:
        who = str(change.get("who") or "actor")
        player_id = role_ids.get(who, who)
        if player_id:
            receipts["attributes"].append(_u_apply_attribute_change(session, player_id, change, str(interaction.get("id") or "interaction")))
    reporter_id = str(interaction.get("reporter_id") or "")
    if reporter_id and actor_id:
        rel = _u_reporter_relationship(session, reporter_id, actor_id)
        for field, delta in (outcome.get("reporter_changes") or {}).items():
            before = float(rel.get(field, 50.0) or 0.0)
            after = _u_clip(before + float(delta))
            rel[field] = after
            receipts["reporter"].append({"field": field, "before": before, "after": after, "delta": round(after - before, 2)})
        rel["interview_count"] = int(rel.get("interview_count", 0) or 0) + 1
        history = list(rel.get("history") or [])
        history.append({"interaction_id": interaction.get("id"), "choice_id": choice.get("id"), "calendar_day": day})
        rel["history"] = history[-20:]
    promise_spec = outcome.get("promise")
    if isinstance(promise_spec, dict):
        promises = list(getattr(session, "universe_promises", None) or [])
        due_games = int(promise_spec.get("due_games") or 5)
        promises.append(
            {
                "id": f"promise_{uuid.uuid4().hex[:10]}",
                "interaction_id": interaction.get("id"),
                "type": promise_spec.get("type"),
                "player_id": promise_spec.get("player_id") or actor_id,
                "description": promise_spec.get("description"),
                "created_day": day,
                "games_remaining": due_games,
                "status": "active",
                "progress": 0,
            }
        )
        session.universe_promises = promises[-80:]
        receipts["promise_id"] = promises[-1]["id"]
    interaction["status"] = "resolved"
    interaction["resolved_day"] = day
    interaction["resolved_iso"] = iso
    interaction["selected_choice_id"] = choice.get("id")
    interaction["resolution"] = receipts
    interaction["effects"] = {"team_morale": sum(r.get("delta", 0) for r in receipts["team"] if r.get("field") in ("unity", "confidence")), "room_tension": sum(r.get("delta", 0) for r in receipts["team"] if r.get("field") == "tension")}
    cause_type = str(outcome.get("cause_type") or ("PLAYER_INTERACTION" if outcome.get("public") else ""))
    storyline = None
    if bool(outcome.get("public")):
        storyline = _u_record_storyline(
            session,
            event=interaction,
            headline=str(interaction.get("title") or "Team interaction"),
            summary=str(interaction.get("summary") or "A team interaction became public."),
            cause_type=cause_type or "PLAYER_INTERACTION",
            category="locker_room" if "reporter" not in str(interaction.get("kind") or "") else "media",
            heat=int(outcome.get("heat") or 38),
            public=True,
        )
        if storyline:
            _u_social_burst(session, storyline, interaction, random.Random(_u_seed(interaction.get("id"), choice.get("id"))))
    _u_append_event(session, {"id": interaction.get("id"), "kind": interaction.get("kind"), "team_id": team_id, "participants": interaction.get("participants"), "choice_id": choice.get("id"), "calendar_day": day, "calendar_iso": iso, "public": bool(outcome.get("public")), "receipts": receipts})
    return {"interaction": interaction, "receipts": receipts, "storyline": storyline}


def resolve_universe_interaction(session: Any, interaction_id: str, choice_id: str) -> Dict[str, Any]:
    """Resolve one EA-style conversation choice and return exact consequences."""
    migrate_session_storyline_state(session)
    all_rows = list(getattr(session, "universe_interactions", None) or [])
    interaction = next((row for row in all_rows if str(row.get("id") or "") == str(interaction_id)), None)
    if interaction is None:
        raise ValueError(f"Universe interaction not found: {interaction_id}")
    if str(interaction.get("status") or "") != "pending":
        raise ValueError(f"Universe interaction is already {interaction.get('status')}: {interaction_id}")
    choice = next((row for row in (interaction.get("choices") or []) if str(row.get("id") or "") == str(choice_id)), None)
    if choice is None:
        raise ValueError(f"Choice not found: {choice_id}")
    result = _u_apply_outcome(session, interaction, choice)
    session.universe_interactions = all_rows[-UNIVERSE_MAX_INTERACTIONS:]
    session.universe_interaction_queue = [row for row in (getattr(session, "universe_interaction_queue", None) or []) if str(row.get("id") or "") != str(interaction_id)]
    return result


def _u_queue_or_resolve(session: Any, interaction: Dict[str, Any]) -> None:
    rows = list(getattr(session, "universe_interactions", None) or [])
    rows.append(interaction)
    session.universe_interactions = rows[-UNIVERSE_MAX_INTERACTIONS:]
    if interaction.get("requires_action"):
        queue = list(getattr(session, "universe_interaction_queue", None) or [])
        queue.append(interaction)
        session.universe_interaction_queue = queue[-12:]
        return
    choices = list(interaction.get("choices") or [])
    default_id = str(interaction.get("default_choice_id") or "")
    choice = next((row for row in choices if str(row.get("id") or "") == default_id), choices[0] if choices else None)
    if choice:
        _u_apply_outcome(session, interaction, choice)


def _u_generate_daily_interactions(session: Any, rng: random.Random) -> int:
    user_team_id = str(getattr(session, "user_team_id", "") or "")
    created = 0
    team_ids = list((getattr(session, "team_by_id", None) or {}).keys())
    if user_team_id in team_ids:
        team_ids.remove(user_team_id)
        team_ids.insert(0, user_team_id)
    pending = len([row for row in (getattr(session, "universe_interaction_queue", None) or []) if str(row.get("status") or "") == "pending"])
    for index, team_id_raw in enumerate(team_ids):
        team_id = str(team_id_raw)
        candidates = _u_interaction_candidates(session, team_id, rng)
        if not candidates:
            continue
        is_user = team_id == user_team_id
        if is_user and pending >= 3:
            continue
        threshold = 42 if is_user else 60
        score, kind, actor_id, target_id = candidates[0]
        roll = rng.uniform(0, 100)
        if score + rng.uniform(-12, 12) < threshold or roll > (54 if is_user else 28):
            continue
        interaction = _u_make_interaction(session, team_id, kind, actor_id, target_id, rng, score)
        _u_queue_or_resolve(session, interaction)
        created += 1
        if is_user:
            continue
        if created >= 8:
            break
    return created


def _u_infer_role_satisfaction(session: Any, team_id: str, entity: Dict[str, Any]) -> float:
    roster_entities = [
        (getattr(session, "universe_players", None) or {}).get(str(getattr(player, "id", "") or "")) or {}
        for player in _u_team_players(session, team_id)
    ]
    ranked = sorted(roster_entities, key=lambda row: -float(row.get("overall", 0)))
    player_id = str(entity.get("player_id") or "")
    rank = next((i + 1 for i, row in enumerate(ranked) if str(row.get("player_id") or "") == player_id), len(ranked))
    expected = 78 if rank <= 3 else 68 if rank <= 8 else 58 if rank <= 14 else 50
    ambition = float((entity.get("personality") or {}).get("ambition", 50))
    ego = float((entity.get("personality") or {}).get("ego", 50))
    return _u_clip(expected - max(0, ambition - 65) * 0.16 - max(0, ego - 70) * 0.18)


def _u_tick_player_life(session: Any, team_id: str, entity: Dict[str, Any], rng: random.Random) -> None:
    day, iso, season = _u_current_meta(session)
    if int(entity.get("last_tick_day", -1) or -1) == day:
        return
    state = entity.get("state") or {}
    life = entity.get("life") or {}
    personality = entity.get("personality") or {}
    previous_role = float(state.get("role_satisfaction", 58))
    role_target = _u_infer_role_satisfaction(session, team_id, entity)
    state["role_satisfaction"] = _u_clip(previous_role + (role_target - previous_role) * 0.18)
    home_stability = float(life.get("home_stability", 65))
    relocation = float(life.get("relocation_strain", 15))
    sleep = float(life.get("sleep_quality", 68))
    personal_target = _u_clip(30 + relocation * 0.30 + max(0, 60 - home_stability) * 0.36 + max(0, 62 - sleep) * 0.28)
    state["personal_stress"] = _u_clip(float(state.get("personal_stress", 30)) * 0.86 + personal_target * 0.14 + rng.uniform(-1.8, 1.8))
    state["energy"] = _u_clip(float(state.get("energy", 70)) + (sleep - 65) * 0.035 - float(state.get("personal_stress", 30)) * 0.012 + rng.uniform(-1.2, 1.2))
    state["focus"] = _u_clip(48 + float(state.get("confidence", 55)) * 0.24 + float(state.get("energy", 70)) * 0.20 - float(state.get("personal_stress", 30)) * 0.14)
    life["relocation_strain"] = _u_clip(relocation - 0.18)
    life["community_connection"] = _u_clip(float(life.get("community_connection", 35)) + rng.uniform(0, 0.25))
    if rng.random() < 0.012 and float(life.get("community_connection", 0)) >= 58:
        life["current_note"] = "A community commitment has strengthened the player's connection to the city."
        state["belonging"] = _u_clip(float(state.get("belonging", 55)) + 2)
    elif float(state.get("personal_stress", 0)) >= 68:
        life["current_note"] = "Travel and home responsibilities are competing with hockey focus."
    else:
        life["current_note"] = "Home life is stable."
    concerns = entity.get("concerns") or {}
    role_concern = concerns.get("role") or {}
    role_concern["trend"] = round(float(state.get("role_satisfaction", 58)) - previous_role, 2)
    role_concern["satisfaction"] = float(state.get("role_satisfaction", 58))
    concerns["role"] = role_concern
    home_concern = concerns.get("home_life") or {}
    home_before = float(home_concern.get("satisfaction", 65))
    home_after = _u_clip(home_stability - relocation * 0.35)
    home_concern.update({"satisfaction": home_after, "trend": round(home_after - home_before, 2)})
    concerns["home_life"] = home_concern
    years = _u_contract_years(_player_from_roster(session, str(entity.get("player_id") or "")) or object(), season)
    contract_concern = concerns.get("contract") or {}
    old_contract = float(contract_concern.get("satisfaction", 55))
    contract_after = _u_clip(38 + years * 18 - max(0, float(personality.get("ambition", 50)) - 70) * 0.18)
    contract_concern.update({"satisfaction": contract_after, "trend": round(contract_after - old_contract, 2)})
    concerns["contract"] = contract_concern
    w, l, o, _ = _team_record(session, team_id)
    gp = w + l + o
    points_pct = (w * 2 + o) / max(2, gp * 2)
    winning = concerns.get("winning") or {}
    old_win = float(winning.get("satisfaction", 55))
    win_after = _u_clip(25 + points_pct * 70)
    winning.update({"satisfaction": win_after, "trend": round(win_after - old_win, 2)})
    concerns["winning"] = winning
    belonging = concerns.get("team_belonging") or {}
    old_belong = float(belonging.get("satisfaction", 60))
    belong_after = float(state.get("belonging", 60))
    belonging.update({"satisfaction": belong_after, "trend": round(belong_after - old_belong, 2)})
    concerns["team_belonging"] = belonging
    entity["top_concerns"] = sorted(
        [
            {"id": key, **value, "pressure": round(float(value.get("importance", 50)) * (100 - float(value.get("satisfaction", 50))) / 100, 1)}
            for key, value in concerns.items()
        ],
        key=lambda row: (-float(row.get("pressure", 0)), row["id"]),
    )[:3]
    entity["last_tick_day"] = day
    entity["last_updated_iso"] = iso


def _u_tick_promises(session: Any) -> Dict[str, int]:
    promises = list(getattr(session, "universe_promises", None) or [])
    counts = {"kept": 0, "broken": 0}
    day, iso, _ = _u_current_meta(session)
    for promise in promises:
        if str(promise.get("status") or "") != "active":
            continue
        if int(promise.get("games_remaining", 1) or 0) > 0:
            continue
        fulfilled = int(promise.get("progress", 0) or 0) > 0
        promise["status"] = "kept" if fulfilled else "broken"
        promise["resolved_day"] = day
        player_id = str(promise.get("player_id") or "")
        entity = (getattr(session, "universe_players", None) or {}).get(player_id) or {}
        delta = 6 if fulfilled else -12
        _u_apply_profile_delta(entity, "state.gm_trust", delta)
        _u_apply_profile_delta(entity, "state.morale", 3 if fulfilled else -6)
        _u_add_memory(entity, kind=promise["status"], summary=f"Management promise {promise['status']}: {promise.get('description')}", day=day, iso=iso, emotional_delta=delta, public=not fulfilled)
        event = {"id": f"uve_{uuid.uuid4().hex[:10]}", "kind": f"promise_{promise['status']}", "team_id": entity.get("team_id"), "participants": [player_id], "player_id": player_id}
        _u_append_event(session, {**event, "calendar_day": day, "calendar_iso": iso, "promise_id": promise.get("id")})
        _u_record_storyline(session, event=event, headline=f"Management promise {promise['status']} with {entity.get('player_name') or 'player'}", summary=str(promise.get("description") or "A management promise reached its deadline."), cause_type="PROMISE_KEPT" if fulfilled else "PROMISE_BROKEN", category="management", heat=32 if fulfilled else 67, public=not fulfilled or bool(promise.get("public")))
        counts[promise["status"]] += 1
    session.universe_promises = promises
    return counts


def record_universe_promise_progress(session: Any, player_id: str, promise_type: str, amount: int = 1) -> int:
    """Call from lineup/contract code when the GM delivers on a promise."""
    migrate_session_storyline_state(session)
    changed = 0
    for promise in getattr(session, "universe_promises", None) or []:
        if str(promise.get("status") or "") == "active" and str(promise.get("player_id") or "") == str(player_id) and str(promise.get("type") or "") == str(promise_type):
            promise["progress"] = int(promise.get("progress", 0) or 0) + int(amount)
            changed += 1
    return changed


def _u_expire_interactions(session: Any) -> int:
    day = _u_current_meta(session)[0]
    expired = 0
    for interaction in list(getattr(session, "universe_interactions", None) or []):
        if str(interaction.get("status") or "") != "pending" or int(interaction.get("expires_day", day + 1) or day + 1) >= day:
            continue
        choices = list(interaction.get("choices") or [])
        default_id = str(interaction.get("default_choice_id") or "")
        choice = next((row for row in choices if str(row.get("id") or "") == default_id), choices[0] if choices else None)
        if choice:
            _u_apply_outcome(session, interaction, choice)
        else:
            interaction["status"] = "expired"
        expired += 1
    session.universe_interaction_queue = [row for row in (getattr(session, "universe_interaction_queue", None) or []) if str(row.get("status") or "") == "pending"]
    return expired


def _u_generate_ambient_social(session: Any, rng: random.Random) -> int:
    entities = getattr(session, "universe_players", None) or {}
    by_team: Dict[str, List[Dict[str, Any]]] = {}
    for row in entities.values():
        if not bool(row.get("active_roster", True)):
            continue
        by_team.setdefault(str(row.get("team_id") or ""), []).append(row)
    team_ids = [tid for tid, rows in by_team.items() if rows]
    if not team_ids:
        return 0
    rng.shuffle(team_ids)
    created = 0
    day = _u_current_meta(session)[0]
    for tid in team_ids[:16]:
        if rng.random() > 0.48:
            continue
        entity = rng.choice(by_team[tid])
        social = entity.get("social") or {}
        if str(social.get("style") or "quiet") == "quiet" and rng.random() > 0.35:
            continue
        if day - int(social.get("last_post_day", -99) or -99) < 3:
            continue
        kind = "community" if float((entity.get("life") or {}).get("community_connection", 0)) >= 60 else rng.choice(["support", "win", "loss"])
        _u_add_social_post(
            session,
            {
                "author_type": "player",
                "author_id": entity.get("player_id"),
                "author_name": entity.get("player_name"),
                "handle": social.get("handle"),
                "verified": True,
                "text": _u_player_post_text(entity, kind),
                "sentiment": kind,
                "likes": rng.randint(300, 9000),
                "reposts": rng.randint(20, 900),
                "replies": rng.randint(15, 500),
            },
        )
        social["last_post_day"] = day
        entity["social"] = social
        created += 1
        if created >= 5:
            break
    return created


def narrative_universe_v2_daily_pass(session: Any, calendar_idx: int, day_meta: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Advance player lives, concerns, room relationships, scenes, and social media."""
    _u_migrate_v2(session)
    if int(getattr(session, "_universe_last_daily_tick", -1) or -1) == int(calendar_idx):
        return {"already_processed": True, "calendar_day": int(calendar_idx), "generated_interactions": 0}
    entities = _u_sync_player_entities(session)
    season = int(getattr(session, "season_calendar_year", 2025) or 2025)
    local_rng = rng or random.Random(_u_seed("universe_daily", season, calendar_idx, getattr(session, "user_team_id", "")))
    for team_id, player in _u_all_players(session):
        entity = entities.get(str(getattr(player, "id", "") or ""))
        if entity:
            _u_tick_player_life(session, team_id, entity, local_rng)
    for team_id in (getattr(session, "team_by_id", None) or {}).keys():
        _u_rebuild_locker_room(session, str(team_id))
    expired = _u_expire_interactions(session)
    interactions = _u_generate_daily_interactions(session, local_rng)
    social_created = _u_generate_ambient_social(session, local_rng)
    promise_counts = _u_tick_promises(session)
    coverage_stats: Dict[str, Any] = {}
    try:
        from app.sim_engine.franchise.storyline_coverage import run_coverage_daily_pass  # noqa: WPS433

        coverage_stats = run_coverage_daily_pass(session, local_rng)
    except Exception:
        coverage_stats = {}
    try:
        from app.sim_engine.franchise.burner_engine import tick_burner_investigation_daily  # noqa: WPS433

        tick_burner_investigation_daily(session)
    except Exception:
        pass
    user_team_id = str(getattr(session, "user_team_id", "") or "")
    user_room = (getattr(session, "universe_locker_rooms", None) or {}).get(user_team_id) or {}
    snapshots = list(getattr(session, "universe_daily_snapshots", None) or [])
    snapshots.append(
        {
            "calendar_day": int(calendar_idx),
            "calendar_iso": str(day_meta.get("iso") or ""),
            "team_id": user_team_id,
            "culture": dict(user_room.get("culture") or {}),
            "pending_interactions": len(getattr(session, "universe_interaction_queue", None) or []),
        }
    )
    session.universe_daily_snapshots = snapshots[-120:]
    session._universe_last_daily_tick = int(calendar_idx)
    return {
        "already_processed": False,
        "calendar_day": int(calendar_idx),
        "player_entities": len(entities),
        "locker_rooms": len(getattr(session, "universe_locker_rooms", None) or {}),
        "generated_interactions": interactions,
        "expired_interactions": expired,
        "ambient_social_created": social_created,
        "promises_kept": promise_counts["kept"],
        "promises_broken": promise_counts["broken"],
        "pending_interactions": len(getattr(session, "universe_interaction_queue", None) or []),
        "coverage": coverage_stats,
    }


def _u_active_modifiers(session: Any, player_id: str) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for row in (getattr(session, "universe_attribute_modifiers", None) or {}).get(str(player_id), []) or []:
        if int(row.get("games_remaining", 0) or 0) <= 0:
            continue
        attribute = str(row.get("attribute") or "")
        totals[attribute] = round(totals.get(attribute, 0.0) + float(row.get("delta") or 0.0), 2)
    return totals


def build_universe_game_context(session: Any, team_id: str, opponent_id: str = "", game_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return sim-ready room and player modifiers.

    Add win_probability_delta to the team's pregame win probability and merge
    player_modifiers into the rating modifiers used for that game. This is the
    bridge that makes character and relationships affect actual results.
    """
    migrate_session_storyline_state(session)
    _u_sync_player_entities(session)
    room = _u_rebuild_locker_room(session, str(team_id))
    culture = room.get("culture") or {}
    entities = getattr(session, "universe_players", None) or {}
    player_modifiers: Dict[str, Dict[str, float]] = {}
    hidden_contributors: List[Dict[str, Any]] = []
    risks: List[Dict[str, Any]] = []
    glue_count = 0
    disruptor_tax = 0.0
    for player in _u_team_players(session, str(team_id)):
        player_id = str(getattr(player, "id", "") or "")
        entity = entities.get(player_id) or {}
        state = entity.get("state") or {}
        personality = entity.get("personality") or {}
        niches = _u_niche_ids(entity)
        morale = float(state.get("morale", 55))
        confidence = float(state.get("confidence", 55))
        focus = float(state.get("focus", 60))
        energy = float(state.get("energy", 70))
        character = float(personality.get("character", 55))
        role_satisfaction = float(state.get("role_satisfaction", 55))
        effort = (morale - 50) * 0.012 + (focus - 50) * 0.010 + (energy - 65) * 0.008
        composure = (confidence - 50) * 0.012 + (float(personality.get("resilience", 50)) - 50) * 0.009
        discipline = (character - 50) * 0.007 - max(0, float(personality.get("volatility", 50)) - 60) * 0.012
        passing = 0.0
        defensive_effort = 0.0
        penalty_risk = 0.0
        if character < 45 and role_satisfaction < 46:
            defensive_effort -= (45 - character) * 0.020 + (46 - role_satisfaction) * 0.014
            passing -= max(0, float(personality.get("ego", 50)) - 60) * 0.014
            penalty_risk += max(0, float(personality.get("volatility", 50)) - 58) * 0.018
            disruptor_tax += float(entity.get("disruption_risk", 0)) * 0.00012
            risks.append({"player_id": player_id, "player_name": entity.get("player_name"), "risk": "Frustration may become selfish decisions", "severity": round(float(entity.get("disruption_risk", 0)), 1)})
        for ability in entity.get("niche_abilities") or []:
            tier_mult = 0.72 + int(ability.get("tier", 1) or 1) * 0.28
            game = ability.get("game_effects") or {}
            effort += float(game.get("effort", 0)) * tier_mult
            effort = max(effort, float(game.get("effort_floor", -99)) * tier_mult)
            composure += float(game.get("composure", 0)) * tier_mult
            discipline += float(game.get("discipline", 0)) * tier_mult
            passing += float(game.get("passing", 0)) * tier_mult
            defensive_effort += float(game.get("defensive_effort", 0)) * tier_mult
            penalty_risk += float(game.get("penalty_risk", 0)) * tier_mult
        if any(n in niches for n in ("glue_guy", "mentor", "peacemaker", "culture_carrier", "quiet_professional")):
            glue_count += 1
        temporary = _u_active_modifiers(session, player_id)
        overall_equivalent = effort * 0.36 + composure * 0.22 + discipline * 0.14 + passing * 0.14 + defensive_effort * 0.14
        player_modifiers[player_id] = {
            "effort": round(effort, 3),
            "composure": round(composure, 3),
            "discipline": round(discipline, 3),
            "passing": round(passing, 3),
            "defensive_effort": round(defensive_effort, 3),
            "penalty_risk": round(max(0.0, penalty_risk), 3),
            "overall_equivalent": round(max(-2.5, min(2.5, overall_equivalent)), 3),
            **temporary,
        }
        if float(entity.get("overall", 99)) < 80 and float(entity.get("room_value", 0)) >= 70:
            hidden_contributors.append(
                {
                    "player_id": player_id,
                    "player_name": entity.get("player_name"),
                    "overall": entity.get("overall"),
                    "room_value": entity.get("room_value"),
                    "impact": "Raises the emotional and effort floor of teammates",
                }
            )
    unity = float(culture.get("unity", 50))
    tension = float(culture.get("tension", 35))
    confidence = float(culture.get("confidence", 50))
    leadership = float(culture.get("leadership", 50))
    accountability = float(culture.get("accountability", 50))
    win_delta = (
        (unity - 50) * 0.00042
        + (confidence - 50) * 0.00032
        + (leadership - 50) * 0.00024
        + (accountability - 50) * 0.00018
        - max(0, tension - 35) * 0.00045
        + min(0.010, glue_count * 0.0012)
        - disruptor_tax
    )
    win_delta = max(-0.075, min(0.075, win_delta))
    context_id = str((game_meta or {}).get("game_id") or f"gamectx_{team_id}_{_u_current_meta(session)[0]}")
    context = {
        "id": context_id,
        "team_id": str(team_id),
        "opponent_id": str(opponent_id or ""),
        "calendar_day": _u_current_meta(session)[0],
        "win_probability_delta": round(win_delta, 4),
        "team_modifiers": {
            "chemistry": round((unity - 50) / 25, 3),
            "composure": round((confidence + leadership - 100) / 50, 3),
            "discipline": round((accountability - tension) / 50, 3),
            "effort_floor": round((accountability + unity - 100) / 55, 3),
        },
        "player_modifiers": player_modifiers,
        "hidden_contributors": hidden_contributors,
        "character_risks": risks,
        "locker_room_snapshot": dict(culture),
        "explanation": f"Locker-room environment changes win probability by {win_delta * 100:+.1f} percentage points.",
    }
    contexts = dict(getattr(session, "universe_game_contexts", None) or {})
    contexts[context_id] = context
    session.universe_game_contexts = dict(list(contexts.items())[-60:])
    return context


def apply_universe_game_context(sim_inputs: Dict[str, Any], context: Dict[str, Any], *, side: str = "team") -> Dict[str, Any]:
    """Merge a universe context into a generic dict used by a game simulator."""
    result = dict(sim_inputs or {})
    prefix = f"{side}_" if side else ""
    probability_key = f"{prefix}win_probability"
    if probability_key in result:
        result[probability_key] = max(0.01, min(0.99, float(result[probability_key]) + float(context.get("win_probability_delta", 0))))
    elif "win_probability" in result:
        result["win_probability"] = max(0.01, min(0.99, float(result["win_probability"]) + float(context.get("win_probability_delta", 0))))
    result[f"{prefix}universe_team_modifiers"] = dict(context.get("team_modifiers") or {})
    existing_players = dict(result.get(f"{prefix}player_modifiers") or {})
    for player_id, changes in (context.get("player_modifiers") or {}).items():
        row = dict(existing_players.get(player_id) or {})
        for field, delta in changes.items():
            row[field] = round(float(row.get(field, 0) or 0) + float(delta), 3)
        existing_players[player_id] = row
    result[f"{prefix}player_modifiers"] = existing_players
    result[f"{prefix}universe_context_id"] = context.get("id")
    return result


def build_universe_matchup_context(session: Any, home_team_id: str, away_team_id: str, game_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build both sides together so probability changes remain complementary."""
    meta = dict(game_meta or {})
    base_game_id = str(meta.get("game_id") or f"matchup_{home_team_id}_{away_team_id}_{_u_current_meta(session)[0]}")
    home = build_universe_game_context(session, home_team_id, away_team_id, {**meta, "game_id": f"{base_game_id}:home"})
    away = build_universe_game_context(session, away_team_id, home_team_id, {**meta, "game_id": f"{base_game_id}:away"})
    net_home_delta = float(home.get("win_probability_delta", 0)) - float(away.get("win_probability_delta", 0))
    return {
        "id": base_game_id,
        "home_team_id": str(home_team_id),
        "away_team_id": str(away_team_id),
        "home": home,
        "away": away,
        "home_win_probability_delta": round(max(-0.10, min(0.10, net_home_delta)), 4),
        "explanation": f"The two locker rooms create a net home-team probability adjustment of {net_home_delta * 100:+.1f} percentage points.",
    }


def apply_universe_matchup_context(sim_inputs: Dict[str, Any], matchup: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a full matchup context to common home/away simulation inputs."""
    result = dict(sim_inputs or {})
    delta = float(matchup.get("home_win_probability_delta", 0) or 0)
    if "home_win_probability" in result:
        home_probability = max(0.01, min(0.99, float(result["home_win_probability"]) + delta))
        result["home_win_probability"] = home_probability
        if "away_win_probability" in result:
            result["away_win_probability"] = round(1.0 - home_probability, 6)
    result["home_universe_team_modifiers"] = dict((matchup.get("home") or {}).get("team_modifiers") or {})
    result["away_universe_team_modifiers"] = dict((matchup.get("away") or {}).get("team_modifiers") or {})
    result["home_player_modifiers"] = dict((matchup.get("home") or {}).get("player_modifiers") or {})
    result["away_player_modifiers"] = dict((matchup.get("away") or {}).get("player_modifiers") or {})
    result["universe_matchup_context_id"] = matchup.get("id")
    return result


def _u_advance_game_based_state(session: Any, team_id: str) -> None:
    modifiers = dict(getattr(session, "universe_attribute_modifiers", None) or {})
    for player in _u_team_players(session, team_id):
        player_id = str(getattr(player, "id", "") or "")
        rows = list(modifiers.get(player_id) or [])
        for row in rows:
            row["games_remaining"] = max(0, int(row.get("games_remaining", 0) or 0) - 1)
        modifiers[player_id] = [row for row in rows if int(row.get("games_remaining", 0) or 0) > 0]
    session.universe_attribute_modifiers = modifiers
    for promise in getattr(session, "universe_promises", None) or []:
        if str(promise.get("status") or "") == "active" and str(((getattr(session, "universe_players", None) or {}).get(str(promise.get("player_id") or "")) or {}).get("team_id") or "") == str(team_id):
            promise["games_remaining"] = max(0, int(promise.get("games_remaining", 0) or 0) - 1)


def apply_universe_postgame(session: Any, team_id: str, game_result: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Record how character affected a completed game and create follow-up beats."""
    migrate_session_storyline_state(session)
    _u_sync_player_entities(session)
    day, iso, season = _u_current_meta(session)
    local_rng = rng or random.Random(_u_seed("postgame", season, day, team_id, game_result.get("game_id", "")))
    won = bool(game_result.get("won"))
    if "result" in game_result:
        won = str(game_result.get("result") or "").upper() in ("W", "WIN")
    entities = getattr(session, "universe_players", None) or {}
    room = _u_rebuild_locker_room(session, str(team_id))
    notable: List[Dict[str, Any]] = []
    player_stats = game_result.get("player_stats") or {}
    for player in _u_team_players(session, str(team_id)):
        player_id = str(getattr(player, "id", "") or "")
        entity = entities.get(player_id) or {}
        personality = entity.get("personality") or {}
        state = entity.get("state") or {}
        stats = player_stats.get(player_id) or {}
        morale_delta = 2.2 if won else -2.0
        confidence_delta = 1.5 if won else -1.2
        if int(stats.get("points", 0) or 0) > 0 or int(stats.get("goals", 0) or 0) > 0:
            confidence_delta += 1.6
        if int(stats.get("penalty_minutes", stats.get("pim", 0)) or 0) >= 4:
            state["media_stress"] = _u_clip(float(state.get("media_stress", 25)) + 1.5)
        state["morale"] = _u_clip(float(state.get("morale", 55)) + morale_delta)
        state["confidence"] = _u_clip(float(state.get("confidence", 55)) + confidence_delta)
        low_character_risk = max(0.0, 48 - float(personality.get("character", 55))) * 0.006 + max(0.0, 48 - float(state.get("role_satisfaction", 55))) * 0.004 + max(0.0, float((room.get("culture") or {}).get("tension", 35)) - 55) * 0.002
        if not won and local_rng.random() < min(0.28, low_character_risk):
            state["coach_trust"] = _u_clip(float(state.get("coach_trust", 55)) - 4)
            event = {"id": f"uve_{uuid.uuid4().hex[:10]}", "kind": "selfish_game_moment", "team_id": str(team_id), "participants": [player_id], "player_id": player_id}
            summary = f"Video review identifies a low-effort recovery and a frustration penalty from {entity.get('player_name')}, turning a difficult game into a character concern."
            storyline = _u_record_storyline(session, event=event, headline=f"{entity.get('player_name')}'s frustration hurts the team", summary=summary, cause_type="LOW_CHARACTER_GAME_IMPACT", category="performance", heat=52, public=True)
            if storyline:
                _u_social_burst(session, storyline, event, local_rng)
            _u_apply_attribute_change(session, player_id, {"attribute": "discipline", "delta": -1, "permanent": False, "duration_games": 3, "reason": "Selfish postgame concern"}, event["id"])
            notable.append({"type": "low_character_cost", "player_id": player_id, "storyline": storyline})
        high_character = float(entity.get("room_value", 0)) >= 72 and float(entity.get("overall", 99)) < 80
        quiet_boxscore = int(stats.get("points", 0) or 0) == 0
        if high_character and quiet_boxscore and local_rng.random() < (0.08 if won else 0.035):
            event = {"id": f"uve_{uuid.uuid4().hex[:10]}", "kind": "winning_detail", "team_id": str(team_id), "participants": [player_id], "player_id": player_id}
            summary = f"Coaches highlight {entity.get('player_name')}'s bench communication, detail away from the puck, and calming shift as a hidden part of the result."
            storyline = _u_record_storyline(session, event=event, headline=f"The winning detail behind {entity.get('player_name')}'s quiet night", summary=summary, cause_type="HIGH_CHARACTER_IMPACT", category="locker_room", heat=28, public=local_rng.random() < 0.55)
            state["belonging"] = _u_clip(float(state.get("belonging", 55)) + 2)
            notable.append({"type": "high_character_value", "player_id": player_id, "storyline": storyline})
    culture = room.get("culture") or {}
    culture["confidence"] = _u_clip(float(culture.get("confidence", 50)) + (3 if won else -3))
    culture["unity"] = _u_clip(float(culture.get("unity", 50)) + (1.2 if won else -0.8))
    culture["tension"] = _u_clip(float(culture.get("tension", 35)) + (-1.0 if won else 1.4))
    _u_advance_game_based_state(session, str(team_id))
    return {"team_id": str(team_id), "won": won, "notable_character_events": notable, "locker_room": dict(culture)}


def _u_public_entity(entity: Dict[str, Any], *, include_private: bool) -> Dict[str, Any]:
    result = dict(entity)
    if not include_private:
        result.pop("life", None)
        result.pop("concerns", None)
        result.pop("memories", None)
    return result


def team_locker_room_profile(session: Any, team_id: str) -> Dict[str, Any]:
    migrate_session_storyline_state(session)
    _u_sync_player_entities(session)
    room = _u_rebuild_locker_room(session, str(team_id))
    entities = getattr(session, "universe_players", None) or {}
    user_team = str(team_id) == str(getattr(session, "user_team_id", "") or "")
    roster = [
        _u_public_entity(entities.get(str(getattr(player, "id", "") or "")) or {}, include_private=user_team)
        for player in _u_team_players(session, str(team_id))
    ]
    return {**room, "players": roster}


def player_universe_profile(session: Any, player_id: str) -> Dict[str, Any]:
    migrate_session_storyline_state(session)
    _u_sync_player_entities(session)
    entity = dict((getattr(session, "universe_players", None) or {}).get(str(player_id)) or {})
    team_id = str(entity.get("team_id") or "")
    room = _u_rebuild_locker_room(session, team_id) if team_id else {}
    relationships: List[Dict[str, Any]] = []
    entities = getattr(session, "universe_players", None) or {}
    for rel in (room.get("relationships") or {}).values():
        ids = list(rel.get("player_ids") or [])
        if str(player_id) not in ids:
            continue
        other_id = ids[0] if ids[1] == str(player_id) else ids[1]
        relationships.append({**rel, "other_player_id": other_id, "other_player_name": (entities.get(other_id) or {}).get("player_name")})
    relationships.sort(key=lambda row: (-float(row.get("tension", 0)), -float(row.get("chemistry", 0))))
    reporter_rows = [row for row in (getattr(session, "universe_reporter_relationships", None) or {}).values() if str(row.get("player_id") or "") == str(player_id)]
    promises = [row for row in (getattr(session, "universe_promises", None) or []) if str(row.get("player_id") or "") == str(player_id)]
    user_team = str(getattr(session, "user_team_id", "") or "")
    include_private = str(entity.get("team_id") or "") == user_team
    player_obj = _player_from_roster(session, str(player_id))
    human_dossier = build_human_dossier_payload(
        session,
        entity,
        player_obj,
        include_private=include_private,
    )
    return {
        **entity,
        "relationships": relationships,
        "reporter_relationships": reporter_rows,
        "active_modifiers": _u_active_modifiers(session, str(player_id)),
        "promises": promises,
        "human_dossier": human_dossier,
    }


def _u_ingest_existing_storyline(session: Any, storyline: Dict[str, Any]) -> None:
    player_id = str(storyline.get("player_id") or "")
    if not player_id:
        return
    entity = (getattr(session, "universe_players", None) or {}).get(player_id)
    if not entity:
        player = _player_from_roster(session, player_id)
        if player is None:
            return
        team_id = str(storyline.get("team_id") or "")
        entity = _u_create_player_entity(session, team_id, player)
        session.universe_players[player_id] = entity
    day, iso, _ = _u_current_meta(session)
    heat = int(storyline.get("heat") or 30)
    tone = str(storyline.get("tone") or "neutral").lower()
    emotional = -max(1, heat / 15) if tone == "negative" else max(1, heat / 20) if tone == "positive" else 0
    _u_add_memory(entity, kind="storyline", summary=str(storyline.get("headline") or "Media story"), day=day, iso=iso, emotional_delta=emotional, public=True)
    if tone == "negative":
        _u_apply_profile_delta(entity, "state.media_stress", min(7, heat / 16))
    elif tone == "positive":
        _u_apply_profile_delta(entity, "state.confidence", min(4, heat / 24))
    reporter_id = str(storyline.get("reporter_id") or "")
    if reporter_id:
        rel = _u_reporter_relationship(session, reporter_id, player_id)
        rel["interview_count"] = int(rel.get("interview_count", 0) or 0) + 1
        if tone == "negative":
            rel["friction"] = _u_clip(float(rel.get("friction", 25)) + heat * 0.045)
            rel["trust"] = _u_clip(float(rel.get("trust", 50)) - heat * 0.025)


def build_narrative_universe_v2_payload(session: Any) -> Dict[str, Any]:
    migrate_session_storyline_state(session)
    entities = _u_sync_player_entities(session)
    user_team_id = str(getattr(session, "user_team_id", "") or "")
    room = _u_rebuild_locker_room(session, user_team_id) if user_team_id else {}
    pending = [row for row in (getattr(session, "universe_interaction_queue", None) or []) if str(row.get("status") or "") == "pending"]
    active_promises = [row for row in (getattr(session, "universe_promises", None) or []) if str(row.get("status") or "") == "active"]
    user_players = [row for row in entities.values() if str(row.get("team_id") or "") == user_team_id and bool(row.get("active_roster", True))]
    user_players.sort(key=lambda row: (-float(row.get("room_value", 0)), str(row.get("player_name") or "")))
    feed = list(getattr(session, "social_posts", None) or [])[-120:]
    reddit = list(getattr(session, "reddit_threads", None) or [])[-100:]
    burner: Dict[str, Any] = {}
    try:
        from app.sim_engine.franchise.burner_engine import burner_state_payload  # noqa: WPS433

        burner = burner_state_payload(session)
    except Exception:
        burner = dict(getattr(session, "gm_burner_account", None) or {})
    extra: Dict[str, Any] = {}
    try:
        from app.sim_engine.franchise.storyline_coverage import coverage_payload_fields  # noqa: WPS433

        extra = coverage_payload_fields(session)
    except Exception:
        extra = {}
    return {
        "engine_version": UNIVERSE_ENGINE_VERSION,
        "players": user_players,
        "player_entity_count": len(entities),
        "locker_room": room,
        "locker_rooms": dict(getattr(session, "universe_locker_rooms", None) or {}),
        "interaction_queue": pending,
        "recent_interactions": list(getattr(session, "universe_interactions", None) or [])[-30:],
        "reporter_relationships": dict(getattr(session, "universe_reporter_relationships", None) or {}),
        "active_promises": active_promises,
        "attribute_modifiers": dict(getattr(session, "universe_attribute_modifiers", None) or {}),
        "recent_universe_events": list(getattr(session, "universe_event_log", None) or [])[-60:],
        "twitter_feed": feed,
        "social_feed": feed,
        "reddit_threads": reddit,
        "reddit_engagement_pulse": list(getattr(session, "reddit_engagement_pulse", None) or [])[-20:],
        "gm_burner_account": burner,
        "daily_snapshots": list(getattr(session, "universe_daily_snapshots", None) or [])[-60:],
        "hallmark_panels": {
            "room_pulse": dict(room.get("culture") or {}),
            "unheralded_leaders": [row for row in user_players if float(row.get("overall", 99)) < 80 and float(row.get("room_value", 0)) >= 70][:6],
            "character_risks": [row for row in user_players if float(row.get("disruption_risk", 0)) >= 45][:6],
            "decisions_waiting": len(pending),
            "active_promises": len(active_promises),
        },
        **extra,
    }


# ---------------------------------------------------------------------------
# Backward-compatible public wrappers. Existing callers receive their original
# fields plus Universe V2 fields; no route or save-file rewrite is required.
# ---------------------------------------------------------------------------

def migrate_session_storyline_state(session: Any) -> None:
    _UNIVERSE_LEGACY_MIGRATE(session)
    _u_migrate_v2(session)


def narrative_universe_daily_pass(session: Any, calendar_idx: int, day_meta: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, Any]:
    legacy = _UNIVERSE_LEGACY_DAILY_PASS(session, calendar_idx, day_meta, rng)
    expanded = narrative_universe_v2_daily_pass(session, calendar_idx, day_meta, rng)
    return {**dict(legacy or {}), "universe_v2": expanded}


def enrich_storyline_for_narrative_universe(session: Any, event: Dict[str, Any]) -> Dict[str, Any]:
    migrate_session_storyline_state(session)
    enriched = _UNIVERSE_LEGACY_ENRICH(session, event)
    _u_ingest_existing_storyline(session, enriched)
    return enriched


def build_narrative_universe_payload(session: Any) -> Dict[str, Any]:
    legacy = _UNIVERSE_LEGACY_PAYLOAD(session)
    expanded = build_narrative_universe_v2_payload(session)
    return {**dict(legacy or {}), **expanded, "legacy_narrative": legacy}


def player_narrative_profile(session: Any, player_id: str) -> Dict[str, Any]:
    legacy = _UNIVERSE_LEGACY_PLAYER_PROFILE(session, player_id)
    expanded = player_universe_profile(session, player_id)
    return {**dict(legacy or {}), **expanded, "legacy_media_profile": legacy}


# ========================= STORYLINE UNIVERSE V3 PATCH =========================
# Paste this block at the VERY BOTTOM of the existing storyline module.
# Later Python definitions override the V2 implementations without deleting old code.

UNIVERSE_ENGINE_VERSION = 3
UNIVERSE_MAJOR_EVENTS_MIN = 3
UNIVERSE_MAJOR_EVENTS_MAX = 5
UNIVERSE_POTENTIAL_SEASON_CAP = 2.5
UNIVERSE_READINESS_MIN = -25.0
UNIVERSE_READINESS_MAX = 6.0

_V3_CAUSE_TYPES = frozenset(
    {
        "MINOR_LIFE_EVENT",
        "POSITIVE_LIFE_EVENT",
        "PLAYER_MEETING_REQUEST",
        "REQUEST_MORE_ICE",
        "REQUEST_PP_TIME",
        "REQUEST_STARTING_ROLE",
        "CONTRACT_CLARITY_REQUEST",
        "DEVELOPMENT_MEETING",
        "WINNING_CONCERN",
        "TRADE_PROPOSAL_EXPOSURE",
        "PRIVATE_TRADE_DEMAND",
        "TRADE_DEMAND",
        "PLAYER_ARRESTED",
        "LEAGUE_SUSPENSION",
        "PLAYER_BANNED",
        "MAJOR_PUBLIC_ALTERCATION",
        "GAMBLING_VIOLATION",
        "PLAYER_DEATH",
        "ILLEGAL_TEAM_WORKOUTS",
        "UNDER_TABLE_PAYMENTS",
        "CAP_CIRCUMVENTION",
        "EXECUTIVE_MISCONDUCT",
    }
)
STORYLINE_CAUSE_TYPES = frozenset(set(STORYLINE_CAUSE_TYPES) | set(_V3_CAUSE_TYPES))
_CLAIM_CAUSE_TYPES = frozenset(set(_CLAIM_CAUSE_TYPES) | {"PRIVATE_TRADE_DEMAND", "TRADE_DEMAND", "TRADE_PROPOSAL_EXPOSURE"})
_FACT_CAUSE_TYPES = frozenset(
    set(_FACT_CAUSE_TYPES)
    | {
        "PLAYER_ARRESTED",
        "LEAGUE_SUSPENSION",
        "PLAYER_BANNED",
        "MAJOR_PUBLIC_ALTERCATION",
        "GAMBLING_VIOLATION",
        "PLAYER_DEATH",
        "ILLEGAL_TEAM_WORKOUTS",
        "UNDER_TABLE_PAYMENTS",
        "CAP_CIRCUMVENTION",
        "EXECUTIVE_MISCONDUCT",
    }
)

# 17 negative + 17 positive small life events. These are minor human events,
# not major conduct incidents. They create tiny but real readiness/stat effects.
MINOR_NEGATIVE_LIFE_EVENTS: List[Dict[str, Any]] = [
    {"id": "headache", "headline": "{name} wakes up feeling off", "summary": "A headache makes today's preparation less comfortable than usual.", "profile": {"state.focus": -2.5, "state.energy": -1.0}, "ovr": -0.35, "attrs": {"offensive_awareness": -0.25}, "days": 2, "heat": 8},
    {"id": "breakup", "headline": "Personal change weighs on {name}", "summary": "A recent breakup creates a difficult stretch away from the rink.", "profile": {"state.morale": -4.0, "state.focus": -3.0, "state.personal_stress": 5.0}, "character": -0.3, "ovr": -1.0, "attrs": {"passing": -0.4, "offensive_awareness": -0.4}, "days": 12, "heat": 18, "requires_partnered": True},
    {"id": "poor_sleep", "headline": "Poor sleep disrupts {name}'s routine", "summary": "A bad night's sleep leaves the player slightly drained.", "profile": {"state.energy": -3.0, "state.focus": -2.0}, "ovr": -0.45, "attrs": {"stamina": -0.4}, "days": 2, "heat": 7},
    {"id": "partner_argument", "headline": "Home tension follows {name} to the rink", "summary": "A minor argument at home adds distraction to the day.", "profile": {"state.morale": -2.0, "state.personal_stress": 3.0}, "ovr": -0.4, "days": 4, "heat": 9, "requires_partnered": True},
    {"id": "family_difficulty", "headline": "Family concern occupies {name}", "summary": "A family member is going through a difficult period and the player is distracted.", "profile": {"state.personal_stress": 5.0, "state.focus": -2.5}, "ovr": -0.7, "days": 8, "heat": 11, "requires_dependents": True, "leave": {"type": "family_leave", "days_min": 2, "days_max": 5, "chance": 0.22, "reason_public": "Unavailable — Family Leave"}},
    {"id": "child_relocation", "headline": "Relocation remains difficult for {name}'s family", "summary": "A child is having trouble adjusting to the new city.", "profile": {"state.personal_stress": 4.0, "state.belonging": -2.0}, "ovr": -0.55, "days": 10, "heat": 10, "requires_dependents": True},
    {"id": "homesick", "headline": "{name} admits to feeling homesick", "summary": "Distance from home is beginning to affect the player's routine.", "profile": {"state.morale": -2.5, "state.belonging": -2.0}, "ovr": -0.45, "days": 8, "heat": 10},
    {"id": "pet_illness", "headline": "Small home-life worry follows {name}", "summary": "A sick pet creates a minor distraction away from hockey.", "profile": {"state.personal_stress": 2.5, "state.focus": -1.0}, "ovr": -0.25, "days": 4, "heat": 6},
    {"id": "home_repairs", "headline": "Home problems add stress for {name}", "summary": "Unexpected home repairs disrupt the player's routine.", "profile": {"state.personal_stress": 2.0, "state.energy": -1.0}, "ovr": -0.25, "days": 4, "heat": 6},
    {"id": "lost_luggage", "headline": "Road-trip inconvenience frustrates {name}", "summary": "Lost luggage makes an already busy travel day more irritating.", "profile": {"state.focus": -1.5, "state.energy": -1.0}, "ovr": -0.25, "days": 2, "heat": 5},
    {"id": "unexpected_expense", "headline": "Unexpected expense bothers {name}", "summary": "A frustrating personal expense creates a small amount of off-ice stress.", "profile": {"state.personal_stress": 2.0, "state.morale": -1.0}, "ovr": -0.2, "days": 5, "heat": 5},
    {"id": "social_media_noise", "headline": "Online criticism gets under {name}'s skin", "summary": "A wave of negative social-media replies becomes a distraction.", "profile": {"state.media_stress": 3.0, "state.confidence": -1.5}, "character": -0.15, "ovr": -0.35, "attrs": {"discipline": -0.2}, "days": 5, "heat": 13},
    {"id": "family_travel_issue", "headline": "Family travel issue distracts {name}", "summary": "Travel plans involving family fall apart at an inconvenient time.", "profile": {"state.personal_stress": 2.0, "state.focus": -1.0}, "ovr": -0.2, "days": 3, "heat": 5},
    {"id": "car_trouble", "headline": "A stressful morning starts badly for {name}", "summary": "Car trouble throws off the player's normal preparation routine.", "profile": {"state.focus": -1.5, "state.energy": -0.5}, "ovr": -0.2, "days": 1, "heat": 4},
    {"id": "missed_family_event", "headline": "Schedule conflict weighs on {name}", "summary": "The NHL schedule forces the player to miss an important family event.", "profile": {"state.morale": -2.0, "state.personal_stress": 2.5}, "ovr": -0.35, "days": 5, "heat": 7},
    {"id": "agent_disagreement", "headline": "{name} and his representation are not fully aligned", "summary": "A small disagreement with the player's agent creates uncertainty.", "profile": {"state.focus": -1.0, "state.personal_stress": 2.0}, "ovr": -0.25, "days": 5, "heat": 12},
    {"id": "privacy_intrusion", "headline": "Unwanted attention frustrates {name}", "summary": "The player feels that public attention has crossed into his private life.", "profile": {"state.media_stress": 4.0, "state.morale": -1.5}, "character": -0.2, "ovr": -0.45, "days": 7, "heat": 15},
]

MINOR_POSITIVE_LIFE_EVENTS: List[Dict[str, Any]] = [
    {"id": "engagement", "headline": "{name} celebrates an engagement", "summary": "A major positive step in the player's personal life lifts the mood around him.", "profile": {"state.morale": 5.0, "state.belonging": 2.0, "state.personal_stress": -2.0}, "character": 0.3, "ovr": 0.5, "attrs": {"composure": 0.3}, "days": 10, "heat": 24, "public_chance": 0.65, "potential_chance": 0.16, "potential": 0.2, "requires_partnered": True},
    {"id": "new_child", "headline": "{name}'s family welcomes a new child", "summary": "The player is energized by a new addition to the family, even as sleep becomes harder to manage.", "profile": {"state.morale": 6.0, "state.belonging": 3.0, "state.personal_stress": 1.5, "state.energy": -1.5}, "character": 0.5, "ovr": 0.35, "attrs": {"stamina": -0.2, "composure": 0.4}, "days": 12, "heat": 28, "public_chance": 0.75, "potential_chance": 0.18, "potential": 0.25, "requires_partnered": True},
    {"id": "wedding", "headline": "{name} celebrates his wedding", "summary": "A joyful family milestone gives the player a noticeable emotional lift.", "profile": {"state.morale": 6.0, "state.belonging": 3.0, "state.personal_stress": -3.0}, "character": 0.4, "ovr": 0.55, "days": 12, "heat": 26, "public_chance": 0.7, "potential_chance": 0.14, "potential": 0.2, "requires_partnered": True},
    {"id": "pregnancy_news", "headline": "Good family news lifts {name}", "summary": "The player's family shares exciting news and his mood noticeably improves.", "profile": {"state.morale": 4.0, "state.belonging": 2.0}, "character": 0.25, "ovr": 0.3, "days": 8, "heat": 18, "public_chance": 0.45, "potential_chance": 0.12, "potential": 0.15, "requires_partnered": True},
    {"id": "family_settles", "headline": "{name}'s family settling comfortably into the city", "summary": "Home life is becoming more stable and the player feels increasingly rooted.", "profile": {"state.belonging": 4.0, "state.personal_stress": -3.0, "state.focus": 1.5}, "character": 0.2, "ovr": 0.3, "days": 12, "heat": 12, "potential_chance": 0.18, "potential": 0.2},
    {"id": "parents_visit", "headline": "Family visit gives {name} a lift", "summary": "A visit from family makes the homestand feel a little easier.", "profile": {"state.morale": 2.5, "state.personal_stress": -1.5}, "ovr": 0.2, "days": 4, "heat": 7},
    {"id": "buys_home", "headline": "{name} puts down roots in the city", "summary": "Buying a home strengthens the player's sense of stability and belonging.", "profile": {"state.belonging": 4.0, "state.personal_stress": -2.0}, "character": 0.25, "ovr": 0.25, "days": 10, "heat": 16, "public_chance": 0.35, "potential_chance": 0.12, "potential": 0.15},
    {"id": "charity_success", "headline": "Community event becomes a meaningful day for {name}", "summary": "A charity event goes exceptionally well and strengthens the player's connection to the city.", "profile": {"state.morale": 3.0, "state.belonging": 4.0, "state.confidence": 1.0}, "character": 0.6, "ovr": 0.25, "days": 7, "heat": 22, "public_chance": 0.8, "potential_chance": 0.10, "potential": 0.15},
    {"id": "fan_moment", "headline": "A fan interaction sticks with {name}", "summary": "A meaningful interaction with a supporter gives the player perspective and energy.", "profile": {"state.morale": 2.0, "state.belonging": 2.0}, "character": 0.25, "ovr": 0.15, "days": 4, "heat": 10, "public_chance": 0.3},
    {"id": "hometown_honor", "headline": "Hometown recognition means a lot to {name}", "summary": "Recognition from home gives the player an extra dose of confidence.", "profile": {"state.confidence": 3.0, "state.morale": 2.0}, "ovr": 0.3, "attrs": {"composure": 0.25}, "days": 6, "heat": 18, "public_chance": 0.6},
    {"id": "old_friend_visit", "headline": "Familiar face helps {name} reset", "summary": "Time with an old friend gives the player a useful mental reset.", "profile": {"state.personal_stress": -2.5, "state.focus": 1.5}, "ovr": 0.2, "days": 4, "heat": 6},
    {"id": "family_break", "headline": "Family time leaves {name} refreshed", "summary": "A successful short break with family restores energy and focus.", "profile": {"state.energy": 2.0, "state.focus": 2.0, "state.morale": 1.5}, "ovr": 0.35, "attrs": {"stamina": 0.3}, "days": 5, "heat": 7},
    {"id": "endorsement", "headline": "New opportunity boosts {name}'s profile", "summary": "A new endorsement opportunity gives the player a small confidence lift.", "profile": {"state.confidence": 2.0, "state.morale": 1.0}, "ovr": 0.15, "days": 4, "heat": 16, "public_chance": 0.75},
    {"id": "youth_hockey", "headline": "{name} connects with local youth hockey", "summary": "Time spent with young players strengthens the player's connection to the community.", "profile": {"state.belonging": 3.0, "state.morale": 2.0}, "character": 0.5, "ovr": 0.15, "days": 6, "heat": 18, "public_chance": 0.7, "potential_chance": 0.08, "potential": 0.1},
    {"id": "partner_settles", "headline": "Home life becoming more comfortable for {name}", "summary": "The player's partner is settling into the city, reducing background stress.", "profile": {"state.personal_stress": -3.0, "state.belonging": 3.0}, "ovr": 0.25, "days": 8, "heat": 8, "requires_partnered": True},
    {"id": "mentor_connection", "headline": "Veteran advice clicks for {name}", "summary": "A strong conversation with a veteran helps the player see his situation more clearly.", "profile": {"state.confidence": 2.0, "state.focus": 2.0}, "character": 0.2, "ovr": 0.25, "attrs": {"offensive_awareness": 0.25}, "days": 6, "heat": 10, "potential_chance": 0.22, "potential": 0.3},
    {"id": "community_award", "headline": "{name} recognized for work away from the rink", "summary": "A community award reinforces the player's positive relationship with the city.", "profile": {"state.morale": 3.0, "state.belonging": 4.0}, "character": 0.75, "ovr": 0.2, "days": 7, "heat": 24, "public_chance": 0.9, "potential_chance": 0.08, "potential": 0.1},
]


# ---------------------------------------------------------------------------
# Human Universe V3.1 — persistent life, pressure tiers, dossier, agent memory
# ---------------------------------------------------------------------------

def _u_tier_label(value: float, *, invert: bool = False) -> str:
    v = float(value or 50)
    if invert:
        v = 100.0 - v
    if v >= 92:
        return "Elite"
    if v >= 82:
        return "Very High"
    if v >= 72:
        return "High"
    if v >= 62:
        return "Above Average"
    if v >= 48:
        return "Average"
    if v >= 38:
        return "Below Average"
    if v >= 28:
        return "Low"
    return "Very Low"


def _u_pressure_tier(pressure: float) -> int:
    p = float(pressure or 0)
    if p < 18:
        return 0
    if p < 35:
        return 1
    if p < 52:
        return 2
    if p < 68:
        return 3
    return 4


def _u_pressure_tier_label(tier: int) -> str:
    return {
        0: "Settled",
        1: "Uneasy",
        2: "Frustrated",
        3: "Relationship Breaking",
        4: "Crisis",
    }.get(int(tier), "Settled")


def _u_family_id(player_id: str, slot: str) -> str:
    return f"fam_{player_id}_{slot}"


def _u_migrate_entity_life(entity: Dict[str, Any], rng: random.Random) -> None:
    """Expand legacy flat life dict once — never reroll on GET."""
    life = entity.setdefault("life", {})
    if life.get("life_v31"):
        return
    player_id = str(entity.get("player_id") or "")
    rel_status = str(life.get("relationship_status") or "single")
    dependents = int(life.get("dependents") or 0)
    personality = entity.get("personality") or {}

    if rel_status in ("partnered", "family_household", "engaged", "married") and not isinstance(life.get("partner"), dict):
        life["partner"] = {
            "id": _u_family_id(player_id, "partner"),
            "name": str(life.get("partner_name") or "Partner"),
            "relationship_strength": _u_clip(68 + rng.uniform(-8, 12)),
            "relocation_satisfaction": _u_clip(float(life.get("city_satisfaction") or 62)),
            "city_satisfaction": _u_clip(float(life.get("city_satisfaction") or 62)),
            "relocation_tolerance": _u_clip(55 + rng.uniform(-15, 20)),
            "status": "married" if rel_status == "married" else "engaged" if rel_status == "engaged" else "partner",
        }

    if not isinstance(life.get("children"), list):
        children: List[Dict[str, Any]] = []
        for i in range(dependents):
            children.append(
                {
                    "id": _u_family_id(player_id, f"child_{i}"),
                    "age_bracket": rng.choice(["infant", "toddler", "school_age", "teen"]),
                    "city_adjustment": _u_clip(55 + rng.uniform(-12, 15)),
                    "school_stability": _u_clip(60 + rng.uniform(-10, 10)),
                    "health_stress": _u_clip(rng.uniform(0, 8)),
                }
            )
        life["children"] = children

    life.setdefault(
        "parents",
        {
            "proximity_score": _u_clip(rng.uniform(25, 75)),
            "importance": _u_clip(float(personality.get("family_orientation", 55))),
            "support_strength": _u_clip(55 + rng.uniform(-15, 20)),
        },
    )
    life.setdefault("city_attachment", _u_clip(float(life.get("community_connection") or 40)))
    life.setdefault("home_owned", bool(life.get("home_owned", False)))
    life.setdefault("housing", "owned" if life.get("home_owned") else "rented")

    if not isinstance(life.get("friends"), list):
        life["friends"] = [
            {
                "id": f"fr_{player_id}_home",
                "name": "Hometown friend",
                "kind": "hometown",
                "closeness": _u_clip(48 + rng.uniform(0, 32)),
                "influence": "positive/supportive",
                "stage": "friend",
                "last_contact_day": -99,
            },
            {
                "id": f"fr_{player_id}_local",
                "name": "Local friend",
                "kind": "local",
                "closeness": _u_clip(35 + rng.uniform(0, 28)),
                "influence": "neutral",
                "stage": "acquaintance",
                "last_contact_day": -99,
            },
        ]

    entity.setdefault(
        "mental_wellbeing",
        {
            "state": "stable",
            "wellbeing_score": _u_clip(72 - float((entity.get("state") or {}).get("personal_stress", 25)) * 0.25),
            "last_shift_day": -1,
            "private": True,
        },
    )
    entity.setdefault("human_pressure", {"score": 0.0, "tier": 0, "tier_label": "Settled", "drivers": []})
    life["life_v31"] = True


def _u_memory_weight(entity: Dict[str, Any], kind: str) -> float:
    weights = {
        "betrayal": 1.0,
        "broken_promise": 0.95,
        "trade_exposure": 0.85,
        "gm_support": 0.9,
        "family_accommodation": 0.88,
        "gratitude": 0.75,
        "embarrassment": 0.65,
        "support": 0.7,
        "playoff_success": 0.55,
    }
    total = 0.0
    for mem in list(entity.get("memories") or [])[-24:]:
        mk = str(mem.get("kind") or "")
        if mk == kind or kind in mk:
            age_days = max(1, abs(int(mem.get("calendar_day") or 0)))
            decay = 0.985 ** min(400, age_days)
            total += float(mem.get("emotional_delta") or 0) * decay * weights.get(kind, 0.5)
    return total


def _u_agent_org_ledger(session: Any) -> Dict[str, Dict[str, Any]]:
    book = getattr(session, "universe_agent_org_relationships", None)
    if not isinstance(book, dict):
        book = {}
        session.universe_agent_org_relationships = book
    return book


def _u_agent_org_trust(session: Any, agent_id: str, team_id: str) -> float:
    if not agent_id or not team_id:
        return 55.0
    key = f"{agent_id}:{team_id}"
    row = _u_agent_org_ledger(session).get(key) or {}
    return float(row.get("trust") or row.get("fairness") or 55.0)


def _u_adjust_agent_org_trust(session: Any, agent_id: str, team_id: str, delta: float, reason: str = "") -> None:
    if not agent_id or not team_id:
        return
    book = _u_agent_org_ledger(session)
    key = f"{agent_id}:{team_id}"
    row = dict(book.get(key) or {})
    before = float(row.get("trust") or 55.0)
    after = _u_clip(before + float(delta))
    row.update({"agent_id": agent_id, "team_id": team_id, "trust": after, "fairness": after, "last_reason": reason})
    history = list(row.get("history") or [])
    history.append({"delta": round(delta, 2), "reason": reason, "trust_after": after})
    row["history"] = history[-16:]
    book[key] = row


def _u_compute_human_pressure(
    entity: Dict[str, Any],
    *,
    room: Optional[Dict[str, Any]] = None,
    agent_org_trust: float = 55.0,
) -> Dict[str, Any]:
    """Competing forces — no single variable triggers escalation alone."""
    state = entity.get("state") or {}
    life = entity.get("life") or {}
    personality = entity.get("personality") or {}
    concerns = entity.get("concerns") or {}

    role_sat = float(state.get("role_satisfaction", 58))
    win_sat = float((concerns.get("winning") or {}).get("satisfaction", 55))
    gm_trust = float(state.get("gm_trust", 65))
    coach_trust = float(state.get("coach_trust", 60))
    belonging = float(state.get("belonging", 60))
    personal_stress = float(state.get("personal_stress", 28))
    media_stress = float(state.get("media_stress", 28))

    role_pressure = max(0.0, (55 - role_sat) * 0.55)
    if float(personality.get("competitiveness", 50)) >= 75 and win_sat >= 62:
        role_pressure *= 0.55
    if float(personality.get("loyalty", 50)) >= 72 and gm_trust >= 58:
        role_pressure *= 0.68

    losing_pressure = max(0.0, (50 - win_sat) * 0.42) * (float(personality.get("competitiveness", 50)) / 100.0)
    trust_pressure = max(0.0, (58 - gm_trust) * 0.35 + (55 - coach_trust) * 0.22)
    life_pressure = max(0.0, personal_stress - 35) * 0.38 + max(0.0, float(life.get("relocation_strain") or 0) - 20) * 0.28
    city_pressure = max(0.0, (55 - float(life.get("city_attachment") or 40)) * 0.22)
    social_pressure = max(0.0, (52 - belonging) * 0.30)

    partner = life.get("partner") if isinstance(life.get("partner"), dict) else {}
    if partner:
        city_pressure += max(0.0, (50 - float(partner.get("city_satisfaction") or 55)) * 0.18)

    memory_pressure = -_u_memory_weight(entity, "gm_support") * 0.12 + _u_memory_weight(entity, "betrayal") * 0.15
    agent_pressure = max(0.0, (55 - agent_org_trust) * 0.12)

    volatility_amp = 1.0 + max(0.0, float(personality.get("volatility", 40)) - 65) * 0.008
    resilience_damp = 1.0 - max(0.0, float(personality.get("resilience", 55)) - 70) * 0.006

    raw = (
        role_pressure * 1.15
        + losing_pressure
        + trust_pressure
        + life_pressure
        + city_pressure
        + social_pressure
        + media_stress * 0.08
        + memory_pressure
        + agent_pressure
    ) * volatility_amp * resilience_damp

    score = _u_clip(raw, 0.0, 100.0)
    tier = _u_pressure_tier(score)
    drivers = []
    for label, val in (
        ("Role", role_pressure),
        ("Winning", losing_pressure),
        ("Management trust", trust_pressure),
        ("Home life", life_pressure),
        ("City fit", city_pressure),
        ("Belonging", social_pressure),
    ):
        if val >= 8:
            drivers.append({"label": label, "pressure": round(val, 1)})
    drivers.sort(key=lambda row: -float(row["pressure"]))
    return {"score": round(score, 1), "tier": tier, "tier_label": _u_pressure_tier_label(tier), "drivers": drivers[:4]}


def _u_tick_mental_wellbeing(entity: Dict[str, Any], day: int) -> None:
    """Mental wellbeing separate from character/morality."""
    state = entity.get("state") or {}
    life = entity.get("life") or {}
    personality = entity.get("personality") or {}
    mw = entity.setdefault("mental_wellbeing", {"state": "stable", "wellbeing_score": 70.0})

    stress = float(state.get("personal_stress", 28))
    sleep = float(life.get("sleep_quality", 68))
    home = float(life.get("home_stability", 65))
    social = float(state.get("belonging", 60))
    media = float(state.get("media_stress", 28))
    resilience = float(personality.get("resilience", 55))

    target = _u_clip(78 - stress * 0.35 - max(0, 62 - sleep) * 0.18 - max(0, 60 - home) * 0.12 - media * 0.08 + social * 0.06 + resilience * 0.08)
    prev = float(mw.get("wellbeing_score") or target)
    score = _u_clip(prev * 0.88 + target * 0.12)

    if score >= 68:
        label = "stable"
    elif score >= 55:
        label = "strained"
    elif score >= 42:
        label = "overloaded"
    elif score >= 30:
        label = "withdrawn"
    elif score >= 18:
        label = "acute_concern"
    else:
        label = "recovering" if prev < score else "acute_concern"

    mw["wellbeing_score"] = round(score, 1)
    mw["state"] = label
    mw["last_shift_day"] = day
    mw["private"] = True


def _u_mutate_life_from_event(entity: Dict[str, Any], spec: Dict[str, Any], rng: random.Random, day: int) -> None:
    """Life events must change persistent state — not just temporary deltas."""
    life = entity.setdefault("life", {})
    event_id = str(spec.get("id") or "")
    player_id = str(entity.get("player_id") or "")
    state = entity.setdefault("state", {})

    if event_id == "breakup":
        life["relationship_status"] = "single"
        life.pop("partner", None)
        life["home_stability"] = _u_clip(float(life.get("home_stability", 65)) - 12)
        state["personal_stress"] = _u_clip(float(state.get("personal_stress", 30)) + 8)
    elif event_id == "engagement":
        life["relationship_status"] = "engaged"
        partner = life.get("partner") if isinstance(life.get("partner"), dict) else {}
        partner.setdefault("id", _u_family_id(player_id, "partner"))
        partner["status"] = "engaged"
        life["partner"] = partner
    elif event_id == "wedding":
        life["relationship_status"] = "married"
        partner = life.get("partner") if isinstance(life.get("partner"), dict) else {"id": _u_family_id(player_id, "partner")}
        partner["status"] = "married"
        life["partner"] = partner
        life["home_stability"] = _u_clip(float(life.get("home_stability", 65)) + 8)
    elif event_id == "new_child":
        life["relationship_status"] = "family_household"
        life["dependents"] = int(life.get("dependents") or 0) + 1
        children = list(life.get("children") or [])
        children.append(
            {
                "id": _u_family_id(player_id, f"child_{len(children)}"),
                "age_bracket": "infant",
                "city_adjustment": _u_clip(50 + rng.uniform(-8, 10)),
                "school_stability": 70.0,
                "health_stress": 0.0,
            }
        )
        life["children"] = children
        life["sleep_quality"] = _u_clip(float(life.get("sleep_quality", 70)) - 8)
    elif event_id == "pregnancy_news":
        partner = life.get("partner") if isinstance(life.get("partner"), dict) else {}
        partner["expecting"] = True
        life["partner"] = partner
    elif event_id == "buys_home":
        life["home_owned"] = True
        life["housing"] = "owned"
        life["city_attachment"] = _u_clip(float(life.get("city_attachment") or 40) + 10)
        life["home_stability"] = _u_clip(float(life.get("home_stability", 65)) + 10)
        life["relocation_strain"] = _u_clip(float(life.get("relocation_strain") or 15) + 6)
    elif event_id == "family_settles":
        life["relocation_strain"] = _u_clip(max(0.0, float(life.get("relocation_strain") or 15) - 10))
        life["city_attachment"] = _u_clip(float(life.get("city_attachment") or 40) + 6)
        partner = life.get("partner") if isinstance(life.get("partner"), dict) else {}
        if partner:
            partner["city_satisfaction"] = _u_clip(float(partner.get("city_satisfaction") or 55) + 8)
            life["partner"] = partner
    elif event_id == "child_relocation":
        children = list(life.get("children") or [])
        if children:
            children[0]["city_adjustment"] = _u_clip(float(children[0].get("city_adjustment") or 55) - 12)
            life["children"] = children
    elif event_id == "old_friend_visit":
        friends = list(life.get("friends") or [])
        if friends:
            friends[0]["closeness"] = _u_clip(float(friends[0].get("closeness") or 45) + 6)
            friends[0]["last_contact_day"] = day
            life["friends"] = friends
    elif event_id == "poor_sleep":
        life["sleep_quality"] = _u_clip(float(life.get("sleep_quality", 70)) - 10)
    elif event_id == "unexpected_expense":
        life["financial_stress"] = _u_clip(float(life.get("financial_stress") or 10) + 8)
    elif event_id == "family_difficulty":
        children = list(life.get("children") or [])
        if children:
            children[0]["health_stress"] = _u_clip(float(children[0].get("health_stress") or 0) + 12)
            life["children"] = children
        state["personal_stress"] = _u_clip(float(state.get("personal_stress", 30)) + 6)

    recent = list(life.get("major_life_events") or [])
    recent.append({"id": event_id, "day": day, "severity": str(spec.get("event_tier") or "minor")})
    life["major_life_events"] = recent[-20:]


def _u_set_player_leave(
    session: Any,
    player_id: str,
    *,
    leave_type: str,
    days_min: int,
    days_max: int,
    reason_public: str,
    rng: random.Random,
) -> None:
    """Personal/family/mental-health leave — not injury."""
    day, iso, _ = _u_current_meta(session)
    duration = rng.randint(int(days_min), int(days_max))
    book = getattr(session, "universe_player_availability", None) or {}
    book[str(player_id)] = {
        "status": leave_type,
        "reason_public": reason_public,
        "return_day": day + duration,
        "calendar_iso": iso,
        "private_detail": True,
    }
    session.universe_player_availability = book


def _u_character_descriptor_tags(personality: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    if float(personality.get("professionalism", 50)) >= 72 and float(personality.get("accountability", 50)) >= 68:
        tags.append("Accountable")
    if float(personality.get("loyalty", 50)) >= 72:
        tags.append("Loyal")
    if float(personality.get("volatility", 50)) <= 38:
        tags.append("Low volatility")
    if float(personality.get("competitiveness", 50)) >= 82:
        tags.append("Elite competitive drive")
    if float(personality.get("family_orientation", 50)) >= 72:
        tags.append("Strong family priority")
    if float(personality.get("sociability", 50)) <= 35:
        tags.append("Private personality")
    if float(personality.get("media_savvy", 50)) <= 40:
        tags.append("Low media comfort")
    return tags[:5]


def _u_scout_confidence_for_trait(base_confidence: float, trait_value: float, interviews: int = 0) -> Tuple[str, int]:
    """Prospect scouting — blur unknowns, never expose exact hidden values."""
    conf = _u_clip(base_confidence + interviews * 4, 20, 95)
    if conf < 45:
        return "Unknown", int(conf)
    if conf < 58:
        return "Mixed reports", int(conf)
    return _u_tier_label(trait_value), int(conf)


def build_human_dossier_payload(
    session: Any,
    entity: Dict[str, Any],
    player: Any = None,
    *,
    scouting_mode: bool = False,
    scout_confidence: float = 82.0,
    include_private: bool = True,
) -> Dict[str, Any]:
    """Readable GM/scouting language — not raw 0-100 dumps."""
    personality = dict(entity.get("personality") or {})
    state = dict(entity.get("state") or {})
    life = dict(entity.get("life") or {}) if include_private else {}
    pressure = dict(entity.get("human_pressure") or {})
    mw = dict(entity.get("mental_wellbeing") or {}) if include_private else {}
    base_ovr = float(entity.get("overall") or _player_ovr99(player or object()))
    readiness_delta = 0.0
    for mod in _u_active_modifiers(session, str(entity.get("player_id") or "")):
        readiness_delta += float(mod.get("ovr_delta") or 0)
    current_ovr = round(_u_clip(base_ovr + readiness_delta, 1, 99), 1)

    char_tier = _u_tier_label(float(personality.get("character", 55)))
    descriptors = _u_character_descriptor_tags(personality)

    def trait_row(key: str, label: str, invert: bool = False) -> Dict[str, Any]:
        val = float(personality.get(key, 50))
        if scouting_mode:
            tier, conf = _u_scout_confidence_for_trait(scout_confidence, val)
            return {"label": label, "tier": tier, "confidence": conf}
        return {"label": label, "tier": _u_tier_label(val, invert=invert)}

    character_block = {
        "headline": char_tier,
        "summary_line": " · ".join(descriptors[:3]) if descriptors else "",
        "descriptors": descriptors,
        "traits": [
            trait_row("competitiveness", "Competitive Drive"),
            trait_row("professionalism", "Professionalism"),
            trait_row("leadership", "Leadership"),
            trait_row("loyalty", "Loyalty"),
            trait_row("volatility", "Volatility", invert=True),
            trait_row("family_orientation", "Family Priority"),
            trait_row("resilience", "Mental Resilience"),
            trait_row("sociability", "Social Fit"),
        ],
        "values": [
            trait_row("family_orientation", "Family"),
            trait_row("competitiveness", "Winning"),
            trait_row("money_focus", "Money"),
            trait_row("ambition", "Career Ambition"),
            trait_row("loyalty", "Loyalty"),
        ],
    }

    life_summary = "Limited information"
    if include_private and life:
        rel = str(life.get("relationship_status") or "single")
        deps = int(life.get("dependents") or 0)
        if rel == "married":
            life_summary = f"Married · {deps} dependent{'s' if deps != 1 else ''}" if deps else "Married"
        elif rel == "family_household":
            life_summary = f"Family household · {deps} child{'ren' if deps != 1 else ''}"
        elif rel == "engaged":
            life_summary = "Engaged"
        elif rel == "partnered":
            life_summary = "Partnered"
        else:
            life_summary = "Single"

    payload: Dict[str, Any] = {
        "character": character_block,
        "current_state": {
            "morale_tier": _u_tier_label(float(state.get("morale", 55))),
            "confidence_tier": _u_tier_label(float(state.get("confidence", 55))),
            "readiness_delta": round(readiness_delta, 2),
            "base_ovr": round(base_ovr, 1),
            "current_ovr": current_ovr,
            "pressure_tier": int(pressure.get("tier") or 0),
            "pressure_label": str(pressure.get("tier_label") or "Settled"),
            "role_satisfaction_tier": _u_tier_label(float(state.get("role_satisfaction", 55))),
        },
        "life": {
            "summary": life_summary,
            "city_attachment_tier": _u_tier_label(float(life.get("city_attachment") or 40)) if include_private else "Limited information",
            "home_stability_tier": _u_tier_label(float(life.get("home_stability") or 60)) if include_private else "Limited information",
            "relocation_tier": _u_tier_label(float(life.get("relocation_strain") or 20), invert=True) if include_private else "Limited information",
        },
        "pressure_drivers": list(pressure.get("drivers") or []),
    }
    if include_private:
        payload["mental_wellbeing"] = {
            "state": str(mw.get("state") or "stable"),
            "tier": _u_tier_label(float(mw.get("wellbeing_score") or 65)),
            "private": True,
        }
    if scouting_mode:
        payload["scouting"] = {
            "confidence": int(_u_clip(scout_confidence, 20, 95)),
            "character_read": char_tier if scout_confidence >= 50 else "Mixed reports",
        }
    return payload


def apply_trade_universe_relocation(
    session: Any,
    *,
    player_id: str,
    from_team_id: str,
    to_team_id: str,
    player_requested: bool = False,
    stability_promised: bool = False,
    newborn_recent: bool = False,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Human relocation consequences after a trade completes."""
    _u_migrate_v2(session)
    entities = getattr(session, "universe_players", None) or {}
    entity = entities.get(str(player_id))
    if not entity:
        return {"applied": False}
    local_rng = rng or random.Random(_u_seed("trade_reloc", player_id, to_team_id))
    life = entity.setdefault("life", {})
    state = entity.setdefault("state", {})
    day, iso, _ = _u_current_meta(session)

    if not newborn_recent:
        for ev in reversed(list(life.get("major_life_events") or [])):
            if str(ev.get("id") or "") == "new_child" and day - int(ev.get("day") or 0) <= 45:
                newborn_recent = True
                break

    strain = 8.0 + local_rng.uniform(0, 12)
    if newborn_recent:
        strain += 14.0
    if player_requested:
        strain -= 10.0
    if stability_promised and not player_requested:
        strain += 18.0

    life["relocation_strain"] = _u_clip(float(life.get("relocation_strain") or 15) + strain)
    life["city_attachment"] = _u_clip(max(15.0, float(life.get("city_attachment") or 40) - 8 - strain * 0.25))
    partner = life.get("partner") if isinstance(life.get("partner"), dict) else {}
    if partner:
        partner["city_satisfaction"] = _u_clip(float(partner.get("city_satisfaction") or 55) - 10 - strain * 0.2)
        life["partner"] = partner
    for child in list(life.get("children") or []):
        child["city_adjustment"] = _u_clip(float(child.get("city_adjustment") or 55) - 8)

    gm_delta = -min(18.0, strain * 0.55) if stability_promised and not player_requested else (-4.0 if not player_requested else 4.0)
    state["gm_trust"] = _u_clip(float(state.get("gm_trust", 65)) + gm_delta)
    entity["team_id"] = str(to_team_id)

    summary = "Relocation stress after trade"
    if player_requested:
        summary = "Trade aligned with player's relocation preference"
        _u_add_memory(entity, kind="gratitude", summary=summary, day=day, iso=iso, emotional_delta=4.0)
    elif stability_promised:
        summary = "Player feels organization broke stability promise with trade timing"
        _u_add_memory(entity, kind="betrayal", summary=summary, day=day, iso=iso, emotional_delta=-8.0)
    else:
        _u_add_memory(entity, kind="trade_relocation", summary=summary, day=day, iso=iso, emotional_delta=-3.0)

    try:
        from app.sim_engine.franchise.player_agent_engine import ensure_player_agent  # noqa: WPS433

        player = _player_from_roster(session, player_id)
        if player is not None:
            agent = ensure_player_agent(player, session)
            agent_id = str(agent.get("id") or "")
            if agent_id:
                _u_adjust_agent_org_trust(session, agent_id, from_team_id, gm_delta * 0.65, reason=summary)
    except Exception:
        pass

    entities[player_id] = entity
    session.universe_players = entities
    return {"applied": True, "relocation_strain": life.get("relocation_strain"), "gm_trust_delta": gm_delta}


def _u_migrate_v2(session: Any) -> None:
    """V3-compatible save migration. Safe for old V2 saves."""
    defaults = {
        "universe_engine_version": UNIVERSE_ENGINE_VERSION,
        "universe_players": {},
        "universe_locker_rooms": {},
        "universe_interactions": [],
        "universe_interaction_queue": [],
        "universe_event_log": [],
        "universe_reporter_relationships": {},
        "universe_attribute_modifiers": {},
        "universe_readiness_modifiers": {},
        "universe_promises": [],
        "universe_trade_demands": [],
        "universe_team_sanctions": [],
        "universe_cap_penalties": {},
        "universe_forfeited_picks": [],
        "universe_player_availability": {},
        "universe_agent_org_relationships": {},
        "universe_major_event_state": {},
        "universe_daily_snapshots": [],
        "universe_game_contexts": {},
        "_universe_last_daily_tick": -1,
    }
    for key, default in defaults.items():
        if getattr(session, key, None) is None:
            if isinstance(default, dict):
                setattr(session, key, default.copy())
            elif isinstance(default, list):
                setattr(session, key, list(default))
            else:
                setattr(session, key, default)
    session.universe_engine_version = UNIVERSE_ENGINE_VERSION


def _u_sync_player_entities(session: Any) -> Dict[str, Dict[str, Any]]:
    """Sync real players into Universe state while preserving V3 mental/life ledgers."""
    _u_migrate_v2(session)
    entities = dict(getattr(session, "universe_players", None) or {})
    active_ids: List[str] = []
    for team_id, player in _u_all_players(session):
        player_id = str(getattr(player, "id", "") or "")
        active_ids.append(player_id)
        entity = dict(entities.get(player_id) or {})
        if not entity:
            entity = _u_create_player_entity(session, team_id, player)
        entity["player_name"] = _u_name(player)
        entity["team_id"] = str(team_id)
        entity["position"] = _u_position(player)
        entity["age"] = _player_age(player)
        entity["overall"] = round(_player_ovr99(player), 1)
        entity.setdefault("state", {})
        entity.setdefault("life", {})
        entity.setdefault("social", {})
        entity.setdefault("concerns", {})
        entity.setdefault("memories", [])
        entity.setdefault("attribute_ledger", [])
        entity.setdefault("potential_ledger", [])
        entity.setdefault("reputation_tags", [])
        entity.setdefault("season_permanent_attribute_delta", {})
        entity.setdefault("season_potential_delta", {})
        entity["state"].setdefault("character_modifier", 0.0)
        entity["state"].setdefault("mental_ovr", None)
        entity["life"].setdefault("last_minor_event_day", -999)
        entity["life"].setdefault("minor_event_history", [])
        base_personality = _u_personality(player, player_id)
        base_personality["character"] = _u_clip(
            float(base_personality.get("character", 55))
            + float(entity["state"].get("character_modifier", 0.0) or 0.0)
        )
        entity["personality"] = base_personality
        ident = getattr(player, "identity", None)
        entity["identity"] = {
            "name": _u_name(player),
            "age": entity["age"],
            "birth_city": str(getattr(ident, "birth_city", "") or ""),
            "birth_country": str(getattr(ident, "birth_country", "") or ""),
            "draft_year": int(getattr(ident, "draft_year", 0) or 0),
            "draft_round": int(getattr(ident, "draft_round", 0) or 0),
            "draft_pick": int(getattr(ident, "draft_pick", 0) or 0),
            "position": entity["position"],
            "overall": entity["overall"],
        }
        entity["trusts"] = {
            "coach": round(_u_psych_value(player, ("coach_trust", "coach_relationship"), 55.0), 1),
            "gm": round(_u_psych_value(player, ("trust_in_management",), float((entity.get("state") or {}).get("gm_trust", 55))), 1),
            "teammates": round(_u_psych_value(player, ("trust_in_teammates",), 55.0), 1),
            "room": round(_u_psych_value(player, ("locker_room_fit",), float((entity.get("state") or {}).get("belonging", 55))), 1),
        }
        entity["personality_tags"] = _u_personality_tags(entity)
        _u_migrate_entity_life(entity, random.Random(_u_seed("life_migrate", player_id)))
        try:
            from app.sim_engine.franchise.player_agent_engine import ensure_player_agent  # noqa: WPS433

            agent = ensure_player_agent(player, session)
            agent_id = str(agent.get("id") or "")
            agent_trust = _u_agent_org_trust(session, agent_id, str(team_id)) if agent_id else 55.0
        except Exception:
            agent_trust = 55.0
        room = (getattr(session, "universe_locker_rooms", None) or {}).get(str(team_id)) or {}
        entity["human_pressure"] = _u_compute_human_pressure(entity, room=room, agent_org_trust=agent_trust)
        entities[player_id] = entity
    for player_id, entity in entities.items():
        entity["active_roster"] = player_id in active_ids
    session.universe_players = entities
    return entities


def _u_mental_ovr(entity: Dict[str, Any], player: Any = None) -> float:
    """Use future Mental OVR if present; otherwise derive a stable temporary fallback."""
    state = entity.get("state") or {}
    for candidate in (
        state.get("mental_ovr"),
        entity.get("mental_ovr"),
        getattr(player, "mental_ovr", None) if player is not None else None,
        getattr(getattr(player, "psych", None), "mental_ovr", None) if player is not None else None,
    ):
        if candidate is None or candidate == "":
            continue
        try:
            value = float(candidate)
            return _u_clip(value * 100.0 if value <= 1.5 else value, 1.0, 99.0)
        except (TypeError, ValueError):
            pass
    personality = entity.get("personality") or {}
    return _u_clip(
        float(personality.get("resilience", 55)) * 0.28
        + float(personality.get("professionalism", 55)) * 0.17
        + float(state.get("confidence", 55)) * 0.23
        + float(state.get("focus", 60)) * 0.18
        + float(state.get("morale", 55)) * 0.14,
        1.0,
        99.0,
    )


def _u_apply_readiness_modifier(
    session: Any,
    player_id: str,
    *,
    source_id: str,
    ovr_delta: float,
    days: int,
    reason: str,
    stat_modifiers: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Apply day-based effective-OVR/stat effects without rewriting base hockey talent."""
    store = dict(getattr(session, "universe_readiness_modifiers", None) or {})
    rows = list(store.get(str(player_id)) or [])
    receipt = {
        "id": f"ready_{uuid.uuid4().hex[:10]}",
        "source_id": str(source_id),
        "player_id": str(player_id),
        "ovr_delta": round(max(UNIVERSE_READINESS_MIN, min(UNIVERSE_READINESS_MAX, float(ovr_delta))), 2),
        "days_remaining": max(1, int(days)),
        "reason": str(reason),
        "stat_modifiers": {str(k): float(v) for k, v in (stat_modifiers or {}).items()},
    }
    rows.append(receipt)
    store[str(player_id)] = rows[-20:]
    session.universe_readiness_modifiers = store
    return receipt


def _u_active_readiness(session: Any, player_id: str) -> Dict[str, Any]:
    total_ovr = 0.0
    stats: Dict[str, float] = {}
    sources: List[Dict[str, Any]] = []
    for row in (getattr(session, "universe_readiness_modifiers", None) or {}).get(str(player_id), []) or []:
        if int(row.get("days_remaining", 0) or 0) <= 0:
            continue
        total_ovr += float(row.get("ovr_delta", 0) or 0)
        for key, value in (row.get("stat_modifiers") or {}).items():
            stats[str(key)] = stats.get(str(key), 0.0) + float(value or 0)
        sources.append(row)
    return {
        "ovr_delta": round(max(UNIVERSE_READINESS_MIN, min(UNIVERSE_READINESS_MAX, total_ovr)), 2),
        "stat_modifiers": {k: round(v, 3) for k, v in stats.items()},
        "sources": sources,
    }


def _u_tick_readiness_modifiers(session: Any) -> int:
    store = dict(getattr(session, "universe_readiness_modifiers", None) or {})
    expired = 0
    for player_id, rows in list(store.items()):
        kept: List[Dict[str, Any]] = []
        for row in rows or []:
            row["days_remaining"] = int(row.get("days_remaining", 0) or 0) - 1
            if row["days_remaining"] > 0:
                kept.append(row)
            else:
                expired += 1
        if kept:
            store[player_id] = kept
        else:
            store.pop(player_id, None)
    session.universe_readiness_modifiers = store
    return expired


def _u_write_potential(player: Any, delta_99: float) -> Tuple[Optional[float], Optional[float], str]:
    """Best-effort writer for the project's known potential field names."""
    for container_name, container in (
        ("player", player),
        ("development", getattr(player, "development", None)),
        ("ratings", getattr(player, "ratings", None)),
    ):
        if container is None:
            continue
        for name in ("dev_potential", "potential", "dev_ceiling"):
            exists = name in container if isinstance(container, dict) else hasattr(container, name)
            if not exists:
                continue
            raw = _u_get(container, name, None)
            if callable(raw):
                continue
            try:
                before = float(raw)
            except (TypeError, ValueError):
                continue
            if before <= 1.5:
                after = max(0.01, min(1.0, before + float(delta_99) / 99.0))
            else:
                after = max(1.0, min(99.0, before + float(delta_99)))
            if isinstance(container, dict):
                container[name] = after
            else:
                try:
                    setattr(container, name, after)
                except Exception:
                    continue
            return before, after, f"{container_name}.{name}"
    return None, None, "ledger_only"


def _u_apply_potential_change(session: Any, player_id: str, delta: float, source_id: str, reason: str) -> Dict[str, Any]:
    """Capped long-term potential movement for genuine development/life arcs."""
    entities = getattr(session, "universe_players", None) or {}
    entity = entities.get(str(player_id)) or {}
    season = _u_current_meta(session)[2]
    caps = entity.setdefault("season_potential_delta", {})
    key = str(season)
    used = float(caps.get(key, 0.0) or 0.0)
    requested = float(delta)
    if requested >= 0:
        applied = min(requested, max(0.0, UNIVERSE_POTENTIAL_SEASON_CAP - used))
    else:
        applied = max(requested, min(0.0, -UNIVERSE_POTENTIAL_SEASON_CAP - used))
    caps[key] = round(used + applied, 3)
    player = _player_from_roster(session, str(player_id))
    before = after = None
    location = "ledger_only"
    if player is not None and abs(applied) > 0.0001:
        before, after, location = _u_write_potential(player, applied)
    receipt = {
        "id": f"pot_{uuid.uuid4().hex[:10]}",
        "source_id": str(source_id),
        "player_id": str(player_id),
        "requested_delta": requested,
        "applied_delta": round(applied, 3),
        "before": before,
        "after": after,
        "location": location,
        "reason": str(reason),
        "season": season,
    }
    ledger = list(entity.get("potential_ledger") or [])
    ledger.append(receipt)
    entity["potential_ledger"] = ledger[-40:]
    return receipt


def _u_notify_user_event(
    session: Any,
    event: Dict[str, Any],
    *,
    presentation_level: int = 1,
    force_league: bool = False,
) -> None:
    """Central notification router: 1=inbox, 2=banner, 3=popup, 4=blocking breaking news."""
    _ensure_session_event_lists(session)
    user_team_id = str(getattr(session, "user_team_id", "") or "")
    team_id = str(event.get("team_id") or "")
    if not force_league and user_team_id and team_id and team_id != user_team_id:
        return
    day, iso, _ = _u_current_meta(session)
    event_id = str(event.get("id") or f"uve_{uuid.uuid4().hex[:10]}")
    notif_id = f"universe_notif:{event_id}"
    if not any(str(row.get("id") or "") == notif_id for row in session.notifications):
        session.notifications.append(
            {
                "id": notif_id,
                "type": "breaking_news" if presentation_level >= 4 else "player_meeting" if event.get("requires_action") else "storyline",
                "priority": "CRITICAL" if presentation_level >= 4 else "HIGH" if presentation_level >= 2 else "MEDIUM",
                "title": str(event.get("notification_title") or event.get("headline") or event.get("title") or "Franchise update")[:110],
                "text": str(event.get("summary") or event.get("description") or "")[:280],
                "date": iso or day,
                "calendar_day": day,
                "calendar_iso": iso,
                "team_id": team_id,
                "player_id": str(event.get("player_id") or event.get("actor_id") or ""),
                "source": "storyline_universe_v3",
                "presentation_level": int(presentation_level),
                "storyline_id": event.get("storyline_id"),
                "interaction_id": event.get("interaction_id") or event.get("id") if event.get("requires_action") else None,
            }
        )
        session.notifications = session.notifications[-180:]
    if presentation_level < 3:
        return
    popup_id = f"universe_popup:{event_id}"
    if any(str(row.get("id") or "") == popup_id for row in session.pending_ui_popups):
        return
    session.pending_ui_popups.append(
        {
            "id": popup_id,
            "kind": "breaking_news" if presentation_level >= 4 else "player_meeting",
            "blocking": presentation_level >= 4,
            "severity": "catastrophic" if str(event.get("event_tier")) == "catastrophic" else "major" if presentation_level >= 4 else "significant",
            "title": str(event.get("popup_title") or event.get("notification_title") or ("BREAKING NEWS" if presentation_level >= 4 else "PLAYER MEETING")),
            "headline": str(event.get("headline") or event.get("title") or "Franchise update"),
            "summary": str(event.get("summary") or event.get("description") or ""),
            "description": str(event.get("summary") or event.get("description") or ""),
            "team_id": team_id,
            "player_id": str(event.get("player_id") or event.get("actor_id") or ""),
            "player_name": str(event.get("player_name") or ""),
            "calendar_day": day,
            "calendar_iso": iso,
            "presentation_level": int(presentation_level),
            "requires_decision": bool(event.get("requires_action")),
            "interaction_id": event.get("interaction_id") or (event.get("id") if event.get("requires_action") else None),
            "storyline_id": event.get("storyline_id"),
            "choices": list(event.get("choices") or []),
        }
    )
    session.pending_ui_popups = session.pending_ui_popups[-30:]


def _u_record_storyline(session: Any, *, event: Dict[str, Any], headline: str, summary: str, cause_type: str, category: str, heat: int, public: bool = True) -> Optional[Dict[str, Any]]:
    """V3 storyline writer: event tier controls severity; heat only controls media attention."""
    record_private = bool(event.get("record_private_storyline"))
    if not public and not record_private:
        return None
    day, iso, _ = _u_current_meta(session)
    participants = list(event.get("participants") or [])
    player_id = str(participants[0] if participants else event.get("player_id") or "")
    entities = getattr(session, "universe_players", None) or {}
    entity = entities.get(player_id) or {}
    team_id = str(event.get("team_id") or entity.get("team_id") or "")
    event_id = str(event.get("id") or f"uve_{uuid.uuid4().hex[:10]}")
    try:
        cause_event_id = record_decision_event(
            session,
            {
                "event_type": cause_type,
                "team_id": team_id,
                "player_id": player_id,
                "player_name": entity.get("player_name") or event.get("player_name"),
                "calendar_day": day,
                "calendar_iso": iso,
                "universe_event_id": event_id,
                "event_tier": event.get("event_tier"),
            },
        )
    except Exception:
        cause_event_id = event_id
    tier = str(event.get("event_tier") or "minor").lower()
    severity_map = {"ambient": "minor", "minor": "minor", "developing": "mid", "major": "major", "catastrophic": "crisis"}
    priority_map = {"ambient": "LOW", "minor": "MEDIUM", "developing": "HIGH", "major": "HIGH", "catastrophic": "CRITICAL"}
    tone = str(event.get("tone") or "")
    if not tone:
        tone = "negative" if cause_type in (
            "TEAMMATE_CONFLICT", "TEAMMATE_FIGHT", "PLAYER_REPORTER_ALTERCATION", "PROMISE_BROKEN",
            "LOW_CHARACTER_GAME_IMPACT", "PLAYER_ARRESTED", "LEAGUE_SUSPENSION", "PLAYER_BANNED",
            "MAJOR_PUBLIC_ALTERCATION", "GAMBLING_VIOLATION", "PLAYER_DEATH", "UNDER_TABLE_PAYMENTS",
            "CAP_CIRCUMVENTION", "ILLEGAL_TEAM_WORKOUTS", "EXECUTIVE_MISCONDUCT", "TRADE_DEMAND"
        ) else "positive" if cause_type in ("HIGH_CHARACTER_IMPACT", "PROMISE_KEPT", "COMMUNITY_MOMENT", "POSITIVE_LIFE_EVENT") else "neutral"
    visibility = str(event.get("visibility") or ("public" if public else "team_only"))
    knowledge_type = str(event.get("knowledge_type") or ("fact" if cause_type in _FACT_CAUSE_TYPES else "claim" if cause_type in _CLAIM_CAUSE_TYPES else "report"))
    public_level = str(event.get("public_knowledge_level") or ("confirmed" if visibility == "public" and knowledge_type == "fact" else "widely_reported" if visibility == "public" else "private"))
    row = {
        "id": f"story_{uuid.uuid4().hex[:12]}",
        "storyline_id": "",
        "type": str(event.get("kind") or category),
        "category": category,
        "cause_type": cause_type,
        "cause_event_id": cause_event_id,
        "universe_event_id": event_id,
        "team_id": team_id,
        "team_name": _team_display(session, team_id) if team_id else "League",
        "player_id": player_id,
        "player_name": entity.get("player_name") or event.get("player_name") or "",
        "related_player_ids": participants,
        "headline": headline,
        "title": headline,
        "summary": summary,
        "short_summary": summary[:180],
        "description": summary,
        "tone": tone,
        "priority": priority_map.get(tier, "MEDIUM"),
        "severity": severity_map.get(tier, "minor"),
        "event_tier": tier,
        "mechanical_severity": int(event.get("mechanical_severity") or 10),
        "visibility": visibility,
        "heat": int(_u_clip(heat, 5, 100)),
        "credibility": int(event.get("credibility") or 90),
        "calendar_day": day,
        "calendar_iso": iso,
        "date": iso or day,
        "source": "storyline_universe_v3",
        "effects": dict(event.get("effects") or {}),
        "stable_key": str(event.get("stable_key") or ""),
        "knowledge_type": knowledge_type,
        "public_knowledge_level": public_level,
        "source_label": event.get("source_label") or "",
        "reporter_name": event.get("reporter_name") or "",
        "outlet_name": event.get("outlet_name") or "",
        "evidence": dict(event.get("evidence") or {}),
        "personality_tags": list(entity.get("personality_tags") or []),
        "top_concerns": list(entity.get("top_concerns") or []),
        "trusts": dict(entity.get("trusts") or {}),
        **lane_flags(
            classify_story_lane(
                cause_type=cause_type,
                category=category,
                knowledge_type=knowledge_type,
                public_level=public_level,
                heat=heat,
                legal_status=str(event.get("legal_status") or ""),
                incident_family=str(event.get("incident_family") or ""),
            )
        ),
    }
    row["storyline_id"] = row["id"]
    try:
        from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433
        _record_storyline(session, row)
    except (ImportError, ModuleNotFoundError):
        enriched = _UNIVERSE_LEGACY_ENRICH(session, row)
        existing = list(getattr(session, "storyline_events", None) or [])
        existing.append(enriched)
        session.storyline_events = existing[-300:]
    return row


def _u_life_event_allowed(entity: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    life = entity.get("life") or {}
    relationship = str(life.get("relationship_status") or "single")
    dependents = int(life.get("dependents", 0) or 0)
    if spec.get("requires_partnered") and relationship == "single":
        return False
    if spec.get("requires_dependents") and dependents <= 0:
        return False
    return True


def _u_generate_minor_life_events(session: Any, rng: random.Random) -> int:
    """Generate diverse small positive/negative off-ice events with subtle real sim effects."""
    user_team_id = str(getattr(session, "user_team_id", "") or "")
    entities = getattr(session, "universe_players", None) or {}
    created = 0
    day, iso, _ = _u_current_meta(session)
    for team_id_raw, team in (getattr(session, "team_by_id", None) or {}).items():
        team_id = str(team_id_raw)
        for player in list(getattr(team, "roster", None) or []):
            player_id = str(getattr(player, "id", "") or "")
            entity = entities.get(player_id)
            if not entity or getattr(player, "retired", False):
                continue
            life = entity.get("life") or {}
            last = int(life.get("last_minor_event_day", -999) or -999)
            if day - last < 18:
                continue
            chance = 0.0045 if team_id == user_team_id else 0.00055
            if rng.random() > chance:
                continue
            positive = rng.random() < 0.52
            pool = MINOR_POSITIVE_LIFE_EVENTS if positive else MINOR_NEGATIVE_LIFE_EVENTS
            allowed = [spec for spec in pool if _u_life_event_allowed(entity, spec)]
            if not allowed:
                continue
            spec = rng.choice(allowed)
            event_id = f"life_{spec['id']}_{player_id}_{day}"
            _u_mutate_life_from_event(entity, spec, rng, day)
            leave_cfg = spec.get("leave") if isinstance(spec.get("leave"), dict) else None
            if leave_cfg and rng.random() < float(leave_cfg.get("chance") or 0):
                _u_set_player_leave(
                    session,
                    player_id,
                    leave_type=str(leave_cfg.get("type") or "personal_leave"),
                    days_min=int(leave_cfg.get("days_min") or 2),
                    days_max=int(leave_cfg.get("days_max") or 5),
                    reason_public=str(leave_cfg.get("reason_public") or "Unavailable — Personal Leave"),
                    rng=rng,
                )
            for field, delta in (spec.get("profile") or {}).items():
                _u_apply_profile_delta(entity, str(field), float(delta))
            char_delta = float(spec.get("character", 0) or 0)
            if char_delta:
                _u_apply_profile_delta(entity, "state.character_modifier", char_delta)
            readiness = _u_apply_readiness_modifier(
                session,
                player_id,
                source_id=event_id,
                ovr_delta=float(spec.get("ovr", 0) or 0),
                days=int(spec.get("days", 3) or 3),
                reason=str(spec.get("summary") or spec.get("headline") or "Minor life event"),
                stat_modifiers={str(k): float(v) for k, v in (spec.get("attrs") or {}).items()},
            )
            potential_receipt = None
            if positive and int(entity.get("age", 30) or 30) <= 25 and rng.random() < float(spec.get("potential_chance", 0) or 0):
                potential_receipt = _u_apply_potential_change(
                    session,
                    player_id,
                    float(spec.get("potential", 0) or 0),
                    event_id,
                    str(spec.get("summary") or "Positive stability/development event"),
                )
            life["last_minor_event_day"] = day
            history = list(life.get("minor_event_history") or [])
            history.append({"id": event_id, "event_type": spec["id"], "positive": positive, "calendar_day": day, "calendar_iso": iso})
            life["minor_event_history"] = history[-30:]
            entity["life"] = life
            headline = str(spec["headline"]).format(name=entity.get("player_name") or "Player")
            summary = str(spec["summary"])
            public = rng.random() < float(spec.get("public_chance", 0.08 if positive else 0.03))
            event = {
                "id": event_id,
                "kind": spec["id"],
                "team_id": team_id,
                "player_id": player_id,
                "player_name": entity.get("player_name"),
                "participants": [player_id],
                "headline": headline,
                "summary": summary,
                "tone": "positive" if positive else "negative",
                "event_tier": "minor",
                "mechanical_severity": 10 if positive else 12,
                "visibility": "public" if public else "team_only",
                "knowledge_type": "fact",
                "record_private_storyline": team_id == user_team_id,
                "evidence": {"readiness": readiness, "potential": potential_receipt},
            }
            _u_append_event(session, {**event, "calendar_day": day, "calendar_iso": iso})
            _u_add_memory(entity, kind=spec["id"], summary=summary, day=day, iso=iso, emotional_delta=2.0 if positive else -2.0, public=public)
            storyline = _u_record_storyline(
                session,
                event=event,
                headline=headline,
                summary=summary,
                cause_type="POSITIVE_LIFE_EVENT" if positive else "MINOR_LIFE_EVENT",
                category="personal_life",
                heat=int(spec.get("heat", 10) or 10),
                public=public,
            )
            if storyline:
                event["storyline_id"] = storyline.get("storyline_id")
            if team_id == user_team_id:
                _u_notify_user_event(session, event, presentation_level=1)
            created += 1
            if created >= 4:
                return created
    return created


def _u_interaction_candidates(session: Any, team_id: str, rng: random.Random) -> List[Tuple[float, str, str, str]]:
    """V3 candidate list with specific role/contract/development meeting families."""
    entities = getattr(session, "universe_players", None) or {}
    room = _u_rebuild_locker_room(session, team_id)
    player_ids = [str(getattr(p, "id", "") or "") for p in _u_team_players(session, team_id)]
    candidates: List[Tuple[float, str, str, str]] = []
    for player_id in player_ids:
        entity = entities.get(player_id) or {}
        state = entity.get("state") or {}
        personality = entity.get("personality") or {}
        concerns = entity.get("concerns") or {}
        niches = _u_niche_ids(entity)
        role_sat = float(state.get("role_satisfaction", 60))
        pos = str(entity.get("position") or "F").upper()
        offense_signal = max(
            _u_read_numeric(_player_from_roster(session, player_id) or object(), ("offensive_awareness", "passing"), 50),
            _u_read_numeric(_player_from_roster(session, player_id) or object(), ("shot_accuracy", "shooting_accuracy"), 50),
        )
        if role_sat < 45:
            kind = "request_starting_role" if pos == "G" else "request_pp_time" if offense_signal >= 78 and float(personality.get("ambition", 50)) >= 60 else "request_more_ice"
            candidates.append((80 - role_sat + float(personality.get("ambition", 50)) * 0.24, kind, player_id, ""))
        contract = concerns.get("contract") or {}
        if float(contract.get("satisfaction", 60)) < 38 and float(contract.get("importance", 50)) >= 58:
            candidates.append((72 - float(contract.get("satisfaction", 60)) + float(personality.get("money_focus", 45)) * 0.20, "contract_clarity", player_id, ""))
        winning = concerns.get("winning") or {}
        if float(winning.get("satisfaction", 60)) < 34 and float(personality.get("competitiveness", 50)) >= 65:
            candidates.append((70 - float(winning.get("satisfaction", 60)) + float(personality.get("competitiveness", 50)) * 0.25, "winning_concern_meeting", player_id, ""))
        development = concerns.get("development") or {}
        if int(entity.get("age", 30) or 30) <= 24 and float(development.get("satisfaction", 60)) < 45:
            candidates.append((66 - float(development.get("satisfaction", 60)) + float(personality.get("ambition", 50)) * 0.20, "development_meeting", player_id, ""))
        if float(state.get("personal_stress", 25)) > 65:
            candidates.append((float(state.get("personal_stress", 25)) + 8, "personal_check_in", player_id, ""))
        reporter_rel = _u_reporter_relationship(session, "hart", player_id)
        media_score = float(state.get("media_stress", 25)) + float(reporter_rel.get("friction", 20)) * 0.55 + float(personality.get("volatility", 40)) * 0.25
        if media_score > 76:
            kind = "reporter_altercation" if media_score > 118 and float(personality.get("volatility", 0)) > 76 and rng.random() < 0.16 else "reporter_confrontation"
            candidates.append((media_score, kind, player_id, ""))
        if float(entity.get("overall", 99)) < 80 and float(entity.get("room_value", 0)) >= 72:
            candidates.append((float(entity.get("room_value", 0)) - 8, "unheralded_leader", player_id, ""))
        if "mentor" in niches or "glue_guy" in niches or "peacemaker" in niches:
            possible_targets = [pid for pid in player_ids if pid != player_id]
            if possible_targets:
                target_id = min(possible_targets, key=lambda pid: float((entities.get(pid) or {}).get("age", 27)) + float(((entities.get(pid) or {}).get("state") or {}).get("confidence", 55)) * 0.08)
                target = entities.get(target_id) or {}
                if "mentor" in niches and int(target.get("age", 30)) <= 24:
                    candidates.append((58 + max(0, 52 - float((target.get("state") or {}).get("confidence", 55))), "mentor_session", player_id, target_id))
                if ("glue_guy" in niches or "peacemaker" in niches) and float((room.get("culture") or {}).get("tension", 30)) >= 44:
                    candidates.append((62 + float((room.get("culture") or {}).get("tension", 30)) * 0.35, "glue_intervention", player_id, target_id))
    for rel in (room.get("relationships") or {}).values():
        ids = list(rel.get("player_ids") or [])
        if len(ids) != 2 or any(pid not in entities for pid in ids):
            continue
        a = entities[ids[0]]
        b = entities[ids[1]]
        tension = float(rel.get("tension", 0))
        volatility = (float((a.get("personality") or {}).get("volatility", 40)) + float((b.get("personality") or {}).get("volatility", 40))) / 2
        character_floor = min(float((a.get("personality") or {}).get("character", 55)), float((b.get("personality") or {}).get("character", 55)))
        if tension >= 62 and character_floor < 52:
            kind = "teammate_fight" if tension >= 82 and volatility >= 72 and rng.random() < 0.24 else "blame_game"
            candidates.append((tension + volatility * 0.35, kind, ids[0], ids[1]))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    return candidates


def _u_make_extended_interaction(session: Any, team_id: str, kind: str, actor_id: str, target_id: str, rng: random.Random, score: float) -> Dict[str, Any]:
    """Build new specific meeting types; fall back to the existing V2 scene builder."""
    if kind not in {"request_more_ice", "request_pp_time", "request_starting_role", "contract_clarity", "winning_concern_meeting", "development_meeting"}:
        return _u_make_interaction(session, team_id, kind, actor_id, target_id, rng, score)
    entity = (getattr(session, "universe_players", None) or {}).get(actor_id) or {}
    name = str(entity.get("player_name") or "Player")
    day, iso, _ = _u_current_meta(session)
    interaction = {
        "id": f"int_{uuid.uuid4().hex[:12]}",
        "kind": kind,
        "team_id": str(team_id),
        "participants": [actor_id],
        "actor_id": actor_id,
        "target_id": "",
        "player_id": actor_id,
        "player_name": name,
        "calendar_day": day,
        "calendar_iso": iso,
        "expires_day": day + 3,
        "status": "pending",
        "score": round(score, 2),
        "requires_action": str(team_id) == str(getattr(session, "user_team_id", "") or ""),
        "private": True,
        "stakes": "high",
        "dialogue": [],
        "choices": [],
        "default_choice_id": "honest",
        "event_tier": "minor",
        "notification_title": "PLAYER MEETING REQUEST",
    }
    if kind == "request_more_ice":
        interaction.update({
            "title": f"{name} asks for more ice time",
            "summary": f"{name} believes his recent work warrants a larger role and wants a direct answer from management.",
            "dialogue": [{"speaker": name, "text": "I want more responsibility. If you think I haven't earned it, tell me exactly what I need to do."}],
            "choices": [
                _u_choice("honest", "Give an honest assessment", "Explain the current role and what must improve.", {"profile_changes": {"actor": {"state.gm_trust": 4, "state.focus": 2}}, "readiness_changes": [{"who": "actor", "ovr_delta": 0.25, "days": 5, "reason": "Clear role expectations", "stats": {"toi_readiness": 0.02}}], "public": False}),
                _u_choice("promise", "Promise a bigger opportunity", "Morale jumps now, but the promise must be delivered.", {"profile_changes": {"actor": {"state.morale": 5, "state.gm_trust": 3}}, "promise": {"type": "role_opportunity", "player_id": actor_id, "due_games": 6, "description": "Give the player a meaningful lineup opportunity within six games.", "success_potential_delta": 0.35, "success_readiness": 0.5, "failure_readiness": -2.0}, "public": False}),
                _u_choice("decline", "Tell him the role is earned", "Push back without offering a timetable.", {"profile_changes": {"actor": {"state.gm_trust": -4, "state.morale": -3, "state.focus": 2}}, "readiness_changes": [{"who": "actor", "ovr_delta": -0.75, "days": 8, "reason": "Frustration after role meeting", "stats": {"offensive_awareness": -0.25}}], "public": False}),
            ],
        })
    elif kind == "request_pp_time":
        interaction.update({
            "title": f"{name} wants a power-play opportunity",
            "summary": f"{name} believes his offensive game deserves a larger special-teams role.",
            "dialogue": [{"speaker": name, "text": "Give me a real look on the power play. I think I can help us there."}],
            "choices": [
                _u_choice("honest", "Explain the current PP hierarchy", "Set expectations without making a promise.", {"profile_changes": {"actor": {"state.gm_trust": 3, "state.focus": 2}}, "public": False}),
                _u_choice("promise", "Promise a PP look", "Create a development opportunity that can pay off if delivered.", {"profile_changes": {"actor": {"state.morale": 4, "state.confidence": 2}}, "promise": {"type": "power_play_opportunity", "player_id": actor_id, "due_games": 8, "description": "Give the player meaningful power-play usage within eight games.", "success_potential_delta": 0.5, "success_attribute": "offensive_awareness", "success_attribute_delta": 0.4, "failure_readiness": -2.0}, "public": False}),
                _u_choice("decline", "Decline the request", "Keep the current units intact.", {"profile_changes": {"actor": {"state.morale": -3, "state.gm_trust": -3}}, "readiness_changes": [{"who": "actor", "ovr_delta": -0.6, "days": 7, "reason": "Power-play role frustration", "stats": {"shot_involvement": -0.015}}], "public": False}),
            ],
        })
    elif kind == "request_starting_role":
        interaction.update({
            "title": f"{name} wants more starts",
            "summary": f"{name} believes his play warrants a larger share of the crease.",
            "dialogue": [{"speaker": name, "text": "I want the net more often. I think I've earned the chance to run with it."}],
            "choices": [
                _u_choice("honest", "Explain the goalie plan", "Give a clear workload explanation.", {"profile_changes": {"actor": {"state.gm_trust": 4, "state.focus": 2}}, "public": False}),
                _u_choice("promise", "Promise more starts", "A real opportunity can raise confidence and development.", {"profile_changes": {"actor": {"state.confidence": 4, "state.morale": 3}}, "promise": {"type": "goalie_start_opportunity", "player_id": actor_id, "due_games": 6, "description": "Give the goalie a meaningful run of starts.", "success_potential_delta": 0.35, "failure_readiness": -2.5}, "public": False}),
                _u_choice("decline", "Stay with the current starter", "The hierarchy remains unchanged.", {"profile_changes": {"actor": {"state.morale": -4, "state.gm_trust": -3}}, "readiness_changes": [{"who": "actor", "ovr_delta": -0.8, "days": 10, "reason": "Goalie role frustration", "stats": {"goalie_positioning": -0.3}}], "public": False}),
            ],
        })
    elif kind == "contract_clarity":
        interaction.update({
            "title": f"{name} wants contract clarity",
            "summary": f"{name} asks where he stands before contract uncertainty becomes a distraction.",
            "dialogue": [{"speaker": name, "text": "I don't need a number today. I need to know whether I'm part of the plan."}],
            "choices": [
                _u_choice("honest", "Be direct", "Give an honest view of the club's plans.", {"profile_changes": {"actor": {"state.gm_trust": 5, "state.focus": 2}}, "public": False}),
                _u_choice("commit", "Express commitment", "Reassure the player without promising contract terms.", {"profile_changes": {"actor": {"state.gm_trust": 4, "state.morale": 3}}, "readiness_changes": [{"who": "actor", "ovr_delta": 0.4, "days": 7, "reason": "Contract reassurance"}], "public": False}),
                _u_choice("deflect", "Avoid committing", "Keep flexibility, but uncertainty grows.", {"profile_changes": {"actor": {"state.gm_trust": -5, "state.media_stress": 2}}, "readiness_changes": [{"who": "actor", "ovr_delta": -0.65, "days": 10, "reason": "Contract uncertainty", "stats": {"focus": -0.2}}], "public": False}),
            ],
        })
    elif kind == "winning_concern_meeting":
        interaction.update({
            "title": f"{name} wants to talk about the club's direction",
            "summary": f"{name} is increasingly concerned about whether the team is positioned to compete.",
            "dialogue": [{"speaker": name, "text": "I want to win. I need to understand what we're building toward."}],
            "choices": [
                _u_choice("honest", "Explain the plan", "A concrete explanation can stabilize trust.", {"profile_changes": {"actor": {"state.gm_trust": 4, "state.focus": 2}}, "public": False}),
                _u_choice("promise_compete", "Promise to improve the roster", "Creates a meaningful management promise.", {"profile_changes": {"actor": {"state.morale": 3, "state.gm_trust": 3}}, "promise": {"type": "winning_commitment", "player_id": actor_id, "due_games": 12, "description": "Show meaningful progress toward a more competitive team.", "success_readiness": 0.75, "failure_readiness": -3.0}, "public": False}),
                _u_choice("dismiss", "Tell him to focus on playing", "The issue is pushed back onto the player.", {"profile_changes": {"actor": {"state.gm_trust": -6, "state.morale": -3}}, "readiness_changes": [{"who": "actor", "ovr_delta": -1.0, "days": 14, "reason": "Winning concern dismissed"}], "public": False}),
            ],
        })
    else:
        interaction.update({
            "title": f"{name} asks about his development plan",
            "summary": f"{name} wants a clearer path for his next stage of development.",
            "dialogue": [{"speaker": name, "text": "Tell me what the organization needs me to become. I want a real plan."}],
            "choices": [
                _u_choice("honest", "Set a clear development plan", "Specific expectations increase focus.", {"profile_changes": {"actor": {"state.focus": 4, "state.gm_trust": 3}}, "readiness_changes": [{"who": "actor", "ovr_delta": 0.35, "days": 8, "reason": "Clear development plan"}], "potential_changes": [{"who": "actor", "delta": 0.2, "reason": "Development plan alignment"}], "public": False}),
                _u_choice("skills_focus", "Assign focused skills work", "Target one area and build confidence through repetition.", {"profile_changes": {"actor": {"state.confidence": 3}}, "attributes": [{"who": "actor", "attribute": "offensive_awareness", "delta": 0.4, "permanent": False, "duration_games": 8, "reason": "Focused development work"}], "potential_changes": [{"who": "actor", "delta": 0.25, "reason": "Focused development opportunity"}], "public": False}),
                _u_choice("wait", "Tell him to be patient", "No change to the development plan.", {"profile_changes": {"actor": {"state.gm_trust": -2, "state.morale": -1}}, "public": False}),
            ],
        })
    return interaction


def _u_apply_outcome(session: Any, interaction: Dict[str, Any], choice: Dict[str, Any]) -> Dict[str, Any]:
    """V3 outcome router: profile, relationships, stats, readiness, potential, promises and coverage."""
    outcome = dict(choice.get("outcome") or {})
    entities = getattr(session, "universe_players", None) or {}
    actor_id = str(interaction.get("actor_id") or "")
    target_id = str(interaction.get("target_id") or "")
    team_id = str(interaction.get("team_id") or "")
    day, iso, _ = _u_current_meta(session)
    receipts: Dict[str, Any] = {"profiles": [], "relationships": [], "attributes": [], "potential": [], "readiness": [], "reporter": [], "team": []}
    role_ids = {"actor": actor_id, "target": target_id}
    for role, changes in (outcome.get("profile_changes") or {}).items():
        player_id = role_ids.get(str(role), str(role))
        entity = entities.get(player_id)
        if not entity:
            continue
        for field, delta in (changes or {}).items():
            receipt = _u_apply_profile_delta(entity, str(field), float(delta))
            receipt["player_id"] = player_id
            receipts["profiles"].append(receipt)
        _u_add_memory(entity, kind=str(interaction.get("kind") or "interaction"), summary=str(interaction.get("summary") or "Team interaction"), day=day, iso=iso, emotional_delta=sum(float(v) for v in (changes or {}).values()), related_ids=[pid for pid in (actor_id, target_id) if pid and pid != player_id], public=bool(outcome.get("public")))
    if actor_id and target_id and outcome.get("relationship"):
        receipts["relationships"] = _u_change_relationship(session, team_id, actor_id, target_id, dict(outcome["relationship"]), str(interaction.get("title") or "Interaction"))
    room = _u_rebuild_locker_room(session, team_id) if team_id else {}
    culture = room.get("culture") or {}
    for field, delta in (outcome.get("team_changes") or {}).items():
        before = float(culture.get(field, 50.0) or 0.0)
        after = _u_clip(before + float(delta))
        culture[field] = after
        receipts["team"].append({"field": field, "before": before, "after": after, "delta": round(after - before, 2)})
    for change in outcome.get("attributes") or []:
        who = str(change.get("who") or "actor")
        player_id = role_ids.get(who, who)
        if player_id:
            receipts["attributes"].append(_u_apply_attribute_change(session, player_id, change, str(interaction.get("id") or "interaction")))
    for change in outcome.get("potential_changes") or []:
        who = str(change.get("who") or "actor")
        player_id = role_ids.get(who, who)
        if player_id:
            receipts["potential"].append(_u_apply_potential_change(session, player_id, float(change.get("delta") or 0), str(interaction.get("id") or "interaction"), str(change.get("reason") or "Narrative development")))
    for change in outcome.get("readiness_changes") or []:
        who = str(change.get("who") or "actor")
        player_id = role_ids.get(who, who)
        if player_id:
            receipts["readiness"].append(_u_apply_readiness_modifier(session, player_id, source_id=str(interaction.get("id") or "interaction"), ovr_delta=float(change.get("ovr_delta") or 0), days=int(change.get("days") or 5), reason=str(change.get("reason") or "Narrative readiness"), stat_modifiers=dict(change.get("stats") or {})))
    reporter_id = str(interaction.get("reporter_id") or "")
    if reporter_id and actor_id:
        rel = _u_reporter_relationship(session, reporter_id, actor_id)
        for field, delta in (outcome.get("reporter_changes") or {}).items():
            before = float(rel.get(field, 50.0) or 0.0)
            after = _u_clip(before + float(delta))
            rel[field] = after
            receipts["reporter"].append({"field": field, "before": before, "after": after, "delta": round(after - before, 2)})
        rel["interview_count"] = int(rel.get("interview_count", 0) or 0) + 1
        history = list(rel.get("history") or [])
        history.append({"interaction_id": interaction.get("id"), "choice_id": choice.get("id"), "calendar_day": day})
        rel["history"] = history[-20:]
    promise_spec = outcome.get("promise")
    if isinstance(promise_spec, dict):
        promises = list(getattr(session, "universe_promises", None) or [])
        due_games = int(promise_spec.get("due_games") or 5)
        promise = {
            "id": f"promise_{uuid.uuid4().hex[:10]}",
            "interaction_id": interaction.get("id"),
            "type": promise_spec.get("type"),
            "player_id": promise_spec.get("player_id") or actor_id,
            "description": promise_spec.get("description"),
            "created_day": day,
            "games_remaining": due_games,
            "status": "active",
            "progress": 0,
            "success_potential_delta": float(promise_spec.get("success_potential_delta", 0) or 0),
            "success_readiness": float(promise_spec.get("success_readiness", 0) or 0),
            "failure_readiness": float(promise_spec.get("failure_readiness", 0) or 0),
            "success_attribute": promise_spec.get("success_attribute"),
            "success_attribute_delta": float(promise_spec.get("success_attribute_delta", 0) or 0),
        }
        promises.append(promise)
        session.universe_promises = promises[-80:]
        receipts["promise_id"] = promise["id"]
    interaction["status"] = "resolved"
    interaction["resolved_day"] = day
    interaction["resolved_iso"] = iso
    interaction["selected_choice_id"] = choice.get("id")
    interaction["resolution"] = receipts
    interaction["effects"] = {"team_morale": sum(r.get("delta", 0) for r in receipts["team"] if r.get("field") in ("unity", "confidence")), "room_tension": sum(r.get("delta", 0) for r in receipts["team"] if r.get("field") == "tension")}
    cause_type = str(outcome.get("cause_type") or ("PLAYER_INTERACTION" if outcome.get("public") else ""))
    storyline = None
    if bool(outcome.get("public")):
        storyline = _u_record_storyline(session, event={**interaction, "event_tier": "major" if str(interaction.get("stakes")) == "critical" else "minor"}, headline=str(interaction.get("title") or "Team interaction"), summary=str(interaction.get("summary") or "A team interaction became public."), cause_type=cause_type or "PLAYER_INTERACTION", category="locker_room" if "reporter" not in str(interaction.get("kind") or "") else "media", heat=int(outcome.get("heat") or 38), public=True)
        if storyline:
            _u_social_burst(session, storyline, interaction, random.Random(_u_seed(interaction.get("id"), choice.get("id"))))
    _u_append_event(session, {"id": interaction.get("id"), "kind": interaction.get("kind"), "team_id": team_id, "participants": interaction.get("participants"), "choice_id": choice.get("id"), "calendar_day": day, "calendar_iso": iso, "public": bool(outcome.get("public")), "receipts": receipts})
    return {"interaction": interaction, "receipts": receipts, "storyline": storyline}


def _u_queue_or_resolve(session: Any, interaction: Dict[str, Any]) -> None:
    """Queue user meetings and always surface them through the notification system."""
    rows = list(getattr(session, "universe_interactions", None) or [])
    rows.append(interaction)
    session.universe_interactions = rows[-UNIVERSE_MAX_INTERACTIONS:]
    if interaction.get("requires_action"):
        queue = list(getattr(session, "universe_interaction_queue", None) or [])
        if not any(str(row.get("id") or "") == str(interaction.get("id") or "") for row in queue):
            queue.append(interaction)
        session.universe_interaction_queue = queue[-12:]
        stakes = str(interaction.get("stakes") or "medium")
        level = 3 if stakes == "critical" else 2 if stakes == "high" else 1
        _u_notify_user_event(session, {**interaction, "interaction_id": interaction.get("id"), "headline": interaction.get("title"), "notification_title": interaction.get("notification_title") or "PLAYER MEETING REQUEST"}, presentation_level=level)
        return
    choices = list(interaction.get("choices") or [])
    default_id = str(interaction.get("default_choice_id") or "")
    choice = next((row for row in choices if str(row.get("id") or "") == default_id), choices[0] if choices else None)
    if choice:
        _u_apply_outcome(session, interaction, choice)


def _u_generate_daily_interactions(session: Any, rng: random.Random) -> int:
    user_team_id = str(getattr(session, "user_team_id", "") or "")
    created = 0
    team_ids = list((getattr(session, "team_by_id", None) or {}).keys())
    if user_team_id in team_ids:
        team_ids.remove(user_team_id)
        team_ids.insert(0, user_team_id)
    pending = len([row for row in (getattr(session, "universe_interaction_queue", None) or []) if str(row.get("status") or "") == "pending"])
    for team_id_raw in team_ids:
        team_id = str(team_id_raw)
        candidates = _u_interaction_candidates(session, team_id, rng)
        if not candidates:
            continue
        is_user = team_id == user_team_id
        if is_user and pending >= 3:
            continue
        threshold = 40 if is_user else 61
        score, kind, actor_id, target_id = candidates[0]
        roll = rng.uniform(0, 100)
        if score + rng.uniform(-12, 12) < threshold or roll > (60 if is_user else 28):
            continue
        interaction = _u_make_extended_interaction(session, team_id, kind, actor_id, target_id, rng, score)
        _u_queue_or_resolve(session, interaction)
        created += 1
        if is_user:
            pending += 1
            continue
        if created >= 8:
            break
    return created


def _u_tick_promises(session: Any) -> Dict[str, int]:
    """Resolve promises with long-term potential rewards and readiness fallout."""
    promises = list(getattr(session, "universe_promises", None) or [])
    counts = {"kept": 0, "broken": 0}
    day, iso, _ = _u_current_meta(session)
    for promise in promises:
        if str(promise.get("status") or "") != "active" or int(promise.get("games_remaining", 1) or 0) > 0:
            continue
        fulfilled = int(promise.get("progress", 0) or 0) > 0
        promise["status"] = "kept" if fulfilled else "broken"
        promise["resolved_day"] = day
        player_id = str(promise.get("player_id") or "")
        entity = (getattr(session, "universe_players", None) or {}).get(player_id) or {}
        delta = 6 if fulfilled else -12
        _u_apply_profile_delta(entity, "state.gm_trust", delta)
        _u_apply_profile_delta(entity, "state.morale", 3 if fulfilled else -6)
        if fulfilled and float(promise.get("success_potential_delta", 0) or 0):
            _u_apply_potential_change(session, player_id, float(promise.get("success_potential_delta") or 0), str(promise.get("id")), "Management delivered a development/role promise")
        readiness_delta = float(promise.get("success_readiness", 0) if fulfilled else promise.get("failure_readiness", 0) or 0)
        if readiness_delta:
            _u_apply_readiness_modifier(session, player_id, source_id=str(promise.get("id")), ovr_delta=readiness_delta, days=10 if fulfilled else 21, reason=f"Management promise {promise['status']}")
        if fulfilled and promise.get("success_attribute") and float(promise.get("success_attribute_delta", 0) or 0):
            _u_apply_attribute_change(session, player_id, {"attribute": promise.get("success_attribute"), "delta": float(promise.get("success_attribute_delta") or 0), "permanent": False, "duration_games": 10, "reason": "Promise delivered"}, str(promise.get("id")))
        _u_add_memory(entity, kind=promise["status"], summary=f"Management promise {promise['status']}: {promise.get('description')}", day=day, iso=iso, emotional_delta=delta, public=not fulfilled)
        event = {"id": f"uve_{uuid.uuid4().hex[:10]}", "kind": f"promise_{promise['status']}", "team_id": entity.get("team_id"), "participants": [player_id], "player_id": player_id, "player_name": entity.get("player_name"), "event_tier": "minor" if fulfilled else "developing", "tone": "positive" if fulfilled else "negative"}
        _u_append_event(session, {**event, "calendar_day": day, "calendar_iso": iso, "promise_id": promise.get("id")})
        storyline = _u_record_storyline(session, event=event, headline=f"Management promise {promise['status']} with {entity.get('player_name') or 'player'}", summary=str(promise.get("description") or "A management promise reached its deadline."), cause_type="PROMISE_KEPT" if fulfilled else "PROMISE_BROKEN", category="management", heat=32 if fulfilled else 67, public=not fulfilled or bool(promise.get("public")))
        if storyline:
            event["storyline_id"] = storyline.get("storyline_id")
        if str(entity.get("team_id") or "") == str(getattr(session, "user_team_id", "") or ""):
            _u_notify_user_event(session, {**event, "headline": f"Promise {promise['status']}: {entity.get('player_name') or 'player'}", "summary": str(promise.get("description") or "")}, presentation_level=1 if fulfilled else 2)
        counts[promise["status"]] += 1
    session.universe_promises = promises
    return counts


def _trade_fallout_magnitude(session: Any, player: Any, attempt_count: int) -> Tuple[int, int, int, str, int]:
    """Scale trade fallout by future Mental OVR, personality and repeated exposure."""
    player_id = str(getattr(player, "id", "") or "")
    entity = (getattr(session, "universe_players", None) or {}).get(player_id) or {}
    char = _player_character_0_100(player)
    mental = _u_mental_ovr(entity, player)
    personality = entity.get("personality") or {}
    pst = _ensure_player_storyline_state(player)
    heat = int(pst.get("trade_rumor_heat") or 0)
    fragility = max(0.0, min(1.0, (80.0 - mental) / 55.0))
    volatility = float(personality.get("volatility", 50)) / 100.0
    professionalism = float(personality.get("professionalism", 55)) / 100.0
    gm_trust_drop = int(round(3 + attempt_count * 2.2 + fragility * 8 + volatility * 3 - professionalism * 2))
    morale_drop = int(round(2 + attempt_count * 2.0 + fragility * 10 + volatility * 2))
    base_penalty = 0.5 + attempt_count * 0.85 + heat * 0.018
    ovr_mod = int(round(base_penalty * (0.65 + fragility * 2.6)))
    if mental < 45:
        ovr_mod += 2
    if mental < 30 and attempt_count >= 2:
        ovr_mod += 3
    ovr_mod = max(1, min(18, ovr_mod))
    duration_days = int(round(7 + attempt_count * 7 + fragility * 52 + max(0, 50 - char) * 0.35))
    duration_days = max(7, min(120, duration_days))
    severity = "minor" if ovr_mod <= 2 else "mid" if ovr_mod <= 6 else "major"
    pst["gm_trust"] = _clamp(float(pst.get("gm_trust", 0.72)) - gm_trust_drop * 0.01)
    pst["last_trade_mental_ovr"] = round(mental, 1)
    pst["last_trade_readiness_penalty"] = -ovr_mod
    pst["last_trade_recovery_days"] = duration_days
    return morale_drop, gm_trust_drop, ovr_mod, severity, duration_days


def _build_trade_rejected_storyline(
    session: Any,
    *,
    player: Any,
    player_id: str,
    player_name: str,
    team_id: str,
    partner_team_id: str,
    attempt_count: int,
    cause_event_id: str,
    calendar_iso: str,
    cur_day: int,
) -> Optional[Dict[str, Any]]:
    tname = _team_display(session, team_id)
    partner = _team_display(session, partner_team_id)
    morale_drop, gm_trust_drop, ovr_mod, severity, duration_days = _trade_fallout_magnitude(session, player, attempt_count)
    mental = _u_mental_ovr((getattr(session, "universe_players", None) or {}).get(player_id) or {}, player)
    if attempt_count >= 4:
        headline = f"{player_name} camp pushing for clarity after repeated trade talks"
        body = f"{tname} has now included {player_name} in repeated trade proposals. The latest discussion with {partner} failed, and the relationship with management is under real strain."
        cause_type = "PLAYER_REPEATEDLY_SHOPPED"
    elif attempt_count >= 2:
        headline = f"Another failed proposal increases tension around {player_name}"
        body = f"{tname}'s attempt to move {player_name} to {partner} was rejected. This is the {_ordinal(attempt_count)} submitted proposal involving him this season."
        cause_type = "PLAYER_REPEATEDLY_SHOPPED"
    else:
        headline = f"Trade proposal involving {player_name} creates an internal ripple"
        body = f"{tname} included {player_name} in a submitted proposal to {partner}. The deal failed, and the player now has to process the fact that management considered moving him."
        cause_type = "TRADE_REJECTED"
    stable_key = f"{cause_type}|{player_id}|{cause_event_id}"
    return {
        "id": _storyline_id(stable_key),
        "storyline_id": _storyline_id(stable_key),
        "stable_key": stable_key,
        "type": "trade_fallout",
        "category": "trade",
        "cause_type": cause_type,
        "cause_event_id": cause_event_id,
        "culprit_player_id": player_id,
        "player_id": player_id,
        "player_name": player_name,
        "affected_player_ids": [player_id],
        "participants": [player_id],
        "team_id": team_id,
        "team_name": tname,
        "severity": severity,
        "event_tier": "major" if ovr_mod >= 7 else "developing" if ovr_mod >= 3 else "minor",
        "mechanical_severity": min(95, 15 + ovr_mod * 5),
        "priority": "HIGH" if ovr_mod >= 4 else "MEDIUM",
        "tone": "negative",
        "headline": headline,
        "title": headline,
        "description": body,
        "short_summary": body[:180],
        "summary": body,
        "cause": f"User submitted a TradeHub proposal involving {player_name}; {partner} rejected it.",
        "calendar_iso": calendar_iso,
        "calendar_day": cur_day,
        "date": calendar_iso or cur_day,
        "source": "TradeHub Fallout",
        "source_label": "TradeHub Fallout",
        "effects": {"player_morale": -float(morale_drop), "gm_trust": -float(gm_trust_drop), "room_tension": float(2 + min(8, attempt_count)), "trade_market_heat": float(2 + attempt_count)},
        "ovr_modifier": -int(ovr_mod),
        "readiness_duration_days": duration_days,
        "mental_ovr_at_event": round(mental, 1),
        "recovery_conditions": ["Win games together", "Restore a meaningful role", "Meet with player privately", "Trade player if relationship is broken", "Allow time to pass"],
        "resolution_condition": "relationship_recovers_or_player_traded",
        "status": "active",
        "resolved": False,
        "requires_action": attempt_count >= 2 or ovr_mod >= 4,
        "visibility": "team_only",
        "knowledge_type": "fact",
        "public_knowledge_level": "private",
    }


def record_trade_hub_evaluation(
    session: Any,
    evaluation: Dict[str, Any],
    assets_by_team: Dict[str, List[Dict[str, Any]]],
    *,
    proposal_submitted: bool = False,
) -> List[Dict[str, Any]]:
    """Track every submitted user trade proposal and apply Mental-OVR-driven fallout when players learn."""
    if not proposal_submitted:
        return []
    migrate_session_storyline_state(session)
    _u_sync_player_entities(session)
    utid = str(getattr(session, "user_team_id", "") or "")
    if not utid:
        return []
    accepted = bool(evaluation.get("accepted"))
    partner_id = ""
    for tid in (evaluation.get("participating_teams") or evaluation.get("team_ids") or []):
        if str(tid) != utid:
            partner_id = str(tid)
            break
    if not partner_id:
        for tid in (assets_by_team or {}).keys():
            if str(tid) != utid:
                partner_id = str(tid)
                break
    generated: List[Dict[str, Any]] = []
    cur, iso, season = _u_current_meta(session)
    rejection_kind = _classify_trade_rumor_verdict(evaluation)
    for pid, pname, player in _outgoing_user_players(session, assets_by_team):
        pst = _ensure_player_storyline_state(player)
        pst["proposal_submission_count"] = int(pst.get("proposal_submission_count", 0) or 0) + 1
        pst["last_proposed_day"] = cur
        pst["last_trade_rumor_context"] = _trade_rumor_context_key(partner_id, str(evaluation.get("verdict") or ""))
        eid = record_decision_event(session, {"event_type": "TRADE_ATTEMPTED_BY_USER", "date": iso or cur, "calendar_day": cur, "team_id": utid, "player_ids": [pid], "player_id": pid, "player_name": pname, "target_team_id": partner_id, "accepted": accepted, "proposal_submitted": True, "proposal_submission_count": pst["proposal_submission_count"]})
        if accepted:
            continue
        if rejection_kind == "technical_no_fallout":
            continue
        if rejection_kind == "soft_blocked":
            pst["trade_rumor_heat"] = min(100, int(pst.get("trade_rumor_heat") or 0) + 1)
            continue
        cooldown_active = _is_trade_rumor_cooldown_active(pst, cur_day=cur, season=season)
        if cooldown_active:
            continue
        pst["trade_attempt_count"] = int(pst.get("trade_attempt_count") or 0) + 1
        pst["trade_rumor_heat"] = min(100, int(pst.get("trade_rumor_heat") or 0) + 12)
        pst["last_trade_rumor_day"] = cur
        pst["last_trade_rumor_week"] = cur // 7
        pst["last_trade_rumor_season"] = season
        attempt_n = int(pst["trade_attempt_count"])
        sl = _build_trade_rejected_storyline(session, player=player, player_id=pid, player_name=pname, team_id=utid, partner_team_id=partner_id, attempt_count=attempt_n, cause_event_id=eid, calendar_iso=iso, cur_day=cur)
        if not sl:
            continue
        _apply_storyline_effects(session, utid, pid, dict(sl.get("effects") or {}))
        readiness = _u_apply_readiness_modifier(session, pid, source_id=eid, ovr_delta=float(sl.get("ovr_modifier") or 0), days=int(sl.get("readiness_duration_days") or 14), reason=str(sl.get("cause") or "Trade proposal fallout"), stat_modifiers={"focus": -0.25 * abs(float(sl.get("ovr_modifier") or 0)), "shot_involvement": -0.004 * abs(float(sl.get("ovr_modifier") or 0)), "assist_involvement": -0.003 * abs(float(sl.get("ovr_modifier") or 0))})
        event = {**sl, "id": f"trade_exposure_{pid}_{cur}_{attempt_n}", "kind": "trade_proposal_exposure", "participants": [pid], "record_private_storyline": True, "evidence": {"readiness": readiness, "attempt_count": attempt_n}}
        story = _u_record_storyline(session, event=event, headline=sl["headline"], summary=sl["description"], cause_type="TRADE_PROPOSAL_EXPOSURE", category="trade", heat=35 + min(35, attempt_n * 9), public=False)
        if story:
            sl["storyline_id"] = story.get("storyline_id")
        _u_notify_user_event(session, {**event, "headline": sl["headline"], "summary": sl["description"], "requires_action": bool(sl.get("requires_action"))}, presentation_level=2 if sl.get("requires_action") else 1)
        generated.append(sl)
        try:
            from app.sim_engine.franchise.trade_stability_engine import apply_trade_hub_exposure  # noqa: WPS433

            apply_trade_hub_exposure(
                session,
                player,
                attempt_n=attempt_n,
                rejection_kind=rejection_kind,
            )
        except Exception:
            pass
    return generated


def _u_create_trade_demand(session: Any, player_id: str, reason: str, rng: random.Random) -> Optional[Dict[str, Any]]:
    """Create a private major trade demand; low-character players can leak immediately."""
    demands = list(getattr(session, "universe_trade_demands", None) or [])
    if any(str(row.get("player_id") or "") == str(player_id) and str(row.get("status") or "") == "active" for row in demands):
        return None
    entity = (getattr(session, "universe_players", None) or {}).get(str(player_id)) or {}
    player = _player_from_roster(session, str(player_id))
    if not entity or player is None:
        return None
    team_id = str(entity.get("team_id") or "")
    day, iso, season = _u_current_meta(session)
    character = float((entity.get("personality") or {}).get("character", _player_character_0_100(player)))
    mental = _u_mental_ovr(entity, player)
    agent = _agent_for_player(session, str(player_id), rng)
    leak_tendency = float(agent.get("leak_tendency") or 0.3)
    immediate_leak_chance = 0.0
    if character < 35:
        immediate_leak_chance = min(0.92, 0.62 + leak_tendency * 0.30)
    elif character < 45:
        immediate_leak_chance = 0.18 + leak_tendency * 0.18
    demand_id = f"trade_demand_{player_id}_{season}_{day}"
    demand = {
        "id": demand_id,
        "kind": "private_trade_demand",
        "event_tier": "major",
        "mechanical_severity": 72,
        "team_id": team_id,
        "player_id": str(player_id),
        "player_name": entity.get("player_name"),
        "participants": [str(player_id)],
        "headline": f"{entity.get('player_name') or 'Player'} formally requests a trade",
        "summary": str(reason),
        "reason": str(reason),
        "created_day": day,
        "calendar_iso": iso,
        "leak_day": day + 7,
        "status": "active",
        "public": False,
        "visibility": "gm_private",
        "knowledge_type": "fact",
        "public_knowledge_level": "private",
        "agent_id": agent.get("id"),
        "mental_ovr": round(mental, 1),
        "character": round(character, 1),
        "immediate_leak_chance": round(immediate_leak_chance, 3),
    }
    demands.append(demand)
    session.universe_trade_demands = demands[-60:]
    _u_apply_profile_delta(entity, "state.gm_trust", -12)
    _u_apply_profile_delta(entity, "state.morale", -8)
    penalty = -max(3.0, min(12.0, 3.0 + (70.0 - mental) * 0.16))
    _u_apply_readiness_modifier(session, str(player_id), source_id=demand_id, ovr_delta=penalty, days=45 if mental >= 55 else 75, reason="Unresolved trade demand", stat_modifiers={"focus": -1.2, "shot_involvement": -0.04, "assist_involvement": -0.035})
    _u_append_event(session, {**demand, "calendar_day": day})
    if team_id == str(getattr(session, "user_team_id", "") or ""):
        _u_notify_user_event(session, {**demand, "notification_title": "URGENT PLAYER MEETING", "requires_action": True}, presentation_level=3)
    if rng.random() < immediate_leak_chance:
        _u_publish_trade_demand(session, demand, leaked_by="player_camp")
    return demand


def _u_publish_trade_demand(session: Any, demand: Dict[str, Any], *, leaked_by: str) -> Optional[Dict[str, Any]]:
    if bool(demand.get("public")):
        return None
    demand["public"] = True
    demand["visibility"] = "public"
    demand["public_knowledge_level"] = "widely_reported"
    demand["knowledge_type"] = "corroborated_claim"
    demand["leaked_by"] = leaked_by
    name = str(demand.get("player_name") or "Player")
    team_id = str(demand.get("team_id") or "")
    headline = f"Sources: {name} has asked {_team_display(session, team_id)} for a trade"
    summary = f"Multiple sources now say {name} formally requested a move. Management had been given time to handle the matter privately before it became public."
    event = {**demand, "headline": headline, "summary": summary, "event_tier": "major", "mechanical_severity": 78, "source_label": "Mark Ellison · NorthStar Hockey", "reporter_name": "Mark Ellison", "outlet_name": "NorthStar Hockey", "credibility": 88}
    storyline = _u_record_storyline(session, event=event, headline=headline, summary=summary, cause_type="TRADE_DEMAND", category="trade", heat=88, public=True)
    if storyline:
        demand["storyline_id"] = storyline.get("storyline_id")
        event["storyline_id"] = storyline.get("storyline_id")
    _u_notify_user_event(session, event, presentation_level=4, force_league=True)
    return storyline


def _u_tick_trade_demands(session: Any, rng: random.Random) -> Dict[str, int]:
    day = _u_current_meta(session)[0]
    leaked = 0
    active = 0
    for demand in list(getattr(session, "universe_trade_demands", None) or []):
        if str(demand.get("status") or "") != "active":
            continue
        active += 1
        if not bool(demand.get("public")) and day >= int(demand.get("leak_day", day + 1) or day + 1):
            _u_publish_trade_demand(session, demand, leaked_by="reporter_deadline")
            leaked += 1
    return {"active": active, "leaked": leaked}


def _u_maybe_create_trade_demand_from_state(session: Any, rng: random.Random) -> int:
    """Deprecated — unified trade_demand_engine.process_trade_demand_day owns formal demands."""
    return 0

    # Legacy V3 path retained below for reference but unreachable.
    created = 0
    season = _u_current_meta(session)[2]
    for team_id, player in _u_all_players(session):
        player_id = str(getattr(player, "id", "") or "")
        entity = (getattr(session, "universe_players", None) or {}).get(player_id) or {}
        state = entity.get("state") or {}
        concerns = entity.get("concerns") or {}
        pst = _ensure_player_storyline_state(player)
        if int(pst.get("trade_demand_season", -1) or -1) == season:
            continue
        role_sat = float(state.get("role_satisfaction", 60))
        gm_trust = float(state.get("gm_trust", 60))
        winning_sat = float((concerns.get("winning") or {}).get("satisfaction", 60))
        attempts = int(pst.get("trade_attempt_count", 0) or 0)
        competitiveness = float((entity.get("personality") or {}).get("competitiveness", 55))
        severe = (role_sat <= 24 and gm_trust <= 42) or attempts >= 3 or (winning_sat <= 23 and competitiveness >= 78 and gm_trust <= 48)
        if not severe or rng.random() > 0.055:
            continue
        reason = "The player no longer believes his role and relationship with management can be repaired." if role_sat <= 24 else "Repeated trade exposure has convinced the player that a move is necessary." if attempts >= 3 else "The player no longer believes the club's competitive direction matches his career goals."
        if _u_create_trade_demand(session, player_id, reason, rng):
            pst["trade_demand_season"] = season
            created += 1
            if created >= 1:
                break
    return created


def resolve_universe_trade_demands_after_trade(session: Any, moved_players: List[Dict[str, Any]]) -> int:
    """Call beside resolve_culprit_traded_storylines after a completed trade."""
    moved = {str(row.get("asset_id") or row.get("player_id") or "") for row in moved_players or []}
    resolved = 0
    for demand in getattr(session, "universe_trade_demands", None) or []:
        if str(demand.get("player_id") or "") in moved and str(demand.get("status") or "") == "active":
            demand["status"] = "resolved"
            demand["resolution"] = "player_traded"
            demand["resolved_day"] = _u_current_meta(session)[0]
            resolved += 1
    return resolved


def _u_set_player_availability(session: Any, player_id: str, *, status: str, reason: str, days: int = 0, games: int = 0) -> Dict[str, Any]:
    store = dict(getattr(session, "universe_player_availability", None) or {})
    row = {"player_id": str(player_id), "status": str(status), "reason": str(reason), "days_remaining": max(0, int(days)), "games_remaining": max(0, int(games)), "active": True}
    store[str(player_id)] = row
    session.universe_player_availability = store
    return row


def universe_player_is_available(session: Any, player_id: str) -> bool:
    row = (getattr(session, "universe_player_availability", None) or {}).get(str(player_id)) or {}
    if not row or not row.get("active"):
        return True
    return str(row.get("status") or "available") not in ("suspended", "banned", "investigative_leave", "deceased")


def _u_tick_player_availability_days(session: Any) -> int:
    store = dict(getattr(session, "universe_player_availability", None) or {})
    cleared = 0
    for row in store.values():
        if not row.get("active"):
            continue
        if int(row.get("days_remaining", 0) or 0) > 0:
            row["days_remaining"] = int(row.get("days_remaining") or 0) - 1
        if int(row.get("days_remaining", 0) or 0) <= 0 and int(row.get("games_remaining", 0) or 0) <= 0 and str(row.get("status")) != "deceased":
            row["active"] = False
            row["status"] = "available"
            cleared += 1
    session.universe_player_availability = store
    return cleared


def tick_universe_player_availability_after_game(session: Any, player_ids: List[str]) -> int:
    """Game-loop hook: decrement suspension/ban games for the involved clubs."""
    store = dict(getattr(session, "universe_player_availability", None) or {})
    changed = 0
    for player_id in player_ids or []:
        row = store.get(str(player_id))
        if not row or not row.get("active"):
            continue
        if int(row.get("games_remaining", 0) or 0) > 0:
            row["games_remaining"] = int(row.get("games_remaining") or 0) - 1
            changed += 1
        if int(row.get("games_remaining", 0) or 0) <= 0 and int(row.get("days_remaining", 0) or 0) <= 0 and str(row.get("status")) != "deceased":
            row["active"] = False
            row["status"] = "available"
    session.universe_player_availability = store
    return changed


def _u_apply_team_sanction(session: Any, team_id: str, event_type: str, rng: random.Random, event_id: str) -> Dict[str, Any]:
    """Persist fictional league sanctions; cap/draft engines can consume the dedicated ledgers."""
    sanctions = list(getattr(session, "universe_team_sanctions", None) or [])
    cap_penalties = dict(getattr(session, "universe_cap_penalties", None) or {})
    forfeited = list(getattr(session, "universe_forfeited_picks", None) or [])
    season = _u_current_meta(session)[2]
    sanction: Dict[str, Any] = {"id": f"sanction_{uuid.uuid4().hex[:10]}", "source_event_id": event_id, "team_id": str(team_id), "event_type": event_type, "season": season, "active": True, "fine_m": 0.0, "cap_penalty_m": 0.0, "forfeited_picks": []}
    if event_type == "ILLEGAL_TEAM_WORKOUTS":
        sanction["fine_m"] = round(rng.uniform(0.25, 1.5), 2)
        if rng.random() < 0.18:
            pick = {"team_id": str(team_id), "draft_year": season + 1, "round": 2, "source_event_id": event_id}
            forfeited.append(pick)
            sanction["forfeited_picks"].append(pick)
    elif event_type == "UNDER_TABLE_PAYMENTS":
        sanction["fine_m"] = round(rng.uniform(2.0, 8.0), 2)
        sanction["cap_penalty_m"] = round(rng.uniform(3.0, 7.5), 2)
        pick = {"team_id": str(team_id), "draft_year": season + 1, "round": 1, "source_event_id": event_id}
        forfeited.append(pick)
        sanction["forfeited_picks"].append(pick)
    elif event_type == "CAP_CIRCUMVENTION":
        sanction["fine_m"] = round(rng.uniform(1.0, 5.0), 2)
        sanction["cap_penalty_m"] = round(rng.uniform(2.0, 6.0), 2)
        if rng.random() < 0.55:
            pick = {"team_id": str(team_id), "draft_year": season + 1, "round": 1, "source_event_id": event_id}
            forfeited.append(pick)
            sanction["forfeited_picks"].append(pick)
    else:
        sanction["fine_m"] = round(rng.uniform(0.5, 4.0), 2)
    if sanction["cap_penalty_m"]:
        cap_penalties[str(team_id)] = round(float(cap_penalties.get(str(team_id), 0) or 0) + float(sanction["cap_penalty_m"]), 2)
    sanctions.append(sanction)
    session.universe_team_sanctions = sanctions[-80:]
    session.universe_cap_penalties = cap_penalties
    session.universe_forfeited_picks = forfeited[-80:]
    return sanction


def _u_major_event_state_for_season(session: Any, rng: random.Random) -> Dict[str, Any]:
    season = _u_current_meta(session)[2]
    state = dict(getattr(session, "universe_major_event_state", None) or {})
    if int(state.get("season", -1) or -1) != season:
        deterministic = random.Random(_u_seed("major_event_budget", season))
        state = {"season": season, "target": deterministic.randint(UNIVERSE_MAJOR_EVENTS_MIN, UNIVERSE_MAJOR_EVENTS_MAX), "generated": 0, "last_event_day": -999, "event_ids": []}
        session.universe_major_event_state = state
    return state


def _u_run_major_league_event_pass(session: Any, rng: random.Random) -> int:
    """League-wide scheduler: 3-5 major incidents per season, never 3-5 per team."""
    state = _u_major_event_state_for_season(session, rng)
    target = int(state.get("target", 4) or 4)
    generated = int(state.get("generated", 0) or 0)
    if generated >= target:
        return 0
    day, iso, season = _u_current_meta(session)
    if day - int(state.get("last_event_day", -999) or -999) < 14:
        return 0
    calendar_len = max(day + 1, len(getattr(session, "nhl_calendar", None) or []))
    remaining_days = max(1, calendar_len - day)
    remaining_events = max(1, target - generated)
    chance_today = min(0.12, max(0.0035, remaining_events / remaining_days * 1.25))
    if rng.random() > chance_today:
        return 0
    teams = [(str(tid), tm) for tid, tm in (getattr(session, "team_by_id", None) or {}).items()]
    if not teams:
        return 0
    event_id = f"major_{season}_{day}_{uuid.uuid4().hex[:8]}"
    is_team_event = rng.random() < 0.24
    storyline = None
    event: Dict[str, Any]
    if is_team_event:
        team_id, _ = rng.choice(teams)
        choices = [
            ("ILLEGAL_TEAM_WORKOUTS", 0.34, "League investigation finds prohibited team workouts", "The league has sanctioned the club after determining that prohibited workouts violated league rules."),
            ("UNDER_TABLE_PAYMENTS", 0.18, "League uncovers undisclosed player compensation", "Investigators found compensation outside registered contracts, triggering major financial, cap and draft sanctions."),
            ("CAP_CIRCUMVENTION", 0.28, "League sanctions club for cap circumvention", "A league investigation found an improper attempt to work around salary-cap rules."),
            ("EXECUTIVE_MISCONDUCT", 0.20, "Team executive misconduct triggers league discipline", "The league announced sanctions after an investigation into serious front-office misconduct."),
        ]
        roll = rng.random()
        acc = 0.0
        chosen = choices[-1]
        for row in choices:
            acc += row[1]
            if roll <= acc:
                chosen = row
                break
        event_type, _, headline, summary = chosen
        sanction = _u_apply_team_sanction(session, team_id, event_type, rng, event_id)
        team = (getattr(session, "team_by_id", None) or {}).get(team_id)
        state_obj = getattr(team, "state", None) if team is not None else None
        if state_obj is not None:
            try:
                state_obj.team_morale = _clamp(float(getattr(state_obj, "team_morale", 0.5)) - 0.08)
                state_obj.organizational_pressure = _clamp(float(getattr(state_obj, "organizational_pressure", 0.5)) + 0.18)
            except Exception:
                pass
        event = {"id": event_id, "kind": event_type.lower(), "event_tier": "major", "mechanical_severity": 88 if event_type in ("UNDER_TABLE_PAYMENTS", "CAP_CIRCUMVENTION") else 76, "team_id": team_id, "headline": headline, "summary": summary, "tone": "negative", "visibility": "public", "knowledge_type": "fact", "public_knowledge_level": "confirmed", "source_label": "League Office", "evidence": {"sanction": sanction}, "incident_family": "organizational_misconduct"}
        storyline = _u_record_storyline(session, event=event, headline=headline, summary=summary, cause_type=event_type, category="league_discipline", heat=94 if event_type in ("UNDER_TABLE_PAYMENTS", "CAP_CIRCUMVENTION") else 84, public=True)
    else:
        nhl_players: List[Tuple[str, Any]] = []
        for team_id, team in teams:
            for player in list(getattr(team, "roster", None) or []):
                if not getattr(player, "retired", False):
                    nhl_players.append((team_id, player))
        if not nhl_players:
            return 0
        team_id, player = rng.choice(nhl_players)
        player_id = str(getattr(player, "id", "") or "")
        entity = (getattr(session, "universe_players", None) or {}).get(player_id) or {}
        name = str(entity.get("player_name") or _u_name(player))
        # PLAYER_DEATH has weight 0.01 against a ~4.5 total pool: exceptionally rare.
        pool = [
            ("PLAYER_ARRESTED", 1.35),
            ("LEAGUE_SUSPENSION", 1.25),
            ("PLAYER_BANNED", 0.35),
            ("MAJOR_PUBLIC_ALTERCATION", 1.15),
            ("GAMBLING_VIOLATION", 0.45),
            ("PLAYER_DEATH", 0.01),
        ]
        total = sum(weight for _, weight in pool)
        pick = rng.random() * total
        acc = 0.0
        event_type = pool[-1][0]
        for key, weight in pool:
            acc += weight
            if pick <= acc:
                event_type = key
                break
        mental = _u_mental_ovr(entity, player)
        if event_type == "PLAYER_ARRESTED":
            headline = f"{name} arrested; league and team reviewing situation"
            summary = f"The club has confirmed that {name} was arrested. The player is away from team activities while the matter is reviewed."
            availability = _u_set_player_availability(session, player_id, status="investigative_leave", reason="Arrest / league review", days=rng.randint(5, 14))
            ovr_delta, days = -max(8, min(18, 11 + (60 - mental) * 0.12)), rng.randint(60, 120)
        elif event_type == "LEAGUE_SUSPENSION":
            games = rng.randint(8, 30)
            headline = f"League suspends {name} for {games} games"
            summary = f"The league announced a {games}-game suspension for {name}. The player is immediately unavailable."
            availability = _u_set_player_availability(session, player_id, status="suspended", reason="League suspension", games=games)
            ovr_delta, days = -max(6, min(16, 8 + (58 - mental) * 0.10)), rng.randint(45, 90)
        elif event_type == "PLAYER_BANNED":
            days_banned = rng.randint(180, 365)
            headline = f"League imposes long-term ban on {name}"
            summary = f"The league has banned {name} from competition while a major disciplinary matter is resolved."
            availability = _u_set_player_availability(session, player_id, status="banned", reason="Long-term league ban", days=days_banned)
            ovr_delta, days = -max(12, min(22, 15 + (55 - mental) * 0.12)), days_banned
        elif event_type == "MAJOR_PUBLIC_ALTERCATION":
            games = rng.randint(3, 8)
            headline = f"Major public altercation puts {name} under league review"
            summary = f"Video of a serious public altercation involving {name} has triggered immediate team and league discipline."
            availability = _u_set_player_availability(session, player_id, status="suspended", reason="Major public altercation", games=games)
            ovr_delta, days = -max(7, min(16, 9 + (55 - mental) * 0.10)), rng.randint(30, 75)
        elif event_type == "GAMBLING_VIOLATION":
            games = rng.randint(20, 50)
            headline = f"League announces major gambling-policy suspension for {name}"
            summary = f"A league investigation found a serious gambling-policy violation. {name} has been suspended for {games} games."
            availability = _u_set_player_availability(session, player_id, status="suspended", reason="Gambling-policy violation", games=games)
            ovr_delta, days = -max(10, min(20, 13 + (55 - mental) * 0.11)), rng.randint(90, 180)
        else:
            headline = f"League mourns the death of {name}"
            summary = f"The organization and league have announced that {name} has died. The player is permanently removed from competition."
            availability = _u_set_player_availability(session, player_id, status="deceased", reason="Player deceased")
            try:
                setattr(player, "retired", True)
                setattr(player, "career_ended", True)
                setattr(player, "_universe_deceased", True)
            except Exception:
                pass
            entity.setdefault("life", {})["status"] = "deceased"
            entity["active_roster"] = False
            ovr_delta, days = 0.0, 1
        readiness = None
        if ovr_delta:
            readiness = _u_apply_readiness_modifier(session, player_id, source_id=event_id, ovr_delta=ovr_delta, days=days, reason=summary, stat_modifiers={"focus": -2.0, "composure": -1.5, "shot_involvement": -0.06, "assist_involvement": -0.05})
        _u_apply_profile_delta(entity, "state.morale", -14 if event_type != "PLAYER_DEATH" else 0)
        _u_apply_profile_delta(entity, "state.confidence", -10 if event_type != "PLAYER_DEATH" else 0)
        _u_apply_profile_delta(entity, "state.media_stress", 25 if event_type != "PLAYER_DEATH" else 0)
        event = {"id": event_id, "kind": event_type.lower(), "event_tier": "catastrophic" if event_type == "PLAYER_DEATH" else "major", "mechanical_severity": 100 if event_type == "PLAYER_DEATH" else 90, "team_id": team_id, "player_id": player_id, "player_name": name, "participants": [player_id], "headline": headline, "summary": summary, "tone": "negative", "visibility": "public", "knowledge_type": "fact", "public_knowledge_level": "confirmed", "source_label": "League Desk", "incident_family": "major_player_conduct", "evidence": {"availability": availability, "readiness": readiness, "mental_ovr": mental}}
        storyline = _u_record_storyline(session, event=event, headline=headline, summary=summary, cause_type=event_type, category="major_news", heat=100 if event_type == "PLAYER_DEATH" else 94, public=True)
        # Team-wide shock for the most serious player events.
        tm = (getattr(session, "team_by_id", None) or {}).get(team_id)
        st = getattr(tm, "state", None) if tm is not None else None
        if st is not None:
            try:
                shock = 0.20 if event_type == "PLAYER_DEATH" else 0.07
                st.team_morale = _clamp(float(getattr(st, "team_morale", 0.5)) - shock)
                st.organizational_pressure = _clamp(float(getattr(st, "organizational_pressure", 0.5)) + (0.20 if event_type != "PLAYER_DEATH" else 0.12))
            except Exception:
                pass
    if storyline:
        event["storyline_id"] = storyline.get("storyline_id")
    _u_append_event(session, {**event, "calendar_day": day, "calendar_iso": iso})
    _u_notify_user_event(session, event, presentation_level=4, force_league=True)
    state["generated"] = generated + 1
    state["last_event_day"] = day
    ids = list(state.get("event_ids") or [])
    ids.append(event_id)
    state["event_ids"] = ids[-10:]
    session.universe_major_event_state = state
    return 1


def _u_tick_player_life(session: Any, team_id: str, entity: Dict[str, Any], rng: random.Random) -> None:
    """State simulation only. Actual discrete life events are generated separately by V3."""
    day, iso, season = _u_current_meta(session)
    if int(entity.get("last_tick_day", -1) or -1) == day:
        return
    state = entity.get("state") or {}
    life = entity.get("life") or {}
    personality = entity.get("personality") or {}
    state["character_modifier"] = round(float(state.get("character_modifier", 0) or 0) * 0.992, 3)
    previous_role = float(state.get("role_satisfaction", 58))
    role_target = _u_infer_role_satisfaction(session, team_id, entity)
    state["role_satisfaction"] = _u_clip(previous_role + (role_target - previous_role) * 0.18)
    home_stability = float(life.get("home_stability", 65))
    relocation = float(life.get("relocation_strain", 15))
    sleep = float(life.get("sleep_quality", 68))
    personal_target = _u_clip(30 + relocation * 0.30 + max(0, 60 - home_stability) * 0.36 + max(0, 62 - sleep) * 0.28)
    state["personal_stress"] = _u_clip(float(state.get("personal_stress", 30)) * 0.86 + personal_target * 0.14 + rng.uniform(-1.8, 1.8))
    state["energy"] = _u_clip(float(state.get("energy", 70)) + (sleep - 65) * 0.035 - float(state.get("personal_stress", 30)) * 0.012 + rng.uniform(-1.2, 1.2))
    state["focus"] = _u_clip(48 + float(state.get("confidence", 55)) * 0.24 + float(state.get("energy", 70)) * 0.20 - float(state.get("personal_stress", 30)) * 0.14)
    life["relocation_strain"] = _u_clip(relocation - 0.18)
    life["community_connection"] = _u_clip(float(life.get("community_connection", 35)) + rng.uniform(0, 0.25))
    life["current_note"] = "Travel and home responsibilities are competing with hockey focus." if float(state.get("personal_stress", 0)) >= 68 else "Home life is stable."
    concerns = entity.get("concerns") or {}
    role_concern = concerns.get("role") or {}
    role_concern["trend"] = round(float(state.get("role_satisfaction", 58)) - previous_role, 2)
    role_concern["satisfaction"] = float(state.get("role_satisfaction", 58))
    concerns["role"] = role_concern
    home_concern = concerns.get("home_life") or {}
    home_before = float(home_concern.get("satisfaction", 65))
    home_after = _u_clip(home_stability - relocation * 0.35)
    home_concern.update({"satisfaction": home_after, "trend": round(home_after - home_before, 2)})
    concerns["home_life"] = home_concern
    years = _u_contract_years(_player_from_roster(session, str(entity.get("player_id") or "")) or object(), season)
    contract_concern = concerns.get("contract") or {}
    old_contract = float(contract_concern.get("satisfaction", 55))
    contract_after = _u_clip(38 + years * 18 - max(0, float(personality.get("ambition", 50)) - 70) * 0.18)
    contract_concern.update({"satisfaction": contract_after, "trend": round(contract_after - old_contract, 2)})
    concerns["contract"] = contract_concern
    w, l, o, _ = _team_record(session, team_id)
    gp = w + l + o
    points_pct = (w * 2 + o) / max(2, gp * 2)
    winning = concerns.get("winning") or {}
    old_win = float(winning.get("satisfaction", 55))
    win_after = _u_clip(25 + points_pct * 70)
    winning.update({"satisfaction": win_after, "trend": round(win_after - old_win, 2)})
    concerns["winning"] = winning
    belonging = concerns.get("team_belonging") or {}
    old_belong = float(belonging.get("satisfaction", 60))
    belong_after = float(state.get("belonging", 60))
    belonging.update({"satisfaction": belong_after, "trend": round(belong_after - old_belong, 2)})
    concerns["team_belonging"] = belonging
    entity["top_concerns"] = sorted([{"id": key, **value, "pressure": round(float(value.get("importance", 50)) * (100 - float(value.get("satisfaction", 50))) / 100, 1)} for key, value in concerns.items()], key=lambda row: (-float(row.get("pressure", 0)), row["id"]))[:3]
    _u_tick_mental_wellbeing(entity, day)
    try:
        from app.sim_engine.franchise.player_agent_engine import ensure_player_agent  # noqa: WPS433

        player_obj = _player_from_roster(session, str(entity.get("player_id") or ""))
        agent_id = ""
        if player_obj is not None:
            agent_id = str((ensure_player_agent(player_obj, session) or {}).get("id") or "")
        agent_trust = _u_agent_org_trust(session, agent_id, str(team_id)) if agent_id else 55.0
    except Exception:
        agent_trust = 55.0
    room = (getattr(session, "universe_locker_rooms", None) or {}).get(str(team_id)) or {}
    entity["human_pressure"] = _u_compute_human_pressure(entity, room=room, agent_org_trust=agent_trust)
    entity["state"] = state
    entity["life"] = life
    entity["concerns"] = concerns
    entity["last_tick_day"] = day
    entity["last_updated_iso"] = iso


def narrative_universe_v2_daily_pass(session: Any, calendar_idx: int, day_meta: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """V3 daily director: lives, meetings, trade-demand deadlines, minor events and 3-5 league-wide major events."""
    _u_migrate_v2(session)
    if int(getattr(session, "_universe_last_daily_tick", -1) or -1) == int(calendar_idx):
        return {"already_processed": True, "calendar_day": int(calendar_idx), "generated_interactions": 0}
    entities = _u_sync_player_entities(session)
    season = int(getattr(session, "season_calendar_year", 2025) or 2025)
    local_rng = rng or random.Random(_u_seed("universe_daily_v3", season, calendar_idx, getattr(session, "user_team_id", "")))
    readiness_expired = _u_tick_readiness_modifiers(session)
    availability_cleared = _u_tick_player_availability_days(session)
    for team_id, player in _u_all_players(session):
        entity = entities.get(str(getattr(player, "id", "") or ""))
        if entity:
            _u_tick_player_life(session, team_id, entity, local_rng)
    for team_id in (getattr(session, "team_by_id", None) or {}).keys():
        _u_rebuild_locker_room(session, str(team_id))
    minor_life = _u_generate_minor_life_events(session, local_rng)
    expired = _u_expire_interactions(session)
    interactions = _u_generate_daily_interactions(session, local_rng)
    trade_demands_created = _u_maybe_create_trade_demand_from_state(session, local_rng)
    trade_demand_tick = _u_tick_trade_demands(session, local_rng)
    major_events = _u_run_major_league_event_pass(session, local_rng)
    social_created = _u_generate_ambient_social(session, local_rng)
    promise_counts = _u_tick_promises(session)
    coverage_stats: Dict[str, Any] = {}
    try:
        from app.sim_engine.franchise.storyline_coverage import run_coverage_daily_pass  # noqa: WPS433
        coverage_stats = run_coverage_daily_pass(session, local_rng)
    except Exception:
        coverage_stats = {}
    user_team_id = str(getattr(session, "user_team_id", "") or "")
    user_room = (getattr(session, "universe_locker_rooms", None) or {}).get(user_team_id) or {}
    snapshots = list(getattr(session, "universe_daily_snapshots", None) or [])
    snapshots.append({"calendar_day": int(calendar_idx), "calendar_iso": str(day_meta.get("iso") or ""), "team_id": user_team_id, "culture": dict(user_room.get("culture") or {}), "pending_interactions": len(getattr(session, "universe_interaction_queue", None) or []), "minor_life_events": minor_life, "major_events": major_events})
    session.universe_daily_snapshots = snapshots[-120:]
    session._universe_last_daily_tick = int(calendar_idx)
    return {
        "already_processed": False,
        "calendar_day": int(calendar_idx),
        "player_entities": len(entities),
        "locker_rooms": len(getattr(session, "universe_locker_rooms", None) or {}),
        "generated_interactions": interactions,
        "minor_life_events": minor_life,
        "major_league_events": major_events,
        "trade_demands_created": trade_demands_created,
        "trade_demands_active": trade_demand_tick["active"],
        "trade_demands_leaked": trade_demand_tick["leaked"],
        "expired_interactions": expired,
        "readiness_modifiers_expired": readiness_expired,
        "availability_cleared": availability_cleared,
        "ambient_social_created": social_created,
        "promises_kept": promise_counts["kept"],
        "promises_broken": promise_counts["broken"],
        "pending_interactions": len(getattr(session, "universe_interaction_queue", None) or []),
        "coverage": coverage_stats,
    }


def build_universe_game_context(session: Any, team_id: str, opponent_id: str = "", game_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """V3 sim bridge with effective OVR and stat-specific fingerprints for realData generation."""
    migrate_session_storyline_state(session)
    _u_sync_player_entities(session)
    room = _u_rebuild_locker_room(session, str(team_id))
    culture = room.get("culture") or {}
    entities = getattr(session, "universe_players", None) or {}
    player_modifiers: Dict[str, Dict[str, Any]] = {}
    hidden_contributors: List[Dict[str, Any]] = []
    risks: List[Dict[str, Any]] = []
    glue_count = 0
    disruptor_tax = 0.0
    readiness_team_tax = 0.0
    for player in _u_team_players(session, str(team_id)):
        player_id = str(getattr(player, "id", "") or "")
        entity = entities.get(player_id) or {}
        state = entity.get("state") or {}
        personality = entity.get("personality") or {}
        niches = _u_niche_ids(entity)
        morale = float(state.get("morale", 55))
        confidence = float(state.get("confidence", 55))
        focus = float(state.get("focus", 60))
        energy = float(state.get("energy", 70))
        character = float(personality.get("character", 55))
        role_satisfaction = float(state.get("role_satisfaction", 55))
        mental = _u_mental_ovr(entity, player)
        effort = (morale - 50) * 0.012 + (focus - 50) * 0.010 + (energy - 65) * 0.008
        composure = (confidence - 50) * 0.012 + (float(personality.get("resilience", 50)) - 50) * 0.009 + (mental - 55) * 0.004
        discipline = (character - 50) * 0.007 - max(0, float(personality.get("volatility", 50)) - 60) * 0.012
        passing = 0.0
        defensive_effort = 0.0
        penalty_risk = 0.0
        if character < 45 and role_satisfaction < 46:
            defensive_effort -= (45 - character) * 0.020 + (46 - role_satisfaction) * 0.014
            passing -= max(0, float(personality.get("ego", 50)) - 60) * 0.014
            penalty_risk += max(0, float(personality.get("volatility", 50)) - 58) * 0.018
            disruptor_tax += float(entity.get("disruption_risk", 0)) * 0.00012
            risks.append({"player_id": player_id, "player_name": entity.get("player_name"), "risk": "Frustration may become selfish decisions", "severity": round(float(entity.get("disruption_risk", 0)), 1)})
        for ability in entity.get("niche_abilities") or []:
            tier_mult = 0.72 + int(ability.get("tier", 1) or 1) * 0.28
            game = ability.get("game_effects") or {}
            effort += float(game.get("effort", 0)) * tier_mult
            effort = max(effort, float(game.get("effort_floor", -99)) * tier_mult)
            composure += float(game.get("composure", 0)) * tier_mult
            discipline += float(game.get("discipline", 0)) * tier_mult
            passing += float(game.get("passing", 0)) * tier_mult
            defensive_effort += float(game.get("defensive_effort", 0)) * tier_mult
            penalty_risk += float(game.get("penalty_risk", 0)) * tier_mult
        if any(n in niches for n in ("glue_guy", "mentor", "peacemaker", "culture_carrier", "quiet_professional")):
            glue_count += 1
        temporary = _u_active_modifiers(session, player_id)
        readiness = _u_active_readiness(session, player_id)
        readiness_ovr = float(readiness.get("ovr_delta", 0) or 0)
        readiness_stats = dict(readiness.get("stat_modifiers") or {})
        readiness_team_tax += max(0.0, -readiness_ovr) * 0.00018
        overall_equivalent = effort * 0.36 + composure * 0.22 + discipline * 0.14 + passing * 0.14 + defensive_effort * 0.14 + readiness_ovr
        specific = {
            "shooting": readiness_stats.pop("shooting", 0.0),
            "shot_accuracy": readiness_stats.pop("shot_accuracy", 0.0),
            "puck_control": readiness_stats.pop("puck_control", 0.0),
            "offensive_awareness": readiness_stats.pop("offensive_awareness", 0.0),
            "defensive_awareness": readiness_stats.pop("defensive_awareness", 0.0),
            "speed": readiness_stats.pop("speed", 0.0),
            "agility": readiness_stats.pop("agility", 0.0),
            "stamina": readiness_stats.pop("stamina", 0.0),
            "faceoffs": readiness_stats.pop("faceoffs", 0.0),
            "goalie_positioning": readiness_stats.pop("goalie_positioning", readiness_stats.pop("positioning", 0.0)),
            "rebound_control": readiness_stats.pop("rebound_control", 0.0),
            "shot_involvement": readiness_stats.pop("shot_involvement", readiness_ovr * 0.006),
            "assist_involvement": readiness_stats.pop("assist_involvement", readiness_ovr * 0.005),
            "turnover_risk": readiness_stats.pop("turnover_risk", max(0.0, -readiness_ovr) * 0.004),
            "toi_readiness": readiness_stats.pop("toi_readiness", readiness_ovr * 0.004),
        }
        player_modifiers[player_id] = {
            "available": universe_player_is_available(session, player_id),
            "mental_ovr": round(mental, 2),
            "readiness_ovr_delta": round(readiness_ovr, 2),
            "effort": round(effort, 3),
            "composure": round(composure, 3),
            "discipline": round(discipline, 3),
            "passing": round(passing, 3),
            "defensive_effort": round(defensive_effort, 3),
            "penalty_risk": round(max(0.0, penalty_risk + max(0.0, -readiness_ovr) * 0.004), 3),
            "overall_equivalent": round(max(-25.0, min(6.0, overall_equivalent)), 3),
            **specific,
            **readiness_stats,
            **temporary,
        }
        if float(entity.get("overall", 99)) < 80 and float(entity.get("room_value", 0)) >= 70:
            hidden_contributors.append({"player_id": player_id, "player_name": entity.get("player_name"), "overall": entity.get("overall"), "room_value": entity.get("room_value"), "impact": "Raises the emotional and effort floor of teammates"})
    unity = float(culture.get("unity", 50))
    tension = float(culture.get("tension", 35))
    confidence = float(culture.get("confidence", 50))
    leadership = float(culture.get("leadership", 50))
    accountability = float(culture.get("accountability", 50))
    win_delta = ((unity - 50) * 0.00042 + (confidence - 50) * 0.00032 + (leadership - 50) * 0.00024 + (accountability - 50) * 0.00018 - max(0, tension - 35) * 0.00045 + min(0.010, glue_count * 0.0012) - disruptor_tax - min(0.035, readiness_team_tax))
    win_delta = max(-0.10, min(0.085, win_delta))
    context_id = str((game_meta or {}).get("game_id") or f"gamectx_{team_id}_{_u_current_meta(session)[0]}")
    context = {
        "id": context_id,
        "team_id": str(team_id),
        "opponent_id": str(opponent_id or ""),
        "calendar_day": _u_current_meta(session)[0],
        "win_probability_delta": round(win_delta, 4),
        "team_modifiers": {"chemistry": round((unity - 50) / 25, 3), "composure": round((confidence + leadership - 100) / 50, 3), "discipline": round((accountability - tension) / 50, 3), "effort_floor": round((accountability + unity - 100) / 55, 3)},
        "player_modifiers": player_modifiers,
        "hidden_contributors": hidden_contributors,
        "character_risks": risks,
        "locker_room_snapshot": dict(culture),
        "explanation": f"Locker-room, mental and readiness state changes win probability by {win_delta * 100:+.1f} percentage points.",
    }
    contexts = dict(getattr(session, "universe_game_contexts", None) or {})
    contexts[context_id] = context
    session.universe_game_contexts = dict(list(contexts.items())[-60:])
    return context


# Extend the existing V2 payload instead of replacing its UI contract.
_UNIVERSE_V2_PAYLOAD_BEFORE_V3 = build_narrative_universe_v2_payload

def build_narrative_universe_v2_payload(session: Any) -> Dict[str, Any]:
    base = dict(_UNIVERSE_V2_PAYLOAD_BEFORE_V3(session) or {})
    state = dict(getattr(session, "universe_major_event_state", None) or {})
    base.update(
        {
            "universe_engine_version": UNIVERSE_ENGINE_VERSION,
            "trade_demands": list(getattr(session, "universe_trade_demands", None) or [])[-30:],
            "team_sanctions": list(getattr(session, "universe_team_sanctions", None) or [])[-30:],
            "cap_penalties": dict(getattr(session, "universe_cap_penalties", None) or {}),
            "forfeited_picks": list(getattr(session, "universe_forfeited_picks", None) or [])[-30:],
            "player_availability": dict(getattr(session, "universe_player_availability", None) or {}),
            "agent_org_relationships": dict(getattr(session, "universe_agent_org_relationships", None) or {}),
            "major_event_budget": state,
        }
    )
    user_tid = str(getattr(session, "user_team_id", "") or "")
    dossiers: Dict[str, Any] = {}
    for row in list(base.get("players") or []):
        pid = str(row.get("player_id") or "")
        if not pid or str(row.get("team_id") or "") != user_tid:
            continue
        entity = (getattr(session, "universe_players", None) or {}).get(pid) or row
        player_obj = _player_from_roster(session, pid)
        dossiers[pid] = build_human_dossier_payload(session, entity, player_obj, include_private=True)
    if dossiers:
        base["human_dossiers"] = dossiers
        if not base.get("player_dossiers"):
            base["player_dossiers"] = list(dossiers.values())
    try:
        base["player_meetings"] = build_player_meetings_payload(session)
    except Exception as exc:
        _log.warning("player_meetings payload failed: %s", exc)
        base["player_meetings"] = {}
    return base


# ========================= PLAYER MEETINGS / GM RELATIONSHIP SYSTEM =========================

GM_MEETING_HISTORY_MAX = 120
GM_MEETING_PRAISE_COOLDOWN_DAYS = 12
GM_MEETING_DEFAULT_COOLDOWN_DAYS = 6
GM_MEETING_NTC_COOLDOWN_DAYS = 21


def _ensure_gm_meeting_state(session: Any) -> None:
    if not hasattr(session, "gm_meeting_history") or session.gm_meeting_history is None:
        session.gm_meeting_history = []
    if not hasattr(session, "gm_active_meetings") or session.gm_active_meetings is None:
        session.gm_active_meetings = {}


def _gm_user_org_players(session: Any) -> List[Tuple[str, Any, str]]:
    """(player_id, player_obj, roster_bucket) for user NHL + AHL."""
    utid = str(getattr(session, "user_team_id", "") or "")
    by_id = getattr(session, "team_by_id", None) or {}
    team = by_id.get(utid)
    if team is None:
        raw = getattr(session, "user_team_id", None)
        team = by_id.get(raw)
    if team is None and utid.isdigit():
        team = by_id.get(int(utid))
    if team is None:
        return []
    rows: List[Tuple[str, Any, str]] = []
    seen: set = set()
    for bucket in ("roster", "ahl_roster", "injured_reserve", "scratches"):
        for player in getattr(team, bucket, None) or []:
            pid = str(getattr(player, "id", "") or "")
            if not pid or pid in seen or getattr(player, "retired", False):
                continue
            seen.add(pid)
            rows.append((pid, player, bucket or "roster"))
    return rows


def _gm_entity(session: Any, player_id: str) -> Dict[str, Any]:
    migrate_session_storyline_state(session)
    _u_sync_player_entities(session)
    entities = getattr(session, "universe_players", None) or {}
    entity = entities.get(str(player_id))
    if entity is None:
        player = _player_from_roster(session, str(player_id))
        if player is None:
            return {}
        utid = str(getattr(session, "user_team_id", "") or "")
        entity = _u_create_player_entity(session, utid, player)
        entities[str(player_id)] = entity
        session.universe_players = entities
    return entity


def _gm_contract_info(player: Any) -> Dict[str, Any]:
    c = getattr(player, "contract", None) or {}
    if not isinstance(c, dict):
        c = {}
    clause = str(c.get("clause") or c.get("clause_type") or "").upper()
    has_ntc = bool(c.get("no_trade_clause") or c.get("ntc") or "NTC" in clause or "NMC" in clause or "NO TRADE" in clause or "NO MOVE" in clause)
    has_nmc = bool(c.get("no_move_clause") or c.get("nmc") or "NMC" in clause or "NO MOVE" in clause)
    years_left = float(c.get("years_remaining") or c.get("term_years_remaining") or 0)
    expiring = years_left <= 1.05
    return {
        "years_remaining": years_left,
        "aav_m": float(c.get("aav_m") or c.get("cap_hit_m") or 0),
        "expiring": expiring,
        "has_ntc": has_ntc,
        "has_nmc": has_nmc,
        "has_trade_protection": has_ntc or has_nmc,
        "clause_label": "NMC" if has_nmc else ("NTC" if has_ntc else "None"),
    }


def _gm_on_cooldown(entity: Dict[str, Any], interaction_type: str, day: int) -> bool:
    cooldowns = dict(entity.get("meeting_cooldowns") or {})
    last = int(cooldowns.get(interaction_type) or -999)
    extra = GM_MEETING_PRAISE_COOLDOWN_DAYS if interaction_type == "praise_performance" else GM_MEETING_NTC_COOLDOWN_DAYS if interaction_type == "ask_ntc_waiver" else GM_MEETING_DEFAULT_COOLDOWN_DAYS
    return day - last < extra


def _gm_set_cooldown(entity: Dict[str, Any], interaction_type: str, day: int) -> None:
    cooldowns = dict(entity.get("meeting_cooldowns") or {})
    cooldowns[str(interaction_type)] = int(day)
    entity["meeting_cooldowns"] = cooldowns


def _gm_broken_promises(session: Any, player_id: str) -> int:
    return sum(
        1
        for p in (getattr(session, "universe_promises", None) or [])
        if str(p.get("player_id") or "") == str(player_id) and str(p.get("status") or "") == "broken"
    )


def _gm_relationship_summary(entity: Dict[str, Any], session: Any, player_id: str) -> Dict[str, Any]:
    state = dict(entity.get("state") or {})
    trusts = dict(entity.get("trusts") or {})
    gm_trust = float(state.get("gm_trust") or trusts.get("gm") or 55)
    morale = float(state.get("morale") or 55)
    role_sat = float(state.get("role_satisfaction") or 55)
    broken = _gm_broken_promises(session, player_id)
    active_promises = [
        p for p in (getattr(session, "universe_promises", None) or [])
        if str(p.get("player_id") or "") == str(player_id) and str(p.get("status") or "") == "active"
    ]
    score = gm_trust * 0.45 + morale * 0.20 + role_sat * 0.20 + max(0, 100 - broken * 14) * 0.15
    if score >= 78:
        label, tone = "Strong", "positive"
    elif score >= 64:
        label, tone = "Good", "neutral"
    elif score >= 48:
        label, tone = "Neutral", "neutral"
    elif score >= 32:
        label, tone = "Strained", "negative"
    else:
        label, tone = "Broken", "negative"
    notes: List[str] = []
    if role_sat < 45:
        notes.append("wants a larger role")
    if broken:
        notes.append(f"still frustrated by {broken} broken promise{'s' if broken != 1 else ''}")
    if gm_trust >= 72:
        notes.append("trusts your communication")
    elif gm_trust < 40:
        notes.append("skeptical of management")
    if active_promises:
        notes.append(f"{len(active_promises)} open promise{'s' if len(active_promises) != 1 else ''}")
    detail = " · ".join(notes[:2]) if notes else "No major friction on file."
    return {
        "label": label,
        "tone": tone,
        "detail": detail,
        "gm_trust": round(gm_trust, 1),
        "morale": round(morale, 1),
        "role_satisfaction": round(role_sat, 1),
        "broken_promises": broken,
        "active_promise_count": len(active_promises),
    }


def build_ovr_trend_explanation(session: Any, player_id: str) -> Dict[str, Any]:
    """Build hockey-language OVR trend explanation from real readiness/performance state."""
    migrate_session_storyline_state(session)
    entity = _gm_entity(session, player_id)
    player = _player_from_roster(session, player_id)
    if not entity:
        return {"direction": "flat", "factors": [], "summary": "Insufficient data for a trend read."}
    base_ovr = float(entity.get("overall") or _player_ovr99(player or object()))
    readiness = _u_active_readiness(session, player_id)
    readiness_ovr = float(readiness.get("ovr_delta") or 0)
    state = dict(entity.get("state") or {})
    factors: List[Dict[str, str]] = []
    direction = "flat"
    if readiness_ovr >= 1.2:
        direction = "up"
    elif readiness_ovr <= -1.2:
        direction = "down"
    conf = float(state.get("confidence") or 55)
    if conf >= 68:
        factors.append({"kind": "confidence", "text": "Confidence is running high after recent results."})
    elif conf <= 42:
        factors.append({"kind": "confidence", "text": "Confidence has declined and is affecting current readiness."})
    role_sat = float(state.get("role_satisfaction") or 55)
    if role_sat <= 42:
        factors.append({"kind": "role", "text": "Current deployment gives fewer meaningful touches than the player expects."})
    elif role_sat >= 68 and direction == "up":
        factors.append({"kind": "role", "text": "Increased usage is helping the player play to his strengths."})
    for row in (readiness.get("modifiers") or []):
        reason = str(row.get("reason") or row.get("source") or "Recent form")
        delta = float(row.get("ovr_delta") or 0)
        if abs(delta) >= 0.4:
            factors.append({"kind": "readiness", "text": f"{reason} ({delta:+.1f} readiness)." if delta else reason})
    pst = _ensure_player_storyline_state(player) if player is not None else {}
    if int(pst.get("last_trade_readiness_penalty") or 0) < 0:
        factors.append({"kind": "trade", "text": "Trade talk or uncertainty has weighed on readiness."})
    age = float(entity.get("age") or _player_age(player or object()) or 28)
    if age >= 33 and direction == "down":
        factors.append({"kind": "age", "text": "Age curve is a factor — recovery and peak burst take longer."})
    perm = dict(entity.get("season_permanent_attribute_delta") or {})
    dev_gain = sum(float(v) for v in perm.values() if float(v) > 0)
    if dev_gain >= 0.5:
        factors.append({"kind": "development", "text": "Underlying skills have improved this season — permanent development, not just a hot stretch."})
    if not factors:
        factors.append({"kind": "baseline", "text": "Core attributes are stable; current rating reflects ordinary readiness variance."})
    headline = "WHY YOUR GAME IS TRENDING UP" if direction == "up" else "WHY YOUR GAME IS TRENDING DOWN" if direction == "down" else "CURRENT FORM SNAPSHOT"
    return {
        "direction": direction,
        "base_ovr": round(base_ovr, 1),
        "readiness_delta": round(readiness_ovr, 2),
        "current_ovr": round(_u_clip(base_ovr + readiness_ovr, 1, 99), 1),
        "headline": headline,
        "factors": factors[:6],
        "permanent_note": "Permanent attribute changes are separate from temporary readiness swings." if dev_gain >= 0.5 else "",
    }


def _gm_build_context(session: Any, player_id: str) -> Dict[str, Any]:
    entity = _gm_entity(session, player_id)
    player = _player_from_roster(session, player_id)
    day, iso, season = _u_current_meta(session)
    state = dict(entity.get("state") or {})
    personality = dict(entity.get("personality") or {})
    life = dict(entity.get("life") or {})
    contract = _gm_contract_info(player) if player is not None else {}
    ovr_trend = build_ovr_trend_explanation(session, player_id)
    pst = _ensure_player_storyline_state(player) if player is not None else {}
    pos = str(entity.get("position") or _u_position(player or object())).upper()
    is_goalie = pos == "G"
    try:
        from app.sim_engine.franchise.player_agent_engine import ensure_player_agent, agent_public_view

        agent = agent_public_view(player, session) if player is not None else {}
    except Exception:
        agent = {}
    return {
        "session": session,
        "player_id": str(player_id),
        "entity": entity,
        "player": player,
        "day": day,
        "iso": iso,
        "state": state,
        "personality": personality,
        "life": life,
        "contract": contract,
        "ovr_trend": ovr_trend,
        "pst": pst,
        "position": pos,
        "is_goalie": is_goalie,
        "agent": agent,
        "relationship": _gm_relationship_summary(entity, session, player_id),
        "trade_rumor_heat": int(pst.get("trade_rumor_heat") or 0),
        "trade_attempts": int(pst.get("trade_attempt_count") or 0),
        "scoring_drought": float(state.get("confidence", 55)) <= 44 and not is_goalie,
        "struggling": ovr_trend.get("direction") == "down" or float(state.get("morale", 55)) <= 42,
        "improving": ovr_trend.get("direction") == "up",
        "injured": bool(getattr(player, "injured", False) or getattr(player, "injury", None)),
        "age": float(entity.get("age") or 26),
    }


GM_MEETING_CATEGORIES = (
    ("role", "Role & Usage"),
    ("performance", "Performance"),
    ("development", "Development & OVR"),
    ("contract", "Contract & Future"),
    ("trade", "Trade & Movement"),
    ("team", "Team & Relationships"),
    ("personal", "Personal"),
)


def _gm_choice(cid: str, label: str, detail: str, outcome: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": cid, "label": label, "detail": detail, "outcome": outcome}


def _gm_apply_meeting_outcome(session: Any, ctx: Dict[str, Any], interaction_type: str, choice: Dict[str, Any]) -> Dict[str, Any]:
    entity = ctx["entity"]
    player_id = ctx["player_id"]
    day, iso = ctx["day"], ctx["iso"]
    outcome = dict(choice.get("outcome") or {})
    fake_interaction = {
        "id": f"gm_{uuid.uuid4().hex[:10]}",
        "kind": interaction_type,
        "team_id": str(getattr(session, "user_team_id", "") or ""),
        "actor_id": player_id,
        "player_id": player_id,
        "player_name": str(entity.get("player_name") or "Player"),
        "summary": str(choice.get("label") or interaction_type),
    }
    receipts = _u_apply_outcome(session, fake_interaction, choice)
    _gm_set_cooldown(entity, interaction_type, day)
    rel = entity.setdefault("gm_relationship", {})
    for key, delta in (outcome.get("relationship") or {}).items():
        rel[key] = _u_clip(float(rel.get(key, 55)) + float(delta))
    history_row = {
        "id": fake_interaction["id"],
        "player_id": player_id,
        "player_name": entity.get("player_name"),
        "interaction_type": interaction_type,
        "initiator": "gm",
        "choice_id": choice.get("id"),
        "choice_label": choice.get("label"),
        "calendar_day": day,
        "calendar_iso": iso,
        "relationship_snapshot": _gm_relationship_summary(entity, session, player_id),
        "receipts": receipts,
    }
    hist = list(getattr(session, "gm_meeting_history", None) or [])
    hist.append(history_row)
    session.gm_meeting_history = hist[-GM_MEETING_HISTORY_MAX:]
    _u_add_memory(
        entity,
        kind="gm_meeting",
        summary=f"GM meeting — {interaction_type.replace('_', ' ')}: {choice.get('label')}",
        day=day,
        iso=iso,
        emotional_delta=float(sum((outcome.get("relationship") or {}).values()) or 0),
        public=bool(outcome.get("public")),
    )
    return {"ok": True, "history": history_row, "receipts": receipts, "relationship": _gm_relationship_summary(entity, session, player_id)}


def _gm_build_interaction(session: Any, ctx: Dict[str, Any], interaction_type: str) -> Optional[Dict[str, Any]]:
    """Build one GM-initiated meeting template with player response + GM choices."""
    entity = ctx["entity"]
    name = str(entity.get("player_name") or "Player")
    p = ctx["personality"]
    rel = ctx["relationship"]
    volatile = float(p.get("volatility", 50)) >= 68
    ambitious = float(p.get("ambition", 50)) >= 70
    loyal = float(p.get("loyalty", 50)) >= 72
    templates: Dict[str, Dict[str, Any]] = {}

    def _reg(iid: str, cat: str, title: str, player_line: str, choices: List[Dict[str, Any]], available: bool = True):
        templates[iid] = {"id": iid, "category": cat, "title": title, "player_line": player_line, "choices": choices, "available": available}

    # Role / usage (1-12)
    _reg("discuss_ice_time", "role", "Discuss current ice time", f"I want to understand where I stand. Am I getting a fair look?", [
        _gm_choice("honest", "Give an honest assessment", "Explain deployment honestly.", {"profile_changes": {"actor": {"state.gm_trust": 4, "state.focus": 2}}, "relationship": {"trust": 3}}),
        _gm_choice("promise", "Promise to revisit usage", "Creates a role opportunity promise.", {"profile_changes": {"actor": {"state.morale": 4, "state.gm_trust": 2}}, "promise": {"type": "role_opportunity", "player_id": ctx["player_id"], "due_games": 6, "description": "Meaningful lineup opportunity within six games.", "success_readiness": 0.5, "failure_readiness": -2.0}, "relationship": {"trust": 2}}),
        _gm_choice("firm", "Hold the current structure", "Keep hierarchy intact.", {"profile_changes": {"actor": {"state.gm_trust": -3, "state.morale": -2}}, "relationship": {"trust": -4}}),
    ], not ctx["is_goalie"])
    _reg("offer_increased_ice_time", "role", "Offer increased ice time", "If you're serious about giving me more, I need to know what that looks like.", [
        _gm_choice("commit", "Commit to increased usage", "Promise meaningful minutes.", {"profile_changes": {"actor": {"state.morale": 5, "state.confidence": 3}}, "promise": {"type": "role_opportunity", "player_id": ctx["player_id"], "due_games": 5, "description": "Increased even-strength usage.", "success_readiness": 0.6, "failure_readiness": -2.5}, "relationship": {"trust": 4}}),
        _gm_choice("conditional", "Make it conditional on performance", "Earn-it message.", {"profile_changes": {"actor": {"state.focus": 3, "state.gm_trust": 1}}, "relationship": {"respect": 2}}),
    ], float(ctx["state"].get("role_satisfaction") or 55) <= 58)
    _reg("explain_reduced_ice_time", "role", "Explain reduced ice time", "My minutes dropped. I'd like the truth.", [
        _gm_choice("transparent", "Explain coaching decision", "Transparency builds trust.", {"profile_changes": {"actor": {"state.gm_trust": 3}}, "relationship": {"trust": 3, "honesty": 2}}),
        _gm_choice("performance", "Tie it to recent performance", "Accountability framing.", {"profile_changes": {"actor": {"state.focus": 2, "state.morale": -1}}, "relationship": {"respect": 1}}),
        _gm_choice("deflect", "Defer to coaching staff", "Avoids direct answer.", {"profile_changes": {"actor": {"state.gm_trust": -4}}, "relationship": {"trust": -5}}),
    ], float(ctx["state"].get("role_satisfaction") or 55) <= 52)
    _reg("discuss_line_assignment", "role", "Discuss line assignment", "Where do you see me fitting in the lineup?", [
        _gm_choice("top_six", "Discuss top-six path", "Offensive role clarity.", {"profile_changes": {"actor": {"state.confidence": 2, "state.focus": 2}}, "relationship": {"communication": 3}}),
        _gm_choice("depth", "Explain depth role", "Honest depth conversation.", {"profile_changes": {"actor": {"state.morale": -1, "state.gm_trust": 2}}, "relationship": {"honesty": 2}}),
    ])
    _reg("discuss_promotion", "role", "Discuss promotion to higher line", "I think I've outgrown my current line.", [
        _gm_choice("support", "Agree — promotion coming", "Boost confidence.", {"profile_changes": {"actor": {"state.morale": 4, "state.confidence": 3}}, "promise": {"type": "role_opportunity", "player_id": ctx["player_id"], "due_games": 4, "description": "Promotion to a higher line.", "success_readiness": 0.45}, "relationship": {"trust": 3}}),
        _gm_choice("wait", "Not yet — keep proving it", "Patience required.", {"profile_changes": {"actor": {"state.morale": -2 if ambitious else 0}}, "relationship": {"respect": 1 if not ambitious else -2}}),
    ], ctx["improving"] or ambitious)
    _reg("discuss_demotion", "role", "Discuss demotion", "Moving down a line stings. Talk to me.", [
        _gm_choice("accountability", "Accountability conversation", "Performance-linked.", {"profile_changes": {"actor": {"state.focus": 3, "state.morale": -2}}, "relationship": {"respect": 2}}),
        _gm_choice("support", "Support through adjustment", "Soft landing.", {"profile_changes": {"actor": {"state.gm_trust": 2, "state.belonging": 2}}, "relationship": {"loyalty": 2}}),
    ], ctx["struggling"] or float(ctx["state"].get("role_satisfaction") or 55) <= 48)
    _reg("offer_pp_opportunity", "role", "Offer power-play opportunity", "Put me on the power play and I'll produce.", [
        _gm_choice("promise_pp", "Promise PP look", "Special teams promise.", {"profile_changes": {"actor": {"state.morale": 4}}, "promise": {"type": "power_play_opportunity", "player_id": ctx["player_id"], "due_games": 8, "description": "Meaningful PP usage.", "success_readiness": 0.5, "failure_readiness": -2.0}, "relationship": {"trust": 3}}),
        _gm_choice("deny", "PP spots are earned", "Hierarchy held.", {"profile_changes": {"actor": {"state.morale": -3}}, "relationship": {"trust": -2}}),
    ], not ctx["is_goalie"])
    _reg("explain_pp_removal", "role", "Explain removal from power play", "I lost my PP spot. Why?", [
        _gm_choice("explain", "Explain decision", "Direct answer.", {"profile_changes": {"actor": {"state.gm_trust": 2}}, "relationship": {"honesty": 2}}),
        _gm_choice("challenge", "Challenge him to earn it back", "Competitive framing.", {"profile_changes": {"actor": {"state.focus": 3, "state.confidence": -1}}, "relationship": {"respect": 1}}),
    ], not ctx["is_goalie"])
    _reg("offer_pk_opportunity", "role", "Offer penalty-kill opportunity", "I can help on the kill if you trust me.", [
        _gm_choice("yes", "Offer PK reps", "Defensive trust.", {"profile_changes": {"actor": {"state.belonging": 3, "state.confidence": 2}}, "relationship": {"trust": 2}}),
        _gm_choice("no", "Stay at even strength", "Role unchanged.", {"profile_changes": {"actor": {"state.morale": -1}}, "relationship": {}}),
    ], not ctx["is_goalie"])
    _reg("discuss_healthy_scratch", "role", "Discuss healthy scratch", "Being scratched hurts. What's the message?", [
        _gm_choice("development", "Development / reset", "Temporary bench.", {"profile_changes": {"actor": {"state.focus": 2, "state.morale": -3}}, "relationship": {"communication": 2}}),
        _gm_choice("performance", "Performance decision", "Hard truth.", {"profile_changes": {"actor": {"state.morale": -4, "state.focus": 3}}, "relationship": {"respect": 1}}),
    ], float(ctx["state"].get("morale") or 55) <= 45)
    _reg("discuss_return_lineup", "role", "Discuss returning to lineup", "I'm ready to come back. What's the plan?", [
        _gm_choice("welcome", "Welcome back with defined role", "Clear return path.", {"profile_changes": {"actor": {"state.morale": 4, "state.confidence": 3}}, "relationship": {"trust": 3}}),
        _gm_choice("earn", "Earn the spot back", "Competition message.", {"profile_changes": {"actor": {"state.focus": 3}}, "relationship": {"respect": 2}}),
    ], ctx["injured"])
    _reg("discuss_goalie_workload", "role", "Discuss goalie starting workload", "I want clarity on the crease rotation.", [
        _gm_choice("promise_starts", "Promise a run of starts", "Starter opportunity.", {"profile_changes": {"actor": {"state.confidence": 4}}, "promise": {"type": "goalie_start_opportunity", "player_id": ctx["player_id"], "due_games": 6, "description": "Meaningful starting workload.", "success_readiness": 0.4, "failure_readiness": -2.5}, "relationship": {"trust": 3}}),
        _gm_choice("platoon", "Explain platoon plan", "Shared net.", {"profile_changes": {"actor": {"state.gm_trust": 2}}, "relationship": {"communication": 2}}),
    ], ctx["is_goalie"])

    # Performance (13-20)
    _reg("praise_performance", "performance", "Praise recent performance", "Appreciate you noticing — means something.", [
        _gm_choice("specific", "Be specific about what worked", "Targeted praise.", {"profile_changes": {"actor": {"state.confidence": 3, "state.morale": 2}}, "relationship": {"respect": 2}}),
        _gm_choice("team", "Praise team-first play", "Culture reinforcement.", {"profile_changes": {"actor": {"state.belonging": 3}}, "relationship": {"loyalty": 2}}),
    ], ctx["improving"] and not _gm_on_cooldown(entity, "praise_performance", ctx["day"]))
    _reg("challenge_performance", "performance", "Challenge poor performance", "I know I haven't been good enough.", [
        _gm_choice("direct", "Direct accountability", "High standards.", {"profile_changes": {"actor": {"state.focus": 4, "state.morale": -2}}, "relationship": {"respect": 2}}),
        _gm_choice("support", "Challenge with support", "Tough love.", {"profile_changes": {"actor": {"state.focus": 3, "state.gm_trust": 1}}, "relationship": {"trust": 2}}),
    ], ctx["struggling"])
    _reg("ask_struggles", "performance", "Ask what is causing struggles", "Honestly? A few things are off.", [
        _gm_choice("listen", "Listen and validate", "Open door.", {"profile_changes": {"actor": {"state.gm_trust": 4, "state.personal_stress": -3}}, "relationship": {"communication": 4}}),
        _gm_choice("fix_role", "Offer role adjustment", "Structural fix.", {"profile_changes": {"actor": {"state.morale": 2}}, "promise": {"type": "role_opportunity", "player_id": ctx["player_id"], "due_games": 7, "description": "Adjusted role to help production.", "success_readiness": 0.4}, "relationship": {"trust": 3}}),
    ], ctx["struggling"])
    _reg("discuss_scoring_drought", "performance", "Discuss scoring drought", "The points aren't coming. I'm aware.", [
        _gm_choice("stay_confident", "Keep shooting — trust process", "Confidence support.", {"profile_changes": {"actor": {"state.confidence": 2}}, "relationship": {"trust": 2}}),
        _gm_choice("usage_change", "Discuss usage to break drought", "Tactical help.", {"profile_changes": {"actor": {"state.focus": 2, "state.morale": 1}}, "relationship": {"communication": 3}}),
    ], ctx["scoring_drought"] and not ctx["is_goalie"])
    _reg("discuss_defensive_struggles", "performance", "Discuss defensive struggles", "My defensive game hasn't been sharp.", [
        _gm_choice("film", "Commit to video work", "Development focus.", {"profile_changes": {"actor": {"state.focus": 3}}, "relationship": {"respect": 2}}),
        _gm_choice("pairing", "Discuss pairing support", "Structural help.", {"profile_changes": {"actor": {"state.belonging": 2}}, "relationship": {"trust": 2}}),
    ], ctx["struggling"] and not ctx["is_goalie"])
    _reg("discuss_improvement", "performance", "Discuss recent improvement", "Feels like things are clicking again.", [
        _gm_choice("encourage", "Encourage the trend", "Positive reinforcement.", {"profile_changes": {"actor": {"state.confidence": 3, "state.morale": 2}}, "relationship": {"trust": 2}}),
        _gm_choice("raise_bar", "Raise the standard", "Push higher.", {"profile_changes": {"actor": {"state.focus": 3, "state.ambition": 1}}, "relationship": {"respect": 3}}),
    ], ctx["improving"])
    _reg("discuss_consistency", "performance", "Discuss consistency", "I know I need to be more consistent.", [
        _gm_choice("routine", "Discuss routine / habits", "Process focus.", {"profile_changes": {"actor": {"state.focus": 3}}, "relationship": {"communication": 2}}),
        _gm_choice("pressure", "Acknowledge pressure", "Empathy.", {"profile_changes": {"actor": {"state.personal_stress": -2, "state.gm_trust": 2}}, "relationship": {"trust": 2}}),
    ])
    _reg("discuss_playoff_performance", "performance", "Discuss playoff performance", "Playoffs are a different animal.", [
        _gm_choice("belief", "Express belief in playoff game", "Big moment trust.", {"profile_changes": {"actor": {"state.confidence": 4, "state.morale": 2}}, "relationship": {"trust": 3}}),
        _gm_choice("prep", "Discuss playoff prep plan", "Preparation focus.", {"profile_changes": {"actor": {"state.focus": 3}}, "relationship": {"respect": 2}}),
    ], str(getattr(session, "phase", "") or "") in ("playoffs", "playoff_ready"))

    # Development / OVR (21-28)
    ovr_factors = ctx["ovr_trend"].get("factors") or []
    ovr_line = ovr_factors[0]["text"] if ovr_factors else "My game feels about the same."
    _reg("explain_ovr_rising", "development", "Explain why overall is rising", f"I've felt the difference. {ovr_line}", [
        _gm_choice("credit", "Credit his work", "Ownership.", {"profile_changes": {"actor": {"state.confidence": 3, "state.morale": 2}}, "relationship": {"respect": 3}}),
        _gm_choice("demand_more", "Demand he keep pushing", "Higher bar.", {"profile_changes": {"actor": {"state.focus": 3}}, "relationship": {"respect": 2}}),
    ], ctx["ovr_trend"].get("direction") == "up")
    _reg("explain_ovr_falling", "development", "Explain why overall is falling", f"I know something's off. {ovr_line}", [
        _gm_choice("support", "Support through slump", "Safety.", {"profile_changes": {"actor": {"state.gm_trust": 3, "state.morale": 2}}, "relationship": {"trust": 3}}),
        _gm_choice("accountability", "Accountability for readiness", "Standards.", {"profile_changes": {"actor": {"state.focus": 3, "state.confidence": -1}}, "relationship": {"respect": 2}}),
    ], ctx["ovr_trend"].get("direction") == "down")
    _reg("discuss_development_plan", "development", "Discuss development plan", "Tell me what the organization needs me to become.", [
        _gm_choice("clear_plan", "Set a clear plan", "Development clarity.", {"profile_changes": {"actor": {"state.focus": 4, "state.gm_trust": 3}}, "potential_changes": [{"who": "actor", "delta": 0.2, "reason": "Development plan alignment"}], "relationship": {"communication": 4}}),
        _gm_choice("skills", "Assign focused skills work", "Targeted growth.", {"profile_changes": {"actor": {"state.confidence": 2}}, "relationship": {"respect": 2}}),
    ], float(ctx["age"]) <= 27)
    _reg("set_development_priority", "development", "Set development priority", "Which part of my game should I prioritize?", [
        _gm_choice("skating", "Prioritize skating", "Skating focus.", {"profile_changes": {"actor": {"state.focus": 3}}, "relationship": {"communication": 2}}),
        _gm_choice("hockey_sense", "Prioritize hockey sense", "IQ focus.", {"profile_changes": {"actor": {"state.focus": 3}}, "relationship": {"communication": 2}}),
    ], float(ctx["age"]) <= 26)
    _reg("discuss_stalled_development", "development", "Discuss stalled development", "I feel like I've plateaued.", [
        _gm_choice("patience", "Explain patience", "Timeline.", {"profile_changes": {"actor": {"state.gm_trust": 1}}, "relationship": {"honesty": 2}}),
        _gm_choice("opportunity", "Promise opportunity to break through", "Chance to grow.", {"profile_changes": {"actor": {"state.morale": 3}}, "promise": {"type": "role_opportunity", "player_id": ctx["player_id"], "due_games": 10, "description": "Development opportunity in lineup.", "success_potential_delta": 0.3}, "relationship": {"trust": 3}}),
    ], float(ctx["age"]) <= 28 and not ctx["improving"])
    _reg("discuss_aging_decline", "development", "Discuss aging / decline", "I know the clock is ticking.", [
        _gm_choice("veteran_role", "Discuss veteran leadership role", "New value path.", {"profile_changes": {"actor": {"state.belonging": 3, "state.morale": 2}}, "relationship": {"loyalty": 3}}),
        _gm_choice("honest", "Honest conversation about minutes", "Transparent.", {"profile_changes": {"actor": {"state.gm_trust": 3}}, "relationship": {"honesty": 3}}),
    ], float(ctx["age"]) >= 32)
    _reg("discuss_conditioning", "development", "Discuss conditioning / readiness", "I'll do whatever it takes to be ready.", [
        _gm_choice("staff", "Connect with training staff", "Support.", {"profile_changes": {"actor": {"state.energy": 4, "state.focus": 2}}, "relationship": {"trust": 2}}),
        _gm_choice("expect", "Set clear expectations", "Standards.", {"profile_changes": {"actor": {"state.focus": 3}}, "relationship": {"respect": 2}}),
    ])
    _reg("discuss_injury_recovery", "development", "Discuss recovery after injury", "I'm doing everything to get back right.", [
        _gm_choice("no_rush", "No rush — health first", "Trust.", {"profile_changes": {"actor": {"state.gm_trust": 4, "state.personal_stress": -3}}, "relationship": {"loyalty": 3}}),
        _gm_choice("timeline", "Discuss return timeline", "Planning.", {"profile_changes": {"actor": {"state.focus": 2}}, "relationship": {"communication": 2}}),
    ], ctx["injured"])

    # Contract (29-36)
    _reg("ask_extension_interest", "contract", "Ask about extension interest", "I'm open to talking about staying — depending on the terms.", [
        _gm_choice("interested", "Express mutual interest", "Goodwill for negotiations.", {"profile_changes": {"actor": {"state.morale": 3, "state.gm_trust": 3}}, "relationship": {"negotiation_goodwill": 5, "loyalty": 3}}),
        _gm_choice("noncommittal", "Stay noncommittal", "Flexibility.", {"profile_changes": {"actor": {"state.gm_trust": -2}}, "relationship": {"negotiation_goodwill": -2}}),
    ], ctx["contract"].get("expiring") or ctx["contract"].get("years_remaining", 0) <= 2)
    _reg("long_term_plans", "contract", "Tell player they are part of long-term plans", "That's what I needed to hear.", [
        _gm_choice("commit", "Commit to long-term vision", "Security.", {"profile_changes": {"actor": {"state.morale": 4, "state.belonging": 3}}, "relationship": {"loyalty": 4, "negotiation_goodwill": 4}}),
        _gm_choice("cautious", "Cautious optimism", "Measured.", {"profile_changes": {"actor": {"state.gm_trust": 1}}, "relationship": {"honesty": 1}}),
    ])
    _reg("future_uncertain", "contract", "Tell player future is uncertain", "That's hard to hear.", [
        _gm_choice("honest", "Be direct about cap/depth chart", "Transparency.", {"profile_changes": {"actor": {"state.gm_trust": 2, "state.morale": -3}}, "relationship": {"honesty": 3, "negotiation_goodwill": -2}}),
        _gm_choice("soften", "Soften with respect", "Dignity.", {"profile_changes": {"actor": {"state.morale": -1}}, "relationship": {"respect": 2}}),
    ], ctx["contract"].get("expiring"))
    _reg("discuss_role_next_season", "contract", "Discuss expected role next season", "What role do you see for me next year?", [
        _gm_choice("top_role", "Top-six / top-pair role", "Ambitious plan.", {"profile_changes": {"actor": {"state.confidence": 3, "state.morale": 2}}, "relationship": {"communication": 3}}),
        _gm_choice("depth", "Depth / specialist role", "Honest depth.", {"profile_changes": {"actor": {"state.morale": -2 if ambitious else 1}}, "relationship": {"honesty": 2}}),
    ])
    _reg("discuss_upcoming_fa", "contract", "Discuss upcoming free agency", "Free agency is on my mind.", [
        _gm_choice("priority", "Say he's a priority to re-sign", "Retention signal.", {"profile_changes": {"actor": {"state.morale": 3}}, "relationship": {"negotiation_goodwill": 5, "loyalty": 3}}),
        _gm_choice("open", "Keep options open", "Neutral.", {"profile_changes": {"actor": {"state.gm_trust": -1}}, "relationship": {"negotiation_goodwill": -1}}),
    ], ctx["contract"].get("expiring"))
    _reg("hometown_discount", "contract", "Ask for hometown discount", "I've given a lot to this organization.", [
        _gm_choice("ask", "Ask for team-friendly term", "Money ask.", {"profile_changes": {"actor": {"state.morale": -2 if not loyal else 1}}, "relationship": {"negotiation_goodwill": 3 if loyal else -4}}),
        _gm_choice("respect", "Respect his market value", "No discount push.", {"profile_changes": {"actor": {"state.gm_trust": 2}}, "relationship": {"respect": 3}}),
    ], loyal or float(ctx["life"].get("community_connection") or 0) >= 55)
    _reg("contract_expectations", "contract", "Discuss contract expectations", "Let's talk about what fair looks like.", [
        _gm_choice("listen", "Listen to expectations", "Open negotiation tone.", {"profile_changes": {"actor": {"state.gm_trust": 2}}, "relationship": {"negotiation_goodwill": 3, "communication": 3}}),
        _gm_choice("anchor", "Anchor below market", "Hard line.", {"profile_changes": {"actor": {"state.morale": -3}}, "relationship": {"negotiation_goodwill": -5, "grievance": 4}}),
    ], ctx["contract"].get("years_remaining", 0) <= 2)
    _reg("discuss_retirement", "contract", "Discuss retirement / future plans", "I'm thinking about how much longer I want to do this.", [
        _gm_choice("support", "Support whatever he decides", "Dignity.", {"profile_changes": {"actor": {"state.gm_trust": 4, "state.belonging": 3}}, "relationship": {"loyalty": 4}}),
        _gm_choice("win_now", "Ask for one more push", "Competitive ask.", {"profile_changes": {"actor": {"state.morale": 1, "state.focus": 2}}, "relationship": {"respect": 2}}),
    ], float(ctx["age"]) >= 34)

    # Trade / movement (37-44)
    _reg("ask_trade_welcome", "trade", "Ask whether player would welcome a trade", "If it's the right situation, I'm willing to listen.", [
        _gm_choice("explore", "Explore mutual interest", "Trade openness.", {"profile_changes": {"actor": {"state.morale": 1}}, "relationship": {"trust": 1}}),
        _gm_choice("stay", "Encourage staying", "Commitment.", {"profile_changes": {"actor": {"state.belonging": 2}}, "relationship": {"loyalty": 2}}),
    ], ctx["struggling"] or float(ctx["state"].get("role_satisfaction") or 55) <= 45)
    _reg("tell_being_shopped", "trade", "Tell player they are being shopped", "So it's true — you're shopping me.", [
        _gm_choice("honest", "Confirm with honesty", "Transparency.", {"profile_changes": {"actor": {"state.morale": -5, "state.gm_trust": -6}}, "relationship": {"trust": -8, "grievance": 6}}),
        _gm_choice("context", "Explain baseball context", "Business framing.", {"profile_changes": {"actor": {"state.morale": -3, "state.gm_trust": -3}}, "relationship": {"trust": -4}}),
    ], ctx["trade_attempts"] >= 1 or ctx["trade_rumor_heat"] >= 20)
    _reg("reassure_not_shopped", "trade", "Reassure player they are not being shopped", "Good — I needed to hear that.", [
        _gm_choice("firm", "Firm reassurance", "Trust rebuild.", {"profile_changes": {"actor": {"state.morale": 3, "state.gm_trust": 4}}, "relationship": {"trust": 5, "loyalty": 3}}),
        _gm_choice("vague", "Vague reassurance", "Weak trust.", {"profile_changes": {"actor": {"state.gm_trust": -2 if rel.get('broken_promises') else 1}}, "relationship": {"trust": -2 if rel.get("broken_promises") else 1}}),
    ], ctx["trade_rumor_heat"] >= 8)
    _reg("ask_ntc_waiver", "trade", "Ask player to waive NTC/NMC", "That's a big ask. I need a good reason.", [
        _gm_choice("destination", "Present specific destination", "Targeted waiver ask.", {"profile_changes": {"actor": {"state.media_stress": 2}}, "relationship": {"trust": -2}, "ntc_waiver_request": True}),
        _gm_choice("withdraw", "Withdraw the request", "Preserve relationship.", {"profile_changes": {"actor": {"state.gm_trust": 2}}, "relationship": {"trust": 2}}),
    ], ctx["contract"].get("has_trade_protection") and not _gm_on_cooldown(entity, "ask_ntc_waiver", ctx["day"]))
    _reg("ask_preferred_destinations", "trade", "Ask for preferred destinations", "If a move happens, here's what I'd consider.", [
        _gm_choice("listen", "Listen to list", "Intel gathering.", {"profile_changes": {"actor": {"state.gm_trust": 1}}, "relationship": {"communication": 2}}),
        _gm_choice("discourage", "Discourage trade talk", "Shut down.", {"profile_changes": {"actor": {"state.morale": -2}}, "relationship": {"trust": -2}}),
    ], ctx["contract"].get("has_trade_protection") or ctx["trade_rumor_heat"] >= 15)
    _reg("discuss_ahl_assignment", "trade", "Discuss AHL assignment", "Sending me down is a message.", [
        _gm_choice("development", "Frame as development", "Soft landing.", {"profile_changes": {"actor": {"state.focus": 2, "state.morale": -4}}, "relationship": {"communication": 2}}),
        _gm_choice("performance", "Performance-based", "Accountability.", {"profile_changes": {"actor": {"state.morale": -5, "state.focus": 3}}, "relationship": {"respect": 1, "grievance": 3}}),
    ], ctx["struggling"] and float(ctx["age"]) <= 28)
    _reg("discuss_call_up", "trade", "Discuss call-up opportunity", "I'm ready for the NHL.", [
        _gm_choice("soon", "Call-up is possible soon", "Hope.", {"profile_changes": {"actor": {"state.morale": 4, "state.confidence": 3}}, "promise": {"type": "call_up_opportunity", "player_id": ctx["player_id"], "due_games": 14, "description": "Evaluate for NHL call-up.", "success_readiness": 0.5}, "relationship": {"trust": 3}}),
        _gm_choice("work", "Keep working in AHL", "Patience.", {"profile_changes": {"actor": {"state.focus": 2, "state.morale": -1}}, "relationship": {"respect": 1}}),
    ], False)  # enabled when player on AHL — checked below
    _reg("address_trade_rumors", "trade", "Address trade rumors", "The rumors are everywhere. What's true?", [
        _gm_choice("deny", "Deny active talks", "Media calm.", {"profile_changes": {"actor": {"state.media_stress": -2}}, "relationship": {"trust": 2 if ctx["trade_attempts"] == 0 else -4}}),
        _gm_choice("acknowledge", "Acknowledge uncertainty", "Honest.", {"profile_changes": {"actor": {"state.gm_trust": 2, "state.morale": -2}}, "relationship": {"honesty": 3}}),
    ], ctx["trade_rumor_heat"] >= 10)

    # Team / relationships (45-48)
    _reg("ask_coaching_relationship", "team", "Ask about coaching relationship", "My relationship with the coach is… complicated.", [
        _gm_choice("mediate", "Offer to mediate with coach", "Bridge building.", {"profile_changes": {"actor": {"state.coach_trust": 3, "state.gm_trust": 3}}, "relationship": {"communication": 3}}),
        _gm_choice("support_coach", "Back coaching staff", "Hierarchy.", {"profile_changes": {"actor": {"state.coach_trust": 1, "state.morale": -1}}, "relationship": {"respect": 1}}),
    ], float(ctx["state"].get("coach_trust") or 55) <= 48)
    _reg("ask_locker_room_concerns", "team", "Ask about locker-room concerns", "There are some things in the room worth discussing.", [
        _gm_choice("listen", "Listen confidentially", "Safe space.", {"profile_changes": {"actor": {"state.gm_trust": 4, "state.belonging": 2}}, "relationship": {"trust": 4}}),
        _gm_choice("address", "Promise to address with leadership", "Action.", {"profile_changes": {"actor": {"state.morale": 2}}, "relationship": {"trust": 3}}),
    ])
    _reg("discuss_leadership", "team", "Discuss leadership / captaincy", "Leadership matters to this team.", [
        _gm_choice("praise_leader", "Praise leadership impact", "Recognition.", {"profile_changes": {"actor": {"state.confidence": 3, "state.morale": 2}}, "relationship": {"respect": 3}}),
        _gm_choice("expect_more", "Expect more vocal leadership", "Challenge.", {"profile_changes": {"actor": {"state.focus": 3}}, "relationship": {"respect": 2}}),
    ], float(p.get("leadership", 50)) >= 65)
    _reg("ask_mentor_younger", "team", "Ask player to mentor younger teammate", "I can help the young guys.", [
        _gm_choice("assign", "Assign mentorship role", "Culture building.", {"profile_changes": {"actor": {"state.belonging": 3, "state.morale": 2}}, "promise": {"type": "mentor_assignment", "player_id": ctx["player_id"], "due_games": 15, "description": "Mentor a younger teammate.", "success_readiness": 0.3}, "relationship": {"loyalty": 3}}),
        _gm_choice("decline", "Not the right time", "Pass.", {"profile_changes": {"actor": {"state.morale": -1}}, "relationship": {}}),
    ], float(p.get("leadership", 50)) >= 60 or float(entity.get("room_value") or 0) >= 65)

    # Personal (49-50)
    _reg("personal_wellbeing_check", "personal", "Check in on personal wellbeing", "I appreciate you asking off the ice.", [
        _gm_choice("support", "Offer support resources", "Care.", {"profile_changes": {"actor": {"state.personal_stress": -6, "state.gm_trust": 5}}, "relationship": {"trust": 5, "loyalty": 2}}),
        _gm_choice("boundaries", "Respect privacy", "Space.", {"profile_changes": {"actor": {"state.gm_trust": 2}}, "relationship": {"respect": 2}}),
    ], float(ctx["state"].get("personal_stress") or 30) >= 45 or float(ctx["life"].get("relocation_strain") or 0) >= 40)
    _reg("repair_relationship", "personal", "Repair relationship / clear the air", "We should talk about where things stand between us.", [
        _gm_choice("apologize", "Apologize for past friction", "Accountability.", {"profile_changes": {"actor": {"state.gm_trust": 6, "state.morale": 3}}, "relationship": {"trust": 6, "grievance": -8}}),
        _gm_choice("reset", "Reset expectations together", "Fresh start.", {"profile_changes": {"actor": {"state.gm_trust": 4, "state.focus": 2}}, "relationship": {"communication": 4}}),
        _gm_choice("deflect", "Minimize past issues", "Avoidance.", {"profile_changes": {"actor": {"state.gm_trust": -4}}, "relationship": {"trust": -5}}),
    ], rel.get("label") in ("Strained", "Broken") or rel.get("broken_promises", 0) > 0)

    tpl = templates.get(interaction_type)
    if tpl is None:
        return None
    if interaction_type == "discuss_call_up":
        bucket = ""
        for pid, pl, b in _gm_user_org_players(session):
            if pid == ctx["player_id"]:
                bucket = b
                break
        if bucket != "ahl_roster":
            return None
    if not tpl.get("available", True):
        return None
    if _gm_on_cooldown(entity, interaction_type, ctx["day"]):
        return None
    meeting_id = f"gmmeet_{uuid.uuid4().hex[:12]}"
    return {
        "id": meeting_id,
        "interaction_type": interaction_type,
        "category": tpl["category"],
        "title": tpl["title"],
        "initiator": "gm",
        "player_id": ctx["player_id"],
        "player_name": name,
        "status": "pending",
        "dialogue": [
            {"speaker": "GM", "text": tpl["title"] + "."},
            {"speaker": name, "text": tpl["player_line"]},
        ],
        "choices": tpl["choices"],
        "ovr_explanation": ctx["ovr_trend"] if interaction_type in ("explain_ovr_rising", "explain_ovr_falling") else None,
        "relationship": rel,
        "expires_day": ctx["day"] + 5,
    }


def get_available_gm_interactions(session: Any, player_id: str) -> List[Dict[str, Any]]:
    day, _, _ = _u_current_meta(session)
    cache = dict(getattr(session, "_gm_interactions_cache", None) or {})
    cache_key = f"{player_id}:{day}"
    if cache_key in cache:
        return list(cache[cache_key])
    ctx = _gm_build_context(session, player_id)
    if not ctx.get("entity"):
        return []
    catalog_ids = [
        "discuss_ice_time", "offer_increased_ice_time", "explain_reduced_ice_time", "discuss_line_assignment",
        "discuss_promotion", "discuss_demotion", "offer_pp_opportunity", "explain_pp_removal", "offer_pk_opportunity",
        "discuss_healthy_scratch", "discuss_return_lineup", "discuss_goalie_workload",
        "praise_performance", "challenge_performance", "ask_struggles", "discuss_scoring_drought",
        "discuss_defensive_struggles", "discuss_improvement", "discuss_consistency", "discuss_playoff_performance",
        "explain_ovr_rising", "explain_ovr_falling", "discuss_development_plan", "set_development_priority",
        "discuss_stalled_development", "discuss_aging_decline", "discuss_conditioning", "discuss_injury_recovery",
        "ask_extension_interest", "long_term_plans", "future_uncertain", "discuss_role_next_season",
        "discuss_upcoming_fa", "hometown_discount", "contract_expectations", "discuss_retirement",
        "ask_trade_welcome", "tell_being_shopped", "reassure_not_shopped", "ask_ntc_waiver",
        "ask_preferred_destinations", "discuss_ahl_assignment", "discuss_call_up", "address_trade_rumors",
        "ask_coaching_relationship", "ask_locker_room_concerns", "discuss_leadership", "ask_mentor_younger",
        "personal_wellbeing_check", "repair_relationship",
    ]
    available: List[Dict[str, Any]] = []
    for iid in catalog_ids:
        built = _gm_build_interaction(session, ctx, iid)
        if built is None:
            continue
        cat_label = next((lbl for cid, lbl in GM_MEETING_CATEGORIES if cid == built["category"]), built["category"])
        available.append({"id": iid, "category": built["category"], "category_label": cat_label, "title": built["title"]})
    cache[cache_key] = available
    session._gm_interactions_cache = cache
    return available


def start_gm_player_meeting(session: Any, player_id: str, interaction_type: str) -> Dict[str, Any]:
    _ensure_gm_meeting_state(session)
    migrate_session_storyline_state(session)
    ctx = _gm_build_context(session, player_id)
    if not ctx.get("entity"):
        raise ValueError("Player not found in organization.")
    meeting = _gm_build_interaction(session, ctx, str(interaction_type))
    if meeting is None:
        raise ValueError("That conversation is not available for this player right now.")
    active = dict(getattr(session, "gm_active_meetings", None) or {})
    active[str(meeting["id"])] = meeting
    session.gm_active_meetings = active
    return {"ok": True, "meeting": meeting}


def resolve_gm_player_meeting(session: Any, meeting_id: str, choice_id: str) -> Dict[str, Any]:
    _ensure_gm_meeting_state(session)
    active = dict(getattr(session, "gm_active_meetings", None) or {})
    meeting = active.get(str(meeting_id))
    if meeting is None:
        raise ValueError(f"Active GM meeting not found: {meeting_id}")
    choice = next((c for c in (meeting.get("choices") or []) if str(c.get("id")) == str(choice_id)), None)
    if choice is None:
        raise ValueError(f"Choice not found: {choice_id}")
    ctx = _gm_build_context(session, str(meeting.get("player_id") or ""))
    outcome = dict(choice.get("outcome") or {})
    result = _gm_apply_meeting_outcome(session, ctx, str(meeting.get("interaction_type") or ""), choice)
    if outcome.get("ntc_waiver_request"):
        player = ctx.get("player")
        utid = str(getattr(session, "user_team_id", "") or "")
        team = (getattr(session, "team_by_id", None) or {}).get(utid)
        try:
            from app.sim_engine.trades.trade_rules import evaluate_ntc_waiver_request
            waiver = evaluate_ntc_waiver_request(player, source_team=team, destination_team=team, context={"meeting_request": True})
            trust = float((ctx["entity"].get("state") or {}).get("gm_trust") or 55)
            chance = float(waiver.get("accept_chance") or 0.35) + (trust - 55) * 0.004
            life = ctx.get("life") or {}
            if float(life.get("community_connection") or 0) >= 60:
                chance -= 0.08
            if float(life.get("relocation_strain") or 0) >= 50:
                chance -= 0.05
            if _gm_broken_promises(session, str(meeting.get("player_id") or "")):
                chance -= 0.12
            chance = max(0.03, min(0.88, chance))
            accepted = float(waiver.get("roll") or 0.5) < chance
            result["ntc_waiver"] = {**waiver, "accept_chance": round(chance, 3), "accepted": accepted, "meeting_context": True}
            entity = ctx["entity"]
            if accepted:
                _u_apply_profile_delta(entity, "state.gm_trust", 2)
                result["message"] = f"{meeting.get('player_name')} agreed to discuss waiving his clause."
            else:
                _u_apply_profile_delta(entity, "state.gm_trust", -5)
                _u_apply_profile_delta(entity, "state.morale", -4)
                result["message"] = f"{meeting.get('player_name')} declined to waive his no-trade protection."
        except Exception as exc:
            result["ntc_waiver"] = {"error": str(exc)}
    meeting["status"] = "resolved"
    active.pop(str(meeting_id), None)
    session.gm_active_meetings = active
    return result


def resolve_player_meeting_interaction(session: Any, interaction_id: str, choice_id: str) -> Dict[str, Any]:
    """Resolve a player-initiated universe meeting and record GM meeting history."""
    _ensure_gm_meeting_state(session)
    result = resolve_universe_interaction(session, interaction_id, choice_id)
    interaction = dict(result.get("interaction") or {})
    player_id = str(interaction.get("player_id") or interaction.get("actor_id") or "")
    entity = _gm_entity(session, player_id) if player_id else {}
    day, iso, _ = _u_current_meta(session)
    hist = list(getattr(session, "gm_meeting_history", None) or [])
    hist.append({
        "id": str(interaction.get("id") or interaction_id),
        "player_id": player_id,
        "player_name": interaction.get("player_name") or entity.get("player_name"),
        "interaction_type": interaction.get("kind"),
        "initiator": "player",
        "choice_id": choice_id,
        "calendar_day": day,
        "calendar_iso": iso,
        "relationship_snapshot": _gm_relationship_summary(entity, session, player_id) if player_id else {},
        "receipts": result.get("receipts"),
    })
    session.gm_meeting_history = hist[-GM_MEETING_HISTORY_MAX:]
    return {"ok": True, **result}


def build_player_meetings_payload(session: Any) -> Dict[str, Any]:
    """Full Player Meetings office payload for frontend."""
    _ensure_gm_meeting_state(session)
    migrate_session_storyline_state(session)
    _u_sync_player_entities(session)
    utid = str(getattr(session, "user_team_id", "") or "")
    day, iso, _ = _u_current_meta(session)
    roster_rows: List[Dict[str, Any]] = []
    needs_attention: List[Dict[str, Any]] = []
    pending_player = list(getattr(session, "universe_interaction_queue", None) or [])
    pending_player = [r for r in pending_player if str(r.get("status") or "") == "pending" and str(r.get("team_id") or "") == utid]
    active_promises = [p for p in (getattr(session, "universe_promises", None) or []) if str(p.get("status") or "") == "active"]
    broken_promises = [p for p in (getattr(session, "universe_promises", None) or []) if str(p.get("status") or "") == "broken"]
    for pid, player, bucket in _gm_user_org_players(session):
        entity = _gm_entity(session, pid)
        rel = _gm_relationship_summary(entity, session, pid)
        contract = _gm_contract_info(player)
        ovr_trend = build_ovr_trend_explanation(session, pid)
        pst = _ensure_player_storyline_state(player)
        try:
            from app.sim_engine.franchise.player_agent_engine import ensure_player_agent, agent_public_view
            agent = agent_public_view(player, session)
        except Exception:
            agent = {}
        requested = any(str(r.get("player_id") or r.get("actor_id") or "") == pid for r in pending_player)
        row = {
            "player_id": pid,
            "player_name": str(entity.get("player_name") or _u_name(player)),
            "position": str(entity.get("position") or _u_position(player)),
            "age": int(entity.get("age") or _player_age(player)),
            "overall": round(float(entity.get("overall") or _player_ovr99(player)), 1),
            "ovr_trend": ovr_trend.get("direction"),
            "readiness_delta": ovr_trend.get("readiness_delta"),
            "roster_bucket": bucket,
            "morale_tier": _u_tier_label(float((entity.get("state") or {}).get("morale", 55))),
            "relationship": rel,
            "contract": contract,
            "agent": agent,
            "requested_meeting": requested,
            "has_active_promise": any(str(p.get("player_id") or "") == pid for p in active_promises),
            "broken_promise_count": sum(1 for p in broken_promises if str(p.get("player_id") or "") == pid),
            "trade_concern": int(pst.get("trade_rumor_heat") or 0) >= 15,
            "concern_label": rel.get("detail"),
        }
        roster_rows.append(row)
        if requested or row["has_active_promise"] or row["broken_promise_count"] or row["trade_concern"] or rel.get("label") in ("Strained", "Broken"):
            needs_attention.append(row)
    hist = list(getattr(session, "gm_meeting_history", None) or [])[-40:]
    active_gm = list((getattr(session, "gm_active_meetings", None) or {}).values())
    return {
        "calendar_day": day,
        "calendar_iso": iso,
        "needs_attention": needs_attention,
        "roster": sorted(roster_rows, key=lambda r: (-int(r.get("requested_meeting") or 0), r.get("relationship", {}).get("label") == "Broken", r.get("relationship", {}).get("label") == "Strained", -float(r.get("overall") or 0))),
        "player_requests": pending_player,
        "active_gm_meetings": active_gm,
        "promises": {
            "active": active_promises,
            "broken": broken_promises[-20:],
            "fulfilled": [p for p in (getattr(session, "universe_promises", None) or []) if str(p.get("status") or "") == "kept"][-20:],
        },
        "history": hist,
        "categories": [{"id": cid, "label": lbl} for cid, lbl in GM_MEETING_CATEGORIES],
    }


def get_player_meeting_detail(session: Any, player_id: str) -> Dict[str, Any]:
    ctx = _gm_build_context(session, player_id)
    profile = player_universe_profile(session, player_id)
    return {
        "player_id": player_id,
        "profile": profile,
        "relationship": ctx.get("relationship"),
        "ovr_explanation": ctx.get("ovr_trend"),
        "available_interactions": get_available_gm_interactions(session, player_id),
        "memories": list((ctx.get("entity") or {}).get("memories") or [])[-12:],
        "promises": [p for p in (getattr(session, "universe_promises", None) or []) if str(p.get("player_id") or "") == str(player_id)],
        "history": [h for h in (getattr(session, "gm_meeting_history", None) or []) if str(h.get("player_id") or "") == str(player_id)][-15:],
    }
