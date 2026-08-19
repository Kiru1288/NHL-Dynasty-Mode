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
import uuid
from typing import Any, Dict, List, Optional, Tuple

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
    for tm in (getattr(session, "team_by_id", None) or {}).values():
        for p in getattr(tm, "roster", None) or []:
            if str(getattr(p, "id", "") or "") == pid:
                return p
    return None


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
    }
    cause_type = cause_type_map.get(str(stype), "")
    return {
        "id": sid,
        "storyline_id": sid,
        "stable_key": stable_key,
        "type": stype,
        "category": category,
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


def _league_points_rank(session: Any, team_id: str) -> int:
    st = getattr(session, "standings", None)
    if st is None:
        return 16
    recs = getattr(st, "records", None) or {}
    rows: List[Tuple[int, str]] = []
    if isinstance(recs, dict):
        iter_rows = recs.items()
    elif isinstance(recs, list):
        iter_rows = ((getattr(rr, "team_id", None) or getattr(rr, "id", i), rr) for i, rr in enumerate(recs))
    else:
        return 16
    for tid, rr in iter_rows:
        pts = int(getattr(rr, "points", 0) or 0)
        rows.append((pts, str(tid)))
    rows.sort(key=lambda x: (-x[0], x[1]))
    for i, (_, tid) in enumerate(rows):
        if tid == str(team_id):
            return i + 1
    return len(rows) or 16


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
    segment = str(day_meta.get("segment") or "")
    if segment not in ("preseason", "regular"):
        return {"generated": 0, "skipped_cooldown": 0, "skipped_sample": 0}

    iso = str(day_meta.get("iso") or "")
    cur_day = int(calendar_idx)
    season = int(getattr(session, "season_calendar_year", 2025) or 2025)
    uid = str(getattr(session, "user_team_id", "") or "")
    stats = dict(getattr(session, "player_season_stats", None) or {})
    generated: List[Dict[str, Any]] = []
    skipped_cd = 0
    skipped_sample = 0

    # Build OVR lookup from rosters
    ovr_by_id: Dict[str, float] = {}
    age_by_id: Dict[str, int] = {}
    name_by_id: Dict[str, str] = {}
    team_name_by_id: Dict[str, str] = {}
    for tid, tm in (getattr(session, "team_by_id", None) or {}).items():
        team_name_by_id[str(tid)] = str(getattr(tm, "name", "") or getattr(tm, "city", "") or tid)
        for p in getattr(tm, "roster", None) or []:
            pid = str(getattr(p, "id", "") or "")
            if not pid:
                continue
            ovr_by_id[pid] = _player_ovr99(p)
            age_by_id[pid] = _player_age(p)
            name_by_id[pid] = str(getattr(p, "name", "") or "Player")

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
            player = _player_from_roster(session, str(pid))
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
        cap = _cap_hit_m(_player_from_roster(session, str(pid)) or object())

        # Star underperforming
        if ovr >= 84 and gp >= SKATER_GP_MINOR:
            major = gp >= SKATER_GP_MAJOR
            threshold = -2.2 if major else -1.4
            if ppg < exp_ppg * (0.45 if major else 0.62):
                sev = "major" if major and delta <= threshold * 2 else "minor"
                pri = "HIGH" if major else "MEDIUM"
                stype = "star_underperforming"
                headlines_minor = [
                    f"{pname} enters witness protection program on the scoresheet",
                    f"Local ${cap:.1f}M forward reportedly still searching for the net" if cap >= 4 else f"{pname}'s scoring lamp remains cold",
                ]
                headlines_major = [
                    "Top-line center producing at the pace of a decorative traffic cone",
                    f"The ${cap:.1f}M man has gone missing" if cap >= 6 else f"{pname}'s drought becoming impossible to ignore",
                ]
                hl = _headline_pick(r, headlines_major if major else headlines_minor)
                if sev == "major" and not _can_fire(session, f"{stype}_esc|{pid}", cur_day, "major")[0]:
                    hl = "Scoring drought becoming impossible to ignore"
                try_emit(
                    stable_key=f"{stype}|{pid}|{season}",
                    stype=stype,
                    category="performance",
                    severity=sev,
                    priority=pri,
                    tone="negative",
                    headline=hl,
                    description=f"{pname} ({round(ovr)} OVR) has {pts} points in {gp} games ({ppg:.2f} P/GP) vs ~{exp_pts:.0f} expected for role.",
                    short_summary=f"{round(ovr)} OVR · {gp} GP · {pts}P · expected ~{exp_pts:.0f}P",
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
                headline="Rookie refuses to wait politely for his development timeline",
                description=f"{pname} is outpacing development expectations with {pts} points in {gp} games.",
                short_summary=f"Rookie surge · {pts}P in {gp}GP",
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
            rank = _league_points_rank(session, tid)
            if rank >= 20 and ppg >= 0.85:
                try_emit(
                    stable_key=f"superstar_carry|{pid}|{season}",
                    stype="superstar_carrying",
                    category="team",
                    severity="minor",
                    priority="MEDIUM",
                    tone="mixed",
                    headline=f"{pname} accidentally becomes the entire offense",
                    description=f"{pname} is producing like a star on a team sitting {trec} (league rank ~{rank}).",
                    short_summary=f"{pts}P on a struggling club",
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
                headline="Cap hit remains elite, production has filed a missing persons report",
                description=f"${cap:.2f}M AAV vs {pts} points in {gp} games.",
                short_summary=f"${cap:.1f}M · {pts}P · {gp}GP",
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

        # Hot / cold skater streak (points in last stretch approximated by season pace)
        if gp >= SKATER_GP_MAJOR:
            if ppg >= exp_ppg * 1.35 and pts >= 8:
                try_emit(
                    stable_key=f"hot_streak|{pid}|{season}",
                    stype="hot_streak",
                    category="performance",
                    severity="minor",
                    priority="LOW",
                    tone="positive",
                    headline=f"{pname} is playing like the game owes him money",
                    description=f"Heating up at {ppg:.2f} points per game over {gp} appearances.",
                    short_summary=f"Hot · {ppg:.2f} P/GP",
                    cause="Sustained scoring pace above expected band.",
                    team_id=tid,
                    team_name=team_name_by_id.get(tid, tid),
                    player_id=str(pid),
                    player_name=pname,
                    player_position=pos,
                    player_overall=round(ovr, 1),
                    evidence={"points_per_game": round(ppg, 3), "games_played": gp, "points": pts},
                    effects={"player_morale": 2, "fan_confidence": 3, "media_pressure": 2},
                    heat=48,
                )
            elif ppg < exp_ppg * 0.5 and ovr >= 78:
                try_emit(
                    stable_key=f"cold_streak|{pid}|{season}",
                    stype="cold_streak",
                    category="performance",
                    severity="minor",
                    priority="LOW",
                    tone="negative",
                    headline=f"{pname}'s stick has gone colder than arena ice at 6 AM",
                    description=f"Cold stretch: {pts} points in {gp} games ({ppg:.2f} P/GP).",
                    short_summary=f"Cold · {ppg:.2f} P/GP",
                    cause="Extended scoring slump vs expected role.",
                    team_id=tid,
                    team_name=team_name_by_id.get(tid, tid),
                    player_id=str(pid),
                    player_name=pname,
                    player_position=pos,
                    player_overall=round(ovr, 1),
                    evidence={"points": pts, "games_played": gp, "expected_points": exp_pts},
                    effects={"player_morale": -2, "media_pressure": 3, "player_confidence": -3},
                    heat=44,
                )

    # --- Goalie triggers ---
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

        if sv_pct < exp_sv - 0.018 and major:
            try_emit(
                stable_key=f"goalie_meltdown|{pid}|{season}",
                stype="goalie_meltdown",
                category="performance",
                severity="major" if sv_pct < exp_sv - 0.028 else "minor",
                priority="HIGH" if sv_pct < exp_sv - 0.028 else "MEDIUM",
                tone="negative",
                headline="Goalie has apparently decided the puck deserves freedom",
                description=f"{pname} at {sv_pct:.3f} SV% and {gaa:.2f} GAA over {gp} starts.",
                short_summary=f"{sv_pct:.3f} SV% · {gp} GP",
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
                headline="Netminder commits grand larceny, steals another two points",
                description=f"{pname} rolling at {sv_pct:.3f} SV% over {gp} starts.",
                short_summary=f"Heater · {sv_pct:.3f} SV%",
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

    # --- Team triggers ---
    for tid, tm in (getattr(session, "team_by_id", None) or {}).items():
        tid = str(tid)
        gp = _team_games_played(session, tid)
        if gp < TEAM_GP_MIN:
            continue
        w, l, o, trec = _team_record(session, tid)
        win_pct = w / max(1, gp)
        rank = _league_points_rank(session, tid)
        strength = float((getattr(session, "strength_map", None) or {}).get(tid, 0.5) or 0.5)
        tname = team_name_by_id.get(tid, tid)

        # Surprise team
        if strength < 0.46 and win_pct >= 0.58 and gp >= TEAM_GP_MIN:
            try_emit(
                stable_key=f"surprise_team|{tid}|{season}",
                stype="surprise_team",
                category="league",
                severity="minor",
                priority="MEDIUM",
                tone="positive",
                headline=f"{tname} keeps winning games nobody scheduled on the calendar",
                description=f"Expected also-ran is {trec} with a {win_pct:.3f} win rate.",
                short_summary=f"Surprise · {trec}",
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
                severity="major",
                priority="HIGH",
                tone="negative",
                headline=f"{tname} begins annual tradition of making everyone nervous",
                description=f"Strong on paper club is {trec} through {gp} games.",
                short_summary=f"Collapse · {trec}",
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
                headline="Every point now feels like it comes with a stress tax",
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
                headline=f"{tname} skid has the room playing like the clock is guilty",
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
                headline=f"{tname} rolling like they found a cheat code for effort",
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

    # Cap storylines per day
    cap = 6
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
    for tm in (getattr(session, "team_by_id", None) or {}).values():
        for pl in getattr(tm, "roster", None) or []:
            _ensure_player_storyline_state(pl)


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
]

_REPORTER_BY_ID = {r["id"]: r for r in MEDIA_REPORTERS}

_CLAIM_CAUSE_TYPES = frozenset({
    "TRADE_DEMAND", "TRADE_RUMOR", "TRADE_HUB_REJECT", "TRADE_HUB_SOFT_BLOCK",
    "LOCKER_ROOM_PULSE", "AGENT_DISSATISFACTION",
})

_FACT_CAUSE_TYPES = frozenset({
    "PLAYER_TRADED", "LINEUP_SCRATCH", "LINEUP_PROMOTION", "INJURY", "PLAYER_LOW_PRODUCTION",
    "GOALIE_BAD_FORM", "LOSING_STREAK", "WINNING_STREAK",
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
    if "injury" in cat or "goalie" in cat:
        return _REPORTER_BY_ID["knox"]
    utid = str(getattr(session, "user_team_id") or "")
    if str(sl.get("team_id") or "") == utid:
        return _REPORTER_BY_ID["morin"]
    if int(sl.get("heat") or 0) >= 70 and sl.get("priority") != "CRITICAL":
        return _REPORTER_BY_ID["hart"]
    return _REPORTER_BY_ID["knox"]


def _knowledge_type_for_storyline(sl: Dict[str, Any]) -> str:
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
    posts = list(getattr(session, "social_posts", None) or [])
    reporter = _REPORTER_BY_ID.get(str(sl.get("reporter_id") or "")) or _pick_reporter_for_storyline(sl, session)
    iso = str(sl.get("calendar_iso") or sl.get("date") or "")
    story_id = str(sl.get("storyline_id") or sl.get("id") or "")

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
            "text": str(sl.get("summary") or sl.get("short_summary") or sl.get("headline") or "")[:280],
            "related_headline": str(sl.get("headline") or ""),
            "calendar_iso": iso,
            "heat": heat,
            "knowledge_type": sl.get("knowledge_type"),
            "likes": int(heat * 140 + random.randint(80, 900)),
            "reposts": int(heat * 35 + random.randint(10, 200)),
            "replies": int(heat * 18 + random.randint(5, 120)),
        }
    )

    pname = str(sl.get("player_name") or "")
    if pname and heat >= 28:
        posts.append(
            {
                "id": f"soc_{uuid.uuid4().hex[:10]}",
                "arc_id": sl.get("arc_id"),
                "storyline_id": story_id,
                "author_type": "fan",
                "author_name": f"{pname.split()[-1]} Sicko",
                "handle": f"@Fan{abs(hash(pname)) % 9000 + 1000}",
                "verified": False,
                "text": f"{'NOT GREAT' if heat >= 55 else 'Interesting'} — {sl.get('headline') or pname}",
                "related_headline": str(sl.get("headline") or ""),
                "calendar_iso": iso,
                "heat": max(10, heat - 15),
                "likes": int(heat * 45 + random.randint(20, 400)),
                "reposts": int(heat * 8),
                "replies": int(heat * 5),
            }
        )

    session.social_posts = posts[-200:]


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
        "social_posts": list(getattr(session, "social_posts", None) or [])[-60:],
        "world_events": list(getattr(session, "decision_event_log", None) or [])[-30:],
        "knowledge_graph": list(getattr(session, "knowledge_graph", None) or [])[-40:],
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
    sl["public_knowledge_level"] = public_level
    sl["gm_knows_more"] = gm_knows_more
    sl["visibility"] = "public" if layers["public"] else ("team_only" if is_user else "private")
    graph = list(getattr(session, "knowledge_graph", None) or [])
    graph.append(
        {
            "world_event_id": sl.get("world_event_id"),
            "storyline_id": sl.get("storyline_id"),
            "team_id": tid,
            "player_id": sl.get("player_id"),
            "knowledge_type": ktype,
            "public_knowledge_level": public_level,
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
    if ctype not in _CLAIM_CAUSE_TYPES and "trade" not in str(sl.get("category") or "").lower():
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
