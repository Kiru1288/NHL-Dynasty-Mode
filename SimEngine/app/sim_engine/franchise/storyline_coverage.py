"""
Published-universe coverage for franchise storylines.

Turns Player.traits / Player.psych, box scores, org desks, and locker
interactions into visible Newsroom / Insiders / Social beats.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.franchise.storyline_engine import (
    PLAYER_AGENTS,
    _AGENT_BY_ID,
    _can_fire,
    _league_points_rank,
    _mark_fired,
    _player_age,
    _player_from_roster,
    _player_ovr99,
    _pos_bucket,
    _stat_int,
    _team_display,
    _team_games_played,
    _team_record,
    _u_add_social_post,
    _u_all_players,
    _u_clip,
    _u_current_meta,
    _u_name,
    _u_personality,
    _u_position,
    _u_psych_value,
    _u_record_storyline,
    _u_sync_player_entities,
    apply_universe_matchup_context,
    apply_universe_postgame,
    build_universe_matchup_context,
)

BEAT_WRITERS: List[Dict[str, Any]] = [
    {"id": "morin", "name": "Rachel Morin", "outlet": "Team Ledger", "role": "beat_reporter", "specialty": "local", "markets": "user"},
    {"id": "ellison", "name": "Mark Ellison", "outlet": "NorthStar Hockey", "role": "national_insider", "specialty": "trades"},
    {"id": "knox", "name": "Derek Knox", "outlet": "NBN", "role": "analyst", "specialty": "performance"},
    {"id": "reid", "name": "Mason Reid", "outlet": "PuckFinance", "role": "cap_specialist", "specialty": "contracts"},
    {"id": "petrov", "name": "Alex Petrov", "outlet": "Future Ice", "role": "prospect_analyst", "specialty": "draft"},
    {"id": "lee", "name": "Jenna Lee", "outlet": "National Sports Desk", "role": "investigative", "specialty": "conduct"},
    {"id": "okada", "name": "Kenji Okada", "outlet": "Pacific Desk", "role": "beat_reporter", "specialty": "west"},
    {"id": "beaumont", "name": "Claire Beaumont", "outlet": "Atlantic Wire", "role": "beat_reporter", "specialty": "east"},
    {"id": "howe", "name": "Sam Howe", "outlet": "Crease Report", "role": "analyst", "specialty": "goalies"},
    {"id": "dops", "name": "League Desk", "outlet": "Department of Player Safety", "role": "officials", "specialty": "discipline"},
]


def personality_from_player(player: Any) -> Dict[str, float]:
    """Map real PersonalityTraits + PsychologyState onto the V2 personality dict."""
    return _u_personality(player, str(getattr(player, "id", "") or ""))


def identity_from_player(player: Any) -> Dict[str, Any]:
    ident = getattr(player, "identity", None)
    return {
        "name": _u_name(player),
        "age": _player_age(player),
        "birth_city": str(getattr(ident, "birth_city", "") or ""),
        "birth_country": str(getattr(ident, "birth_country", "") or ""),
        "draft_year": int(getattr(ident, "draft_year", 0) or 0),
        "draft_round": int(getattr(ident, "draft_round", 0) or 0),
        "draft_pick": int(getattr(ident, "draft_pick", 0) or 0),
        "position": _u_position(player),
        "overall": round(_player_ovr99(player), 1),
    }


def trusts_from_player(player: Any) -> Dict[str, float]:
    return {
        "coach": round(_u_psych_value(player, ("coach_trust", "coach_relationship"), 55.0), 1),
        "gm": round(_u_psych_value(player, ("trust_in_management",), 55.0), 1),
        "teammates": round(_u_psych_value(player, ("trust_in_teammates",), 55.0), 1),
        "room": round(_u_psych_value(player, ("locker_room_fit",), 55.0), 1),
    }


def refresh_entity_from_player(session: Any, team_id: str, player: Any, entity: Dict[str, Any]) -> Dict[str, Any]:
    entity["personality"] = personality_from_player(player)
    entity["identity"] = identity_from_player(player)
    entity["trusts"] = trusts_from_player(player)
    entity["player_name"] = _u_name(player)
    entity["team_id"] = str(team_id)
    entity["position"] = _u_position(player)
    entity["age"] = _player_age(player)
    entity["overall"] = round(_player_ovr99(player), 1)
    state = dict(entity.get("state") or {})
    state["morale"] = _u_psych_value(player, ("morale",), float(state.get("morale", 55)))
    state["confidence"] = _u_psych_value(player, ("confidence_level", "confidence"), float(state.get("confidence", 55)))
    state["coach_trust"] = _u_psych_value(player, ("coach_trust",), float(state.get("coach_trust", 55)))
    state["role_satisfaction"] = _u_psych_value(
        player, ("role_satisfaction", "ice_time_satisfaction"), float(state.get("role_satisfaction", 55))
    )
    state["media_stress"] = _u_psych_value(player, ("media_stress",), float(state.get("media_stress", 28)))
    entity["state"] = state
    return entity


def _emit_public(
    session: Any,
    *,
    headline: str,
    summary: str,
    cause_type: str,
    category: str,
    heat: int,
    team_id: str = "",
    player_id: str = "",
    player_name: str = "",
    evidence: Optional[Dict[str, Any]] = None,
    knowledge_type: str = "report",
    public_level: str = "reported",
    source_label: str = "",
    reporter: Optional[Dict[str, Any]] = None,
    stable_key: str = "",
) -> Optional[Dict[str, Any]]:
    writer = reporter or {}
    event = {
        "id": stable_key or f"cov_{cause_type}_{player_id}_{headline[:24]}",
        "kind": cause_type.lower(),
        "team_id": team_id,
        "player_id": player_id,
        "player_name": player_name,
        "participants": [pid for pid in (player_id,) if pid],
        "effects": {},
        "stable_key": stable_key,
        "knowledge_type": knowledge_type,
        "public_knowledge_level": public_level,
        "source_label": source_label or writer.get("name") or "League desk",
        "reporter_name": writer.get("name") or "",
        "outlet_name": writer.get("outlet") or "",
        "evidence": dict(evidence or {}),
    }
    return _u_record_storyline(
        session,
        event=event,
        headline=headline,
        summary=summary,
        cause_type=cause_type,
        category=category,
        heat=heat,
        public=True,
    )


def emit_concern_threshold_storylines(session: Any, rng: random.Random) -> int:
    """Publish when role / contract / belonging / winning pressure crosses a line."""
    entities = getattr(session, "universe_players", None) or {}
    day, _iso, season = _u_current_meta(session)
    emitted = 0
    for team_id, player in _u_all_players(session):
        player_id = str(getattr(player, "id", "") or "")
        entity = entities.get(player_id)
        if not entity:
            continue
        refresh_entity_from_player(session, str(team_id), player, entity)
        concerns = dict(entity.get("concerns") or {})
        psych = getattr(player, "psych", None)
        if psych is not None:
            role_sat = float(getattr(psych, "role_satisfaction", 0.5) or 0.5)
            ice_sat = float(getattr(psych, "ice_time_satisfaction", 0.5) or 0.5)
            belonging = float(getattr(psych, "locker_room_fit", 0.5) or 0.5)
            contract_p = float(getattr(psych, "contract_pressure", 0.5) or 0.5)
            role_sat = role_sat * 100.0 if role_sat <= 1.5 else role_sat
            ice_sat = ice_sat * 100.0 if ice_sat <= 1.5 else ice_sat
            belonging = belonging * 100.0 if belonging <= 1.5 else belonging
            contract_p = contract_p * 100.0 if contract_p <= 1.5 else contract_p
            role = concerns.setdefault("role", {"label": "Role and ice time", "importance": 50, "satisfaction": 58, "trend": 0})
            role["satisfaction"] = round((role_sat + ice_sat) / 2.0, 1)
            belong = concerns.setdefault(
                "team_belonging", {"label": "Belonging in the room", "importance": 50, "satisfaction": 60, "trend": 0}
            )
            belong["satisfaction"] = round(belonging, 1)
            contract = concerns.setdefault(
                "contract", {"label": "Contract security", "importance": 50, "satisfaction": 55, "trend": 0}
            )
            contract["satisfaction"] = round(max(8.0, 100.0 - contract_p), 1)
        w, l, o, _ = _team_record(session, str(team_id))
        gp = w + l + o
        pts_pct = (w * 2 + o) / max(2, gp * 2)
        winning = concerns.setdefault(
            "winning", {"label": "Competing for a winner", "importance": 50, "satisfaction": 55, "trend": 0}
        )
        winning["satisfaction"] = round(_u_clip(25 + pts_pct * 70), 1)
        entity["concerns"] = concerns
        personality = entity.get("personality") or {}
        name = entity.get("player_name") or "Player"
        tname = _team_display(session, str(team_id))
        checks = [
            (
                "role",
                "PLAYER_ROLE_FRUSTRATION",
                "locker_room",
                f"{name} wants a clearer read on his role",
                f"{name}'s ice-time and role satisfaction have dropped. Teammates notice the body language.",
                58,
            ),
            (
                "contract",
                "CONTRACT_YEAR_HEAT",
                "business",
                f"{name}'s camp is restless about the next contract",
                f"With security slipping, {name}'s agent is expected to test the market temperature.",
                62,
            ),
            (
                "team_belonging",
                "ROOM_BELONGING",
                "locker_room",
                f"{name} is drifting from the room",
                f"Belonging inside the {tname} locker room has thinned. Quiet players feel it first.",
                52,
            ),
            (
                "winning",
                "WINNING_CONCERN",
                "team",
                f"{name} is done pretending the record is fine",
                f"{tname}'s results are wearing on a competitor who measures himself against the standings.",
                55,
            ),
        ]
        for key, cause, category, headline, summary, heat in checks:
            row = concerns.get(key) or {}
            importance = float(row.get("importance", 50) or 50)
            satisfaction = float(row.get("satisfaction", 60) or 60)
            pressure = importance * (100.0 - satisfaction) / 100.0
            ambition_boost = float(personality.get("ambition", 50)) >= 68 and key in ("role", "winning")
            if pressure < (28 if ambition_boost else 34):
                continue
            stable = f"concern|{key}|{player_id}|{season}"
            ok, _rep = _can_fire(session, stable, day, "minor")
            if not ok:
                continue
            writers = [w for w in BEAT_WRITERS if w.get("specialty") in ("local", "performance", "contracts")]
            writer = rng.choice(writers or BEAT_WRITERS)
            _emit_public(
                session,
                headline=headline,
                summary=summary,
                cause_type=cause,
                category=category,
                heat=int(heat + min(18, pressure / 4)),
                team_id=str(team_id),
                player_id=player_id,
                player_name=name,
                evidence={
                    "concern": key,
                    "satisfaction": round(satisfaction, 1),
                    "importance": round(importance, 1),
                    "pressure": round(pressure, 1),
                },
                knowledge_type="claim" if key == "contract" else "report",
                public_level="rumour" if key == "contract" else "reported",
                reporter=writer,
                stable_key=stable,
            )
            _mark_fired(session, stable, day, "minor", 0)
            emitted += 1
            if emitted >= 8:
                return emitted
    return emitted


def _skater_rows(box: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    rows: List[Tuple[str, Dict[str, Any]]] = []
    hid = str(box.get("home_id") or "")
    aid = str(box.get("away_id") or "")
    for side, tid in (("home_skaters", hid), ("away_skaters", aid)):
        for row in list(box.get(side) or []):
            if isinstance(row, dict):
                rows.append((tid, row))
    return rows


def ingest_game_box_storylines(session: Any, box: Dict[str, Any]) -> int:
    """Hat tricks, shutouts, OT winners, first goals, fights from a finished box."""
    if not isinstance(box, dict):
        return 0
    day, _iso, season = _u_current_meta(session)
    hid = str(box.get("home_id") or "")
    aid = str(box.get("away_id") or "")
    hg = int(box.get("home_goals") or box.get("home_score") or 0)
    ag = int(box.get("away_goals") or box.get("away_score") or 0)
    ot = bool(box.get("overtime") or box.get("ot"))
    events = [ev for ev in list(box.get("scoring_events") or []) if isinstance(ev, dict)]
    emitted = 0
    log = dict(getattr(session, "player_recent_games", None) or {})
    bumped: set = set()

    def _bump_log(pid: str, goals: int, assists: int, points: int) -> None:
        if not pid:
            return
        hist = list(log.get(pid) or [])
        hist.append({"g": int(goals), "a": int(assists), "pts": int(points), "day": day})
        log[pid] = hist[-12:]
        bumped.add(pid)

    for tid, row in _skater_rows(box):
        pid = str(row.get("player_id") or row.get("id") or "")
        if not pid:
            continue
        g = int(row.get("g") or 0)
        a = int(row.get("a") or 0)
        pim = int(row.get("pim") or 0)
        _bump_log(pid, g, a, g + a)
        name = str(row.get("name") or "Player")
        tname = _team_display(session, tid)
        if g >= 3:
            stable = f"hattrick|{pid}|{day}"
            if _can_fire(session, stable, day, "minor")[0]:
                _emit_public(
                    session,
                    headline=f"{name} records a hat trick",
                    summary=f"{name} scored {g} as {tname} finished {hg}-{ag}.",
                    cause_type="HAT_TRICK",
                    category="performance",
                    heat=78,
                    team_id=tid,
                    player_id=pid,
                    player_name=name,
                    evidence={"goals": g, "assists": a, "final": f"{hg}-{ag}"},
                    reporter={"name": "Derek Knox", "outlet": "NBN", "specialty": "performance"},
                    stable_key=stable,
                )
                _mark_fired(session, stable, day, "minor", 0)
                emitted += 1
        if pim >= 15:
            stable = f"fight|{pid}|{day}"
            if _can_fire(session, stable, day, "minor")[0]:
                _emit_public(
                    session,
                    headline=f"{name} piles up penalty minutes as tempers go",
                    summary=f"{name} finished with {pim} PIM. The Department of Player Safety will have the tape.",
                    cause_type="ON_ICE_ALTERCATION",
                    category="league",
                    heat=64,
                    team_id=tid,
                    player_id=pid,
                    player_name=name,
                    evidence={"pim": pim},
                    knowledge_type="report",
                    reporter={"name": "League Desk", "outlet": "Department of Player Safety", "specialty": "discipline"},
                    stable_key=stable,
                )
                _mark_fired(session, stable, day, "minor", 0)
                emitted += 1
        stats = (getattr(session, "player_season_stats", None) or {}).get(pid) or {}
        career_g = _stat_int(stats, "g", "goals")
        if g >= 1 and career_g == g:
            player = _player_from_roster(session, pid)
            ovr = _player_ovr99(player) if player is not None else 99
            if ovr <= 82:
                stable = f"firstgoal|{pid}|{season}"
                if _can_fire(session, stable, day, "minor")[0]:
                    _emit_public(
                        session,
                        headline=f"{name} scores his first of the season",
                        summary=f"The first one is in the book. {name} got on the scoresheet in a {hg}-{ag} result.",
                        cause_type="FIRST_GOAL",
                        category="performance",
                        heat=48,
                        team_id=tid,
                        player_id=pid,
                        player_name=name,
                        evidence={"goals": g},
                        stable_key=stable,
                    )
                    _mark_fired(session, stable, day, "minor", 0)
                    emitted += 1

    if not bumped and events:
        goals_by: Dict[str, Dict[str, Any]] = {}
        for ev in events:
            pid = str(ev.get("scorer_id") or "")
            if not pid:
                continue
            row = goals_by.setdefault(pid, {"g": 0, "name": str(ev.get("scorer") or "Player"), "tid": str(ev.get("for_team_id") or "")})
            row["g"] += 1
        for pid, row in goals_by.items():
            _bump_log(pid, int(row["g"]), 0, int(row["g"]))
            if int(row["g"]) >= 3:
                stable = f"hattrick|{pid}|{day}"
                if _can_fire(session, stable, day, "minor")[0]:
                    _emit_public(
                        session,
                        headline=f"{row['name']} records a hat trick",
                        summary=f"{row['name']} scored {row['g']} in a {hg}-{ag} result.",
                        cause_type="HAT_TRICK",
                        category="performance",
                        heat=78,
                        team_id=str(row["tid"]),
                        player_id=pid,
                        player_name=str(row["name"]),
                        evidence={"goals": row["g"], "final": f"{hg}-{ag}"},
                        reporter={"name": "Derek Knox", "outlet": "NBN", "specialty": "performance"},
                        stable_key=stable,
                    )
                    _mark_fired(session, stable, day, "minor", 0)
                    emitted += 1

    if hg == 0 or ag == 0:
        shut_tid = hid if ag == 0 else aid
        other = aid if shut_tid == hid else hid
        goalie_blob = box.get("home_goalie") if shut_tid == hid else box.get("away_goalie")
        gname = ""
        gid = ""
        if isinstance(goalie_blob, dict):
            gname = str(goalie_blob.get("name") or "")
            gid = str(goalie_blob.get("player_id") or goalie_blob.get("id") or "")
        if not gname:
            gname = f"{_team_display(session, shut_tid)} netminder"
        stable = f"shutout|{shut_tid}|{day}"
        if _can_fire(session, stable, day, "minor")[0]:
            _emit_public(
                session,
                headline=f"{gname} throws a shutout",
                summary=f"{_team_display(session, shut_tid)} blanked {_team_display(session, other)} {max(hg, ag)}-0.",
                cause_type="SHUTOUT",
                category="goalie",
                heat=72,
                team_id=shut_tid,
                player_id=gid,
                player_name=gname,
                evidence={"final": f"{hg}-{ag}", "overtime": ot},
                reporter={"name": "Sam Howe", "outlet": "Crease Report", "specialty": "goalies"},
                stable_key=stable,
            )
            _mark_fired(session, stable, day, "minor", 0)
            emitted += 1

    if ot and events:
        last = events[-1]
        scorer = str(last.get("scorer") or "")
        scorer_id = str(last.get("scorer_id") or "")
        win_tid = str(last.get("for_team_id") or (hid if hg > ag else aid))
        if scorer:
            stable = f"otwinner|{scorer_id or scorer}|{day}"
            if _can_fire(session, stable, day, "minor")[0]:
                _emit_public(
                    session,
                    headline=f"{scorer} wins it in overtime",
                    summary=f"{scorer} ended it. {_team_display(session, win_tid)} leaves with two points in extra time.",
                    cause_type="OT_WINNER",
                    category="performance",
                    heat=70,
                    team_id=win_tid,
                    player_id=scorer_id,
                    player_name=scorer,
                    evidence={"overtime": True, "final": f"{hg}-{ag}"},
                    stable_key=stable,
                )
                _mark_fired(session, stable, day, "minor", 0)
                emitted += 1

    session.player_recent_games = log
    for tid, won in ((hid, hg > ag), (aid, ag > hg)):
        if tid:
            try:
                apply_universe_postgame(session, tid, {"won": won, "game_id": box.get("game_id"), "player_stats": {}})
            except Exception:
                pass
    return emitted


def emit_rolling_form_storylines(session: Any) -> int:
    """Hot/cold off last-10 game log, not season PPG."""
    log = dict(getattr(session, "player_recent_games", None) or {})
    if not log:
        return 0
    day, _iso, season = _u_current_meta(session)
    emitted = 0
    team_of: Dict[str, str] = {}
    for tid, player in _u_all_players(session):
        team_of[str(getattr(player, "id", "") or "")] = str(tid)
    for pid, hist in log.items():
        recent = list(hist)[-10:]
        if len(recent) < 5:
            continue
        pts = sum(int(g.get("pts") or 0) for g in recent)
        gp = len(recent)
        ppg = pts / max(1, gp)
        player = _player_from_roster(session, pid)
        if player is None:
            continue
        ovr = _player_ovr99(player)
        name = _u_name(player)
        team_id = team_of.get(pid, "")
        expected = max(0.15, (ovr - 62.0) / 42.0)
        if _pos_bucket(_u_position(player)) == "D":
            expected *= 0.55
        if ppg >= expected * 1.55 and pts >= 5:
            stable = f"formhot|{pid}|{season}|{day // 6}"
            if _can_fire(session, stable, day, "minor")[0]:
                _emit_public(
                    session,
                    headline=f"{name} is rolling over his last {gp}",
                    summary=f"{pts} points in {gp} games ({ppg:.2f} P/GP), well above a {round(ovr)} OVR baseline.",
                    cause_type="ROLLING_HOT",
                    category="performance",
                    heat=56,
                    team_id=team_id,
                    player_id=pid,
                    player_name=name,
                    evidence={"last_n": gp, "points": pts, "ppg": round(ppg, 3)},
                    stable_key=stable,
                )
                _mark_fired(session, stable, day, "minor", 0)
                emitted += 1
        elif ppg <= expected * 0.42 and ovr >= 78 and gp >= 6:
            stable = f"formcold|{pid}|{season}|{day // 6}"
            if _can_fire(session, stable, day, "minor")[0]:
                _emit_public(
                    session,
                    headline=f"{name}'s last {gp} have gone cold",
                    summary=f"{pts} points in {gp} games. This is a form slump, not a season-long indictment.",
                    cause_type="ROLLING_COLD",
                    category="performance",
                    heat=50,
                    team_id=team_id,
                    player_id=pid,
                    player_name=name,
                    evidence={"last_n": gp, "points": pts, "ppg": round(ppg, 3)},
                    stable_key=stable,
                )
                _mark_fired(session, stable, day, "minor", 0)
                emitted += 1
        if emitted >= 6:
            break
    return emitted


def publish_cpu_interaction_rumors(session: Any, rng: random.Random) -> int:
    """Turn auto-resolved locker scenes on other clubs into public rumors."""
    user_tid = str(getattr(session, "user_team_id", "") or "")
    emitted = 0
    entities = getattr(session, "universe_players", None) or {}
    for row in list(getattr(session, "universe_interactions", None) or [])[-24:]:
        if str(row.get("status") or "") != "resolved":
            continue
        if str(row.get("team_id") or "") == user_tid:
            continue
        if row.get("_published_rumor"):
            continue
        kind = str(row.get("kind") or "")
        if kind in ("mentor_session", "glue_intervention") and rng.random() > 0.22:
            row["_published_rumor"] = True
            continue
        writer = rng.choice(BEAT_WRITERS)
        actor = entities.get(str(row.get("actor_id") or "")) or {}
        headline = str(row.get("title") or "Locker-room chatter")
        _emit_public(
            session,
            headline=headline,
            summary=str(row.get("summary") or "Sources inside the room describe a notable interaction."),
            cause_type="LOCKER_ROOM_PULSE",
            category="rumor",
            heat=int(row.get("score") or 40),
            team_id=str(row.get("team_id") or ""),
            player_id=str(row.get("actor_id") or ""),
            player_name=str(actor.get("player_name") or ""),
            knowledge_type="speculation" if kind in ("mentor_session", "glue_intervention") else "claim",
            public_level="chatter" if kind in ("mentor_session", "glue_intervention") else "rumour",
            reporter=writer,
            stable_key=f"cpurumor|{row.get('id') or headline}",
        )
        row["_published_rumor"] = True
        emitted += 1
        if emitted >= 4:
            break
    return emitted


def emit_league_social(session: Any, rng: random.Random) -> int:
    """Agents leak on a schedule; beat writers file ambient notes."""
    entities = getattr(session, "universe_players", None) or {}
    created = 0
    rel = dict(getattr(session, "agent_relationships", None) or {})
    if rel and rng.random() < 0.42:
        pid = rng.choice(list(rel.keys()))
        agent_id = str((rel.get(pid) or {}).get("agent_id") or "")
        agent = _AGENT_BY_ID.get(agent_id) or rng.choice(PLAYER_AGENTS)
        if float(agent.get("leak_tendency") or 0) >= 0.35:
            entity = entities.get(pid) or {}
            _u_add_social_post(
                session,
                {
                    "author_type": "agent",
                    "author_id": agent.get("id"),
                    "author_name": agent.get("name"),
                    "agency": agent.get("agency"),
                    "handle": f"@{(agent.get('name') or 'Agent').replace(' ', '')}HQ",
                    "verified": True,
                    "text": f"Just know {entity.get('player_name') or 'my client'} is focused on winning. The rest will take care of itself.",
                    "sentiment": "leak",
                    "likes": rng.randint(400, 9000),
                    "knowledge_type": "claim",
                },
            )
            created += 1
    if rng.random() < 0.55:
        writer = rng.choice(BEAT_WRITERS)
        pool = [row for row in entities.values() if bool(row.get("active_roster", True))]
        entity = rng.choice(pool) if pool else {}
        _u_add_social_post(
            session,
            {
                "author_type": "reporter",
                "author_id": writer.get("id"),
                "author_name": writer.get("name"),
                "handle": f"@{(writer.get('name') or 'Desk').replace(' ', '')}",
                "verified": True,
                "outlet": writer.get("outlet"),
                "text": f"Checking in on {entity.get('player_name') or 'the room'} — {writer.get('specialty') or 'league'} desk is watching this one.",
                "sentiment": "desk",
                "likes": rng.randint(80, 4000),
                "knowledge_type": "report",
            },
        )
        created += 1
    return created


def emit_org_desk_storylines(session: Any, rng: random.Random) -> int:
    """Coaches, GMs, captains, AHL shuttle, DOPS-flavored league notes."""
    day, _iso, season = _u_current_meta(session)
    emitted = 0
    user_tid = str(getattr(session, "user_team_id", "") or "")
    for tid, tm in (getattr(session, "team_by_id", None) or {}).items():
        tid = str(tid)
        gp = _team_games_played(session, tid)
        rank = _league_points_rank(session, tid)
        tname = _team_display(session, tid)
        coach = str(getattr(tm, "coach_name", None) or getattr(tm, "coach", None) or "")
        if tid == user_tid:
            coach = coach or str(getattr(session, "head_coach_name", "") or "Head Coach")
        coach = coach or f"{tname} bench"
        if gp >= 18 and rank >= 24:
            stable = f"coachhot|{tid}|{season}"
            if _can_fire(session, stable, day, "minor")[0]:
                _emit_public(
                    session,
                    headline=f"Heat rising on {coach}",
                    summary=f"{tname} sits near the bottom (rank {rank}) and the bench is the first place the noise lands.",
                    cause_type="COACH_HOT_SEAT",
                    category="league",
                    heat=68 if tid == user_tid else 54,
                    team_id=tid,
                    evidence={"rank": rank, "games_played": gp},
                    knowledge_type="claim",
                    public_level="rumour",
                    reporter={"name": "Mark Ellison", "outlet": "NorthStar Hockey", "specialty": "trades"},
                    stable_key=stable,
                )
                _mark_fired(session, stable, day, "minor", 0)
                emitted += 1
        if gp >= 22 and rank >= 26:
            gm_name = str(getattr(tm, "gm_name", None) or getattr(session, "gm_name", None) or "the general manager")
            stable = f"gmseat|{tid}|{season}"
            if _can_fire(session, stable, day, "minor")[0]:
                _emit_public(
                    session,
                    headline=f"Ownership patience with {tname}'s front office is a live question",
                    summary=f"League sources say {gm_name} is absorbing the same standings pressure as the bench.",
                    cause_type="GM_JOB_SECURITY",
                    category="league",
                    heat=60,
                    team_id=tid,
                    evidence={"rank": rank},
                    knowledge_type="speculation",
                    public_level="chatter",
                    stable_key=stable,
                )
                _mark_fired(session, stable, day, "minor", 0)
                emitted += 1

        captains = []
        for p in getattr(tm, "roster", None) or []:
            if getattr(p, "is_captain", False) or getattr(p, "captain", False):
                captains.append(p)
            elif str(getattr(tm, "captain_id", "") or "") == str(getattr(p, "id", "") or ""):
                captains.append(p)
        if captains and rng.random() < 0.08:
            cap = captains[0]
            cname = _u_name(cap)
            stable = f"captain|{getattr(cap, 'id', '')}|{season}|{day // 12}"
            if _can_fire(session, stable, day, "minor")[0]:
                _emit_public(
                    session,
                    headline=f"{cname} is still the room's clearest voice",
                    summary=f"The {tname} captaincy remains the public face of a group that needs one.",
                    cause_type="CAPTAINCY_PULSE",
                    category="locker_room",
                    heat=36,
                    team_id=tid,
                    player_id=str(getattr(cap, "id", "") or ""),
                    player_name=cname,
                    stable_key=stable,
                )
                _mark_fired(session, stable, day, "minor", 0)
                emitted += 1

        ahl = [str(getattr(p, "id", "") or "") for p in (getattr(tm, "ahl_roster", None) or []) if getattr(p, "id", None)]
        snap = dict(getattr(session, "_ahl_roster_snapshot", None) or {})
        prev = list(snap.get(tid) or [])
        if prev:
            nhl_ids = {str(getattr(p, "id", "") or "") for p in (getattr(tm, "roster", None) or [])}
            called = [pid for pid in prev if pid in nhl_ids]
            sent = [pid for pid in ahl if pid not in prev]
            if called:
                pl = _player_from_roster(session, called[0])
                pname = _u_name(pl) if pl is not None else "A prospect"
                stable = f"callup|{called[0]}|{day}"
                if _can_fire(session, stable, day, "minor")[0]:
                    _emit_public(
                        session,
                        headline=f"{pname} recalled from the AHL",
                        summary=f"{tname} pulled {pname} onto the NHL roster. The shuttle is moving.",
                        cause_type="AHL_CALLUP",
                        category="league",
                        heat=44,
                        team_id=tid,
                        player_id=called[0],
                        player_name=pname,
                        stable_key=stable,
                    )
                    _mark_fired(session, stable, day, "minor", 0)
                    emitted += 1
            if sent:
                stable = f"senddown|{sent[0]}|{day}"
                if _can_fire(session, stable, day, "minor")[0]:
                    _emit_public(
                        session,
                        headline=f"{tname} assigns a body to the AHL",
                        summary="A roster crunch turned into a send-down. Waivers may still be in play.",
                        cause_type="AHL_SENDDOWN",
                        category="league",
                        heat=40,
                        team_id=tid,
                        stable_key=stable,
                    )
                    _mark_fired(session, stable, day, "minor", 0)
                    emitted += 1
        snap[tid] = ahl
        session._ahl_roster_snapshot = snap
        if emitted >= 5:
            break
    return emitted


def apply_matchup_to_scales(
    session: Any,
    hid: str,
    aid: str,
    h_scale: float,
    a_scale: float,
    game_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Fold locker-room matchup context into the live strength scales."""
    matchup = build_universe_matchup_context(session, str(hid), str(aid), game_meta)
    applied = apply_universe_matchup_context(
        {"home_win_probability": 0.5, "away_win_probability": 0.5},
        matchup,
    )
    delta = float(matchup.get("home_win_probability_delta") or 0.0)
    return (
        max(0.90, min(1.10, float(h_scale) * (1.0 + delta))),
        max(0.90, min(1.10, float(a_scale) * (1.0 - delta))),
        applied,
    )


def ensure_player_agents(session: Any, rng: random.Random) -> int:
    rel = dict(getattr(session, "agent_relationships", None) or {})
    assigned = 0
    for _tid, player in _u_all_players(session):
        pid = str(getattr(player, "id", "") or "")
        if not pid or pid in rel:
            continue
        agent = rng.choice(PLAYER_AGENTS)
        rel[pid] = {"agent_id": agent["id"], "trust": 0.55, "gm_trust": 0.5}
        assigned += 1
    session.agent_relationships = rel
    return assigned


def run_coverage_daily_pass(session: Any, rng: Optional[random.Random] = None) -> Dict[str, int]:
    r = rng or random.Random()
    entities = _u_sync_player_entities(session)
    for tid, player in _u_all_players(session):
        pid = str(getattr(player, "id", "") or "")
        entity = entities.get(pid)
        if entity:
            refresh_entity_from_player(session, str(tid), player, entity)
    ensure_player_agents(session, r)
    return {
        "concerns": emit_concern_threshold_storylines(session, r),
        "form": emit_rolling_form_storylines(session),
        "cpu_rumors": publish_cpu_interaction_rumors(session, r),
        "social": emit_league_social(session, r),
        "org_desk": emit_org_desk_storylines(session, r),
        "agents": len(getattr(session, "agent_relationships", None) or {}),
    }


def coverage_payload_fields(session: Any) -> Dict[str, Any]:
    entities = getattr(session, "universe_players", None) or {}
    user_tid = str(getattr(session, "user_team_id", "") or "")
    dossiers = []
    for entity in entities.values():
        if str(entity.get("team_id") or "") != user_tid:
            continue
        if not bool(entity.get("active_roster", True)):
            continue
        dossiers.append(
            {
                "player_id": entity.get("player_id"),
                "player_name": entity.get("player_name"),
                "identity": entity.get("identity") or {},
                "wants": entity.get("top_concerns") or [],
                "trusts": entity.get("trusts") or {},
                "remembers": list(entity.get("memories") or [])[-6:],
                "reputation": list(entity.get("reputation_tags") or entity.get("personality_tags") or []),
                "personality_tags": entity.get("personality_tags") or [],
                "niches": [n.get("label") for n in (entity.get("niche_abilities") or []) if n.get("label")],
                "overall": entity.get("overall"),
                "position": entity.get("position"),
            }
        )
    dossiers.sort(key=lambda row: (-float(row.get("overall") or 0), str(row.get("player_name") or "")))
    graph = list(getattr(session, "knowledge_graph", None) or [])
    return {
        "beat_writers": list(BEAT_WRITERS),
        "player_dossiers": dossiers[:28],
        "insider_items": graph[-80:],
        "knowledge_graph": graph[-80:],
    }
