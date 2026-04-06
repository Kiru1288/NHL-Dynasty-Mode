"""
Interactive franchise day engine: advances the league calendar one day at a time
by reusing SimEngine's season simulation loop (per-game) without modifying SimEngine.
"""

from __future__ import annotations

import random
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from services.franchise_paths import ensure_simengine_path

ensure_simengine_path()
import run_sim as rs  # noqa: E402

from app.sim_engine.entities.coach import CoachRole, generate_coach  # noqa: E402
from app.sim_engine.league import (
    compute_awards,
    generate_regular_season_schedule,
    simulate_playoffs,
)
from app.sim_engine.league.standings import StandingsTable

from services.franchise_session import FranchiseSession

try:
    from app.sim_engine.world import calendar as world_calendar
    from app.sim_engine.world import chemistry as world_chemistry
    from app.sim_engine.world import durability as world_durability
    from app.sim_engine.world import fatigue as world_fatigue
    from app.sim_engine.world import injuries as world_injuries
    from app.sim_engine.world import morale as world_morale
    from app.sim_engine.world import momentum as world_momentum
except Exception:
    world_momentum = None  # type: ignore
    world_fatigue = None  # type: ignore
    world_morale = None  # type: ignore
    world_chemistry = None  # type: ignore
    world_injuries = None  # type: ignore
    world_durability = None  # type: ignore
    world_calendar = None  # type: ignore


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _display_team(t: Any) -> str:
    city = str(getattr(t, "city", "") or "").strip()
    name = str(getattr(t, "name", "") or "").strip()
    if city and name:
        return f"{city} {name}"
    return rs._team_name(t)


def resolve_user_team(teams: List[Any], query: str) -> Any:
    q = (query or "").strip().lower()
    if not q:
        raise ValueError("Team query is empty.")
    matches: List[Any] = []
    for t in teams:
        raw_tid = getattr(t, "team_id", None)
        if raw_tid is not None and str(raw_tid).lower() == q:
            matches.append(t)
            continue
        tid = str(rs._team_id(t)).lower()
        disp = _display_team(t).lower()
        nm = str(getattr(t, "name", "") or "").lower()
        ct = str(getattr(t, "city", "") or "").lower()
        if q == tid or q in disp or q in nm or q in ct or q in f"{ct} {nm}".strip():
            matches.append(t)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No team matched {query!r}. Try city, nickname, or numeric team id.")
    hint = ", ".join(_display_team(x) for x in matches[:6])
    raise ValueError(f"Ambiguous team {query!r}; matches include: {hint}")


def _use_world_modules() -> bool:
    return all(
        m is not None
        for m in (
            world_momentum,
            world_fatigue,
            world_morale,
            world_chemistry,
            world_injuries,
            world_durability,
            world_calendar,
        )
    )


def apply_coach_archetype(coach: Any, archetype: str, rng: random.Random) -> None:
    arch = (archetype or "balanced").lower().replace(" ", "_").replace("-", "_")
    try:
        if arch in ("development", "development_first", "teacher"):
            coach.usage.trust_youth = _clamp(float(coach.usage.trust_youth) + 0.12)
            coach.usage.trust_veterans = _clamp(float(coach.usage.trust_veterans) - 0.05)
            coach.development.skill_growth_multiplier = min(
                1.15, float(coach.development.skill_growth_multiplier) + 0.06
            )
        elif arch in ("defense_first", "defensive", "structure"):
            coach.tactics.risk_tolerance = _clamp(float(coach.tactics.risk_tolerance) - 0.10)
            coach.usage.penalty_kill_conservatism = _clamp(
                float(coach.usage.penalty_kill_conservatism) + 0.08
            )
        elif arch in ("aggressive", "attack", "offense_first"):
            coach.tactics.risk_tolerance = _clamp(float(coach.tactics.risk_tolerance) + 0.12)
            coach.tactics.offensive_activation = _clamp(
                float(coach.tactics.offensive_activation) + 0.08
            )
        elif arch in ("players_coach", "culture", "leader"):
            coach.usage.meritocracy = _clamp(float(coach.usage.meritocracy) + 0.10)
            coach.room_temperature = _clamp(float(coach.room_temperature) + 0.08)
        else:
            # balanced: small random identity nudge
            coach.tactics.risk_tolerance = _clamp(float(coach.tactics.risk_tolerance) + rng.uniform(-0.03, 0.03))
    except Exception:
        pass


def _chaos_index(sim: Any, league: Any) -> float:
    ctx = getattr(league, "_tuning_context", None) or {}
    return float(ctx.get("chaos_index", getattr(league, "_chaos_index", 0.5)) or 0.5)


def start_franchise(
    *,
    team_query: str,
    head_coach_name: str,
    coach_archetype: str,
    seed: Optional[int] = None,
) -> FranchiseSession:
    ensure_simengine_path()
    from app.sim_engine.engine import SimEngine

    master = seed if seed is not None else random.randrange(1, 10**9)
    sim = SimEngine(seed=master, debug=False)
    league = sim.league
    try:
        setattr(league, "_runner_sim_engine", sim)
    except Exception:
        pass

    teams = list(getattr(league, "teams", None) or [])
    if not teams:
        raise RuntimeError("League has no teams after initialization.")

    user_team = resolve_user_team(teams, team_query)
    _tid = getattr(user_team, "team_id", None)
    if _tid is not None:
        uid = str(_tid)
    else:
        _oid = getattr(user_team, "id", None)
        uid = str(_oid) if _oid is not None else rs._team_id(user_team)
    sim.team = user_team

    coach = generate_coach(sim.rng, f"HIRE_{uid}", CoachRole.HEAD_COACH)
    coach.name = (head_coach_name or "Head Coach").strip() or "Head Coach"
    apply_coach_archetype(coach, coach_archetype, sim.rng)
    user_team.coach = coach
    sim.coach = coach

    schedule = generate_regular_season_schedule(sim.rng, teams, 82)
    by_day: Dict[int, List[Any]] = defaultdict(list)
    for slot in schedule:
        by_day[int(slot.day)].append(slot)
    days_sorted = sorted(by_day.keys())

    standings = StandingsTable(teams)
    team_by_id: Dict[str, Any] = {}
    team_ids: List[str] = []
    for idx, t in enumerate(teams):
        tid = getattr(t, "team_id", None) or getattr(t, "id", None) or f"T{idx:02d}"
        tid = str(tid)
        team_ids.append(tid)
        team_by_id[tid] = t

    sim._preseason_line_synergy_refresh(teams, sim.rng)
    strength_map = sim._build_strength_map(teams)
    use_world = _use_world_modules()
    play_days: Dict[str, Any] = {}
    if use_world and world_calendar is not None:
        play_days = world_calendar.build_team_play_days(schedule)

    session = FranchiseSession(
        session_id=FranchiseSession.new_id(),
        sim=sim,
        user_team_id=uid,
        head_coach_name=coach.name,
        coach_archetype=coach_archetype,
        season_calendar_year=2025,
        schedule=schedule,
        by_day=dict(by_day),
        days_sorted=days_sorted,
        day_index=0,
        standings=standings,
        team_by_id=team_by_id,
        team_ids=team_ids,
        strength_map=strength_map,
        prev_calendar_day=None,
        last_game_day={tid: None for tid in team_ids},
        play_days=play_days,
        injury_log_major=[],
        chaos_index=_chaos_index(sim, league),
        use_world=use_world,
        preseason_applied=True,
    )

    session.notifications.append(f"Franchise ready — {_display_team(user_team)} ({uid}).")
    session.notifications.append(f"Hired {coach.name} ({coach_archetype}). {len(days_sorted)} calendar days scheduled.")
    session.timeline.append("Welcome to Franchise Mode. Advance the day to begin the regular season.")
    return session


def _simulate_slots_for_day(session: FranchiseSession, calendar_day: int, slots: List[Any]) -> List[str]:
    sim = session.sim
    teams = list(sim.league.teams)
    r = sim.rng
    lines: List[str] = []
    user_tid = session.user_team_id

    for slot in slots:
        home = session.team_by_id.get(slot.home_id)
        away = session.team_by_id.get(slot.away_id)
        if home is None or away is None:
            continue
        d = int(slot.day)
        hid, aid = str(slot.home_id), str(slot.away_id)

        if session.use_world and world_momentum is not None:
            if session.prev_calendar_day is not None and d > session.prev_calendar_day:
                span = float(d - session.prev_calendar_day)
                world_momentum.decay_all_teams(teams, span * 0.06)
            session.prev_calendar_day = d

            for tid, tm in ((hid, home), (aid, away)):
                lg = session.last_game_day.get(tid)
                if lg is not None:
                    gap = d - lg - 1
                    if gap > 0:
                        world_fatigue.rest_roster(tm, gap, r)
                session.last_game_day[tid] = d

            hb2b = bool(
                session.play_days and world_calendar.is_back_to_back(session.play_days.get(hid, set()), d)
            )
            ab2b = bool(
                session.play_days and world_calendar.is_back_to_back(session.play_days.get(aid, set()), d)
            )

            hm = world_momentum.team_strength_modifier(home)
            am = world_momentum.team_strength_modifier(away)
            hc = world_chemistry.team_strength_modifier(home)
            ac = world_chemistry.team_strength_modifier(away)
            hf = world_fatigue.team_fatigue_strength_factor(home)
            af = world_fatigue.team_fatigue_strength_factor(away)
            hmr = world_morale.team_morale_strength_factor(home)
            amr = world_morale.team_morale_strength_factor(away)

            h_scale = max(0.93, min(1.07, hm * hc * hf * hmr))
            a_scale = max(0.93, min(1.07, am * ac * af * amr))

            base_noise = 1.0 + 0.22 * (session.chaos_index - 0.5)
            nh = world_chemistry.chemistry_chaos_dampen(home, base_noise)
            na = world_chemistry.chemistry_chaos_dampen(away, base_noise)
            _, ih = sim._identity_runner_strength_noise_factors(home)
            _, ia = sim._identity_runner_strength_noise_factors(away)
            noise_scale = 0.5 * (nh + na) * (0.5 * (ih + ia))

            world_fatigue.tick_roster_fatigue_for_game(home, r, hb2b, session.schedule, d, hid)
            world_fatigue.tick_roster_fatigue_for_game(away, r, ab2b, session.schedule, d, aid)

            hg, ag, ot = sim._simulate_game(
                r,
                home,
                away,
                session.strength_map,
                home_strength_scale=h_scale,
                away_strength_scale=a_scale,
                noise_scale=noise_scale,
            )

            world_momentum.update_momentum_after_game(home, hg, ag, r)
            world_momentum.update_momentum_after_game(away, ag, hg, r)
            blow = abs(hg - ag) >= 3
            world_chemistry.update_after_game(home, hg > ag, blow, r)
            world_chemistry.update_after_game(away, ag > hg, blow, r)

            for p in getattr(home, "roster", None) or []:
                if getattr(p, "retired", False):
                    continue
                world_morale.update_after_team_result(
                    p,
                    hg > ag,
                    hg - ag,
                    r,
                    role_satisfaction_proxy=float(
                        getattr(getattr(p, "psych", None), "role_satisfaction", 0.5) or 0.5
                    ),
                )
            for p in getattr(away, "roster", None) or []:
                if getattr(p, "retired", False):
                    continue
                world_morale.update_after_team_result(
                    p,
                    ag > hg,
                    ag - hg,
                    r,
                    role_satisfaction_proxy=float(
                        getattr(getattr(p, "psych", None), "role_satisfaction", 0.5) or 0.5
                    ),
                )

            for tm in (home, away):
                for pl in getattr(tm, "roster", None) or []:
                    if int(getattr(pl, "_world_injury_games_remaining", 0) or 0) > 0:
                        world_injuries.tick_games_missed(pl)

            for tm in (home, away):
                ev = world_injuries.maybe_injure_roster_subset(
                    tm, r, session.chaos_index, max_checks=7
                )
                for label, tier, games in ev:
                    if tier == "major":
                        tid_inj = str(getattr(tm, "team_id", None) or getattr(tm, "id", None) or "")
                        session.injury_log_major.append(
                            {
                                "player": label,
                                "tier": tier,
                                "games": games,
                                "team_id": tid_inj,
                            }
                        )
        else:
            _, nh = sim._identity_runner_strength_noise_factors(home)
            _, na = sim._identity_runner_strength_noise_factors(away)
            id_noise = 0.5 * (nh + na)
            hg, ag, ot = sim._simulate_game(r, home, away, session.strength_map, noise_scale=id_noise)

        session.standings.record_game(slot.home_id, slot.away_id, hg, ag, overtime=ot)

        if hid == user_tid or aid == user_tid:
            opp = away if hid == user_tid else home
            won = (hg > ag) if hid == user_tid else (ag > hg)
            wl = "W" if won else "L"
            gs = f"{hg}-{ag}"
            if ot:
                gs += " OT"
            lines.append(
                f"{wl} vs {_display_team(opp)} ({gs}) — calendar day {d}"
            )

    return lines


def _maybe_enqueue_post_day_decisions(session: FranchiseSession, user_lines: List[str]) -> None:
    """Lightweight GM prompts derived from engine state (no extra full-season sim)."""
    sim = session.sim
    user_team = session.team_by_id.get(session.user_team_id)
    if user_team is None:
        return
    r = sim.rng

    # Injury headline for user club
    for inj in reversed(session.injury_log_major[-5:]):
        if inj.get("team_id") != session.user_team_id:
            continue
        dec_id = f"dec_{uuid.uuid4().hex[:12]}"
        session.pending_decisions.append(
            {
                "id": dec_id,
                "kind": "injury_protocol",
                "title": "Medical staff report",
                "description": f"{inj.get('player')} — major injury (~{inj.get('games')} games). Choose how you message the room.",
                "options": [
                    {"id": "transparent", "label": "Transparent update (builds trust)"},
                    {"id": "minimize", "label": "Minimize publicly (reduces media noise)"},
                    {"id": "next_man", "label": "Next-man-up rhetoric (pressure on depth)"},
                ],
                "meta": {"injury": inj},
            }
        )
        break

    if user_lines and r.random() < 0.22:
        roster = [p for p in (getattr(user_team, "roster", None) or []) if not getattr(p, "retired", False)]
        if roster:
            p = r.choice(roster)
            ident = getattr(p, "identity", None)
            nm = str(getattr(ident, "name", None) or getattr(p, "name", None) or "Player")
            role = float(getattr(getattr(p, "psych", None), "role_satisfaction", 0.55) or 0.55)
            if role < 0.62 or r.random() < 0.3:
                dec_id = f"dec_{uuid.uuid4().hex[:12]}"
                session.pending_decisions.append(
                    {
                        "id": dec_id,
                        "kind": "ice_time",
                        "title": f"{nm} wants a larger role",
                        "description": "Agents and internal scouts disagree on fit. Your call.",
                        "options": [
                            {"id": "promote", "label": "Promote usage (+ morale short-term, fatigue risk)"},
                            {"id": "steady", "label": "Hold structure (stable room)"},
                            {"id": "bench_msg", "label": "Send message with minutes cut (discipline)"},
                        ],
                        "meta": {"player_name": nm},
                    }
                )

    if not user_lines and r.random() < 0.12:
        dec_id = f"dec_{uuid.uuid4().hex[:12]}"
        session.pending_decisions.append(
            {
                "id": dec_id,
                "kind": "trade_inquiry",
                "title": "Trade desk ping",
                "description": "Rival GM floats a futures-for-help concept. No names on paper yet.",
                "options": [
                    {"id": "listen", "label": "Stay open — scouting will dig"},
                    {"id": "decline", "label": "Decline politely"},
                    {"id": "counter", "label": "Counter with salary retention ask"},
                ],
                "meta": {},
            }
        )


def advance_franchise_day(session: FranchiseSession) -> Dict[str, Any]:
    if session.pending_decisions:
        raise ValueError("Resolve pending decisions before advancing.")

    if session.phase == "complete":
        return {"status": "complete", "message": "Season and playoffs finished. Start a new franchise to continue."}

    # Regular season days
    if session.day_index < len(session.days_sorted):
        d = session.days_sorted[session.day_index]
        slots = session.by_day.get(d, [])
        user_lines = _simulate_slots_for_day(session, d, slots)
        session.day_index += 1

        day_label = f"Season {session.season_calendar_year} · calendar day {d} ({session.day_index}/{len(session.days_sorted)})"
        session.timeline.append(day_label)
        for ln in user_lines[:6]:
            session.timeline.append(ln)
        if len(session.timeline) > 200:
            session.timeline = session.timeline[-200:]

        _maybe_enqueue_post_day_decisions(session, user_lines)
        return {"status": "ok", "calendar_day": d, "user_game_summaries": user_lines}

    # End regular season — run playoffs once
    if not session.playoffs_simulated:
        sim = session.sim
        teams = list(sim.league.teams)
        playoff_result = simulate_playoffs(sim.rng, session.standings, teams, session.strength_map)
        awards = compute_awards(session.standings, playoff_result, teams)
        session.playoffs_simulated = True
        session.phase = "complete"
        session.champion_id = str(getattr(playoff_result, "champion_id", "") or "")
        ch_name = session.team_by_id.get(session.champion_id)
        ch_disp = _display_team(ch_name) if ch_name else session.champion_id
        session.notifications.append(f"Playoffs simulated. Cup: {ch_disp}")
        session.timeline.append(f"POSTSEASON: Champion {ch_disp}")
        return {
            "status": "postseason",
            "champion_id": session.champion_id,
            "awards_keys": list(awards.keys()) if awards else [],
        }

    session.phase = "complete"
    return {"status": "complete"}


def apply_decision(session: FranchiseSession, decision_id: str, choice_id: str) -> None:
    for i, d in enumerate(session.pending_decisions):
        if d.get("id") == decision_id:
            kind = d.get("kind")
            session.pending_decisions.pop(i)
            session.timeline.append(f"Decision ({kind}): {choice_id}")
            user_team = session.team_by_id.get(session.user_team_id)

            if kind == "ice_time" and user_team is not None:
                roster = [p for p in (getattr(user_team, "roster", None) or []) if not getattr(p, "retired", False)]
                nm = (d.get("meta") or {}).get("player_name")
                target = None
                for p in roster:
                    ident = getattr(p, "identity", None)
                    pn = str(getattr(ident, "name", None) or getattr(p, "name", None) or "")
                    if nm and pn == nm:
                        target = p
                        break
                if target is not None and getattr(target, "psych", None) is not None:
                    psych = target.psych
                    if choice_id == "promote":
                        psych.role_satisfaction = _clamp(float(psych.role_satisfaction) + 0.1)
                    elif choice_id == "bench_msg":
                        psych.role_satisfaction = max(0.15, float(psych.role_satisfaction) - 0.12)
                    else:
                        psych.role_satisfaction = _clamp(float(psych.role_satisfaction) + 0.02)

            if kind == "injury_protocol":
                boost = 0.03 if choice_id == "transparent" else 0.01 if choice_id == "minimize" else 0.0
                if user_team is not None and boost:
                    for p in getattr(user_team, "roster", None) or []:
                        if getattr(p, "psych", None) is None:
                            continue
                        p.psych.role_satisfaction = _clamp(float(p.psych.role_satisfaction) + boost * 0.25)
            return
    raise ValueError("Decision not found")


def list_teams_summary() -> List[Dict[str, str]]:
    """Lightweight listing for setup UI (bootstraps engine, then throws away)."""
    ensure_simengine_path()
    from app.sim_engine.engine import SimEngine

    sim = SimEngine(seed=1, debug=False)
    teams = list(getattr(sim.league, "teams", None) or [])
    out: List[Dict[str, str]] = []
    for t in teams:
        raw = getattr(t, "team_id", None)
        tid = str(raw) if raw is not None else str(rs._team_id(t))
        out.append({"team_id": tid, "name": _display_team(t)})
    out.sort(key=lambda x: x["name"])
    return out


def build_state_payload(session: FranchiseSession) -> Dict[str, Any]:
    sim = session.sim
    user_team = session.team_by_id.get(session.user_team_id)
    rec = None
    if session.standings and user_team is not None:
        tid = str(
            getattr(user_team, "team_id", None)
            if getattr(user_team, "team_id", None) is not None
            else rs._team_id(user_team)
        )
        rec = session.standings.records.get(tid) or session.standings.records.get(session.user_team_id)

    roster_rows: List[Dict[str, Any]] = []
    if user_team is not None:
        for p in getattr(user_team, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            ident = getattr(p, "identity", None)
            ovr_f = getattr(p, "ovr", None)
            try:
                ov = float(ovr_f() if callable(ovr_f) else ovr_f)
            except Exception:
                ov = 0.0
            roster_rows.append(
                {
                    "name": str(getattr(ident, "name", None) or "?"),
                    "position": str(getattr(getattr(ident, "position", None), "value", ident) if ident else "?"),
                    "ovr": round(ov * 99, 1) if ov <= 1.5 else round(ov, 1),
                    "morale": round(float(getattr(getattr(p, "psych", None), "morale", 0.5) or 0.5), 3),
                }
            )
        roster_rows.sort(key=lambda x: -float(x.get("ovr") or 0))

    standings_rows: List[Dict[str, Any]] = []
    if session.standings:
        for tid, r in session.standings.records.items():
            standings_rows.append(
                {
                    "team_id": tid,
                    "name": getattr(r, "name", tid),
                    "gp": getattr(r, "gp", 0),
                    "w": getattr(r, "wins", 0),
                    "l": getattr(r, "losses", 0),
                    "otl": getattr(r, "otl", 0),
                    "pts": getattr(r, "points", 0),
                }
            )
        standings_rows.sort(key=lambda x: (-x["pts"], -(x["w"] - x["l"])))

    cap_hint = str(getattr(user_team, "cap_pressure", "moderate") if user_team else "?")
    strat = str(getattr(user_team, "strategy", "balanced") if user_team else "?")

    day_display = "Off-season"
    prog = None
    if session.phase == "regular" and session.days_sorted and session.day_index < len(session.days_sorted):
        cd = session.days_sorted[session.day_index]
        prog = f"{session.day_index + 1} / {len(session.days_sorted)}"
        day_display = f"{session.season_calendar_year} · Day {cd} (scheduled)"
    elif session.phase == "regular" and session.day_index >= len(session.days_sorted):
        day_display = "Regular season complete"
    elif session.phase == "complete":
        day_display = f"Season complete — Cup: {session.champion_id or '?'}"

    return {
        "session_id": session.session_id,
        "phase": session.phase,
        "season_year": session.season_calendar_year,
        "calendar_summary": day_display,
        "progress": prog,
        "team": {
            "id": session.user_team_id,
            "name": _display_team(user_team) if user_team else session.user_team_id,
            "coach": session.head_coach_name,
            "coach_archetype": session.coach_archetype,
            "record": (
                {
                    "gp": getattr(rec, "gp", 0),
                    "w": getattr(rec, "wins", 0),
                    "l": getattr(rec, "losses", 0),
                    "otl": getattr(rec, "otl", 0),
                    "pts": getattr(rec, "points", 0),
                }
                if rec
                else None
            ),
            "cap_pressure": cap_hint,
            "strategy": strat,
        },
        "pending_decisions": list(session.pending_decisions),
        "notifications": list(session.notifications[-24:]),
        "timeline": list(session.timeline[-80:]),
        "roster": roster_rows[:28],
        "standings": standings_rows[:32],
        "flags": {
            "playoffs_done": session.playoffs_simulated,
            "can_advance": len(session.pending_decisions) == 0 and session.phase != "complete",
        },
    }
