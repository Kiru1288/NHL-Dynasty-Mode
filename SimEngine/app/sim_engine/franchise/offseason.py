"""
Franchise offseason phase controller — year-over-year continuation after Stanley Cup.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.franchise.session import FranchiseSession
from app.sim_engine.franchise.calendar import (
    build_season_calendar,
    calendar_day_to_dict,
    last_regular_season_index,
    map_abstract_schedule_to_calendar,
    season_anchor_event_markers,
)

OFFSEASON_STAGES: Tuple[str, ...] = (
    "awards",
    "retirements",
    "salary_cap",
    "development_report",
    "draft_lottery",
    "draft",
    "re_sign",
    "free_agency",
    "roster_cleanup",
    "next_season_reveal",
)

STAGE_NEXT_EVENT: Dict[str, str] = {
    "awards": "retirements",
    "retirements": "salary_cap",
    "salary_cap": "development_report",
    "development_report": "draft_lottery",
    "draft_lottery": "draft",
    "draft": "re_sign",
    "re_sign": "free_agency",
    "free_agency": "roster_cleanup",
    "roster_cleanup": "generate_next_season",
    "next_season_reveal": "preseason_start",
}


def _sync_phase_fields(session: FranchiseSession) -> None:
    ph = str(getattr(session, "phase", "regular") or "regular")
    if ph == "complete":
        if session.playoffs_simulated and not getattr(session, "offseason_stage", None):
            session.phase = "post_cup"
            ph = "post_cup"
        elif getattr(session, "next_season_generated", False):
            session.phase = "preseason"
            ph = "preseason"
        else:
            session.phase = "offseason"
            session.offseason_stage = session.offseason_stage or "awards"
            ph = "offseason"
    session.season_phase = ph
    if session.champion_id and not session.stanley_cup_winner:
        session.stanley_cup_winner = session.champion_id
    if session.stanley_cup_winner and not session.champion_id:
        session.champion_id = session.stanley_cup_winner


def _serialize_award(award: Any) -> Dict[str, Any]:
    if isinstance(award, dict):
        return dict(award)
    return {
        "name": str(getattr(award, "name", "") or ""),
        "winner_name": str(getattr(award, "winner_name", "") or ""),
        "winner_team_id": str(getattr(award, "winner_team_id", "") or ""),
        "winner_player_id": str(getattr(award, "winner_player_id", "") or ""),
        "winner_team_name": str(getattr(award, "winner_team_name", "") or ""),
        "finalists": list(getattr(award, "finalists", None) or []),
        "candidates": list(getattr(award, "candidates", None) or []),
        "winner_stats": dict(getattr(award, "winner_stats", None) or {}),
        "rationale": str(getattr(award, "rationale", "") or ""),
    }


def _enqueue_offseason_popup(session: FranchiseSession, kind: str, title: str, headline: str) -> None:
    from app.sim_engine.franchise.engine import _append_unique_dict_event

    popup = {
        "id": f"{kind}_{int(session.season_calendar_year)}",
        "kind": kind,
        "title": title,
        "headline": headline,
        "priority": "CRITICAL",
    }
    _append_unique_dict_event(session.pending_ui_popups, popup)


def _playoff_lifecycle_log(session: FranchiseSession, event: str, **kwargs: Any) -> None:
    bits = [str(event or "event")]
    for key, value in kwargs.items():
        if value is not None and str(value).strip():
            bits.append(f"{key}={value}")
    session.timeline.append("PLAYOFFS: " + " ".join(bits))


def _serialize_playoff_series(series: Any) -> Dict[str, Any]:
    high_id = str(getattr(series, "team_high_id", "") or "")
    low_id = str(getattr(series, "team_low_id", "") or "")
    return {
        "round_index": int(getattr(series, "round_index", 1) or 1),
        "conference": getattr(series, "conference", None),
        "seed_high": int(getattr(series, "seed_high", 0) or 0),
        "seed_low": int(getattr(series, "seed_low", 0) or 0),
        "team_high_id": high_id,
        "team_low_id": low_id,
        "team_high": high_id,
        "team_low": low_id,
        "home_id": high_id,
        "away_id": low_id,
        "wins_high": int(getattr(series, "wins_high", 0) or 0),
        "wins_low": int(getattr(series, "wins_low", 0) or 0),
    }


def _build_playoff_payload(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.league.playoffs import build_playoff_first_round
    from app.sim_engine.franchise.serialization import _display_team, _franchise_team_abbrev

    standings = getattr(session, "standings", None)
    if standings is None:
        return {}

    first_round, playoff_teams = build_playoff_first_round(standings)
    team_rows: List[Dict[str, Any]] = []
    for rec in playoff_teams:
        tid = str(getattr(rec, "team_id", "") or "")
        tm = (getattr(session, "team_by_id", None) or {}).get(tid)
        team_rows.append(
            {
                "team_id": tid,
                "name": _display_team(tm) if tm else str(getattr(rec, "name", tid) or tid),
                "abbrev": _franchise_team_abbrev(tm) if tm else tid[:3].upper(),
                "seed": int(getattr(rec, "playoff_seed", 0) or 0),
                "w": int(getattr(rec, "wins", 0) or 0),
                "l": int(getattr(rec, "losses", 0) or 0),
                "otl": int(getattr(rec, "otl", 0) or 0),
                "pts": int(getattr(rec, "points", 0) or 0),
            }
        )

    matchups = [_serialize_playoff_series(s) for s in first_round]
    return {
        "first_round": matchups,
        "first_round_matchups": matchups,
        "matchups": matchups,
        "series": matchups,
        "playoff_teams": team_rows,
        "teams": team_rows,
    }


def _transition_to_playoff_ready(session: FranchiseSession) -> Dict[str, Any]:
    """Regular season finished — show playoff bracket UI before simulating the postseason."""
    from app.sim_engine.franchise.engine import invalidate_session_payload_caches
    from app.sim_engine.franchise.progression import _run_franchise_season_end_progression

    if session.playoffs_simulated:
        _sync_phase_fields(session)
        return {
            "status": str(session.phase),
            "season_phase": str(session.season_phase),
            "champion_id": session.champion_id,
        }

    if not getattr(session, "_year_end_progression_done", False):
        prog = _run_franchise_season_end_progression(session)
        sp = (prog.get("lifecycle") or {}).get("special_events", 0) if isinstance(prog.get("lifecycle"), dict) else 0
        session.timeline.append(
            f"YEAR-END: Roster aging + progression ({int(sp)} major career events league-wide)."
        )
        if int(prog.get("retired_removed", 0) or 0) > 0:
            session.notifications.append(
                f"{prog['retired_removed']} player(s) left active NHL rosters (retirement)."
            )
        setattr(session, "_year_end_progression_done", True)

    session.regular_season_complete = True
    session.phase = "playoff_ready"
    session.season_phase = "playoff_ready"
    session.next_important_event = "enter_playoffs"
    session.playoff_payload = _build_playoff_payload(session)
    session.playoffs_generated = True
    _enqueue_offseason_popup(
        session,
        "playoff_start",
        "Stanley Cup Playoffs",
        "The bracket is set — postseason begins",
    )
    invalidate_session_payload_caches(session, "playoff_ready")

    return {
        "status": "playoff_ready",
        "season_phase": "playoff_ready",
        "next_important_event": "enter_playoffs",
    }


def complete_playoffs(session: FranchiseSession) -> Dict[str, Any]:
    """Simulate playoffs only — transition to post_cup, do not end franchise."""
    from app.sim_engine.league import compute_awards, simulate_playoffs
    from app.sim_engine.franchise.serialization import _display_team
    from app.sim_engine.franchise.engine import invalidate_session_payload_caches

    if session.playoffs_simulated and str(getattr(session, "phase", "")) in ("post_cup", "offseason", "preseason", "complete"):
        return {
            "status": "post_cup",
            "season_phase": str(getattr(session, "phase", "post_cup")),
            "champion_id": session.champion_id,
            "already_done": True,
        }

    session.phase = "playoffs"
    session.season_phase = "playoffs"
    _playoff_lifecycle_log(session, "season_phase_updated", season_phase="playoffs")

    sim = session.sim
    teams = list(sim.league.teams)
    playoff_result = simulate_playoffs(sim.rng, session.standings, teams, session.strength_map)
    award_rows = list((session.player_season_stats or {}).values()) if session.player_season_stats else []
    awards = compute_awards(session.standings, playoff_result, teams, player_season_stats=award_rows)

    session.playoffs_simulated = True
    session.playoffs_done = True
    session.phase = "post_cup"
    session.season_phase = "post_cup"
    session.champion_id = str(getattr(playoff_result, "champion_id", "") or "") if playoff_result else ""
    session.stanley_cup_winner = session.champion_id

    try:
        from app.sim_engine.league.awards import apply_career_award_history, build_awards_payload

        season_year = int(getattr(session, "season_calendar_year", 0) or 0)
        season_seed = getattr(session, "franchise_seed", None) or season_year
        payload = build_awards_payload(awards, season=season_year, season_seed=season_seed)
        payload["metadata"] = dict(payload.get("metadata") or {})
        payload["metadata"]["season_year"] = season_year
        payload["metadata"]["result_id"] = f"awards:{season_year}:{season_seed}"
        session.awards_payload = payload
        try:
            apply_career_award_history(teams, awards, season_year, result_id=payload["metadata"]["result_id"])
        except Exception:
            pass
        awards_dict = payload.get("awards") or {}
    except Exception:
        awards_dict = {k: _serialize_award(v) for k, v in (awards or {}).items()}
        session.awards_payload = {"awards": awards_dict, "items": list(awards_dict.values())}
    session.awards_generated = True

    ch_name = session.team_by_id.get(session.champion_id)
    ch_disp = _display_team(ch_name) if ch_name else session.champion_id
    session.notifications.append(f"Playoffs complete. Stanley Cup: {ch_disp}")
    session.timeline.append(f"POSTSEASON: Champion {ch_disp}")
    _playoff_lifecycle_log(session, "stanley_cup_winner", champion=session.champion_id)

    payload = dict(getattr(session, "playoff_payload", None) or {})
    if playoff_result is not None:
        payload["champion_id"] = session.champion_id
        payload["finalist_ids"] = list(getattr(playoff_result, "finalist_ids", None) or [])
        payload["series_list"] = [
            {
                "round_index": int(getattr(s, "round_index", 0) or 0),
                "conference": getattr(s, "conference", None),
                "seed_high": int(getattr(s, "seed_high", 0) or 0),
                "seed_low": int(getattr(s, "seed_low", 0) or 0),
                "team_high_id": str(getattr(s, "team_high_id", "")),
                "team_low_id": str(getattr(s, "team_low_id", "")),
                "wins_high": int(getattr(s, "wins_high", 0) or 0),
                "wins_low": int(getattr(s, "wins_low", 0) or 0),
                "series_score": getattr(s, "series_score", lambda: "")(),
            }
            for s in (getattr(playoff_result, "series_list", None) or [])
        ]
    session.playoff_payload = payload
    session.next_important_event = "awards"
    _enqueue_offseason_popup(session, "awards", "Awards Night", "Season hardware awaits")
    invalidate_session_payload_caches(session, "post_cup")

    return {
        "status": "post_cup",
        "season_phase": "post_cup",
        "champion_id": session.champion_id,
        "awards_keys": list(awards_dict.keys()),
    }


def advance_season_phase(session: FranchiseSession, target: Optional[str] = None) -> Dict[str, Any]:
    """Single source-of-truth season phase controller."""
    from app.sim_engine.franchise.serialization import _regular_season_is_truly_complete

    _sync_phase_fields(session)
    phase = str(getattr(session, "phase", "regular") or "regular")
    tgt = str(target or "").strip().lower()

    if phase == "complete":
        _sync_phase_fields(session)
        phase = str(session.phase)

    if phase == "regular":
        if tgt in ("playoff_ready", "playoffs"):
            if _regular_season_is_truly_complete(session):
                return _transition_to_playoff_ready(session)
            raise RuntimeError("Regular season not complete.")
        if _regular_season_is_truly_complete(session):
            return _transition_to_playoff_ready(session)
        return {"status": "regular", "season_phase": "regular", "message": "Regular season in progress."}

    if phase == "playoff_ready":
        if tgt == "playoffs":
            return complete_playoffs(session)
        return {
            "status": "playoff_ready",
            "season_phase": "playoff_ready",
            "next_important_event": "enter_playoffs",
        }

    if phase == "playoffs":
        return complete_playoffs(session)

    if phase == "post_cup":
        session.phase = "offseason"
        session.season_phase = "offseason"
        session.offseason_stage = "awards"
        session.next_important_event = "awards"
        _enqueue_offseason_popup(session, "awards", "Awards Night", "Season hardware awaits")
        return {"status": "offseason", "season_phase": "offseason", "offseason_stage": "awards"}

    if phase == "offseason":
        return continue_offseason(session)

    if phase == "preseason":
        if tgt == "regular":
            session.phase = "regular"
            session.season_phase = "regular"
            session.next_important_event = ""
            return {"status": "regular", "season_phase": "regular"}
        return {"status": "preseason", "season_phase": "preseason", "next_important_event": "preseason_start"}

    return {"status": phase, "season_phase": phase}


def _offseason_stage_ready(session: FranchiseSession, stage: str) -> bool:
    stage = str(stage or "")
    if stage == "awards":
        return bool(session.awards_payload)
    if stage == "retirements":
        return bool(session.retirements_processed)
    if stage == "salary_cap":
        return bool((session.salary_cap_payload or {}).get("new_season_cap"))
    if stage == "development_report":
        return bool(getattr(session, "development_report_done", False))
    if stage == "draft_lottery":
        return bool(session.draft_lottery_done)
    if stage == "draft":
        return bool(session.draft_payload)
    if stage == "re_sign":
        return bool(session.resign_payload)
    if stage == "free_agency":
        return bool(session.free_agency_open or session.free_agents_payload)
    if stage == "roster_cleanup":
        return bool(session.roster_cleanup_payload)
    if stage == "next_season_reveal":
        return bool(session.next_season_payload)
    return False


def _ensure_offseason_stage_hydrated(session: FranchiseSession) -> None:
    """Populate the current offseason screen payload when the user lands on it."""
    _sync_phase_fields(session)
    phase = str(session.phase)
    if phase == "post_cup":
        if not _offseason_stage_ready(session, "awards"):
            _enter_awards_stage(session)
        return
    if phase != "offseason":
        return
    stage = str(getattr(session, "offseason_stage", "") or "awards")
    if stage == "next_season_reveal":
        return
    if _offseason_stage_ready(session, stage):
        return
    _stage_handler(session, stage)


def _stage_handler(session: FranchiseSession, stage: str) -> Dict[str, Any]:
    handlers = {
        "awards": _enter_awards_stage,
        "retirements": _process_retirements,
        "salary_cap": _advance_salary_cap,
        "development_report": _run_offseason_development,
        "draft_lottery": _run_draft_lottery,
        "draft": _prepare_draft_payload,
        "re_sign": _prepare_resign_payload,
        "free_agency": _open_free_agency,
        "roster_cleanup": _run_roster_cleanup,
        "next_season_reveal": _finalize_next_season_reveal,
    }
    fn = handlers.get(stage)
    if fn is None:
        raise ValueError(f"Unknown offseason stage: {stage!r}")
    return fn(session)


def continue_offseason(session: FranchiseSession) -> Dict[str, Any]:
    """Advance one offseason UI stage."""
    from app.sim_engine.franchise.engine import invalidate_session_payload_caches

    _sync_phase_fields(session)
    if str(session.phase) == "post_cup":
        session.phase = "offseason"
        session.offseason_stage = "awards"
        _enter_awards_stage(session)

    if str(session.phase) != "offseason":
        raise ValueError(f"Cannot continue offseason from phase {session.phase!r}")

    current = str(getattr(session, "offseason_stage", "") or "awards")
    if current not in OFFSEASON_STAGES:
        current = "awards"
        session.offseason_stage = current

    idx = OFFSEASON_STAGES.index(current)
    result: Dict[str, Any] = {}

    if current == "roster_cleanup" and not session.next_season_generated:
        _ensure_offseason_stage_hydrated(session)
        session.offseason_stage = "roster_cleanup"
        session.next_important_event = "generate_next_season"
        invalidate_session_payload_caches(session, "roster_cleanup")
        return {
            **result,
            "status": "offseason",
            "season_phase": "offseason",
            "offseason_stage": "roster_cleanup",
            "next_important_event": "generate_next_season",
            "needs_generate_next_season": True,
        }

    if current == "next_season_reveal":
        result = _finalize_next_season_reveal(session)
        session.phase = "preseason"
        session.season_phase = "preseason"
        session.offseason_stage = None
        session.next_important_event = "preseason_start"
        invalidate_session_payload_caches(session, "preseason")
        return {
            **result,
            "status": "preseason",
            "season_phase": "preseason",
            "offseason_stage": None,
            "next_important_event": "preseason_start",
        }

    if idx + 1 < len(OFFSEASON_STAGES):
        next_stage = OFFSEASON_STAGES[idx + 1]
        session.offseason_stage = next_stage
        result = _stage_handler(session, next_stage)
    else:
        session.phase = "preseason"
        session.season_phase = "preseason"
        session.offseason_stage = None
        session.next_important_event = "preseason_start"
        invalidate_session_payload_caches(session, "preseason")
        return {
            **result,
            "status": "preseason",
            "season_phase": "preseason",
            "offseason_stage": None,
            "next_important_event": "preseason_start",
        }

    session.next_important_event = STAGE_NEXT_EVENT.get(next_stage, next_stage)
    invalidate_session_payload_caches(session, f"offseason_{next_stage}")
    return {
        **result,
        "status": "offseason",
        "season_phase": "offseason",
        "offseason_stage": next_stage,
        "next_important_event": session.next_important_event,
    }


def _enter_awards_stage(session: FranchiseSession) -> Dict[str, Any]:
    if not session.awards_generated:
        session.awards_payload = session.awards_payload or {"awards": {}, "items": []}
    return {"awards": session.awards_payload}


def _process_retirements(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.franchise.retirement import run_franchise_retirement_pass

    payload = run_franchise_retirement_pass(session)
    return {"retirements": payload}


def _tick_league_contracts(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.franchise.engine import (
        _build_free_agent_row,
        _contract_years_remaining,
        _is_true_free_agent,
        _serialize_player_row,
        player_cap_hit_millions,
    )

    sim = session.sim
    league = getattr(sim, "league", None)
    teams = list(getattr(league, "teams", None) or [])
    expired_ufas: List[Dict[str, Any]] = []
    expired_rfas: List[Dict[str, Any]] = []

    fa_pool = list(getattr(league, "free_agents", None) or [])
    fa_ids = {str(getattr(p, "player_id", getattr(p, "id", "")) or "") for p in fa_pool}

    for team in teams:
        roster = list(getattr(team, "roster", None) or [])
        kept = []
        for p in roster:
            if getattr(p, "retired", False):
                continue
            c = getattr(p, "contract", None)
            if c is not None and hasattr(c, "tick_year"):
                try:
                    c.tick_year()
                except Exception:
                    yrs = _contract_years_remaining(p)
                    if yrs > 0 and hasattr(c, "years_remaining"):
                        c.years_remaining = max(0, int(getattr(c, "years_remaining", yrs)) - 1)
            yrs_left = _contract_years_remaining(p)
            if yrs_left <= 0:
                pid = str(getattr(p, "player_id", getattr(p, "id", "")) or "")
                rights = str(getattr(c, "rights_status", getattr(p, "rights_status", "UFA")) or "UFA").upper()
                row = _serialize_player_row(p, include_ratings=True, session=session, _team=team)
                if "RFA" in rights:
                    expired_rfas.append(row)
                else:
                    expired_ufas.append(row)
                    if pid and pid not in fa_ids:
                        fa_pool.append(p)
                        fa_ids.add(pid)
                continue
            kept.append(p)
        team.roster = kept

    if league is not None:
        setattr(league, "free_agents", fa_pool)

    session.contracts_ticked = True
    return {"expired_ufas": expired_ufas, "expired_rfas": expired_rfas}


def _advance_salary_cap(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.economy.cap_engine import advance_league_salary_cap, calculate_team_cap_snapshot
    from app.sim_engine.franchise.engine import _team_cap_snapshot

    tick = _tick_league_contracts(session)
    sim = session.sim
    league = getattr(sim, "league", None)
    sy = int(session.season_calendar_year)
    cap_row: Dict[str, Any] = {}
    try:
        cap_row = advance_league_salary_cap(league, sim.rng, season_year=sy + 1)
    except Exception:
        cap_row = {"upperLimit": float(getattr(league, "salary_cap_m", 88.0) or 88.0)}

    user_team = session.team_by_id.get(session.user_team_id)
    user_cap = _team_cap_snapshot(user_team, sim, session) if user_team else {}
    over_cap_teams: List[Dict[str, Any]] = []
    for tid, tm in (session.team_by_id or {}).items():
        snap = calculate_team_cap_snapshot(tm, league) if tm and league else {}
        space = float(snap.get("cap_space", snap.get("capSpace", 0)) or 0)
        if space < 0:
            over_cap_teams.append({"team_id": tid, "cap_space": space, "cap_hit": snap.get("cap_hit", 0)})

    prev_cap = float(getattr(session, "salary_cap_payload", {}).get("current_cap", 0) or 0)
    new_cap = float(cap_row.get("upperLimit", getattr(league, "salary_cap_m", 0)) or 0)
    payload = {
        "last_season_cap": prev_cap or new_cap,
        "new_season_cap": new_cap,
        "change": new_cap - (prev_cap or new_cap),
        "user_team_cap": user_cap,
        "over_cap_teams": over_cap_teams,
        "expired_ufas": tick.get("expired_ufas", []),
        "expired_rfas": tick.get("expired_rfas", []),
    }
    session.salary_cap_payload = payload
    session.timeline.append(f"OFFSEASON: Salary cap set to ${new_cap:.1f}M.")
    return {"salary_cap": payload}


def _run_offseason_development(session: FranchiseSession) -> Dict[str, Any]:
    if getattr(session, "development_report_done", False) and session.development_report_payload:
        return {"development_report": session.development_report_payload}

    import run_sim as rs
    from app.sim_engine.franchise.engine import _franchise_nhl_age_and_phase_tick

    sim = session.sim
    league = getattr(sim, "league", None)
    teams = list(getattr(league, "teams", None) or [])
    rng = sim.rng

    risers: List[Dict[str, Any]] = []
    fallers: List[Dict[str, Any]] = []

    before: Dict[str, float] = {}
    for team in teams:
        for p in getattr(team, "roster", None) or []:
            pid = str(getattr(p, "player_id", getattr(p, "id", "")) or "")
            before[pid] = float(getattr(p, "overall", getattr(p, "ovr", 0)) or 0)

    if getattr(rs, "_run_player_progression_pass", None):
        try:
            rs._run_player_progression_pass(teams, rng, None)
        except Exception:
            pass

    for team in teams:
        for p in getattr(team, "roster", None) or []:
            pid = str(getattr(p, "player_id", getattr(p, "id", "")) or "")
            after = float(getattr(p, "overall", getattr(p, "ovr", 0)) or 0)
            diff = after - before.get(pid, after)
            if abs(diff) >= 0.5:
                row = {
                    "player_id": pid,
                    "name": str(getattr(getattr(p, "identity", None), "name", getattr(p, "name", "")) or ""),
                    "delta": round(diff, 1),
                    "overall": after,
                }
                if diff > 0:
                    risers.append(row)
                else:
                    fallers.append(row)

    risers.sort(key=lambda r: -float(r.get("delta", 0)))
    fallers.sort(key=lambda r: float(r.get("delta", 0)))
    payload = {
        "risers": risers[:12],
        "fallers": fallers[:12],
        "prospects_ready": [],
    }
    session.development_report_payload = payload
    session.development_report_done = True
    return {"development_report": payload}


def _run_draft_lottery(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.franchise.engine import _build_standings_rows, _display_team

    if session.draft_lottery_done and session.draft_lottery_payload:
        return {"draft_lottery": session.draft_lottery_payload}

    sim = session.sim
    standings_rows = _build_standings_rows(session)
    ordered = sorted(standings_rows, key=lambda r: (int(r.get("pts", 0)), -int(r.get("w", 0))))
    picks: List[Dict[str, Any]] = []
    try:
        from app.sim_engine.draft.draft_lottery import LotteryTeam, run_draft_lottery

        lot_teams = []
        for i, row in enumerate(ordered[:16]):
            lot_teams.append(LotteryTeam(team_id=str(row.get("team_id", "")), points=int(row.get("pts", 0))))
        seed = hash((int(session.season_calendar_year), int(getattr(sim.rng, "getstate", lambda: (0,))()[1][0] if hasattr(sim.rng, "getstate") else 0))) % (2**31)
        result = run_draft_lottery(teams=lot_teams, seed=seed)
        order = list(getattr(result, "pick_order", None) or [])
        for pick_num, tid in enumerate(order[:16], start=1):
            orig_rank = next((i + 1 for i, r in enumerate(ordered) if str(r.get("team_id")) == str(tid)), pick_num)
            tm = session.team_by_id.get(str(tid))
            picks.append({
                "pick": pick_num,
                "team_id": str(tid),
                "team_name": _display_team(tm) if tm else str(tid),
                "original_rank": orig_rank,
                "movement": orig_rank - pick_num,
            })
    except Exception:
        for i, row in enumerate(ordered[:16], start=1):
            tid = str(row.get("team_id", ""))
            tm = session.team_by_id.get(tid)
            picks.append({
                "pick": i,
                "team_id": tid,
                "team_name": _display_team(tm) if tm else tid,
                "original_rank": i,
                "movement": 0,
            })

    payload = {"picks": picks, "order": picks}
    session.draft_lottery_payload = payload
    session.draft_lottery_done = True
    return {"draft_lottery": payload}


def _prepare_draft_payload(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.franchise.engine import get_cached_draft_class_rankings

    rankings = get_cached_draft_class_rankings(session, session.sim)
    prospects = list(rankings.get("prospects") or rankings.get("ranked") or [])
    user_picks = [p for p in (session.draft_lottery_payload or {}).get("picks", []) if str(p.get("team_id")) == str(session.user_team_id)]
    payload = {
        "prospects": prospects[:32],
        "board": prospects[:32],
        "current_pick": user_picks[0] if user_picks else None,
        "user_picks": user_picks,
        "draft_year": int(session.season_calendar_year) + 1,
    }
    session.draft_payload = payload
    session.draft_completed = True
    return {"draft": payload}


def _prepare_resign_payload(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.franchise.engine import (
        _contract_years_remaining,
        _serialize_player_row,
        player_cap_hit_millions,
    )

    user_team = session.team_by_id.get(session.user_team_id)
    expiring: List[Dict[str, Any]] = []
    if user_team:
        for p in getattr(user_team, "roster", None) or []:
            yrs = _contract_years_remaining(p)
            if yrs <= 1:
                row = _serialize_player_row(p, include_ratings=True, session=session, _team=user_team)
                row["years_remaining"] = yrs
                row["cap_hit"] = player_cap_hit_millions(p)
                c = getattr(p, "contract", None)
                rights = str(getattr(c, "rights_status", "UFA") or "UFA").upper()
                row["rights"] = rights
                expiring.append(row)

    sim = session.sim
    user_cap = {}
    try:
        from app.sim_engine.franchise.engine import _team_cap_snapshot
        user_cap = _team_cap_snapshot(user_team, sim, session) if user_team else {}
    except Exception:
        pass

    payload = {"expiring_contracts": expiring, "cap_snapshot": user_cap}
    session.resign_payload = payload
    return {"contracts": payload, "re_sign": payload}


def _open_free_agency(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.franchise.engine import get_contract_office

    office = get_contract_office(session)
    fas = list(office.get("free_agents") or office.get("freeAgents") or [])
    session.free_agents_payload = fas
    session.free_agency_open = True
    return {"free_agents": fas}


def _run_roster_cleanup(session: FranchiseSession) -> Dict[str, Any]:
    user_team = session.team_by_id.get(session.user_team_id)
    roster_count = len(getattr(user_team, "roster", None) or []) if user_team else 0
    issues: List[str] = []
    warnings: List[str] = []

    if roster_count < 18:
        warnings.append(f"NHL roster light ({roster_count} skaters)")
    if roster_count > 23:
        issues.append(f"NHL roster over limit ({roster_count})")

    goalies = sum(
        1 for p in (getattr(user_team, "roster", None) or [])
        if str(getattr(p, "position", "")).upper() == "G"
    ) if user_team else 0
    if goalies < 2:
        warnings.append(f"Goalie count low ({goalies})")

    payload = {
        "nhl_roster_count": roster_count,
        "goalie_count": goalies,
        "issues": issues,
        "warnings": warnings,
        "valid": len(issues) == 0,
    }
    session.roster_cleanup_payload = payload
    session.next_important_event = "generate_next_season"
    return {"roster_cleanup": payload}


def generate_next_season(session: FranchiseSession) -> Dict[str, Any]:
    """Build new schedule/calendar — only increments year when data exists."""
    from app.sim_engine.league import generate_regular_season_schedule
    from app.sim_engine.league.schedule_generator import _safe_team_id
    from app.sim_engine.league.standings import StandingsTable
    from app.sim_engine.franchise.engine import (
        _finalize_schedule_after_generation,
        _merge_abstract_schedule_to_by_day,
        invalidate_session_payload_caches,
    )

    if session.next_season_generated and session.next_season_payload:
        return {"next_season": session.next_season_payload, "already_generated": True}

    sim = session.sim
    teams = list(getattr(sim, "league", None).teams or [])
    gp = int(getattr(session, "games_per_team_schedule", 82) or 82)
    next_sy = int(session.season_calendar_year) + 1

    history_entry = {
        "season_year": int(session.season_calendar_year),
        "champion_id": session.champion_id,
        "game_results_count": len(getattr(session, "game_results", None) or []),
    }
    session.season_history.append(history_entry)

    schedule_raw = generate_regular_season_schedule(sim.rng, teams, gp)
    by_abs: Dict[int, List[Any]] = defaultdict(list)
    for slot in schedule_raw:
        by_abs[int(slot.day)].append(slot)
    abstract_keys = sorted(by_abs.keys())

    cal_objs = build_season_calendar(next_sy)
    nhl_cal = [calendar_day_to_dict(c) for c in cal_objs]
    last_reg_idx = last_regular_season_index(cal_objs)
    day_map = map_abstract_schedule_to_calendar(cal_objs, abstract_keys)
    by_day, schedule = _merge_abstract_schedule_to_by_day(by_abs, abstract_keys, day_map, nhl_cal)
    by_day, schedule, _ = _finalize_schedule_after_generation(by_day, nhl_cal, user_id=str(session.user_team_id))

    session.schedule = schedule
    session.by_day = dict(by_day)
    session.days_sorted = sorted(by_day.keys())
    session.nhl_calendar = nhl_cal
    session.calendar_cursor = 0
    session.nhl_regular_season_last_index = last_reg_idx
    session.standings = StandingsTable(teams)
    session.player_season_stats = {}
    session.game_results = []
    session.regular_season_complete = False
    session.playoffs_generated = False
    session.playoffs_simulated = False
    session.playoffs_done = False
    session.playoff_payload = {}
    session.champion_id = None
    session.stanley_cup_winner = None
    session.preseason_applied = False
    session.season_calendar_year = next_sy

    first_opp = ""
    uid = str(session.user_team_id)
    for slot in schedule[:40]:
        hid = str(getattr(slot, "home_id", getattr(slot, "home_team_id", "")) or "")
        aid = str(getattr(slot, "away_id", getattr(slot, "away_team_id", "")) or "")
        if uid in (hid, aid):
            opp_id = aid if hid == uid else hid
            opp = session.team_by_id.get(opp_id)
            from app.sim_engine.franchise.engine import _display_team
            first_opp = _display_team(opp) if opp else opp_id
            break

    anchors = season_anchor_event_markers(next_sy)
    opening = next((a for a in anchors if "opening" in str(a.get("key", "")).lower()), None)
    preseason = next((a for a in anchors if "preseason_start" in str(a.get("key", "")).lower()), None)

    payload = {
        "season_year": next_sy,
        "season_label": f"{next_sy}–{next_sy + 1}",
        "opening_night": opening.get("iso") if opening else None,
        "preseason_start": preseason.get("iso") if preseason else (nhl_cal[0].get("iso") if nhl_cal else None),
        "first_opponent": first_opp,
        "schedule_games": len(schedule),
    }
    session.next_season_payload = payload
    session.next_season_generated = True
    session.offseason_stage = "next_season_reveal"
    session.next_important_event = "preseason_start"
    session.phase = "offseason"
    session.season_phase = "offseason"
    session.timeline.append(f"NEW SEASON: {next_sy}–{next_sy + 1} schedule generated.")
    invalidate_session_payload_caches(session, "next_season")
    _enqueue_offseason_popup(session, "next_season_reveal", "New Season", f"{next_sy}–{next_sy + 1} ready")
    return {"next_season": payload, "status": "next_season_reveal"}


def _finalize_next_season_reveal(session: FranchiseSession) -> Dict[str, Any]:
    session.phase = "preseason"
    session.season_phase = "preseason"
    session.offseason_stage = None
    session.next_important_event = "preseason_start"
    return {"next_season": session.next_season_payload, "season_phase": "preseason"}


def build_offseason_state_extras(session: FranchiseSession) -> Dict[str, Any]:
    """Extra payload fields for build_state_payload."""
    _sync_phase_fields(session)
    _ensure_offseason_stage_hydrated(session)
    phase = str(session.phase)
    stage = getattr(session, "offseason_stage", None)

    can_advance = (
        phase in ("preseason", "regular")
        and len(getattr(session, "pending_decisions", None) or []) == 0
        and phase not in ("post_cup", "offseason")
    )
    can_continue_offseason = phase in ("post_cup", "offseason")
    can_generate = phase == "offseason" and stage == "roster_cleanup" and not session.next_season_generated
    is_terminal = False

    return {
        "offseason_stage": stage,
        "playoffs_done": bool(getattr(session, "playoffs_done", session.playoffs_simulated)),
        "stanley_cup_winner": session.stanley_cup_winner or session.champion_id,
        "awards": session.awards_payload,
        "retirements": session.retirements_payload,
        "retired_players_archive": list(getattr(session, "retired_players_archive", None) or []),
        "draft_lottery": session.draft_lottery_payload,
        "draft": session.draft_payload,
        "free_agents": session.free_agents_payload,
        "contracts": session.resign_payload,
        "salary_cap": session.salary_cap_payload,
        "development_report": session.development_report_payload,
        "roster_cleanup": session.roster_cleanup_payload,
        "next_season": session.next_season_payload,
        "season_history": list(getattr(session, "season_history", None) or []),
        "flags": {
            "playoffs_done": bool(getattr(session, "playoffs_done", session.playoffs_simulated)),
            "can_advance_day": can_advance,
            "can_enter_playoffs": phase == "playoff_ready",
            "can_advance_phase": phase in ("regular", "playoff_ready", "post_cup", "preseason"),
            "can_continue_offseason": can_continue_offseason,
            "can_generate_next_season": can_generate,
            "is_terminal_dead_end": is_terminal,
        },
    }
