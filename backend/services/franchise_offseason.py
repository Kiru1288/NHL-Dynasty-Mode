"""
Franchise offseason phase controller — year-over-year continuation after Stanley Cup.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from services.franchise_session import FranchiseSession
from services.nhl_season_calendar import (
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
    "draft_combine",
    "draft",
    "draft_review",
    "prospect_rights",
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
    "draft_lottery": "draft_combine",
    "draft_combine": "draft",
    "draft": "draft_review",
    "draft_review": "prospect_rights",
    "prospect_rights": "re_sign",
    "re_sign": "free_agency",
    "free_agency": "roster_cleanup",
    "roster_cleanup": "generate_next_season",
    "next_season_reveal": "preseason_start",
}

# Schema version each stage handler stamps on its payload. `_offseason_stage_ready`
# reads the same table: when a handler's shape changes, bump it here only. A handler
# writing a version the readiness check does not accept makes the stage look
# un-hydrated forever, so every state fetch re-runs its side effects.
STAGE_PAYLOAD_VERSION: Dict[str, int] = {
    "draft_review": 4,
    "prospect_rights": 5,
    "re_sign": 6,
    "free_agency": 4,
    "roster_cleanup": 4,
}

# Post-draft slice used by Hub timeline / resume labels (keeps pre-draft stages intact).
POST_DRAFT_STAGES: Tuple[str, ...] = (
    "draft",
    "draft_review",
    "prospect_rights",
    "re_sign",
    "free_agency",
    "roster_cleanup",
    "next_season_reveal",
)

# Signing bonuses unlock once club revenue clears this floor (stars / wins / global draw help).
SIGNING_BONUS_REVENUE_FLOOR_M = 155.0
OWN_FA_MORATORIUM_DAYS = 6
INSTANT_ACCEPT_INTEREST = 88.0


def signing_bonus_max_pct_for_revenue(revenue_m: float) -> float:
    """Higher revenue → larger signing-bonus room (massive cash-upfront deals)."""
    rev = float(revenue_m or 0)
    if rev < SIGNING_BONUS_REVENUE_FLOOR_M:
        return 0.0
    if rev >= 230:
        return 0.32
    if rev >= 210:
        return 0.26
    if rev >= 190:
        return 0.20
    if rev >= 170:
        return 0.14
    return 0.08



def _safe_attr_float(obj: Any, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        raw = getattr(obj, key, None)
        if raw is None:
            continue
        if callable(raw) and not isinstance(raw, (int, float)):
            try:
                raw = raw()
            except TypeError:
                continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return default


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mark_stage_entered(session: FranchiseSession, stage: str) -> None:
    entered = getattr(session, "offseason_stage_entered_at", None)
    if not isinstance(entered, dict):
        session.offseason_stage_entered_at = {}
        entered = session.offseason_stage_entered_at
    if stage and stage not in entered:
        entered[stage] = _now_iso()


def _mark_stage_completed(session: FranchiseSession, stage: str) -> None:
    done = list(getattr(session, "offseason_completed_stages", None) or [])
    if stage and stage not in done:
        done.append(stage)
        session.offseason_completed_stages = done
    completed = getattr(session, "offseason_stage_completed_at", None)
    if not isinstance(completed, dict):
        session.offseason_stage_completed_at = {}
        completed = session.offseason_stage_completed_at
    if stage:
        completed[stage] = _now_iso()


def invalidate_offseason_decision_payloads(session: FranchiseSession, *, reason: str = "") -> None:
    """Force rebuild of decision-sensitive stage payloads after Cap Ledger / rights actions."""
    session.prospect_rights_payload = {}
    session.resign_payload = {}
    existing_cleanup = getattr(session, "roster_cleanup_payload", None)
    # Soft-refresh Roster Check counts in place. Wiping to {} would re-run the
    # full compliance pipeline (buyouts / cap casualties) on the next hydrate.
    if isinstance(existing_cleanup, dict) and existing_cleanup.get("version"):
        try:
            _revalidate_roster_cleanup(session, existing_cleanup)
        except Exception:
            session.roster_cleanup_payload = {}
    else:
        session.roster_cleanup_payload = {}
    if reason:
        try:
            from services.franchise_sim import invalidate_session_payload_caches
            invalidate_session_payload_caches(session, f"offseason_decision_{reason}")
        except Exception:
            pass


def team_signing_bonus_eligibility(session: FranchiseSession, team_id: Optional[str] = None) -> Dict[str, Any]:
    """Signing bonuses require NHL revenue eligibility (stars / wins / global draw lift revenue)."""
    tid = str(team_id or session.user_team_id)
    team = session.team_by_id.get(tid)
    revenue_m = None
    league_revenue_m = None
    try:
        from services.league_operations import calculate_team_revenue, calculate_league_revenue

        row = calculate_team_revenue(session, team, tid, is_user=(tid == str(session.user_team_id)))
        revenue_m = float(row.get("revenue") or row.get("revenue_m") or 0) or None
        try:
            team_rows = []
            for oid, ot in (session.team_by_id or {}).items():
                try:
                    team_rows.append(
                        calculate_team_revenue(
                            session, ot, str(oid), is_user=(str(oid) == str(session.user_team_id))
                        )
                    )
                except Exception:
                    continue
            if team_rows:
                league_revenue_m = float(calculate_league_revenue(team_rows) or 0) or None
        except Exception:
            league_revenue_m = None
    except Exception:
        revenue_m = None
        try:
            revenue_m = float(getattr(team, "revenue_m", None) or getattr(team, "annual_revenue_m", None) or 0) or None
        except Exception:
            revenue_m = None
    if revenue_m is None:
        return {
            "eligible": False,
            "revenue_m": None,
            "league_revenue_m": league_revenue_m,
            "floor_m": SIGNING_BONUS_REVENUE_FLOOR_M,
            "max_bonus_pct": 0.0,
            "reason": "revenue_unavailable",
            "label": "Signing bonuses locked — revenue unavailable",
        }
    eligible = revenue_m >= SIGNING_BONUS_REVENUE_FLOOR_M
    max_pct = signing_bonus_max_pct_for_revenue(revenue_m) if eligible else 0.0
    return {
        "eligible": eligible,
        "revenue_m": round(revenue_m, 1),
        "league_revenue_m": round(league_revenue_m, 1) if league_revenue_m is not None else None,
        "floor_m": SIGNING_BONUS_REVENUE_FLOOR_M,
        "max_bonus_pct": max_pct,
        "reason": None if eligible else "below_revenue_floor",
        "label": (
            None
            if eligible
            else f"Signing bonuses require NHL revenue ≥ ${SIGNING_BONUS_REVENUE_FLOOR_M:.0f}M (club at ${revenue_m:.1f}M)"
        ),
    }


def ensure_own_fa_window(session: FranchiseSession) -> Dict[str, Any]:
    """Start the 6-day exclusive window to sign your own free agents."""
    if not getattr(session, "own_fa_window_active", False) and not session.free_agency_open:
        session.own_fa_window_active = True
        session.own_fa_window_day = int(getattr(session, "own_fa_window_day", 0) or 0)
        if not isinstance(getattr(session, "own_fa_window_signings", None), list):
            session.own_fa_window_signings = []
    day = int(getattr(session, "own_fa_window_day", 0) or 0)
    remaining = max(0, OWN_FA_MORATORIUM_DAYS - day)
    return {
        "active": bool(getattr(session, "own_fa_window_active", False)) and not session.free_agency_open,
        "day": day,
        "days_total": OWN_FA_MORATORIUM_DAYS,
        "days_remaining": remaining,
        "complete": day >= OWN_FA_MORATORIUM_DAYS,
        "recent_signings": list(getattr(session, "own_fa_window_signings", None) or [])[-8:],
    }


def own_fa_window_status(session: FranchiseSession) -> Dict[str, Any]:
    day = int(getattr(session, "own_fa_window_day", 0) or 0)
    active = bool(getattr(session, "own_fa_window_active", False)) and not bool(session.free_agency_open)
    remaining = max(0, OWN_FA_MORATORIUM_DAYS - day)
    return {
        "active": active,
        "day": day,
        "days_total": OWN_FA_MORATORIUM_DAYS,
        "days_remaining": remaining,
        "complete": day >= OWN_FA_MORATORIUM_DAYS or bool(session.free_agency_open),
        "recent_signings": list(getattr(session, "own_fa_window_signings", None) or [])[-8:],
        "instant_accept_interest": INSTANT_ACCEPT_INTEREST,
    }


RESIGN_PHASE_TERMINAL = frozenset({"accepted", "released", "lapsed"})
RESIGN_PHASE_STATUSES = frozenset({
    "open", "pending", "countered", "accepted", "rejected", "released", "lapsed",
})


def ensure_resign_phase_outcomes(session: FranchiseSession) -> Dict[str, Any]:
    outcomes = getattr(session, "resign_phase_outcomes", None)
    if not isinstance(outcomes, dict):
        session.resign_phase_outcomes = {}
        outcomes = session.resign_phase_outcomes
    return outcomes


def upsert_resign_phase_outcome(
    session: FranchiseSession,
    *,
    player_id: str,
    phase_status: str,
    snapshot_row: Optional[Dict[str, Any]] = None,
    terms: Optional[Dict[str, Any]] = None,
    last_offer: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Record/update a re-sign desk outcome for the duration of the re_sign phase."""
    pid = str(player_id or "").strip()
    if not pid or session is None:
        return {}
    status = str(phase_status or "open").strip().lower()
    if status not in RESIGN_PHASE_STATUSES:
        status = "open"
    outcomes = ensure_resign_phase_outcomes(session)
    existing = outcomes.get(pid) if isinstance(outcomes.get(pid), dict) else {}
    # Terminal statuses stick unless an explicit later terminal overwrite (accepted/released/lapsed).
    prior = str(existing.get("phase_status") or "")
    if prior in RESIGN_PHASE_TERMINAL and status not in RESIGN_PHASE_TERMINAL:
        return existing

    row_snap = dict(snapshot_row) if isinstance(snapshot_row, dict) else dict(existing.get("snapshot_row") or {})
    if name and not row_snap.get("name"):
        row_snap["name"] = name
    if not row_snap.get("player_id"):
        row_snap["player_id"] = pid
    row_snap["phase_status"] = status
    if status == "accepted":
        row_snap["can_negotiate"] = False
        row_snap["available_actions"] = []
        if terms:
            if terms.get("aav_m") is not None:
                row_snap["aav_m"] = terms.get("aav_m")
                row_snap["cap_hit_m"] = terms.get("aav_m")
                row_snap["current_cap_hit"] = terms.get("aav_m")
            if terms.get("years") is not None:
                row_snap["years_remaining"] = terms.get("years")
                row_snap["years"] = terms.get("years")
            if terms.get("expiry_year") is not None:
                row_snap["expiry_year"] = terms.get("expiry_year")
        row_snap["contract_status"] = "signed"
        row_snap["negotiation_state"] = "accepted"
    elif status == "released":
        row_snap["can_negotiate"] = False
        row_snap["can_qualify"] = False
        row_snap["can_release_rights"] = False
        row_snap["available_actions"] = []
        row_snap["contract_status"] = "released"
        row_snap["negotiation_state"] = "released"
    elif status == "lapsed":
        row_snap["negotiation_state"] = "lapsed"
        row_snap["pending_offer"] = None
    elif status == "rejected":
        row_snap["negotiation_state"] = "rejected"
        # Rejected offers remain retryable while the player is still eligible.
        if row_snap.get("can_negotiate") is None:
            row_snap["can_negotiate"] = True
    elif status == "countered":
        row_snap["negotiation_state"] = "countered"
        row_snap["can_negotiate"] = True
    elif status == "pending":
        row_snap["negotiation_state"] = "pending"

    entry = {
        "player_id": pid,
        "name": row_snap.get("name") or name or existing.get("name") or pid,
        "phase_status": status,
        "snapshot_row": row_snap,
        "terms": dict(terms) if isinstance(terms, dict) else (existing.get("terms") or None),
        "last_offer": dict(last_offer) if isinstance(last_offer, dict) else (existing.get("last_offer") or None),
        "reason": reason if reason is not None else existing.get("reason"),
        "updated_at": _now_iso(),
        "window_day": int(getattr(session, "own_fa_window_day", 0) or 0),
        "terminal": status in RESIGN_PHASE_TERMINAL,
    }
    outcomes[pid] = entry
    return entry


def clear_resign_phase_state(session: FranchiseSession) -> None:
    """Archive/clear re-sign negotiation state when leaving the re_sign stage."""
    session.resign_phase_outcomes = {}
    session.resign_negotiations = {}
    session.resign_payload = {}
    session.own_fa_window_signings = list(getattr(session, "own_fa_window_signings", None) or [])


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
    try:
        from app.sim_engine.league.awards import serialize_award as _canon_serialize

        return _canon_serialize(award)
    except Exception:
        pass
    return {
        "name": str(getattr(award, "name", "") or ""),
        "award_id": str(getattr(award, "award_id", "") or ""),
        "winner_name": str(getattr(award, "winner_name", "") or ""),
        "winner_team_id": str(getattr(award, "winner_team_id", "") or ""),
        "winner_player_id": str(getattr(award, "winner_player_id", "") or ""),
        "winner_team_name": str(getattr(award, "winner_team_name", "") or ""),
        "finalists": list(getattr(award, "finalists", None) or []),
        "candidates": list(getattr(award, "candidates", None) or []),
        "winners": list(getattr(award, "winners", None) or []),
        "full_results": list(getattr(award, "full_results", None) or []),
        "shared": bool(getattr(award, "shared", False)),
        "status": str(getattr(award, "status", "complete") or "complete"),
        "official": bool(getattr(award, "official", True)),
        "display_metric": str(getattr(award, "display_metric", "") or ""),
        "calculation_quality": str(getattr(award, "calculation_quality", "full") or "full"),
        "fallback_reason": getattr(award, "fallback_reason", None),
        "unavailable_reason": getattr(award, "unavailable_reason", None),
        "winner_stats": dict(getattr(award, "winner_stats", None) or {}),
        "rationale": str(getattr(award, "rationale", "") or ""),
        "public_rationale": str(getattr(award, "public_rationale", "") or getattr(award, "rationale", "") or ""),
        "voting": getattr(award, "voting", None),
        "result": dict(getattr(award, "result", None) or {}),
    }


def _enqueue_offseason_popup(session: FranchiseSession, kind: str, title: str, headline: str) -> None:
    from services.franchise_sim import _append_unique_dict_event

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
    from services.franchise_sim import _display_team, _franchise_team_abbrev

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
    try:
        from services.franchise_playoffs import sanitize_first_round_matchups

        matchups = sanitize_first_round_matchups(matchups)
    except Exception:
        pass
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
    from services.franchise_sim import invalidate_session_payload_caches
    from services.franchise_sim import _run_franchise_season_end_progression

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


def complete_playoffs_from_live_result(session: FranchiseSession, live: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finish postseason using an already-simulated live bracket (interactive mode).
    Reuses awards / post_cup plumbing from complete_playoffs without re-simulating.
    """
    from app.sim_engine.league import compute_awards
    from app.sim_engine.league.awards import apply_career_award_history, build_awards_payload
    from app.sim_engine.league.playoffs import PlayoffResult, PlayoffSeries
    from services.franchise_sim import _display_team, invalidate_session_payload_caches

    if session.playoffs_simulated and str(getattr(session, "phase", "")) in (
        "post_cup",
        "offseason",
        "preseason",
        "complete",
    ):
        return {
            "status": "post_cup",
            "season_phase": str(getattr(session, "phase", "post_cup")),
            "champion_id": session.champion_id,
            "already_done": True,
        }

    champion = str(live.get("champion_id") or "")
    finalists = [str(x) for x in (live.get("finalist_ids") or []) if x]
    series_objs: List[PlayoffSeries] = []
    for s in live.get("series") or []:
        if not s.get("team_high_id") or not s.get("team_low_id"):
            continue
        if str(s.get("status")) not in ("complete", "active") and int(s.get("wins_high") or 0) + int(s.get("wins_low") or 0) == 0:
            # Skip untouched pending slots
            if str(s.get("status")) == "pending":
                continue
        series_objs.append(
            PlayoffSeries(
                round_index=int(s.get("round_index") or 1),
                conference=s.get("conference"),
                seed_high=int(s.get("seed_high") or 0),
                seed_low=int(s.get("seed_low") or 0),
                team_high_id=str(s.get("team_high_id")),
                team_low_id=str(s.get("team_low_id")),
                wins_high=int(s.get("wins_high") or 0),
                wins_low=int(s.get("wins_low") or 0),
            )
        )
    playoff_result = PlayoffResult(
        champion_id=champion,
        finalist_ids=finalists or [champion],
        series_list=series_objs,
    )

    sim = session.sim
    teams = list(sim.league.teams)
    season_year = int(getattr(session, "season_calendar_year", 0) or 0)
    award_rows = [dict(v) for v in (list((session.player_season_stats or {}).values()) if session.player_season_stats else [])]
    for row in award_rows:
        row.setdefault("stat_scope", "regular_season")
    playoff_rows = []
    for key in ("playoff_player_stats", "player_playoff_stats", "playoff_stats"):
        raw = getattr(session, key, None)
        if isinstance(raw, dict):
            playoff_rows = [dict(v) for v in raw.values()]
            break
        if isinstance(raw, list):
            playoff_rows = [dict(v) for v in raw if isinstance(v, dict)]
            break
    for row in playoff_rows:
        row["stat_scope"] = "playoffs"

    season_seed = getattr(session, "franchise_seed", None)
    if season_seed is None:
        rng = getattr(sim, "rng", None)
        raw_seed = getattr(rng, "_seed", None) if rng is not None else None
        # Never use rng.seed — that is Random.seed (a method), not an int.
        if isinstance(raw_seed, (int, float)) and not isinstance(raw_seed, bool):
            season_seed = int(raw_seed) & 0xFFFFFFFF
        else:
            season_seed = int(season_year) & 0xFFFFFFFF
    elif not isinstance(season_seed, (int, float)) or isinstance(season_seed, bool):
        season_seed = int(season_year) & 0xFFFFFFFF
    else:
        season_seed = int(season_seed) & 0xFFFFFFFF

    history_by_player = dict(getattr(session, "player_award_history", None) or {})
    awards = compute_awards(
        session.standings,
        playoff_result,
        teams,
        player_season_stats=award_rows,
        playoff_player_stats=playoff_rows,
        season_seed=season_seed,
        season_year=season_year,
        season_length=int(getattr(session, "season_length", 82) or 82),
        history_by_player=history_by_player,
    )

    session.playoffs_simulated = True
    session.playoffs_done = True
    session.phase = "post_cup"
    session.season_phase = "post_cup"
    session.champion_id = champion
    session.stanley_cup_winner = champion
    session.next_important_event = "awards"

    result_id = f"awards:{season_year}:{season_seed}"
    payload = build_awards_payload(
        awards,
        season=season_year,
        season_seed=season_seed,
        season_length=int(getattr(session, "season_length", 82) or 82),
    )
    payload["metadata"] = dict(payload.get("metadata") or {})
    payload["metadata"]["season_year"] = season_year
    payload["metadata"]["result_id"] = result_id
    # Ceremony-sized client payload — full ballots caused ~15MB Network Errors on Cup night.
    session.awards_payload = slim_awards_payload_for_client(payload)
    session.awards_generated = True
    try:
        apply_career_award_history(teams, awards, season_year, result_id=result_id)
    except Exception:
        pass

    ch_name = session.team_by_id.get(champion)
    ch_disp = _display_team(ch_name) if ch_name else champion
    session.notifications.append(f"Playoffs complete. Stanley Cup: {ch_disp}")
    session.timeline.append(f"POSTSEASON: Champion {ch_disp}")

    payload_po = dict(getattr(session, "playoff_payload", None) or {})
    payload_po["champion_id"] = champion
    payload_po["finalist_ids"] = finalists
    payload_po["series_list"] = list(live.get("series") or [])
    payload_po["completed"] = True
    payload_po["live"] = False
    session.playoff_payload = payload_po
    if isinstance(getattr(session, "playoff_live", None), dict):
        session.playoff_live["completed"] = True
        session.playoff_live["champion_id"] = champion

    _enqueue_offseason_popup(session, "awards", "Awards Night", "Season hardware awaits")
    invalidate_session_payload_caches(session, "post_cup")
    return {
        "status": "post_cup",
        "season_phase": "post_cup",
        "champion_id": champion,
        "awards_keys": list((session.awards_payload.get("awards") or {}).keys()),
    }


def complete_playoffs(session: FranchiseSession) -> Dict[str, Any]:
    """Simulate playoffs only — transition to post_cup, do not end franchise."""
    # Prefer finishing interactive live playoffs if already started.
    live = getattr(session, "playoff_live", None)
    if isinstance(live, dict) and live.get("started") and not session.playoffs_simulated:
        try:
            from services.franchise_playoffs import handle_playoff_action

            res = handle_playoff_action(session, "sim_rest")
            if isinstance(res.get("finish"), dict):
                return res["finish"]
            if session.playoffs_simulated:
                return {
                    "status": "post_cup",
                    "season_phase": "post_cup",
                    "champion_id": session.champion_id,
                }
        except Exception:
            pass

    from app.sim_engine.league import compute_awards, simulate_playoffs
    from app.sim_engine.league.awards import apply_career_award_history, build_awards_payload
    from services.franchise_sim import _display_team
    from services.franchise_sim import invalidate_session_payload_caches

    if session.playoffs_simulated and str(getattr(session, "phase", "")) in ("post_cup", "offseason", "preseason", "complete"):
        return {
            "status": "post_cup",
            "season_phase": str(getattr(session, "phase", "post_cup")),
            "champion_id": session.champion_id,
            "already_done": True,
        }

    # Idempotent: if awards already frozen for this season, do not recompute/duplicate history.
    existing = dict(getattr(session, "awards_payload", None) or {})
    existing_meta = dict(existing.get("metadata") or {})
    season_year = int(getattr(session, "season_calendar_year", 0) or 0)
    if existing.get("awards") and str(existing_meta.get("season_year") or "") == str(season_year) and existing_meta.get("computed_at_stage"):
        session.playoffs_simulated = True
        session.playoffs_done = True
        session.phase = "post_cup"
        session.season_phase = "post_cup"
        return {
            "status": "post_cup",
            "season_phase": "post_cup",
            "champion_id": session.champion_id,
            "already_done": True,
            "awards_keys": list((existing.get("awards") or {}).keys()),
        }

    session.phase = "playoffs"
    session.season_phase = "playoffs"
    _playoff_lifecycle_log(session, "season_phase_updated", season_phase="playoffs")

    sim = session.sim
    teams = list(sim.league.teams)
    playoff_result = simulate_playoffs(sim.rng, session.standings, teams, session.strength_map)

    # Freeze input snapshots before any offseason mutation.
    award_rows = [dict(v) for v in (list((session.player_season_stats or {}).values()) if session.player_season_stats else [])]
    for row in award_rows:
        row.setdefault("stat_scope", "regular_season")
    playoff_rows = []
    for key in ("playoff_player_stats", "player_playoff_stats", "playoff_stats"):
        raw = getattr(session, key, None)
        if isinstance(raw, dict):
            playoff_rows = [dict(v) for v in raw.values()]
            break
        if isinstance(raw, list):
            playoff_rows = [dict(v) for v in raw if isinstance(v, dict)]
            break
    for row in playoff_rows:
        row["stat_scope"] = "playoffs"

    season_seed = getattr(session, "franchise_seed", None)
    if season_seed is None:
        rng = getattr(sim, "rng", None)
        raw_seed = getattr(rng, "_seed", None) if rng is not None else None
        # Never use rng.seed — that is Random.seed (a method), not an int.
        if isinstance(raw_seed, (int, float)) and not isinstance(raw_seed, bool):
            season_seed = int(raw_seed) & 0xFFFFFFFF
        else:
            season_seed = int(season_year) & 0xFFFFFFFF
    elif not isinstance(season_seed, (int, float)) or isinstance(season_seed, bool):
        season_seed = int(season_year) & 0xFFFFFFFF
    else:
        season_seed = int(season_seed) & 0xFFFFFFFF

    history_by_player = dict(getattr(session, "player_award_history", None) or {})
    awards = compute_awards(
        session.standings,
        playoff_result,
        teams,
        player_season_stats=award_rows,
        playoff_player_stats=playoff_rows,
        season_seed=season_seed,
        season_year=season_year,
        season_length=int(getattr(session, "season_length", 82) or 82),
        history_by_player=history_by_player,
    )

    session.playoffs_simulated = True
    session.playoffs_done = True
    session.phase = "post_cup"
    session.season_phase = "post_cup"
    session.champion_id = str(getattr(playoff_result, "champion_id", "") or "") if playoff_result else ""
    session.stanley_cup_winner = session.champion_id

    result_id = f"awards:{season_year}:{season_seed}"
    payload = build_awards_payload(awards, season=season_year, season_seed=season_seed, season_length=int(getattr(session, "season_length", 82) or 82))
    payload["metadata"] = dict(payload.get("metadata") or {})
    payload["metadata"]["season_year"] = season_year
    payload["metadata"]["result_id"] = result_id
    payload["frozen_inputs"] = {
        "regular_season_player_stats": award_rows,
        "playoff_player_stats": playoff_rows,
        "standings_snapshot": [
            {
                "team_id": str(getattr(r, "team_id", "")),
                "points": int(getattr(r, "points", 0) or 0),
                "wins": int(getattr(r, "wins", 0) or 0),
                "losses": int(getattr(r, "losses", 0) or 0),
                "otl": int(getattr(r, "otl", 0) or 0),
                "gf": int(getattr(r, "gf", 0) or 0),
                "ga": int(getattr(r, "ga", 0) or 0),
            }
            for r in list(session.standings.league_table() or [])
        ],
    }
    # Drop frozen_inputs + ballot bloat before storing — state responses were ~15MB.
    session.awards_payload = slim_awards_payload_for_client(payload)
    session.awards_generated = True

    try:
        apply_career_award_history(teams, awards, season_year, result_id=result_id)
    except Exception:
        pass

    ch_name = session.team_by_id.get(session.champion_id)
    ch_disp = _display_team(ch_name) if ch_name else session.champion_id
    session.notifications.append(f"Playoffs complete. Stanley Cup: {ch_disp}")
    session.timeline.append(f"POSTSEASON: Champion {ch_disp}")
    _playoff_lifecycle_log(session, "stanley_cup_winner", champion=session.champion_id)

    payload_po = dict(getattr(session, "playoff_payload", None) or {})
    if playoff_result is not None:
        payload_po["champion_id"] = session.champion_id
        payload_po["finalist_ids"] = list(getattr(playoff_result, "finalist_ids", None) or [])
        payload_po["series_list"] = [
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
        # Conference champions from awards payload (team achievements).
        conf = next((a for a in (payload.get("team_achievements") or []) if str(a.get("award_id")) == "conference_champions"), None)
        if conf:
            payload_po["conference_champions"] = list(conf.get("winners") or conf.get("full_results") or [])
    session.playoff_payload = payload_po
    session.next_important_event = "awards"
    _enqueue_offseason_popup(session, "awards", "Awards Night", "Season hardware awaits")
    invalidate_session_payload_caches(session, "post_cup")

    return {
        "status": "post_cup",
        "season_phase": "post_cup",
        "champion_id": session.champion_id,
        "awards_keys": list((session.awards_payload.get("awards") or {}).keys()),
    }


def advance_season_phase(session: FranchiseSession, target: Optional[str] = None) -> Dict[str, Any]:
    """Single source-of-truth season phase controller."""
    from services.franchise_sim import _regular_season_is_truly_complete

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
            session.next_season_generated = False
            session.next_season_payload = {}
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
    if stage == "draft_combine":
        return bool(getattr(session, "draft_combine_done", False))
    if stage == "draft":
        state = getattr(session, "draft_state", None) or {}
        return bool(session.draft_payload) or bool(state.get("draft_started"))
    def _versioned(payload: Any, key: str) -> bool:
        return isinstance(payload, dict) and int(payload.get("version") or 0) >= STAGE_PAYLOAD_VERSION[key]

    if stage == "draft_review":
        p = getattr(session, "draft_review_payload", None)
        return _versioned(p, "draft_review") and p.get("user_picks") is not None
    if stage == "prospect_rights":
        return _versioned(getattr(session, "prospect_rights_payload", None), "prospect_rights")
    if stage == "re_sign":
        return _versioned(session.resign_payload, "re_sign")
    if stage == "free_agency":
        payload = getattr(session, "free_agency_market_payload", None)
        if not _versioned(payload, "free_agency"):
            return False
        # Stale empty boards (version stamped, 0 agents) must rehydrate so
        # overseas / July 1 pools are never locked out of the Wire.
        fa_rows = payload.get("free_agents") if isinstance(payload, dict) else None
        count = int(payload.get("available_count") or 0) if isinstance(payload, dict) else 0
        if count <= 0 and not (isinstance(fa_rows, list) and len(fa_rows) > 0):
            return False
        return True
    if stage == "roster_cleanup":
        return _versioned(session.roster_cleanup_payload, "roster_cleanup")
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
        "draft_combine": _run_draft_combine,
        "draft": _prepare_draft_payload,
        "draft_review": _run_draft_review,
        "prospect_rights": _run_prospect_rights_stage,
        "re_sign": _prepare_resign_payload,
        "free_agency": _open_free_agency,
        "roster_cleanup": _run_roster_cleanup,
        "next_season_reveal": _finalize_next_season_reveal,
    }
    fn = handlers.get(stage)
    if fn is None:
        raise ValueError(f"Unknown offseason stage: {stage!r}")
    return fn(session)


def continue_offseason(
    session: FranchiseSession,
    *,
    from_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Advance one offseason UI stage.

    ``from_stage`` is the stage the client currently shows. When a prior continue
    advanced the server (e.g. to retirements) but the response failed to serialize,
    the client may still be on awards — we re-deliver retirements instead of
    skipping ahead to salary_cap.
    """
    from services.franchise_sim import invalidate_session_payload_caches

    _sync_phase_fields(session)
    client_stage = str(from_stage or "").strip().lower()

    # Playoff hub "Continue to Awards" used to call this while phase was still
    # "playoffs" (Cup decided in live state, finish/awards not yet committed).
    if str(session.phase) in ("playoffs", "playoff_ready"):
        live = getattr(session, "playoff_live", None)
        try:
            from services.franchise_playoffs import handle_playoff_action, finish_live_playoffs

            if isinstance(live, dict) and live.get("started"):
                if live.get("completed") or live.get("champion_id"):
                    finish_live_playoffs(session)
                else:
                    handle_playoff_action(session, "sim_rest")
            else:
                complete_playoffs(session)
        except Exception:
            complete_playoffs(session)
        _sync_phase_fields(session)

    # Awards Night is shown while phase is still post_cup. Continue from that screen
    # must move into offseason and advance past awards → retirements.
    # Only park on awards if awards data is still missing.
    if str(session.phase) == "post_cup":
        session.phase = "offseason"
        session.season_phase = "offseason"
        session.offseason_stage = "awards"
        session.next_important_event = "awards"
        result = _enter_awards_stage(session)
        if not _offseason_stage_ready(session, "awards"):
            invalidate_session_payload_caches(session, "offseason_awards")
            return {
                **result,
                "status": "offseason",
                "season_phase": "offseason",
                "offseason_stage": "awards",
                "next_important_event": "awards",
            }
        # Awards already populated from Cup finish — fall through to next stage.

    if str(session.phase) != "offseason":
        raise ValueError(f"Cannot continue offseason from phase {session.phase!r}")

    current = str(getattr(session, "offseason_stage", "") or "awards")
    if current not in OFFSEASON_STAGES:
        current = "awards"
        session.offseason_stage = current

    # Client still on Awards / post_cup, but server already processed Final Skate.
    # Do not treat a missing from_stage as awards — that deadlocks automated continue.
    if (
        client_stage in ("awards", "post_cup")
        and current == "retirements"
        and bool(getattr(session, "retirements_processed", False))
        and isinstance(getattr(session, "retirements_payload", None), dict)
    ):
        session.next_important_event = STAGE_NEXT_EVENT.get("retirements", "retirements")
        invalidate_session_payload_caches(session, "offseason_retirements")
        return {
            "retirements": session.retirements_payload,
            "status": "offseason",
            "season_phase": "offseason",
            "offseason_stage": "retirements",
            "next_important_event": session.next_important_event,
            "replayed": True,
        }

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

    if current == "draft_combine" and not getattr(session, "draft_combine_done", False):
        raise ValueError("Complete the Draft Combine before continuing offseason")

    if current == "draft" and not session.draft_completed:
        raise ValueError("Complete the Entry Draft before continuing offseason")

    if current == "re_sign":
        # Exclusive window is optional negotiating time — opening FA ends it.
        resign = session.resign_payload if isinstance(session.resign_payload, dict) else {}
        if resign.get("can_continue") is False and list(resign.get("blocking_decisions") or []):
            raise ValueError(
                list(resign.get("blocking_reasons") or ["Resolve required RFA decisions before Free Agency"])[0]
            )
        # Leaving the re-sign desk — archive outcomes so Free Agency starts clean.
        clear_resign_phase_state(session)

    if idx + 1 < len(OFFSEASON_STAGES):
        next_stage = OFFSEASON_STAGES[idx + 1]
        _mark_stage_completed(session, current)
        session.offseason_stage = next_stage
        _mark_stage_entered(session, next_stage)
        result = _stage_handler(session, next_stage)
    else:
        _mark_stage_completed(session, current)
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


def build_free_agency_desk(session: FranchiseSession, *, open_market: bool = False) -> Dict[str, Any]:
    """Build the Free Agency Wire payload for any season phase.

    Does not change ``offseason_stage``. Used by the Hub / standalone Free Agency
    screen so regular-season and playoff access share the offseason Wire UI.
    """
    from services.contract_economy import build_contract_office, sync_all_team_cap_fields
    from services.fa_market_engine import (
        annotate_fa_rows_with_decisions,
        ensure_fa_market_book,
    )

    phase = str(getattr(session, "phase", "") or "").lower()
    stage = str(getattr(session, "offseason_stage", "") or "")
    should_open = (
        bool(open_market)
        or bool(getattr(session, "free_agency_open", False))
        or (phase == "offseason" and stage == "free_agency")
    )
    if should_open and phase == "offseason":
        # When already in the July FA window / Free Agency stage, keep the full
        # open-market path so the Wire is never stuck on exclusive-only emptiness.
        return _open_free_agency(session, force=bool(open_market) or stage == "free_agency")

    try:
        league = getattr(getattr(session, "sim", None), "league", None)
        sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
        if league is not None:
            from app.sim_engine.league_hierarchy_bootstrap import ensure_overseas_fa_pool

            rng = getattr(getattr(session, "sim", None), "rng", None)
            if rng is not None:
                ensure_overseas_fa_pool(league, rng, min_count=120)
            sync_all_team_cap_fields(league, getattr(session, "sim", None), season_year=sy)
    except Exception:
        pass

    ensure_fa_market_book(session)
    office = build_contract_office(session)
    fa_list = annotate_fa_rows_with_decisions(
        session, list(office.get("free_agents") or office.get("freeAgents") or [])
    )
    overseas_list = annotate_fa_rows_with_decisions(
        session, list(office.get("overseas_free_agents") or [])
    )
    session.free_agents_payload = fa_list
    bonus = team_signing_bonus_eligibility(session)
    cap = office.get("cap_snapshot") or {}
    needs = (office.get("team") or {}).get("needs") or {}
    summary = office.get("summary") or {}
    top = sorted(fa_list, key=lambda r: -float(r.get("ovr") or r.get("overall") or 0))[:24]
    cpu = getattr(session, "cpu_fa_signings", None) or {}
    recent = list(cpu.get("signings") or [])[-12:]
    book = getattr(session, "fa_market_book", None) or {}
    book_log = list(book.get("log") or [])[-20:]
    news = []
    for s in recent[-8:]:
        club = s.get("team_name") or s.get("team_abbrev") or "A club"
        news.append({
            "kind": "signing",
            "text": (
                f"{club} signs "
                f"{s.get('name') or s.get('player_id') or 'a free agent'} · "
                f"{s.get('aav_m')}M × {s.get('years')}y"
            ),
        })
    for entry in book_log[-8:]:
        if isinstance(entry, dict) and entry.get("text"):
            news.append({"kind": entry.get("kind") or "market", "text": entry.get("text")})
        elif isinstance(entry, str):
            news.append({"kind": "market", "text": entry})

    day = int(getattr(session, "fa_market_day", 0) or book.get("day") or 0)
    awaiting_july1 = (
        not bool(session.free_agency_open)
        and phase == "offseason"
        and stage in ("re_sign", "salary_cap", "draft", "draft_combine", "awards", "retirements")
    )
    market = {
        "version": STAGE_PAYLOAD_VERSION["free_agency"],
        "market_status": "awaiting_open" if awaiting_july1 else ("open" if session.free_agency_open else "in_season"),
        "wave": int(getattr(session, "cpu_fa_wave", 0) or 0),
        "fa_market_day": day,
        "market_phase": "awaiting_open" if awaiting_july1 else ("in_season" if not session.free_agency_open else "open_market"),
        "market_phase_label": (
            "Open Free Agency from Re-Sign to populate the July 1 board"
            if awaiting_july1
            else (
                "Opening Day"
                if session.free_agency_open and day <= 1
                else ("Open market" if session.free_agency_open else "Free Agency Board")
            )
        ),
        "empty_reason": (
            "July 1 free agents are still under exclusive negotiating / pending expiry. "
            "Use Open Free Agency on the Re-Sign desk to open the market."
            if awaiting_july1 and not fa_list
            else None
        ),
        "available_count": len(fa_list),
        "major_available": top,
        "free_agents": fa_list,
        "overseas_free_agents": overseas_list,
        "market_news": news[-16:],
        "cap_space_m": float(cap.get("usable_cap_space_m") or cap.get("cap_space_m") or 0),
        "cap_snapshot": cap,
        "contract_slots": office.get("contract_slots") or {},
        "needs": needs,
        "pending_rfa_count": summary.get("rfaCount") or 0,
        "signing_bonus": bonus,
        "recent_league_signings": recent,
        "cpu_signings_count": len(list(cpu.get("signings") or [])),
        "stage_status": "ready",
        "can_continue": True,
        "standalone": True,
        "available_actions": [
            "advance_fa_day",
            "open_cap_ledger_fa",
            "back_to_hub",
        ],
    }
    session.free_agency_market_payload = market
    return {
        "ok": True,
        "free_agents": fa_list,
        "free_agency_market": market,
    }


def reopen_offseason_stage(session: FranchiseSession, stage: str) -> Dict[str, Any]:
    """Step back to an earlier offseason desk (Roster Check → Free Agency).

    Used when Roster Check is blocked and the GM needs to sign free agents
    without leaving the offseason timeline.
    """
    from services.franchise_sim import invalidate_session_payload_caches

    _sync_phase_fields(session)
    if str(session.phase) != "offseason":
        raise ValueError(f"Cannot reopen offseason stage from phase {session.phase!r}")

    target = str(stage or "").strip().lower()
    current = str(getattr(session, "offseason_stage", "") or "")
    if target not in OFFSEASON_STAGES:
        raise ValueError(f"Unknown offseason stage {target!r}")

    allowed = {
        ("roster_cleanup", "free_agency"),
        ("roster_cleanup", "re_sign"),
        ("next_season_reveal", "roster_cleanup"),
        ("next_season_reveal", "free_agency"),
    }
    # Also allow reopening free_agency while already there (idempotent refresh).
    if (current, target) not in allowed and not (current == target == "free_agency"):
        raise ValueError(f"Cannot reopen {target!r} from {current!r}")

    completed = [
        s
        for s in list(getattr(session, "offseason_completed_stages", None) or [])
        if s not in OFFSEASON_STAGES[OFFSEASON_STAGES.index(target) :]
    ]
    session.offseason_completed_stages = completed
    session.offseason_stage = target
    session.next_important_event = STAGE_NEXT_EVENT.get(target, target)
    _mark_stage_entered(session, target)

    if target == "free_agency":
        session.free_agency_open = True
        # Force rebuild so stale empty boards (version stamped, 0 agents) refill.
        existing = getattr(session, "free_agency_market_payload", None)
        empty = (
            not isinstance(existing, dict)
            or int(existing.get("available_count") or 0) <= 0
            or not list(existing.get("free_agents") or [])
        )
        result = _open_free_agency(session, force=bool(empty))
        market = result.get("free_agency_market") or getattr(session, "free_agency_market_payload", None) or {}
        invalidate_session_payload_caches(session, "offseason_reopen_free_agency")
        return {
            "ok": True,
            "status": "offseason",
            "season_phase": "offseason",
            "offseason_stage": "free_agency",
            "next_important_event": session.next_important_event,
            "free_agency_market": market,
            "reopened_from": current,
        }

    if target == "re_sign":
        result = _prepare_resign_payload(session, force=True)
        invalidate_session_payload_caches(session, "offseason_reopen_re_sign")
        return {
            "ok": True,
            "status": "offseason",
            "season_phase": "offseason",
            "offseason_stage": "re_sign",
            "next_important_event": session.next_important_event,
            "re_sign": result.get("re_sign") or result.get("contracts"),
            "reopened_from": current,
        }

    result = _run_roster_cleanup(session, force=False)
    invalidate_session_payload_caches(session, "offseason_reopen_roster_cleanup")
    return {
        "ok": True,
        "status": "offseason",
        "season_phase": "offseason",
        "offseason_stage": target,
        "next_important_event": session.next_important_event,
        **result,
        "reopened_from": current,
    }


def _enter_awards_stage(session: FranchiseSession) -> Dict[str, Any]:
    if not session.awards_generated:
        session.awards_payload = session.awards_payload or {"awards": {}, "items": []}
    return {"awards": slim_awards_payload_for_client(session.awards_payload)}


def _process_retirements(session: FranchiseSession) -> Dict[str, Any]:
    from services.franchise_retirement import run_franchise_retirement_pass
    from services.json_safe import json_safe

    payload = run_franchise_retirement_pass(session)
    session.retirements_payload = json_safe(payload)
    return {"retirements": session.retirements_payload}


def _tick_league_contracts(session: FranchiseSession) -> Dict[str, Any]:
    from services.contract_economy import handle_player_contract_expiry
    from services.franchise_sim import _serialize_player_row

    # One tick per offseason salary-cap stage. Re-entry must not decrement twice.
    if bool(getattr(session, "contracts_ticked", False)):
        return {
            "expired_ufas": [],
            "expired_rfas": [],
            "skipped": True,
            "reason": "contracts_already_ticked",
        }

    sim = session.sim
    league = getattr(sim, "league", None)
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)

    # Evaluate ELC slides before year burn / expiry.
    slide_result = {}
    try:
        from services.elc_offer_engine import process_elc_slides

        slide_result = process_elc_slides(session, season_year)
    except Exception:
        slide_result = {}

    teams = list(getattr(league, "teams", None) or [])
    expired_ufas: List[Dict[str, Any]] = []
    expired_rfas: List[Dict[str, Any]] = []

    # Affiliate SPCs count against the 50-contract limit, so their years must burn
    # on the same tick as the NHL list or they never expire.
    for team in teams:
        for attr in ("roster", "ahl_roster", "echl_roster"):
            roster = list(getattr(team, attr, None) or [])
            if not roster:
                continue
            kept = []
            for p in roster:
                if getattr(p, "retired", False):
                    continue
                outcome = handle_player_contract_expiry(
                    p, team, league, season_year, defer_july1_ufa=True
                )
                if outcome != "kept":
                    c = getattr(p, "contract", None)
                    rights = str(getattr(c, "rights_status", getattr(p, "rights_status", "UFA")) or "UFA").upper()
                    row = _serialize_player_row(p, include_ratings=True, session=session, _team=team)
                    if outcome == "rfa_rights" or "RFA" in rights:
                        expired_rfas.append(row)
                    else:
                        expired_ufas.append(row)
                    continue
                kept.append(p)
            setattr(team, attr, kept)

    # Refresh every club's cached cap mirrors (team.cap_space / total_cap_hit /
    # cap_snapshot) now that rosters changed — not just the user's team — so
    # any downstream read of those mirrored fields reflects freed-up space
    # immediately instead of a stale pre-expiry snapshot.
    try:
        from services.contract_economy import sync_team_cap_fields

        for team in teams:
            try:
                sync_team_cap_fields(team, league, sim, season_year=season_year)
            except Exception:
                continue
    except Exception:
        pass

    session.contracts_ticked = True
    return {
        "expired_ufas": expired_ufas,
        "expired_rfas": expired_rfas,
        "elc_slides": slide_result,
    }


def _cap_status_label(cap_space_m: float) -> str:
    space = float(cap_space_m or 0)
    if space < 0:
        return "Over cap"
    if space <= 1.0:
        return "Tight"
    if space <= 4.0:
        return "Deadline limited"
    if space >= 10.0:
        return "Offseason flexible"
    return "Healthy"


def _build_cap_report(
    session: FranchiseSession,
    *,
    cap_row: Dict[str, Any],
    user_team: Any,
    user_snap: Dict[str, Any],
    tick: Dict[str, Any],
    over_cap_teams: List[Dict[str, Any]],
    season_year: int,
) -> Dict[str, Any]:
    from services.franchise_sim import _display_team

    prev_cap = float(cap_row.get("previous_cap", 0) or 0)
    new_cap = float(cap_row.get("upperLimit", cap_row.get("current_cap", 0)) or 0)
    if prev_cap <= 0:
        prev_cap = max(0.0, new_cap - float(cap_row.get("change", 0) or 0))
    change = float(cap_row.get("cap_change", cap_row.get("change", new_cap - prev_cap)) or 0)
    pct = float(cap_row.get("cap_change_percent", 0) or 0)
    if prev_cap > 0 and pct == 0 and abs(change) > 1e-6:
        pct = round((change / prev_cap) * 100.0, 1)

    payroll = float(user_snap.get("totalCapHit", user_snap.get("activeRosterCapHit", 0)) or 0)
    cap_space = float(user_snap.get("capSpace", user_snap.get("usableCapSpace", 0)) or 0)
    dead_cap = round(
        float(user_snap.get("buriedCapHit", 0) or 0)
        + float(user_snap.get("buyoutCapHit", 0) or 0)
        + float(user_snap.get("otherDeadCap", 0) or 0),
        3,
    )
    retained = float(user_snap.get("retainedSalary", 0) or 0)
    bonus = float(user_snap.get("bonusOverage", 0) or 0)
    projected = float(user_snap.get("projectedDeadlineSpace", cap_space) or cap_space)

    notes: List[str] = []
    movement_label = str(cap_row.get("movement_label") or "").strip()
    if movement_label:
        notes.append(movement_label)
    if over_cap_teams:
        notes.append(f"{len(over_cap_teams)} teams over cap")
    expired_ufas = list(tick.get("expired_ufas") or [])
    if expired_ufas:
        notes.append(f"{len(expired_ufas)} UFAs available")

    sy = int(season_year)
    season_label = cap_row.get("season") or f"{sy + 1}-{(sy + 2) % 100:02d}"

    return {
        "season": season_label,
        "previous_cap": round(prev_cap, 3),
        "current_cap": round(new_cap, 3),
        "cap_change": round(change, 3),
        "cap_change_percent": pct,
        "movement_type": cap_row.get("movement_type", ""),
        "movement_label": movement_label,
        "movement_reason": str(cap_row.get("movement_reason") or "").strip(),
        "notes": notes[:4],
        "user_team": {
            "team_id": str(session.user_team_id or ""),
            "team_name": _display_team(user_team) if user_team is not None else "",
            "payroll": round(payroll, 3),
            "cap_space": round(cap_space, 3),
            "dead_cap": dead_cap,
            "retained_salary": round(retained, 3),
            "bonus_overages": round(bonus, 3),
            "projected_space": round(projected, 3),
            "cap_status": _cap_status_label(cap_space),
        },
    }


def _advance_salary_cap(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.economy.cap_engine import (
        advance_league_salary_cap,
        calculate_team_cap_snapshot,
        cleanup_league_retained_salary_records,
    )
    from services.franchise_sim import _team_cap_snapshot

    tick = _tick_league_contracts(session)
    sim = session.sim
    league = getattr(sim, "league", None)
    sy = int(session.season_calendar_year)
    if league is not None:
        try:
            cleanup_league_retained_salary_records(league)
        except Exception:
            pass
    cap_row: Dict[str, Any] = {}
    try:
        cap_row = advance_league_salary_cap(league, sim.rng, season_year=sy + 1)
    except Exception:
        fallback_cap = float(getattr(league, "salary_cap_m", 88.0) or 88.0)
        cap_row = {
            "previous_cap": fallback_cap,
            "upperLimit": fallback_cap,
            "current_cap": fallback_cap,
            "change": 0.0,
            "cap_change": 0.0,
            "cap_change_percent": 0.0,
            "movement_type": "flat_cap",
            "movement_label": "Flat cap year",
            "movement_reason": "League held the cap steady.",
        }

    user_team = session.team_by_id.get(session.user_team_id)
    user_snap = calculate_team_cap_snapshot(user_team, league) if user_team and league else {}
    user_cap = _team_cap_snapshot(user_team, sim) if user_team else {}
    over_cap_teams: List[Dict[str, Any]] = []
    for tid, tm in (session.team_by_id or {}).items():
        snap = calculate_team_cap_snapshot(tm, league) if tm and league else {}
        space = float(snap.get("cap_space", snap.get("capSpace", 0)) or 0)
        if space < 0:
            over_cap_teams.append({"team_id": tid, "cap_space": space, "cap_hit": snap.get("cap_hit", snap.get("totalCapHit", 0))})

    prev_cap = float(cap_row.get("previous_cap", 0) or 0)
    new_cap = float(cap_row.get("upperLimit", cap_row.get("current_cap", getattr(league, "salary_cap_m", 0))) or 0)
    if prev_cap <= 0:
        prev_cap = max(0.0, new_cap - float(cap_row.get("change", 0) or 0))
    change = float(cap_row.get("cap_change", cap_row.get("change", new_cap - prev_cap)) or 0)

    cap_report = _build_cap_report(
        session,
        cap_row=cap_row,
        user_team=user_team,
        user_snap=user_snap,
        tick=tick,
        over_cap_teams=over_cap_teams,
        season_year=sy,
    )

    payload = {
        "last_season_cap": prev_cap,
        "previous_cap": prev_cap,
        "new_season_cap": new_cap,
        "current_cap": new_cap,
        "change": change,
        "cap_change": change,
        "cap_change_percent": cap_report.get("cap_change_percent", 0),
        "movement_type": cap_report.get("movement_type", ""),
        "movement_label": cap_report.get("movement_label", ""),
        "movement_reason": cap_report.get("movement_reason", ""),
        "cap_report": cap_report,
        "user_team_cap": user_cap,
        "over_cap_teams": over_cap_teams,
        "expired_ufas": tick.get("expired_ufas", []),
        "expired_rfas": tick.get("expired_rfas", []),
        "notes": cap_report.get("notes", []),
    }
    session.salary_cap_payload = payload
    session.timeline.append(f"OFFSEASON: Salary cap set to ${new_cap:.1f}M.")
    return {"salary_cap": payload}


# ---------------------------------------------------------------------------
# Organizational Development Review — normalized report builders
# ---------------------------------------------------------------------------

_DEV_TREND_BREAKOUT = "Breakout"
_DEV_TREND_IMPROVED = "Improved"
_DEV_TREND_STABLE = "Stable"
_DEV_TREND_STALLED = "Stalled"
_DEV_TREND_REGRESSED = "Regressed"

_DEV_READINESS_NHL_READY = "NHL Ready"
_DEV_READINESS_CLOSE = "Close"
_DEV_READINESS_DEVELOPING = "Developing"
_DEV_READINESS_LONG_TERM = "Long-Term"
_DEV_READINESS_AT_RISK = "At Risk"

_DEV_ATTR_LABELS = {
    "skating": "Skating",
    "speed": "Speed",
    "acceleration": "Acceleration",
    "agility": "Agility",
    "balance": "Balance",
    "strength": "Strength",
    "endurance": "Endurance",
    "durability": "Durability",
    "offensive_awareness": "Offensive awareness",
    "defensive_awareness": "Defensive awareness",
    "passing": "Passing",
    "puck_control": "Puck control",
    "deking": "Deking",
    "shot_power": "Shot power",
    "shot_accuracy": "Shot accuracy",
    "wrist_shot": "Wrist shot",
    "slap_shot": "Slap shot",
    "faceoffs": "Faceoffs",
    "positioning": "Positioning",
    "rebounds": "Rebound control",
    "glove_high": "Glove high",
    "glove_low": "Glove low",
    "five_hole": "Five hole",
    "stick_handling": "Stick handling",
    "poke_check": "Poke check",
}

_DEV_ATTR_PREFIXES = (
    "off_",
    "pm_",
    "def_",
    "phy_",
    "skg_",
    "iqm_",
    "pc_",
    "dev_",
    "per_",
    "st_",
    "g_",
)

_DEV_META_ATTR_KEYS = frozenset(
    {
        "dev_potential",
        "dev_ceiling",
        "dev_growth_rate",
        "dev_work_ethic",
        "dev_coachability",
        "dev_learning_ability",
        "potential",
        "overall",
        "ovr",
        "true_potential",
        "true_ceiling",
        "_generated_profile",
        "generated_profile",
    }
)


def _dev_player_id(player: Any) -> str:
    return str(getattr(player, "player_id", getattr(player, "id", "")) or "").strip()


def _dev_player_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    return str(getattr(ident, "name", None) or getattr(player, "name", "") or "Unknown")


def _dev_player_position(player: Any) -> str:
    ident = getattr(player, "identity", None)
    pos = getattr(ident, "position", None) if ident else None
    if pos is not None and hasattr(pos, "value"):
        return str(pos.value).upper()
    raw = str(pos or getattr(player, "position", "") or "F").upper()
    return raw or "F"


def _dev_player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    try:
        return int(getattr(ident, "age", None) or getattr(player, "age", 0) or 0)
    except Exception:
        return 0


def _dev_is_goalie(player: Any) -> bool:
    return _dev_player_position(player) == "G"


def _dev_is_defense(player: Any) -> bool:
    pos = _dev_player_position(player)
    return pos in ("D", "LD", "RD", "DEF", "DEFENSE")


def _dev_is_signed(player: Any) -> bool:
    return str(getattr(player, "signed_status", "unsigned") or "unsigned").lower() == "signed"


def _dev_contract_status(player: Any) -> str:
    signed = _dev_is_signed(player)
    if signed:
        return str(getattr(player, "contract_type", "") or getattr(getattr(player, "contract", None), "contract_type", "") or "signed")
    if bool(getattr(player, "drafted", False)) or getattr(player, "nhl_rights_team_id", None):
        return "unsigned_rights"
    return "unsigned"


def _dev_extract_attributes(player: Any) -> Dict[str, float]:
    ratings = getattr(player, "ratings", None)
    if not isinstance(ratings, dict):
        return {}
    out: Dict[str, float] = {}
    for key, val in ratings.items():
        try:
            fv = float(val)
        except Exception:
            continue
        if fv <= 1.5:
            fv = round(fv * 99.0, 2)
        out[str(key)] = round(fv, 2)
    return out


def _dev_is_skill_attr(key: str) -> bool:
    return str(key or "").lower() not in _DEV_META_ATTR_KEYS and not str(key or "").startswith("_")


def _dev_attr_deltas(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for key in set(before) | set(after):
        if not _dev_is_skill_attr(key):
            continue
        diff = round(float(after.get(key, before.get(key, 0))) - float(before.get(key, 0)), 2)
        if abs(diff) >= 0.01:
            deltas[key] = diff
    return deltas


def _dev_attr_label(key: str) -> str:
    k = str(key or "").lower()
    if k in _DEV_ATTR_LABELS:
        return _DEV_ATTR_LABELS[k]
    for prefix in _DEV_ATTR_PREFIXES:
        if k.startswith(prefix):
            k = k[len(prefix) :]
            break
    return k.replace("_", " ").strip().title()


def _dev_ledger_attr_deltas(ledger: Dict[str, Any]) -> Dict[str, float]:
    """Return display-scale attribute deltas from the seasonal ledger (never rescale)."""
    out: Dict[str, float] = {}
    raw = ledger.get("attribute_deltas") if isinstance(ledger, dict) else None
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if not _dev_is_skill_attr(k):
            continue
        try:
            fv = round(float(v), 2)
        except Exception:
            continue
        if abs(fv) < 0.01:
            continue
        out[str(k)] = fv
    return out


def _dev_ovr_from_ledger(ledger: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Read ovr_before/ovr_after from ledger (stored 0–1) as display-scale OVR."""
    if not isinstance(ledger, dict) or not ledger.get("development_applied"):
        return None, None
    try:
        before = ledger.get("ovr_before")
        after = ledger.get("ovr_after")
        if before is None or after is None:
            return None, None
        prev = int(round(float(before) * 99.0))
        cur = int(round(float(after) * 99.0))
        if prev <= 0 and cur <= 0:
            return None, None
        return float(prev), float(cur)
    except Exception:
        return None, None


def _dev_snapshot_player(
    player: Any,
    *,
    team_id: str = "",
    league_id: str = "",
) -> Dict[str, Any]:
    from services.franchise_sim import _player_ovr99

    return {
        "overall": round(_player_ovr99(player), 2),
        "attributes": _dev_extract_attributes(player),
        "team_id": str(team_id or getattr(player, "team_id", "") or ""),
        "league_id": str(
            league_id
            or getattr(player, "current_league_id", "")
            or getattr(player, "development_path", "")
            or ""
        ),
        "age": _dev_player_age(player),
        "position": _dev_player_position(player),
        "potential": _dev_player_potential_display(player),
        "contract_status": _dev_contract_status(player),
        "signed": _dev_is_signed(player),
        "readiness_score": _dev_readiness_score_raw(player),
        "readiness_tier": "",
    }


def _dev_player_potential_display(player: Any) -> float:
    try:
        from app.sim_engine.entities.player import display_rating, normalize_rating

        pot = getattr(player, "potential", None)
        if pot is not None:
            return float(display_rating(normalize_rating(pot)))
        ratings = getattr(player, "ratings", None) or {}
        if ratings.get("dev_potential") is not None:
            return float(display_rating(normalize_rating(ratings["dev_potential"])))
    except Exception:
        pass
    try:
        from services.franchise_sim import _player_ovr99

        return float(int(round(_player_ovr99(player) + 4.0)))
    except Exception:
        return 0.0


def _dev_readiness_score_raw(player: Any) -> float:
    try:
        from app.sim_engine.progression.development import calculate_nhl_readiness_score

        return round(float(calculate_nhl_readiness_score(player)), 1)
    except Exception:
        raw = float(getattr(player, "nhl_readiness", 0) or 0)
        return raw if raw > 1.5 else round(raw * 99.0, 1)


def _dev_readiness_tier(player: Any, score: float) -> str:
    s = float(score or 0)
    goalie = _dev_is_goalie(player)
    defense = _dev_is_defense(player)
    if goalie:
        if s >= 72:
            return _DEV_READINESS_NHL_READY
        if s >= 62:
            return _DEV_READINESS_CLOSE
        if s >= 50:
            return _DEV_READINESS_DEVELOPING
        if s >= 38:
            return _DEV_READINESS_LONG_TERM
        return _DEV_READINESS_AT_RISK
    if defense:
        if s >= 70:
            return _DEV_READINESS_NHL_READY
        if s >= 60:
            return _DEV_READINESS_CLOSE
        if s >= 48:
            return _DEV_READINESS_DEVELOPING
        if s >= 36:
            return _DEV_READINESS_LONG_TERM
        return _DEV_READINESS_AT_RISK
    if s >= 68:
        return _DEV_READINESS_NHL_READY
    if s >= 58:
        return _DEV_READINESS_CLOSE
    if s >= 46:
        return _DEV_READINESS_DEVELOPING
    if s >= 34:
        return _DEV_READINESS_LONG_TERM
    return _DEV_READINESS_AT_RISK


def _dev_development_phase(player: Any) -> str:
    try:
        from app.sim_engine.progression.development import (
            PHASE_DECLINING,
            PHASE_EMERGING,
            PHASE_PRIME,
            PHASE_PROSPECT,
            PHASE_VETERAN,
            career_phase_for_age,
            determine_development_career_stage,
        )

        if _dev_is_goalie(player) and _dev_player_age(player) < 24:
            return "Goalie Development"
        phase = str(getattr(player, "career_phase", "") or career_phase_for_age(_dev_player_age(player)))
        stage = str(getattr(player, "development_career_stage", "") or determine_development_career_stage(player))
        if phase == PHASE_PROSPECT or stage == "prospect":
            return "Early Development"
        if phase == PHASE_EMERGING:
            return "Growth"
        if phase == PHASE_PRIME:
            return "Prime"
        if phase in (PHASE_VETERAN, PHASE_DECLINING) or stage in ("veteran", "declining", "late_career"):
            return "Decline"
        return "Refinement"
    except Exception:
        age = _dev_player_age(player)
        if _dev_is_goalie(player) and age < 24:
            return "Goalie Development"
        if age <= 22:
            return "Early Development"
        if age <= 26:
            return "Growth"
        if age <= 30:
            return "Refinement"
        if age <= 34:
            return "Prime"
        return "Decline"


def _dev_normalize_trend(
    player: Any,
    ovr_delta: float,
    attr_deltas: Dict[str, float],
    *,
    session: Optional[FranchiseSession] = None,
    season_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Classify trend from recomputed OVR delta; stale player metadata cannot override."""
    attr_sum = sum(v for v in attr_deltas.values() if v > 0)
    attr_neg = sum(v for v in attr_deltas.values() if v < 0)

    if ovr_delta >= 1.2:
        return _DEV_TREND_BREAKOUT
    if ovr_delta >= 0.35:
        return _DEV_TREND_IMPROVED
    if ovr_delta <= -0.8:
        return _DEV_TREND_REGRESSED
    if ovr_delta <= -0.35:
        return _DEV_TREND_REGRESSED

    if abs(ovr_delta) < 0.15:
        if attr_sum >= 0.75 and attr_neg > -0.4:
            return _DEV_TREND_IMPROVED
        if attr_neg <= -0.75 and attr_sum < 0.4:
            return _DEV_TREND_REGRESSED
        gp = 0
        if isinstance(season_stats, dict):
            gp = int(season_stats.get("gp", 0) or 0)
        if gp <= 0 and player is not None:
            gp = _dev_games_played(player, session)
        # Only call it a stall when they barely played — not for 80-point seasons.
        if player is not None and gp < 20 and _dev_player_age(player) <= 26 and attr_sum < 0.35:
            return _DEV_TREND_STALLED
        return _DEV_TREND_STABLE

    if ovr_delta > 0:
        return _DEV_TREND_IMPROVED
    return _DEV_TREND_REGRESSED


def _dev_games_played(player: Any, session: Optional[FranchiseSession]) -> int:
    try:
        from app.sim_engine.progression.development import _get_games_played

        gp = int(_get_games_played(player) or 0)
        if gp > 0:
            return gp
    except Exception:
        pass
    if session is not None:
        pid = _dev_player_id(player)
        row = dict((getattr(session, "player_season_stats", None) or {}).get(pid) or {})
        return int(row.get("gp", 0) or 0)
    return int(getattr(player, "gp", 0) or getattr(player, "games_played", 0) or 0)


def _dev_franchise_team_name(session: FranchiseSession) -> str:
    uid = str(getattr(session, "user_team_id", "") or "")
    user_team = session.team_by_id.get(uid) if uid else None
    if user_team is None:
        return ""
    try:
        from services.franchise_sim import _display_team

        return str(_display_team(user_team) or "")
    except Exception:
        return str(getattr(user_team, "name", "") or getattr(user_team, "city", "") or "")


def _dev_org_group(pool: str, league_id: str, signed: bool) -> str:
    p = str(pool or "").lower()
    lid = str(league_id or "").upper()
    if p in ("nhl", "ahl"):
        return "NHL / AHL"
    if p == "echl":
        return "ECHL"
    if not signed or p == "unsigned":
        return "Unsigned"
    if "NCAA" in lid or lid in ("OHL", "WHL", "QMJHL", "USHL") or "CHL" in lid:
        return "Junior / NCAA"
    if lid.startswith("EU_") or any(x in lid for x in ("SHL", "LIIGA", "DEL", "KHL", "NL")):
        return "Europe"
    if p == "development":
        if "NCAA" in lid or "CHL" in lid or lid in ("OHL", "WHL", "QMJHL", "USHL"):
            return "Junior / NCAA"
        return "Europe"
    return "Junior / NCAA"


def _dev_collect_org_entries(session: FranchiseSession) -> List[Dict[str, Any]]:
    uid = str(getattr(session, "user_team_id", "") or "")
    user_team = session.team_by_id.get(uid)
    entries: List[Dict[str, Any]] = []
    seen: set = set()

    def add(player: Any, pool: str, team_id: str, league_id: str, team_name: str = "") -> None:
        if player is None or getattr(player, "retired", False):
            return
        pid = _dev_player_id(player)
        if not pid:
            return
        norm = pid.lower()
        if norm in seen:
            return
        seen.add(norm)
        signed = _dev_is_signed(player)
        entries.append(
            {
                "player": player,
                "player_id": pid,
                "pool": pool,
                "organization_id": uid,
                "team_id": str(team_id or uid),
                "league_id": str(league_id or ""),
                "team_name": str(team_name or ""),
                "signed": signed,
                "org_group": _dev_org_group(pool, league_id, signed),
            }
        )

    if user_team is not None:
        from services.franchise_sim import _display_team

        tname = _display_team(user_team)
        for p in getattr(user_team, "roster", None) or []:
            add(p, "nhl", uid, "NHL", tname)
        for p in getattr(user_team, "ahl_roster", None) or []:
            add(p, "ahl", uid, "AHL", tname)
        for p in getattr(user_team, "echl_roster", None) or []:
            add(p, "echl", uid, "ECHL", tname)
        for p in getattr(user_team, "prospect_pool", None) or getattr(user_team, "prospects", None) or []:
            if str(getattr(p, "status", "") or "").lower() in ("nhl", "active"):
                continue
            lid = str(getattr(p, "current_league_id", "") or getattr(p, "development_path", "") or "Unsigned")
            pool = "prospects" if _dev_is_signed(p) else "unsigned"
            add(p, pool, uid, lid, tname)

    league = getattr(getattr(session, "sim", None), "league", None)
    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "")
        for tm in block.get("teams") or []:
            tname = str(tm.get("name") or "")
            for p in tm.get("players") or []:
                rights = str(
                    getattr(p, "nhl_rights_team_id", None)
                    or getattr(p, "rights_team_id", None)
                    or ""
                )
                if rights != uid:
                    continue
                add(p, "development", uid, code, tname)
    return entries


def _dev_season_stats(session: FranchiseSession, player: Any) -> Dict[str, Any]:
    pid = _dev_player_id(player)
    row = dict((getattr(session, "player_season_stats", None) or {}).get(pid) or {})
    gp = int(row.get("gp", 0) or getattr(player, "gp", 0) or 0)
    if gp <= 0:
        gp = int(getattr(player, "games_played", 0) or 0)
    goals = int(row.get("g", row.get("goals", 0)) or 0)
    assists = int(row.get("a", row.get("assists", 0)) or 0)
    points = int(row.get("pts", goals + assists) or 0)
    out: Dict[str, Any] = {
        "gp": gp,
        "goals": goals,
        "assists": assists,
        "points": points,
        "league": str(row.get("league", "") or getattr(player, "current_league_id", "") or ""),
    }
    if gp > 0:
        out["ppg"] = round(points / gp, 2)
    toi = row.get("toi_sec")
    if toi:
        out["toi_sec"] = int(toi)
    if _dev_is_goalie(player):
        starts = int(row.get("starts", row.get("gs", 0)) or 0)
        sv = row.get("sv_pct", row.get("save_pct"))
        gaa = row.get("gaa", row.get("goals_against_avg"))
        if starts:
            out["starts"] = starts
        if sv is not None:
            out["save_pct"] = round(float(sv), 3) if float(sv) <= 1.5 else round(float(sv), 1)
        if gaa is not None:
            out["gaa"] = round(float(gaa), 2)
        if gp:
            out["workload"] = starts or gp
    return out


def _dev_league_adjusted_context(player: Any, season_stats: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from services.unsigned_prospect_development import _league_quality

        lid = str(
            season_stats.get("league")
            or getattr(player, "current_league_id", "")
            or getattr(player, "development_path", "")
            or ""
        )
        lq = float(_league_quality(lid))
        gp = int(season_stats.get("gp", 0) or 0)
        ppg = float(season_stats.get("ppg", 0) or 0)
        out: Dict[str, Any] = {"league_quality": round(lq, 2)}
        if gp >= 10 and ppg > 0:
            out["production_rate"] = round(ppg * lq, 2)
        prod = float(getattr(player, "production_score", 0) or getattr(player, "ppg", 0) or 0)
        if prod > 0 and gp >= 10:
            out["production_score"] = round(prod, 2)
        return out
    except Exception:
        return {}


def _dev_build_reasons(
    player: Any,
    session: FranchiseSession,
    *,
    trend: str,
    ovr_delta: float,
    attr_deltas: Dict[str, float],
    season_stats: Dict[str, Any],
    league_ctx: Dict[str, Any],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    pos_deltas = sorted(
        ((k, v) for k, v in attr_deltas.items() if v > 0 and _dev_is_skill_attr(k)),
        key=lambda x: -x[1],
    )
    neg_deltas = sorted(
        ((k, v) for k, v in attr_deltas.items() if v < 0 and _dev_is_skill_attr(k)),
        key=lambda x: x[1],
    )
    injured = bool(getattr(player, "injured", False) or getattr(player, "injury_flag", False))
    gp = int(season_stats.get("gp", 0) or 0)

    if trend not in (_DEV_TREND_REGRESSED,) and ovr_delta >= -0.1:
        if len(pos_deltas) >= 2:
            reasons.append(f"{_dev_attr_label(pos_deltas[0][0])} and {_dev_attr_label(pos_deltas[1][0])} improved")
        elif len(pos_deltas) == 1:
            reasons.append(f"{_dev_attr_label(pos_deltas[0][0])} improved")

    if league_ctx.get("production_rate") and float(league_ctx["production_rate"]) >= 0.55 and trend in (
        _DEV_TREND_IMPROVED,
        _DEV_TREND_BREAKOUT,
    ):
        lid = str(season_stats.get("league", "") or "").upper()
        if "AHL" in lid:
            reasons.append("Strong AHL production accelerated readiness")
        elif gp >= 20:
            reasons.append("League production supported development")

    if gp > 0 and gp < 15 and _dev_player_age(player) <= 24 and trend == _DEV_TREND_STALLED:
        reasons.append("Limited games slowed development")

    if injured and neg_deltas:
        skate_keys = [k for k, _ in neg_deltas if "skat" in k.lower() or "speed" in k.lower()]
        if skate_keys:
            reasons.append("Injury reduced skating growth")
        elif trend == _DEV_TREND_REGRESSED:
            reasons.append("Injury limited offseason gains")

    if _dev_player_age(player) >= 27 and trend == _DEV_TREND_REGRESSED and ovr_delta < 0:
        reasons.append("Regression driven by age decline")

    if _dev_is_goalie(player) and _dev_player_age(player) < 23 and _dev_readiness_tier(player, _dev_readiness_score_raw(player)) in (
        _DEV_READINESS_LONG_TERM,
        _DEV_READINESS_DEVELOPING,
    ):
        reasons.append("Goalie development remains long-term")

    dev_type = str(getattr(player, "dev_type", "") or getattr(player, "_dev_archetype", "") or "").lower()
    if "late_bloomer" in dev_type and ovr_delta >= 0.5 and _dev_player_age(player) >= 22:
        reasons.append("Late-bloomer growth exceeded expectations")

    if trend == _DEV_TREND_STALLED and not reasons:
        reasons.append("Development plateaued this cycle")

    if trend == _DEV_TREND_REGRESSED and ovr_delta < -0.3 and not reasons:
        if len(neg_deltas) >= 2:
            reasons.append(f"{_dev_attr_label(neg_deltas[0][0])} and {_dev_attr_label(neg_deltas[1][0])} declined")
        elif len(neg_deltas) == 1:
            reasons.append(f"{_dev_attr_label(neg_deltas[0][0])} declined")
        else:
            reasons.append("Overall skills declined this cycle")

    if trend == _DEV_TREND_STABLE and not reasons and abs(ovr_delta) < 0.2:
        reasons.append("Maintained current development level")

    trimmed = [r for r in reasons if r and len(r.split()) <= 15]
    if not trimmed:
        return "", []
    return trimmed[0], trimmed[1:3]


def _dev_notable_category(
    player: Any,
    *,
    trend: str,
    ovr_delta: float,
    readiness_tier: str,
    prev_tier: str,
    attr_deltas: Dict[str, float],
) -> Tuple[bool, str]:
    attr_gain = sum(v for v in attr_deltas.values() if v > 0)
    dev_type = str(getattr(player, "dev_type", "") or "").lower()
    if trend == _DEV_TREND_BREAKOUT and ovr_delta >= 1.0:
        return True, "Top Riser"
    if ovr_delta >= 1.5:
        return True, "Top Riser"
    if trend == _DEV_TREND_REGRESSED and ovr_delta <= -0.5:
        return True, "Regressed"
    if ovr_delta <= -1.0:
        return True, "Regressed"
    if readiness_tier == _DEV_READINESS_NHL_READY and prev_tier != _DEV_READINESS_NHL_READY:
        return True, "Newly NHL Ready"
    if trend == _DEV_TREND_STALLED:
        return True, "Stalled"
    if "late_bloomer" in dev_type and _dev_player_age(player) >= 22 and ovr_delta >= 0.5:
        return True, "Late Bloomer"
    if readiness_tier == _DEV_READINESS_AT_RISK and (trend == _DEV_TREND_REGRESSED or ovr_delta < -0.5):
        return True, "High Risk"
    if attr_gain >= 2.5 and ovr_delta >= 0.4:
        return True, "Top Riser"
    return False, ""


def _dev_append_history(player: Any, record: Dict[str, Any], season: int) -> None:
    hist = getattr(player, "development_history", None)
    if not isinstance(hist, list):
        hist = []
        try:
            setattr(player, "development_history", hist)
        except Exception:
            return
    key_deltas = {
        k: v
        for k, v in sorted(
            (record.get("attribute_deltas") or {}).items(),
            key=lambda x: -abs(x[1]),
        )[:6]
    }
    entry = {
        "season": int(season),
        "previous_ovr": record.get("previous_overall"),
        "new_ovr": record.get("current_overall"),
        "ovr_delta": record.get("overall_delta"),
        "attribute_deltas": key_deltas,
        "development_trend": record.get("development_trend"),
        "league": record.get("current_league_id"),
        "readiness_tier": record.get("readiness_tier"),
        "kind": "offseason_report",
    }
    if hist and isinstance(hist[-1], dict) and int(hist[-1].get("season", -1)) == int(season) and hist[-1].get("kind") == "offseason_report":
        hist[-1] = entry
    else:
        hist.append(entry)


def _dev_validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Reject impossible combinations of labels, attribute deltas, and OVR movement."""
    ovr_delta = float(record.get("overall_delta", 0) or 0)
    attr_deltas = dict(record.get("attribute_deltas") or {})
    trend = str(record.get("development_trend") or _DEV_TREND_STABLE)

    # Detect legacy corrupt scaling (uniform ±24–25 on many keys).
    if len(attr_deltas) >= 5:
        rounded = [round(float(v), 1) for v in attr_deltas.values()]
        if len(set(rounded)) == 1 and abs(rounded[0]) >= 20:
            record["attribute_deltas"] = {}
            record["previous_attributes"] = {}
            record["current_attributes"] = {}
            attr_deltas = {}

    # Reconcile trend with authoritative OVR delta.
    attr_sum = sum(v for v in attr_deltas.values() if v > 0)
    attr_neg = sum(v for v in attr_deltas.values() if v < 0)
    expected = _dev_normalize_trend(None, ovr_delta, attr_deltas)
    if trend in (_DEV_TREND_BREAKOUT, _DEV_TREND_REGRESSED) and abs(ovr_delta) < 0.15:
        trend = expected
    elif trend == _DEV_TREND_BREAKOUT and ovr_delta < 0.8:
        trend = expected
    elif trend == _DEV_TREND_REGRESSED and ovr_delta > -0.25:
        trend = expected
    record["development_trend"] = trend

    primary = str(record.get("primary_reason") or "")
    if primary and "improved" in primary.lower() and trend == _DEV_TREND_REGRESSED:
        record["primary_reason"] = ""
        record["secondary_reasons"] = []
    if primary and "declined" in primary.lower() and trend in (_DEV_TREND_IMPROVED, _DEV_TREND_BREAKOUT):
        record["primary_reason"] = ""
        record["secondary_reasons"] = []

    notable_cat = str(record.get("notable_category") or "")
    if notable_cat == "Regressed" and ovr_delta > -0.4:
        record["notable"] = False
        record["notable_category"] = ""
    if notable_cat == "Top Riser" and ovr_delta < 0.35 and attr_sum < 1.0:
        record["notable"] = False
        record["notable_category"] = ""
    if trend == _DEV_TREND_STALLED and abs(ovr_delta) >= 0.5:
        record["notable"] = bool(record.get("notable")) and notable_cat not in ("Stalled", "")

    # Drop readiness headline when regression dominates.
    if (
        record.get("readiness_tier") == _DEV_READINESS_NHL_READY
        and trend == _DEV_TREND_REGRESSED
        and ovr_delta < -0.25
        and attr_neg <= -0.5
    ):
        record["readiness_tier"] = _DEV_READINESS_CLOSE

    return record


def _dev_build_record(
    session: FranchiseSession,
    entry: Dict[str, Any],
    before: Dict[str, Any],
    after: Dict[str, Any],
    *,
    season: int,
) -> Dict[str, Any]:
    player = entry["player"]
    ledger = getattr(player, "development_ledger", None) or {}
    if not isinstance(ledger, dict):
        ledger = {}

    snap_deltas = _dev_attr_deltas(before.get("attributes") or {}, after.get("attributes") or {})
    ledger_deltas = _dev_ledger_attr_deltas(ledger)

    # Integer OVRs everywhere in the Organizational Development Review.
    # Prefer season-start snapshot for cumulative season growth, then ledger, then snap.
    ledger_prev, ledger_cur = _dev_ovr_from_ledger(ledger)
    snap_prev = int(round(float(before.get("overall", 0) or 0)))
    snap_cur = int(round(float(after.get("overall", 0) or 0)))
    season_start = None
    try:
        raw_start = getattr(player, "season_start_ovr", None)
        if raw_start is None:
            raw_start = getattr(player, "_season_start_ovr", None)
        if raw_start is not None:
            season_start = int(round(float(raw_start)))
    except Exception:
        season_start = None
    if season_start is not None and season_start > 0:
        prev_ovr = season_start
        cur_ovr = snap_cur if snap_cur > 0 else int(round(ledger_cur or snap_cur))
        ovr_delta = int(cur_ovr - prev_ovr)
    elif ledger_prev is not None:
        prev_ovr = int(round(ledger_prev))
        # Live OVR after any soft-regression / follow-on passes.
        cur_ovr = snap_cur if snap_cur > 0 else int(round(ledger_cur or snap_cur))
        ovr_delta = int(cur_ovr - prev_ovr)
    else:
        prev_ovr = snap_prev
        cur_ovr = snap_cur
        ovr_delta = int(cur_ovr - prev_ovr)

    if snap_deltas:
        attr_deltas = dict(snap_deltas)
        if ledger_deltas:
            for k, v in ledger_deltas.items():
                if k not in attr_deltas or abs(float(attr_deltas.get(k, 0) or 0)) < abs(float(v)):
                    attr_deltas[k] = v
    elif ledger_deltas:
        attr_deltas = dict(ledger_deltas)
    else:
        attr_deltas = {}

    # Round attribute deltas to whole points for the report UI.
    attr_deltas = {
        k: int(round(float(v)))
        for k, v in attr_deltas.items()
        if abs(float(v)) >= 0.5
    }
    attr_deltas = {k: v for k, v in attr_deltas.items() if v != 0}
    readiness_score = _dev_readiness_score_raw(player)
    readiness_tier = _dev_readiness_tier(player, readiness_score)
    prev_tier = _dev_readiness_tier(player, float(before.get("readiness_score", 0) or 0))
    season_stats = _dev_season_stats(session, player)
    trend = _dev_normalize_trend(
        player,
        ovr_delta,
        attr_deltas,
        session=session,
        season_stats=season_stats,
    )
    league_ctx = _dev_league_adjusted_context(player, season_stats)
    primary, secondary = _dev_build_reasons(
        player,
        session,
        trend=trend,
        ovr_delta=ovr_delta,
        attr_deltas=attr_deltas,
        season_stats=season_stats,
        league_ctx=league_ctx,
    )
    notable, notable_category = _dev_notable_category(
        player,
        trend=trend,
        ovr_delta=ovr_delta,
        readiness_tier=readiness_tier,
        prev_tier=prev_tier,
        attr_deltas=attr_deltas,
    )
    # Keep only attributes that moved (plus a tiny floor set) — avoids huge/unsafe dumps.
    changed_keys = set(attr_deltas.keys())
    prev_attrs = {
        k: v for k, v in (before.get("attributes") or {}).items() if k in changed_keys
    }
    cur_attrs = {
        k: v for k, v in (after.get("attributes") or {}).items() if k in changed_keys
    }
    record = {
        "player_id": entry["player_id"],
        "organization_id": entry["organization_id"],
        "player_name": _dev_player_name(player),
        "position": _dev_player_position(player),
        "age": _dev_player_age(player),
        "contract_status": after.get("contract_status") or before.get("contract_status") or "",
        "signed": bool(entry.get("signed")),
        "previous_team_id": str(before.get("team_id", "") or ""),
        "current_team_id": str(after.get("team_id", "") or entry.get("team_id", "")),
        "previous_league_id": str(before.get("league_id", "") or ""),
        "current_league_id": str(after.get("league_id", "") or entry.get("league_id", "")),
        "previous_overall": prev_ovr,
        "current_overall": cur_ovr,
        "overall_delta": ovr_delta,
        "previous_attributes": prev_attrs,
        "current_attributes": cur_attrs,
        "attribute_deltas": attr_deltas,
        "potential": after.get("potential") or before.get("potential") or 0,
        "development_phase": _dev_development_phase(player),
        "development_trend": trend,
        "readiness_tier": readiness_tier,
        "readiness_score": readiness_score,
        "primary_reason": primary,
        "secondary_reasons": secondary,
        "season_stats": season_stats,
        "league_adjusted_context": league_ctx,
        "goalie": _dev_is_goalie(player),
        "notable": notable,
        "notable_category": notable_category,
        "org_group": entry.get("org_group", ""),
        "pool": entry.get("pool", ""),
        "team_name": (
            _dev_franchise_team_name(session)
            if (
                str(entry.get("league_id") or "").upper() in ("NHL", "AHL", "ECHL")
                or str(entry.get("org_group") or "").startswith("NHL")
                or str(entry.get("pool") or "").lower() in ("nhl", "ahl", "echl")
            )
            else str(entry.get("team_name", "") or "")
        ),
        "report_season": int(season),
        "development_history": _dev_compact_history(player),
    }
    _dev_append_history(player, record, season)
    return _dev_validate_record(record)


def _dev_compact_history(player: Any) -> List[Dict[str, Any]]:
    hist = getattr(player, "development_history", None)
    if not isinstance(hist, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in hist:
        if not isinstance(item, dict):
            continue
        if item.get("kind") not in ("offseason_report", None) and "ovr_delta" not in item:
            continue
        prev = item.get("previous_ovr", item.get("ovr_before"))
        new = item.get("new_ovr", item.get("ovr_after"))
        if prev is None and new is None:
            continue
        # History / ledger may store 0–1; promote to display if clearly fractional.
        try:
            prev_raw = float(prev) if prev is not None else None
            new_raw = float(new) if new is not None else None
        except Exception:
            continue
        if prev_raw is not None and abs(prev_raw) <= 1.5:
            prev_f = int(round(prev_raw * 99.0))
        else:
            prev_f = int(round(prev_raw)) if prev_raw is not None else None
        if new_raw is not None and abs(new_raw) <= 1.5:
            new_f = int(round(new_raw * 99.0))
        else:
            new_f = int(round(new_raw)) if new_raw is not None else None
        delta = item.get("ovr_delta")
        if delta is None and prev_f is not None and new_f is not None:
            delta = int(new_f - prev_f)
        else:
            try:
                d_raw = float(delta) if delta is not None else None
                if d_raw is not None and abs(d_raw) <= 1.5 and prev_f is not None and new_f is not None:
                    delta = int(new_f - prev_f)
                elif d_raw is not None:
                    delta = int(round(d_raw))
                else:
                    delta = None
            except Exception:
                delta = None
        rows.append(
            {
                "season": item.get("season"),
                "previous_ovr": prev_f,
                "new_ovr": new_f,
                "ovr_delta": delta,
                "development_trend": item.get("development_trend"),
                "readiness_tier": item.get("readiness_tier"),
                "league": item.get("league"),
            }
        )
    return rows[-5:]


def _run_user_org_depth_progression(session: FranchiseSession, season_id: int) -> None:
    from app.sim_engine.progression import run_player_progression

    entries = _dev_collect_org_entries(session)
    rng = session.sim.rng
    for entry in entries:
        pool = str(entry.get("pool") or "")
        if pool in ("nhl", "unsigned"):
            continue
        player = entry["player"]
        try:
            _dev_stamp_season_production(session, player)
            setattr(player, "_active_dev_season", season_id)
            setattr(player, "_dev_source_path", f"org_{pool}")
            run_player_progression(
                player,
                rng,
                season_id=season_id,
                source_path=f"org_{pool}",
            )
        except Exception:
            pass


def _dev_stamp_season_production(session: FranchiseSession, player: Any) -> None:
    """Feed real season production into growth modifiers (PPG / workload)."""
    stats = _dev_season_stats(session, player)
    gp = int(stats.get("gp", 0) or 0)
    if gp <= 0:
        return
    try:
        # Season production stamp — replace, never accumulate across years.
        setattr(player, "games_played", int(gp))
        setattr(player, "gp", int(gp))
    except Exception:
        pass
    if _dev_is_goalie(player):
        sv = float(stats.get("save_pct") or stats.get("sv_pct") or 0.0)
        if sv > 1.5:
            sv = sv / 100.0
        # .900 → ~0.55, .920 → ~0.78, .930 → ~0.90
        score = 0.35 + max(0.0, (sv - 0.880) * 8.5) if sv > 0 else 0.5
    else:
        pts = float(stats.get("points", 0) or 0)
        ppg = pts / float(gp)
        # 0.40 PPG ~ depth, 0.70 solid, 0.95+ star season
        score = 0.32 + ppg * 0.55
        if ppg >= 0.85:
            score = max(score, 0.82)
        if ppg >= 0.95:
            score = max(score, 0.90)
        if ppg >= 1.10:
            score = max(score, 0.95)
    score = max(0.22, min(0.98, float(score)))
    for key in ("production_score", "recent_performance_score", "points_signal", "production"):
        try:
            setattr(player, key, score)
        except Exception:
            pass


def _dev_season_gap_ovr(player: Any) -> float:
    """Runway to potential in display OVR points."""
    try:
        from services.franchise_sim import _player_ovr99

        ovr = float(_player_ovr99(player))
    except Exception:
        ovr = 0.0
    pot = float(_dev_player_potential_display(player) or 0)
    return max(0.0, pot - ovr)


def _dev_expected_display_growth(player: Any) -> float:
    """Individualized expected displayed OVR growth for catch-up comparisons."""
    try:
        from app.sim_engine.progression.development import (
            calculate_season_growth_budget,
            resolve_development_profile,
            _SEASON_END_POOL_SHARE,
        )

        profile = resolve_development_profile(player)
        # Deterministic mid-noise estimate (no RNG) via phase NORMAL budget.
        class _FixedRng:
            def uniform(self, a, b):
                return (float(a) + float(b)) * 0.5

            def random(self):
                return 0.5

            def choice(self, seq):
                return seq[0] if seq else None

        budget = calculate_season_growth_budget(
            player, None, profile, rng=_FixedRng(), dev_phase="NORMAL"
        )
        # Season-end pool only — mid-season is separate.
        return max(0.0, float(budget) * 99.0 * float(_SEASON_END_POOL_SHARE))
    except Exception:
        gap = _dev_season_gap_ovr(player)
        age = int(_dev_player_age(player) or 99)
        if age <= 20 and gap >= 10:
            return 4.5
        if age <= 23 and gap >= 7:
            return 3.5
        if gap >= 4:
            return 2.5
        return 1.5


def _dev_needs_growth_catchup(player: Any) -> bool:
    """True when actual growth falls materially below individualized expected growth."""
    age = int(_dev_player_age(player) or 99)
    if age > 28:
        return False
    if _dev_season_gap_ovr(player) < 3.0:
        return False
    ledger = getattr(player, "development_ledger", None) or {}
    if not isinstance(ledger, dict) or not ledger.get("development_applied"):
        return True
    lp, lc = _dev_ovr_from_ledger(ledger)
    if lp is None or lc is None:
        return True
    actual = float(lc) - float(lp)
    # Prefer season-start cumulative if present.
    try:
        start = getattr(player, "season_start_ovr", None)
        if start is None:
            start = getattr(player, "_season_start_ovr", None)
        if start is not None:
            from services.franchise_sim import _player_ovr99

            actual = float(_player_ovr99(player)) - float(start)
    except Exception:
        pass
    expected = _dev_expected_display_growth(player)
    # Catch up when shortfall is large (design §12) — not a universal +1.5 gate.
    if expected <= 1.0:
        return actual < 0.75
    return actual < (expected * 0.55)


def _run_development_growth_catchup(session: FranchiseSession, season_id: int) -> int:
    """
    Re-open development for players who already ran under the old tiny budgets
    (or stalled to ~0) despite real potential runway. Preserves season-start OVR.
    """
    from app.sim_engine.entities.player import player_current_ovr_01
    from app.sim_engine.progression.development import apply_player_development
    from app.sim_engine.progression.potential import ensure_development_ledger

    rng = session.sim.rng
    touched = 0
    seen: set = set()

    def _one(player: Any) -> None:
        nonlocal touched
        if player is None or getattr(player, "retired", False):
            return
        pid = _dev_player_id(player)
        if not pid or pid.lower() in seen:
            return
        seen.add(pid.lower())
        _dev_stamp_season_production(session, player)
        if not _dev_needs_growth_catchup(player):
            return
        try:
            setattr(player, "_active_dev_season", season_id)
            ledger = ensure_development_ledger(player, season_id)
            # Catchup is a single corrective pass — never reopen every stage entry.
            if ledger.get("catchup_applied"):
                return
            orig_before = ledger.get("ovr_before")
            if orig_before is None:
                orig_before = float(player_current_ovr_01(player))
            prev_attrs = dict(ledger.get("attribute_deltas") or {})
            ledger["development_applied"] = False
            apply_player_development(player, rng)
            ledger = getattr(player, "development_ledger", None) or ledger
            if not isinstance(ledger, dict):
                return
            ledger["ovr_before"] = float(orig_before)
            merged = {str(k): float(v) for k, v in prev_attrs.items()}
            for k, v in (ledger.get("attribute_deltas") or {}).items():
                merged[str(k)] = float(merged.get(str(k), 0.0)) + float(v)
            ledger["attribute_deltas"] = merged
            try:
                ledger["ovr_after"] = float(player_current_ovr_01(player))
            except Exception:
                pass
            ledger["source_path"] = "apply_player_development:catchup_v5"
            ledger["catchup_applied"] = True
            ledger["development_applied"] = True
            try:
                setattr(player, "development_ledger", ledger)
            except Exception:
                pass
            touched += 1
        except Exception:
            pass

    league = getattr(getattr(session, "sim", None), "league", None)
    for team in list(getattr(league, "teams", None) or []):
        for p in getattr(team, "roster", None) or []:
            _one(p)
    for entry in _dev_collect_org_entries(session):
        if str(entry.get("pool") or "") == "nhl":
            continue
        _one(entry.get("player"))
    return touched


def _dev_summary_line(summary: Dict[str, Any]) -> str:
    parts = []
    if summary.get("improved"):
        parts.append(f"{summary['improved']} improved")
    if summary.get("nhl_ready"):
        parts.append(f"{summary['nhl_ready']} NHL ready")
    if summary.get("stalled"):
        parts.append(f"{summary['stalled']} stalled")
    if summary.get("regressed"):
        parts.append(f"{summary['regressed']} regressed")
    return " · ".join(parts) if parts else "No notable movement"


def _run_offseason_development(session: FranchiseSession) -> Dict[str, Any]:
    from datetime import datetime, timezone

    season_id = int(getattr(session, "season_calendar_year", 2025) or 2025)
    completed = int(getattr(session, "development_report_completed_season", 0) or 0)

    def _safe_payload(raw: Any) -> Dict[str, Any]:
        try:
            from services.json_safe import json_safe

            cleaned = json_safe(raw)
            return cleaned if isinstance(cleaned, dict) else {}
        except Exception:
            return raw if isinstance(raw, dict) else {}

    if completed == season_id and session.development_report_payload:
        session.development_report_payload = _safe_payload(session.development_report_payload)
        cached = session.development_report_payload
        if int(cached.get("schema_version", 1) or 1) >= 5:
            return {"development_report": cached}
    if getattr(session, "development_report_done", False) and session.development_report_payload:
        payload = _safe_payload(session.development_report_payload)
        session.development_report_payload = payload
        if (
            int(payload.get("report_season", 0) or 0) == season_id
            and int(payload.get("schema_version", 1) or 1) >= 5
        ):
            return {"development_report": payload}

    import run_sim as rs
    from services.franchise_sim import _player_ovr99

    sim = session.sim
    league = getattr(sim, "league", None)
    teams = list(getattr(league, "teams", None) or [])
    rng = sim.rng

    org_entries = _dev_collect_org_entries(session)
    # Stamp production so growth budgets see real PPG / workload.
    for entry in org_entries:
        try:
            _dev_stamp_season_production(session, entry["player"])
        except Exception:
            pass

    before_snapshots: Dict[str, Dict[str, Any]] = {}
    for entry in org_entries:
        player = entry["player"]
        pid = entry["player_id"]
        snap = _dev_snapshot_player(
            player,
            team_id=entry.get("team_id", ""),
            league_id=entry.get("league_id", ""),
        )
        snap["readiness_tier"] = _dev_readiness_tier(player, snap["readiness_score"])
        before_snapshots[pid] = snap

    league_before: Dict[str, float] = {}
    for team in teams:
        for p in getattr(team, "roster", None) or []:
            pid = _dev_player_id(p)
            if not pid:
                continue
            try:
                _dev_stamp_season_production(session, p)
            except Exception:
                pass
            league_before[pid] = _player_ovr99(p)
            try:
                setattr(p, "_active_dev_season", season_id)
            except Exception:
                pass

    if getattr(rs, "_run_player_progression_pass", None) and not getattr(
        session, "_year_end_progression_done", False
    ):
        try:
            rs._run_player_progression_pass(teams, rng, None)
        except Exception:
            pass

    try:
        _run_user_org_depth_progression(session, season_id)
    except Exception:
        pass

    # Year-end often already applied weak/zero growth under the old formula.
    # Catch up high-runway players (e.g. 84 OVR / 94 POT) so the review isn't flat.
    try:
        _run_development_growth_catchup(session, season_id)
    except Exception:
        pass

    unsigned_dev: Dict[str, Any] = {}
    try:
        from services.unsigned_prospect_development import run_unsigned_prospect_development_pass

        unsigned_dev = run_unsigned_prospect_development_pass(session, season_year=season_id)
    except Exception:
        unsigned_dev = {"developed": 0, "results": []}

    risers: List[Dict[str, Any]] = []
    fallers: List[Dict[str, Any]] = []
    for team in teams:
        tname = ""
        try:
            from services.franchise_sim import _display_team

            tname = _display_team(team)
        except Exception:
            tname = str(getattr(team, "name", "") or getattr(team, "city", "") or "")
        for p in getattr(team, "roster", None) or []:
            pid = _dev_player_id(p)
            if not pid:
                continue
            after = _player_ovr99(p)
            before_v = float(league_before.get(pid, after))
            # Prefer true season-start OVR for cumulative year growth display.
            try:
                start = getattr(p, "season_start_ovr", None)
                if start is None:
                    start = getattr(p, "_season_start_ovr", None)
                if start is not None:
                    before_v = float(start)
            except Exception:
                pass
            # Year-end progression often already applied — recover true delta from ledger.
            ledger = getattr(p, "development_ledger", None) or {}
            lp, lc = _dev_ovr_from_ledger(ledger if isinstance(ledger, dict) else {})
            if before_v == float(league_before.get(pid, after)) and lp is not None and lc is not None:
                before_v = float(lp)
                after = float(lc)
            elif lc is not None:
                after = float(lc)
            diff = after - before_v
            if abs(diff) >= 0.5:
                row = {
                    "player_id": pid,
                    "name": _dev_player_name(p),
                    "player_name": _dev_player_name(p),
                    "position": _dev_player_position(p),
                    "age": _dev_player_age(p),
                    "delta": int(round(diff)),
                    "overall_delta": int(round(diff)),
                    "overall": int(round(after)),
                    "current_overall": int(round(after)),
                    "previous_overall": int(round(before_v)),
                    "team_name": tname,
                    "current_league_id": "NHL",
                    "org_group": "NHL / AHL",
                    "league_id": "NHL",
                    "potential": int(round(_dev_player_potential_display(p) or 0)),
                }
                if isinstance(ledger, dict):
                    row["attribute_deltas"] = {
                        k: int(round(float(v)))
                        for k, v in (_dev_ledger_attr_deltas(ledger) or {}).items()
                        if abs(float(v)) >= 0.5
                    }
                if diff > 0:
                    risers.append(row)
                else:
                    fallers.append(row)

    risers.sort(key=lambda r: -int(r.get("delta", 0) or 0))
    fallers.sort(key=lambda r: int(r.get("delta", 0) or 0))

    organization_players: List[Dict[str, Any]] = []
    for entry in org_entries:
        player = entry["player"]
        pid = entry["player_id"]
        before = before_snapshots.get(pid) or _dev_snapshot_player(
            player,
            team_id=entry.get("team_id", ""),
            league_id=entry.get("league_id", ""),
        )
        after = _dev_snapshot_player(
            player,
            team_id=entry.get("team_id", ""),
            league_id=entry.get("league_id", ""),
        )
        after["readiness_tier"] = _dev_readiness_tier(player, after["readiness_score"])
        record = _dev_build_record(session, entry, before, after, season=season_id)
        organization_players.append(record)

    organization_players.sort(
        key=lambda r: (
            -float(r.get("current_overall", 0) or 0),
            -abs(float(r.get("overall_delta", 0) or 0)),
        )
    )

    prospects_ready = [
        {
            "player_id": r["player_id"],
            "name": r["player_name"],
            "position": r["position"],
            "overall": r["current_overall"],
            "readiness_tier": r["readiness_tier"],
            "readiness_score": r["readiness_score"],
        }
        for r in organization_players
        if r.get("readiness_tier") == _DEV_READINESS_NHL_READY
    ]

    summary = {
        "improved": sum(1 for r in organization_players if r.get("development_trend") in (_DEV_TREND_IMPROVED, _DEV_TREND_BREAKOUT)),
        "nhl_ready": len(prospects_ready),
        "stalled": sum(1 for r in organization_players if r.get("development_trend") == _DEV_TREND_STALLED),
        "regressed": sum(1 for r in organization_players if r.get("development_trend") == _DEV_TREND_REGRESSED),
        "total": len(organization_players),
    }
    summary["line"] = _dev_summary_line(summary)

    payload = {
        "schema_version": 5,
        "report_season": season_id,
        "organization_players": organization_players,
        "league_risers": risers[:48],
        "league_fallers": fallers[:24],
        "prospects_ready": prospects_ready,
        "summary": summary,
        "risers": risers[:48],
        "fallers": fallers[:24],
        "unsigned_prospect_development": unsigned_dev,
        "org_prospect_deltas": list(unsigned_dev.get("results") or [])[:12],
    }
    try:
        from services.json_safe import json_safe

        payload = json_safe(payload)
    except Exception:
        pass
    session.development_report_payload = payload if isinstance(payload, dict) else {}
    session.development_report_done = True
    session.development_report_completed_season = season_id
    session.development_report_generated_at = datetime.now(timezone.utc).isoformat()
    return {"development_report": session.development_report_payload}


def _run_draft_lottery(session: FranchiseSession) -> Dict[str, Any]:
    from services.franchise_sim import (
        _build_standings_rows,
        _display_team,
        invalidate_session_payload_caches,
    )
    from datetime import datetime, timezone

    if session.draft_lottery_done and session.draft_lottery_payload:
        payload = dict(session.draft_lottery_payload)
        if not payload.get("ownership_annotated"):
            try:
                from services.draft_pick_ownership import annotate_lottery_picks_with_ownership

                picks = annotate_lottery_picks_with_ownership(
                    session,
                    list(payload.get("picks") or payload.get("final_order") or payload.get("order") or []),
                )
                payload["picks"] = picks
                payload["final_order"] = picks
                payload["order"] = picks
                payload["ownership_annotated"] = True
                session.draft_lottery_payload = payload
            except Exception:
                pass
        return {"draft_lottery": payload}

    sim = session.sim
    standings_rows = _build_standings_rows(session)
    ordered = sorted(standings_rows, key=lambda r: (int(r.get("pts", 0)), -int(r.get("w", 0))))
    pre_lottery_order: List[Dict[str, Any]] = []
    for i, row in enumerate(ordered[:16], start=1):
        tid = str(row.get("team_id", ""))
        tm = session.team_by_id.get(tid)
        pre_lottery_order.append({
            "lottery_rank": i,
            "team_id": tid,
            "team_name": _display_team(tm) if tm else tid,
            "points": int(row.get("pts", 0)),
            "wins": int(row.get("w", 0)),
        })

    picks: List[Dict[str, Any]] = []
    draw_results: List[Dict[str, Any]] = []
    seed = hash(
        (
            int(session.season_calendar_year),
            int(getattr(sim.rng, "getstate", lambda: (0,))()[1][0] if hasattr(sim.rng, "getstate") else 0),
        )
    ) % (2**31)
    try:
        from app.sim_engine.draft.draft_lottery import LotteryTeam, run_draft_lottery

        lot_teams = [
            LotteryTeam(team_id=str(row.get("team_id", "")), points=int(row.get("pts", 0)))
            for row in ordered[:16]
        ]
        result = run_draft_lottery(teams=lot_teams, seed=seed)
        order = list(getattr(result, "pick_order", None) or [])
        winners = list(getattr(result, "lottery_winners", None) or [])
        for i, tid in enumerate(winners[:2], start=1):
            draw_results.append({"draw": i, "team_id": str(tid), "won_pick": i})
        for pick_num, tid in enumerate(order[:16], start=1):
            orig_rank = next((i + 1 for i, r in enumerate(ordered) if str(r.get("team_id")) == str(tid)), pick_num)
            tm = session.team_by_id.get(str(tid))
            picks.append({
                "pick": pick_num,
                "team_id": str(tid),
                "team_name": _display_team(tm) if tm else str(tid),
                "original_rank": orig_rank,
                "movement": orig_rank - pick_num,
                "won_pick": pick_num if pick_num <= 2 and str(tid) in {str(w) for w in winners[:2]} else None,
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
                "won_pick": None,
            })

    # Finalize protections against lottery outcome before draft order creation later.
    # Use lottery *earners* (standings teams), not current pick owners.
    try:
        from services.draft_pick_conditions import resolve_pick_protections

        league = getattr(sim, "league", None)
        if league is not None:
            resolve_pick_protections(
                league,
                draft_year=int(session.season_calendar_year) + 1,
                lottery_order=[str(p.get("lottery_team_id") or p["team_id"]) for p in picks],
            )
    except Exception:
        pass

    # Resolve traded ownership so lottery UI shows selecting team via original owner.
    try:
        from services.draft_pick_ownership import annotate_lottery_picks_with_ownership

        picks = annotate_lottery_picks_with_ownership(session, picks)
    except Exception:
        pass

    payload = {
        "lottery_seed": seed,
        "pre_lottery_order": pre_lottery_order,
        "draw_results": draw_results,
        "final_order": picks,
        "picks": picks,
        "order": picks,
        "movement": [
            {
                "team_id": p.get("lottery_team_id") or p["team_id"],
                "movement": p.get("movement", 0),
            }
            for p in picks
        ],
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ownership_annotated": True,
    }
    session.draft_lottery_payload = payload
    session.draft_lottery_done = True
    invalidate_session_payload_caches(session, reason="draft_lottery")
    return {"draft_lottery": payload}


# ── Draft Review (post-draft evaluation) ─────────────────────────────────────

_DR_LEAGUE_LABELS = {
    "CHL_QMJHL": "QMJHL",
    "CHL_OHL": "OHL",
    "CHL_WHL": "WHL",
    "QMJHL": "QMJHL",
    "OHL": "OHL",
    "WHL": "WHL",
    "USHL": "USHL",
    "NCAA": "NCAA",
    "NCAA_D1": "NCAA",
    "EU_SHL": "SHL",
    "SHL": "SHL",
    "EU_LIIGA": "Liiga",
    "LIIGA": "Liiga",
    "EU_DEL": "DEL",
    "DEL": "DEL",
    "EU_KHL": "KHL",
    "KHL": "KHL",
    "EU_CZE": "Czech Extraliga",
    "CZE": "Czech Extraliga",
    "EU_SUI": "NL",
    "SUI": "NL",
    "ALLSV": "Allsvenskan",
    "AHL": "AHL",
    "NHL": "NHL",
    "JUNIOR": "Junior",
    "EUROPE": "Europe",
}


def _dr_league_label(raw: Any) -> Optional[str]:
    s = str(raw or "").strip()
    if not s:
        return None
    key = s.upper().replace(" ", "_").replace("-", "_")
    if key in _DR_LEAGUE_LABELS:
        return _DR_LEAGUE_LABELS[key]
    if key.startswith("CHL_"):
        return key.split("_", 1)[1]
    if key.startswith("EU_"):
        return _DR_LEAGUE_LABELS.get(key, key[3:].replace("_", " ").title())
    if "_" in key and key.split("_")[-1] in _DR_LEAGUE_LABELS:
        return _DR_LEAGUE_LABELS[key.split("_")[-1]]
    return s.replace("_", " ")


def _dr_to_float(raw: Any, default: Optional[float] = None) -> Optional[float]:
    if raw is None or raw == "":
        return default
    if callable(raw) and not isinstance(raw, (int, float)):
        try:
            raw = raw()
        except TypeError:
            return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _dr_to_int(raw: Any, default: Optional[int] = None) -> Optional[int]:
    v = _dr_to_float(raw, None)
    if v is None:
        return default
    return int(round(v))


def _dr_merge_stat_blob(*blobs: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for blob in blobs:
        if not isinstance(blob, dict):
            continue
        for src_key, dest_key in (
            ("gp", "gp"),
            ("games", "gp"),
            ("games_played", "gp"),
            ("g", "goals"),
            ("goals", "goals"),
            ("a", "assists"),
            ("assists", "assists"),
            ("pts", "points"),
            ("points", "points"),
            ("ppg", "ppg"),
            ("points_per_game", "ppg"),
            ("shots", "shots"),
            ("sog", "shots"),
            ("toi_sec", "toi_sec"),
            ("toi", "toi_sec"),
            ("primary_points", "primary_points"),
            ("p1", "primary_points"),
            ("starts", "starts"),
            ("gs", "starts"),
            ("sv_pct", "save_pct"),
            ("save_pct", "save_pct"),
            ("save_percentage", "save_pct"),
            ("gaa", "gaa"),
            ("goals_against_avg", "gaa"),
            ("so", "shutouts"),
            ("shutouts", "shutouts"),
            ("league", "league"),
            ("league_name", "league"),
            ("league_code", "league"),
        ):
            if dest_key in out and out[dest_key] not in (None, "", 0, 0.0):
                continue
            if blob.get(src_key) is not None and blob.get(src_key) != "":
                out[dest_key] = blob.get(src_key)
    return out


def _dr_player_stat_blob(player: Any) -> Dict[str, Any]:
    if player is None:
        return {}
    blobs = [
        getattr(player, "season_stats", None),
        getattr(player, "stats", None),
        getattr(player, "last_season_stats", None),
        getattr(player, "prospect_stats", None),
    ]
    flat = {
        "gp": getattr(player, "gp", None) or getattr(player, "games_played", None),
        "goals": getattr(player, "goals", None) or getattr(player, "g", None),
        "assists": getattr(player, "assists", None) or getattr(player, "a", None),
        "points": getattr(player, "points", None) or getattr(player, "pts", None),
        "ppg": getattr(player, "ppg", None) or getattr(player, "points_per_game", None),
        "league": getattr(player, "current_league_id", None) or getattr(player, "league", None),
        "save_pct": getattr(player, "save_pct", None) or getattr(player, "sv_pct", None),
        "gaa": getattr(player, "gaa", None),
        "starts": getattr(player, "starts", None),
    }
    return _dr_merge_stat_blob(flat, *[b for b in blobs if isinstance(b, dict)])


def _dr_public_ability(pick: Dict[str, Any], board: Dict[str, Any], player: Any = None) -> float:
    for raw in (
        pick.get("nhl_readiness"),
        pick.get("floor_grade"),
        pick.get("current_ovr_estimate"),
        pick.get("scouted_overall_estimate"),
        board.get("nhl_readiness"),
        board.get("current_ovr_estimate"),
        board.get("scouted_overall_estimate"),
        board.get("floor_score"),
        getattr(player, "nhl_readiness", None) if player is not None else None,
    ):
        v = _dr_to_float(raw)
        if v is None:
            continue
        if v <= 1.5:
            v *= 99.0
        if v > 0:
            return v
    return 0.0


def _dr_public_ceiling(pick: Dict[str, Any], board: Dict[str, Any]) -> Optional[str]:
    for raw in (
        pick.get("potential_grade"),
        pick.get("ceiling_grade"),
        pick.get("talent_grade"),
        pick.get("scout_tier"),
        board.get("talent_grade"),
        board.get("potential_grade"),
        board.get("scout_tier"),
        board.get("ceiling_label"),
    ):
        if raw is None or raw == "":
            continue
        # Avoid leaking raw numeric hidden potential; allow letter grades / tiers.
        if isinstance(raw, (int, float)):
            v = float(raw)
            if v <= 1.5:
                continue
            if v >= 90:
                return "Elite ceiling"
            if v >= 84:
                return "High ceiling"
            if v >= 78:
                return "Upside starter"
            if v >= 72:
                return "Solid NHL tools"
            return "Developmental ceiling"
        s = str(raw).strip()
        if s.lower() in ("true", "false") or s.replace(".", "", 1).isdigit():
            continue
        return s
    return None


def _dr_public_floor(pick: Dict[str, Any], board: Dict[str, Any]) -> Optional[str]:
    for raw in (
        pick.get("floor_grade"),
        board.get("floor_label"),
        board.get("floor_score"),
        pick.get("current_ovr_estimate"),
    ):
        if raw is None or raw == "":
            continue
        if isinstance(raw, (int, float)):
            v = float(raw)
            if v <= 1.5:
                v *= 99.0
            if v >= 70:
                return "NHL-ready floor"
            if v >= 64:
                return "AHL / depth floor"
            if v >= 58:
                return "Org depth floor"
            return "Project floor"
        s = str(raw).strip()
        if s.replace(".", "", 1).isdigit():
            continue
        return s
    return None


def _dr_resolve_eta_years(pick: Dict[str, Any], board: Dict[str, Any], player: Any = None) -> Tuple[int, str]:
    """ETA years from public signals — never requires hidden true OVR."""
    overall = _dr_to_int(pick.get("overall_pick"), 99) or 99
    rank = _dr_to_int(pick.get("final_rank") or board.get("rank"), overall) or overall
    ability = _dr_public_ability(pick, board, player)
    age = _dr_to_int(pick.get("age") or board.get("age"), 18) or 18
    conf = _dr_confidence_label(pick.get("scouting_confidence") or board.get("scouting_confidence"))
    # Prefer stored draft ETA when present, then refine by ability/rank.
    stored = _dr_to_int(pick.get("nhl_eta") or board.get("nhl_eta") or board.get("eta_years"), None)
    years = stored if stored is not None else 4
    if ability >= 72 and rank <= 15:
        years = min(years, 1)
    elif ability >= 68 and rank <= 32:
        years = min(years, 2)
    elif ability >= 64 and overall <= 32:
        years = min(years, 3)
    elif ability > 0 and ability < 55 and overall > 64:
        years = max(years, 5)
    elif ability > 0 and ability < 58 and overall <= 32:
        years = max(years, 3)
    if age <= 17:
        years = max(years, 3)
    elif age >= 20 and ability >= 64:
        years = min(years, max(1, years - 1))
    # Top pick should not default to mid-range project without signals.
    if overall <= 10 and ability >= 60:
        years = min(years, 2)
    elif overall <= 10 and ability <= 0:
        years = min(years if stored is not None else 3, 3)
    years = max(0, min(int(years), 8))
    return years, conf


def _dr_scouting_summary(pick: Dict[str, Any], board: Dict[str, Any], selection: Dict[str, Any]) -> Dict[str, Any]:
    ceiling = _dr_public_ceiling(pick, board)
    floor = _dr_public_floor(pick, board)
    conf = selection.get("scouting_confidence_label") or _dr_confidence_label(
        pick.get("scouting_confidence") or board.get("scouting_confidence")
    )
    risk = str(selection.get("risk_level") or pick.get("risk_score") or board.get("risk") or "Medium")
    board_bits = []
    if selection.get("board_range"):
        board_bits.append(str(selection["board_range"]))
    if selection.get("selection_delta_label"):
        board_bits.append(str(selection["selection_delta_label"]))
    notes = []
    if pick.get("pick_reason"):
        notes.append(str(pick["pick_reason"]))
    snap = pick.get("board_snapshot") if isinstance(pick.get("board_snapshot"), dict) else {}
    if snap.get("pick_reasoning") and snap["pick_reasoning"] not in notes:
        notes.append(str(snap["pick_reasoning"]))
    if snap.get("stock_movement"):
        notes.append(f"Stock: {snap['stock_movement']}")
    if selection.get("selection_reason"):
        notes.append(str(selection["selection_reason"]))
    headline_parts = [p for p in (floor, ceiling) if p]
    if not headline_parts:
        headline = f"{selection.get('selection_verdict') or 'Scouting'} profile · {conf} confidence"
    else:
        headline = " · ".join(headline_parts[:2])
    return {
        "mode": "scouting",
        "headline": headline,
        "floor_label": floor,
        "ceiling_label": ceiling,
        "potential_label": ceiling,
        "scouting_confidence_label": conf,
        "risk_level": risk,
        "board_context": " · ".join(board_bits) if board_bits else None,
        "notes": notes[:3],
        "league_context": "Production sample unavailable — scouting profile shown",
        "data_confidence": conf,
    }


_DR_RIGHTS_STATUS_LABELS = {
    "exclusive_rights": "Exclusive rights",
    "college_rights": "College rights",
    "indefinite_european_rights": "Indefinite European rights",
    "drafted_unsigned": "Drafted, unsigned",
    "signed": "Signed",
    "rights_relinquished": "Rights relinquished",
    "unrestricted_free_agent": "Unrestricted free agent",
    "draft_reentry": "Draft re-entry",
}

_DR_ENV_GRADE_LABELS = {
    "ideal": "Ideal environment",
    "good": "Strong environment",
    "acceptable": "Acceptable environment",
    "risky": "Risky environment",
    "poor": "Poor environment",
}

_DR_CONFIDENCE_LABELS = (
    (75, "High"),
    (50, "Medium"),
    (0, "Low"),
)


def _dr_confidence_label(raw: Any) -> str:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return "Medium"
    if v <= 1.5:
        v *= 100.0
    for threshold, label in _DR_CONFIDENCE_LABELS:
        if v >= threshold:
            return label
    return "Low"


def _dr_pos_keys(pos: str) -> Tuple[str, ...]:
    bucket = str(pos or "").upper()
    if bucket in ("LW", "RW", "W"):
        return ("LW", "RW", "W")
    if bucket in ("D", "LD", "RD", "LHD", "RHD") or bucket.endswith("D"):
        return ("D", "LD", "RD", "LHD", "RHD")
    if bucket == "G":
        return ("G",)
    if bucket == "C":
        return ("C",)
    return (bucket,) if bucket else ()


def _dr_pos_group(pos: str) -> str:
    p = str(pos or "").upper()
    if p == "G":
        return "G"
    if p in ("D", "LD", "RD", "LHD", "RHD") or p.endswith("D"):
        return "D"
    return "F"


def _dr_player_pos(player: Any, fallback: str = "") -> str:
    if player is None:
        return str(fallback or "").upper()
    try:
        return _dev_player_position(player) or str(fallback or "").upper()
    except Exception:
        return str(getattr(player, "position", None) or fallback or "").upper()


def _dr_player_matches_pos(player: Any, pos: str) -> bool:
    keys = _dr_pos_keys(pos)
    if not keys:
        return False
    ppos = _dr_player_pos(player)
    if ppos in keys:
        return True
    sec = str(getattr(player, "secondary_position", "") or "").upper()
    return sec in keys


def _dr_eta_range(eta: Any, confidence: str = "Medium") -> Tuple[str, str]:
    try:
        e = int(eta)
    except (TypeError, ValueError):
        e = 4
    e = max(0, min(e, 8))
    bands = {
        0: "0–1 years",
        1: "0–2 years",
        2: "1–3 years",
        3: "2–4 years",
        4: "3–5 years",
        5: "4–6 years",
        6: "5–7 years",
    }
    label = bands.get(e, "5–8 years")
    return label, confidence or "Medium"


def _dr_pub_delta(pick: Dict[str, Any]) -> Optional[int]:
    d = pick.get("public_rank_delta")
    if d is None:
        d = pick.get("public_board_delta")
    if d is None:
        try:
            fr = int(pick.get("final_rank") or 0)
            op = int(pick.get("overall_pick") or 0)
            if fr > 0 and op > 0:
                d = op - fr
        except (TypeError, ValueError):
            d = None
    try:
        return int(d) if d is not None else None
    except (TypeError, ValueError):
        return None


def _dr_selection_review(pick: Dict[str, Any], needs: List[Dict[str, Any]]) -> Dict[str, Any]:
    label = str(pick.get("selection_label") or pick.get("pick_classification") or "").strip()
    conf_raw = pick.get("scouting_confidence")
    if conf_raw is None:
        conf_raw = (pick.get("board_snapshot") or {}).get("scouting_confidence")
    conf_label = _dr_confidence_label(conf_raw)
    delta = _dr_pub_delta(pick)
    risk = str(pick.get("risk_score") or pick.get("risk") or "Medium")
    need_cats = {str(n.get("category") or "") for n in (needs or [])[:4]}
    pos = str(pick.get("position") or "").upper()
    fills_need = any(
        (n in ("Franchise Center", "Center Depth") and pos == "C")
        or (n in ("Top-Six Winger", "Wing Depth") and pos in ("LW", "RW", "W"))
        or (n == "Goalie Pipeline" and pos == "G")
        or (n == "Right-Shot Defense" and pos in ("D", "RD", "RHD"))
        for n in need_cats
    )

    board_range = None
    snap = pick.get("board_snapshot") if isinstance(pick.get("board_snapshot"), dict) else {}
    cr = snap.get("consensus_range") or pick.get("consensus_range")
    if isinstance(cr, (list, tuple)) and len(cr) >= 2:
        board_range = f"#{cr[0]}–#{cr[1]}"
    elif snap.get("public_rank") is not None:
        board_range = f"Board #{snap.get('public_rank')}"
    elif pick.get("final_rank") is not None:
        board_range = f"Board #{pick.get('final_rank')}"

    if conf_label == "Low" and label in ("", "Off Board", "Expected"):
        return {
            "selection_grade": "C",
            "selection_grade_label": "Incomplete read",
            "selection_verdict": "Uncertain value",
            "selection_reason": "Limited scouting confidence leaves board value unclear.",
            "board_range": board_range,
            "selection_delta_label": "Board incomplete",
            "scouting_confidence": conf_raw,
            "scouting_confidence_label": conf_label,
            "risk_level": risk,
            "risk_reason": "Uncertainty from thin scouting coverage.",
        }

    if label == "Steal":
        grade, glabel, verdict = "A+", "Outstanding value", "Best value"
        reason = "Fell well past public consensus at selection."
    elif label == "Value":
        grade, glabel, verdict = "B+", "Strong value", "Good value"
        reason = "Available later than the public board expected."
    elif label == "Expected":
        grade, glabel, verdict = "B", "On the board", "Expected range"
        reason = "Taken inside the expected public consensus window."
    elif label == "Early" and fills_need:
        grade, glabel, verdict = "B-", "Need fill", "Need-based selection"
        reason = "Slightly early relative to the board, but fills an organizational need."
    elif label == "Early":
        grade, glabel, verdict = "B-", "Slight reach", "Aggressive projection"
        reason = "Selected ahead of public consensus on upside projection."
    elif label == "Reach":
        grade, glabel, verdict = "C+", "Reach", "Aggressive projection"
        reason = "Taken well ahead of public consensus ranking."
    elif label == "Off Board":
        grade, glabel, verdict = "C", "Off-board swing", "Long-term swing"
        reason = "Selected outside the published consensus board."
    elif fills_need:
        grade, glabel, verdict = "B", "Need fill", "Need-based selection"
        reason = "Addresses a documented organizational gap."
    else:
        grade, glabel, verdict = "B-", "Standard selection", "Expected range"
        reason = pick.get("pick_reason") or "Standard selection relative to available information."

    if delta is None:
        delta_label = "Board incomplete"
    elif delta >= 15:
        delta_label = f"Fell {delta} spots"
    elif delta >= 5:
        delta_label = f"Value of +{delta}"
    elif delta <= -15:
        delta_label = f"Reached {-delta} spots"
    elif delta <= -5:
        delta_label = f"Early by {-delta}"
    else:
        delta_label = "Near board rank"

    risk_reason = {
        "High": "Projection carries meaningful bust risk.",
        "Low": "Tools and path project with lower variance.",
        "Medium": "Balanced risk relative to draft slot.",
    }.get(risk, "Standard developmental risk.")

    return {
        "selection_grade": grade,
        "selection_grade_label": glabel,
        "selection_verdict": verdict,
        "selection_reason": reason,
        "board_range": board_range,
        "selection_delta_label": delta_label,
        "scouting_confidence": conf_raw,
        "scouting_confidence_label": conf_label,
        "risk_level": risk,
        "risk_reason": risk_reason,
    }


def _dr_production_snapshot(
    session: FranchiseSession,
    player: Any,
    pick: Dict[str, Any],
    board: Optional[Dict[str, Any]] = None,
    selection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    board = board or {}
    selection = selection or {}
    stats = _dr_merge_stat_blob(
        _dev_season_stats(session, player) if player is not None else {},
        _dr_player_stat_blob(player),
        pick.get("season_stats") if isinstance(pick.get("season_stats"), dict) else {},
        pick.get("stats") if isinstance(pick.get("stats"), dict) else {},
        board,
        {
            "gp": pick.get("gp") or pick.get("games") or pick.get("games_played"),
            "goals": pick.get("goals") or pick.get("g"),
            "assists": pick.get("assists") or pick.get("a"),
            "points": pick.get("points") or pick.get("pts"),
            "ppg": pick.get("ppg") or pick.get("points_per_game"),
            "league": pick.get("league") or pick.get("league_code"),
        },
    )
    league_raw = str(
        stats.get("league")
        or pick.get("league")
        or board.get("league")
        or board.get("league_code")
        or getattr(player, "current_league_id", "")
        or ""
    )
    league = _dr_league_label(league_raw) or league_raw or None
    year = int(getattr(session, "season_calendar_year", 0) or 0)
    season_label = f"{year}–{str(year + 1)[2:]}" if year else None
    gp = _dr_to_int(stats.get("gp"), 0) or 0
    conf = "High" if gp >= 20 else ("Medium" if gp >= 5 else "Low")
    is_g = (
        _dev_is_goalie(player)
        if player is not None
        else str(pick.get("position") or board.get("position") or "").upper() == "G"
    )

    if gp <= 0:
        scout = _dr_scouting_summary(pick, board, selection)
        scout["season"] = season_label
        scout["league"] = league
        scout["club"] = pick.get("club") or board.get("club") or board.get("team_name") or board.get("team")
        return scout

    if is_g:
        starts = _dr_to_int(stats.get("starts"), 0) or 0
        out: Dict[str, Any] = {
            "mode": "stats",
            "season": season_label,
            "league": league,
            "games": gp,
            "starts": starts or None,
            "save_percentage": stats.get("save_pct"),
            "goals_against_average": stats.get("gaa"),
            "shutouts": stats.get("shutouts"),
            "workload": starts or gp,
            "data_confidence": conf,
            "floor_label": _dr_public_floor(pick, board),
            "ceiling_label": _dr_public_ceiling(pick, board),
            "potential_label": _dr_public_ceiling(pick, board),
        }
        if stats.get("save_pct") is not None:
            sv = float(stats["save_pct"])
            sv = sv if sv > 1.5 else sv * 100.0
            if sv >= 91.5:
                out["league_context"] = "High-workload starter production"
            elif sv >= 90.0:
                out["league_context"] = "Solid starting goalie workload"
            else:
                out["league_context"] = "Developmental starter minutes"
            out["production_trend"] = "Season body of work"
        else:
            out["league_context"] = f"{league or 'League'} workload tracked"
            out["production_trend"] = "Partial goalie sample"
        return {k: v for k, v in out.items() if v is not None}

    goals = _dr_to_int(stats.get("goals"), 0) or 0
    assists = _dr_to_int(stats.get("assists"), 0) or 0
    points = _dr_to_int(stats.get("points"), goals + assists) or (goals + assists)
    ppg = _dr_to_float(stats.get("ppg"))
    if ppg is None and gp > 0:
        ppg = round(points / gp, 2)
    toi = None
    if stats.get("toi_sec"):
        try:
            toi = round(float(stats["toi_sec"]) / max(1, gp) / 60.0, 1)
        except (TypeError, ValueError):
            toi = None
    out = {
        "mode": "stats",
        "season": season_label,
        "league": league,
        "games": gp,
        "goals": goals,
        "assists": assists,
        "points": points,
        "points_per_game": ppg,
        "primary_points": stats.get("primary_points"),
        "shots": stats.get("shots"),
        "toi_per_game": toi,
        "data_confidence": conf,
        "floor_label": _dr_public_floor(pick, board),
        "ceiling_label": _dr_public_ceiling(pick, board),
        "potential_label": _dr_public_ceiling(pick, board),
    }
    if gp >= 20 and ppg is not None:
        if float(ppg) >= 1.2:
            out["league_context"] = "Top-line junior production"
        elif float(ppg) >= 0.85:
            out["league_context"] = "Top-six scoring pace"
        elif float(ppg) >= 0.55:
            out["league_context"] = "Middle-six production"
        else:
            out["league_context"] = "Bottom-six / sheltered offence"
        out["production_trend"] = "Stable season body of work"
    else:
        out["league_context"] = f"{league or 'League'} sample · {gp} GP"
        out["production_trend"] = "Limited sample"
    return {k: v for k, v in out.items() if v is not None}


def _dr_org_depth_counts(
    team: Any,
    pos: str,
    *,
    exclude_id: str = "",
) -> Dict[str, Any]:
    nhl = list(getattr(team, "roster", None) or []) if team else []
    ahl = list(getattr(team, "ahl_roster", None) or []) if team else []
    pool = list(getattr(team, "prospect_pool", None) or getattr(team, "prospects", None) or []) if team else []

    def _id(p: Any) -> str:
        return str(getattr(p, "player_id", None) or getattr(p, "id", "") or "")

    def _filt(players: List[Any]) -> List[Any]:
        out = []
        for p in players:
            if exclude_id and _id(p) == exclude_id:
                continue
            if _dr_player_matches_pos(p, pos):
                out.append(p)
        return out

    nhl_m = _filt(nhl)
    ahl_m = _filt(ahl)
    pool_m = _filt(pool)
    # Ready-ish NHL blockers: prefer higher OVR / readiness.
    blockers = []
    for p in sorted(nhl_m, key=lambda x: -_safe_attr_float(x, "overall", "ovr", "nhl_readiness"))[:3]:
        blockers.append({
            "name": getattr(getattr(p, "identity", None), "name", None) or getattr(p, "name", None),
            "ovr": _safe_attr_float(p, "overall", "ovr") or None,
        })
    return {
        "nhl_ahead": len(nhl_m),
        "ahl_ahead": len(ahl_m),
        "prospects_ahead": len(pool_m),
        "blockers": blockers,
        "nhl_players": nhl_m,
        "ahl_players": ahl_m,
        "prospect_players": pool_m,
    }


def _dr_organizational_fit(
    team: Any,
    pick: Dict[str, Any],
    player: Any,
    needs: List[Dict[str, Any]],
    env: Dict[str, Any],
) -> Dict[str, Any]:
    pos = str(pick.get("position") or _dr_player_pos(player) or "")
    pid = str(pick.get("prospect_id") or "")
    depth = _dr_org_depth_counts(team, pos, exclude_id=pid)
    nhl_n = depth["nhl_ahead"]
    ahl_n = depth["ahl_ahead"]
    pool_n = depth["prospects_ahead"]
    total_ahead = nhl_n + ahl_n + max(0, pool_n)

    if total_ahead <= 1:
        congestion = "Low"
        depth_status = f"Thin at {pos or 'position'}"
        fit_grade, fit_label = "A-", "Clear organizational need"
    elif total_ahead <= 3:
        congestion = "Moderate"
        depth_status = f"Developing depth at {pos or 'position'}"
        fit_grade, fit_label = "B", "Useful pipeline addition"
    elif total_ahead <= 5:
        congestion = "Elevated"
        depth_status = f"Crowded path at {pos or 'position'}"
        fit_grade, fit_label = "C+", "Competitive depth battle"
    else:
        congestion = "High"
        depth_status = f"Heavy congestion at {pos or 'position'}"
        fit_grade, fit_label = "C", "Longer wait for opportunity"

    need_cats = [str(n.get("category") or "") for n in (needs or [])[:4]]
    need_filled = None
    if pos == "C" and any("Center" in n for n in need_cats):
        need_filled = "Center depth"
    elif pos in ("LW", "RW", "W") and any("Wing" in n for n in need_cats):
        need_filled = f"{'Right wing' if pos == 'RW' else 'Left wing' if pos == 'LW' else 'Wing'} scoring depth"
    elif pos == "G" and any("Goalie" in n for n in need_cats):
        need_filled = "Goalie pipeline"
    elif _dr_pos_group(pos) == "D" and any("Defense" in n or "Right-Shot" in n for n in need_cats):
        need_filled = "Puck-moving defence" if "Right-Shot" in "".join(need_cats) else "Defence depth"
    elif total_ahead <= 2:
        group = _dr_pos_group(pos)
        need_filled = {
            "F": "Forward pipeline depth",
            "D": "Defence pipeline depth",
            "G": "Goalie pipeline depth",
        }.get(group, "Organizational depth")
    else:
        need_filled = "General organizational depth"

    if any("Center" in n or "Wing" in n or "Goalie" in n or "Defense" in n or "Right-Shot" in n for n in need_cats):
        if congestion == "Low":
            fit_grade, fit_label = "A-", "Clear organizational need"

    env_grade = str((env or {}).get("grade") or "acceptable")
    env_reason = ((env or {}).get("reasons") or ["Standard developmental placement"])[0]
    opportunities = []
    if pool_n == 0:
        opportunities.append(f"No established {pos or 'position'} prospect ahead")
    if nhl_n <= 2:
        opportunities.append("NHL depth remains thin")
    if not opportunities:
        opportunities.append("Earn role through production")

    blockers_txt = []
    if nhl_n >= 3:
        blockers_txt.append("Multiple NHL players ahead")
    if ahl_n >= 2:
        blockers_txt.append("AHL depth already set")
    age = pick.get("age")
    if age is not None and int(age or 99) <= 18:
        blockers_txt.append("Needs another junior or amateur season")
    if not blockers_txt:
        blockers_txt.append("Development timeline")

    pipeline_rank = max(1, pool_n + 1)
    if pipeline_rank == 1:
        pipeline_label = "Top prospect at position"
    elif pipeline_rank == 2:
        pipeline_label = "Second in positional pipeline"
    elif congestion == "High":
        pipeline_label = f"#{pipeline_rank} — long wait likely"
    else:
        pipeline_label = f"#{pipeline_rank} in positional pipeline"

    env_conflict = None
    if congestion in ("Elevated", "High") and env_grade in ("ideal", "good"):
        env_conflict = (
            f"{_DR_ENV_GRADE_LABELS.get(env_grade, 'Strong environment')} reflects role/ice time fit; "
            f"{congestion.lower()} congestion means NHL opportunity still waits behind depth."
        )
    elif congestion == "Low" and env_grade in ("risky", "poor"):
        env_conflict = (
            "Path is open organizationally, but the current development setting is a concern."
        )

    return {
        "fit_grade": fit_grade,
        "fit_label": fit_label,
        "depth_status": depth_status,
        "nhl_players_ahead": nhl_n,
        "ahl_players_ahead": ahl_n,
        "prospects_ahead": pool_n,
        "path_congestion": congestion,
        "need_filled": need_filled,
        "expected_pipeline_rank": pipeline_rank,
        "pipeline_label": pipeline_label,
        "blockers": blockers_txt[:3],
        "opportunities": opportunities[:3],
        "environment_grade": _DR_ENV_GRADE_LABELS.get(env_grade, env_grade.title()),
        "environment_reason": env_reason,
        "fit_tension_note": env_conflict,
        "depth_at_position": {
            "count": nhl_n,
            "blockers": depth["blockers"],
        },
    }


def _dr_development_plan(
    pick: Dict[str, Any],
    player: Any,
    card: Dict[str, Any],
    production: Dict[str, Any],
    fit: Dict[str, Any],
    board: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    board = board or {}
    path = str(
        card.get("returning_to")
        or pick.get("development_path")
        or getattr(player, "development_path", "")
        or ""
    ).upper()
    league_raw = str(
        production.get("league")
        or pick.get("league")
        or board.get("league")
        or board.get("league_code")
        or card.get("current_league_id")
        or ""
    )
    league = _dr_league_label(league_raw) or _dr_league_label(path) or league_raw
    club = (
        pick.get("club")
        or board.get("club")
        or board.get("team_name")
        or board.get("team")
        or getattr(player, "current_team_name", None)
        or getattr(player, "junior_team", None)
    )
    age = int(pick.get("age") or board.get("age") or (_dev_player_age(player) if player is not None else 18) or 18)
    pos = str(pick.get("position") or board.get("position") or _dr_player_pos(player) or "F").upper()
    is_g = pos == "G"
    overall = _dr_to_int(pick.get("overall_pick"), 99) or 99
    readiness = _dr_public_ability(pick, board, player)
    ahl_eligible = any(
        a.get("id") == "assign_ahl" and a.get("enabled")
        for a in (card.get("available_actions") or [])
    )
    if age >= 20 and ("EUROPE" in path or "NCAA" not in path):
        ahl_eligible = ahl_eligible or age >= 20
    elc_slide = bool(
        card.get("elc_slide_eligible")
        if card.get("elc_slide_eligible") is not None
        else pick.get("can_slide")
    )
    ppg = float(production.get("points_per_game") or 0)
    gp = int(production.get("games") or 0)
    eta_years, eta_conf = _dr_resolve_eta_years(pick, board, player)
    eta_range, _ = _dr_eta_range(eta_years, eta_conf)

    # Draft-slot aware role bands (scouted ability can still override downward).
    if overall <= 10:
        jr_role = f"First-line {pos}" if pos in ("C", "LW", "RW") else ("Top-pair defence" if not is_g else "Franchise junior starter")
        jr_minutes = "19–22 minutes" if not is_g else "Workhorse starter"
        jr_st = "PP1 / primary unit" if not is_g else "None"
        nhl_proj = (
            "Top-six scoring winger" if pos in ("LW", "RW")
            else ("1C / 2C driver" if pos == "C" else ("Top-four defence" if not is_g else "NHL starter trajectory"))
        )
        nhl_conf = "Medium" if readiness >= 60 else "Low-Medium"
        nhl_reason = f"Pick #{overall} carries featured-role expectation; timeline depends on physical maturity and production."
        obj = "Dominate age-group minutes and drive offence as a primary option"
    elif overall <= 32:
        jr_role = f"Top-six {pos}" if pos in ("C", "LW", "RW") else ("Top-four defence" if not is_g else "Junior starter")
        jr_minutes = "17–20 minutes" if not is_g else "Primary starter"
        jr_st = "PP1 / PP2" if not is_g else "None"
        nhl_proj = (
            "Middle-six / top-nine winger" if pos in ("LW", "RW")
            else ("Middle-six centre" if pos == "C" else ("NHL defence depth / 2nd pair upside" if not is_g else "NHL depth / tandem upside"))
        )
        nhl_conf = "Medium"
        nhl_reason = f"First-round capital supports a regular NHL role if development holds."
        obj = "Produce in a top-six junior role without sheltered usage"
    elif overall <= 96:
        jr_role = f"Top-six {pos}" if pos != "D" and not is_g else ("Second-pair defence" if not is_g else "Split starter")
        if gp >= 20 and ppg >= 1.0 and not is_g:
            jr_role = f"First-line {pos}" if pos != "D" else "Top-pair defence"
        jr_minutes = "16–19 minutes" if not is_g else "Shared starter"
        jr_st = "Power-play contributor" if not is_g else "None"
        nhl_proj = "NHL depth / call-up contributor" if not is_g else "Organizational goalie depth"
        nhl_conf = "Low-Medium"
        nhl_reason = "Mid-round projection; role clarity comes from sustained production."
        obj = "Earn harder minutes through production and two-way reliability"
    else:
        jr_role = f"Middle-six {pos}" if not is_g and pos != "D" else ("Second-pair defence" if not is_g else "Developmental starter")
        jr_minutes = "14–17 minutes" if not is_g else "Backup / developmental starts"
        jr_st = "Secondary special teams" if not is_g else "None"
        nhl_proj = "Long-term org depth" if not is_g else "Long-term goalie depth"
        nhl_conf = "Low"
        nhl_reason = "Later-round variance stays high until the tools translate."
        obj = "Raise tools and earn trust in tougher minutes"

    # Production can upgrade mid/late roles.
    if not is_g and gp >= 20 and ppg >= 1.15 and overall > 10:
        jr_role = f"First-line {pos}" if pos != "D" else "Top-pair defence"
        jr_minutes = "18–21 minutes"
        jr_st = "First power-play unit"
        obj = "Drive offence as a primary option"
    elif not is_g and gp >= 20 and ppg >= 0.85 and overall > 32:
        jr_role = f"Top-six {pos}" if pos != "D" else "Top-four defence"
        obj = "Sustain top-six production against better competition"

    if readiness >= 72 and age >= 19 and ahl_eligible:
        next_dest, next_label = "NHL", "Compete for NHL roster"
        club_line = None
        role = "Bottom-six audition" if not is_g else "Camp / third-goalie look"
        minutes = "Limited NHL minutes" if not is_g else "Practice / emergency look"
        st = "Situational" if not is_g else "None"
        obj = "Earn a roster foothold without forced ice time"
        alt = "AHL featured role if NHL minutes are unavailable"
        steps = [
            {"stage": "NHL camp", "status": "next", "detail": role},
            {"stage": "AHL", "status": "future", "detail": "Featured fallback"},
            {"stage": "NHL", "status": "projection", "detail": nhl_proj},
        ]
    elif "NCAA" in path or "NCAA" in str(league_raw).upper() or league == "NCAA":
        next_dest = "NCAA"
        next_label = f"Remain at {club}" if club else "Remain in NCAA"
        club_line = club
        role = "Top-six college forward" if not is_g else "College starter"
        if _dr_pos_group(pos) == "D":
            role = "Top-pair college defence"
        if overall <= 32:
            role = "Featured college role"
        minutes = "18–22 minutes" if not is_g else "Starting workload"
        st = "First power-play unit" if not is_g else "None"
        obj = "Lead at the college level and refine pro tools"
        alt = "Sign and join AHL after the eligibility window"
        steps = [
            {"stage": "NCAA", "status": "next", "detail": role},
            {"stage": "AHL", "status": "future", "detail": "Pro introduction"},
            {"stage": "NHL", "status": "projection", "detail": nhl_proj},
        ]
    elif "EUROPE" in path or any(
        x in str(league_raw).upper() for x in ("SHL", "LIIGA", "DEL", "KHL", "CZE", "SUI", "SVK", "ALLSV")
    ) or league in ("SHL", "Liiga", "DEL", "KHL", "Czech Extraliga", "NL", "Allsvenskan"):
        if ahl_eligible and age >= 20 and readiness >= 64:
            next_dest, next_label = "AHL", "Sign and join AHL"
            club_line = None
            role = "AHL top nine" if not is_g else ("AHL starter" if readiness >= 66 else "AHL backup")
            minutes = "16–19 minutes" if not is_g else "Shared starter workload"
            st = "Second power-play look" if not is_g else "None"
            obj = "Transition the North American game"
            alt = f"Remain with {club}" if club else "Remain in Europe another season"
            steps = [
                {"stage": "AHL", "status": "next", "detail": role},
                {"stage": "NHL", "status": "future", "detail": "Depth introduction"},
                {"stage": "NHL", "status": "projection", "detail": nhl_proj},
            ]
        else:
            next_dest = league or "Europe"
            next_label = f"Remain with {club}" if club else f"Remain in {league or 'Europe'}"
            club_line = club
            role = "European pro role" if not is_g else "European starting goalie"
            if overall <= 32:
                role = "Featured European minutes"
            minutes = "Featured pro minutes" if not is_g else "Starter / 1B"
            st = "Power-play usage" if not is_g else "None"
            obj = "Produce against men and add strength"
            alt = "Sign and join AHL when ready"
            steps = [
                {"stage": league or "Europe", "status": "next", "detail": role},
                {"stage": "AHL", "status": "future", "detail": "NA transition"},
                {"stage": "NHL", "status": "projection", "detail": nhl_proj},
            ]
    elif ahl_eligible and age >= 20 and readiness >= 60:
        next_dest, next_label = "AHL", "AHL featured role" if not is_g else ("AHL starter" if readiness >= 64 else "AHL backup")
        club_line = None
        role = "AHL top nine" if not is_g else next_label
        minutes = "16–19 minutes" if not is_g else "Shared starter workload"
        st = "Second power-play look" if not is_g else "None"
        obj = "Dominate AHL minutes before the NHL push"
        alt = "Challenge for NHL depth if production spikes"
        steps = [
            {"stage": "AHL", "status": "next", "detail": role},
            {"stage": "NHL", "status": "future", "detail": "Depth introduction"},
            {"stage": "NHL", "status": "projection", "detail": nhl_proj},
        ]
    else:
        # Junior / CHL / USHL — use specific league/club, never raw enum.
        next_dest = league or "Junior"
        if club and league:
            next_label = f"Return to {club}"
        elif league:
            next_label = f"Return to {league}"
        elif club:
            next_label = f"Return to {club}"
        else:
            next_label = "Return to junior club"
        club_line = club
        if is_g:
            role = "Junior starting goalie" if overall <= 64 else "Developmental starter"
            minutes = "Primary starter workload"
            st = "None"
            obj = "Own the crease and raise save percentage"
            alt = "AHL backup look after junior season if eligible"
            steps = [
                {"stage": league or "Junior", "status": "next", "detail": role},
                {"stage": "AHL", "status": "future", "detail": "Pro crease introduction"},
                {"stage": "NHL", "status": "projection", "detail": nhl_proj},
            ]
        else:
            role, minutes, st = jr_role, jr_minutes, jr_st
            alt = (
                f"AHL challenge after {league} season"
                if age >= 19 and league
                else (f"Another {league} season if still eligible" if league else "Reassess after next season")
            )
            steps = [
                {"stage": league or "Junior", "status": "next", "detail": role},
                {"stage": "AHL", "status": "future", "detail": "Pro introduction"},
                {"stage": "NHL", "status": "projection", "detail": nhl_proj},
            ]

    secondary = "Improve defensive-zone exits" if _dr_pos_group(pos) != "G" else "Improve rebound control and composure"
    if fit.get("path_congestion") in ("Elevated", "High"):
        secondary = "Outproduce depth competitors for the next role"
    if overall <= 10:
        secondary = "Add strength and pace for an NHL top-six translation"

    return {
        "next_destination": next_dest,
        "next_destination_label": next_label,
        "next_club": club_line,
        "recommended_role": role,
        "minutes_target": minutes,
        "special_teams_role": st,
        "season_objective": obj,
        "secondary_objective": secondary,
        "alternate_path": alt,
        "path_steps": steps,
        "eta_range": eta_range,
        "eta_years": eta_years,
        "eta_confidence": eta_conf,
        "nhl_projection": nhl_proj,
        "nhl_projection_confidence": nhl_conf,
        "nhl_projection_reason": nhl_reason,
        "ahl_eligible": bool(ahl_eligible),
        "elc_can_slide": bool(elc_slide),
    }


def _dr_rights_preview(card: Dict[str, Any], plan: Dict[str, Any], pick: Dict[str, Any]) -> Dict[str, Any]:
    status = str(card.get("rights_status") or "exclusive_rights")
    expiry = card.get("rights_through") or card.get("rights_expiry_year")
    years_rem = None
    try:
        if expiry is not None and pick.get("draft_year") is not None:
            years_rem = max(0, int(expiry) - int(pick.get("draft_year")))
        elif expiry is not None:
            years_rem = None
    except (TypeError, ValueError):
        years_rem = None
    elc_slide = bool(plan.get("elc_can_slide") if plan.get("elc_can_slide") is not None else card.get("elc_slide_eligible"))
    next_label = str(plan.get("next_destination_label") or "")
    readiness = float(pick.get("nhl_readiness") or 0)
    if readiness and readiness <= 1.5:
        readiness *= 99.0

    if readiness >= 70 and plan.get("next_destination") == "NHL":
        rec, reason = "Sign", "Near-ready; ELC opens a pro path immediately."
        contract_now = True
    elif plan.get("next_destination") in ("AHL",) or "AHL" in next_label:
        rec, reason = "Sign", "AHL assignment requires an ELC."
        contract_now = True
    elif "NCAA" in str(plan.get("next_destination") or "").upper() or "college" in next_label.lower():
        rec, reason = "Wait", "Keep him in college; signing is not required now."
        contract_now = False
    elif any(x in next_label for x in ("Europe", "SHL", "Liiga", "DEL", "KHL")) or "EUROPE" in str(plan.get("next_destination") or "").upper():
        rec, reason = "Wait", "Leave him in Europe until a North American push is clear."
        contract_now = False
    elif plan.get("elc_can_slide"):
        dest = plan.get("next_club") or plan.get("next_destination") or "his development league"
        rec, reason = "Wait", f"Send him back to {dest} and preserve the ELC slide."
        contract_now = False
    else:
        rec, reason = "Wait", "No immediate contract pressure; review at Prospect Rights."
        contract_now = False

    deadline = f"Rights expire in {expiry}" if expiry is not None else "Rights window open"
    return {
        "rights_status": status,
        "rights_status_label": _DR_RIGHTS_STATUS_LABELS.get(status, status.replace("_", " ").title()),
        "rights_years_remaining": years_rem,
        "rights_deadline_label": deadline,
        "rights_through": expiry,
        "signing_recommendation": rec,
        "signing_reason": reason,
        "elc_can_slide": elc_slide,
        "contract_required_now": contract_now,
        "entry_level_contract_eligible": card.get("entry_level_contract_eligible", True),
        "recommended_action": card.get("recommended_action"),
        "recommended_label": card.get("recommended_label"),
        "available_actions": card.get("available_actions") or [],
        "elc_slide_eligible": card.get("elc_slide_eligible"),
        "development_environment": card.get("development_environment"),
        "eta": card.get("eta"),
        "path_visual": [s.get("stage") for s in (plan.get("path_steps") or [])],
        "returning_to": card.get("returning_to"),
    }


def _dr_review_line(
    pick: Dict[str, Any],
    selection: Dict[str, Any],
    plan: Dict[str, Any],
    fit: Dict[str, Any],
) -> str:
    name_pos = str(pick.get("position") or "prospect")
    archetype = str(pick.get("archetype") or pick.get("player_type") or "").strip()
    who = f"A {archetype.lower()} {name_pos}" if archetype else f"A {name_pos}"
    verdict = str(selection.get("selection_verdict") or "Expected range").lower()
    dest = str(plan.get("next_destination_label") or "development path")
    line = f"{who} drafted with {verdict} and a clear path to {dest.lower()}."
    words = line.split()
    if len(words) > 22:
        line = " ".join(words[:22])
    if fit.get("fit_label") and "need" in str(fit.get("fit_label")).lower():
        alt = f"{who} fills a real need with {verdict} at the pick."
        if len(alt.split()) <= 22:
            line = alt
    return line[0].upper() + line[1:] if line else "Selection under organizational review."


def _dr_identity_fields(
    pick: Dict[str, Any],
    player: Any,
    board_entry: Dict[str, Any],
) -> Dict[str, Any]:
    ident = getattr(player, "identity", None) if player is not None else None
    age = pick.get("age")
    if age is None:
        age = board_entry.get("age")
    if age is None and player is not None:
        age = _dev_player_age(player) or None
    shoots = (
        pick.get("shoots")
        or board_entry.get("shoots")
        or board_entry.get("handedness")
        or (getattr(ident, "shoots", None) if ident else None)
        or getattr(player, "shoots", None)
    )
    height = (
        pick.get("height")
        or board_entry.get("height")
        or board_entry.get("height_display")
        or getattr(player, "height", None)
        or (getattr(ident, "height", None) if ident else None)
    )
    weight = (
        pick.get("weight")
        or board_entry.get("weight")
        or getattr(player, "weight", None)
        or (getattr(ident, "weight", None) if ident else None)
    )
    club = (
        pick.get("club")
        or pick.get("team_name_junior")
        or board_entry.get("club")
        or board_entry.get("team")
        or getattr(player, "current_team_name", None)
        or getattr(player, "junior_team", None)
    )
    league = pick.get("league") or board_entry.get("league") or board_entry.get("league_name")
    nationality = pick.get("nationality") or board_entry.get("nationality") or getattr(player, "nationality", None)
    if nationality is None and ident is not None:
        nationality = getattr(ident, "nationality", None) or getattr(ident, "nation", None)
    secondary = (
        pick.get("secondary_position")
        or board_entry.get("secondary_position")
        or getattr(player, "secondary_position", None)
    )
    archetype = (
        pick.get("archetype")
        or pick.get("player_type")
        or board_entry.get("archetype")
        or board_entry.get("player_type")
        or getattr(player, "archetype", None)
        or getattr(player, "player_type", None)
    )
    return {
        "age": age,
        "shoots": shoots,
        "height": height,
        "weight": weight,
        "club": club,
        "league": league,
        "nationality": nationality,
        "secondary_position": secondary,
        "archetype": archetype,
    }


def _dr_pick_score(selection: Dict[str, Any], fit: Dict[str, Any], pick: Dict[str, Any]) -> float:
    grade_map = {"A+": 96, "A": 92, "A-": 88, "B+": 84, "B": 78, "B-": 72, "C+": 66, "C": 60, "C-": 54, "D": 45}
    score = float(grade_map.get(str(selection.get("selection_grade") or "B"), 75))
    fit_map = {"A-": 8, "A": 10, "B": 4, "C+": 0, "C": -4}
    score += float(fit_map.get(str(fit.get("fit_grade") or "B"), 2))
    conf = str(selection.get("scouting_confidence_label") or "")
    if conf == "High":
        score += 3
    elif conf == "Low":
        score -= 2
    # Do not punish late rounds for uncertainty alone.
    try:
        rnd = int(pick.get("round") or 1)
        if rnd >= 5 and str(selection.get("selection_verdict")) == "Uncertain value":
            score += 4
    except (TypeError, ValueError):
        pass
    return score


def _dr_letter_from_score(score: float) -> Tuple[str, str]:
    if score >= 90:
        return "A", "Excellent haul"
    if score >= 85:
        return "A-", "Very strong haul"
    if score >= 80:
        return "B+", "Strong value"
    if score >= 75:
        return "B", "Solid haul"
    if score >= 70:
        return "B-", "Mixed value"
    if score >= 64:
        return "C+", "Uneven haul"
    return "C", "Developmental haul"


def _dr_haul_summary(
    enriched: List[Dict[str, Any]],
    needs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n = len(enriched)
    mix = {"F": 0, "D": 0, "G": 0}
    for p in enriched:
        g = _dr_pos_group(str(p.get("position") or ""))
        mix[g] = mix.get(g, 0) + 1

    scores = []
    for p in enriched:
        scores.append(_dr_pick_score(
            {
                "selection_grade": p.get("selection_grade"),
                "scouting_confidence_label": p.get("scouting_confidence_label"),
                "selection_verdict": p.get("selection_verdict"),
            },
            p.get("organizational_fit") or {},
            p,
        ))
    avg = sum(scores) / max(1, len(scores)) if scores else 70.0
    grade, grade_label = _dr_letter_from_score(avg)

    steals = sum(1 for p in enriched if p.get("was_steal") or p.get("selection_verdict") == "Best value")
    values = sum(1 for p in enriched if p.get("was_value") or p.get("selection_verdict") == "Good value")
    reaches = sum(1 for p in enriched if p.get("was_reach") or p.get("selection_verdict") == "Aggressive projection")
    expected = sum(1 for p in enriched if p.get("selection_verdict") == "Expected range")
    high_risk = sum(1 for p in enriched if str(p.get("risk_level") or p.get("risk_score")) == "High")
    low_risk = sum(1 for p in enriched if str(p.get("risk_level") or p.get("risk_score")) == "Low")
    near_ready = sum(
        1 for p in enriched
        if int((p.get("development_plan") or {}).get("eta_years") if (p.get("development_plan") or {}).get("eta_years") is not None else p.get("nhl_eta") or 99) <= 2
    )
    long_term = sum(
        1 for p in enriched
        if int((p.get("development_plan") or {}).get("eta_years") if (p.get("development_plan") or {}).get("eta_years") is not None else p.get("nhl_eta") or 0) >= 4
    )

    needs_addressed: List[str] = []
    for p in enriched:
        nf = (p.get("organizational_fit") or {}).get("need_filled")
        if nf and nf not in needs_addressed and "General" not in str(nf):
            needs_addressed.append(str(nf))
    for n_item in needs[:3]:
        cat = str(n_item.get("category") or "")
        if cat and cat not in needs_addressed and any(
            (cat == "Wing Depth" and str(p.get("position")) in ("LW", "RW", "W"))
            or (cat == "Center Depth" and str(p.get("position")) == "C")
            or (cat == "Goalie Pipeline" and str(p.get("position")) == "G")
            or ("Defense" in cat and _dr_pos_group(str(p.get("position") or "")) == "D")
            for p in enriched
        ):
            needs_addressed.append(cat)
    needs_addressed = needs_addressed[:4]

    reason_bits = []
    if reaches >= 2:
        reason_bits.append(f"{reaches} aggressive early picks")
    if steals + values == 0 and n:
        reason_bits.append("few clear value wins vs the public board")
    if high_risk >= max(2, n // 2):
        reason_bits.append(f"{high_risk} high-risk projections")
    if long_term >= max(3, n - 1) and n:
        reason_bits.append("most picks are long-term projects")
    if not needs_addressed and n:
        reason_bits.append("limited direct need fills")
    if steals >= 1:
        reason_bits.append(f"{steals} steal-level value hit{'s' if steals != 1 else ''}")
    if not reason_bits:
        reason_bits.append("balanced mix of board-range selections")
    grade_reason = f"{grade_label} because of " + ", ".join(reason_bits[:3]) + "."

    def _value_score(p: Dict[str, Any]) -> float:
        verdict = {
            "Best value": 50, "Good value": 35, "Need-based selection": 22,
            "Expected range": 12, "Aggressive projection": 4, "Long-term swing": 8, "Uncertain value": 0,
        }.get(str(p.get("selection_verdict")), 10)
        delta = _dr_pub_delta(p) or 0
        return verdict + max(-20, min(30, float(delta)))

    def _close_score(p: Dict[str, Any]) -> float:
        plan = p.get("development_plan") or {}
        years = plan.get("eta_years")
        if years is None:
            years = p.get("nhl_eta")
        years = int(years if years is not None else 99)
        ready = _dr_public_ability(p, {}, None)
        # Lower is closer; subtract readiness so near-ready high picks win.
        return float(years) * 10.0 - ready

    best_value = max(enriched, key=_value_score) if enriched else None
    closest = min(enriched, key=_close_score) if enriched else None
    # Prefer distinct highlight players when possible.
    if (
        best_value and closest
        and str(best_value.get("prospect_id")) == str(closest.get("prospect_id"))
        and len(enriched) > 1
    ):
        others = [p for p in enriched if str(p.get("prospect_id")) != str(best_value.get("prospect_id"))]
        closest = min(others, key=_close_score) if others else closest
    largest = max(
        enriched,
        key=lambda p: int((p.get("development_plan") or {}).get("eta_years") if (p.get("development_plan") or {}).get("eta_years") is not None else p.get("nhl_eta") or 0),
        default=None,
    ) if enriched else None
    highest_conf = max(
        enriched,
        key=lambda p: float(p.get("scouting_confidence") or 0),
        default=None,
    ) if enriched else None

    balance = f"{mix.get('F', 0)}F / {mix.get('D', 0)}D / {mix.get('G', 0)}G"
    bits = []
    if n:
        bits.append(f"{n} selection{'s' if n != 1 else ''}")
    if needs_addressed:
        bits.append(f"addressed {needs_addressed[0].lower()}")
    if long_term:
        bits.append(f"{long_term} four-year-plus project{'s' if long_term != 1 else ''}")
    if mix.get("G"):
        bits.append("added goaltending")
    summary_line = (", ".join(bits).capitalize() + ".") if bits else "Draft class under review."

    def _pid(p: Optional[Dict[str, Any]]) -> Optional[str]:
        return str(p.get("prospect_id")) if p and p.get("prospect_id") else None

    return {
        "total_picks": n,
        "position_mix": mix,
        "position_balance_label": balance,
        "haul_grade": grade,
        "haul_grade_label": grade_label,
        "haul_grade_reason": grade_reason,
        "needs_addressed": needs_addressed,
        "best_value_pick_id": _pid(best_value),
        "closest_to_nhl_pick_id": _pid(closest),
        "largest_project_pick_id": _pid(largest),
        "highest_confidence_pick_id": _pid(highest_conf),
        "summary_line": summary_line,
        "long_term_projects": long_term,
        "long_term_label": f"{long_term} projects (4+ years)" if long_term else "No 4+ year projects",
        "near_ready_count": near_ready,
        "steals": steals,
        "value_picks": values,
        "reaches": reaches,
        "expected_picks": expected,
        "risk_distribution": {"High": high_risk, "Low": low_risk, "Medium": max(0, n - high_risk - low_risk)},
        "analysis_chips": [
            balance,
            f"{steals} steal{'s' if steals != 1 else ''}" if steals else None,
            f"{reaches} reach{'es' if reaches != 1 else ''}" if reaches else None,
            f"{near_ready} near-ready" if near_ready else None,
            f"{high_risk} high-risk" if high_risk else None,
            f"{long_term} long projects" if long_term else None,
        ],
    }


def _enrich_draft_review_pick(
    session: FranchiseSession,
    pick: Dict[str, Any],
    *,
    team: Any,
    needs: List[Dict[str, Any]],
    board_by_id: Dict[str, Dict[str, Any]],
    league: Any,
) -> Dict[str, Any]:
    from services.draft_rights_engine import rights_card_payload, development_path_for
    from services.draft_player_registry import get_player

    pid = str(pick.get("prospect_id") or pick.get("player_id") or pick.get("selected_prospect_id") or "")
    player = None
    if league is not None and pid:
        try:
            player = get_player(league, pid)
        except Exception:
            player = None
    board_entry = board_by_id.get(pid) or {}
    for alt in (pick.get("key"), pick.get("player_id")):
        if not board_entry and alt:
            board_entry = board_by_id.get(str(alt)) or {}

    try:
        card = rights_card_payload(player) if player is not None else {}
    except Exception:
        card = {}

    identity = _dr_identity_fields(pick, player, board_entry)
    if identity.get("league"):
        identity["league"] = _dr_league_label(identity["league"]) or identity["league"]
    path_raw = card.get("returning_to") or pick.get("development_path") or development_path_for(pick)
    path = _dr_league_label(path_raw) or path_raw
    selection = _dr_selection_review({**pick, **identity}, needs)
    production = _dr_production_snapshot(
        session, player, {**pick, **identity}, board=board_entry, selection=selection
    )
    env = card.get("development_environment") or {}
    fit = _dr_organizational_fit(team, {**pick, **identity, "prospect_id": pid}, player, needs, env)
    plan = _dr_development_plan(
        {**pick, **identity, **selection, "prospect_id": pid},
        player,
        card,
        production,
        fit,
        board=board_entry,
    )
    rights = _dr_rights_preview(card, plan, {**pick, "prospect_id": pid, "draft_year": pick.get("draft_year")})
    review_line = _dr_review_line({**pick, **identity}, selection, plan, fit)
    eta_years = plan.get("eta_years")
    if eta_years is None:
        eta_years, _ = _dr_resolve_eta_years({**pick, **identity}, board_entry, player)

    base = {k: v for k, v in pick.items() if k != "player_ref"}
    return {
        **base,
        "prospect_id": pid or base.get("prospect_id"),
        "prospect_name": pick.get("prospect_name") or pick.get("name") or board_entry.get("name"),
        "overall_pick": pick.get("overall_pick"),
        "round": pick.get("round"),
        "round_pick": pick.get("pick_in_round") or pick.get("round_pick"),
        "position": pick.get("position") or identity.get("secondary_position") or _dr_player_pos(player),
        **identity,
        **selection,
        "production": production,
        "development_plan": plan,
        "organizational_fit": fit,
        "rights_card": rights,
        "review_line": review_line,
        "potential_label": production.get("potential_label") or _dr_public_ceiling({**pick, **identity}, board_entry),
        "floor_label": production.get("floor_label") or _dr_public_floor({**pick, **identity}, board_entry),
        "ceiling_label": production.get("ceiling_label") or _dr_public_ceiling({**pick, **identity}, board_entry),
        # Backward-compatible fields used by older UI / stages
        "development_path": path,
        "path_visual": [s.get("stage") for s in (plan.get("path_steps") or [])] or card.get("path_visual"),
        "nhl_eta": eta_years,
        "recommended_path": plan.get("next_destination_label") or card.get("recommended_label"),
        "elc_eligible": rights.get("entry_level_contract_eligible", True),
        "can_slide": rights.get("elc_can_slide"),
        "ahl_eligible": plan.get("ahl_eligible"),
        "uses_contract_slot_if_signed": True,
        "depth_at_position": fit.get("depth_at_position") or {"count": 0, "blockers": []},
        "development_environment": env,
        "development_risks": (env or {}).get("reasons") or [],
    }


def _run_draft_review(session: FranchiseSession) -> Dict[str, Any]:
    """Post-draft organizational review — value, paths, fit, and rights preview."""
    existing = getattr(session, "draft_review_payload", None)
    if isinstance(existing, dict) and existing.get("user_picks") is not None and existing.get("version") == 4:
        return {"draft_review": existing}

    from services.franchise_entry_draft import get_draft_recap, calculate_team_needs
    from services.franchise_sim import get_cached_draft_class_rankings

    state = getattr(session, "draft_state", None) or {}
    completed = list(state.get("completed_picks") or state.get("draft_results") or [])
    user_id = str(session.user_team_id)
    user_picks = [p for p in completed if str(p.get("team_id")) == user_id]
    league = getattr(session.sim, "league", None)
    team = session.team_by_id.get(user_id)

    needs: List[Dict[str, Any]] = []
    try:
        needs = calculate_team_needs(session, user_id)
    except Exception:
        needs = []

    board_by_id: Dict[str, Dict[str, Any]] = {}
    try:
        board = get_cached_draft_class_rankings(session, session.sim)
        for e in (board.get("entries") or []):
            for ident in (e.get("key"), e.get("prospect_id"), e.get("player_id")):
                if ident is not None:
                    board_by_id.setdefault(str(ident), e)
    except Exception:
        board_by_id = {}

    enriched: List[Dict[str, Any]] = []
    draft_year = state.get("draft_year") or (int(session.season_calendar_year) + 1)
    for pick in user_picks:
        try:
            row = dict(pick) if isinstance(pick, dict) else {}
            row.setdefault("draft_year", draft_year)
            enriched.append(
                _enrich_draft_review_pick(
                    session,
                    row,
                    team=team,
                    needs=needs,
                    board_by_id=board_by_id,
                    league=league,
                )
            )
        except Exception:
            # Degrade gracefully — keep a minimal row so the stage still opens.
            pid = str((pick or {}).get("prospect_id") or "")
            enriched.append({
                **{k: v for k, v in (pick or {}).items() if k != "player_ref"},
                "prospect_id": pid,
                "prospect_name": (pick or {}).get("prospect_name") or (pick or {}).get("name"),
                "selection_grade": "C",
                "selection_verdict": "Uncertain value",
                "selection_reason": "Incomplete prospect data for full review.",
                "scouting_confidence_label": "Low",
                "review_line": "Selection recorded with incomplete evaluation data.",
                "production": {"league_context": "Season stats unavailable", "data_confidence": "Low"},
                "development_plan": {
                    "next_destination": (pick or {}).get("development_path") or "Junior",
                    "next_destination_label": "Development path under review",
                    "recommended_role": "Org prospect",
                    "minutes_target": "Standard development minutes",
                    "special_teams_role": "TBD",
                    "season_objective": "Establish a development baseline",
                    "secondary_objective": "Gather more evaluation data",
                    "alternate_path": "Reassess at Prospect Rights",
                    "path_steps": [
                        {"stage": "Development", "status": "next", "detail": "Evaluation"},
                        {"stage": "AHL", "status": "future", "detail": "Pro path"},
                        {"stage": "NHL", "status": "projection", "detail": "Projection"},
                    ],
                    "eta_range": "3–5 years",
                    "eta_confidence": "Low",
                    "ahl_eligible": False,
                    "elc_can_slide": True,
                },
                "organizational_fit": {
                    "fit_grade": "B",
                    "fit_label": "Pipeline addition",
                    "depth_status": "Depth under review",
                    "nhl_players_ahead": 0,
                    "ahl_players_ahead": 0,
                    "prospects_ahead": 0,
                    "path_congestion": "Low",
                    "need_filled": "Organizational depth",
                    "expected_pipeline_rank": 1,
                    "blockers": ["Incomplete data"],
                    "opportunities": ["Earn evaluation through play"],
                    "environment_grade": "Acceptable environment",
                    "environment_reason": "Insufficient signals",
                },
                "rights_card": {
                    "rights_status": "exclusive_rights",
                    "rights_status_label": "Exclusive rights",
                    "signing_recommendation": "Wait",
                    "signing_reason": "Review details at Prospect Rights.",
                    "elc_can_slide": True,
                    "contract_required_now": False,
                },
            })

    haul = _dr_haul_summary(enriched, needs)

    recap = None
    try:
        recap = get_draft_recap(session)
    except Exception:
        recap = None

    payload = {
        "version": STAGE_PAYLOAD_VERSION["draft_review"],
        "draft_year": state.get("draft_year") or (int(session.season_calendar_year) + 1),
        "total_picks": len(enriched),
        "user_picks": enriched,
        "haul_summary": haul,
        "headline": haul.get("summary_line") or (
            f"{len(enriched)} selections for your club" if enriched else "Draft complete"
        ),
        "next_stage": "prospect_rights",
        "user_grade": haul.get("haul_grade") or (recap or {}).get("user_grade"),
        "stage_status": "ready",
        "can_continue": True,
        "blocking_reasons": [],
        "warning_reasons": [],
        "available_actions": ["continue_to_prospect_rights", "back_to_hub"],
    }
    session.draft_review_payload = payload
    return {"draft_review": payload}


def _run_prospect_rights_stage(session: FranchiseSession, *, force: bool = False) -> Dict[str, Any]:
    """
    Post-draft rights management: ELC offers, return-to-league, rights review, slots.
    """
    from services.draft_rights_engine import process_draft_rights_deadlines, rights_card_payload
    from services.contract_economy import (
        CONTRACT_SLOTS_LIMIT,
        validate_contract_slots,
        add_to_reserve_list,
        run_cpu_prospect_rights_pass,
    )

    existing = getattr(session, "prospect_rights_payload", None)
    if (
        not force
        and isinstance(existing, dict)
        and existing.get("version") == STAGE_PAYLOAD_VERSION["prospect_rights"]
        and existing.get("prospects") is not None
    ):
        return {"prospect_rights": existing}

    league = getattr(session.sim, "league", None)
    season_year = int(session.season_calendar_year)
    rights_result = {}
    if league is not None:
        rights_result = process_draft_rights_deadlines(session, league, season_year)

    # CPU orgs make rights decisions independently (idempotent via team flags).
    cpu_rights = {}
    try:
        cpu_rights = run_cpu_prospect_rights_pass(session)
    except Exception:
        cpu_rights = {}

    team = session.team_by_id.get(session.user_team_id)
    if team is not None:
        for p in list(getattr(team, "prospect_pool", None) or []):
            if bool(getattr(p, "entry_level_contract_eligible", False)):
                add_to_reserve_list(team, p, added_season=season_year)

    reserve = list(getattr(team, "reserve_list", None) or []) if team is not None else []
    unsigned = [e for e in reserve if isinstance(e, dict) and str(e.get("signed_status", "unsigned")).lower() != "signed"]
    expiring = [
        e for e in unsigned
        if e.get("rights_expiry_year") is not None and int(e.get("rights_expiry_year") or 0) <= season_year + 1
    ]
    slots = validate_contract_slots(team, league, additional=0) if team is not None else {}
    used = int(slots.get("contract_slots_used") or 0)
    prospect_cards = []
    for e in unsigned:
        pid = str(e.get("player_id") or "")
        player = None
        try:
            from services.draft_player_registry import get_player
            player = get_player(league, pid) if league is not None else None
        except Exception:
            player = None
        card = (
            rights_card_payload(player, team=team, season_year=season_year)
            if player is not None
            else {
                "rights_through": e.get("rights_expiry_year"),
                "returning_to": e.get("current_league_id"),
                "elc_decision": "Unsigned",
                "available_actions": [
                    {
                        "id": "keep_unsigned",
                        "label": "Keep unsigned",
                        "enabled": True,
                        "pros": ["Preserves a contract slot for other signings"],
                        "cons": ["No roster control until signed"],
                        "summary": "Leave unsigned and revisit later",
                    }
                ],
            }
        )
        prospect_cards.append({
            "player_id": pid,
            "name": e.get("name") or card.get("name"),
            "position": e.get("position"),
            "age": e.get("age") or getattr(getattr(player, "identity", None), "age", None) if player else None,
            "decision_status": card.get("decision_status") or e.get("decision_status") or "pending",
            "contract_slot_impact": 1 if card.get("entry_level_contract_eligible", True) else 0,
            **card,
        })

    warnings = []
    if used >= int(slots.get("contract_slots_limit", CONTRACT_SLOTS_LIMIT)):
        warnings.append("Contract slots full — ELC signings blocked until a slot opens")
    if expiring:
        warnings.append(f"{len(expiring)} rights nearing expiry")

    payload = {
        "version": STAGE_PAYLOAD_VERSION["prospect_rights"],
        "season_year": season_year,
        "contracts": f"{used}/{slots.get('contract_slots_limit', CONTRACT_SLOTS_LIMIT)}",
        "contract_slots_used": used,
        "contract_slots_limit": slots.get("contract_slots_limit", CONTRACT_SLOTS_LIMIT),
        "reserve_rights": len(unsigned),
        "elc_slots_available": max(0, int(slots.get("contract_slots_limit", CONTRACT_SLOTS_LIMIT)) - used),
        "expiring_this_year": expiring,
        "reentry_eligible": rights_result.get("reentry_eligible") or [],
        "notifications": rights_result.get("notifications") or [],
        "prospects": prospect_cards,
        "recommended_signing_priority": [
            c for c in prospect_cards
            if c.get("rights_through") is not None and int(c.get("rights_through") or 9999) <= season_year + 1
        ][:8],
        "rights_review": rights_result,
        "cpu_rights": cpu_rights,
        "stage_status": "ready",
        "can_continue": True,
        "blocking_reasons": [],
        "warning_reasons": warnings,
        "available_actions": ["sign_elc", "keep_unsigned", "continue_to_re_sign", "back_to_hub", "open_cap_ledger"],
    }
    session.prospect_rights_payload = payload
    session.draft_rights_review_payload = rights_result
    return {"prospect_rights": payload}


def _run_draft_combine(session: FranchiseSession) -> Dict[str, Any]:
    """Hydrate Draft Combine stage — prospect testing, team impressions, final board prep."""
    from services.franchise_scouting import run_franchise_draft_combine

    if getattr(session, "draft_combine_done", False) and session.draft_combine_payload:
        return {"draft_combine": session.draft_combine_payload}
    if not session.draft_lottery_done:
        _run_draft_lottery(session)
    payload = run_franchise_draft_combine(session)
    session.draft_combine_payload = payload
    session.draft_combine_done = True
    return {"draft_combine": payload}


def _prepare_draft_payload(session: FranchiseSession) -> Dict[str, Any]:
    from services.franchise_entry_draft import prepare_offseason_draft_payload

    if not getattr(session, "draft_combine_done", False):
        raise ValueError("Draft Combine must be completed before the Entry Draft")
    return prepare_offseason_draft_payload(session)


def _prepare_resign_payload(session: FranchiseSession, *, force: bool = False) -> Dict[str, Any]:
    from services.contract_economy import (
        build_contract_office,
        compute_player_demand,
        contract_row_available_actions,
    )

    existing = getattr(session, "resign_payload", None)
    if not force and isinstance(existing, dict) and existing.get("version") == STAGE_PAYLOAD_VERSION["re_sign"]:
        return {"contracts": existing, "re_sign": existing}

    office = build_contract_office(session)
    expiring = list(office.get("expiring") or [])
    rfa_rows = list(office.get("rfa_rights") or [])
    contracts = list(office.get("contracts") or [])
    summary = dict(office.get("summary") or {})
    user_team = session.team_by_id.get(session.user_team_id)
    league = getattr(session.sim, "league", None)
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)

    def _enrich_demand(row: Dict[str, Any]) -> Dict[str, Any]:
        pid = str(row.get("player_id") or "")
        player = None
        try:
            from services.draft_player_registry import get_player
            player = get_player(league, pid) if league is not None else None
        except Exception:
            player = None
        if player is None and user_team is not None:
            for p in list(getattr(user_team, "roster", None) or []):
                if str(getattr(p, "id", "")) == pid:
                    player = p
                    break
        # Own expired UFAs sit in the FA pool during the exclusive window.
        if player is None and league is not None and pid:
            for p in list(getattr(league, "free_agents", None) or []):
                if str(getattr(p, "id", "")) == pid:
                    player = p
                    break
        # RFA rights holders are off-roster — resolve via rights entry / player_ref.
        if player is None and user_team is not None and (
            row.get("contract_status") == "rfa_rights" or row.get("can_qualify")
        ):
            try:
                from services.contract_economy import find_rfa_rights, resolve_rfa_player

                entry = find_rfa_rights(user_team, pid)
                player = resolve_rfa_player(entry, league)
            except Exception:
                player = None
        if player is None:
            # Still surface a usable QO-based ask so Rights rows are not blank.
            if row.get("contract_status") == "rfa_rights":
                qo = row.get("qualifying_offer_aav_m") or row.get("previous_aav_m")
                if qo is not None:
                    row.setdefault("player_ask_aav_m", qo)
                    row.setdefault("requested_cap_hit", qo)
                    row.setdefault("requested_term", 1)
                    row.setdefault("aav_m", row.get("previous_aav_m") or qo)
                    row.setdefault("current_cap_hit", row.get("previous_aav_m") or qo)
                row.setdefault("available_actions", contract_row_available_actions(row))
            else:
                row.setdefault("available_actions", contract_row_available_actions(row))
            return row
        try:
            demand = compute_player_demand(player, user_team, league, context="re_sign")
            ask = demand.get("want_aav_m")
            years = int(demand.get("want_years") or 2)
            row["player_ask_aav_m"] = ask
            row["requested_cap_hit"] = ask
            row["requested_term"] = years
            row["expected_aav_range"] = [
                demand.get("min_acceptable_aav_m"),
                round(float(ask or 0) * 1.08, 3) if ask else None,
            ]
            row["expected_term_range"] = [max(1, years - 1), min(8, years + 1)]
            morale = float(getattr(player, "morale", None) or getattr(player, "happiness", 70) or 70)
            row["morale"] = round(morale, 1)
            loyalty = float((demand.get("profile") or {}).get("loyalty") or 0.5)
            importance = float(demand.get("importance") or 0.5)
            stay_interest = max(
                0.0,
                min(
                    100.0,
                    50.0 + (loyalty - 0.5) * 40.0 + (morale - 50.0) * 0.45 + importance * 20.0,
                ),
            )
            interest = "High" if stay_interest >= 70 else ("Low" if stay_interest < 45 else "Medium")
            row["interest_label"] = interest
            row["interest_level"] = interest
            row["stay_interest"] = round(stay_interest, 1)
            row["clause_ask"] = "NMC" if years >= 5 and _safe_attr_float(player, "overall", "ovr") >= 88 else (
                "NTC" if years >= 4 and _safe_attr_float(player, "overall", "ovr") >= 84 else "None"
            )
            row["negotiation_state"] = "open" if row.get("can_negotiate") or row.get("can_qualify") else "closed"
            # Persist negotiation baseline on session so reopening does not invent new demands.
            neg_map = getattr(session, "resign_negotiations", None)
            if not isinstance(neg_map, dict):
                session.resign_negotiations = {}
                neg_map = session.resign_negotiations
            if pid and pid not in neg_map:
                neg_map[pid] = {
                    "negotiation_id": f"resign-{pid}-{season_year}",
                    "player_id": pid,
                    "team_id": str(getattr(user_team, "team_id", "") or ""),
                    "negotiation_type": "re_sign",
                    "status": "open",
                    "opened_season": season_year,
                    "current_round": 0,
                    "baseline_demand": {
                        "minimum_acceptance": demand.get("min_acceptable_aav_m"),
                        "target_aav": ask,
                        "opening_request": round(float(ask or 0) * 1.06, 3) if ask else None,
                        "target_term": years,
                        "preferred_clause": row.get("clause_ask"),
                    },
                    "team_offers": [],
                    "player_counters": [],
                }
            elif pid and pid in neg_map:
                row["negotiation_id"] = neg_map[pid].get("negotiation_id")
                row["negotiation_round"] = neg_map[pid].get("current_round", 0)
                row["negotiation_status"] = neg_map[pid].get("status")
                pending = neg_map[pid].get("pending_offer")
                if isinstance(pending, dict):
                    row["pending_offer"] = {
                        "aav_m": pending.get("aav_m"),
                        "years": pending.get("years"),
                        "interest": pending.get("interest"),
                        "resolve_days": pending.get("resolve_days"),
                        "days_held": pending.get("days_held"),
                        "days_remaining": max(
                            0,
                            int(pending.get("resolve_days") or 0) - int(pending.get("days_held") or 0),
                        ),
                    }
                    row["negotiation_state"] = "pending"
                base = neg_map[pid].get("baseline_demand") or {}
                if base.get("target_aav") is not None:
                    row["player_ask_aav_m"] = base.get("target_aav")
                    row["requested_cap_hit"] = base.get("target_aav")
                if base.get("target_term") is not None:
                    row["requested_term"] = base.get("target_term")
                counters = neg_map[pid].get("player_counters") or []
                if counters:
                    last = counters[-1]
                    row["last_counter"] = last
            row["legal_contract_types"] = _resign_legal_contract_types(row, player)
        except Exception:
            pass
        row["expiry_type"] = row.get("expiry_status") or row.get("rights_status") or row.get("expiry_type")
        row["current_cap_hit"] = row.get("aav_m") or row.get("cap_hit_m")
        row["current_salary"] = row.get("aav_m") or row.get("cap_hit_m")
        row["available_actions"] = contract_row_available_actions(row)
        return row

    def _resign_legal_contract_types(row: Dict[str, Any], player: Any) -> List[Dict[str, Any]]:
        types = []
        ovr = _safe_attr_float(player, "overall", "ovr")
        age = int(getattr(player, "age", 25) or 25)
        types.append({"id": "nhl_one_way", "label": "NHL one-way", "enabled": True})
        types.append({
            "id": "nhl_two_way",
            "label": "NHL two-way",
            "enabled": ovr < 82 or age <= 24,
            "blocked_reason": None if (ovr < 82 or age <= 24) else "Player expects one-way security",
        })
        if ovr < 72 or str(row.get("league") or "").upper() in ("AHL", "ECHL"):
            types.append({"id": "ahl", "label": "AHL contract", "enabled": True})
            types.append({"id": "ahl_echl_two_way", "label": "AHL/ECHL two-way", "enabled": True})
        if ovr < 65:
            types.append({"id": "echl", "label": "ECHL contract", "enabled": True})
        if age >= 30 and ovr < 74:
            types.append({"id": "pto", "label": "Professional tryout", "enabled": True})
        return types

    # Attach demand bands without exposing hidden formulas.
    expiring = [_enrich_demand(dict(r)) for r in expiring]
    rfa_rows = [_enrich_demand(dict(r)) for r in rfa_rows]
    contracts = [_enrich_demand(dict(r)) for r in contracts]

    # Deduped table universe: all org contracts + RFA rights + own UFAs not already listed.
    contract_ids = {str(r.get("player_id") or "") for r in contracts}
    table_rows = list(contracts)
    for r in rfa_rows:
        pid = str(r.get("player_id") or "")
        if pid and pid not in contract_ids:
            table_rows.append(r)
            contract_ids.add(pid)
    for r in expiring:
        pid = str(r.get("player_id") or "")
        if pid and pid not in contract_ids and (
            bool(r.get("own_ufa")) or str(r.get("contract_status") or "") == "own_ufa"
        ):
            table_rows.append(r)
            contract_ids.add(pid)

    # Phase outcomes keep Accepted / Rejected / Released rows visible until Free Agency.
    outcomes = ensure_resign_phase_outcomes(session)
    live_by_id = {str(r.get("player_id") or ""): r for r in table_rows if r.get("player_id")}

    # Seed open outcomes for anyone currently pending so filters can keep them later.
    for r in list(expiring) + list(rfa_rows):
        pid = str(r.get("player_id") or "")
        if not pid:
            continue
        if pid not in outcomes:
            upsert_resign_phase_outcome(
                session,
                player_id=pid,
                phase_status="open",
                snapshot_row=dict(r),
                name=r.get("name"),
            )

    for pid, outcome in list(outcomes.items()):
        if not isinstance(outcome, dict):
            continue
        status = str(outcome.get("phase_status") or "open")
        snap = dict(outcome.get("snapshot_row") or {})
        live = live_by_id.get(pid)
        if live is not None:
            # Annotate the live row; preserve a snapshot for terminal display if they leave later.
            live["phase_status"] = status
            live["phase_terminal"] = bool(outcome.get("terminal") or status in RESIGN_PHASE_TERMINAL)
            if outcome.get("terms"):
                live["phase_terms"] = outcome.get("terms")
            if outcome.get("last_offer"):
                live["phase_last_offer"] = outcome.get("last_offer")
            if outcome.get("reason"):
                live["phase_reason"] = outcome.get("reason")
            if status in RESIGN_PHASE_TERMINAL:
                live["can_negotiate"] = False
                if status in ("accepted", "released"):
                    live["available_actions"] = []
            # Refresh snapshot from live row while they remain on the board.
            upsert_resign_phase_outcome(
                session,
                player_id=pid,
                phase_status=status,
                snapshot_row=dict(live),
                terms=outcome.get("terms"),
                last_offer=outcome.get("last_offer"),
                reason=outcome.get("reason"),
                name=live.get("name") or outcome.get("name"),
            )
        elif status in RESIGN_PHASE_TERMINAL or status in ("rejected", "countered", "pending"):
            # Player left live eligibility (signed / walked) — keep frozen snapshot on the desk.
            row = dict(snap) if snap else {"player_id": pid, "name": outcome.get("name") or pid}
            row["player_id"] = pid
            row["phase_status"] = status
            row["phase_terminal"] = bool(outcome.get("terminal") or status in RESIGN_PHASE_TERMINAL)
            if outcome.get("terms"):
                row["phase_terms"] = outcome.get("terms")
            if outcome.get("last_offer"):
                row["phase_last_offer"] = outcome.get("last_offer")
            if outcome.get("reason"):
                row["phase_reason"] = outcome.get("reason")
            if status == "accepted":
                row["can_negotiate"] = False
                row["available_actions"] = []
                row["contract_status"] = row.get("contract_status") or "signed"
                if int(row.get("years_remaining") or 0) <= 1 and outcome.get("terms"):
                    yrs = outcome["terms"].get("years")
                    if yrs is not None:
                        row["years_remaining"] = yrs
            elif status == "released":
                row["can_negotiate"] = False
                row["can_qualify"] = False
                row["can_release_rights"] = False
                row["available_actions"] = []
                row["contract_status"] = "released"
            table_rows.append(row)
            contract_ids.add(pid)

    grouped = {
        "pending_ufa": [r for r in expiring if str(r.get("expiry_status") or r.get("rights") or "").upper() == "UFA"],
        "pending_rfa": (
            [r for r in expiring if str(r.get("expiry_status") or r.get("rights") or "").upper() == "RFA"]
            + rfa_rows
        ),
        "buyout_candidates": list(office.get("buyout_candidates") or [])[:12],
        "signed_next_season": [r for r in contracts if int(r.get("years_remaining") or 0) > 1],
        "goalies": [r for r in contracts if str(r.get("position") or "").upper() == "G"],
        "extension_eligible": [r for r in table_rows if r.get("extension_eligible")],
        "phase_accepted": [r for r in table_rows if str(r.get("phase_status") or "") == "accepted"],
        "phase_rejected": [r for r in table_rows if str(r.get("phase_status") or "") == "rejected"],
        "phase_released": [r for r in table_rows if str(r.get("phase_status") or "") == "released"],
    }

    blocking_decisions = []
    rfa_warnings = []
    for r in rfa_rows:
        if r.get("can_qualify") or r.get("can_release_rights"):
            # RFA rights can carry into Free Agency (qualify / offer sheet later).
            # Do not hard-block July 1 open — warn only.
            rfa_warnings.append({
                "player_id": r.get("player_id"),
                "name": r.get("name"),
                "code": "rfa_rights",
                "message": f"{r.get('name') or 'Player'}: qualify or release RFA rights",
            })

    pending_decisions = list(grouped["pending_ufa"]) + list(grouped["pending_rfa"])
    # Deduplicate pending by player_id
    seen_pending = set()
    pending_unique = []
    for r in pending_decisions:
        pid = str(r.get("player_id") or "")
        if not pid or pid in seen_pending:
            continue
        seen_pending.add(pid)
        pending_unique.append(r)

    warnings = []
    if summary.get("ufaCount"):
        warnings.append(f"{summary.get('ufaCount')} pending UFAs")
    if summary.get("rfaCount"):
        warnings.append(f"{summary.get('rfaCount')} RFA situations")
    if rfa_warnings:
        warnings.append(f"{len(rfa_warnings)} RFA rights still open (can resolve during Free Agency)")

    remaining_count = len(rfa_warnings)
    resolved_count = max(0, int(summary.get("rfaCount") or 0) - remaining_count)
    window = ensure_own_fa_window(session)
    bonus = team_signing_bonus_eligibility(session)
    # Opening FA is always allowed from re-sign; exclusive days and open RFAs are optional.
    can_continue = True
    if not window.get("complete"):
        warnings.append(
            f"Exclusive window Day {window.get('day', 0)}/{window.get('days_total', 6)} — "
            "Sim Day to progress offers, or Open Free Agency to end exclusivity"
        )

    payload = {
        "version": STAGE_PAYLOAD_VERSION["re_sign"],
        "season_year": season_year,
        "contracts": table_rows,
        "expiring_contracts": expiring,
        "cap_snapshot": office.get("cap_snapshot") or office.get("team_cap") or {},
        "contract_slots": office.get("contract_slots") or {},
        "summary": {
            **summary,
            "pendingDecisions": len(pending_unique),
            "blockingDecisions": 0,
            "openRfaRights": remaining_count,
            "tableRows": len(table_rows),
            "phaseAccepted": len(grouped.get("phase_accepted") or []),
            "phaseRejected": len(grouped.get("phase_rejected") or []),
            "phaseReleased": len(grouped.get("phase_released") or []),
        },
        "grouped": grouped,
        "rfa_rights": rfa_rows,
        "pending_decisions": pending_unique,
        "blocking_decisions": blocking_decisions,
        "open_rfa_rights": rfa_warnings,
        "phase_outcomes": dict(ensure_resign_phase_outcomes(session)),
        "needs": (office.get("team") or {}).get("needs") or {},
        "team": office.get("team") or {},
        "stage_status": "ready",
        "can_continue": can_continue,
        "blocking_reasons": [],
        "warning_reasons": warnings,
        "resolved_count": resolved_count,
        "remaining_count": remaining_count,
        "own_fa_window": window,
        "signing_bonus": bonus,
        "instant_accept_interest": INSTANT_ACCEPT_INTEREST,
        "available_actions": ["sim_negotiation_day", "open_cap_ledger", "continue_to_free_agency", "back_to_hub"],
    }
    session.resign_payload = payload
    return {"contracts": payload, "re_sign": payload}


def _open_free_agency(session: FranchiseSession, *, force: bool = False) -> Dict[str, Any]:
    from services.contract_economy import (
        build_contract_office,
        expire_pending_july1_contracts,
        run_cpu_own_ufa_resign,
        run_cpu_rfa_decisions,
        sync_all_team_cap_fields,
    )
    from services.fa_market_engine import (
        annotate_fa_rows_with_decisions,
        ensure_fa_market_book,
        tick_free_agency_market,
    )

    window = own_fa_window_status(session)
    # Entering the Free Agency stage always opens the market (July 1). The
    # exclusive window is optional negotiating time on the re-sign desk — it must
    # not leave the Wire empty when the GM chooses Open Free Agency.
    if not force and not window.get("complete") and not session.free_agency_open:
        force = True

    existing_market = getattr(session, "free_agency_market_payload", None)
    already_open = bool(session.free_agency_open)
    wave = int(getattr(session, "cpu_fa_wave", 0) or 0)

    # Idempotent: July 1 burn → RFAs + own UFAs, then opens the living market.
    if not already_open or force:
        # Final-year UFAs deferred at salary-cap become free agents here (July 1).
        if not getattr(session, "july1_contracts_expired", False):
            session.july1_expiry_report = expire_pending_july1_contracts(session)
        if not getattr(session, "cpu_rfa_decisions", None):
            session.cpu_rfa_decisions = run_cpu_rfa_decisions(session)
        if not getattr(session, "cpu_own_ufa_resign", None):
            # Retain stars on CPU clubs BEFORE exclusivity clears onto Opening Day.
            session.cpu_own_ufa_resign = run_cpu_own_ufa_resign(session)
        try:
            league = getattr(session.sim, "league", None)
            sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
            if league is not None:
                from app.sim_engine.league_hierarchy_bootstrap import ensure_overseas_fa_pool

                ensure_overseas_fa_pool(league, session.sim.rng, min_count=120)
                sync_all_team_cap_fields(league, session.sim, season_year=sy)
        except Exception:
            pass
        ensure_fa_market_book(session)
        if wave < 1:
            # Opening day: offers circulate; only a couple fringe deals may close.
            tick = tick_free_agency_market(
                session,
                days=1,
                opening_day=True,
                max_signings_per_day=2,
                max_offers_per_day=2000,
            )
            session.cpu_fa_wave = 1
            session._last_fa_market_tick = tick
        elif force and int(getattr(session, "fa_market_day", 0) or 0) < 2:
            tick_free_agency_market(session, days=1, max_signings_per_day=3)
        session.free_agency_open = True
        session.own_fa_window_active = False
        # Release exclusive home-team UFAs onto the open market.
        try:
            league = getattr(session.sim, "league", None)
            for pool_attr in ("free_agents", "overseas_free_agents"):
                for p in list(getattr(league, pool_attr, None) or []) if league else []:
                    try:
                        setattr(p, "ufa_exclusive", False)
                    except Exception:
                        pass
        except Exception:
            pass

    try:
        league = getattr(session.sim, "league", None)
        sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
        if league is not None:
            from app.sim_engine.league_hierarchy_bootstrap import ensure_overseas_fa_pool
            from services.franchise_sim import resync_league_ages_to_session

            ensure_overseas_fa_pool(league, session.sim.rng, min_count=120)
            sync_all_team_cap_fields(league, session.sim, season_year=sy)
            resync_league_ages_to_session(session)
    except Exception:
        pass

    ensure_fa_market_book(session)
    office = build_contract_office(session)
    fa_list = annotate_fa_rows_with_decisions(
        session, list(office.get("free_agents") or office.get("freeAgents") or [])
    )
    overseas_list = annotate_fa_rows_with_decisions(
        session, list(office.get("overseas_free_agents") or [])
    )
    session.free_agents_payload = fa_list
    bonus = team_signing_bonus_eligibility(session)
    cap = office.get("cap_snapshot") or {}
    needs = (office.get("team") or {}).get("needs") or {}
    summary = office.get("summary") or {}
    # Keep a short "headline board" of stars, but the full pool is free_agents.
    top = sorted(fa_list, key=lambda r: -float(r.get("ovr") or r.get("overall") or 0))[:24]
    cpu = getattr(session, "cpu_fa_signings", None) or {}
    recent = list(cpu.get("signings") or [])[-12:]
    book = getattr(session, "fa_market_book", None) or {}
    book_log = list(book.get("log") or [])[-20:]
    news = []
    for s in recent[-8:]:
        news.append({
            "kind": "signing",
            "text": (
                f"{s.get('team_name') or s.get('team_id') or 'A club'} signs "
                f"{s.get('name') or s.get('player_id') or 'a free agent'} · "
                f"{s.get('aav_m')}M × {s.get('years')}y"
            ),
        })
    for entry in book_log[-8:]:
        if isinstance(entry, dict) and entry.get("text"):
            news.append({"kind": entry.get("kind") or "market", "text": entry.get("text")})
        elif isinstance(entry, str):
            news.append({"kind": "market", "text": entry})
    decisions = (getattr(session, "_last_fa_market_tick", None) or {}).get("decision_snapshot")
    if not decisions:
        from services.fa_market_engine import _decision_snapshot
        decisions = _decision_snapshot(book)

    day = int(getattr(session, "fa_market_day", 0) or book.get("day") or 0)
    if day <= 1:
        phase = "opening_day"
    elif day <= 3:
        phase = "initial_rush"
    elif day <= 7:
        phase = "first_week"
    elif day <= 14:
        phase = "secondary_market"
    elif day <= 30:
        phase = "late_summer"
    else:
        phase = "camp_tryout_market"

    market = {
        "version": STAGE_PAYLOAD_VERSION["free_agency"],
        "market_status": "open" if session.free_agency_open else "closed",
        "wave": int(getattr(session, "cpu_fa_wave", 0) or 0),
        "fa_market_day": day,
        "market_phase": phase,
        "market_phase_label": {
            "opening_day": "Opening Day",
            "initial_rush": "Initial Rush",
            "first_week": "First Week",
            "secondary_market": "Secondary Market",
            "late_summer": "Late Summer",
            "camp_tryout_market": "Camp / Tryout Market",
        }.get(phase, phase),
        "available_count": len(fa_list),
        "major_available": top,
        "free_agents": fa_list,
        "overseas_free_agents": overseas_list,
        "market_news": news[-16:],
        "cap_space_m": (
            float(cap["usable_cap_space_m"])
            if cap.get("usable_cap_space_m") is not None
            else (float(cap["cap_space_m"]) if cap.get("cap_space_m") is not None else 0.0)
        ),
        "cap_snapshot": cap,
        "contract_slots": office.get("contract_slots") or {},
        "needs": needs,
        "pending_rfa_count": summary.get("rfaCount") or 0,
        "signing_bonus": bonus,
        "recent_league_signings": recent,
        "cpu_signings_count": len(list(cpu.get("signings") or [])),
        "decision_snapshot": decisions,
        "stage_status": "ready",
        "can_continue": True,
        "blocking_reasons": [],
        "warning_reasons": (
            ["Pending RFAs still unresolved"] if (summary.get("rfaCount") or 0) > 0 else []
        ),
        "available_actions": [
            "advance_fa_day",
            "advance_fa_week",
            "advance_fa_month",
            "open_cap_ledger_fa",
            "continue_to_roster_check",
            "back_to_hub",
        ],
    }
    if isinstance(existing_market, dict) and already_open and not force:
        market["wave"] = existing_market.get("wave", market["wave"])
        market["fa_market_day"] = existing_market.get("fa_market_day", market["fa_market_day"])
    session.free_agency_market_payload = market
    return {
        "free_agents": session.free_agents_payload,
        "free_agency_market": market,
        "cpu_signings": session.cpu_fa_signings,
        "cpu_rfa_decisions": session.cpu_rfa_decisions,
    }


def advance_contract_negotiation_day(session: FranchiseSession, *, days: int = 1) -> Dict[str, Any]:
    """Advance the exclusive own-FA window and resolve pending offers.

    Insane offers already signed instantly. Competitive offers sit on the table and
    resolve after their resolve_days — Sim Day is how you watch players sign.
    """
    from services.contract_economy import (
        _find_player_in_league,
        sign_player_to_team,
    )

    days = max(1, min(14, int(days or 1)))
    ensure_own_fa_window(session)
    if session.free_agency_open:
        return {
            "ok": False,
            "reason": "Open free agency has already started",
            "own_fa_window": own_fa_window_status(session),
        }

    sim = session.sim
    league = getattr(sim, "league", None)
    user_team = session.team_by_id.get(str(session.user_team_id))
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    neg_map = getattr(session, "resign_negotiations", None)
    if not isinstance(neg_map, dict):
        session.resign_negotiations = {}
        neg_map = session.resign_negotiations

    signed: List[Dict[str, Any]] = []
    still_pending: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for _ in range(days):
        session.own_fa_window_day = int(getattr(session, "own_fa_window_day", 0) or 0) + 1
        for pid, entry in list(neg_map.items()):
            if not isinstance(entry, dict):
                continue
            pending = entry.get("pending_offer")
            if not isinstance(pending, dict):
                continue
            pending["days_held"] = int(pending.get("days_held") or 0) + 1
            need = max(1, int(pending.get("resolve_days") or 2))
            interest = float(pending.get("interest") or 0)
            if pending["days_held"] < need:
                still_pending.append(
                    {
                        "player_id": pid,
                        "days_held": pending["days_held"],
                        "resolve_days": need,
                        "interest": interest,
                    }
                )
                continue
            player, _owner = _find_player_in_league(league, str(pid))
            if player is None or user_team is None:
                entry["status"] = "lapsed"
                entry["pending_offer"] = None
                rejected.append({"player_id": pid, "reason": "player_missing"})
                try:
                    upsert_resign_phase_outcome(
                        session,
                        player_id=str(pid),
                        phase_status="lapsed",
                        reason="player_missing",
                    )
                except Exception:
                    pass
                continue
            offer = {
                "aav_m": pending.get("aav_m"),
                "years": pending.get("years"),
                "ntc": pending.get("ntc"),
                "nmc": pending.get("nmc"),
                "signing_bonus_m": pending.get("signing_bonus_m") or 0,
                "two_way": pending.get("two_way"),
                "contract_category": pending.get("contract_category") or "nhl_one_way",
                "context": pending.get("context") or "re_sign",
                "force": True,
                "resolve_pending": True,
                "_session": session,
            }
            result = sign_player_to_team(player, user_team, league, season_year, offer)
            if result.get("ok") and result.get("status") == "accepted":
                entry["status"] = "accepted"
                entry["pending_offer"] = None
                name = str(getattr(player, "name", None) or getattr(player, "full_name", pid) or pid)
                signing = {
                    "player_id": pid,
                    "name": name,
                    "aav_m": pending.get("aav_m"),
                    "years": pending.get("years"),
                    "window_day": int(session.own_fa_window_day),
                    "interest": interest,
                }
                signed.append(signing)
                session.own_fa_window_signings = list(getattr(session, "own_fa_window_signings", None) or [])
                session.own_fa_window_signings.append(signing)
                try:
                    upsert_resign_phase_outcome(
                        session,
                        player_id=str(pid),
                        phase_status="accepted",
                        name=name,
                        terms={"aav_m": pending.get("aav_m"), "years": pending.get("years")},
                        last_offer=dict(pending),
                    )
                except Exception:
                    pass
            else:
                entry["status"] = "lapsed"
                entry["pending_offer"] = None
                rejected.append({"player_id": pid, "reason": result.get("reason") or "failed"})
                try:
                    upsert_resign_phase_outcome(
                        session,
                        player_id=str(pid),
                        phase_status="lapsed",
                        name=str(getattr(player, "name", None) or pid),
                        reason=str(result.get("reason") or "failed"),
                        last_offer=dict(pending),
                    )
                except Exception:
                    pass

    invalidate_offseason_decision_payloads(session, reason="advance_negotiation_day")
    refreshed = _prepare_resign_payload(session, force=True)
    window = own_fa_window_status(session)
    return {
        "ok": True,
        "days_advanced": days,
        "signed": signed,
        "still_pending": still_pending,
        "rejected": rejected,
        "own_fa_window": window,
        "re_sign": refreshed.get("re_sign") or refreshed.get("contracts"),
        "contracts": refreshed.get("contracts") or refreshed.get("re_sign"),
    }


def resolve_user_fa_pending_offers(session: FranchiseSession, *, days: int = 1) -> Dict[str, Any]:
    """Resolve pending user FA offers when Sim Day advances the open market.

    Same mechanics as the exclusive re-sign window — competitive offers sit for
    resolve_days, then force-sign. Instant accepts already cleared the pool.
    """
    from services.contract_economy import _find_player_in_league, sign_player_to_team
    from services.fa_market_engine import mark_fa_player_signed, record_user_fa_offer

    days = max(1, min(45, int(days or 1)))
    sim = session.sim
    league = getattr(sim, "league", None)
    user_team = session.team_by_id.get(str(session.user_team_id))
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    neg_map = getattr(session, "resign_negotiations", None)
    if not isinstance(neg_map, dict):
        return {"signed": [], "still_pending": [], "rejected": []}

    signed: List[Dict[str, Any]] = []
    still_pending: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for _ in range(days):
        for pid, entry in list(neg_map.items()):
            if not isinstance(entry, dict):
                continue
            pending = entry.get("pending_offer")
            if not isinstance(pending, dict):
                continue
            ctx = str(pending.get("context") or "").lower()
            if ctx and ctx not in ("ufa", "free_agency", "fa", ""):
                # Exclusive re-sign pending offers use advance_contract_negotiation_day.
                if ctx in ("re_sign", "resign", "extension"):
                    continue
            pending["days_held"] = int(pending.get("days_held") or 0) + 1
            need = max(1, int(pending.get("resolve_days") or 2))
            interest = float(pending.get("interest") or 0)
            if pending["days_held"] < need:
                still_pending.append(
                    {
                        "player_id": pid,
                        "days_held": pending["days_held"],
                        "resolve_days": need,
                        "interest": interest,
                    }
                )
                continue
            player, owner = _find_player_in_league(league, str(pid))
            if player is None or user_team is None:
                entry["status"] = "lapsed"
                entry["pending_offer"] = None
                rejected.append({"player_id": pid, "reason": "player_missing"})
                continue
            if owner is not None and owner is not user_team:
                entry["status"] = "lapsed"
                entry["pending_offer"] = None
                name = str(getattr(player, "name", None) or getattr(player, "full_name", pid) or pid)
                rejected.append({
                    "player_id": pid,
                    "name": name,
                    "reason": "signed_elsewhere",
                    "feedback": f"{name} signed elsewhere.",
                })
                continue
            offer = {
                "aav_m": pending.get("aav_m"),
                "years": pending.get("years"),
                "ntc": pending.get("ntc"),
                "nmc": pending.get("nmc"),
                "signing_bonus_m": pending.get("signing_bonus_m") or 0,
                "two_way": pending.get("two_way"),
                "contract_category": pending.get("contract_category") or "nhl_one_way",
                "context": "ufa",
                "force": True,
                "resolve_pending": True,
                "_session": session,
            }
            result = sign_player_to_team(player, user_team, league, season_year, offer)
            if result.get("ok") and result.get("status") == "accepted":
                entry["status"] = "accepted"
                entry["pending_offer"] = None
                name = str(getattr(player, "name", None) or getattr(player, "full_name", pid) or pid)
                signing = {
                    "player_id": pid,
                    "name": name,
                    "aav_m": pending.get("aav_m"),
                    "years": pending.get("years"),
                    "interest": interest,
                    "team_id": str(session.user_team_id),
                }
                signed.append(signing)
                try:
                    mark_fa_player_signed(session, str(pid))
                    record_user_fa_offer(
                        session,
                        player_id=str(pid),
                        aav_m=float(pending.get("aav_m") or 0),
                        years=int(pending.get("years") or 1),
                        ntc=bool(pending.get("ntc")),
                        nmc=bool(pending.get("nmc")),
                        status="accepted",
                    )
                except Exception:
                    pass
                cpu = getattr(session, "cpu_fa_signings", None)
                if not isinstance(cpu, dict):
                    cpu = {"signings": []}
                    session.cpu_fa_signings = cpu
                cpu.setdefault("signings", []).append({
                    **signing,
                    "team_name": "Your club",
                    "source": "user_pending_resolve",
                })
            else:
                entry["status"] = "lapsed"
                entry["pending_offer"] = None
                name = str(getattr(player, "name", None) or getattr(player, "full_name", pid) or pid)
                pr = result.get("player_response")
                feedback = None
                if isinstance(pr, dict):
                    feedback = pr.get("feedback")
                rejected.append({
                    "player_id": pid,
                    "name": name,
                    "reason": result.get("reason") or "failed",
                    "feedback": feedback or result.get("reason") or f"{name} declined.",
                })

    return {"signed": signed, "still_pending": still_pending, "rejected": rejected}


def advance_free_agency_day(session: FranchiseSession, *, days: int = 1) -> Dict[str, Any]:
    """
    Advance free-agency market time by N days (default 1).
    CPU teams extend offers; players evaluate / wait / sign on staggered schedules.
    User pending FA offers also age and resolve on the same clock.
    """
    from services.fa_market_engine import tick_free_agency_market

    days = max(1, min(45, int(days or 1)))
    if not getattr(session, "free_agency_open", False):
        opened = _open_free_agency(session, force=False)
        # Opening already ticked day 1; for multi-day requests continue remaining days
        remaining = days - 1
        if remaining <= 0:
            market = opened.get("free_agency_market") or {}
            return {
                "ok": True,
                "free_agency_market": market,
                "free_agents": opened.get("free_agents"),
                "cpu_signings": session.cpu_fa_signings,
                "day": int(getattr(session, "fa_market_day", 0) or 0),
                "phase": market.get("market_phase"),
                "tick": getattr(session, "_last_fa_market_tick", None),
            }
        days = remaining

    user_resolve = resolve_user_fa_pending_offers(session, days=days)
    try:
        from services.contract_economy import run_cpu_offer_sheet_pass, tick_offer_sheets

        run_cpu_offer_sheet_pass(session, max_sheets=2)
        tick_offer_sheets(session)
    except Exception:
        pass
    tick = tick_free_agency_market(session, days=days)
    session._last_fa_market_tick = tick
    refreshed = _open_free_agency(session, force=False)
    market = dict(refreshed.get("free_agency_market") or {})
    market["version"] = STAGE_PAYLOAD_VERSION["free_agency"]
    market["fa_market_day"] = int(tick.get("day") or getattr(session, "fa_market_day", 0) or 0)
    # Merge user signings into the wire so Sim Day always moves the feed.
    wire = list(market.get("market_news") or [])
    for s in list(user_resolve.get("signed") or []):
        wire.append({
            "kind": "signing",
            "text": (
                f"Your club signs {s.get('name') or s.get('player_id')} · "
                f"{s.get('aav_m')}M × {s.get('years')}y"
            ),
        })
    for p in list(user_resolve.get("still_pending") or [])[:4]:
        wire.append({
            "kind": "pending",
            "text": (
                f"Your offer still on the table "
                f"({p.get('days_held')}/{p.get('resolve_days')} days) · {p.get('player_id')}"
            ),
        })
    market["market_news"] = wire[-24:]
    market["day_events"] = {
        "cpu_signings": len(tick.get("signings") or []),
        "new_offers": len(tick.get("offers") or []),
        "recent_signings": list(tick.get("signings") or [])[:8],
        "user_signings": list(user_resolve.get("signed") or []),
        "user_pending": list(user_resolve.get("still_pending") or []),
        "decision_snapshot": tick.get("decision_snapshot"),
        "days_advanced": days,
    }
    market["decision_snapshot"] = tick.get("decision_snapshot")
    session.free_agency_market_payload = market
    return {
        "ok": True,
        "free_agency_market": market,
        "free_agents": refreshed.get("free_agents") or market.get("free_agents"),
        "cpu_signings": session.cpu_fa_signings,
        "day": market["fa_market_day"],
        "phase": market.get("market_phase"),
        "tick": tick,
        "user_resolve": user_resolve,
    }


def _position_code(player: Any) -> str:
    """Position code ("C"/"LW"/"RW"/"D"/"G") — delegates to shared roster_compliance."""
    from services.roster_compliance import position_code

    return position_code(player)


def _revalidate_roster_cleanup(session: FranchiseSession, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh the Roster Check gate from live rosters without replaying any moves."""
    from services.contract_economy import get_team_cap_snapshot_full
    from services.roster_compliance import evaluate_roster_compliance

    user_team = session.team_by_id.get(session.user_team_id)
    league = getattr(session.sim, "league", None)
    season_year = int(session.season_calendar_year)

    cap_snap: Dict[str, Any] = {}
    cap_error: Optional[str] = None
    if user_team is not None:
        try:
            cap_snap = get_team_cap_snapshot_full(user_team, league, session.sim, season_year=season_year) or {}
        except Exception as exc:
            cap_error = str(exc) or "unknown error"

    evaluation = evaluate_roster_compliance(
        user_team,
        league=league,
        sim=session.sim,
        season_year=season_year,
        cap_snap=cap_snap,
        cap_error=cap_error,
    )
    capacity = evaluation.get("capacity") or {}
    slots = evaluation.get("contract_slots") or {}
    valid = bool(evaluation.get("valid"))

    refreshed = dict(payload)
    refreshed.update({
        "nhl_roster_count": evaluation.get("nhl_roster_count"),
        "forward_count": evaluation.get("forward_count"),
        "defense_count": evaluation.get("defense_count"),
        "goalie_count": evaluation.get("goalie_count"),
        "composition": capacity.get("composition"),
        "ir_count": evaluation.get("ir_count"),
        "ltir_count": evaluation.get("ltir_count"),
        "buried_count": capacity.get("buried_count"),
        "payroll_m": evaluation.get("payroll_m"),
        "cap_space_m": evaluation.get("cap_space_m"),
        "contract_slots_used": evaluation.get("contract_slots_used"),
        "contract_slots_limit": evaluation.get("contract_slots_limit"),
        "contract_slots_available": slots.get("available"),
        "blocking": list(evaluation.get("blocking") or []),
        "warnings": list(evaluation.get("warnings") or []),
        "issues": list(evaluation.get("issues") or []),
        "warning_messages": list(evaluation.get("warning_messages") or []),
        "valid": valid,
        "status": "ready" if valid else "blocking",
        "can_continue": valid,
        "blocking_reasons": list(evaluation.get("blocking_reasons") or []),
        "warning_reasons": list(evaluation.get("warning_reasons") or []),
        "available_actions": (
            ["generate_next_season", "back_to_hub"] if valid else ["resolve_issues", "open_cap_ledger", "back_to_hub"]
        ),
    })
    session.roster_cleanup_payload = refreshed
    session.next_important_event = "generate_next_season"
    return refreshed


def _run_roster_cleanup(session: FranchiseSession, *, force: bool = False) -> Dict[str, Any]:
    from services.contract_economy import (
        resolve_offer_sheets,
        run_cap_compliance_pipeline,
        run_prospect_promotion_pass,
        run_roster_fill_pass,
        get_team_cap_snapshot_full,
    )
    from services.roster_compliance import (
        ACTIVE_ROSTER_MAX,
        ACTIVE_ROSTER_MIN,
        MIN_DEFENSE,
        MIN_FORWARDS,
        MIN_GOALIES,
        evaluate_roster_compliance,
    )

    existing = getattr(session, "roster_cleanup_payload", None)
    if (
        not force
        and isinstance(existing, dict)
        and existing.get("version") == STAGE_PAYLOAD_VERSION["roster_cleanup"]
        and existing.get("valid") is not None
    ):
        # Re-validate against live rosters so Cap Ledger fixes unlock Generate, but
        # never replay the moves: promotions, waivers, buyouts and cap-casualty
        # trades all live below and must fire once per visit to this desk.
        return {"roster_cleanup": _revalidate_roster_cleanup(session, existing)}

    offer_sheets = resolve_offer_sheets(session)
    session.offer_sheet_resolutions = offer_sheets
    promo = run_prospect_promotion_pass(session)
    compliance = run_cap_compliance_pipeline(session, include_buyouts=True)
    # Trim first, then fill: relief can free the spots the floor pass needs.
    roster_fill = run_roster_fill_pass(session)
    user_team = session.team_by_id.get(session.user_team_id)
    league = getattr(session.sim, "league", None)
    sim = session.sim
    season_year = int(session.season_calendar_year)

    cap_snap: Dict[str, Any] = {}
    cap_error: Optional[str] = None
    if user_team is not None:
        try:
            cap_snap = get_team_cap_snapshot_full(user_team, league, sim, season_year=season_year) or {}
        except Exception as exc:
            cap_error = str(exc) or "unknown error"

    evaluation = evaluate_roster_compliance(
        user_team,
        league=league,
        sim=sim,
        season_year=season_year,
        cap_snap=cap_snap,
        cap_error=cap_error,
    )
    capacity = evaluation.get("capacity") or {}
    slots = evaluation.get("contract_slots") or {}
    valid = bool(evaluation.get("valid"))

    payload = {
        "version": STAGE_PAYLOAD_VERSION["roster_cleanup"],
        "nhl_roster_count": evaluation.get("nhl_roster_count"),
        "nhl_roster_max": ACTIVE_ROSTER_MAX,
        "nhl_roster_min": ACTIVE_ROSTER_MIN,
        "forward_count": evaluation.get("forward_count"),
        "defense_count": evaluation.get("defense_count"),
        "goalie_count": evaluation.get("goalie_count"),
        "min_forwards": MIN_FORWARDS,
        "min_defense": MIN_DEFENSE,
        "min_goalies": MIN_GOALIES,
        "composition": capacity.get("composition"),
        "ir_count": evaluation.get("ir_count"),
        "ltir_count": evaluation.get("ltir_count"),
        "buried_count": capacity.get("buried_count"),
        "payroll_m": evaluation.get("payroll_m"),
        "cap_space_m": evaluation.get("cap_space_m"),
        "contract_slots_used": evaluation.get("contract_slots_used"),
        "contract_slots_limit": evaluation.get("contract_slots_limit"),
        "contract_slots_available": slots.get("available"),
        "prospect_promotions": promo,
        "cap_compliance": {
            "buried": len(compliance.get("buried") or []),
            "waived": len(compliance.get("waived") or []),
            "claims": len(compliance.get("claims") or []),
            "cleared": len(compliance.get("cleared") or []),
            "buyouts": len(compliance.get("buyouts") or []),
        },
        "roster_fill": {
            "recalls": int(roster_fill.get("recall_count") or 0),
            "teams_filled": int(roster_fill.get("teams_filled") or 0),
            "unresolved": len(roster_fill.get("unresolved") or []),
        },
        "offer_sheets_resolved": offer_sheets.get("count", 0),
        "blocking": list(evaluation.get("blocking") or []),
        "warnings": list(evaluation.get("warnings") or []),
        "issues": list(evaluation.get("issues") or []),
        "warning_messages": list(evaluation.get("warning_messages") or []),
        "valid": valid,
        "status": "ready" if valid else "blocking",
        "can_continue": valid,
        "blocking_reasons": list(evaluation.get("blocking_reasons") or []),
        "warning_reasons": list(evaluation.get("warning_reasons") or []),
        "available_actions": (
            ["generate_next_season", "back_to_hub"] if valid else ["resolve_issues", "open_cap_ledger", "back_to_hub"]
        ),
    }
    session.roster_cleanup_payload = payload
    session.next_important_event = "generate_next_season"
    return {"roster_cleanup": payload}


def _roll_development_league_draft_class(session: FranchiseSession, season_year: int) -> Dict[str, Any]:
    """Age junior clubs one year, inject fresh draft-age talent, reset prospect season lines.

    Drafted players stay on their clubs for rights/development but are filtered out of
    the upcoming Prospect Board. Undrafted players age; new 16–17 year olds refill depth.
    """
    import random as _random

    from app.sim_engine.entities.player import Position
    from app.sim_engine.generation.prospect_league_scoring import initialize_prospect_season
    from app.sim_engine.league_hierarchy_bootstrap import _set_assignment, _spawn_player
    from services.franchise_sim import (
        _bump_prospect_revision,
        invalidate_session_payload_caches,
    )

    sim = getattr(session, "sim", None)
    league = getattr(sim, "league", None) if sim else None
    if league is None or sim is None:
        return {"ok": False, "error": "no league"}

    rng = getattr(sim, "rng", None) or _random.Random(int(season_year) * 9973)
    try:
        from app.sim_engine.league_hierarchy_bootstrap import set_spawn_as_of_year

        set_spawn_as_of_year(int(season_year))
    except Exception:
        pass
    aged = 0
    reset_stats = 0
    injected = 0
    culled = 0
    used_names: set = set()
    league_players = list(getattr(league, "players", None) or [])
    for p in league_players:
        ident = getattr(p, "identity", None)
        nm = str(getattr(ident, "name", "") or "")
        if nm:
            used_names.add(nm)

    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "")
        for tm in block.get("teams") or []:
            players = list(tm.get("players") or [])
            kept: List[Any] = []
            for p in players:
                if getattr(p, "retired", False):
                    culled += 1
                    continue
                ident = getattr(p, "identity", None)
                try:
                    from services.franchise_sim import sync_player_age_to_session

                    sync_player_age_to_session(p, session)
                    aged += 1
                except Exception:
                    try:
                        if ident is not None and hasattr(ident, "age"):
                            ident.age = int(getattr(ident, "age", 17) or 17) + 1
                        else:
                            p.age = int(getattr(p, "age", 17) or 17) + 1
                        aged += 1
                    except Exception:
                        pass
                age_now = 99
                try:
                    age_now = int(getattr(ident, "age", 99) or 99) if ident else int(getattr(p, "age", 99) or 99)
                except Exception:
                    age_now = 99
                # Age out undrafted overagers from junior clubs.
                drafted = bool(getattr(p, "drafted", False)) or bool(
                    getattr(p, "nhl_rights_team_id", None) or getattr(p, "rights_team_id", None)
                )
                if (not drafted) and age_now > 20:
                    culled += 1
                    continue
                try:
                    initialize_prospect_season(
                        p,
                        code,
                        rng=rng,
                        season_year=int(season_year),
                        calendar_iso=None,
                        force=True,
                    )
                    reset_stats += 1
                except Exception:
                    try:
                        setattr(p, "_prospect_season_stats", None)
                        setattr(p, "_prospect_season_year", int(season_year))
                        setattr(p, "_prospect_last_stat_update_iso", "")
                    except Exception:
                        pass
                kept.append(p)

            # Target ~18 skaters + 2 goalies per junior club; refill with new draft-age kids.
            goalies = sum(
                1
                for p in kept
                if str(getattr(getattr(p, "identity", None), "position", "") or "").upper().endswith("G")
            )
            need_g = max(0, 2 - goalies)
            need_sk = max(0, 18 - (len(kept) - goalies))
            for _ in range(need_g + need_sk):
                is_g = need_g > 0
                if is_g:
                    need_g -= 1
                    pos = Position.G
                    # Match bootstrap junior bands; pipeline shaping adds star power.
                    ovr_lo, ovr_hi = 0.30, 0.48
                else:
                    need_sk -= 1
                    pos = rng.choice([Position.C, Position.LW, Position.RW, Position.D])
                    roll = int(rng.randint(1, 100))
                    if roll <= 6:
                        ovr_lo, ovr_hi = 0.44, 0.54
                    elif roll <= 22:
                        ovr_lo, ovr_hi = 0.38, 0.50
                    elif roll <= 55:
                        ovr_lo, ovr_hi = 0.34, 0.46
                    else:
                        ovr_lo, ovr_hi = 0.30, 0.42
                try:
                    newbie = _spawn_player(
                        rng,
                        pos=pos,
                        ovr_lo=ovr_lo,
                        ovr_hi=ovr_hi,
                        # Next-year CHL intake: 17–18 first-year kids, not overagers.
                        age_lo=17,
                        age_hi=18,
                        used_names=used_names,
                        league_players=league_players,
                        pool_context="junior",
                        league_code=code,
                    )
                    _set_assignment(
                        newbie,
                        level="junior",
                        league_code=code,
                        club=str(tm.get("name") or ""),
                    )
                    try:
                        newbie.context.current_team_id = str(tm.get("team_id") or "")
                    except Exception:
                        pass
                    try:
                        initialize_prospect_season(
                            newbie,
                            code,
                            rng=rng,
                            season_year=int(season_year),
                            calendar_iso=None,
                            force=True,
                        )
                    except Exception:
                        pass
                    kept.append(newbie)
                    league_players.append(newbie)
                    injected += 1
                except Exception:
                    pass
            tm["players"] = kept

    try:
        league.players = league_players
    except Exception:
        pass

    # Rebuild NHL-scale star tiers from junior-raw injects (same as franchise start).
    try:
        from app.sim_engine.league_hierarchy_bootstrap import _shape_draft_class_pipeline

        _shape_draft_class_pipeline(league, rng)
    except Exception:
        pass

    # Keep the unsigned global pool aging so future drafts aren't a stale cohort.
    try:
        if hasattr(sim, "_advance_global_prospect_season"):
            sim._advance_global_prospect_season(int(season_year), rng)
    except Exception:
        pass

    session._prospect_stats_synced_iso = ""
    session._prospect_sync_rows = None
    session._prospect_sync_cache_key = None
    session._prospect_retune_v4_applied = False
    session.draft_rank_prev = {}
    session.draft_preseason_rank = {}
    session.draft_midseason_rank = {}
    try:
        _bump_prospect_revision(session)
        invalidate_session_payload_caches(session, reason="season_reset")
    except Exception:
        pass

    return {
        "ok": True,
        "aged": aged,
        "stats_reset": reset_stats,
        "injected": injected,
        "culled": culled,
        "season_year": int(season_year),
    }


def _ensure_undrafted_draft_depth(session: FranchiseSession, season_year: int) -> Dict[str, Any]:
    """Inject draft-age kids when the undrafted board is thin (no re-aging)."""
    import random as _random

    from app.sim_engine.entities.player import Position
    from app.sim_engine.generation.prospect_league_scoring import initialize_prospect_season
    from app.sim_engine.league_hierarchy_bootstrap import _set_assignment, _spawn_player

    sim = getattr(session, "sim", None)
    league = getattr(sim, "league", None) if sim else None
    if league is None or sim is None:
        return {"ok": False}

    undrafted = 0
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if getattr(p, "retired", False):
                    continue
                if bool(getattr(p, "drafted", False)):
                    continue
                if str(
                    getattr(p, "nhl_rights_team_id", None)
                    or getattr(p, "rights_team_id", None)
                    or getattr(p, "drafted_by", None)
                    or ""
                ).strip():
                    continue
                ident = getattr(p, "identity", None)
                age = int(getattr(ident, "age", 99) or 99) if ident else 99
                if age <= 20:
                    undrafted += 1
    if undrafted >= 180:
        return {"ok": True, "undrafted": undrafted, "injected": 0}

    rng = getattr(sim, "rng", None) or _random.Random(int(season_year) * 4243)
    try:
        from app.sim_engine.league_hierarchy_bootstrap import set_spawn_as_of_year

        set_spawn_as_of_year(int(season_year))
    except Exception:
        pass
    used_names: set = set()
    league_players = list(getattr(league, "players", None) or [])
    for p in league_players:
        ident = getattr(p, "identity", None)
        nm = str(getattr(ident, "name", "") or "")
        if nm:
            used_names.add(nm)

    injected = 0
    need = max(0, 220 - undrafted)
    targets: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            if isinstance(tm, dict):
                targets.append((block, tm))
    if not targets:
        return {"ok": True, "undrafted": undrafted, "injected": 0}

    for i in range(need):
        block, tm = targets[i % len(targets)]
        code = str(block.get("league_code") or "CHL_OHL")
        is_g = (i % 12) == 0
        pos = Position.G if is_g else rng.choice([Position.C, Position.LW, Position.RW, Position.D])
        try:
            newbie = _spawn_player(
                rng,
                pos=pos,
                # Junior-raw bands only — never NHL-starter OVRs on inject.
                ovr_lo=0.30 if is_g else 0.32,
                ovr_hi=0.48 if is_g else 0.52,
                age_lo=17,
                age_hi=19,
                used_names=used_names,
                league_players=league_players,
                pool_context="junior",
                league_code=code,
            )
            _set_assignment(newbie, level="junior", league_code=code, club=str(tm.get("name") or ""))
            try:
                newbie.context.current_team_id = str(tm.get("team_id") or "")
            except Exception:
                pass
            try:
                initialize_prospect_season(
                    newbie, code, rng=rng, season_year=int(season_year), force=True
                )
            except Exception:
                pass
            roster = list(tm.get("players") or [])
            roster.append(newbie)
            tm["players"] = roster
            league_players.append(newbie)
            injected += 1
        except Exception:
            continue
    try:
        league.players = league_players
    except Exception:
        pass
    try:
        from app.sim_engine.league_hierarchy_bootstrap import _shape_draft_class_pipeline

        _shape_draft_class_pipeline(league, rng)
    except Exception:
        pass
    try:
        from services.franchise_sim import _bump_prospect_revision, invalidate_session_payload_caches

        _bump_prospect_revision(session)
        invalidate_session_payload_caches(session, reason="season_reset")
    except Exception:
        pass
    return {"ok": True, "undrafted": undrafted, "injected": injected}


def _retune_inflated_underage_prospects(session: FranchiseSession) -> Dict[str, Any]:
    """One-shot: crush NHL-starter OVRs on pre-draft-age kids left by bad year-roll injects."""
    if bool(getattr(session, "_underage_ovr_retune_v1", False)):
        return {"ok": True, "skipped": True}

    import random as _random

    from app.sim_engine.league_hierarchy_bootstrap import (
        _apply_shaped_player,
        _player_ovr_frac,
        _shape_draft_class_pipeline,
    )

    sim = getattr(session, "sim", None)
    league = getattr(sim, "league", None) if sim else None
    if league is None:
        setattr(session, "_underage_ovr_retune_v1", True)
        return {"ok": False, "error": "no league"}

    rng = getattr(sim, "rng", None) or _random.Random(7711)
    fixed = 0
    code_by_id: Dict[int, str] = {}
    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "JUNIOR")
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if getattr(p, "retired", False):
                    continue
                if bool(getattr(p, "drafted", False)):
                    continue
                if str(
                    getattr(p, "nhl_rights_team_id", None)
                    or getattr(p, "rights_team_id", None)
                    or getattr(p, "drafted_by", None)
                    or ""
                ).strip():
                    continue
                ident = getattr(p, "identity", None)
                try:
                    age = int(getattr(ident, "age", 99) or 99) if ident else 99
                except Exception:
                    age = 99
                if age > 16:
                    continue
                try:
                    ovr = float(_player_ovr_frac(p))
                except Exception:
                    continue
                # Age 16 should never sit near NHL starter ability.
                if ovr <= 0.50 + 1e-6:
                    continue
                code_by_id[id(p)] = code
                try:
                    _apply_shaped_player(
                        p,
                        tier="pool",
                        lo=0.34,
                        hi=0.48,
                        pot_lo=78,
                        pot_hi=92,
                        rng=rng,
                        code_by_id=code_by_id,
                        rng_inst=rng,
                    )
                    fixed += 1
                except Exception:
                    continue

    if fixed:
        try:
            _shape_draft_class_pipeline(league, rng)
        except Exception:
            pass
        try:
            from services.franchise_sim import _bump_prospect_revision, invalidate_session_payload_caches

            _bump_prospect_revision(session)
            invalidate_session_payload_caches(session, reason="underage_ovr_retune")
        except Exception:
            pass

    setattr(session, "_underage_ovr_retune_v1", True)
    return {"ok": True, "fixed": fixed}


def generate_next_season(session: FranchiseSession) -> Dict[str, Any]:
    """Build new schedule/calendar — only increments year when data exists."""
    from services.contract_economy import run_cap_compliance_before_season
    # Re-validate roster before generation; never generate with blocking issues.
    cleanup = _run_roster_cleanup(session, force=True)
    payload = (cleanup or {}).get("roster_cleanup") or session.roster_cleanup_payload or {}
    if not payload.get("valid", False):
        reasons = payload.get("blocking_reasons") or payload.get("issues") or ["Roster not compliant"]
        raise ValueError("Cannot generate next season: " + "; ".join(str(r) for r in reasons[:4]))
    run_cap_compliance_before_season(session)
    from app.sim_engine.league import generate_regular_season_schedule
    from app.sim_engine.league.schedule_generator import _safe_team_id
    from app.sim_engine.league.standings import StandingsTable
    from services.franchise_sim import (
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
    source_sy = int(session.season_calendar_year)
    source_phase = str(getattr(session, "phase", "") or "")
    if getattr(session, "_audit_lifecycle_trace", None) is not None or getattr(session, "_audit_schedule_invariants", False):
        trace_row = {
            "transition": "generate_next_season",
            "source_phase": source_phase,
            "source_season_year": source_sy,
            "source_calendar_year": source_sy,
            "event": "season_year_increment",
            "destination_phase": "offseason_roster_cleanup",
            "destination_season_year": next_sy,
            "destination_calendar_year": next_sy,
        }
        buf = list(getattr(session, "_audit_lifecycle_trace", None) or [])
        buf.append(trace_row)
        session._audit_lifecycle_trace = buf

    history_entry = {
        "season_year": int(session.season_calendar_year),
        "champion_id": session.champion_id,
        "game_results_count": len(getattr(session, "game_results", None) or []),
        "draft_results": list((getattr(session, "draft_state", None) or {}).get("draft_results") or []),
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
    session.processed_game_ids = set()
    session._regular_stats_split_done = False
    session.regular_season_complete = False
    session.playoffs_generated = False
    session.playoffs_simulated = False
    session.playoffs_done = False
    session.playoff_payload = {}
    session.champion_id = None
    session.stanley_cup_winner = None
    session.preseason_applied = False
    # Season-scoped lifecycle flags that previously leaked across years.
    session._year_end_progression_done = False
    session.contracts_ticked = False
    session.development_report_done = False
    session.development_report_completed_season = 0
    session.development_report_generated_at = ""
    # Final Skate is latched by retirements_processed; leaving it set meant no
    # club retired anyone from year two onward.
    session.retirements_processed = False
    session.retirements_payload = {}
    session.awards_generated = False
    session.awards_payload = {}
    session.offer_sheet_resolutions = {}
    # Reset per-team goalie workload state so Year 2+ does not inherit fatigue.
    try:
        for team in teams:
            if hasattr(team, "_gm_goalie_start_state"):
                setattr(team, "_gm_goalie_start_state", {})
            if hasattr(team, "_gm_goalie_usage_strategy"):
                try:
                    delattr(team, "_gm_goalie_usage_strategy")
                except Exception:
                    setattr(team, "_gm_goalie_usage_strategy", None)
    except Exception:
        pass
    # Clear stamped season production so year-2 GP cannot inherit year-1 totals.
    try:
        from services.franchise_sim import _iter_league_players_for_aging

        league_clear = getattr(sim, "league", None)
        for pl in _iter_league_players_for_aging(league_clear) if league_clear is not None else []:
            for attr in (
                "games_played",
                "gp",
                "nhl_games_played_this_season",
                "goals",
                "assists",
                "points",
            ):
                try:
                    if hasattr(pl, attr):
                        setattr(pl, attr, 0)
                except Exception:
                    pass
    except Exception:
        pass
    # Keep next_season_generated False until payload is ready below; cleared again
    # when the new season actually starts (_finalize_next_season_reveal).
    session.season_calendar_year = next_sy
    # Re-sync ages to Sept 15 of the new season year (birth-date accurate).
    try:
        from services.franchise_sim import _iter_league_players_for_aging, sync_player_age_to_season

        league_obj = getattr(sim, "league", None)
        for pl in _iter_league_players_for_aging(league_obj) if league_obj is not None else []:
            sync_player_age_to_season(pl, next_sy)
    except Exception:
        pass
    session.draft_completed = False
    session.draft_lottery_done = False
    session.draft_lottery_payload = {}
    session.draft_combine_done = False
    session.draft_combine_payload = {}
    session.draft_payload = {}
    session.draft_state = {}
    session.draft_review_payload = {}
    session.prospect_rights_payload = {}
    session.draft_rights_review_payload = {}
    session.resign_payload = {}
    session.free_agency_open = False
    session.free_agents_payload = []
    session.free_agency_market_payload = {}
    session.fa_market_book = {}
    session.fa_market_day = 0
    try:
        from app.sim_engine.trades.trade_pick_registry import ensure_franchise_pick_registry, upcoming_draft_year

        league = getattr(sim, "league", None)
        if league is not None:
            setattr(league, "season_year", next_sy)
            setattr(league, "current_season", next_sy)
            setattr(league, "draft_year", upcoming_draft_year(next_sy))
            setattr(league, "season_is_calendar", True)
            ensure_franchise_pick_registry(league, season_calendar_year=next_sy, years_ahead=4)
    except Exception:
        pass
    session.cpu_fa_signings = {}
    session.cpu_rfa_decisions = {}
    session.cpu_fa_wave = 0
    session.roster_cleanup_payload = {}
    session.offseason_completed_stages = []
    session.offseason_stage_entered_at = {}
    session.offseason_stage_completed_at = {}
    session.draft_rank_prev = {}
    session.draft_preseason_rank = {}
    session.draft_midseason_rank = {}
    session.draft_rank_snapshot_week = ""

    # World Juniors is season-scoped. Leaving last year's completed bundle /
    # loan latch made September still look like a finished WJC and blocked
    # the next Christmas loan prompts.
    session.wjc_tournament_bundle = None
    session.wjc_loan_prompts_enqueued = False
    session.wjc_nhl_u20_loan = {}
    session.wjc_draft_score_boosts = {}
    try:
        arch = list(getattr(session, "showcase_archive", None) or [])
        session.showcase_archive = [
            a
            for a in arch
            if not (
                isinstance(a, dict)
                and (a.get("kind") == "wjc_tournament" or a.get("wjc_live") or a.get("wjc_phase"))
            )
        ]
    except Exception:
        pass

    # Age juniors, inject a fresh undrafted draft-age class, and zero prospect
    # season lines so the Prospect Board is not last year's class with 60+ GP.
    draft_roll: Dict[str, Any] = {}
    try:
        draft_roll = _roll_development_league_draft_class(session, next_sy)
        session._draft_class_roll_year = int(next_sy)
    except Exception as exc:
        draft_roll = {"ok": False, "error": str(exc)}

    # Drop last year's lifecycle cinematics so Enter Preseason lands on the hub,
    # not a stale awards night / playoff bracket popup.
    _scrub_lifecycle_popups_for_new_season(session)
    try:
        session.playoff_live = None
    except Exception:
        pass
    if hasattr(session, "playoff_live"):
        try:
            delattr(session, "playoff_live")
        except Exception:
            session.playoff_live = None
    _clear_trade_acquisition_cooldowns(session, teams)
    try:
        from services.league_operations import invalidate_league_ops_cache
        invalidate_league_ops_cache(session)
    except Exception:
        pass

    first_opp = ""
    uid = str(session.user_team_id)
    for slot in schedule[:40]:
        hid = str(getattr(slot, "home_id", getattr(slot, "home_team_id", "")) or "")
        aid = str(getattr(slot, "away_id", getattr(slot, "away_team_id", "")) or "")
        if uid in (hid, aid):
            opp_id = aid if hid == uid else hid
            opp = session.team_by_id.get(opp_id)
            from services.franchise_sim import _display_team
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
        "calendar_markers": [
            {"key": a.get("key"), "iso": a.get("iso"), "label": a.get("label") or a.get("name")}
            for a in (anchors or [])[:12]
            if isinstance(a, dict)
        ],
        "salary_cap_m": (getattr(session, "salary_cap_payload", None) or {}).get("new_season_cap"),
        "generation_status": "ready",
        "stage_status": "ready",
        "can_continue": True,
        "blocking_reasons": [],
        "warning_reasons": [],
        "available_actions": ["enter_preseason", "back_to_hub"],
        "draft_class_roll": draft_roll,
    }
    session.next_season_payload = payload
    session.next_season_generated = True
    _mark_stage_completed(session, "roster_cleanup")
    # Seamless handoff: skip the reveal cinematic and park the club in September camp.
    # Hub world opens with the new calendar; players are already aged from year-end.
    session.offseason_stage = "next_season_reveal"
    _mark_stage_entered(session, "next_season_reveal")
    _finalize_next_season_reveal(session)
    session.timeline.append(f"NEW SEASON: {next_sy}–{next_sy + 1} schedule generated — camp opens.")
    invalidate_session_payload_caches(session, "next_season")
    return {"next_season": payload, "status": "preseason", "season_phase": "preseason"}


def _scrub_lifecycle_popups_for_new_season(session: FranchiseSession) -> None:
    """Remove awards / playoff / prior-year offseason stage popups after rollover."""
    drop_kinds = {
        "playoff_start",
        "awards",
        "retirements",
        "salary_cap",
        "development_report",
        "draft_lottery",
        "draft_combine",
        "draft",
        "draft_review",
        "prospect_rights",
        "re_sign",
        "free_agency",
        "roster_cleanup",
        "next_season_reveal",
        "stanley_cup",
        "playoffs",
    }
    pops = list(getattr(session, "pending_ui_popups", None) or [])
    kept = []
    for p in pops:
        if not isinstance(p, dict):
            continue
        kind = str(p.get("kind") or p.get("type") or "").lower()
        eid = str(p.get("id") or "").lower()
        if kind in drop_kinds or any(k in eid for k in ("awards", "playoff", "stanley", "offseason")):
            continue
        if p.get("playoff_live") or p.get("wjc_live"):
            continue
        kept.append(p)
    session.pending_ui_popups = kept


def _clear_trade_acquisition_cooldowns(session: FranchiseSession, teams: Optional[List[Any]] = None) -> int:
    """Wipe acquisition stamps so year-N deals cannot lock year-N+1 trades.

    ``calendar_cursor`` resets to 0 on rollover; without clearing ``last_acquired_day``
    the cooldown math ``(0 - old_day) < 7`` is always true.
    """
    cleared = 0
    league = getattr(getattr(session, "sim", None), "league", None)
    team_list = list(teams if teams is not None else (getattr(league, "teams", None) or []))
    pools = []
    for team in team_list:
        for attr in ("roster", "ahl_roster", "echl_roster", "prospect_pool"):
            pools.append(list(getattr(team, attr, None) or []))
    pools.append(list(getattr(league, "free_agents", None) or []))
    for pool in pools:
        for p in pool:
            if p is None:
                continue
            if not (
                getattr(p, "acquired_via_trade", False)
                or getattr(p, "last_acquired_day", None) is not None
                or getattr(p, "last_acquired_date", None)
            ):
                continue
            try:
                p.acquired_via_trade = False
                p.last_acquired_day = None
                p.last_acquired_date = None
                p.acquired_via_trade_season = None
                p.acquired_from_team_id = None
                cleared += 1
            except Exception:
                continue
    return cleared


def _scrub_lines_of_departed_players(session: FranchiseSession) -> int:
    """Blank saved line slots holding players who retired or left the club.

    Deploy only warns about stale ids, so a lineup saved before the offseason
    silently carries last year's roster into camp. Only ids that are provably
    players — and provably no longer ours — are cleared.
    """
    lines = getattr(session, "lines", None)
    if not isinstance(lines, dict) or not lines:
        return 0

    user_team = session.team_by_id.get(str(session.user_team_id))
    ours = {
        str(getattr(p, "id", "") or "")
        for p in (getattr(user_team, "roster", None) or [])
        if getattr(p, "id", None) and not getattr(p, "retired", False)
    }
    if not ours:
        return 0

    known: set = set()
    league = getattr(session.sim, "league", None)
    for team in list(getattr(league, "teams", None) or []):
        for attr in ("roster", "ahl_roster", "echl_roster", "prospect_pool"):
            for p in list(getattr(team, attr, None) or []):
                pid = str(getattr(p, "id", "") or "")
                if pid:
                    known.add(pid)
    for p in list(getattr(league, "free_agents", None) or []):
        pid = str(getattr(p, "id", "") or "")
        if pid:
            known.add(pid)

    departed = {pid for pid in known if pid not in ours}
    if not departed:
        return 0

    cleared = 0

    def _walk(node: Any) -> Any:
        nonlocal cleared
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, str) and node in departed:
            cleared += 1
            return ""
        return node

    session.lines = _walk(lines)
    return cleared


def _finalize_next_season_reveal(session: FranchiseSession) -> Dict[str, Any]:
    _mark_stage_completed(session, "next_season_reveal")
    session.phase = "preseason"
    session.season_phase = "preseason"
    session.offseason_stage = None
    session.next_important_event = "preseason_start"
    # Season has begun — clear the generation latch so the next offseason can regenerate.
    session.next_season_generated = False
    # generate_next_season still reads last year's cap row to label the reveal, so
    # the salary-cap desk is only cleared once the reveal is behind us.
    session.salary_cap_payload = {}
    _scrub_lines_of_departed_players(session)
    return {"next_season": session.next_season_payload, "season_phase": "preseason"}


def slim_awards_payload_for_client(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Awards Night needs winners + a short nominee list — not triple-copied full ballots.
    Full compute payloads routinely exceed 10MB and crash the browser as a Network Error
    right when playoffs finish / offseason opens.
    """
    if not isinstance(payload, dict):
        return {}

    candidate_cap = 5
    drop_row_keys = frozenset(
        {
            "component_scores",
            "eligibility",
            "result",
            "votes_detail",
            "raw_components",
            "debug",
            "breakdown",
            "vote_share",
            "ballot_share",
        }
    )
    drop_award_keys = frozenset({"voting", "result", "frozen_inputs", "ballots", "voters"})

    def _slim_entity(row: Any) -> Any:
        if not isinstance(row, dict):
            return row
        out = {k: v for k, v in row.items() if k not in drop_row_keys}
        for rk in ("public_rationale", "rationale", "blurb", "summary"):
            if rk in out and isinstance(out[rk], str) and len(out[rk]) > 220:
                out[rk] = out[rk][:217] + "..."
        return out

    def _slim_award(ser: Any) -> Any:
        if not isinstance(ser, dict):
            return ser
        out = {k: v for k, v in ser.items() if k not in drop_award_keys}
        full = [_slim_entity(x) for x in list(out.get("full_results") or [])[:candidate_cap]]
        out["full_results"] = full
        # Avoid shipping candidates + full_results + finalists as three near-copies.
        out["candidates"] = list(full)
        finals = list(out.get("finalists") or [])
        if finals and isinstance(finals[0], dict):
            out["finalists"] = [_slim_entity(x) for x in finals[: min(3, candidate_cap)]]
        else:
            out["finalists"] = list(full[:3])
        winners = list(out.get("winners") or [])
        if winners and isinstance(winners[0], dict):
            out["winners"] = [_slim_entity(x) for x in winners[:3]]
        for rk in ("public_rationale", "rationale"):
            if rk in out and isinstance(out[rk], str) and len(out[rk]) > 320:
                out[rk] = out[rk][:317] + "..."
        return out

    official = [_slim_award(x) for x in list(payload.get("official_results") or [])]
    if not official:
        # Legacy shape: awards dict / items list only.
        legacy = payload.get("items")
        if isinstance(legacy, list) and legacy:
            official = [_slim_award(x) for x in legacy]
        elif isinstance(payload.get("awards"), dict):
            official = [_slim_award(v) for v in payload["awards"].values()]

    team_achievements = [_slim_award(x) for x in list(payload.get("team_achievements") or [])]
    all_star_raw = payload.get("all_star_teams") or {}
    all_star_teams = (
        {k: _slim_award(v) for k, v in all_star_raw.items()}
        if isinstance(all_star_raw, dict)
        else {}
    )

    # Thin legacy map — winner fields only (UI prefers official_results).
    thin_awards: Dict[str, Any] = {}
    for row in official:
        if not isinstance(row, dict):
            continue
        key = str(row.get("award_id") or row.get("name") or "").strip()
        if not key:
            continue
        thin_awards[key] = {
            "name": row.get("name"),
            "award_id": row.get("award_id"),
            "winner_name": row.get("winner_name"),
            "winner_player_id": row.get("winner_player_id"),
            "winner_team_id": row.get("winner_team_id"),
            "winner_team_name": row.get("winner_team_name"),
            "status": row.get("status"),
            "winners": row.get("winners") or [],
            "finalists": row.get("finalists") or [],
            "public_rationale": row.get("public_rationale") or row.get("rationale"),
            "display_metric": row.get("display_metric"),
            "recipient_type": row.get("recipient_type"),
        }

    meta = dict(payload.get("metadata") or {})
    meta.pop("frozen_inputs", None)
    seed_val = meta.get("seed")
    if seed_val is not None and (callable(seed_val) or not isinstance(seed_val, (int, float, str))):
        meta["seed"] = int(payload.get("season") or 0) & 0xFFFFFFFF
    elif isinstance(seed_val, float):
        meta["seed"] = int(seed_val) & 0xFFFFFFFF

    return {
        "season": payload.get("season"),
        "status": payload.get("status") or "complete",
        "official_results": official,
        "ceremony": payload.get("ceremony") or {},
        "metadata": meta,
        "team_achievements": team_achievements,
        "all_star_teams": all_star_teams,
        "awards": thin_awards,
        "items": official,
    }


def build_offseason_state_extras(session: FranchiseSession, *, lean: bool = False, hydrate_stages: bool = False) -> Dict[str, Any]:
    """Extra payload fields for build_state_payload."""
    _sync_phase_fields(session)
    if hydrate_stages:
        _ensure_offseason_stage_hydrated(session)
    phase = str(session.phase)
    stage = getattr(session, "offseason_stage", None)
    completed = list(getattr(session, "offseason_completed_stages", None) or [])

    can_advance = (
        phase in ("preseason", "regular")
        and len(getattr(session, "pending_decisions", None) or []) == 0
        and phase not in ("post_cup", "offseason")
    )
    can_continue_offseason = phase in ("post_cup", "offseason")
    roster_payload = session.roster_cleanup_payload or {}
    can_generate = (
        phase == "offseason"
        and stage == "roster_cleanup"
        and not session.next_season_generated
        and bool(roster_payload.get("valid", True))
    )
    is_terminal = False

    stage_idx = None
    try:
        if stage in OFFSEASON_STAGES:
            stage_idx = OFFSEASON_STAGES.index(str(stage))
    except Exception:
        stage_idx = None
    post_idx = None
    try:
        if stage in POST_DRAFT_STAGES:
            post_idx = POST_DRAFT_STAGES.index(str(stage))
    except Exception:
        post_idx = None

    # Pull stage-local gate fields when present.
    stage_blob = None
    if stage == "draft_review":
        stage_blob = getattr(session, "draft_review_payload", None)
    elif stage == "prospect_rights":
        stage_blob = getattr(session, "prospect_rights_payload", None)
    elif stage == "re_sign":
        stage_blob = getattr(session, "resign_payload", None)
    elif stage == "free_agency":
        stage_blob = getattr(session, "free_agency_market_payload", None)
    elif stage == "roster_cleanup":
        stage_blob = roster_payload
    elif stage == "next_season_reveal":
        stage_blob = session.next_season_payload
    stage_blob = stage_blob if isinstance(stage_blob, dict) else {}

    blocking = list(stage_blob.get("blocking_reasons") or [])
    warnings = list(stage_blob.get("warning_reasons") or [])
    can_continue_stage = bool(stage_blob.get("can_continue", True)) if stage_blob else True
    if stage == "roster_cleanup":
        can_continue_stage = bool(roster_payload.get("valid", False))

    timeline = {
        "season": int(getattr(session, "season_calendar_year", 0) or 0),
        "current_stage": stage,
        "previous_stage": completed[-1] if completed else None,
        "next_stage": STAGE_NEXT_EVENT.get(str(stage or ""), None),
        "stage_index": stage_idx,
        "total_stages": len(OFFSEASON_STAGES),
        "post_draft_index": post_idx,
        "post_draft_total": len(POST_DRAFT_STAGES),
        "completed_stages": completed,
        "current_stage_status": stage_blob.get("stage_status") or stage_blob.get("status") or (
            "ready" if stage else None
        ),
        "can_continue": can_continue_stage,
        "blocking_reasons": blocking,
        "warning_reasons": warnings,
        "resume_available": can_continue_offseason and bool(stage or phase == "post_cup"),
        "primary_action": (
            "generate_next_season" if stage == "roster_cleanup"
            else "enter_preseason" if stage == "next_season_reveal"
            else "continue_offseason" if can_continue_offseason else "advance_day"
        ),
        "secondary_actions": list(stage_blob.get("available_actions") or ["back_to_hub"]),
        "entered_at": (getattr(session, "offseason_stage_entered_at", None) or {}).get(str(stage or "")),
        "completed_at": (getattr(session, "offseason_stage_completed_at", None) or {}).get(str(stage or "")),
        "is_complete": str(stage or "") in completed,
    }

    payload = {
        "offseason_stage": stage,
        "offseason_timeline": timeline,
        "playoffs_done": bool(getattr(session, "playoffs_done", session.playoffs_simulated)),
        "stanley_cup_winner": session.stanley_cup_winner or session.champion_id,
        "awards": slim_awards_payload_for_client(session.awards_payload),
        "retirements": session.retirements_payload,
        "retired_players_archive": list(getattr(session, "retired_players_archive", None) or []),
        "draft_lottery": session.draft_lottery_payload,
        "draft_combine": session.draft_combine_payload,
        "draft": session.draft_payload,
        "draft_review": getattr(session, "draft_review_payload", None),
        "prospect_rights": getattr(session, "prospect_rights_payload", None),
        "free_agency_open": bool(getattr(session, "free_agency_open", False)),
        "fa_market_day": int(getattr(session, "fa_market_day", 0) or 0),
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
    if lean:
        fa_market = getattr(session, "free_agency_market_payload", None) or {}
        fa_agents = session.free_agents_payload or {}
        if isinstance(fa_agents, dict):
            agent_list = fa_agents.get("agents") or fa_agents.get("free_agents") or []
            payload["free_agents_count"] = len(agent_list) if isinstance(agent_list, list) else 0
        else:
            payload["free_agents_count"] = 0
        payload["free_agency_market_summary"] = {
            "day": int(getattr(session, "fa_market_day", 0) or 0),
            "open": bool(getattr(session, "free_agency_open", False)),
            "available_count": len(list((fa_market or {}).get("available") or (fa_market or {}).get("free_agents") or [])),
        }
        return payload
    payload["free_agents"] = session.free_agents_payload
    payload["free_agency_market"] = getattr(session, "free_agency_market_payload", None)
    return payload
