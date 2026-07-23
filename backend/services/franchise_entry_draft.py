"""
NHL Entry Draft execution for franchise mode offseason.

SOURCE OF TRUTH: backend/services/franchise_entry_draft.py (live FastAPI franchise
sessions). SimEngine/app/sim_engine/franchise/ is a legacy mirror — do not duplicate
draft execution there. Calendar, prospect stats, and roster assignment flow through
backend/services/franchise_sim.py wrapping the live SimEngine league object.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from services.franchise_session import FranchiseSession

# Guards creation of per-session draft locks.
_DRAFT_LOCK_REGISTRY_GUARD = threading.Lock()
_DRAFT_LOCKS: "Dict[str, threading.Lock]" = {}


def _draft_lock(session: FranchiseSession) -> threading.Lock:
    """Return a process-wide lock unique to this franchise session.

    Serializes pick execution so two concurrent requests can't both read the same
    current pick, pass validation, and then mutate draft state / registry / rights.
    """
    key = str(getattr(session, "session_id", None) or id(session))
    with _DRAFT_LOCK_REGISTRY_GUARD:
        lock = _DRAFT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DRAFT_LOCKS[key] = lock
        return lock

DRAFT_ROUNDS = 7
DRAFT_TEAMS = 32
PICKS_PER_ROUND = DRAFT_TEAMS
TOTAL_PICKS = DRAFT_ROUNDS * PICKS_PER_ROUND

DRAFT_PHILOSOPHIES = (
    "bpa_heavy",
    "need_focused",
    "safe_floor",
    "high_upside",
    "boom_bust_gambler",
    "analytics_driven",
    "defense_first",
    "center_priority",
    "goalie_tolerant",
    "off_board_scout",
    "rebuilder_upside",
    "contender_timeline",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _display_team(session: FranchiseSession, team_id: str) -> str:
    from services.franchise_sim import _display_team as _dt

    tm = session.team_by_id.get(str(team_id))
    return _dt(tm) if tm else str(team_id)


def _rng_float(session: FranchiseSession, *parts: Any) -> float:
    seed = hashlib.md5(
        f"{session.session_id}:{':'.join(str(p) for p in parts)}".encode()
    ).hexdigest()
    return int(seed[:8], 16) / 0xFFFFFFFF


def _has_playoff_results(session: FranchiseSession) -> bool:
    payload = getattr(session, "playoff_payload", None) or {}
    return bool(
        session.champion_id
        or session.stanley_cup_winner
        or payload.get("champion_id")
        or payload.get("series_list")
    )


def _playoff_team_pick_order(session: FranchiseSession, lottery_ids: List[str]) -> Tuple[List[str], str]:
    """Order playoff-team slots 17–32: earlier elimination → earlier pick (NHL-style buckets)."""
    from services.franchise_sim import _build_standings_rows

    standings = sorted(
        _build_standings_rows(session),
        key=lambda r: (int(r.get("pts", 0)), -int(r.get("w", 0))),
    )
    lottery_set = set(lottery_ids)
    playoff_teams = [str(r["team_id"]) for r in standings if str(r["team_id"]) not in lottery_set]
    pts_map = {str(r["team_id"]): int(r.get("pts", 0)) for r in standings}
    div_winner = {
        str(r["team_id"]): bool(r.get("division_winner") or r.get("is_division_winner"))
        for r in standings
    }

    if not _has_playoff_results(session):
        return playoff_teams[:16], "standings_fallback"

    payload = getattr(session, "playoff_payload", None) or {}
    champion = str(
        session.champion_id or session.stanley_cup_winner or payload.get("champion_id") or ""
    )
    finalists = [str(x) for x in (payload.get("finalist_ids") or []) if x]
    elim_round: Dict[str, int] = {}
    conf_final_losers: set = set()

    # First pass: resolve every decided series independently of list order.
    series_results: List[Tuple[int, str, str]] = []  # (round_index, winner, loser)
    for s in payload.get("series_list") or []:
        ri = int(s.get("round_index") or 0)
        th = str(s.get("team_high_id") or "")
        tl = str(s.get("team_low_id") or "")
        wh = int(s.get("wins_high") or 0)
        wl = int(s.get("wins_low") or 0)
        if wh >= 4:
            loser, winner = tl, th
        elif wl >= 4:
            loser, winner = th, tl
        else:
            continue
        elim_round[loser] = max(elim_round.get(loser, 0), ri)
        series_results.append((ri, winner, loser))

    # The Cup Final is the single highest-round decided series; conference finals
    # are the round below it. Order of series_list is irrelevant.
    final_round = max((ri for ri, _, _ in series_results), default=0)
    cup_final_loser = ""
    if final_round >= 3:
        for ri, _winner, loser in series_results:
            if ri == final_round:
                cup_final_loser = loser
            elif ri == final_round - 1:
                conf_final_losers.add(loser)

    cup_finalist = cup_final_loser or (finalists[0] if finalists else "")
    if cup_finalist and champion and cup_finalist == champion and len(finalists) > 1:
        cup_finalist = finalists[1]
    if not cup_finalist and finalists:
        cup_finalist = finalists[0]

    def sort_key(tid: str) -> Tuple[int, int, int]:
        if champion and tid == champion:
            return (999, pts_map.get(tid, 0), 0)
        if cup_finalist and tid == cup_finalist:
            return (998, pts_map.get(tid, 0), 0)
        if tid in conf_final_losers:
            return (900, pts_map.get(tid, 0), 0 if div_winner.get(tid) else 1)
        er = elim_round.get(tid, 1)
        return (100 + er * 10, pts_map.get(tid, 0), 0 if div_winner.get(tid) else 1)

    ordered = sorted(playoff_teams, key=sort_key)
    if champion and champion in ordered:
        ordered.remove(champion)
        ordered.append(champion)
    if cup_finalist and cup_finalist in ordered and cup_finalist != champion:
        ordered.remove(cup_finalist)
        ordered.insert(max(0, len(ordered) - (1 if champion else 0)), cup_finalist)
    return ordered[:16], "lottery_playoff_results"


def _build_round1_slot_order(session: FranchiseSession) -> Tuple[List[str], str]:
    """Original slot owners for picks 1–32 (before traded ownership)."""
    from services.franchise_sim import _build_standings_rows

    if session.draft_lottery_payload and session.draft_lottery_payload.get("picks"):
        lot = sorted(
            list(session.draft_lottery_payload["picks"]),
            key=lambda p: int(p.get("pick", 99)),
        )
        lot_ids = [str(p["team_id"]) for p in lot[:16]]
    else:
        standings = sorted(
            _build_standings_rows(session),
            key=lambda r: (int(r.get("pts", 0)), -int(r.get("w", 0))),
        )
        lot_ids = [str(r["team_id"]) for r in standings[:16]]

    playoff_ids, po_source = _playoff_team_pick_order(session, lot_ids)
    order = lot_ids[:16] + playoff_ids[:16]
    all_ids = [str(r["team_id"]) for r in sorted(
        _build_standings_rows(session),
        key=lambda r: (int(r.get("pts", 0)), -int(r.get("w", 0))),
    )]
    while len(order) < DRAFT_TEAMS:
        for tid in all_ids:
            if tid not in order:
                order.append(tid)
            if len(order) >= DRAFT_TEAMS:
                break
    source = po_source
    return order[:DRAFT_TEAMS], source


def _build_later_round_slot_order(session: FranchiseSession) -> List[str]:
    """Rounds 2–7 use strict reverse regular-season standings (worst team picks
    first). The draft lottery only affects the first round; it must never carry
    into later rounds."""
    from services.franchise_sim import _build_standings_rows

    standings = sorted(
        _build_standings_rows(session),
        key=lambda r: (int(r.get("pts", 0)), -int(r.get("w", 0))),
    )
    order = [str(r["team_id"]) for r in standings]
    # Pad defensively if standings are short so every round has DRAFT_TEAMS slots.
    if len(order) < DRAFT_TEAMS:
        for tid in [str(t) for t in getattr(session, "team_ids", [])]:
            if tid not in order:
                order.append(tid)
            if len(order) >= DRAFT_TEAMS:
                break
    return order[:DRAFT_TEAMS]


def _apply_registry_to_slot(
    session: FranchiseSession,
    slot: Dict[str, Any],
    draft_year: int,
) -> Dict[str, Any]:
    """Merge trade_pick_registry current owner onto a draft slot."""
    from services.draft_pick_ownership import apply_registry_owner_to_slot

    return apply_registry_owner_to_slot(session, slot, draft_year)


def _mark_pick_resolved(session: FranchiseSession, slot: Dict[str, Any], prospect_id: str) -> None:
    pick_id = slot.get("pick_id")
    league = getattr(session.sim, "league", None)
    if not pick_id:
        raise ValueError("Draft slot missing pick_id during resolution")
    if league is None:
        raise ValueError("League unavailable while resolving draft pick")
    reg = getattr(league, "draft_pick_registry", None)
    if not isinstance(reg, dict) or pick_id not in reg:
        raise ValueError(f"Pick registry missing row for drafted pick: {pick_id}")
    row = reg[pick_id]
    if row.get("resolved") and str(row.get("selected_prospect_id") or "") == str(prospect_id):
        return
    if row.get("resolved") and row.get("selected_prospect_id"):
        raise ValueError(f"Pick already resolved: {pick_id}")
    row["resolved"] = True
    row["selected_prospect_id"] = str(prospect_id)
    row["resolved_at"] = _now_iso()
    row["selecting_team_id"] = str(slot.get("team_id") or row.get("current_owner_team_id") or "")


def _unmark_pick_resolved(session: FranchiseSession, slot: Dict[str, Any]) -> None:
    """Reverse _mark_pick_resolved (used to roll back a failed pick)."""
    pick_id = slot.get("pick_id")
    league = getattr(session.sim, "league", None)
    if not pick_id or league is None:
        return
    reg = getattr(league, "draft_pick_registry", None)
    if not isinstance(reg, dict) or pick_id not in reg:
        return
    row = reg[pick_id]
    row["resolved"] = False
    row["selected_prospect_id"] = None
    row["resolved_at"] = None
    row["selecting_team_id"] = None


def build_full_draft_order(session: FranchiseSession) -> List[Dict[str, Any]]:
    """7-round order: lottery + playoff slots, then traded ownership from pick registry."""
    draft_year = int(session.season_calendar_year) + 1
    round1_slots, order_source = _build_round1_slot_order(session)
    later_slots = _build_later_round_slot_order(session)

    lottery_ids: List[str] = []
    payload = getattr(session, "draft_lottery_payload", None) or {}
    for row in payload.get("picks") or payload.get("final_order") or []:
        tid = str(row.get("team_id") or "")
        if tid:
            lottery_ids.append(tid)
    if not lottery_ids:
        lottery_ids = list(round1_slots[:16])

    league = getattr(session.sim, "league", None)
    if league is not None:
        try:
            from services.draft_pick_conditions import finalize_draft_pick_registry

            finalize_draft_pick_registry(league, draft_year=draft_year, lottery_order=lottery_ids)
        except Exception:
            pass

    picks: List[Dict[str, Any]] = []
    overall = 0
    for rnd in range(1, DRAFT_ROUNDS + 1):
        round_slots = round1_slots if rnd == 1 else later_slots
        for pick_in_round, slot_owner in enumerate(round_slots[:DRAFT_TEAMS], start=1):
            overall += 1
            slot = {
                "round": rnd,
                "pick_in_round": pick_in_round,
                "overall_pick": overall,
                "team_id": slot_owner,
                "original_owner_team_id": slot_owner,
                "team_name": _display_team(session, slot_owner),
            }
            slot = _apply_registry_to_slot(session, slot, draft_year)
            picks.append(slot)

    if picks:
        picks[0]["playoff_order_source"] = order_source
        picks[0]["draft_order_source"] = order_source
    return picks

def append_stock_history_snapshot(
    session: FranchiseSession,
    entry: Dict[str, Any],
    *,
    event_source: str,
    date_label: str = "",
) -> None:
    hist = getattr(session, "draft_stock_history", None)
    if not isinstance(hist, dict):
        session.draft_stock_history = {}
        hist = session.draft_stock_history
    key = str(entry.get("key") or entry.get("prospect_id") or "")
    if not key:
        return
    prev_list = list(hist.get(key) or [])
    prev_rank = int(prev_list[-1]["rank"]) if prev_list else int(entry.get("rank") or 0)
    rank = int(entry.get("rank") or prev_rank)
    snap = {
        "date": date_label or _now_iso()[:10],
        "rank": rank,
        "previous_rank": prev_rank,
        "movement": prev_rank - rank,
        "reason": str(entry.get("stock_reason") or entry.get("stock_label") or ""),
        "gp": entry.get("gp"),
        "goals": entry.get("goals"),
        "assists": entry.get("assists"),
        "points": entry.get("points"),
        "ppg": entry.get("ppg"),
        "scouting_confidence": entry.get("scouting_confidence"),
        "stock_label": entry.get("stock_label"),
        "risk_score": entry.get("risk_score"),
        "production_score": entry.get("production_adjusted_score"),
        "event_source": event_source,
    }
    if prev_list and prev_list[-1].get("rank") == rank and prev_list[-1].get("event_source") == event_source:
        return
    prev_list.append(snap)
    hist[key] = prev_list[-12:]


def finalize_draft_class_for_event(session: FranchiseSession) -> Dict[str, Any]:
    from services.franchise_sim import build_draft_class_rankings, get_cached_draft_class_rankings

    board = get_cached_draft_class_rankings(session, session.sim)
    entries = list(board.get("entries") or [])
    if getattr(session, "draft_combine_done", False):
        scouting = getattr(session, "scouting_state", None) or {}
        combine_map = scouting.get("combine_results") if isinstance(scouting.get("combine_results"), dict) else {}
        if combine_map:
            try:
                from services.franchise_scouting import _apply_public_combine_adjustments

                entries = _apply_public_combine_adjustments(session, entries, combine_map)
                board = {**board, "entries": entries}
            except Exception:
                pass
    for e in entries:
        append_stock_history_snapshot(session, e, event_source="final_ranking", date_label="Draft")
    preseason = dict(getattr(session, "draft_preseason_rank", None) or {})
    if not preseason:
        session.draft_preseason_rank = {
            str(e.get("key")): int(e.get("preseason_rank") or e.get("rank") or 0)
            for e in entries
            if e.get("key")
        }
    return board


def _team_status(session: FranchiseSession, team_id: str) -> str:
    from services.franchise_sim import _build_standings_rows

    rows = sorted(
        _build_standings_rows(session),
        key=lambda r: -int(r.get("pts", 0)),
    )
    ids = [str(r["team_id"]) for r in rows]
    if team_id not in ids:
        return "bubble"
    idx = ids.index(team_id)
    if idx < 8:
        return "contender"
    if idx < 16:
        return "playoff"
    if idx < 24:
        return "bubble"
    return "rebuilder"


def get_team_draft_philosophy(session: FranchiseSession, team_id: str) -> Dict[str, Any]:
    team = session.team_by_id.get(str(team_id))
    status = _team_status(session, team_id)
    archetype = str(getattr(team, "archetype", "") or "").lower()
    risk = float(getattr(team, "risk_tolerance", 0.5) or 0.5)
    dev_q = float(getattr(team, "development_quality", 0.5) or 0.5)

    if status == "rebuilder":
        key = "rebuilder_upside" if risk > 0.55 else "high_upside"
    elif status == "contender":
        key = "contender_timeline" if dev_q < 0.55 else "safe_floor"
    elif "defense" in archetype:
        key = "defense_first"
    elif risk > 0.7:
        key = "boom_bust_gambler"
    elif risk < 0.35:
        key = "safe_floor"
    elif dev_q > 0.65:
        key = "analytics_driven"
    else:
        key = "bpa_heavy"

    return {
        "philosophy": key,
        "label": key.replace("_", " ").title(),
        "team_status": status,
        "risk_tolerance": round(risk, 2),
    }


def calculate_team_needs(session: FranchiseSession, team_id: str) -> List[Dict[str, Any]]:
    team = session.team_by_id.get(str(team_id))
    if not team:
        return []
    roster = list(getattr(team, "roster", None) or [])
    pool = list(getattr(team, "prospect_pool", None) or getattr(team, "prospects", None) or [])

    def count_pos(players, pos):
        return sum(1 for p in players if str(getattr(p, "position", "") or "").upper() == pos)

    def count_rhd(players):
        n = 0
        for p in players:
            pos = str(getattr(p, "position", "") or "").upper()
            shoots = str(getattr(getattr(p, "identity", None), "shoots", "") or getattr(p, "shoots", "") or "").upper()
            is_d = pos in ("D", "LD", "RD", "LHD", "RHD") or pos.endswith("D")
            is_right = shoots.startswith("R") or pos in ("RD", "RHD")
            if is_d and is_right:
                n += 1
        return n

    needs: List[Tuple[str, float]] = []
    c_nhl, c_pool = count_pos(roster, "C"), count_pos(pool, "C")
    g_nhl, g_pool = count_pos(roster, "G"), count_pos(pool, "G")
    rhd_nhl, rhd_pool = count_rhd(roster), count_rhd(pool)
    wing_nhl = count_pos(roster, "LW") + count_pos(roster, "RW")

    if c_nhl + c_pool < 4:
        needs.append((
            "Center Depth",
            0.9,
            f"Only {c_nhl} NHL centers and {c_pool} center prospects.",
        ))
    if wing_nhl < 6:
        needs.append((
            "Wing Depth",
            0.75,
            f"Only {wing_nhl} NHL wingers currently rostered.",
        ))
    if rhd_nhl + rhd_pool < 3:
        needs.append((
            "Right-Shot Defense",
            0.85,
            f"Only {rhd_nhl + rhd_pool} right-shot defensemen across roster and pool.",
        ))
    if g_pool < 2:
        needs.append((
            "Goalie Pipeline",
            0.7,
            f"Only {g_pool} goalie prospect(s) in the organizational pool.",
        ))
    u23 = sum(
        1
        for p in roster
        if int(getattr(getattr(p, "identity", None), "age", getattr(p, "age", 30)) or 30) < 24
    )
    if u23 < 8:
        needs.append((
            "Young NHL Depth",
            0.65,
            f"Only {u23} under-23 players on the NHL roster.",
        ))
    status = _team_status(session, team_id)
    if status == "rebuilder":
        needs.append(("High-Upside Swing", 0.8, "Rebuild window favors longer-timeline upside."))
    elif status == "contender":
        needs.append(("Near-Ready Help", 0.75, "Competitive window favors nearer NHL readiness."))

    needs.sort(key=lambda x: -x[1])
    return [
        {"category": cat, "priority": round(pri, 2), "detail": detail}
        for cat, pri, detail in needs[:6]
    ]


def _scouting_overlay(session: FranchiseSession, prospect_id: str, team_id: Optional[str] = None) -> Dict[str, Any]:
    scouting = getattr(session, "scouting_state", None) or {}
    overlay = dict((scouting.get("prospects") or {}).get(str(prospect_id)) or {})
    tid = str(team_id or session.user_team_id or "")
    if tid == str(session.user_team_id):
        return overlay
    try:
        from services.franchise_scouting import get_team_prospect_impression

        imp = get_team_prospect_impression(session, tid, str(prospect_id))
        if imp:
            merged = dict(imp)
            if imp.get("scout_favorite"):
                merged["target"] = True
            if imp.get("do_not_draft"):
                merged["do_not_draft"] = True
            if imp.get("scout_note"):
                merged["notes"] = [str(imp.get("scout_note"))]
            if imp.get("private_meeting_summary"):
                merged["dinner_status"] = "Completed" if "dinner" in str(imp.get("private_meeting_impression") or "").lower() else merged.get("dinner_status")
            return merged
    except Exception:
        pass
    return overlay


def _team_scouting_profile(session: FranchiseSession, team_id: str) -> Dict[str, Any]:
    try:
        from services.franchise_scouting import get_team_scouting_profile

        return get_team_scouting_profile(session, str(team_id))
    except Exception:
        return {}


def _scouting_event_adjustments(
    session: FranchiseSession,
    team_id: str,
    entry: Dict[str, Any],
    overlay: Dict[str, Any],
    phil: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    """Combine/interview/dinner impact on internal team board (not public board)."""
    delta = 0.0
    meta: Dict[str, Any] = {}
    notes: List[str] = []
    risk_tol = float(phil.get("risk_tolerance") or 0.5)
    profile = _team_scouting_profile(session, team_id)

    scouting = getattr(session, "scouting_state", None) or {}
    combine_map = scouting.get("combine_results") if isinstance(scouting.get("combine_results"), dict) else {}
    pid = str(entry.get("key") or "")
    comb = combine_map.get(pid) or {}
    if comb.get("combine_attended"):
        cs = float(comb.get("combine_score") or 60)
        delta += (cs - 60) * 0.08 * float(profile.get("combine_trust") or 0.5)
        meta["combine_score"] = "Strong" if cs >= 72 else "Average" if cs >= 58 else "Weak"
        if cs >= 72:
            notes.append("Combine: Team scouts loved the testing")
        elif cs < 55:
            notes.append("Combine: Athletic testing raised concerns")
        if comb.get("medical_flag"):
            med_pen = 5.0 if comb.get("medical_risk_level") == "High" else 2.0
            if float(profile.get("red_flag_detection") or 0.5) > 0.6:
                med_pen *= 1.4
            delta -= med_pen if risk_tol < 0.55 else med_pen * 0.4
            meta["medical_flag"] = True
            notes.append("Medical concern caused other teams to pass.")

    if overlay.get("board_delta") is not None:
        delta += float(overlay.get("board_delta") or 0)
    if overlay.get("interview_impression"):
        imp = str(overlay.get("interview_impression") or "")
        if imp in ("Elite", "Strong"):
            delta += 2.5
            notes.append("Selected after strong internal interview.")
        elif imp in ("Poor", "Below Average"):
            delta -= 3.0
            notes.append("Interview: Compete level questioned")
    if overlay.get("combine_impression") in ("Elite", "Strong"):
        notes.append("Team scouts loved the combine testing.")

    interview = str(overlay.get("interview_status") or entry.get("interview_status") or "")
    if interview.lower().startswith("complet"):
        traits = overlay.get("traits") or []
        red = overlay.get("red_flags") or []
        if any("character" in str(t).lower() or "leadership" in str(t).lower() for t in traits):
            delta += 2.0
            meta["interview_score"] = "Positive"
            notes.append("Interview: Leadership impressed scouts")
        elif red:
            delta -= 4.0 if risk_tol < 0.5 else 1.0
            meta["interview_score"] = "Concern"
            notes.append("Interview: Character concerns noted")
        else:
            delta += 1.0
            meta["interview_score"] = "Neutral"

    dinner = str(overlay.get("dinner_status") or entry.get("dinner_status") or "")
    if dinner.lower().startswith("complet") or overlay.get("private_meeting_impression") not in (None, "", "Not held"):
        delta += 3.5 if overlay.get("target") or overlay.get("scout_favorite") else 1.5
        meta["dinner_score"] = "Strong fit"
        notes.append("Private meeting made him a scout favorite.")
        if overlay.get("private_meeting_summary"):
            meta["draft_meeting_summary"] = str(overlay.get("private_meeting_summary"))

    if overlay.get("do_not_draft"):
        delta -= 60.0
        notes.append("Do-not-draft flag")
    if overlay.get("scout_favorite"):
        delta += 3.0
        notes.append("Scout favorite despite public-board ranking.")

    for flag in overlay.get("red_flags") or []:
        fs = str(flag).lower()
        if "medical" in fs:
            delta -= 3.0 if risk_tol < 0.55 else 1.0
            meta["medical_flag"] = True
        if "character" in fs and risk_tol < 0.45:
            delta -= 5.0
            notes.append("Character risk on file")

    scouted = float(overlay.get("scouted_percentage") or entry.get("scouting_confidence") or 40)
    meta["scout_confidence_delta"] = round((scouted - 50) * 0.06, 2)
    meta["internal_board_delta"] = round(delta, 2)
    meta["team_fit_score"] = round(max(0, min(100, 50 + delta * 3)), 1)
    if overlay.get("scout_note") and not meta.get("draft_meeting_summary"):
        meta["draft_meeting_summary"] = str(overlay.get("scout_note"))
    elif overlay.get("notes"):
        meta["draft_meeting_summary"] = str((overlay.get("notes") or [])[-1])

    return delta, meta, notes


def _entry_is_defense(entry: Dict[str, Any]) -> bool:
    pos = str(entry.get("position") or "").upper()
    return pos in ("D", "LD", "RD", "LHD", "RHD") or pos.endswith("D")


def _entry_is_right_shot_d(entry: Dict[str, Any]) -> bool:
    """A Right-Shot Defense need must be filled by a right-handed defenseman."""
    if not _entry_is_defense(entry):
        return False
    pos = str(entry.get("position") or "").upper()
    shoots = str(entry.get("handedness") or entry.get("shoots") or "").upper()
    return shoots.startswith("R") or pos in ("RD", "RHD")


def _prospect_base_score(entry: Dict[str, Any]) -> float:
    """Public-anchored base score (no hidden true_ovr).

    Ordering starts from observable signals — public consensus rank and junior
    production. Team-specific scouted OVR/potential estimates are layered on top
    inside build_team_draft_board so each club works from its own fogged view.
    """
    pub_rank = int(entry.get("rank") or entry.get("public_rank") or 120)
    rank_score = max(0.0, 120.0 - pub_rank) * 0.9
    return (
        rank_score
        + float(entry.get("potential_score") or 0) * 0.20
        + float(entry.get("production_adjusted_score") or 0) * 3.0
    )


def _board_revision(session: FranchiseSession, board: Dict[str, Any]) -> str:
    """Stable digest of the inputs that should invalidate the draft cache.

    Length alone is not enough — public ranks, scouting confidence, potential
    estimates, combine results and the draft year can all change while the
    prospect count stays constant.
    """
    entries = board.get("entries") or []
    parts: List[str] = [
        str(len(entries)),
        str(board.get("class_strength") or ""),
        str(int(getattr(session, "season_calendar_year", 0) or 0)),
    ]
    for e in entries:
        parts.append(
            f"{e.get('key')}:{e.get('rank')}:{e.get('scouting_confidence')}:"
            f"{e.get('potential_score')}:{e.get('stock_delta')}"
        )
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _ensure_draft_cache(session: FranchiseSession, board: Dict[str, Any]) -> Dict[str, Any]:
    state = getattr(session, "draft_state", None) or {}
    entries = board.get("entries") or []
    version = _board_revision(session, board)
    cache = state.get("_cache") or {}
    if cache.get("version") == version and cache.get("entry_by_key"):
        return cache

    entry_by_key = {str(e["key"]): e for e in entries if e.get("key")}
    philosophies = {str(tid): get_team_draft_philosophy(session, str(tid)) for tid in session.team_ids}
    needs = dict(state.get("team_needs_snapshot") or {})
    for tid in session.team_ids:
        if str(tid) not in needs:
            needs[str(tid)] = calculate_team_needs(session, str(tid))

    cache = {
        "version": version,
        "entry_by_key": entry_by_key,
        "base_scores": {pid: _prospect_base_score(e) for pid, e in entry_by_key.items()},
        "team_philosophies": philosophies,
        "team_needs": needs,
    }
    state["_cache"] = cache
    session.draft_state = state
    return cache


def build_team_draft_board(
    session: FranchiseSession,
    team_id: str,
    available: List[Dict[str, Any]],
    *,
    cache: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    phil = (cache or {}).get("team_philosophies", {}).get(str(team_id)) or get_team_draft_philosophy(session, team_id)
    needs = (cache or {}).get("team_needs", {}).get(str(team_id)) or calculate_team_needs(session, team_id)
    need_cats = {n["category"] for n in needs[:3]}
    base_scores = (cache or {}).get("base_scores") or {}
    profile = _team_scouting_profile(session, team_id)
    draft_year = int(getattr(session, "season_calendar_year", 2025) or 2025) + 1
    board: List[Dict[str, Any]] = []

    from services.draft_board_engine import enrich_board_entry_with_team_scouting

    scout_dept_quality = float(profile.get("scouting_quality") or 60)
    for p in available:
        pid = str(p.get("key") or p.get("prospect_id") or "")
        overlay = _scouting_overlay(session, pid, team_id=team_id)
        pub_rank = int(p.get("rank") or 999)

        # Team-specific scouted estimate FIRST — the board is built on the scouts'
        # fogged view (scouted OVR/potential), never on hidden true_ovr.
        interview = {
            "quality": overlay.get("interview_impression") or overlay.get("interview_status"),
            "willingness_to_sign": overlay.get("willingness_to_sign"),
            "ncaa_commitment": overlay.get("ncaa_commitment"),
            "european_contract": overlay.get("european_contract"),
            "medical_flag": overlay.get("medical_flag") or overlay.get("medical_concern"),
            "market_preference": overlay.get("market_preference"),
        }
        enriched = enrich_board_entry_with_team_scouting(
            p,
            team_id=str(team_id),
            scouting_quality=scout_dept_quality,
            draft_year=draft_year,
            interview=interview,
        )
        scouted_ovr = float(enriched.get("scouted_overall_estimate") or 0)
        pot_range = enriched.get("scouted_potential_range") or []
        pot_mid = (
            (float(pot_range[0]) + float(pot_range[1])) / 2.0
            if isinstance(pot_range, (list, tuple)) and len(pot_range) >= 2
            else 0.0
        )
        # Public-board anchor (observable) + scouted ability estimate (fogged).
        score = float(base_scores.get(pid) or _prospect_base_score(p))
        score += scouted_ovr * 0.35 + pot_mid * 0.30

        scouted = float(overlay.get("scouted_percentage") or enriched.get("scouting_confidence") or 40)
        score += (scouted - 50) * 0.08

        pub_trust = float(profile.get("public_board_trust") or 0.5)
        # High public-board trust pulls the internal board toward public order:
        # a better public rank (low number) is boosted, a worse rank faded.
        score += (40 - pub_rank) * (pub_trust - 0.5) * 0.04

        league = str(p.get("league_code") or p.get("league") or "").upper()
        if "NCAA" in league:
            score += (float(profile.get("NCAA_scouting_quality") or 60) - 65) * 0.06
        elif league.startswith("EU_"):
            score += (float(profile.get("European_scouting_quality") or 60) - 65) * 0.06
        else:
            score += (float(profile.get("CHL_scouting_quality") or 60) - 65) * 0.05

        scout_delta, scout_meta, scout_notes = _scouting_event_adjustments(session, team_id, p, overlay, phil)
        score += scout_delta

        pos = str(p.get("position") or "").upper()
        if ("Franchise Center" in need_cats or "Center Depth" in need_cats) and pos == "C":
            score += 3.5
        if "Right-Shot Defense" in need_cats and _entry_is_right_shot_d(p):
            score += 2.5
        if "Goalie Pipeline" in need_cats and pos == "G":
            score += 2.0 + (float(profile.get("goalie_scouting_quality") or 60) - 65) * 0.05
        if "Top-Six Winger" in need_cats and pos in ("LW", "RW"):
            score += 2.0

        if phil["philosophy"] == "safe_floor" and str(p.get("risk") or "") == "Low":
            score += 2.0
        if phil["philosophy"] in ("high_upside", "boom_bust_gambler", "rebuilder_upside"):
            score += float(p.get("potential_score") or 0) * 0.05
        if phil["philosophy"] == "contender_timeline" and float(p.get("scouting_confidence") or 0) > 75:
            score += 1.5
        if overlay.get("target") or overlay.get("scout_favorite"):
            score += 4.0

        noise_scale = 8.0 - scout_dept_quality * 0.05
        noise = (_rng_float(session, team_id, pid) - 0.5) * max(3.0, noise_scale)
        if phil["philosophy"] in ("off_board_scout", "boom_bust_gambler") or float(profile.get("off_board_tendency") or 0) > 0.35:
            noise += (_rng_float(session, team_id, pid, "off") - 0.3) * 10.0

        board.append({
            **enriched,
            "team_board_score": score + noise + (float(enriched.get("scouting_confidence") or 50) - 50) * 0.03,
            "team_board_rank": 0,
            "public_rank": pub_rank,
            **scout_meta,
            "scouting_notes": scout_notes,
        })

    board.sort(key=lambda x: -float(x.get("team_board_score") or 0))
    for i, row in enumerate(board):
        row["team_board_rank"] = i + 1
    return board

def _prospect_storyline(entry: Dict[str, Any]) -> str:
    labels = []
    if int(entry.get("rank") or 99) == 1:
        labels.append("Consensus #1")
    if int(entry.get("stock_delta") or 0) >= 6:
        labels.append("Late-season riser")
    if entry.get("is_gem"):
        labels.append("Scout favorite")
    if entry.get("character_concerns"):
        labels.append("Character concern but elite tools")
    if str(entry.get("position") or "").upper() == "G" and entry.get("generational_goalie"):
        labels.append("Goalie with rare upside")
    if str(entry.get("risk") or "") == "High":
        labels.append("Boom/bust profile")
    if int(entry.get("age") or 18) >= 20:
        labels.append("Overager")
    if not labels:
        labels.append(str(entry.get("stock_label") or "On the board"))
    return labels[0]


def _public_board_thresholds(overall_pick: int) -> Dict[str, int]:
    """Round-sensitive public-consensus bands. Later rounds = wider uncertainty."""
    if overall_pick <= 32:
        return {"expected": 4, "early_value": 9, "reach_steal": 10}
    if overall_pick <= 96:
        return {"expected": 8, "early_value": 17, "reach_steal": 18}
    return {"expected": 14, "early_value": 28, "reach_steal": 29}


def _selection_label_from_public(
    public_rank: Optional[int],
    overall_pick: int,
    *,
    consensus_low: Optional[int] = None,
    consensus_high: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Grade a pick against the PUBLIC consensus board at selection time.
    Never uses the user/team private board for Reach/Steal/Value labels.
    """
    pub = int(public_rank) if public_rank is not None else None
    if pub is None or pub <= 0 or pub >= 500:
        return {
            "selection_label": "Off Board",
            "was_reach": False,
            "was_steal": False,
            "was_value": False,
            "was_early": False,
            "was_expected": False,
            "was_off_board": True,
            "public_rank_delta": None,
            "public_rank_at_pick": None,
        }

    # Positive = fell past public rank (value/steal). Negative = taken early (early/reach).
    # Spec: publicRankDifference = pickNumber - publicRank
    delta = int(overall_pick) - pub
    bands = _public_board_thresholds(overall_pick)
    exp = bands["expected"]
    mid = bands["early_value"]

    # Consensus range softens grades when the board already expects a wide band.
    if consensus_low is not None and consensus_high is not None:
        try:
            lo, hi = int(consensus_low), int(consensus_high)
            if lo <= overall_pick <= hi:
                return {
                    "selection_label": "Expected",
                    "was_reach": False,
                    "was_steal": False,
                    "was_value": False,
                    "was_early": False,
                    "was_expected": True,
                    "was_off_board": False,
                    "public_rank_delta": delta,
                    "public_rank_at_pick": pub,
                }
        except (TypeError, ValueError):
            pass

    if abs(delta) <= exp:
        label = "Expected"
        flags = dict(was_reach=False, was_steal=False, was_value=False, was_early=False, was_expected=True)
    elif delta < 0 and abs(delta) <= mid:
        label = "Early"
        flags = dict(was_reach=False, was_steal=False, was_value=False, was_early=True, was_expected=False)
    elif delta < 0:
        label = "Reach"
        flags = dict(was_reach=True, was_steal=False, was_value=False, was_early=False, was_expected=False)
    elif delta <= mid:
        label = "Value"
        flags = dict(was_reach=False, was_steal=False, was_value=True, was_early=False, was_expected=False)
    else:
        label = "Steal"
        flags = dict(was_reach=False, was_steal=True, was_value=True, was_early=False, was_expected=False)

    return {
        "selection_label": label,
        **flags,
        "was_off_board": False,
        "public_rank_delta": delta,
        "public_rank_at_pick": pub,
    }


def _classify_pick(
    entry: Dict[str, Any],
    public_rank: int,
    team_board_rank: int,
    overall_pick: int,
    philosophy: str,
    needs: List[Dict[str, Any]],
    *,
    best_available_rank: Optional[int] = None,
) -> Dict[str, Any]:
    need_pos = str(entry.get("position") or "").upper()
    need_match = any(
        (n.get("category") in ("Franchise Center", "Center Depth") and need_pos == "C")
        or (n.get("category") == "Right-Shot Defense" and _entry_is_right_shot_d(entry))
        or (n.get("category") == "Goalie Pipeline" and need_pos == "G")
        or (n.get("category") in ("Top-Six Winger", "Wing Depth") and need_pos in ("LW", "RW", "W"))
        for n in (needs or [])[:2]
        if isinstance(n, dict)
    )
    consensus = entry.get("consensus_range") or entry.get("rank_range") or entry.get("projected_pick_range")
    c_lo = c_hi = None
    if isinstance(consensus, (list, tuple)) and len(consensus) >= 2:
        c_lo, c_hi = consensus[0], consensus[1]
    elif isinstance(consensus, dict):
        c_lo, c_hi = consensus.get("low"), consensus.get("high")

    grade = _selection_label_from_public(
        public_rank,
        overall_pick,
        consensus_low=c_lo,
        consensus_high=c_hi,
    )
    # True BPA: the selected prospect is (near) the highest-ranked prospect STILL
    # AVAILABLE, not merely close to the pick slot. best_available_rank is the
    # minimum public rank across the live board at selection time.
    if best_available_rank is not None:
        was_bpa = int(public_rank or 999) <= int(best_available_rank) + 2
    else:
        was_bpa = (
            int(public_rank or 999) <= max(3, int(overall_pick))
            and int(public_rank or 999) <= int(overall_pick) + 2
        )
    was_need = need_match and not grade.get("was_reach")

    tags: List[str] = [str(grade["selection_label"])]
    if was_bpa and grade["selection_label"] in ("Expected", "Value", "Steal"):
        tags.append("BPA")
    if was_need:
        tags.append("Need Fit")
    if str(entry.get("risk") or "") == "Low" and grade["selection_label"] in ("Expected", "Value"):
        tags.append("Safe Floor")
    if str(entry.get("risk") or "") == "High":
        tags.append("High Upside")
    if need_pos == "G" and overall_pick <= 32:
        tags.append("Early Goalie")
    elif need_pos == "G":
        tags.append("Goalie")

    strategy = str(philosophy or "").replace("_", " ")
    return {
        "was_bpa": was_bpa,
        "was_team_need": was_need,
        "was_reach": bool(grade.get("was_reach")),
        "was_steal": bool(grade.get("was_steal")),
        "was_value": bool(grade.get("was_value")),
        "was_early": bool(grade.get("was_early")),
        "was_expected": bool(grade.get("was_expected")),
        "was_off_board": bool(grade.get("was_off_board")),
        "was_goalie_exception": need_pos == "G" and overall_pick <= 15,
        "selection_label": grade["selection_label"],
        "public_rank_at_pick": grade.get("public_rank_at_pick"),
        "public_rank_delta": grade.get("public_rank_delta"),
        "user_rank_at_pick": int(team_board_rank) if team_board_rank else None,
        "draft_strategy_used": strategy,
        "pick_classification": grade["selection_label"],
        "pick_class": str(grade["selection_label"]).lower().replace(" ", "_"),
        "pick_tags": tags,
        "public_board_delta": grade.get("public_rank_delta"),
        "team_board_delta": (int(team_board_rank) - int(overall_pick)) if team_board_rank else None,
    }


def _pick_reason(
    entry: Dict[str, Any],
    cls: Dict[str, Any],
    phil: Dict[str, Any],
    needs: List[Dict[str, Any]],
    public_rank: int,
    team_board_rank: int,
    scout_notes: Optional[List[str]] = None,
) -> str:
    notes = scout_notes or entry.get("scouting_notes") or []
    if notes:
        for n in notes:
            nl = str(n)
            if "Interview: Leadership" in nl or "strong internal interview" in nl:
                return "Moved up internal board after strong interview."
            if "Combine" in nl and ("loved" in nl or "Strong" in nl.title()):
                return "Combine testing confirmed his tools."
            if "Medical" in nl:
                return "Medical concern lowered team confidence."
            if "Private meeting" in nl or "scout favorite" in nl or "GM favorite" in nl:
                return "Private meeting made him a scout favorite."
            if "Character" in nl and phil.get("risk_tolerance", 0.5) < 0.45:
                return "Character concerns pushed him down conservative boards."
    label = str(cls.get("selection_label") or "")
    if cls.get("was_bpa"):
        return "Best player available on the public board."
    if label == "Steal":
        return f"Public board had him at #{public_rank}; selected later than consensus."
    if label == "Value":
        return "Selected later than public consensus."
    if label == "Reach":
        return f"Selected well ahead of public rank #{public_rank}."
    if label == "Early":
        return f"Taken slightly ahead of public rank #{public_rank}."
    if label == "Off Board":
        return "Not ranked on the public board at this slot."
    if cls.get("was_team_need") and needs:
        cat = needs[0].get("category") if isinstance(needs[0], dict) else None
        if cat:
            return f"Addresses organizational need: {cat}."
        return f"Filled a major organizational need at {entry.get('position', 'position')}."
    if phil.get("philosophy") == "rebuilder_upside":
        return "Higher-variance upside pick for a rebuilding club."
    if phil.get("philosophy") == "contender_timeline":
        return "Nearer-term ability fits a competitive window."
    if str(entry.get("position") or "").upper() == "G":
        return "Goaltender selected based on board and pipeline."
    if team_board_rank and public_rank and team_board_rank < public_rank - 5:
        return "Internal board ranked this prospect higher than public consensus."
    return "Best available fit for team philosophy and board."


def _cpu_select_prospect(
    session: FranchiseSession,
    owner: str,
    overall: int,
    available: List[Dict[str, Any]],
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    from services.draft_selection_engine import cpu_select_from_board

    phil = cache["team_philosophies"].get(owner, get_team_draft_philosophy(session, owner))
    needs = cache["team_needs"].get(owner, [])
    base_pool = 80 if overall <= 10 else 50 if overall <= 64 else 28
    # Off-board scouting must be able to reach beyond the public slice, otherwise
    # the philosophy can never actually go off the board.
    phil_name = phil.get("philosophy") if isinstance(phil, dict) else str(phil)
    profile = _team_scouting_profile(session, owner)
    off_board = phil_name in ("off_board_scout", "boom_bust_gambler") or float(profile.get("off_board_tendency") or 0) > 0.35
    pool_size = max(base_pool, 150) if off_board else base_pool
    candidates = available[:pool_size]
    if not candidates:
        raise ValueError("No draft-eligible prospects available for CPU selection")
    team_board = build_team_draft_board(session, owner, candidates, cache=cache)

    def _noise(e: Dict[str, Any]) -> float:
        return _rng_float(session, owner, overall, e.get("key")) - 0.5

    # Ideology nudges philosophy weights without replacing board logic.
    profiles = dict(getattr(session, "cpu_franchise_profiles", None) or {})
    ideo = dict((profiles.get(str(owner)) or {}).get("ideology") or {})
    phil_override = phil
    if isinstance(phil, dict) and ideo:
        phil_override = dict(phil)
        bpa = float(ideo.get("best_player_available_bias", 0.55) or 0.55)
        need_bias = float(ideo.get("positional_need_draft_bias", 0.45) or 0.45)
        if bpa >= 0.62:
            phil_override["philosophy"] = phil_override.get("philosophy") or "best_player_available"
        elif need_bias >= 0.62 and needs:
            phil_override["philosophy"] = "positional_need"

    return cpu_select_from_board(
        team_board,
        overall_pick=overall,
        philosophy=phil_override.get("philosophy") if isinstance(phil_override, dict) else str(phil_override),
        needs=needs,
        noise_fn=_noise,
    )


def _cpu_draft_score(
    session: FranchiseSession,
    team_id: str,
    entry: Dict[str, Any],
    overall_pick: int,
    tb_entry: Optional[Dict[str, Any]],
    needs: List[Dict[str, Any]],
    phil: Dict[str, Any],
) -> float:
    from services.draft_board_engine import score_candidate_for_team

    row = dict(tb_entry or entry)
    return score_candidate_for_team(
        row,
        overall_pick=overall_pick,
        philosophy=phil.get("philosophy") if isinstance(phil, dict) else str(phil),
        needs=needs,
        team_board_score=float(row.get("team_board_score") or 0),
        rng_noise=_rng_float(session, team_id, overall_pick, entry.get("key")) - 0.5,
    )


def _find_prospect_player(session: FranchiseSession, prospect_id: str) -> Tuple[Any, Optional[Dict], Optional[Dict]]:
    league = getattr(session.sim, "league", None)
    if league is None:
        return None, None, None
    from services.draft_player_registry import find_development_home, register_player, rebuild_players_by_id

    rebuild_players_by_id(league, only_if_missing=True)
    player, block, tm = find_development_home(league, str(prospect_id))
    if player is not None:
        register_player(league, player)
    return player, block, tm


def _build_dev_home_index(session: FranchiseSession) -> Dict[str, Tuple[Any, Optional[Dict], Optional[Dict]]]:
    """One-pass id -> (player, block, team) map for live draft payload lookups."""
    league = getattr(session.sim, "league", None)
    index: Dict[str, Tuple[Any, Optional[Dict], Optional[Dict]]] = {}
    if league is None:
        return index
    try:
        from services.draft_player_registry import rebuild_players_by_id

        rebuild_players_by_id(league, only_if_missing=True)
    except Exception:
        pass
    for block in getattr(league, "development_leagues", None) or []:
        if not isinstance(block, dict):
            continue
        for tm in block.get("teams") or []:
            if not isinstance(tm, dict):
                continue
            for p in tm.get("players") or []:
                pid = str(getattr(p, "id", "") or "")
                if pid and pid not in index:
                    index[pid] = (p, block, tm)
    return index


def _lookup_prospect_player(
    session: FranchiseSession,
    prospect_id: str,
    home_index: Optional[Dict[str, Tuple[Any, Optional[Dict], Optional[Dict]]]] = None,
) -> Tuple[Any, Optional[Dict], Optional[Dict]]:
    pid = str(prospect_id or "")
    if not pid:
        return None, None, None
    if home_index is not None and pid in home_index:
        return home_index[pid]
    return _find_prospect_player(session, pid)


def _development_path_for(entry: Dict[str, Any], block: Optional[Dict]) -> str:
    from services.draft_rights_engine import development_path_for

    return development_path_for(entry, block)


def _nhl_eta_years(entry: Dict[str, Any], overall: int) -> int:
    """Delegate to the canonical ETA helper so the draft result, rights screen and
    development screen never disagree on a prospect's timeline."""
    from services.draft_ranking_logic import calculate_prospect_eta

    try:
        eta = calculate_prospect_eta(entry, final_rank=int(entry.get("rank") or overall))
        return int(eta.get("years") if eta.get("years") is not None else 4)
    except Exception:
        return 4


def _assign_drafted_prospect(
    session: FranchiseSession,
    player: Any,
    team_id: str,
    pick_meta: Dict[str, Any],
    block: Optional[Dict],
    tm: Optional[Dict],
    entry: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Transfer NHL rights only. Leave the player on his junior / NCAA / European club.
    """
    team = session.team_by_id.get(str(team_id))
    if team is None:
        raise ValueError(f"Unknown team {team_id}")

    from services.draft_rights_engine import apply_draft_rights, rights_card_payload
    from services.draft_player_registry import register_player

    draft_year = int(pick_meta.get("draft_year") or session.season_calendar_year + 1)
    overall = int(pick_meta.get("overall_pick") or 1)
    ent = entry or {}

    # Intentionally do NOT remove player from development league roster.
    rights = apply_draft_rights(
        player,
        nhl_team_id=str(team_id),
        draft_year=draft_year,
        pick_meta=pick_meta,
        block=block,
        tm=tm,
        entry=ent,
    )
    eta = _nhl_eta_years(ent, overall)
    setattr(player, "nhl_eta", eta)

    hist = getattr(session, "draft_stock_history", None) or {}
    pid = str(getattr(player, "id", "") or ent.get("key") or "")
    profile = {
        "draft_profile_summary": (
            f"Rd {pick_meta.get('round')} pick {overall} — "
            f"{rights.get('development_path')} path, ETA {eta}yr "
            f"(rights: {rights.get('rights_status')})"
        ),
        "pre_draft_stats": {
            "gp": ent.get("gp"),
            "points": ent.get("points"),
            "ppg": ent.get("ppg"),
        },
        "stock_history": list(hist.get(pid) or []),
        "rights_card": rights_card_payload(player),
    }
    setattr(player, "draft_profile_summary", profile["draft_profile_summary"])
    setattr(player, "pre_draft_stats", profile["pre_draft_stats"])
    setattr(player, "stock_history", profile["stock_history"])

    league = getattr(session.sim, "league", None)
    if league is not None:
        register_player(league, player)

    pool = getattr(team, "prospect_pool", None)
    if pool is None:
        team.prospect_pool = []
        pool = team.prospect_pool
    # Prospect pool holds organizational membership (by reference), not physical roster move.
    if all(getattr(p, "id", None) != getattr(player, "id", object()) for p in pool):
        pool.append(player)

    affiliations = getattr(team, "prospect_affiliations", None)
    if not isinstance(affiliations, list):
        try:
            team.prospect_affiliations = []
            affiliations = team.prospect_affiliations
        except Exception:
            affiliations = []
    if isinstance(affiliations, list):
        if not any(str(a.get("player_id")) == pid for a in affiliations if isinstance(a, dict)):
            affiliations.append({
                "player_id": pid,
                "rights_team_id": str(team_id),
                "current_team_id": rights.get("current_team_id"),
                "current_league_id": rights.get("current_league_id"),
                "signed_status": "unsigned",
                "rights_status": rights.get("rights_status"),
                "rights_expiry_year": rights.get("rights_expiry_year"),
            })

    try:
        from services.contract_economy import add_to_reserve_list

        add_to_reserve_list(
            team,
            player,
            draft_year=draft_year,
            draft_overall=overall,
            added_season=int(getattr(session, "season_calendar_year", draft_year) or draft_year),
        )
    except Exception:
        pass

    legacy = getattr(team, "prospects", None)
    if isinstance(legacy, list) and all(getattr(p, "id", None) != getattr(player, "id", object()) for p in legacy):
        legacy.append(player)

    archive = getattr(session, "draft_results_archive", None)
    if not isinstance(archive, list):
        session.draft_results_archive = []
        archive = session.draft_results_archive
    archive.append({
        "draft_year": draft_year,
        "prospect_id": pid,
        "team_id": str(team_id),
        "original_owner_team_id": pick_meta.get("original_owner_team_id"),
        "overall_pick": overall,
        "round": pick_meta.get("round"),
        "pick_in_round": pick_meta.get("pick_in_round"),
        "rights_status": rights.get("rights_status"),
        "rights_type": rights.get("rights_type"),
        "rights_expiry_year": rights.get("rights_expiry_year"),
        "current_league_id": rights.get("current_league_id"),
        "current_team_id": rights.get("current_team_id"),
        "signed_status": "unsigned",
        "signing_date": None,
        "nhl_debut": None,
        "selection_context": {
            "development_path": rights.get("development_path"),
            "public_rank": ent.get("rank"),
        },
    })


def _rollback_assigned_prospect(
    session: FranchiseSession,
    player: Any,
    team_id: str,
    pick_meta: Dict[str, Any],
) -> None:
    """Undo every side effect of _assign_drafted_prospect so a failed pick leaves
    no duplicate pool / affiliation / reserve / archive / rights records."""
    team = session.team_by_id.get(str(team_id))
    pid = str(getattr(player, "id", "") or "")
    overall = int(pick_meta.get("overall_pick") or 0)

    def _same(p: Any) -> bool:
        return getattr(p, "id", object()) == getattr(player, "id", object())

    if team is not None:
        pool = getattr(team, "prospect_pool", None)
        if isinstance(pool, list):
            team.prospect_pool = [p for p in pool if not _same(p)]
        legacy = getattr(team, "prospects", None)
        if isinstance(legacy, list):
            team.prospects = [p for p in legacy if not _same(p)]
        affs = getattr(team, "prospect_affiliations", None)
        if isinstance(affs, list):
            team.prospect_affiliations = [
                a for a in affs if not (isinstance(a, dict) and str(a.get("player_id")) == pid)
            ]
        reserve = getattr(team, "reserve_list", None)
        if isinstance(reserve, list):
            team.reserve_list = [
                r for r in reserve
                if not (isinstance(r, dict) and str(r.get("player_id") or r.get("id")) == pid)
            ]

    archive = getattr(session, "draft_results_archive", None)
    if isinstance(archive, list):
        session.draft_results_archive = [
            a for a in archive
            if not (
                isinstance(a, dict)
                and str(a.get("prospect_id")) == pid
                and int(a.get("overall_pick") or 0) == overall
            )
        ]

    for attr in ("drafted", "nhl_rights_team_id", "rights_team_id", "rights_status", "organizational_status", "nhl_eta"):
        try:
            setattr(player, attr, False if attr == "drafted" else None)
        except Exception:
            pass


def _prospect_entry_key(entry: Dict[str, Any]) -> str:
    return str(entry.get("key") or entry.get("prospect_id") or entry.get("player_id") or "")


def _draft_live_eligible(
    session: FranchiseSession,
    state: Dict[str, Any],
    board: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Board entries that are not drafted and still exist as live player entities."""
    available = _available_entries(state, board)
    league = getattr(session.sim, "league", None)
    if league is None:
        return list(available)

    # Build a single id -> player index for this pass instead of scanning the
    # whole development pool once per entry. The previous implementation called
    # _find_prospect_player() for every available prospect, and each of those
    # scanned every development-league player — turning eligibility into an
    # O(available x pool) operation on every pick. Indexing once makes it O(pool
    # + available) while performing the exact same drafted / rights checks.
    player_index: Dict[str, Any] = {}
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                ppid = str(getattr(p, "id", "") or "")
                if ppid:
                    player_index.setdefault(ppid, p)
    for tm in getattr(league, "teams", None) or []:
        for roster_attr in ("ahl_roster", "echl_roster", "prospect_roster"):
            for p in getattr(tm, roster_attr, None) or []:
                ppid = str(getattr(p, "id", "") or "")
                if ppid:
                    player_index.setdefault(ppid, p)
    try:
        from services.draft_player_registry import ensure_players_by_id
        reg = ensure_players_by_id(league)
        if isinstance(reg, dict):
            for ppid, p in reg.items():
                player_index.setdefault(str(ppid), p)
    except Exception:
        pass

    # Empty live pool (tests / early bootstrap): board availability is authoritative.
    if not player_index:
        return list(available)

    eligible: List[Dict[str, Any]] = []
    for entry in available:
        pid = _prospect_entry_key(entry)
        if not pid:
            continue
        player = player_index.get(pid)
        if player is None:
            # Fall back to the exhaustive lookup only for the rare id the index
            # missed, so correctness never depends on the fast path alone.
            player, _, _ = _find_prospect_player(session, pid)
        if player is None:
            continue
        if bool(getattr(player, "drafted", False)):
            continue
        if str(getattr(player, "nhl_rights_team_id", "") or getattr(player, "rights_team_id", "") or ""):
            # Already held by an NHL org
            continue
        eligible.append(entry)
    return eligible


def _classify_pick_team_relative(
    entry: Dict[str, Any],
    pub_rank: int,
    tb_rank: int,
    overall: int,
    philosophy: str,
    needs: List[Dict[str, Any]],
    *,
    best_available_rank: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Public-board grades drive Reach/Steal/Value/Expected/Off Board.
    Team-board delta is stored only as private context (never overwrites public labels).
    """
    base = _classify_pick(
        entry, pub_rank, tb_rank, overall, philosophy, needs,
        best_available_rank=best_available_rank,
    )
    team_delta = int(tb_rank) - int(overall) if tb_rank else None
    base["team_board_delta"] = team_delta
    base["was_public_reach"] = bool(base.get("was_reach"))
    # Keep pick_class aligned with public selection_label
    base["pick_class"] = str(base.get("selection_label") or base.get("pick_classification") or "expected").lower().replace(" ", "_")
    return base


def _selection_board_snapshot(
    entry: Dict[str, Any],
    tb_entry: Dict[str, Any],
    *,
    pub_rank: int,
    tb_rank: int,
    needs: List[Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "public_rank": pub_rank,
        "team_board_rank": tb_rank,
        "consensus_range": entry.get("consensus_range") or entry.get("rank_range"),
        "scouting_confidence": entry.get("scouting_confidence") or tb_entry.get("scouting_confidence"),
        "estimated_potential": entry.get("potential_score") or tb_entry.get("potential_score"),
        "team_need": needs[0] if needs else None,
        "pick_reasoning": reason,
        "comparable": entry.get("comparable") or entry.get("comp"),
        "combine_result": entry.get("combine_result") or entry.get("combine_summary"),
        "stock_movement": entry.get("stock_label") or entry.get("stock_reason"),
    }


def _raise_impossible_draft_state(
    session: FranchiseSession,
    state: Dict[str, Any],
    board: Dict[str, Any],
) -> None:
    available = _available_entries(state, board)
    eligible = _draft_live_eligible(session, state, board)
    drafted = list(state.get("drafted_prospect_ids") or [])
    raise ValueError(
        "Impossible draft state: no eligible prospects "
        f"(overall_pick={state.get('overall_pick')}, draft_year={state.get('draft_year')}, "
        f"board_size={len(board.get('entries') or [])}, pool_eligible={len(eligible)}, "
        f"available_board={len(available)}, drafted={len(drafted)}, "
        f"unavailable={len(available) - len(eligible)})"
    )


def _available_entries(state: Dict[str, Any], board: Dict[str, Any]) -> List[Dict[str, Any]]:
    drafted = set(str(x) for x in (state.get("drafted_prospect_ids") or []))
    # Match on ANY identity field the entry might carry (key / prospect_id /
    # player_id) so a prospect can never remain "available" after being drafted
    # under a different identity field.
    out: List[Dict[str, Any]] = []
    for e in (board.get("entries") or []):
        ids = {
            str(e.get("key") or ""),
            str(e.get("prospect_id") or ""),
            str(e.get("player_id") or ""),
        }
        ids.discard("")
        if ids & drafted:
            continue
        out.append(e)
    return out


def _build_round_recap(session: FranchiseSession, rnd: int, picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    round_picks = [p for p in picks if int(p.get("round") or 0) == rnd]
    if not round_picks:
        return {}
    uid = str(session.user_team_id)
    user_round = [p for p in round_picks if str(p.get("team_id")) == uid]
    steals = sorted(round_picks, key=lambda p: int(p.get("final_rank") or 99) - int(p.get("overall_pick") or 0), reverse=True)
    reaches = [p for p in round_picks if p.get("was_reach")]
    goalies = [p for p in round_picks if str(p.get("position") or "").upper() == "G"]
    available_rank = {str(p.get("prospect_id")): int(p.get("final_rank") or 0) for p in picks}
    headline = ""
    if steals:
        s = steals[0]
        delta = int(s.get("final_rank") or 0) - int(s.get("overall_pick") or 0)
        if delta >= 8:
            headline = f"Steal: {s.get('prospect_name')} (#{s.get('final_rank')}) at pick {s.get('overall_pick')}"
    if reaches and not headline:
        r = reaches[0]
        headline = f"Reach: {r.get('prospect_name')} at pick {r.get('overall_pick')}"
    if goalies and len(goalies) >= 2:
        headline = headline or f"Goalie run: {len(goalies)} goalies in Round {rnd}"
    pos_counts: Dict[str, int] = {}
    for p in round_picks:
        pos = str(p.get("position") or "?").upper()
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    return {
        "round": rnd,
        "picks_count": len(round_picks),
        "headline": headline or f"Round {rnd} complete",
        "best_steal": steals[0] if steals else None,
        "biggest_reach": reaches[0] if reaches else None,
        "user_picks": user_round,
        "goalie_count": len(goalies),
        "position_summary": pos_counts,
    }


def initialize_entry_draft(session: FranchiseSession) -> Dict[str, Any]:
    if not getattr(session, "draft_combine_done", False):
        raise ValueError("Complete the Draft Combine before starting the Entry Draft")
    try:
        from services.franchise_scouting import ensure_team_scouting_profiles

        ensure_team_scouting_profiles(session)
    except Exception:
        pass
    board = finalize_draft_class_for_event(session)
    order = build_full_draft_order(session)
    draft_year = int(session.season_calendar_year) + 1
    po_source = "standings_fallback"
    if order:
        po_source = str(order[0].get("playoff_order_source") or "standings_fallback")
    cache = _ensure_draft_cache(session, board)
    state = {
        "draft_id": f"draft-{draft_year}-{session.session_id[:8]}",
        "draft_year": draft_year,
        "current_round": 1,
        "current_pick": 1,
        "overall_pick": 1,
        "current_team_id": order[0]["team_id"] if order else "",
        "draft_order": order,
        "pick_ownership": order,
        "completed_picks": [],
        "available_prospects": [e.get("key") for e in board.get("entries") or [] if e.get("key")],
        "drafted_prospect_ids": [],
        "user_team_id": str(session.user_team_id),
        "is_user_pick": str(order[0]["team_id"]) == str(session.user_team_id) if order else False,
        "draft_started": True,
        "draft_completed": False,
        "event_status": "live",
        "draft_results": [],
        "round_recaps": {},
        "team_draft_boards": {},
        "public_draft_board": board.get("entries") or [],
        "team_needs_snapshot": cache.get("team_needs") or {},
        "playoff_order_source": po_source,
        "draft_order_note": (
            "Order: lottery + playoff results"
            if po_source == "lottery_playoff_results"
            else "Order: standings fallback"
        ),
        "draft_order_source": po_source,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "location": "NHL Draft Floor",
        "class_strength": board.get("class_strength"),
        "storylines": [_prospect_storyline(e) for e in (board.get("entries") or [])[:8]],
        "_cache": cache,
    }
    session.draft_state = state
    session.draft_completed = False
    return get_entry_draft_payload(session)

def get_entry_draft_state(session: FranchiseSession) -> Dict[str, Any]:
    state = getattr(session, "draft_state", None) or {}
    if not state.get("draft_started"):
        return {"draft_started": False, "draft_completed": bool(session.draft_completed)}
    return get_entry_draft_payload(session)


# Fields that encode hidden truth or internal engine state and must never reach
# the frontend. Anything with a leading underscore is also stripped.
_PRIVATE_PROSPECT_KEYS = frozenset({
    "true_ovr",
    "true_potential_score",
    "true_potential",
    "dev_potential",
    "true_center_score",
    "base_score",
    "sanity_penalty",
    "is_transcendent",
    "transcendent_talent",
    "generational_goalie",
    "generational_cut",
    "pipeline_tier",
    "hidden_tier",
    "consensus_seed",
    "true_shot_score",
    "draft_audit",
    "aura_tier",
    "draft_hype_tier",
    "special_fx",
})


def _strip_private_fields(d: Any) -> Any:
    """Remove hidden-truth / internal fields from a serialized dict (shallow).

    Drops any key beginning with '_' (engine caches, scratch state) plus the
    explicit private-truth allowlist. Applied to every prospect row, board entry
    and the draft state before they leave the backend.
    """
    if not isinstance(d, dict):
        return d
    return {
        k: v
        for k, v in d.items()
        if not (isinstance(k, str) and k.startswith("_")) and k not in _PRIVATE_PROSPECT_KEYS
    }


def _sanitize_board_for_client(board: Dict[str, Any]) -> Dict[str, Any]:
    clean = _strip_private_fields(board)
    clean["entries"] = [_strip_private_fields(e) for e in (board.get("entries") or [])]
    return clean


def get_entry_draft_payload(
    session: FranchiseSession,
    *,
    refresh_ownership: bool = True,
) -> Dict[str, Any]:
    from services.franchise_sim import get_cached_draft_class_rankings
    from services.draft_pick_ownership import refresh_draft_order_ownership
    from services.draft_day_trade_offers import generate_draft_day_trade_offers
    from services.draft_rights_engine import rights_card_payload

    # Ownership is already refreshed on pick/trade execute paths. Skip the full
    # order re-walk when the caller says the registry is already current.
    if refresh_ownership:
        refresh_draft_order_ownership(session)

    state = dict(getattr(session, "draft_state", None) or {})
    board = get_cached_draft_class_rankings(session, session.sim)
    cache = _ensure_draft_cache(session, board)
    available = _available_entries(state, board)
    overall = int(state.get("overall_pick") or 1)
    order = list(state.get("draft_order") or [])
    current_slot = order[overall - 1] if overall <= len(order) else None
    current_team = str(current_slot.get("team_id") if current_slot else state.get("current_team_id") or "")
    user_id = str(session.user_team_id)

    offers = generate_draft_day_trade_offers(session, state, max_offers=3)

    # Team board only needs a window of the public board — not every remaining name.
    team_board_pool = available[:80]
    team_board = build_team_draft_board(session, user_id, team_board_pool, cache=cache) if user_id else []
    needs = list((state.get("team_needs_snapshot") or {}).get(current_team) or calculate_team_needs(session, current_team))
    phil = get_team_draft_philosophy(session, current_team) if current_team else {}
    # The user's private board must always be enriched with the USER's philosophy
    # and scouting department — never the CPU team currently on the clock.
    user_phil = get_team_draft_philosophy(session, user_id) if user_id else {}
    user_scout_quality = float(_team_scouting_profile(session, user_id).get("scouting_quality") or 60) if user_id else 60.0

    hist = getattr(session, "draft_stock_history", None) or {}
    overlay_cache = {}
    enriched_available = []
    from services.draft_prospect_profile import build_prospect_profile
    from services.draft_board_engine import enrich_board_entry_with_team_scouting

    draft_year = int(state.get("draft_year") or session.season_calendar_year + 1)
    user_team = session.team_by_id.get(user_id) if user_id else None
    roster_rows: List[Dict[str, Any]] = []
    team_status = None
    if user_team is not None:
        for p in list(getattr(user_team, "roster", None) or []) + list(
            getattr(user_team, "prospect_pool", None) or getattr(user_team, "prospects", None) or []
        ):
            roster_rows.append({
                "position": getattr(p, "position", None),
                "handedness": getattr(getattr(p, "identity", None), "shoots", None) or getattr(p, "shoots", None),
                "age": getattr(getattr(p, "identity", None), "age", None) or getattr(p, "age", None),
            })
        try:
            status_key = _team_status(session, user_id)
            team_status = {"key": "rebuilding" if status_key == "rebuilder" else ("cup_contender" if status_key == "contender" else status_key)}
        except Exception:
            team_status = None

    # Deep enrich only the pick window the UI actually browses. Remaining names
    # stay on the public board as light rows — building dossiers for ~250+
    # prospects every pick was a major per-pick cost.
    DEEP_ENRICH_LIMIT = 64
    team_board_by_key = {str(r.get("key") or ""): r for r in team_board}
    deep_keys = {str(r.get("key") or "") for r in available[:DEEP_ENRICH_LIMIT]}
    deep_keys.update(k for k in team_board_by_key.keys() if k)
    home_index = _build_dev_home_index(session)

    for idx, row in enumerate(available):
        pid = str(row.get("key") or "")
        ub = team_board_by_key.get(pid) or row
        deep = pid in deep_keys or idx < DEEP_ENRICH_LIMIT
        if deep:
            if pid not in overlay_cache:
                ov = _scouting_overlay(session, pid, team_id=user_id)
                _, _, notes = _scouting_event_adjustments(
                    session, user_id, row, ov, user_phil
                )
                overlay_cache[pid] = notes
            player, _, _ = _lookup_prospect_player(session, pid, home_index)
            rights = rights_card_payload(player) if player is not None else {}
            try:
                scouted = enrich_board_entry_with_team_scouting(
                    row,
                    team_id=user_id or "league",
                    scouting_quality=user_scout_quality,
                    draft_year=draft_year,
                )
            except Exception:
                scouted = {}
            try:
                dossier = build_prospect_profile(row, roster_rows=roster_rows, team_status=team_status)
            except Exception:
                dossier = {}
            enriched_available.append(
                {
                    **_strip_private_fields(row),
                    **_strip_private_fields(scouted),
                    "team_board_rank": ub.get("team_board_rank"),
                    "stock_history": list(hist.get(pid) or []),
                    "storyline": _prospect_storyline(row),
                    "scouting_event_notes": ub.get("scouting_notes") or overlay_cache.get(pid) or [],
                    "combine_status": ub.get("combine_score") or row.get("combine_status"),
                    "interview_notes": next((n for n in (ub.get("scouting_notes") or []) if "Interview" in n), None),
                    "rights_card": rights,
                    "dossier": dossier,
                    "intel_layers": {
                        "public": {"rank": row.get("rank"), "stock_label": row.get("stock_label")},
                        "scouting": {
                            "team_board_rank": ub.get("team_board_rank"),
                            "confidence": row.get("scouting_confidence"),
                            "notes": ub.get("scouting_notes") or [],
                            "scouted_overall_estimate": scouted.get("scouted_overall_estimate"),
                            "scouted_potential_range": scouted.get("scouted_potential_range"),
                        },
                        "combine": {
                            "status": ub.get("combine_score") or row.get("combine_status"),
                            "interview": next((n for n in (ub.get("scouting_notes") or []) if "Interview" in n), None),
                        },
                    },
                }
            )
        else:
            enriched_available.append(
                {
                    **_strip_private_fields(row),
                    "team_board_rank": ub.get("team_board_rank"),
                    "stock_history": list(hist.get(pid) or [])[-3:],
                    "storyline": _prospect_storyline(row),
                }
            )

    completed = list(state.get("completed_picks") or [])
    for pick in completed:
        if pick.get("rights_card"):
            continue
        p, _, _ = _lookup_prospect_player(session, str(pick.get("prospect_id") or ""), home_index)
        if p is not None:
            pick["rights_card"] = rights_card_payload(p)

    payload = {
        **_strip_private_fields(state),
        "current_pick": current_slot,
        "current_team_id": current_team,
        "current_team_name": _display_team(session, current_team) if current_team else "",
        "is_user_pick": current_team == user_id and not state.get("draft_completed"),
        "is_traded_pick": bool(current_slot.get("is_traded")) if current_slot else False,
        "via_team_name": current_slot.get("via_team_name") if current_slot else None,
        "picks_until_user": state.get("picks_until_user"),
        "available_prospects": enriched_available,
        "public_draft_board": [_strip_private_fields(e) for e in (board.get("entries") or [])],
        "team_board": [_strip_private_fields(r) for r in team_board[:60]],
        "team_needs": needs,
        "team_needs_snapshot": dict(state.get("team_needs_snapshot") or cache.get("team_needs") or {}),
        "team_philosophy": phil,
        "draft_class_rankings": _sanitize_board_for_client(board),
        "completed_picks": [_strip_private_fields(p) for p in completed],
        "draft_results": [_strip_private_fields(r) for r in (state.get("draft_results") or completed)],
        "total_picks": TOTAL_PICKS,
        "picks_remaining": max(0, TOTAL_PICKS - len(completed)),
        "round_recaps": dict(state.get("round_recaps") or {}),
        "trade_offers": offers,
        "draft_day_trade_offers": offers,
        "pick_trade_offers": offers,
    }
    # A completed draft always carries the post-draft recap so the recap screen
    # (including reloads) renders entirely from backend-computed data.
    if state.get("draft_completed"):
        recap = get_draft_recap(session)
        state["draft_recap"] = recap
        session.draft_state = state
        payload["recap"] = recap
    # Serializer output only — mutable authority remains draft_state + pick registry.
    session.draft_payload = payload
    return payload


def _execute_pick(
    session: FranchiseSession,
    prospect_id: str,
    *,
    team_id: Optional[str] = None,
    user_initiated: bool = False,
    defer_payload: bool = False,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    # Serialize the entire read-validate-mutate cycle per session (item: concurrency).
    with _draft_lock(session):
        return _execute_pick_locked(
            session,
            prospect_id,
            team_id=team_id,
            user_initiated=user_initiated,
            defer_payload=defer_payload,
            request_id=request_id,
        )


def _execute_pick_locked(
    session: FranchiseSession,
    prospect_id: str,
    *,
    team_id: Optional[str] = None,
    user_initiated: bool = False,
    defer_payload: bool = False,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    from services.franchise_sim import get_cached_draft_class_rankings, invalidate_session_payload_caches
    from services.draft_pick_ownership import apply_registry_owner_to_slot, resolve_slot_owner
    from services.draft_rights_engine import rights_card_payload

    state = getattr(session, "draft_state", None) or {}
    if state.get("draft_completed"):
        raise ValueError("Draft already completed")
    if not state.get("draft_started"):
        raise ValueError("Draft has not started")

    # Idempotency: repeated request for an already-completed pick returns the prior result.
    req = str(request_id or "")
    if req:
        prior = next(
            (p for p in (state.get("completed_picks") or []) if str(p.get("request_id") or "") == req),
            None,
        )
        if prior:
            out = {"ok": True, "pick_result": prior, "idempotent": True}
            if not defer_payload:
                out["draft"] = get_entry_draft_payload(session, refresh_ownership=False)
            return out

    board = get_cached_draft_class_rankings(session, session.sim)
    cache = _ensure_draft_cache(session, board)
    available = _draft_live_eligible(session, state, board)
    pid = str(prospect_id or "").strip()
    entry = next((e for e in available if _prospect_entry_key(e) == pid), None) if pid else None
    if entry is None:
        if not available:
            _raise_impossible_draft_state(session, state, board)
        if pid:
            # A specific prospect was requested but is not eligible. NEVER silently
            # substitute another player — surface a clear conflict instead so the
            # frontend can refresh and the user's click can't draft someone else.
            raise ValueError(
                f"Prospect '{pid}' is not available to draft "
                "(already selected, invalid ID, or not in this draft class)."
            )
        # No prospect specified → deliberate auto-pick (best available).
        entry = available[0]
        pid = _prospect_entry_key(entry)
        if not pid:
            _raise_impossible_draft_state(session, state, board)

    overall = int(state.get("overall_pick") or 1)
    draft_year = int(state.get("draft_year") or session.season_calendar_year + 1)
    order = list(state.get("draft_order") or [])
    if overall > len(order):
        raise ValueError("No picks remaining")

    # Live ownership — never trust the snapshot alone.
    slot = apply_registry_owner_to_slot(session, dict(order[overall - 1]), draft_year)
    order[overall - 1] = slot
    state["draft_order"] = order
    owner = resolve_slot_owner(session, slot)
    slot["team_id"] = owner
    if team_id and str(team_id) != owner:
        raise ValueError("Team does not own this pick")
    if user_initiated and owner != str(session.user_team_id):
        raise ValueError("Not your pick")

    # Registry must not already be resolved for this slot (idempotent duplicate pick_id).
    league = getattr(session.sim, "league", None)
    pick_id = str(slot.get("pick_id") or "")
    if league is not None and pick_id:
        row = (getattr(league, "draft_pick_registry", None) or {}).get(pick_id)
        if isinstance(row, dict) and row.get("resolved") and row.get("selected_prospect_id"):
            existing = next(
                (
                    p
                    for p in (state.get("completed_picks") or [])
                    if str(p.get("prospect_id")) == str(row.get("selected_prospect_id"))
                    and int(p.get("overall_pick") or 0) == overall
                ),
                None,
            )
            if existing:
                out = {"ok": True, "pick_result": existing, "idempotent": True}
                if not defer_payload:
                    out["draft"] = get_entry_draft_payload(session, refresh_ownership=False)
                return out
            raise ValueError(f"Pick already resolved: {pick_id}")

    player, block, tm = _find_prospect_player(session, pid)
    if player is None:
        raise ValueError("Prospect player entity not found")
    if bool(getattr(player, "drafted", False)):
        raise ValueError("Prospect already drafted")

    phil = cache["team_philosophies"].get(owner, get_team_draft_philosophy(session, owner))
    needs = cache["team_needs"].get(owner, [])
    team_board = build_team_draft_board(session, owner, available[:50], cache=cache)
    tb_entry = next((r for r in team_board if str(r.get("key")) == pid), entry)
    pub_rank = int(entry.get("rank") or overall)
    tb_rank = int(tb_entry.get("team_board_rank") or pub_rank)
    # Highest-ranked prospect still on the live board (BPA reference point).
    best_available_rank = min(
        (int(e.get("rank") or 999) for e in available),
        default=pub_rank,
    )
    cls = _classify_pick_team_relative(
        entry, pub_rank, tb_rank, overall, phil["philosophy"], needs,
        best_available_rank=best_available_rank,
    )
    scout_notes = tb_entry.get("scouting_notes") or []
    reason = _pick_reason(entry, cls, phil, needs, pub_rank, tb_rank, scout_notes)
    board_snap = _selection_board_snapshot(
        entry, tb_entry, pub_rank=pub_rank, tb_rank=tb_rank, needs=needs, reason=reason
    )

    pick_meta = {
        "round": int(slot.get("round") or 1),
        "pick_in_round": int(slot.get("pick_in_round") or 1),
        "overall_pick": overall,
        "draft_year": draft_year,
        "original_owner_team_id": str(slot.get("original_owner_team_id") or owner),
        "pick_id": pick_id,
    }

    # Atomic order: resolve the registry FIRST (the single conflict-detecting commit
    # under the session lock), then perform the heavy assignment. If assignment
    # fails, fully roll back BOTH the assignment and the registry resolution so a
    # retry can't create duplicate or ghost draft records.
    _mark_pick_resolved(session, slot, pid)
    try:
        _assign_drafted_prospect(session, player, owner, pick_meta, block, tm, entry)
    except Exception:
        try:
            _rollback_assigned_prospect(session, player, owner, pick_meta)
        except Exception:
            pass
        try:
            _unmark_pick_resolved(session, slot)
        except Exception:
            pass
        raise

    prev_round = int(slot.get("round") or 1)
    orig_owner = str(slot.get("original_owner_team_id") or owner)
    is_traded = bool(slot.get("is_traded")) or orig_owner != owner

    result = {
        "draft_id": state.get("draft_id"),
        "pick_id": pick_id,
        "request_id": req or f"{state.get('draft_id')}:{pick_id}:{pid}",
        "resolved_at": _now_iso(),
        "selected_prospect_id": pid,
        "round": pick_meta["round"],
        "pick_in_round": pick_meta["pick_in_round"],
        "overall_pick": overall,
        "team_id": owner,
        "team_name": _display_team(session, owner),
        "original_owner_team_id": orig_owner,
        "original_owner_team_name": slot.get("original_owner_team_name") or _display_team(session, orig_owner),
        "is_traded": is_traded,
        "via_team_name": slot.get("via_team_name") if is_traded else None,
        "prospect_id": pid,
        "prospect_name": str(entry.get("name") or ""),
        "position": str(entry.get("position") or ""),
        "league": str(entry.get("league") or entry.get("league_name") or ""),
        "nationality": str(entry.get("nationality") or ""),
        "final_rank": pub_rank,
        "preseason_rank": int(entry.get("preseason_rank") or pub_rank),
        "rank_delta": int(entry.get("preseason_rank") or pub_rank) - overall,
        "stock_label": entry.get("stock_label"),
        "stock_reason": entry.get("stock_reason"),
        "scouting_confidence": entry.get("scouting_confidence"),
        "potential_grade": entry.get("talent_grade") or entry.get("scout_tier"),
        # Scouted/public estimates only — never raw hidden ability.
        "floor_grade": (
            entry.get("current_ovr_estimate")
            or entry.get("scouted_overall_estimate")
            or entry.get("floor_score")
        ),
        "ceiling_grade": (
            entry.get("potential_score")
            or entry.get("expected_ceiling_estimate")
            or entry.get("ceiling_score")
        ),
        "risk_score": entry.get("risk"),
        "nhl_readiness": entry.get("nhl_readiness"),
        "player_type": entry.get("player_type"),
        "development_path": _development_path_for(entry, block),
        "nhl_eta": _nhl_eta_years(entry, overall),
        "pick_reason": reason,
        "board_snapshot": board_snap,
        "rights_card": rights_card_payload(player),
        **cls,
        "timestamp": _now_iso(),
    }

    completed = list(state.get("completed_picks") or [])
    completed.append(result)
    drafted_ids = list(state.get("drafted_prospect_ids") or [])
    # Record every identity field this entry carries so availability checks catch
    # it regardless of which id the board/registry uses downstream.
    for ident in (pid, entry.get("key"), entry.get("prospect_id"), entry.get("player_id")):
        s = str(ident or "")
        if s and s not in drafted_ids:
            drafted_ids.append(s)
    next_overall = overall + 1
    next_slot = (
        apply_registry_owner_to_slot(session, dict(order[next_overall - 1]), draft_year)
        if next_overall <= len(order)
        else None
    )
    if next_slot is not None:
        order[next_overall - 1] = next_slot
    done = next_overall > len(order)

    round_recaps = dict(state.get("round_recaps") or {})
    if next_slot and int(next_slot.get("round") or 1) != prev_round:
        round_recaps[str(prev_round)] = _build_round_recap(session, prev_round, completed)
    if done:
        round_recaps[str(prev_round)] = _build_round_recap(session, prev_round, completed)

    state.update(
        {
            "draft_order": order,
            "overall_pick": next_overall if not done else overall,
            "current_round": int(next_slot.get("round") if next_slot else prev_round),
            "current_pick": int(next_slot.get("pick_in_round") if next_slot else 0),
            "current_team_id": str(next_slot.get("team_id") if next_slot else ""),
            "completed_picks": completed,
            "draft_results": completed,
            "drafted_prospect_ids": drafted_ids,
            "draft_completed": done,
            "event_status": "complete" if done else "live",
            "round_recaps": round_recaps,
            "updated_at": _now_iso(),
            "is_user_pick": (
                str(next_slot.get("team_id")) == str(session.user_team_id) if next_slot and not done else False
            ),
            "trade_offers": [],
            "draft_day_trade_offers": [],
        }
    )
    # Refresh organizational needs after the selection so subsequent picks see updated depth.
    try:
        snap = dict(state.get("team_needs_snapshot") or {})
        snap[str(owner)] = calculate_team_needs(session, str(owner))
        if next_slot and next_slot.get("team_id"):
            nxt = str(next_slot.get("team_id"))
            snap[nxt] = calculate_team_needs(session, nxt)
        state["team_needs_snapshot"] = snap
        cache = state.get("_cache") or {}
        if isinstance(cache, dict):
            cache_needs = dict(cache.get("team_needs") or {})
            cache_needs[str(owner)] = snap[str(owner)]
            if next_slot and next_slot.get("team_id"):
                cache_needs[str(next_slot.get("team_id"))] = snap[str(next_slot.get("team_id"))]
            cache["team_needs"] = cache_needs
            state["_cache"] = cache
    except Exception:
        pass
    session.draft_state = state
    session.draft_completed = done
    invalidate_session_payload_caches(session, "draft_pick")
    out = {"ok": True, "pick_result": result}
    if not defer_payload:
        # Callers already refreshed ownership before executing the pick.
        out["draft"] = get_entry_draft_payload(session, refresh_ownership=False)
    return out

def execute_user_draft_pick(
    session: FranchiseSession,
    prospect_id: str,
    *,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    from services.draft_pick_ownership import refresh_draft_order_ownership, resolve_slot_owner

    refresh_draft_order_ownership(session)
    state = getattr(session, "draft_state", None) or {}
    overall = int(state.get("overall_pick") or 1)
    order = list(state.get("draft_order") or [])
    slot = order[overall - 1] if overall <= len(order) else {}
    owner = resolve_slot_owner(session, slot) if slot else ""
    if owner != str(session.user_team_id):
        raise ValueError("Not your turn to pick")
    return _execute_pick(
        session,
        prospect_id,
        team_id=str(session.user_team_id),
        user_initiated=True,
        request_id=request_id,
    )


def execute_cpu_draft_pick(session: FranchiseSession) -> Dict[str, Any]:
    from services.draft_pick_ownership import refresh_draft_order_ownership, resolve_slot_owner

    refresh_draft_order_ownership(session)
    if _maybe_cpu_draft_day_pick_swap(session):
        refresh_draft_order_ownership(session)
    state = getattr(session, "draft_state", None) or {}
    overall = int(state.get("overall_pick") or 1)
    order = list(state.get("draft_order") or [])
    if overall > len(order):
        raise ValueError("Draft complete")
    slot = order[overall - 1]
    owner = resolve_slot_owner(session, slot)
    if owner == str(session.user_team_id):
        raise ValueError("User team on the clock — make a selection")

    from services.franchise_sim import get_cached_draft_class_rankings

    board = get_cached_draft_class_rankings(session, session.sim)
    cache = _ensure_draft_cache(session, board)
    # Select from the SAME pool that execution validates against, so the CPU can
    # never choose a player it is then not allowed to draft.
    available = _draft_live_eligible(session, state, board)
    if not available:
        _raise_impossible_draft_state(session, state, board)

    chosen = _cpu_select_prospect(session, owner, overall, available, cache)
    return _execute_pick(session, _prospect_entry_key(chosen), team_id=owner, user_initiated=False)


def _maybe_cpu_draft_day_pick_swap(session: FranchiseSession) -> Optional[Dict[str, Any]]:
    """Bounded CPU-CPU pick swap before a selection; routes through validated trade + popup."""
    state = getattr(session, "draft_state", None) or {}
    if not state.get("draft_started") or state.get("draft_completed"):
        return None
    overall = int(state.get("overall_pick") or 1)
    order = list(state.get("draft_order") or [])
    if overall < 1 or overall > len(order):
        return None
    # Cap draft-day CPU swaps so draft flow stays bounded.
    swaps_done = int(state.get("cpu_draft_swaps_completed", 0) or 0)
    if swaps_done >= 12:
        return None
    if swaps_done >= 1 and overall % 7 != 0:
        # Sparse attempts after the first few successful swaps.
        if overall > 32 and overall % 11 != 0:
            return None

    user_id = str(getattr(session, "user_team_id", "") or "")
    on_slot = order[overall - 1]
    on_clock = str(on_slot.get("team_id") or "")
    if not on_clock or on_clock == user_id:
        return None

    profiles = dict(getattr(session, "cpu_franchise_profiles", None) or {})
    ideo = dict((profiles.get(on_clock) or {}).get("ideology") or {})
    aggression = float(ideo.get("aggression", 0.5) or 0.5)
    pick_protect = float(ideo.get("draft_pick_protection", 0.5) or 0.5)
    # Trade-up preference: aggressive / low pick-protection teams more often.
    want_up = aggression >= 0.52 and pick_protect <= 0.62
    rng = getattr(getattr(session, "sim", None), "rng", None)
    roll = float(rng.random()) if rng is not None else 0.9
    p = 0.08 + (0.1 if want_up else 0.0) + (0.04 if overall <= 20 else 0.0)
    if roll > min(0.28, p):
        return None

    # Partner: prefer adjacent slots first (smaller value gap), then look ahead/back.
    partner_idx = None
    if want_up and overall > 1:
        for delta in (1, 2, 3, 4):
            if overall - 1 - delta < 0:
                break
            cand = order[overall - 1 - delta]
            partner = str(cand.get("team_id") or "")
            if partner and partner != on_clock and partner != user_id:
                partner_idx = overall - 1 - delta
                break
    if partner_idx is None:
        for delta in (1, 2, 3, 4, 5):
            idx = overall - 1 + delta
            if idx >= len(order):
                break
            cand = order[idx]
            partner = str(cand.get("team_id") or "")
            if partner and partner != on_clock and partner != user_id:
                partner_idx = idx
                break
    if partner_idx is None:
        return None

    partner_slot = order[partner_idx]
    partner_id = str(partner_slot.get("team_id") or "")
    pick_a = str(on_slot.get("pick_id") or "")
    pick_b = str(partner_slot.get("pick_id") or "")
    if not pick_a or not pick_b:
        return None

    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return None
    try:
        setattr(league, "_franchise_user_team_id", user_id)
    except Exception:
        pass

    team_by_id = dict(getattr(session, "team_by_id", None) or {})
    package = {
        on_clock: [{"type": "pick", "id": pick_b, "team": partner_id}],
        partner_id: [{"type": "pick", "id": pick_a, "team": on_clock}],
    }
    try:
        from app.sim_engine.trades.trade_evaluator import evaluate_trade_package
        from app.sim_engine.trades.trade_executor import execute_validated_trade
        from app.sim_engine.trades.cpu_trade_proposer import build_league_trade_context
        from services.draft_pick_ownership import sync_draft_clock_after_trade
        from services.franchise_sim import _enqueue_cpu_trade_popup
    except Exception:
        return None

    try:
        ctx = build_league_trade_context(
            league,
            calendar_cursor=int(getattr(session, "calendar_cursor", 0) or 0),
            regular_season_last_index=int(getattr(session, "nhl_regular_season_last_index", 192) or 192),
        )
        ctx["cpu_ambient_trade"] = True
        ctx["draft_day_trade"] = True
        evaluation = evaluate_trade_package(
            package,
            league=league,
            team_by_id=team_by_id,
            context=ctx,
            user_team_id=None,
        )
        if not evaluation.get("can_execute") or not evaluation.get("accepted"):
            return None
        result = execute_validated_trade(
            evaluation,
            league=league,
            team_by_id=team_by_id,
            context=ctx,
            user_team_id=None,
        )
    except Exception:
        return None

    sync_draft_clock_after_trade(session)
    moving_up = partner_idx < (overall - 1)
    reason_codes = ["DRAFT_TRADE_UP" if moving_up else "DRAFT_TRADE_DOWN", "PICK_VALUE_REALLOCATION"]
    if moving_up:
        reason_text = "Moved up to select a priority organizational target."
    else:
        reason_text = "Moved down while remaining inside the same prospect tier."
    headline = (
        f"Draft floor: {on_clock} moves {'up' if moving_up else 'down'} with {partner_id}"
    )
    ev = {
        "trade_id": result.get("trade_id"),
        "from_team_id": on_clock if moving_up else partner_id,
        "to_team_id": partner_id if moving_up else on_clock,
        "team": partner_id if moving_up else on_clock,
        "headline": headline,
        "execution": result,
        "trade_category": "draft_trade",
        "importance": "standard",
        "reason_codes": reason_codes,
        "reason_text": reason_text,
        "draft_context": True,
        "outgoing": [f"Pick #{overall}"],
        "incoming": [f"Pick #{partner_idx + 1}"],
    }
    cal_idx = int(getattr(session, "calendar_cursor", 0) or 0)
    iso = ""
    cal = getattr(session, "nhl_calendar", None) or []
    if 0 <= cal_idx < len(cal):
        iso = str(cal[cal_idx].get("iso") or "")
    _enqueue_cpu_trade_popup(session, ev, calendar_idx=cal_idx, iso=iso)
    state = getattr(session, "draft_state", None) or {}
    state["cpu_draft_swaps_completed"] = swaps_done + 1
    log = list(state.get("draft_trade_log") or [])
    log.append(
        {
            "trade_id": result.get("trade_id"),
            "overall_pick": overall,
            "from_team_id": on_clock,
            "to_team_id": partner_id,
            "reason_codes": reason_codes,
        }
    )
    state["draft_trade_log"] = log[-40:]
    session.draft_state = state
    return ev


def _batch_cpu_picks(
    session: FranchiseSession,
    *,
    until_user: bool = True,
    until_round_end: bool = False,
    complete_all: bool = False,
) -> List[Dict[str, Any]]:
    from services.franchise_sim import get_cached_draft_class_rankings
    from services.draft_pick_ownership import refresh_draft_order_ownership

    picks: List[Dict[str, Any]] = []
    cache = None
    safety = 0
    start_round = None
    # Ownership + draft-day swap attempts are expensive; do them once up front and
    # only again after a successful mid-draft trade, not on every CPU selection.
    try:
        refresh_draft_order_ownership(session)
    except Exception:
        pass

    while safety < TOTAL_PICKS:
        safety += 1
        state = getattr(session, "draft_state", None) or {}
        if state.get("draft_completed"):
            break
        overall = int(state.get("overall_pick") or 1)
        order = list(state.get("draft_order") or [])
        if overall > len(order):
            break
        # Attempt a draft-day swap occasionally in early rounds only — not every pick.
        if overall <= 64 and (overall == 1 or overall % 4 == 1):
            try:
                swapped = _maybe_cpu_draft_day_pick_swap(session)
                if swapped:
                    refresh_draft_order_ownership(session)
                    state = getattr(session, "draft_state", None) or {}
                    order = list(state.get("draft_order") or [])
                    overall = int(state.get("overall_pick") or 1)
                    if overall > len(order):
                        break
            except Exception:
                pass
        slot = order[overall - 1]
        if start_round is None:
            start_round = int(slot.get("round") or 1)

        if until_user and not complete_all and str(slot.get("team_id")) == str(session.user_team_id):
            break
        if until_round_end and int(slot.get("round") or 1) != start_round:
            break

        board = get_cached_draft_class_rankings(session, session.sim)
        if cache is None:
            cache = _ensure_draft_cache(session, board)
        available = _draft_live_eligible(session, state, board)
        if not available:
            _raise_impossible_draft_state(session, state, board)
        owner = str(slot.get("team_id") or "")
        chosen = _cpu_select_prospect(session, owner, overall, available, cache)
        res = _execute_pick(session, _prospect_entry_key(chosen), team_id=owner, defer_payload=True)
        picks.append(res.get("pick_result") or {})
        cache = _ensure_draft_cache(session, board)

        if not complete_all and not until_user and not until_round_end:
            break

    return picks


def _build_batch_summary(session: FranchiseSession, picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not picks:
        return {}
    from services.franchise_sim import get_cached_draft_class_rankings

    state = getattr(session, "draft_state", None) or {}
    rankings = get_cached_draft_class_rankings(session, session.sim)
    avail = _available_entries(state, rankings)

    steals = sorted(
        picks,
        key=lambda p: int(p.get("final_rank") or 0) - int(p.get("overall_pick") or 0),
        reverse=True,
    )
    reaches = sorted(
        picks,
        key=lambda p: int(p.get("overall_pick") or 0) - int(p.get("final_rank") or 0),
        reverse=True,
    )
    uid = str(session.user_team_id)
    next_user = None
    order = list(state.get("draft_order") or [])
    overall = int(state.get("overall_pick") or 1)
    if overall <= len(order) and str(order[overall - 1].get("team_id")) == uid:
        next_user = order[overall - 1]
    best_avail = avail[0] if avail else None
    return {
        "picks_made": len(picks),
        "biggest_steal": steals[0] if steals else None,
        "biggest_reach": reaches[0] if reaches else None,
        "user_next_pick": next_user,
        "best_available": {
            "prospect_id": best_avail.get("key"),
            "name": best_avail.get("name"),
            "rank": best_avail.get("rank"),
            "position": best_avail.get("position"),
        } if best_avail else None,
    }


def sim_entry_draft_to_user_pick(session: FranchiseSession) -> Dict[str, Any]:
    picks = _batch_cpu_picks(session, until_user=True)
    summary = _build_batch_summary(session, picks)
    payload = get_entry_draft_payload(session, refresh_ownership=False)
    payload["recap"] = get_draft_recap(session)
    return {
        "ok": True,
        "simulated_picks": picks,
        "batch_count": len(picks),
        "batch_summary": summary,
        "draft": payload,
    }


def sim_entry_draft_round(session: FranchiseSession) -> Dict[str, Any]:
    picks = _batch_cpu_picks(session, until_round_end=True)
    summary = _build_batch_summary(session, picks)
    payload = get_entry_draft_payload(session, refresh_ownership=False)
    payload["recap"] = get_draft_recap(session)
    return {
        "ok": True,
        "simulated_picks": picks,
        "batch_count": len(picks),
        "batch_summary": summary,
        "draft": payload,
    }


def complete_entry_draft(session: FranchiseSession) -> Dict[str, Any]:
    state = getattr(session, "draft_state", None) or {}
    if not state.get("draft_started"):
        raise ValueError("Draft not started")
    picks = _batch_cpu_picks(session, until_user=False, complete_all=True)
    state = getattr(session, "draft_state", None) or {}
    # Guarantee the draft is flagged complete once the order is exhausted, even if
    # the final pick loop terminated on the safety counter.
    order = list(state.get("draft_order") or [])
    if int(state.get("overall_pick") or 1) > len(order):
        state["draft_completed"] = True
        session.draft_state = state
    session.draft_completed = bool(state.get("draft_completed"))
    recap = get_draft_recap(session)
    state["draft_recap"] = recap
    session.draft_state = state
    payload = get_entry_draft_payload(session, refresh_ownership=False)
    payload["recap"] = recap
    return {
        "ok": True,
        "simulated_picks": picks,
        "batch_count": len(picks),
        "batch_summary": _build_batch_summary(session, picks),
        "draft": payload,
        "recap": recap,
    }

def get_draft_recap(session: FranchiseSession) -> Dict[str, Any]:
    from services.franchise_sim import get_cached_draft_class_rankings

    state = getattr(session, "draft_state", None) or {}
    results = list(state.get("draft_results") or [])
    user_id = str(session.user_team_id)
    user_picks = [r for r in results if str(r.get("team_id")) == user_id]
    drafted_ids = {str(r.get("prospect_id")) for r in results}

    def pub_delta(r: Dict[str, Any]) -> int:
        """Value vs PUBLIC board. Positive = fell to you (value/steal); negative = reach."""
        d = r.get("public_rank_delta")
        if d is None:
            d = r.get("public_board_delta")
        if d is None:
            try:
                d = int(r.get("overall_pick") or 0) - int(r.get("final_rank") or 0)
            except (TypeError, ValueError):
                d = 0
        try:
            return int(d)
        except (TypeError, ValueError):
            return 0

    # Steal/reach classification is anchored ONLY to the public board delta so a
    # single pick can never be labelled both a steal and a reach (item 19 & 21).
    # Sign guards guarantee steals always carry a positive value delta (fell to
    # the team) and reaches a negative one (taken early), so the displayed value
    # never contradicts the label and the sort direction is unambiguous.
    steals = sorted(
        [r for r in results if r.get("was_steal") and pub_delta(r) > 0],
        key=pub_delta,
        reverse=True,
    )[:8]
    reaches = sorted(
        [r for r in results if r.get("was_reach") and pub_delta(r) < 0],
        key=pub_delta,
    )[:8]
    steal_ids = {str(r.get("prospect_id")) for r in steals}
    reaches = [r for r in reaches if str(r.get("prospect_id")) not in steal_ids][:8]
    surprises = [
        r for r in results
        if "Off Board" in (r.get("pick_tags") or [])
        or str(r.get("pick_classification") or "") == "Off Board"
        or bool(r.get("was_off_board"))
    ][:8]
    first_round = [r for r in results if int(r.get("round") or 0) == 1]
    goalies = [r for r in results if str(r.get("position") or "").upper() == "G"]

    board = get_cached_draft_class_rankings(session, session.sim)
    undrafted = [
        e for e in (board.get("entries") or [])
        if str(e.get("key")) not in drafted_ids
    ][:12]

    # Enrich the user's picks with profile fields (age, draft eligibility, handedness)
    # so the recap can open a full player card. Sourced from the board entry first,
    # then the live prospect/player object — never fabricated.
    board_by_id: Dict[str, Dict[str, Any]] = {}
    for e in (board.get("entries") or []):
        for ident in (e.get("key"), e.get("prospect_id"), e.get("player_id")):
            if ident is not None:
                board_by_id.setdefault(str(ident), e)
    for p in user_picks:
        pid = str(p.get("prospect_id") or "")
        entry = board_by_id.get(pid) or {}
        age = p.get("age") if p.get("age") is not None else entry.get("age")
        elig = p.get("draft_eligibility_year") or entry.get("draft_eligibility_year")
        shoots = p.get("shoots") or entry.get("handedness") or entry.get("shoots")
        if age is None or elig is None or not shoots:
            try:
                player, _, _ = _find_prospect_player(session, pid)
            except Exception:
                player = None
            if player is not None:
                pident = getattr(player, "identity", None)
                if age is None:
                    age = getattr(pident, "age", None)
                    if age is None:
                        age = getattr(player, "age", None)
                if elig is None:
                    elig = getattr(player, "draft_eligibility_year", None)
                if not shoots:
                    shoots = getattr(pident, "shoots", None) or getattr(player, "shoots", None)
        if age is not None:
            p["age"] = age
        if elig is not None:
            p["draft_eligibility_year"] = elig
        if shoots:
            p["shoots"] = shoots

    biggest_faller = None
    for r in results:
        delta = pub_delta(r)
        if biggest_faller is None or delta > pub_delta(biggest_faller):
            biggest_faller = r

    best_goalie = next((r for r in goalies if pub_delta(r) >= 0), goalies[0] if goalies else None)
    weirdest = surprises[0] if surprises else None

    team_picks: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        team_picks.setdefault(str(r.get("team_id")), []).append(r)

    need_scores: Dict[str, float] = {}
    risk_scores: Dict[str, float] = {}
    safe_scores: Dict[str, float] = {}
    for tid, picks in team_picks.items():
        needs = calculate_team_needs(session, tid)
        need_cats = {n["category"] for n in needs[:3]}
        # Count each pick at most once (a pick either fills a need or it doesn't).
        need_hits = sum(
            1 for p in picks
            if any(
                (n in ("Franchise Center", "Center Depth") and str(p.get("position")) == "C")
                or (n in ("Top-Six Winger", "Wing Depth") and str(p.get("position")) in ("LW", "RW", "W"))
                or (n == "Goalie Pipeline" and str(p.get("position")) == "G")
                or (n == "Right-Shot Defense" and _entry_is_right_shot_d(p))
                for n in need_cats
            )
        )
        need_scores[tid] = need_hits / max(1, len(picks))
        risk_scores[tid] = sum(1 for p in picks if str(p.get("risk_score") or p.get("risk")) == "High")
        safe_scores[tid] = sum(1 for p in picks if str(p.get("risk_score") or p.get("risk")) == "Low")

    best_needs_team = max(need_scores, key=need_scores.get) if need_scores else None
    riskiest_team = max(risk_scores, key=risk_scores.get) if risk_scores else None
    safest_team = max(safe_scores, key=safe_scores.get) if safe_scores else None
    aggressive_team = max(
        team_picks.keys(),
        key=lambda t: sum(1 for p in team_picks[t] if p.get("was_reach")),
        default=None,
    )
    conservative_team = min(
        team_picks.keys(),
        key=lambda t: sum(1 for p in team_picks[t] if p.get("was_reach") or p.get("was_steal")),
        default=None,
    ) if team_picks else None

    round_value: Dict[int, int] = {}
    for r in results:
        rnd = int(r.get("round") or 1)
        if int(r.get("final_rank") or 999) <= 75:
            round_value[rnd] = round_value.get(rnd, 0) + 1
    best_value_round = max(round_value, key=round_value.get) if round_value else None

    fall_reasons: List[str] = []
    scouting = getattr(session, "scouting_state", None) or {}
    combine_map = scouting.get("combine_results") if isinstance(scouting.get("combine_results"), dict) else {}
    for e in undrafted[:8]:
        pid = str(e.get("key") or "")
        comb = combine_map.get(pid) or {}
        if comb.get("medical_flag"):
            fall_reasons.append(f"Medical slide: {e.get('name')} fell after combine medical flags.")
        elif e.get("character_concerns"):
            fall_reasons.append(f"Character concerns kept {e.get('name')} on the board.")
        elif str(e.get("position") or "").upper() == "G":
            fall_reasons.append(f"Goalie volatility: {e.get('name')} went undrafted.")
        elif float(e.get("scouting_confidence") or 50) < 45:
            fall_reasons.append(f"Low scouting confidence hurt {e.get('name')}.")

    user_identity: List[str] = []
    user_pos: Dict[str, int] = {}
    for r in user_picks:
        pos = str(r.get("position") or "?").upper()
        user_pos[pos] = user_pos.get(pos, 0) + 1
    if user_pos.get("C", 0) >= 2:
        user_identity.append("center depth")
    if user_pos.get("D", 0) >= 2:
        user_identity.append("defense pipeline")
    if user_pos.get("G", 0) >= 1:
        user_identity.append("goalie investment")
    if sum(1 for r in user_picks if r.get("was_steal")) >= 2:
        user_identity.append("value hunting")
    if sum(1 for r in user_picks if str(r.get("risk_score") or r.get("risk")) == "High") >= 2:
        user_identity.append("risk swing")

    headlines: List[str] = []
    storyline_ids: set = set()
    if steals:
        s = steals[0]
        storyline_ids.add(str(s.get("prospect_id")))
        headlines.append(
            f"Biggest steal: {s.get('team_name')} landed #{s.get('final_rank')} {s.get('prospect_name')} at pick {s.get('overall_pick')}."
        )
    if reaches:
        r = reaches[0]
        if str(r.get("prospect_id")) not in storyline_ids:
            storyline_ids.add(str(r.get("prospect_id")))
            spot = int(r.get("final_rank") or 0) - int(r.get("overall_pick") or 0)
            headlines.append(
                f"Biggest reach: {r.get('team_name')} took {r.get('prospect_name')} {spot} spots early."
            )
    # A faller IS a steal from the team's view; only surface it if it is a
    # different prospect than the headline steal to avoid contradictory labels.
    if (
        biggest_faller
        and pub_delta(biggest_faller) >= 12
        and str(biggest_faller.get("prospect_id")) not in storyline_ids
    ):
        storyline_ids.add(str(biggest_faller.get("prospect_id")))
        headlines.append(
            f"Biggest slide: {biggest_faller.get('prospect_name')} ranked #{biggest_faller.get('final_rank')}, went #{biggest_faller.get('overall_pick')}."
        )
    r2_goalies = [r for r in results if int(r.get("round") or 0) == 2 and str(r.get("position") or "").upper() == "G"]
    if len(r2_goalies) >= 2:
        headlines.append(f"Goalie run: {len(r2_goalies)} goalies came off the board in Round 2.")
    if best_value_round:
        headlines.append(f"Best value round: Round {best_value_round} produced {round_value[best_value_round]} players ranked inside the top 75.")
    if user_identity:
        headlines.append(f"Your draft identity: {' and '.join(user_identity)}.")

    pos_breakdown: Dict[str, int] = {}
    for r in user_picks:
        pos = str(r.get("position") or "?").upper()
        pos_breakdown[pos] = pos_breakdown.get(pos, 0) + 1

    first_round_storylines = [
        f"{r.get('prospect_name')} — {r.get('pick_reason') or r.get('pick_classification')}"
        for r in first_round[:8]
    ]

    # ---- Extended user-focused recap. Every string is kept to <=15 words. ----
    def _clip15(text: Any) -> str:
        return " ".join(str(text or "").split()[:15])

    def _class_score(picks: List[Dict[str, Any]]) -> float:
        score = 75.0
        for p in picks:
            cls = str(p.get("pick_classification") or "")
            if p.get("was_steal") or cls == "Steal":
                score += 5
            if p.get("was_reach") or cls == "Reach":
                score -= 5
            if p.get("was_value"):
                score += 3
            if p.get("was_bpa"):
                score += 2
            if p.get("was_team_need"):
                score += 2
            if str(p.get("risk_score")) == "High":
                score -= 1
        return max(0.0, min(100.0, score))

    def _letter(score: float) -> str:
        for cut, lab in (
            (94, "A+"), (90, "A"), (86, "A-"), (82, "B+"), (78, "B"), (74, "B-"),
            (70, "C+"), (65, "C"), (60, "C-"), (50, "D"), (0, "F"),
        ):
            if score >= cut:
                return lab
        return "F"

    def _ordinal(n: int) -> str:
        if 10 <= (n % 100) <= 20:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    def _clamp100(v: float) -> int:
        return int(max(0, min(100, round(v))))

    def _ceil(p: Dict[str, Any]) -> float:
        try:
            return float(p.get("ceiling_grade") or 0)
        except (TypeError, ValueError):
            return 0.0

    user_score = _class_score(user_picks)
    user_grade = _letter(user_score)

    team_scores = {tid: _class_score(ps) for tid, ps in team_picks.items()}
    ordered_scores = sorted(team_scores.values(), reverse=True)
    try:
        user_rank_idx = ordered_scores.index(team_scores.get(user_id, user_score)) + 1
    except ValueError:
        user_rank_idx = len(ordered_scores) or 1
    total_teams = max(1, len(team_scores))
    user_class_rank = f"{_ordinal(user_rank_idx)} of {total_teams}"

    star_rating = int(max(1, min(5, round(user_score / 20))))

    n_user = max(1, len(user_picks))
    steal_ct = sum(1 for p in user_picks if p.get("was_steal"))
    reach_ct = sum(1 for p in user_picks if p.get("was_reach"))
    value_ct = sum(1 for p in user_picks if p.get("was_value"))
    need_ct = sum(1 for p in user_picks if p.get("was_team_need"))
    high_risk_ct = sum(1 for p in user_picks if str(p.get("risk_score")) == "High")
    low_risk_ct = sum(1 for p in user_picks if str(p.get("risk_score")) == "Low")

    grade_breakdown = [
        {"label": "Value", "value": _clamp100(55 + steal_ct * 12 - reach_ct * 12)},
        {"label": "Need Fit", "value": _clamp100(need_ct / n_user * 100)},
        {"label": "Upside", "value": _clamp100(high_risk_ct / n_user * 100)},
        {"label": "Floor", "value": _clamp100(low_risk_ct / n_user * 100)},
    ]

    avg_value_delta = round(sum(pub_delta(p) for p in user_picks) / n_user, 1) if user_picks else 0.0

    best_user_pick = max(user_picks, key=pub_delta) if user_picks else None
    riskiest_user_pick = next((p for p in user_picks if str(p.get("risk_score")) == "High"), None)
    highest_ceiling_pick = max(user_picks, key=_ceil) if user_picks else None
    safest_user_pick = next((p for p in user_picks if str(p.get("risk_score")) == "Low"), None)
    first_round_user_pick = next((p for p in user_picks if int(p.get("round") or 0) == 1), None)
    boom_bust_pick = next(
        (p for p in user_picks if str(p.get("risk_score")) == "High" and _ceil(p) >= 80),
        riskiest_user_pick,
    )

    user_needs = calculate_team_needs(session, user_id)

    def _need_filled(cat: str) -> bool:
        return any(
            (cat in ("Franchise Center", "Center Depth") and str(p.get("position")) == "C")
            or (cat in ("Top-Six Winger", "Wing Depth") and str(p.get("position")) in ("LW", "RW", "W"))
            or (cat == "Goalie Pipeline" and str(p.get("position")) == "G")
            or (cat == "Right-Shot Defense" and _entry_is_right_shot_d(p))
            for p in user_picks
        )

    top_need_addressed = next((n["category"] for n in user_needs[:4] if _need_filled(n["category"])), None)
    biggest_need_ignored = next((n["category"] for n in user_needs[:4] if not _need_filled(n["category"])), None)

    # Real roster-need results for the recap panel (item 15). Empty list -> the
    # frontend shows an explicit "no pressing needs" empty state.
    needs_report: List[Dict[str, Any]] = [
        {
            "category": n.get("category"),
            "detail": _clip15(n.get("detail") or ""),
            "priority": n.get("priority"),
            "filled": _need_filled(n.get("category")),
        }
        for n in user_needs[:5]
        if isinstance(n, dict) and n.get("category")
    ]

    # Still on the board: projected draft slot vs. actual (undrafted) reality (item 23).
    still_on_board: List[Dict[str, Any]] = []
    for e in undrafted[:6]:
        rank = e.get("rank")
        rng = e.get("consensus_range") or e.get("rank_range") or e.get("projected_pick_range")
        if isinstance(rng, (list, tuple)) and len(rng) >= 2 and rng[0] and rng[1]:
            projected = f"#{int(rng[0])}-{int(rng[1])}"
        elif rank:
            projected = f"#{int(rank)}"
        else:
            projected = "Unranked"
        still_on_board.append({
            "name": e.get("name"),
            "position": e.get("position"),
            "projected": projected,
            "projected_rank": rank,
            "status": "Undrafted",
        })

    pipeline_grades: List[Dict[str, Any]] = []
    for pos, label in (("C", "Center"), ("LW", "Left Wing"), ("RW", "Right Wing"), ("D", "Defense"), ("G", "Goalie")):
        group = [p for p in user_picks if str(p.get("position")).upper() == pos]
        if not group:
            continue
        gscore = 70 + sum(6 if p.get("was_steal") else (-4 if p.get("was_reach") else 2) for p in group)
        pipeline_grades.append({
            "position": pos,
            "label": label,
            "count": len(group),
            "grade": _letter(max(0.0, min(100.0, gscore))),
        })

    if user_score >= 88:
        fan_reaction = "Fans love this class — the championship pipeline just got a jolt."
        media_reaction = "Analysts call it one of the night's smartest, most disciplined hauls."
        scout_reaction = "Scouts praise clean value picks with strong projectable NHL upside."
        gm_quote = "We stuck to our board and the value fell perfectly."
        rival_reaction = "Rival GMs quietly admit this class made the division tougher."
    elif user_score >= 78:
        fan_reaction = "Fans are pleased — a solid, sensible haul with real upside."
        media_reaction = "Media grades it a steady, needs-first draft with a few gambles."
        scout_reaction = "Scouts see dependable pros with one or two high-ceiling swings."
        gm_quote = "We addressed needs and grabbed value where it made sense."
        rival_reaction = "Rivals see a competent class that won't shift the balance."
    elif user_score >= 68:
        fan_reaction = "Fans are split — a few reaches offset some intriguing selections."
        media_reaction = "Media calls it uneven: nice swings undercut by early reaches."
        scout_reaction = "Scouts flag risk; upside exists but the floor looks shaky."
        gm_quote = "We trusted our evaluations even when they differed from consensus."
        rival_reaction = "Rival GMs think this class left better players available."
    else:
        fan_reaction = "Fans are frustrated — too many reaches and puzzling early gambles."
        media_reaction = "Media pans it: value ignored, needs unaddressed, questions everywhere."
        scout_reaction = "Scouts see boom-or-bust bets with little dependable floor."
        gm_quote = "We swung for upside and we'll live with the risk."
        rival_reaction = "Rival GMs are thrilled to see picks used this way."

    if star_rating >= 4:
        three_year_outlook = "Multiple picks project as NHL regulars within three seasons."
    elif star_rating == 3:
        three_year_outlook = "One or two picks should push for NHL roles by year three."
    else:
        three_year_outlook = "Most picks are long-term projects needing patient development."

    grade_summary = _clip15(
        f"Grade {user_grade}: {steal_ct} steals, {reach_ct} reaches, {need_ct} needs filled."
    )

    next_steps_detail = [
        {"label": "Re-Sign Phase", "note": "Lock up your restricted and unrestricted free agents."},
        {"label": "Free Agency", "note": "Chase the market to fill remaining roster holes."},
        {"label": "Development Camp", "note": "Get these new prospects into your system."},
    ]

    return {
        "user_grade": user_grade,
        "user_grade_score": _clamp100(user_score),
        "grade_summary": grade_summary,
        "user_class_rank": user_class_rank,
        "user_star_rating": star_rating,
        "grade_breakdown": grade_breakdown,
        "user_pick_count": len(user_picks),
        "user_steal_count": steal_ct,
        "user_reach_count": reach_ct,
        "user_value_count": value_ct,
        "user_need_count": need_ct,
        "user_avg_value_delta": avg_value_delta,
        "best_user_pick": best_user_pick,
        "riskiest_user_pick": riskiest_user_pick,
        "highest_ceiling_pick": highest_ceiling_pick,
        "safest_user_pick": safest_user_pick,
        "first_round_user_pick": first_round_user_pick,
        "boom_bust_pick": boom_bust_pick,
        "top_need_addressed": top_need_addressed,
        "biggest_need_ignored": biggest_need_ignored,
        "needs_report": needs_report,
        "still_on_board": still_on_board,
        "pipeline_grades": pipeline_grades,
        "fan_reaction": fan_reaction,
        "media_reaction": media_reaction,
        "scout_reaction": scout_reaction,
        "gm_quote": gm_quote,
        "rival_reaction": rival_reaction,
        "three_year_outlook": three_year_outlook,
        "next_steps_detail": next_steps_detail,
        "user_picks": user_picks,
        "user_draft_class": user_picks,
        "all_picks": results,
        "best_steals": steals,
        "biggest_steal": steals[0] if steals else None,
        "biggest_reaches": reaches,
        "biggest_reach": reaches[0] if reaches else None,
        "biggest_faller": biggest_faller,
        "surprising_picks": surprises,
        "weirdest_off_board": weirdest,
        "best_goalie_pick": best_goalie,
        "team_addressed_needs_best": _display_team(session, best_needs_team) if best_needs_team else None,
        "riskiest_draft_class_team": _display_team(session, riskiest_team) if riskiest_team else None,
        "safest_draft_class_team": _display_team(session, safest_team) if safest_team else None,
        "most_aggressive_team": _display_team(session, aggressive_team) if aggressive_team else None,
        "most_conservative_team": _display_team(session, conservative_team) if conservative_team else None,
        "best_value_round": best_value_round,
        "first_round_summary": first_round,
        "first_round_storylines": first_round_storylines,
        "goalie_summary": goalies,
        "position_breakdown": pos_breakdown,
        "top_undrafted": undrafted,
        "fall_reasons": fall_reasons[:6],
        "headlines": headlines,
        "user_position_breakdown": pos_breakdown,
        "user_draft_identity": user_identity,
        "round_recaps": dict(state.get("round_recaps") or {}),
        "draft_year": state.get("draft_year"),
        "class_strength": state.get("class_strength"),
        "total_picks": len(results),
        "next_steps": ["Re-sign phase", "Free agency", "Development camp"],
    }


def prepare_offseason_draft_payload(session: FranchiseSession) -> Dict[str, Any]:
    """Hydrate draft stage without auto-completing. Live Entry Draft source of truth — do not duplicate in SimEngine franchise mirror."""
    if not getattr(session, "draft_combine_done", False):
        raise ValueError("Draft Combine must be completed before the Entry Draft")
    state = getattr(session, "draft_state", None) or {}
    draft_year = int(session.season_calendar_year) + 1

    # A COMPLETED draft is not stale — re-hydrate and return the finished results.
    # Only a mismatched draft year (a genuinely new draft) resets state. This
    # prevents reopening the stage from wiping results and re-drafting prospects.
    if state.get("draft_started") and int(state.get("draft_year") or 0) == draft_year:
        payload = get_entry_draft_payload(session)
        session.draft_payload = payload
        session.draft_completed = bool(state.get("draft_completed"))
        return {"draft": payload}

    stale = state.get("draft_started") and int(state.get("draft_year") or 0) != draft_year
    if stale:
        session.draft_state = {}
        session.draft_completed = False
        state = {}
    if not state.get("draft_started"):
        if not session.draft_lottery_done:
            from services.franchise_offseason import _run_draft_lottery

            _run_draft_lottery(session)
        initialize_entry_draft(session)
    payload = get_entry_draft_payload(session)
    session.draft_payload = payload
    session.draft_completed = bool((getattr(session, "draft_state", None) or {}).get("draft_completed"))
    return {"draft": payload}