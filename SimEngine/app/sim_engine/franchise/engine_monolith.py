"""
Interactive franchise day engine: advances the league calendar one day at a time
by reusing SimEngine's season simulation loop (per-game) without modifying SimEngine.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import os
import random
import time
import uuid
from dataclasses import is_dataclass, replace
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.franchise.paths import ensure_simengine_path

ensure_simengine_path()
import run_sim as rs  # noqa: E402

_startup_log = logging.getLogger("uvicorn.error")


def _franchise_startup_stage(msg: str) -> None:
    """Always-on lightweight startup tracing (see post /api/franchise/start)."""
    _startup_log.info("[franchise start] %s", msg)

from app.sim_engine.entities.coach import CoachRole, generate_coach  # noqa: E402
from app.sim_engine.entities.player import (  # noqa: E402
    DEFENSE_ATTRS,
    DEV_ATTRS,
    GOALIE_ATTRS,
    IQ_ATTRS,
    OFFENSE_ATTRS,
    PERSONALITY_ATTRS,
    PHYSICAL_ATTRS,
    PLAYMAKING_ATTRS,
    SKILL_ATTRS,
    SKATING_ATTRS,
    SPECIAL_ATTRS,
    height_cm_to_imperial,
)
from app.sim_engine.league import (
    compute_awards,
    generate_regular_season_schedule,
    simulate_playoffs,
)
from app.sim_engine.league.schedule_generator import GameSlot, _safe_team_id, _safe_id_str, _safe_slot_team_id
from app.sim_engine.league.standings import StandingsTable

from app.sim_engine.franchise.session import FranchiseSession
from app.sim_engine.franchise.calendar import (
    build_season_calendar,
    calendar_day_to_dict,
    last_regular_season_index,
    map_abstract_schedule_to_calendar,
    season_anchor_event_markers,
)

# Canonical NHL names (lowercase) -> 3-letter code. Avoids "Stars" -> STA when team_id is numeric.
_NHL_DISPLAY_LOWER_TO_ABBR: Dict[str, str] = {
    "anaheim ducks": "ANA",
    "boston bruins": "BOS",
    "buffalo sabres": "BUF",
    "calgary flames": "CGY",
    "carolina hurricanes": "CAR",
    "chicago blackhawks": "CHI",
    "colorado avalanche": "COL",
    "columbus blue jackets": "CBJ",
    "dallas stars": "DAL",
    "detroit red wings": "DET",
    "edmonton oilers": "EDM",
    "florida panthers": "FLA",
    "los angeles kings": "LAK",
    "minnesota wild": "MIN",
    "montreal canadiens": "MTL",
    "nashville predators": "NSH",
    "new jersey devils": "NJD",
    "new york islanders": "NYI",
    "new york rangers": "NYR",
    "ottawa senators": "OTT",
    "philadelphia flyers": "PHI",
    "pittsburgh penguins": "PIT",
    "seattle kraken": "SEA",
    "san jose sharks": "SJS",
    "st. louis blues": "STL",
    "tampa bay lightning": "TBL",
    "toronto maple leafs": "TOR",
    "utah hockey club": "UTA",
    "vancouver canucks": "VAN",
    "vegas golden knights": "VGK",
    "washington capitals": "WSH",
    "winnipeg jets": "WPG",
}


def _fr_dbg_enabled() -> bool:
    return os.environ.get("NHL_FRANCHISE_DEBUG", "0") == "1"


def _fr_dbg(msg: str) -> None:
    if _fr_dbg_enabled():
        print(f"[franchise debug] {msg}")


def _sync_nhl_calendar_bounds(session: FranchiseSession) -> None:
    """Recompute last preseason+regular calendar index from stored rows (guards stale default 0)."""
    cal = getattr(session, "nhl_calendar", None) or []
    if not cal:
        return
    last = 0
    for i, row in enumerate(cal):
        seg = str(row.get("segment") or "")
        if seg in ("preseason", "regular"):
            last = i
    session.nhl_regular_season_last_index = int(last)


def _calendar_iso_for_day(session: FranchiseSession, day_idx: int) -> str:
    """Best-effort ISO date lookup for a franchise calendar index."""
    cal = getattr(session, "nhl_calendar", None) or []
    try:
        idx = int(day_idx)
    except Exception:
        idx = 0

    if 0 <= idx < len(cal):
        row = cal[idx] or {}
        iso = row.get("iso") or row.get("date") or row.get("calendar_iso")
        if iso:
            return str(iso)

    return ""

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


def _team_plays_on_day(by_day: Dict[int, List[Any]], day_idx: int, team_id: str) -> bool:
    tid = _safe_id_str(team_id)
    for sl in by_day.get(int(day_idx), []) or []:
        if _safe_slot_team_id(sl, "home_id") == tid or _safe_slot_team_id(sl, "away_id") == tid:
            return True
    return False


def _can_place_user_game(
    by_day: Dict[int, List[Any]],
    day_idx: int,
    user_id: str,
    opp_id: str,
    nhl_cal: List[Dict[str, Any]],
) -> bool:
    """
    Legacy/user-game placement gate.

    This now mirrors the strict schedule rules:
    - only preseason/regular
    - allowed game date
    - no double-booking
    - no 4-in-4 / 5-in-7 created for either team
    """
    idx = int(day_idx)

    if idx < 0 or idx >= len(nhl_cal):
        return False

    row = nhl_cal[idx] or {}
    seg = str(row.get("segment") or row.get("season_segment") or "")

    if seg not in ("preseason", "regular"):
        return False

    ag = row.get("allows_games")
    if ag is None:
        ag = row.get("allowsGames")

    if ag is False:
        return False

    if _team_plays_on_day(by_day, idx, user_id):
        return False

    if _team_plays_on_day(by_day, idx, opp_id):
        return False

    # Build a temporary fake slot so the same cadence validator is used.
    probe = GameSlot(
        day=idx,
        home_id=_safe_id_str(user_id),
        away_id=_safe_id_str(opp_id),
        is_playoff=False,
    )

    if _would_create_bad_cadence_for_slot(
        probe,
        idx,
        by_day,
        old_day=None,
        max_games_in_4=3,
        max_games_in_7=4,
    ):
        return False

    return True


def _slot_key(slot: Any) -> Tuple[str, str, bool]:
    return (
        _safe_slot_team_id(slot, "home_id"),
        _safe_slot_team_id(slot, "away_id"),
        bool(getattr(slot, "is_playoff", False)),
    )


def _team_ids_for_slot(slot: Any) -> Tuple[str, str]:
    return (_safe_slot_team_id(slot, "home_id"), _safe_slot_team_id(slot, "away_id"))


def _slot_has_team(slot: Any, team_id: str) -> bool:
    tid = _safe_id_str(team_id)
    h, a = _team_ids_for_slot(slot)
    return tid == h or tid == a


def _regular_game_indices(nhl_cal: List[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    for i, day in enumerate(nhl_cal or []):
        seg = str(day.get("segment") or day.get("season_segment") or "")
        if seg != "regular":
            continue
        ag = day.get("allows_games")
        if ag is None:
            ag = day.get("allowsGames")
        if ag is False:
            continue
        out.append(i)
    return out


def _slot_is_regular_season(slot: Any, nhl_cal: List[Dict[str, Any]], day_idx: int) -> bool:
    if bool(getattr(slot, "is_playoff", False)):
        return False
    idx = int(day_idx)
    if idx < 0 or idx >= len(nhl_cal):
        return False
    row = nhl_cal[idx]
    seg = str(row.get("segment") or row.get("season_segment") or "")
    return seg == "regular"


def _build_team_game_days(
    by_day: Dict[int, List[Any]],
    *,
    nhl_cal: Optional[List[Dict[str, Any]]] = None,
    regular_only: bool = False,
) -> Dict[str, List[int]]:
    acc: Dict[str, List[int]] = defaultdict(list)
    for d, slots in (by_day or {}).items():
        di = int(d)
        for sl in slots or []:
            if regular_only and nhl_cal is not None:
                if not _slot_is_regular_season(sl, nhl_cal, di):
                    continue
            h, a = _team_ids_for_slot(sl)
            if h:
                acc[h].append(di)
            if a:
                acc[a].append(di)
    return {tid: sorted(set(ds)) for tid, ds in acc.items()}


def _team_schedule_penalty(days: List[int]) -> float:
    """Deterministic cadence penalty for one team's sorted unique regular-season game days."""
    if not days:
        return 0.0
    if len(days) != len(set(days)):
        return 1e12
    ds = sorted(days)
    ds_set = set(ds)
    pen = 0.0
    lo, hi = ds[0], ds[-1]

    for start in range(lo, hi - 1):
        if all((start + k) in ds_set for k in range(3)):
            pen += 7500.0
    for start in range(lo, hi - 2):
        if all((start + k) in ds_set for k in range(4)):
            pen += 60000.0

    for i in range(len(ds) - 1):
        gap = ds[i + 1] - ds[i]
        if gap == 1:
            pen += 28.0
        elif gap == 2:
            pen += 5.0
        elif gap == 3:
            pass
        elif gap == 4:
            pen += 15.0
        elif gap >= 5:
            pen += 95.0 + float(gap - 5) * 38.0

    for w in range(lo, hi - 5):
        inc = sum(1 for x in ds if w <= x <= w + 6)
        if inc > 4:
            pen += float(inc - 4) * 3200.0
        elif inc < 2:
            if w >= ds[0] + 7 and w + 6 <= ds[-1] - 7:
                pen += 650.0 * float(2 - inc)

    return pen

def _team_has_impossible_cadence(days: List[int]) -> Optional[str]:
    """
    Hard NHL-style cadence validation for one team's regular-season game days.
    This is stricter than the soft penalty model.
    """
    if not days:
        return None

    ds = sorted(int(x) for x in days)

    if len(ds) != len(set(ds)):
        return "duplicate game day"

    lo, hi = ds[0], ds[-1]

    # HARD: no 4 games in 4 nights.
    for start in range(lo, hi + 1):
        games_4 = sum(1 for d in ds if start <= d <= start + 3)
        if games_4 >= 4:
            return f"4 games in 4 days starting day {start}"

    # HARD: no 5 games in 7 nights.
    for start in range(lo, hi + 1):
        games_7 = sum(1 for d in ds if start <= d <= start + 6)
        if games_7 >= 5:
            return f"5 games in 7 days starting day {start}"

    return None


def _validate_league_cadence_hard(
    by_day: Dict[int, List[Any]],
    nhl_cal: List[Dict[str, Any]],
) -> List[str]:
    """
    League-wide hard cadence validator.
    Separate from slot integrity: this catches the stuff that makes the schedule feel fake.
    """
    errors: List[str] = []

    team_days = _build_team_game_days(
        by_day or {},
        nhl_cal=nhl_cal,
        regular_only=True,
    )

    for team_id, days in sorted(team_days.items()):
        bad = _team_has_impossible_cadence(days)
        if bad:
            errors.append(f"Team {team_id}: {bad}.")

    return errors


def _get_regular_row_allows_games(nhl_cal: List[Dict[str, Any]], day_idx: int) -> bool:
    """
    Defensive calendar read. Some rows use allows_games, some allowsGames.
    """
    idx = int(day_idx)

    if idx < 0 or idx >= len(nhl_cal):
        return False

    row = nhl_cal[idx] or {}
    seg = str(row.get("segment") or row.get("season_segment") or "")

    if seg != "regular":
        return False

    allows = row.get("allows_games")
    if allows is None:
        allows = row.get("allowsGames")

    return allows is not False


def _slot_would_be_legal_after_move(
    slot: Any,
    target_day: int,
    by_day: Dict[int, List[Any]],
    nhl_cal: List[Dict[str, Any]],
    *,
    old_day: int,
    max_games_per_day: int = 18,
) -> bool:
    """
    One strict legality gate used by BOTH smoothing and conflict repair.
    This prevents the repair pass from creating the exact schedule nonsense
    the smoother was trying to remove.
    """
    target_day = int(target_day)
    old_day = int(old_day)

    if target_day == old_day:
        return False

    if not _get_regular_row_allows_games(nhl_cal, target_day):
        return False

    slots_on_target = list(by_day.get(target_day, []) or [])

    if len(slots_on_target) >= int(max_games_per_day):
        return False

    home_id, away_id = _team_ids_for_slot(slot)

    if not home_id or not away_id:
        return False

    if home_id == away_id:
        return False

    if _team_plays_on_day(by_day, target_day, home_id):
        return False

    if _team_plays_on_day(by_day, target_day, away_id):
        return False

    if _would_create_bad_cadence_for_slot(
        slot,
        target_day,
        by_day,
        old_day=old_day,
        max_games_in_4=3,
        max_games_in_7=4,
    ):
        return False

    return True
def _league_schedule_penalty(by_day: Dict[int, List[Any]], nhl_cal: List[Dict[str, Any]]) -> float:
    raw: Dict[str, List[int]] = defaultdict(list)
    for d, slots in (by_day or {}).items():
        di = int(d)
        for sl in slots or []:
            if not _slot_is_regular_season(sl, nhl_cal, di):
                continue
            for tid in _team_ids_for_slot(sl):
                if tid != "":
                    raw[tid].append(di)
    total = 0.0
    for _tid, days in raw.items():
        if len(days) != len(set(days)):
            total += 1e12
        total += _team_schedule_penalty(sorted(set(days)))
    return total


def _league_penalty_from_team_days(team_days: Dict[str, List[int]]) -> float:
    """League penalty as sum of per-team penalties (no cross-team terms)."""
    total = 0.0
    for ds in team_days.values():
        if len(ds) != len(set(ds)):
            return 1e12
        total += _team_schedule_penalty(sorted(set(ds)))
    return total


def _pair_penalty_after_day_move(
    team_days: Dict[str, List[int]],
    t_home: str,
    t_away: str,
    old_d: int,
    new_d: int,
) -> float:
    """Penalty contribution for the two teams if a shared game moves old_d -> new_d."""
    acc = 0.0
    for tid in sorted({str(t_home or ""), str(t_away or "")}):
        if tid == "":
            continue
        base = sorted(set(team_days.get(tid, [])))
        if int(old_d) not in base:
            return 1e12
        nxt = sorted((set(base) - {int(old_d)}) | {int(new_d)})
        if len(nxt) != len(set(nxt)):
            acc += 1e12
        acc += _team_schedule_penalty(nxt)
    return acc


def _can_place_slot_on_day(
    slot: Any,
    target_day: int,
    by_day: Dict[int, List[Any]],
    *,
    old_day: int,
    eligible_set: set,
    max_games_per_day: int,
    nhl_cal: List[Dict[str, Any]],
) -> bool:
    target_day = int(target_day)
    old_day = int(old_day)

    if target_day == old_day:
        return False
    if target_day not in eligible_set:
        return False
    if target_day < 0 or target_day >= len(nhl_cal):
        return False

    row = nhl_cal[target_day] or {}
    seg = str(row.get("segment") or row.get("season_segment") or "")
    if seg != "regular":
        return False

    allows_games = row.get("allows_games")
    if allows_games is None:
        allows_games = row.get("allowsGames")
    if allows_games is False:
        return False

    if len(by_day.get(target_day, []) or []) >= int(max_games_per_day):
        return False

    t1, t2 = _team_ids_for_slot(slot)

    if not t1 or not t2:
        return False
    if t1 == t2:
        return False

    if _team_plays_on_day(by_day, target_day, t1):
        return False
    if _team_plays_on_day(by_day, target_day, t2):
        return False

    # HARD cadence rules: do not allow smoothing to create unrealistic stretches.
    if _would_create_bad_cadence_for_slot(
        slot,
        target_day,
        by_day,
        old_day=old_day,
        max_games_in_4=3,
        max_games_in_7=4,
    ):
        return False

    return True

def _clone_slot_at_day(slot: Any, new_day: int) -> Any:
    nd = int(new_day)
    if is_dataclass(slot) and not isinstance(slot, type):
        try:
            return replace(slot, day=nd)
        except TypeError:
            pass
    return GameSlot(
        day=nd,
        home_id=_safe_slot_team_id(slot, "home_id"),
        away_id=_safe_slot_team_id(slot, "away_id"),
        is_playoff=bool(getattr(slot, "is_playoff", False)),
    )


def _find_slot_on_day(by_day: Dict[int, List[Any]], day: int, key: Tuple[str, str, bool]) -> Optional[Any]:
    for sl in by_day.get(int(day), []) or []:
        if _slot_key(sl) == key:
            return sl
    return None


def _move_slot(by_day: Dict[int, List[Any]], slot: Any, old_day: int, new_day: int) -> None:
    """Move one slot between day buckets and keep slot.day in sync."""
    old_day = int(old_day)
    new_day = int(new_day)
    src = by_day.get(old_day, []) or []
    key = _slot_key(slot)
    pick_idx = -1
    for i, sl in enumerate(src):
        if _slot_key(sl) == key:
            pick_idx = i
            break
    if pick_idx < 0:
        return
    removed = src.pop(pick_idx)
    if not src:
        by_day.pop(old_day, None)
    else:
        by_day[old_day] = src
    moved = _clone_slot_at_day(removed, new_day)
    by_day.setdefault(new_day, []).append(moved)


def _schedule_quality_summary(
    by_day: Dict[int, List[Any]],
    nhl_cal: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    td = _build_team_game_days(
        by_day or {},
        nhl_cal=nhl_cal,
        regular_only=bool(nhl_cal),
    )

    max4 = 0
    max7 = 0
    n4 = 0
    n5in7 = 0
    n3 = 0
    ng5 = 0
    worst_gap = 0
    worst_team = ""

    for tid, days in td.items():
        if not days:
            continue

        ds = sorted(days)
        ds_set = set(ds)
        lo, hi = ds[0], ds[-1]

        team_max4 = 0
        team_max7 = 0

        for start in range(lo, hi + 1):
            c4 = sum(1 for x in ds if start <= x <= start + 3)
            c7 = sum(1 for x in ds if start <= x <= start + 6)
            team_max4 = max(team_max4, c4)
            team_max7 = max(team_max7, c7)
            max4 = max(max4, c4)
            max7 = max(max7, c7)

        if team_max4 >= 4:
            n4 += 1

        if team_max7 >= 5:
            n5in7 += 1

        has3 = any(
            all((start + k) in ds_set for k in range(3))
            for start in range(lo, hi - 1)
        )

        if has3 and team_max4 < 4:
            n3 += 1

        for i in range(len(ds) - 1):
            gap = ds[i + 1] - ds[i]

            if gap >= 5:
                ng5 += 1

            if gap > worst_gap:
                worst_gap = gap
                worst_team = str(tid)

    lp = _league_schedule_penalty(by_day or {}, nhl_cal or []) if nhl_cal else 0.0

    return {
        "max_games_in_4_days": int(max4),
        "max_games_in_7_days": int(max7),
        "teams_with_4_in_4": int(n4),
        "teams_with_5_in_7": int(n5in7),
        "teams_with_3_in_3": int(n3),
        "teams_with_5_day_gaps": int(ng5),
        "worst_gap_days": int(worst_gap),
        "worst_gap_team": worst_team,
        "league_penalty": float(lp),
        "hard_cadence_ok": bool(n4 == 0 and n5in7 == 0),
    }


def _ensure_session_event_lists(session: FranchiseSession) -> None:
    if not hasattr(session, "calendar_events") or session.calendar_events is None:
        session.calendar_events = []
    if not hasattr(session, "pending_ui_popups") or session.pending_ui_popups is None:
        session.pending_ui_popups = []
    if not hasattr(session, "pending_decisions") or session.pending_decisions is None:
        session.pending_decisions = []
    if not hasattr(session, "notifications") or session.notifications is None:
        session.notifications = []
    if not hasattr(session, "timeline") or session.timeline is None:
        session.timeline = []


def _append_unique_dict_event(rows: List[Dict[str, Any]], event: Dict[str, Any]) -> None:
    eid = str(event.get("id") or "").strip()
    if eid and any(isinstance(x, dict) and str(x.get("id") or "") == eid for x in rows):
        return
    rows.append(event)


def _normalized_notification(
    *,
    notification_id: str,
    notification_type: str,
    text: str,
    priority: str = "LOW",
    calendar_day: int = 0,
    calendar_iso: str = "",
    team_id: str = "",
    player_id: str = "",
    source: str = "franchise",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "id": str(notification_id),
        "type": str(notification_type or "system"),
        "text": str(text or ""),
        "priority": str(priority or "LOW").upper(),
        "calendar_day": int(calendar_day),
        "date": int(calendar_day),
        "calendar_iso": str(calendar_iso or ""),
        "team_id": _safe_id_str(team_id),
        "player_id": str(player_id or ""),
        "source": str(source or "franchise"),
    }
    if extra:
        row.update(extra)
    return row


def _normalized_timeline_event(
    *,
    event_id: str,
    event_type: str,
    text: str,
    calendar_day: int = 0,
    calendar_iso: str = "",
    team_id: str = "",
    player_id: str = "",
    priority: str = "LOW",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "id": str(event_id),
        "type": str(event_type or "system"),
        "text": str(text or ""),
        "headline": str(text or ""),
        "calendar_day": int(calendar_day),
        "date": int(calendar_day),
        "calendar_iso": str(calendar_iso or ""),
        "team_id": _safe_id_str(team_id),
        "player_id": str(player_id or ""),
        "priority": str(priority or "LOW").upper(),
    }
    if extra:
        row.update(extra)
    return row


def _would_create_bad_cadence_for_slot(
    slot: Any,
    target_day: int,
    by_day: Dict[int, List[Any]],
    *,
    old_day: Optional[int],
    max_games_in_4: int = 3,
    max_games_in_7: int = 4,
) -> bool:
    """
    Hard rejection used by the schedule smoother.
    Prevents the move from creating 4-in-4 or 5-in-7 type nonsense.
    """
    target_day = int(target_day)
    team_ids = [x for x in _team_ids_for_slot(slot) if x]

    for tid in team_ids:
        days = []
        for d, slots in (by_day or {}).items():
            di = int(d)
            for sl in slots or []:
                if old_day is not None and di == int(old_day) and _slot_key(sl) == _slot_key(slot):
                    continue
                if _slot_has_team(sl, tid):
                    days.append(di)

        days.append(target_day)
        days = sorted(set(days))

        for start in range(target_day - 3, target_day + 1):
            games_4 = sum(1 for d in days if start <= d <= start + 3)
            if games_4 > int(max_games_in_4):
                return True

        for start in range(target_day - 6, target_day + 1):
            games_7 = sum(1 for d in days if start <= d <= start + 6)
            if games_7 > int(max_games_in_7):
                return True

    return False


def _finalize_schedule_after_generation(
    by_day: Dict[int, List[Any]],
    nhl_cal: List[Dict[str, Any]],
    *,
    user_id: Optional[str] = None,
) -> Tuple[Dict[int, List[Any]], List[Any], Dict[str, Any]]:
    """
    Startup-only schedule repair/finalizer.

    This is now much stricter:
    - smoother runs harder
    - conflict repair obeys cadence
    - validation includes hard NHL cadence rules
    - schedule diagnostics tell you exactly whether startup schedule is trustworthy
    """
    started_at = time.perf_counter()
    before = _schedule_quality_summary(dict(by_day), nhl_cal)

    raw_by_day: Dict[int, List[Any]] = {
        int(k): list(v or [])
        for k, v in (by_day or {}).items()
    }

    smooth_error: Optional[str] = None
    repair_error: Optional[str] = None
    second_smooth_error: Optional[str] = None

    try:
        _franchise_startup_stage("schedule finalize: smoothing pass 1")
        fixed = _smooth_league_schedule(
            dict(raw_by_day),
            nhl_cal=nhl_cal,
            user_id=user_id,
            max_passes=1,
            max_games_per_day=18,
            max_teams_per_round=10,
            max_inner_steps=10,
            max_suspicious_pairs=12,
        )
        _franchise_startup_stage("schedule finalize: smoothing pass 1 complete")
    except Exception as e:
        _startup_log.exception(
            "[franchise start] _smooth_league_schedule failed; using mapped schedule."
        )
        fixed = dict(raw_by_day)
        smooth_error = str(e)

    try:
        _franchise_startup_stage("schedule finalize: conflict repair")
        fixed = _repair_regular_day_conflicts(fixed, nhl_cal)
        _franchise_startup_stage("schedule finalize: conflict repair complete")
    except Exception as e:
        _startup_log.exception(
            "[franchise start] _repair_regular_day_conflicts failed; continuing with pre-repair slate."
        )
        repair_error = str(e)

    # Skip the second expensive smoothing pass at startup.
    # The first pass + strict repair + hard validation is enough and avoids long UI stalls.

    schedule: List[Any] = []
    aligned: Dict[int, List[Any]] = {}

    for d in sorted(fixed.keys()):
        di = int(d)
        row: List[Any] = []

        for sl in fixed[di]:
            if int(getattr(sl, "day", -1) or -1) != di:
                sl = _clone_slot_at_day(sl, di)

            row.append(sl)
            schedule.append(sl)

        aligned[di] = row

    hard_errors = _validate_schedule_hard(aligned, nhl_cal)
    after = _schedule_quality_summary(aligned, nhl_cal)
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    startup_validation_ok = len(hard_errors) == 0

    diagnostics: Dict[str, Any] = {
        "before_smooth": before,
        "after_smooth": after,
        "hard_errors": hard_errors,
        "startup_validation_ok": startup_validation_ok,
        "locked_at_startup": True,
        "smooth_error": smooth_error,
        "repair_error": repair_error,
        "second_smooth_error": second_smooth_error,
        "finalize_elapsed_ms": elapsed_ms,
        "strict_schedule_rules": {
            "max_games_in_4_days": 3,
            "max_games_in_7_days": 4,
            "max_games_per_day": 18,
            "regular_only_conflict_repair": True,
            "blocked_dates_respected": True,
        },
    }

    if hard_errors:
        _startup_log.warning(
            "[franchise start] schedule hard-validation reported %s issue(s); first: %s",
            len(hard_errors),
            hard_errors[0],
        )
    else:
        _startup_log.info(
            "[franchise start] schedule hard-validation passed. Quality=%s elapsed_ms=%s",
            after,
            elapsed_ms,
        )

    return aligned, schedule, diagnostics

def _validate_schedule_hard(
    by_day: Dict[int, List[Any]],
    nhl_cal: List[Dict[str, Any]],
    *,
    day_filter: Optional[int] = None,
) -> List[str]:
    """
    Hard schedule integrity validator used at startup and before daily simulation.

    This now checks:
    - games outside allowed calendar segments
    - blocked dates
    - slot.day mismatch
    - missing teams
    - self-play
    - same-day double-booking
    - duplicate exact matchup on same day
    - impossible NHL cadence like 4-in-4 and 5-in-7
    """
    errs: List[str] = []

    if day_filter is not None:
        days = [int(day_filter)]
    else:
        days = sorted(int(d) for d in (by_day or {}).keys())

    seen_matchups: set = set()

    for d in days:
        slots = list((by_day or {}).get(d, []) or [])

        if not slots:
            continue

        row = nhl_cal[d] if 0 <= d < len(nhl_cal) else {}
        seg = str(row.get("segment") or row.get("season_segment") or "")

        allows = row.get("allows_games")
        if allows is None:
            allows = row.get("allowsGames")

        in_game_segment = seg in ("preseason", "regular")

        if not in_game_segment:
            errs.append(
                f"Day {d}: games scheduled outside preseason/regular segment ({seg or 'unknown'})."
            )

        if allows is False:
            errs.append(f"Day {d}: games scheduled on a blocked calendar date.")

        strict_team_uniqueness = seg == "regular"
        team_seen: set = set()

        for sl in slots:
            hid = _safe_slot_team_id(sl, "home_id")
            aid = _safe_slot_team_id(sl, "away_id")
            sday = int(getattr(sl, "day", d) or d)

            if sday != d:
                errs.append(f"Day {d}: slot day mismatch ({sday}).")

            if not hid or not aid:
                errs.append(f"Day {d}: slot missing team id(s).")
                continue

            if hid == aid:
                errs.append(f"Day {d}: self-play slot ({hid}).")

            if strict_team_uniqueness and (hid in team_seen or aid in team_seen):
                errs.append(f"Day {d}: team double-booked ({hid} vs {aid}).")

            team_seen.add(hid)
            team_seen.add(aid)

            key = (d, min(hid, aid), max(hid, aid))

            if strict_team_uniqueness and key in seen_matchups:
                errs.append(f"Day {d}: duplicate matchup detected ({hid} vs {aid}).")

            seen_matchups.add(key)

    # Only run full cadence scan when validating the whole schedule.
    # If day_filter is supplied, we are doing a quick daily slot check.
    if day_filter is None:
        errs.extend(_validate_league_cadence_hard(by_day, nhl_cal))

    return errs


def _repair_regular_day_conflicts(
    by_day: Dict[int, List[Any]],
    nhl_cal: List[Dict[str, Any]],
) -> Dict[int, List[Any]]:
    """
    Resolve regular-season same-day double-bookings by moving extra slots
    to the nearest legal NHL-style date.

    Important:
    - Does not move games onto blocked dates.
    - Does not move games outside regular season.
    - Does not create 4-in-4.
    - Does not create 5-in-7.
    - Does not double-book either team.
    - Prefers nearby dates, not random future dumping.
    """
    if not by_day:
        return by_day

    regular_days = _regular_game_indices(nhl_cal)

    if not regular_days:
        return by_day

    regular_days = sorted(int(x) for x in regular_days)
    moved_total = 0
    failed_moves: List[str] = []

    def _candidate_repair_days(old_day: int) -> List[int]:
        """
        Prefer close dates around the original slot, alternating future/past.
        This keeps the schedule stable and stops everything from being dumped forward.
        """
        old_day = int(old_day)
        out: List[int] = []
        seen: set = set()

        for radius in range(1, 15):
            for cand in (old_day + radius, old_day - radius):
                if cand in seen:
                    continue

                seen.add(cand)

                if cand in regular_days:
                    out.append(cand)

        # If near search fails, use the full regular list by distance.
        if not out:
            out = sorted(
                regular_days,
                key=lambda d: (abs(int(d) - old_day), int(d)),
            )

        return out

    def _pick_target(old_day: int, slot: Any) -> Optional[int]:
        for cand in _candidate_repair_days(old_day):
            if _slot_would_be_legal_after_move(
                slot,
                cand,
                by_day,
                nhl_cal,
                old_day=old_day,
                max_games_per_day=18,
            ):
                return int(cand)

        return None

    for d in sorted(int(x) for x in list(by_day.keys())):
        if d not in regular_days:
            continue

        slots = list(by_day.get(d, []) or [])

        if len(slots) <= 1:
            continue

        team_counts: Dict[str, int] = defaultdict(int)

        for sl in slots:
            h = _safe_slot_team_id(sl, "home_id")
            a = _safe_slot_team_id(sl, "away_id")

            if h:
                team_counts[h] += 1

            if a:
                team_counts[a] += 1

        conflict_slots: List[Any] = []

        for sl in slots:
            h = _safe_slot_team_id(sl, "home_id")
            a = _safe_slot_team_id(sl, "away_id")

            if team_counts.get(h, 0) > 1 or team_counts.get(a, 0) > 1:
                conflict_slots.append(sl)

        for sl in conflict_slots:
            cur_slots = list(by_day.get(d, []) or [])
            h = _safe_slot_team_id(sl, "home_id")
            a = _safe_slot_team_id(sl, "away_id")

            # Recompute after prior moves.
            c_h = sum(1 for x in cur_slots if _slot_has_team(x, h))
            c_a = sum(1 for x in cur_slots if _slot_has_team(x, a))

            if c_h <= 1 and c_a <= 1:
                continue

            target = _pick_target(d, sl)

            if target is None:
                failed_moves.append(f"Day {d}: could not legally move {h} vs {a}.")
                continue

            moved = _clone_slot_at_day(sl, int(target))

            removed = False
            next_src: List[Any] = []

            for existing in cur_slots:
                if not removed and _slot_key(existing) == _slot_key(sl):
                    removed = True
                    continue

                next_src.append(existing)

            if not removed:
                failed_moves.append(f"Day {d}: failed to remove source slot {h} vs {a}.")
                continue

            if next_src:
                by_day[d] = next_src
            else:
                by_day.pop(d, None)

            by_day.setdefault(int(target), []).append(moved)
            moved_total += 1

    if moved_total:
        _fr_dbg(f"schedule conflict repair moved {moved_total} regular-season slots")

    if failed_moves:
        _fr_dbg("schedule conflict repair unresolved: " + " | ".join(failed_moves[:8]))

    return by_day
def _schedule_suspicious_pairs(
    days: List[int],
    by_day: Dict[int, List[Any]],
    team_id: str,
    nhl_cal: List[Dict[str, Any]],
    *,
    max_pairs: int,
) -> List[Tuple[int, Any]]:
    if not days:
        return []
    ds = sorted(days)
    ds_set = set(ds)
    flagged: set = set()
    lo, hi = ds[0], ds[-1]
    for start in range(lo, hi - 2):
        if all((start + k) in ds_set for k in range(4)):
            for x in (start, start + 1, start + 2, start + 3):
                flagged.add(x)
    for start in range(lo, hi - 1):
        if all((start + k) in ds_set for k in range(3)):
            flagged.update([start, start + 1, start + 2])
    for i in range(len(ds) - 1):
        if ds[i + 1] - ds[i] >= 5:
            flagged.add(ds[i])
            flagged.add(ds[i + 1])
    for w in range(lo, hi - 5):
        inc = sum(1 for x in ds if w <= x <= w + 6)
        if inc >= 5:
            for x in ds:
                if w <= x <= w + 6:
                    flagged.add(x)
    pairs: List[Tuple[int, Any]] = []
    for d in sorted(flagged):
        slots = list(by_day.get(d, []) or [])
        slots.sort(key=lambda s: _slot_key(s))
        for sl in slots:
            if _slot_is_regular_season(sl, nhl_cal, d) and _slot_has_team(sl, team_id):
                pairs.append((d, sl))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _candidate_eligible_calendar_days(
    old_day: int,
    eligible_list: List[int],
    *,
    max_index_radius: int,
) -> List[int]:
    """Neighbors along the ordered list of eligible regular game nights (skips league off-days)."""
    if not eligible_list:
        return []
    old_day = int(old_day)
    pos = bisect.bisect_left(eligible_list, old_day)
    if pos < len(eligible_list) and eligible_list[pos] == old_day:
        center = pos
    elif pos > 0 and eligible_list[pos - 1] == old_day:
        center = pos - 1
    else:
        center = min(max(pos - 1, 0), len(eligible_list) - 1)
    seen: set = set()
    out: List[int] = []
    for w in range(1, int(max_index_radius) + 1):
        for j in (center - w, center + w):
            if 0 <= j < len(eligible_list):
                c = int(eligible_list[j])
                if c != old_day and c not in seen:
                    seen.add(c)
                    out.append(c)
    return out


def _smooth_league_schedule(
    by_day: Dict[int, List[Any]],
    *,
    nhl_cal: List[Dict[str, Any]],
    user_id: Optional[str] = None,
    max_passes: int = 2,
    max_games_per_day: int = 22,
    max_teams_per_round: int = 8,
    max_inner_steps: int = 12,
    max_suspicious_pairs: int = 10,
) -> Dict[int, List[Any]]:
    _ = user_id
    if not nhl_cal:
        return {int(k): list(v or []) for k, v in (by_day or {}).items()}
    out: Dict[int, List[Any]] = {int(k): list(v or []) for k, v in (by_day or {}).items()}
    eligible_list = _regular_game_indices(nhl_cal)
    eligible_set = set(eligible_list)
    if not eligible_set:
        return out

    n_teams_round = max(4, min(int(max_teams_per_round), 32))
    n_inner = max(4, min(int(max_inner_steps), 60))
    n_sus = max(4, min(int(max_suspicious_pairs), 40))
    for _pass_i in range(max(1, int(max_passes))):
        improved_outer = False
        for _inner in range(n_inner):
            team_days = _build_team_game_days(out, nhl_cal=nhl_cal, regular_only=True)
            pen0 = _league_penalty_from_team_days(team_days)
            best_pen = pen0
            best_tuple: Optional[Tuple[int, int, Tuple[str, str, bool]]] = None
            team_order = sorted(team_days.keys(), key=lambda t: (-_team_schedule_penalty(team_days[t]), t))[
                :n_teams_round
            ]
            for tid in team_order:
                sus = _schedule_suspicious_pairs(
                    team_days.get(tid, []),
                    out,
                    tid,
                    nhl_cal,
                    max_pairs=n_sus,
                )
                for old_d, sl in sus:
                    key = _slot_key(sl)
                    near4 = _candidate_eligible_calendar_days(int(old_d), eligible_list, max_index_radius=3)
                    near4_set = set(near4)
                    wide8_only = [
                        c
                        for c in _candidate_eligible_calendar_days(int(old_d), eligible_list, max_index_radius=6)
                        if c not in near4_set
                    ]
                    t_home, t_away = _team_ids_for_slot(sl)
                    before_pair = 0.0
                    for tmid in sorted({str(t_home or ""), str(t_away or "")}):
                        if tmid:
                            before_pair += _team_schedule_penalty(sorted(set(team_days.get(tmid, []))))
                    for new_d in near4 + wide8_only:
                        if not _can_place_slot_on_day(
                            sl,
                            new_d,
                            out,
                            old_day=int(old_d),
                            eligible_set=eligible_set,
                            max_games_per_day=int(max_games_per_day),
                            nhl_cal=nhl_cal,
                        ):
                            continue
                        sl_here = _find_slot_on_day(out, int(old_d), key)
                        if sl_here is None:
                            continue
                        after_pair = _pair_penalty_after_day_move(
                            team_days, t_home, t_away, int(old_d), int(new_d)
                        )
                        new_pen = pen0 - before_pair + after_pair
                        if new_pen >= pen0 - 1e-9:
                            continue
                        if new_pen < best_pen - 1e-12:
                            best_pen = new_pen
                            best_tuple = (int(old_d), int(new_d), key)
                        elif abs(new_pen - best_pen) < 1e-12:
                            cand = (int(old_d), int(new_d), key)
                            if best_tuple is None or cand < (best_tuple[0], best_tuple[1], best_tuple[2]):
                                best_tuple = cand
            if best_tuple is None or best_pen >= pen0 - 1e-9:
                break
            o_d, n_d, key = best_tuple
            sl_apply = _find_slot_on_day(out, o_d, key)
            if sl_apply is None:
                break
            _move_slot(out, sl_apply, o_d, n_d)
            improved_outer = True
        if not improved_outer:
            break
    return out


def _smooth_user_team_schedule(by_day: Dict[int, List[Any]], *, user_id: str, nhl_cal: List[Dict[str, Any]]) -> Dict[int, List[Any]]:
    """Backward-compatible entry: smooth the full league (user + CPU) on regular-season nights."""
    if not nhl_cal:
        return {int(k): list(v or []) for k, v in (by_day or {}).items()}
    return _smooth_league_schedule(
        by_day,
        nhl_cal=nhl_cal,
        user_id=(user_id or None),
        max_passes=2,
        max_games_per_day=22,
        max_teams_per_round=8,
        max_inner_steps=12,
        max_suspicious_pairs=10,
    )


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _display_team(t: Any) -> str:
    city = str(getattr(t, "city", "") or "").strip()
    name = str(getattr(t, "name", "") or "").strip()
    if city and name:
        return f"{city} {name}"
    return rs._team_name(t)


def _franchise_team_abbrev(tm: Any) -> str:
    if tm is None:
        return "?"
    for attr in ("abbr", "code", "abbreviation", "short_name"):
        v = getattr(tm, attr, None)
        if v is not None and str(v).strip():
            s = "".join(c for c in str(v).upper() if c.isalnum())
            if 2 <= len(s) <= 5:
                return s[:3]
    disp_raw = _display_team(tm).strip()
    disp = disp_raw.lower()
    if disp in _NHL_DISPLAY_LOWER_TO_ABBR:
        return _NHL_DISPLAY_LOWER_TO_ABBR[disp]
    for full_name, abbr in _NHL_DISPLAY_LOWER_TO_ABBR.items():
        if full_name in disp or disp in full_name:
            return abbr
    city = str(getattr(tm, "city", "") or "").strip()
    name = str(getattr(tm, "name", "") or "").strip()
    if city and name:
        combo = f"{city} {name}".strip().lower()
        if combo in _NHL_DISPLAY_LOWER_TO_ABBR:
            return _NHL_DISPLAY_LOWER_TO_ABBR[combo]
    tid_v = getattr(tm, "team_id", None)
    if tid_v is None:
        tid_v = getattr(tm, "id", None)
    if tid_v is None:
        tid_v = rs._team_id(tm)
    tid = str(tid_v) if tid_v is not None else ""
    u = "".join(c for c in tid.upper() if c.isalnum())
    if len(u) >= 3:
        return u[:3]
    nm = name.upper() if name else ""
    u2 = "".join(c for c in nm if c.isalnum())
    if len(u2) >= 3:
        return u2[:3]
    return (tid[:3] if tid else "?").upper()


def _franchise_log_injury_and_ui(
    session: FranchiseSession,
    *,
    player_id: str,
    player_name: str,
    team_id: str,
    team_abbrev: str,
    tier: str,
    games: int,
    injury_type: str,
    calendar_day: int,
    calendar_iso: str = "",
    game_day_injury: bool = False,
) -> None:
    _ensure_session_event_lists(session)

    if getattr(session, "injury_log_all", None) is None:
        session.injury_log_all = []
    if getattr(session, "injury_log_major", None) is None:
        session.injury_log_major = []

    cur_date = int(calendar_day)
    pid = str(player_id or "")
    tid_cmp = _safe_id_str(team_id).strip()
    utid = str(getattr(session, "user_team_id", "") or "").strip()
    tier_l = str(tier or "").lower().strip()
    team_abbr = str(team_abbrev or "").strip()
    player_label = str(player_name or "Player").strip()

    if not calendar_iso:
        calendar_iso = _calendar_iso_for_day(session, cur_date)

    if any(
        str(i.get("player_id") or "") == pid and int(i.get("calendar_day", i.get("date", -1))) == cur_date
        for i in session.injury_log_all
        if isinstance(i, dict)
    ):
        return

    is_user_team = bool(tid_cmp and utid) and tid_cmp.lower() == utid.lower()
    is_major = tier_l == "major"
    is_moderate = tier_l == "moderate"

    base_id = f"injury:{cur_date}:{tid_cmp}:{pid}"

    injury_log_row = {
        "id": f"injlog:{cur_date}:{tid_cmp}:{pid}",
        "player_id": pid,
        "player_name": player_label,
        "team_id": tid_cmp,
        "team_abbrev": team_abbr,
        "team_abbr": team_abbr,
        "tier": str(tier or ""),
        "severity": str(tier or ""),
        "injury_type": str(injury_type or tier or ""),
        "games": int(games),
        "games_initial": int(games),
        "games_remaining": int(games),
        "games_remaining_at_log": int(games),
        "type": str(injury_type or tier or ""),
        "date": cur_date,
        "calendar_day": cur_date,
        "calendar_iso": str(calendar_iso or ""),
        "status": "INJURED",
    }

    session.injury_log_all.append(injury_log_row)

    if is_major:
        session.injury_log_major.append(
            {
                "id": f"injmajor:{cur_date}:{tid_cmp}:{pid}",
                "player": player_label,
                "player_name": player_label,
                "player_id": pid,
                "tier": tier,
                "games": int(games),
                "team_id": tid_cmp,
                "team_abbrev": team_abbr,
                "calendar_day": cur_date,
                "calendar_iso": str(calendar_iso or ""),
            }
        )

    priority = "HIGH" if is_major else "MEDIUM"
    summary = (
        f"{player_label} ({team_abbr or tid_cmp}) is expected to miss {int(games)} games "
        f"with {injury_type or tier}."
    )
    if game_day_injury:
        summary += " This occurred during today's scheduled game."

    calendar_event = {
        "id": base_id,
        "kind": "injury",
        "type": "injury_report",
        "calendar_day": cur_date,
        "date": cur_date,
        "calendar_iso": str(calendar_iso or ""),
        "title": "Injury Report",
        "headline": f"{player_label} injured",
        "summary": summary,
        "description": summary,
        "team_id": tid_cmp,
        "team_abbrev": team_abbr,
        "team_abbr": team_abbr,
        "player_id": pid,
        "player_name": player_label,
        "priority": priority,
        "tier": str(tier or ""),
        "severity": str(tier or ""),
        "injury_type": str(injury_type or tier or ""),
        "games": int(games),
        "games_remaining": int(games),
        "game_day_injury": bool(game_day_injury),
        "surfaces": ["calendar", "storylines", "notifications"] + (["popup"] if is_user_team or is_major else []),
        "effects": {
            "availability_games_delta": -int(games),
            "depth_stress_delta": 1 if int(games) >= 2 else 0,
        },
        "effect_summary": f"Projected absence: {int(games)} game(s).",
    }

    _append_unique_dict_event(session.calendar_events, calendar_event)

    session.notifications.append(
        _normalized_notification(
            notification_id=f"notif:{base_id}",
            notification_type="injury",
            text=f"{player_label} ({team_abbr or tid_cmp}) out {int(games)} games ({injury_type or tier}).",
            priority=priority,
            calendar_day=cur_date,
            calendar_iso=str(calendar_iso or ""),
            team_id=tid_cmp,
            player_id=pid,
            source="injury_engine",
            extra={"game_day_injury": bool(game_day_injury)},
        )
    )

    _record_storyline(
        session,
        {
            "id": f"story:{base_id}",
            "type": "injury",
            "kind": "injury",
            "headline": f"{player_label} sidelined",
            "details": f"{team_abbr or tid_cmp} lose {player_label} for {int(games)} games ({injury_type or tier})",
            "cause": f"{player_label} suffered a {injury_type or tier} injury.",
            "effects": {
                "availability_games_delta": -int(games),
                "depth_stress_delta": 1 if int(games) >= 2 else 0,
            },
            "effect_summary": f"Projected absence: {int(games)} game(s).",
            "team": tid_cmp,
            "team_id": tid_cmp,
            "team_abbrev": team_abbr,
            "player_id": pid,
            "player_name": player_label,
            "players": [player_label],
            "priority": priority,
            "date": cur_date,
            "calendar_day": cur_date,
            "calendar_iso": str(calendar_iso or ""),
            "surfaces": ["storylines", "calendar"],
        },
    )

    session.timeline.append(
        _normalized_timeline_event(
            event_id=f"timeline:{base_id}",
            event_type="injury",
            text=f"{team_abbr or tid_cmp}: {player_label} injured ({int(games)}g)",
            calendar_day=cur_date,
            calendar_iso=str(calendar_iso or ""),
            team_id=tid_cmp,
            player_id=pid,
            priority=priority,
            extra={"surfaces": ["timeline", "calendar"]},
        )
    )

    # User-team injuries always popup. Major league injuries also popup as league news.
    should_popup = is_user_team or is_major
    if should_popup:
        same_day_injury_popups = [
            p
            for p in session.pending_ui_popups
            if isinstance(p, dict)
            and str(p.get("kind") or "") == "injury"
            and int(p.get("calendar_day", p.get("date", -1))) == cur_date
        ]

        if len(same_day_injury_popups) < 4:
            requires_decision = bool(is_user_team and (is_moderate or is_major))
            popup = {
                **calendar_event,
                "id": base_id,
                "kind": "injury",
                "date": cur_date,
                "calendar_day": cur_date,
                "requires_decision": requires_decision,
                "decision_id": base_id if requires_decision else "",
                "popup_scope": "user_team" if is_user_team else "league_news",
                "choices": (
                    [
                        {"id": "call_up_player", "label": "Call Up Depth Player"},
                        {"id": "shuffle_lines", "label": "Shuffle Lines"},
                        {"id": "play_short_roster", "label": "Play Short Roster"},
                        {"id": "place_on_ir", "label": "Place On IR"},
                    ]
                    if requires_decision
                    else []
                ),
            }

            _append_unique_dict_event(session.pending_ui_popups, popup)

            if requires_decision:
                _append_unique_dict_event(
                    session.pending_decisions,
                    {
                        "id": base_id,
                        "kind": "injury_decision",
                        "type": "injury_decision",
                        "calendar_day": cur_date,
                        "date": cur_date,
                        "calendar_iso": str(calendar_iso or ""),
                        "team_id": tid_cmp,
                        "team_abbrev": team_abbr,
                        "player_id": pid,
                        "player_name": player_label,
                        "title": "Injury Decision Required",
                        "summary": summary,
                        "choices": popup["choices"],
                        "resolved": False,
                    },
                )
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
    games_per_team: int = 82,
    season_start_year: Optional[int] = None,
    injuries_enabled: bool = True,
) -> FranchiseSession:
    ensure_simengine_path()
    from app.sim_engine.engine import SimEngine

    _franchise_startup_stage("SimEngine import complete; constructing engine")
    master = seed if seed is not None else random.randrange(1, 10**9)
    sim = SimEngine(seed=master, debug=False)
    _franchise_startup_stage("SimEngine constructed")
    league = sim.league
    try:
        setattr(league, "_runner_sim_engine", sim)
    except Exception:
        pass

    teams = list(getattr(league, "teams", None) or [])
    if not teams:
        raise RuntimeError("League has no teams after initialization.")
    _franchise_startup_stage(f"team resolution: {len(teams)} clubs in league")

    user_team = resolve_user_team(teams, team_query)
    _tid = getattr(user_team, "team_id", None)
    if _tid is not None:
        uid = str(_tid)
    else:
        _oid = getattr(user_team, "id", None)
        uid = str(_oid) if _oid is not None else rs._team_id(user_team)
    sim.team = user_team
    _franchise_startup_stage(f"user team resolved -> {uid}")

    coach = generate_coach(sim.rng, f"HIRE_{uid}", CoachRole.HEAD_COACH)
    coach.name = (head_coach_name or "Head Coach").strip() or "Head Coach"
    apply_coach_archetype(coach, coach_archetype, sim.rng)
    user_team.coach = coach
    sim.coach = coach
    _franchise_startup_stage("head coach generated and assigned")

    gp = int(games_per_team)
    if gp < 4:
        gp = 4
    if gp > 82:
        gp = 82
    season_y = int(season_start_year) if season_start_year is not None else 2025
    _franchise_startup_stage(f"generating abstract schedule ({gp} GP template)")
    schedule_raw = generate_regular_season_schedule(sim.rng, teams, gp)
    _franchise_startup_stage(f"abstract schedule slots={len(schedule_raw)}")
    by_abs: Dict[int, List[Any]] = defaultdict(list)
    for slot in schedule_raw:
        by_abs[int(slot.day)].append(slot)
    abstract_keys = sorted(by_abs.keys())

    _franchise_startup_stage(f"building NHL season calendar year={season_y}")
    cal_objs = build_season_calendar(season_y)
    nhl_cal = [calendar_day_to_dict(c) for c in cal_objs]
    last_reg_idx = last_regular_season_index(cal_objs)
    _franchise_startup_stage(f"calendar rows={len(nhl_cal)} last_regular_idx={last_reg_idx}")
    _franchise_startup_stage("map_abstract_schedule_to_calendar")
    day_map = map_abstract_schedule_to_calendar(cal_objs, abstract_keys)
    _franchise_startup_stage("abstract days mapped to calendar indices")
    by_day: Dict[int, List[Any]] = defaultdict(list)
    schedule: List[Any] = []
    for old in abstract_keys:
        nid = int(day_map[int(old)])
        for slot in by_abs[old]:
            gs = GameSlot(day=nid, home_id=slot.home_id, away_id=slot.away_id, is_playoff=slot.is_playoff)
            by_day[nid].append(gs)
            schedule.append(gs)
    _sched_dbg = os.environ.get("NHL_FRANCHISE_SCHEDULE_DEBUG") == "1"

    _franchise_startup_stage("_finalize_schedule_after_generation (smooth + repair + validate)")
    by_day, schedule, schedule_diagnostics = _finalize_schedule_after_generation(
        by_day,
        nhl_cal,
        user_id=uid,
    )
    schedule_diagnostics["quality"] = _schedule_quality_summary(by_day, nhl_cal)
    if _sched_dbg:
        print("[franchise schedule] diagnostics", schedule_diagnostics)
    he = list(schedule_diagnostics.get("hard_errors") or [])
    if he:
        _fr_dbg(f"schedule hard-validation warning at startup: {he[0]}")

    if not schedule:
        raise RuntimeError(
            "Franchise startup failed: schedule is empty after calendar mapping and finalization."
        )

    days_sorted = sorted(by_day.keys())
    _franchise_startup_stage(
        f"schedule finalized game_dates={len(days_sorted)} total_slots={len(schedule)} "
        f"validation_ok={bool(schedule_diagnostics.get('startup_validation_ok'))}"
    )
    standings = StandingsTable(teams)
    team_by_id: Dict[str, Any] = {}
    team_ids: List[str] = []
    for idx, t in enumerate(teams):
        # Must match schedule_generator._safe_team_id: team_id may be 0 (falsy) but is valid.
        tid = _safe_team_id(t, idx)
        team_ids.append(tid)
        team_by_id[tid] = t

    sim._preseason_line_synergy_refresh(teams, sim.rng)
    strength_map = sim._build_strength_map(teams)
    use_world = _use_world_modules()
    play_days: Dict[str, Any] = {}
    if use_world and world_calendar is not None:
        play_days = world_calendar.build_team_play_days(schedule)

    _franchise_startup_stage("creating FranchiseSession")
    session = FranchiseSession(
        session_id=FranchiseSession.new_id(),
        sim=sim,
        user_team_id=uid,
        head_coach_name=coach.name,
        coach_archetype=coach_archetype,
        season_calendar_year=season_y,
        games_per_team_schedule=gp,
        calendar_days_finished=0,
        schedule=schedule,
        by_day=dict(by_day),
        days_sorted=days_sorted,
        nhl_calendar=nhl_cal,
        calendar_cursor=0,
        nhl_regular_season_last_index=last_reg_idx,
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
        injuries_enabled=bool(injuries_enabled),
        preseason_applied=True,
    )
    session.schedule_diagnostics = schedule_diagnostics
    session.notifications = getattr(session, "notifications", None) or []
    session.timeline = getattr(session, "timeline", None) or []
    session.pending_ui_popups = getattr(session, "pending_ui_popups", None) or []
    session.calendar_events = getattr(session, "calendar_events", None) or []
    session.pending_decisions = getattr(session, "pending_decisions", None) or []
    start_iso = _calendar_iso_for_day(session, 0)

    session.notifications.append(
        _normalized_notification(
            notification_id=f"system:franchise_ready:{uid}",
            notification_type="system",
            text=f"Franchise ready ΓÇö {_display_team(user_team)} ({uid}).",
            priority="LOW",
            calendar_day=0,
            calendar_iso=start_iso,
            team_id=uid,
        )
    )

    session.notifications.append(
        _normalized_notification(
            notification_id=f"system:coach_hired:{uid}:{season_y}",
            notification_type="system",
            text=(
                f"Hired {coach.name} ({coach_archetype}). NHL calendar {season_y}ΓÇô{season_y + 1} ┬╖ "
                f"{len(nhl_cal)} days ┬╖ {len(days_sorted)} game dates ┬╖ ~{gp} GP."
            ),
            priority="LOW",
            calendar_day=0,
            calendar_iso=start_iso,
            team_id=uid,
        )
    )

    session.timeline.append(
        _normalized_timeline_event(
            event_id=f"timeline:welcome:{uid}:{season_y}",
            event_type="system",
            text="Welcome to Franchise Mode. Advance the day to begin the regular season.",
            calendar_day=0,
            calendar_iso=start_iso,
            team_id=uid,
            priority="LOW",
        )
    )
    try:
        from app.sim_engine.league_hierarchy_bootstrap import bootstrap_full_league_hierarchy

        bootstrap_full_league_hierarchy(league, sim.rng)
        npl = len(getattr(league, "players", None) or [])
        session.notifications.append(
            f"League depth online ΓÇö NHL affiliates (AHL/ECHL), UFA pools, overseas, juniors (~{npl} player records)."
        )
    except Exception as e:
        session.notifications.append(f"League depth bootstrap skipped: {e}")
    try:
        snapshot_draft_rank_prev(session, sim)
    except Exception:
        pass
    _franchise_startup_stage("start_franchise complete; returning session")
    return session


def _name_str(p: Any) -> str:
    ident = getattr(p, "identity", None)
    return str(getattr(ident, "name", None) or "?")


def _pos_str(p: Any) -> str:
    ident = getattr(p, "identity", None)
    pos = getattr(ident, "position", None) if ident else None
    return str(getattr(pos, "value", pos) or "?")


def _player_cap_hit_millions(player: Any) -> float:
    for key in ("cap_hit_m", "contract_aav_m", "aav_m"):
        try:
            v = float(getattr(player, key, 0) or 0)
            if v > 0:
                return v
        except Exception:
            pass
    c = getattr(player, "contract", None)
    if c is not None:
        for key in ("cap_hit_m", "cap_hit", "aav_m", "aav", "salary_aav"):
            try:
                v = float(getattr(c, key, 0) or 0)
                if v <= 0:
                    continue
                # Convert dollars to millions when stored as raw salary.
                if key in ("salary_aav", "aav", "cap_hit") and v > 250:
                    return v / 1_000_000.0
                return v
            except Exception:
                pass
    return 0.0


def _team_cap_snapshot(team: Any, sim: Any) -> Dict[str, float]:
    econ = ((getattr(getattr(sim, "league", None), "get_league_context", lambda: {})() or {}).get("economics") or {})
    cap_raw = float(econ.get("salary_cap", 92_000_000.0) or 92_000_000.0)
    salary_cap_m = cap_raw / 1_000_000.0 if cap_raw > 250 else cap_raw
    payroll_m = 0.0
    for p in (getattr(team, "roster", None) or []):
        if getattr(p, "retired", False):
            continue
        payroll_m += _player_cap_hit_millions(p)
    if payroll_m <= 0.0:
        payroll_m = float(getattr(team, "total_cap_hit", 0) or 0)
        if payroll_m > 250:
            payroll_m /= 1_000_000.0
    cap_space_m = max(0.0, salary_cap_m - payroll_m)
    return {
        "salary_cap": round(float(salary_cap_m), 3),
        "cap_hit": round(float(payroll_m), 3),
        "cap_space": round(float(cap_space_m), 3),
    }


def _rating_label(key: str) -> str:
    for pfx in ("off_", "pm_", "def_", "phy_", "skg_", "iqm_", "pc_", "dev_", "per_", "st_", "g_"):
        if key.startswith(pfx):
            return key[len(pfx) :].replace("_", " ").title()
    return key.replace("_", " ").title()


def _rating_groups_for_player(p: Any) -> List[Dict[str, Any]]:
    pos = (_pos_str(p) or "").strip().upper()
    is_g = pos == "G"
    r = getattr(p, "ratings", None) or {}

    def rows(keys: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for k in keys:
            out.append({"id": k, "label": _rating_label(k), "v": int(float(r.get(k, 68)))})
        return out

    groups: List[Dict[str, Any]] = [
        {"title": "Offense", "rows": rows(OFFENSE_ATTRS)},
        {"title": "Playmaking", "rows": rows(PLAYMAKING_ATTRS)},
        {"title": "Defense", "rows": rows(DEFENSE_ATTRS)},
        {"title": "Physical", "rows": rows(PHYSICAL_ATTRS)},
        {"title": "Skating", "rows": rows(SKATING_ATTRS)},
        {"title": "IQ / Mental", "rows": rows(IQ_ATTRS)},
        {"title": "Skill / Puck Control", "rows": rows(SKILL_ATTRS)},
        {"title": "Development", "rows": rows(DEV_ATTRS)},
        {"title": "Personality", "rows": rows(PERSONALITY_ATTRS)},
        {"title": "Special Traits", "rows": rows(SPECIAL_ATTRS)},
    ]
    if is_g:
        groups.insert(0, {"title": "Goalie", "rows": rows(GOALIE_ATTRS)})
    return groups


def _active_roster(team: Any) -> List[Any]:
    return [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]


def _skaters(team: Any) -> List[Any]:
    return [p for p in _active_roster(team) if _pos_str(p).upper() != "G"]


def _goalies(team: Any) -> List[Any]:
    return [p for p in _active_roster(team) if _pos_str(p).upper() == "G"]


def _available_goalies(team: Any) -> List[Any]:
    return [g for g in _goalies(team) if not _is_player_live_injured(g)]


def _goalie_availability_status(team: Any) -> Dict[str, Any]:
    all_goalies = _goalies(team)
    healthy = _available_goalies(team)
    return {
        "total": int(len(all_goalies)),
        "healthy": int(len(healthy)),
        "forced_injured_start": bool(all_goalies and not healthy),
    }


def _ovr_weight(p: Any) -> float:
    try:
        fn = getattr(p, "ovr", None)
        o = float(fn() if callable(fn) else fn)
        if o <= 1.5:
            o *= 99.0
        return max(14.0, o) ** 1.2
    except Exception:
        return 52.0


def _player_role_usage_mult(p: Any) -> float:
    """Approximate TOI/opportunity weighting by lineup role and quality."""
    role_raw = str(
        getattr(p, "line_role", None)
        or getattr(p, "role", None)
        or getattr(p, "depth_role", None)
        or ""
    ).lower()
    if "top" in role_raw or "line1" in role_raw or "first" in role_raw:
        return 2.35
    if "second" in role_raw or "line2" in role_raw:
        return 1.65
    if "third" in role_raw or "line3" in role_raw or "middle" in role_raw:
        return 0.92
    if "fourth" in role_raw or "line4" in role_raw or "depth" in role_raw:
        return 0.58
    return 0.85


def _rating_avg(p: Any, keys: List[str], default: float = 68.0) -> float:
    r = getattr(p, "ratings", None) or {}
    vals = [float(r.get(k, default) or default) for k in keys]
    if not vals:
        return default
    return sum(vals) / len(vals)


def _offense_opportunity_weight(p: Any) -> float:
    """
    Heavily bias offensive event participation to top-end talent and usage.
    """
    ovr = _ovr_weight(p)
    off = _rating_avg(p, OFFENSE_ATTRS)
    pm = _rating_avg(p, PLAYMAKING_ATTRS)
    iq = _rating_avg(p, IQ_ATTRS)
    pos = _pos_str(p).upper()
    pos_mult = 0.86 if pos == "D" else 1.0
    usage_mult = _player_role_usage_mult(p)
    base = (0.40 * off + 0.34 * pm + 0.26 * iq) / 99.0
    return max(0.04, (ovr ** 1.22) * (base ** 1.08) * usage_mult * pos_mult)


def _team_shooting_profile(team: Any) -> float:
    """Return team shooting baseline ~8%..12% with roster skill scaling."""
    sk = _skaters(team)
    if not sk:
        return 0.095
    top = sorted(sk, key=_offense_opportunity_weight, reverse=True)[:10]
    off = sum(_rating_avg(p, OFFENSE_ATTRS) for p in top) / max(1, len(top))
    iq = sum(_rating_avg(p, IQ_ATTRS) for p in top) / max(1, len(top))
    profile = 0.092 + 0.00028 * (off - 68.0) + 0.00012 * (iq - 68.0)
    return max(0.08, min(0.12, profile))


def _stat_ensure(session: FranchiseSession, p: Any, team_id: str) -> Dict[str, Any]:
    reg = session.player_season_stats
    pid = str(getattr(p, "id", "") or "")
    if not pid:
        return {}
    if pid not in reg:
        reg[pid] = {
            "player_id": pid,
            "name": _name_str(p),
            "team_id": str(team_id),
            "position": _pos_str(p),
            "gp": 0,
            "g": 0,
            "a": 0,
            "pts": 0,
            "sog": 0,
            "pim": 0,
            "hit": 0,
            "blk": 0,
            "toi_sec": 0,
            "ga": 0,
            "w": 0,
            "l": 0,
            "otl": 0,
        }
    row = reg[pid]
    row["name"] = _name_str(p)
    row["position"] = _pos_str(p)
    row["team_id"] = str(team_id)
    return row


def _stat_add(session: FranchiseSession, p: Any, team_id: str, **kwargs: int) -> None:
    row = _stat_ensure(session, p, team_id)
    if not row:
        return
    for k, v in kwargs.items():
        if v:
            row[k] = int(row.get(k, 0)) + int(v)
    row["pts"] = int(row.get("g", 0)) + int(row.get("a", 0))


def _pick_assist(rng: random.Random, skaters: List[Any], scorer: Any) -> Optional[Any]:
    pool = [s for s in skaters if s is not scorer]
    if not pool:
        return None
    w = [max(0.001, _offense_opportunity_weight(s) ** 1.35) for s in pool]
    return rng.choices(pool, weights=w, k=1)[0]


def _scoring_chunk(
    session: FranchiseSession,
    rng: random.Random,
    skaters: List[Any],
    tid: str,
    goals: int,
) -> List[str]:
    if not skaters or goals <= 0:
        return []
    w = [max(0.001, _offense_opportunity_weight(s) ** 1.45) for s in skaters]
    scorers = rng.choices(skaters, weights=w, k=int(goals))
    high: List[str] = []
    for scorer, ng in Counter(scorers).items():
        _stat_add(session, scorer, tid, g=int(ng))
        for _ in range(int(ng)):
            if rng.random() < 0.78:
                ap = _pick_assist(rng, skaters, scorer)
                if ap:
                    _stat_add(session, ap, tid, a=1)
                    if rng.random() < 0.46:
                        ap2 = _pick_assist(rng, [x for x in skaters if x is not scorer and x is not ap], scorer)
                        if ap2:
                            _stat_add(session, ap2, tid, a=1)
        nm = _name_str(scorer)
        high.append(f"{nm} ├ù{ng}" if ng > 1 else nm)
    return high


def _goalie_game(
    session: FranchiseSession,
    rng: random.Random,
    goalies: List[Any],
    tid: str,
    ga: int,
    won: bool,
    otl_loss: bool,
) -> Optional[Dict[str, Any]]:
    if not goalies:
        return None
    w = [_ovr_weight(g) for g in goalies]
    g0 = rng.choices(goalies, weights=w, k=1)[0]
    if won:
        _stat_add(session, g0, tid, gp=1, ga=int(ga), w=1)
    elif otl_loss:
        _stat_add(session, g0, tid, gp=1, ga=int(ga), otl=1)
    else:
        _stat_add(session, g0, tid, gp=1, ga=int(ga), l=1)
    shots_against = max(int(ga) * 3 + rng.randint(18, 34), int(ga) + 12)
    return {
        "player_id": str(getattr(g0, "id", "") or ""),
        "name": _name_str(g0),
        "ga": int(ga),
        "saves": int(shots_against - int(ga)),
        "shots_against": int(shots_against),
    }


def _skater_box_rows(
    session: FranchiseSession,
    rng: random.Random,
    team: Any,
    tid: str,
    team_shots: int,
) -> Dict[str, Dict[str, Any]]:
    """Per-skater game row shells (g/a filled by play-by-play)."""
    rows: Dict[str, Dict[str, Any]] = {}
    sk = _skaters(team)
    if not sk:
        return rows
    shot_weights = [max(0.001, _offense_opportunity_weight(p) ** 1.35) for p in sk]
    shot_owners = rng.choices(sk, weights=shot_weights, k=max(0, int(team_shots)))
    shot_counts = Counter(shot_owners)

    for p in sk:
        pid = str(getattr(p, "id", "") or "")
        if not pid:
            continue
        sog = int(shot_counts.get(p, 0))
        pos = _pos_str(p).upper()
        usage = _player_role_usage_mult(p)
        if pos == "D":
            toi_min = int(rng.randint(16, 24) * usage)
        else:
            toi_min = int(rng.randint(10, 18) * usage)
        toi_min = max(7, min(28, toi_min))
        toi = int(toi_min * 60 + rng.randint(0, 55))
        pim = int(rng.choices([0, 2, 4, 6], weights=[0.56, 0.28, 0.12, 0.04], k=1)[0])
        if pos == "D":
            hit = int(rng.choices([0, 1, 2, 3, 4], weights=[0.17, 0.28, 0.30, 0.18, 0.07], k=1)[0] * max(0.7, min(1.5, usage)))
            blk = int(rng.choices([0, 1, 2, 3, 4], weights=[0.10, 0.26, 0.33, 0.21, 0.10], k=1)[0] * max(0.8, min(1.5, usage)))
        else:
            hit = int(rng.choices([0, 1, 2, 3], weights=[0.29, 0.37, 0.24, 0.10], k=1)[0] * max(0.7, min(1.4, usage)))
            blk = int(rng.choices([0, 1, 2], weights=[0.64, 0.29, 0.07], k=1)[0] * max(0.7, min(1.3, usage)))
        _stat_add(session, p, tid, gp=1, sog=sog, pim=pim, hit=hit, blk=blk, toi_sec=toi)
        rows[pid] = {
            "player_id": pid,
            "name": _name_str(p),
            "position": _pos_str(p),
            "g": 0,
            "a": 0,
            "sog": sog,
            "pim": pim,
            "hit": hit,
            "blk": blk,
            "toi_sec": toi,
        }
    return rows


def _goals_play_by_play(
    session: FranchiseSession,
    rng: random.Random,
    skaters: List[Any],
    tid: str,
    goals: int,
    rows_by_pid: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Register goals + assists on season stats; return summary strings and scoring events."""
    events: List[Dict[str, Any]] = []
    high: List[str] = []
    if not skaters or goals <= 0:
        return high, events
    w = [max(0.001, _offense_opportunity_weight(s) ** 1.5) for s in skaters]
    for _ in range(int(goals)):
        scorer = rng.choices(skaters, weights=w, k=1)[0]
        spid = str(getattr(scorer, "id", "") or "")
        _stat_add(session, scorer, tid, g=1)
        if spid in rows_by_pid:
            rows_by_pid[spid]["g"] = int(rows_by_pid[spid].get("g", 0)) + 1
        assist_names: List[str] = []
        if rng.random() < 0.78:
            ap = _pick_assist(rng, skaters, scorer)
            if ap:
                _stat_add(session, ap, tid, a=1)
                apid = str(getattr(ap, "id", "") or "")
                if apid in rows_by_pid:
                    rows_by_pid[apid]["a"] = int(rows_by_pid[apid].get("a", 0)) + 1
                assist_names.append(_name_str(ap))
                if rng.random() < 0.46:
                    pool2 = [x for x in skaters if x is not scorer and x is not ap]
                    if pool2:
                        ap2 = _pick_assist(rng, pool2, scorer)
                        if ap2:
                            _stat_add(session, ap2, tid, a=1)
                            ap2id = str(getattr(ap2, "id", "") or "")
                            if ap2id in rows_by_pid:
                                rows_by_pid[ap2id]["a"] = int(rows_by_pid[ap2id].get("a", 0)) + 1
                            assist_names.append(_name_str(ap2))
        per = int(rng.choices([1, 2, 3], weights=[0.34, 0.42, 0.24])[0])
        mm = int(rng.randint(0, 19))
        ss = int(rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]))
        rv = rng.random()
        strength = "EV"
        if rv < 0.23:
            strength = "PP"
        elif rv < 0.29:
            strength = "SH"
        events.append(
            {
                "for_team_id": str(tid),
                "period": per,
                "clock": f"{mm}:{ss:02d}",
                "scorer": _name_str(scorer),
                "scorer_id": spid,
                "assists": assist_names,
                "strength": strength,
            }
        )
        high.append(_name_str(scorer))
    return high, events


def _accumulate_franchise_game_stats(
    session: FranchiseSession,
    *,
    home: Any,
    away: Any,
    hid: str,
    aid: str,
    hg: int,
    ag: int,
    ot: bool,
    calendar_day: int,
    rng: random.Random,
    calendar_iso: str = "",
) -> None:
    """
    Single pipeline: SimEngine accumulates skater/goalie stats from the same _simulate_game outcome.

    Hard rules:
    - game box must be a dict
    - final score in box must match simulated score
    - no tied final
    - no negative score
    - append only a clean completed game box
    """
    sim = session.sim
    gid = uuid.uuid4().hex[:14]

    hg = _coerce_final_score(hg)
    ag = _coerce_final_score(ag)

    if hg == ag:
        raise RuntimeError(
            f"Stat accumulation refused tied final on day {calendar_day}: {hid} {hg}, {aid} {ag}"
        )

    box = sim.accumulate_unified_game_stats(
        rng,
        home,
        away,
        str(hid),
        str(aid),
        int(hg),
        int(ag),
        bool(ot),
        session.player_season_stats,
        build_game_payload=True,
        calendar_day=int(calendar_day),
        calendar_iso=str(calendar_iso or ""),
        game_id=gid,
    )

    if not isinstance(box, dict):
        raise RuntimeError(
            f"Stat accumulation failed on calendar day {calendar_day}: SimEngine returned no game box."
        )

    box_hg = _coerce_final_score(box.get("home_goals", box.get("home_score", hg)))
    box_ag = _coerce_final_score(box.get("away_goals", box.get("away_score", ag)))

    if box_hg != int(hg) or box_ag != int(ag):
        raise RuntimeError(
            f"Stat/game mismatch on day {calendar_day}: sim score {hid} {hg}, {aid} {ag}; "
            f"box score {box_hg}-{box_ag}."
        )

    box.update(
        {
            "game_id": str(box.get("game_id") or gid),
            "id": str(box.get("id") or box.get("game_id") or gid),
            "home_id": str(hid),
            "away_id": str(aid),
            "home_name": _display_team(home),
            "away_name": _display_team(away),
            "home_goals": int(hg),
            "away_goals": int(ag),
            "home_score": int(hg),
            "away_score": int(ag),
            "overtime": bool(ot),
            "ot": bool(ot),
            "day": int(calendar_day),
            "calendar_day": int(calendar_day),
            "iso": str(calendar_iso or box.get("iso") or ""),
            "calendar_iso": str(calendar_iso or box.get("calendar_iso") or box.get("iso") or ""),
            "status": "final",
            "completed": True,
            "is_final": True,
            "simmed": True,
        }
    )

    session.game_results.append(box)

    if len(session.game_results) > 2400:
        session.game_results = session.game_results[-1800:]
def _build_stats_central_payload(session: FranchiseSession) -> Dict[str, Any]:
    """
    StatsCentral payload.

    Important:
    - uses only the game-derived stat ledger
    - normalizes skaters and goalies separately
    - exposes league leaders, user leaders, goalie leaders, and diagnostics
    - never invents scoring
    """
    all_results = [
        g for g in list(getattr(session, "game_results", None) or [])
        if isinstance(g, dict)
    ]

    games = list(reversed(all_results[-100:]))

    by_day: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for g in all_results:
        try:
            di = int(g.get("day", g.get("calendar_day", -1)))
        except (TypeError, ValueError):
            continue

        if di < 0:
            continue

        by_day[di].append(g)

    day_keys = sorted(by_day.keys(), reverse=True)
    calendar = [
        {
            "day": int(d),
            "games": list(reversed(by_day[d])),
        }
        for d in day_keys
    ]

    raw_rows = list((getattr(session, "player_season_stats", None) or {}).values())
    rows = [_normalize_player_stat_row(dict(r or {})) for r in raw_rows]

    skaters = [r for r in rows if not r.get("is_goalie")]
    goalies = [r for r in rows if r.get("is_goalie")]

    skaters.sort(
        key=lambda r: (
            -int(r.get("pts", 0) or 0),
            -int(r.get("g", 0) or 0),
            -int(r.get("a", 0) or 0),
            -int(r.get("sog", 0) or 0),
            str(r.get("name") or ""),
        )
    )

    leaders = skaters[:45]

    uid = str(getattr(session, "user_team_id", "") or "")

    user_only = [r for r in skaters if str(r.get("team_id") or "") == uid]
    user_only.sort(
        key=lambda r: (
            -int(r.get("pts", 0) or 0),
            -int(r.get("g", 0) or 0),
            -int(r.get("a", 0) or 0),
            str(r.get("name") or ""),
        )
    )

    goalie_leaders = [
        r for r in goalies
        if int(r.get("shots_against", 0) or 0) >= 50 or int(r.get("gp", 0) or 0) >= 3
    ]

    goalie_leaders.sort(
        key=lambda r: (
            -float(r.get("save_pct", 0.0) or 0.0),
            float(r.get("gaa", 99.0) or 99.0),
            -int(r.get("gp", 0) or 0),
            str(r.get("name") or ""),
        )
    )

    team_totals: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "team_id": "",
            "gp_player_rows": 0,
            "goals": 0,
            "assists": 0,
            "points": 0,
            "shots": 0,
            "hits": 0,
            "blocks": 0,
            "pim": 0,
        }
    )

    for r in skaters:
        tid = str(r.get("team_id") or "")
        if tid == "":
            continue

        trow = team_totals[tid]
        trow["team_id"] = tid
        trow["gp_player_rows"] += int(r.get("gp", 0) or 0)
        trow["goals"] += int(r.get("g", 0) or 0)
        trow["assists"] += int(r.get("a", 0) or 0)
        trow["points"] += int(r.get("pts", 0) or 0)
        trow["shots"] += int(r.get("sog", 0) or 0)
        trow["hits"] += int(r.get("hit", 0) or 0)
        trow["blocks"] += int(r.get("blk", 0) or 0)
        trow["pim"] += int(r.get("pim", 0) or 0)

    integrity = _stats_integrity_payload(rows, all_results)
    from app.sim_engine.franchise.serialization import _build_league_teams_payload

    league_teams, teams_directory = _build_league_teams_payload(session, team_totals=team_totals)

    return {
        "games": games,
        "calendar": calendar,
        "players": rows,
        "skaters": skaters,
        "goalies": goalies,
        "leaders": leaders,
        "league_leaders": leaders,
        "user_leaders": user_only[:20],
        "goalie_leaders": goalie_leaders[:20],
        "team_totals": list(team_totals.values()),
        "teams": league_teams,
        "league_teams": league_teams,
        "league_team_stats": league_teams,
        "teams_directory": teams_directory,
        "integrity": integrity,
        "meta": {
            "source": "game_derived_player_season_stats",
            "stat_source": "game_ledger",
            "player_rows": len(rows),
            "skater_rows": len(skaters),
            "goalie_rows": len(goalies),
            "completed_games": len(all_results),
            "leader_limit": 45,
            "game_derived_only": True,
        },
    }
def _stat_int(row: Dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        if key in row:
            try:
                return int(round(float(row.get(key) or 0)))
            except (TypeError, ValueError):
                continue
    return int(default)


def _stat_float(row: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row:
            try:
                return float(row.get(key) or 0.0)
            except (TypeError, ValueError):
                continue
    return float(default)


def _normalize_stat_position(value: Any) -> str:
    pos = str(value or "").strip().upper()
    if pos in {"G", "GOALIE", "GK"}:
        return "G"
    if pos in {"D", "LD", "RD"}:
        return "D"
    if pos in {"C", "LW", "RW", "F"}:
        return pos
    return pos or "F"


def _is_goalie_stat_row(row: Dict[str, Any]) -> bool:
    pos = _normalize_stat_position(row.get("position") or row.get("pos"))
    if pos == "G":
        return True

    goalie_markers = (
        "shots_against",
        "saves",
        "save_pct",
        "gaa",
        "ga",
        "w",
        "l",
        "otl",
    )

    return any(k in row for k in goalie_markers) and not (_stat_int(row, "g") or _stat_int(row, "a"))


def _normalize_player_stat_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean one player season stat row before sending it to frontend.

    Important:
    - does not create fake production
    - only derives totals from existing game-derived stats
    - splits skater and goalie interpretation
    """
    if not isinstance(row, dict):
        row = {}

    player_id = str(row.get("player_id") or row.get("id") or "")
    team_id = str(row.get("team_id") or row.get("team") or "")
    name = str(row.get("name") or row.get("player") or row.get("player_name") or "Player")
    position = _normalize_stat_position(row.get("position") or row.get("pos") or "F")

    gp = max(0, _stat_int(row, "gp", "games_played"))
    goals = max(0, _stat_int(row, "g", "goals"))
    assists = max(0, _stat_int(row, "a", "assists"))
    points = goals + assists

    sog = max(0, _stat_int(row, "sog", "shots", "shots_on_goal"))
    pim = max(0, _stat_int(row, "pim", "penalty_minutes"))
    hits = max(0, _stat_int(row, "hit", "hits"))
    blocks = max(0, _stat_int(row, "blk", "blocks"))
    toi_sec = max(0, _stat_int(row, "toi_sec", "time_on_ice_sec", "toi_total_sec"))

    is_goalie = _is_goalie_stat_row(row)

    normalized = {
        **row,
        "player_id": player_id,
        "id": player_id,
        "team_id": team_id,
        "name": name,
        "player_name": name,
        "position": position,
        "gp": gp,
        "g": goals,
        "goals": goals,
        "a": assists,
        "assists": assists,
        "pts": points,
        "points": points,
        "sog": sog,
        "shots": sog,
        "pim": pim,
        "hit": hits,
        "hits": hits,
        "blk": blocks,
        "blocks": blocks,
        "toi_sec": toi_sec,
        "toi_total_sec": toi_sec,
        "toi": round((toi_sec / max(1, gp)) / 60.0, 1) if gp > 0 else 0.0,
        "points_per_game": round(points / max(1, gp), 3) if gp > 0 else 0.0,
        "goals_per_game": round(goals / max(1, gp), 3) if gp > 0 else 0.0,
        "assists_per_game": round(assists / max(1, gp), 3) if gp > 0 else 0.0,
        "is_goalie": bool(is_goalie),
        "stat_type": "goalie" if is_goalie else "skater",
    }

    if is_goalie:
        shots_against = max(0, _stat_int(row, "shots_against", "sa"))
        saves = max(0, _stat_int(row, "saves", "sv"))
        goals_against = max(0, _stat_int(row, "ga", "goals_against"))

        if shots_against <= 0 and saves > 0:
            shots_against = saves + goals_against

        if saves <= 0 and shots_against > 0:
            saves = max(0, shots_against - goals_against)

        save_pct = saves / max(1, shots_against)
        goalie_toi_sec = max(0, _stat_int(row, "toi_sec", "time_on_ice_sec", "toi_total_sec"))
        if goalie_toi_sec <= 0 and gp > 0:
            goalie_toi_sec = gp * 3600
        gaa = (goals_against * 3600.0) / goalie_toi_sec if goalie_toi_sec > 0 else 0.0

        normalized.update(
            {
                "shots_against": shots_against,
                "saves": saves,
                "ga": goals_against,
                "goals_against": goals_against,
                "w": max(0, _stat_int(row, "w", "wins")),
                "l": max(0, _stat_int(row, "l", "losses")),
                "otl": max(0, _stat_int(row, "otl", "ot_losses")),
                "toi_sec": goalie_toi_sec,
                "toi_total_sec": goalie_toi_sec,
                "toi": round((goalie_toi_sec / max(1, gp)) / 60.0, 1) if gp > 0 else 0.0,
                "save_pct": round(save_pct, 4),
                "gaa": round(gaa, 3),
            }
        )

    return normalized


def _stats_integrity_payload(rows: List[Dict[str, Any]], game_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Dev-facing truth meter for stat sanity.
    """
    warnings: List[str] = []

    skaters = [r for r in rows if not r.get("is_goalie")]
    goalies = [r for r in rows if r.get("is_goalie")]

    total_player_goals = sum(int(r.get("g", 0) or 0) for r in skaters)

    total_box_goals = 0
    valid_games = 0

    for g in game_results or []:
        if not isinstance(g, dict):
            continue

        try:
            hg = int(round(float(g.get("home_goals", g.get("home_score", 0)) or 0)))
            ag = int(round(float(g.get("away_goals", g.get("away_score", 0)) or 0)))
        except (TypeError, ValueError):
            continue

        if hg < 0 or ag < 0:
            continue

        # Do not count obvious placeholders.
        status = str(g.get("status") or g.get("game_status") or "").lower()
        if hg == 0 and ag == 0 and status not in {"final", "completed", "complete", "simmed", "played"}:
            continue

        total_box_goals += hg + ag
        valid_games += 1

    if valid_games and abs(total_player_goals - total_box_goals) > 0:
        warnings.append(
            f"PLAYER_GOALS_MISMATCH: skater goals {total_player_goals} != game box goals {total_box_goals}."
        )

    top_pts = max([int(r.get("pts", 0) or 0) for r in skaters], default=0)
    top_gp = max([int(r.get("gp", 0) or 0) for r in skaters], default=0)

    if valid_games >= 300 and top_pts < 45:
        warnings.append(
            f"LOW_LEAGUE_SCORING: top scorer has only {top_pts} points after {valid_games} completed games."
        )

    return {
        "skater_rows": len(skaters),
        "goalie_rows": len(goalies),
        "valid_games_counted": int(valid_games),
        "total_player_goals": int(total_player_goals),
        "total_box_goals": int(total_box_goals),
        "goals_match_boxscores": bool(total_player_goals == total_box_goals),
        "top_scorer_points": int(top_pts),
        "top_player_gp": int(top_gp),
        "warnings": warnings,
    }
def _build_schedule_upcoming(session: FranchiseSession, *, limit: int = 14) -> List[Dict[str, Any]]:
    """Next NHL calendar days from the current cursor (real dates) ΓÇö hub / calendar UI."""
    if str(getattr(session, "phase", "")) != "regular":
        return []
    cal = getattr(session, "nhl_calendar", None) or []
    if not cal:
        return []
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    last = int(getattr(session, "nhl_regular_season_last_index", 0) or 0)
    hi = min(last + 1, cur + int(limit))
    by_day = getattr(session, "by_day", None) or {}
    out: List[Dict[str, Any]] = []
    uid = str(getattr(session, "user_team_id", "") or "")
    for idx in range(cur, hi):
        if idx > last:
            break
        row = cal[idx] if idx < len(cal) else {}
        slots = by_day.get(idx, []) or []
        games: List[Dict[str, Any]] = []
        for sl in slots:
            hid = _safe_slot_team_id(sl, "home_id")
            aid = _safe_slot_team_id(sl, "away_id")
            ht = session.team_by_id.get(hid)
            at = session.team_by_id.get(aid)
            games.append(
                {
                    "home_id": hid,
                    "away_id": aid,
                    "home_name": _display_team(ht) if ht else hid,
                    "away_name": _display_team(at) if at else aid,
                }
            )
        user_plays = bool(uid) and any(uid in (g["home_id"], g["away_id"]) for g in games)
        out.append(
            {
                "day": int(idx),
                "calendar_index": int(idx),
                "iso": str(row.get("iso") or ""),
                "weekday": str(row.get("weekday") or ""),
                "segment": str(row.get("segment") or ""),
                "ui_phase": str(row.get("ui_phase") or ""),
                "tags": list(row.get("tags") or []),
                "games": games,
                "user_plays": user_plays,
            }
        )
    return out


def _nhl_today_payload(session: FranchiseSession) -> Dict[str, Any]:
    """Current calendar row + gameday headline for the hub command deck."""
    cal = getattr(session, "nhl_calendar", None) or []
    if not cal or str(getattr(session, "phase", "")) != "regular":
        return {}
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    last = int(getattr(session, "nhl_regular_season_last_index", 0) or 0)
    if cur > last:
        return {"headline": "Regular season complete ΓÇö advance for playoffs", "iso": "", "segment": "regular", "calendar_index": cur}
    cur = max(0, min(cur, len(cal) - 1))
    row = dict(cal[cur])
    row["calendar_index"] = int(cur)
    slots = list((getattr(session, "by_day", None) or {}).get(int(session.calendar_cursor), []) or [])
    uid = str(session.user_team_id)
    user_plays = any(uid in (str(s.home_id), str(s.away_id)) for s in slots)
    row["has_league_games"] = bool(slots)
    row["user_game_today"] = bool(user_plays)
    opp = None
    for s in slots:
        hid, aid = str(s.home_id), str(s.away_id)
        if hid == uid:
            opp = session.team_by_id.get(aid)
        elif aid == uid:
            opp = session.team_by_id.get(hid)
        if opp:
            break
    if user_plays and opp:
        row["headline"] = f"Game day vs {_display_team(opp)}"
    elif slots:
        row["headline"] = "League gameday (your club is off)"
    else:
        row["headline"] = str(row.get("ui_note") or "Off day")
    return row


def _results_by_calendar_index(session: FranchiseSession) -> Dict[int, List[Dict[str, Any]]]:
    """
    Simmed box scores keyed by franchise calendar index.

    Important:
    Only valid final games get scores.
    Scheduled placeholders must not leak 0-0 scores to the frontend.
    """
    by_idx: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for g in getattr(session, "game_results", None) or []:
        if not isinstance(g, dict):
            continue

        try:
            di = int(g.get("day", -1))
        except (TypeError, ValueError):
            continue

        if di < 0:
            continue

        if not _saved_game_is_final(g):
            continue

        hid = str(g.get("home_id") or "")
        aid = str(g.get("away_id") or "")

        try:
            hg, ag = _validate_final_game_result_payload(
                home_id=hid,
                away_id=aid,
                home_goals=g.get("home_goals", g.get("home_score")),
                away_goals=g.get("away_goals", g.get("away_score")),
                calendar_day=di,
            )
        except ValueError:
            continue

        by_idx[di].append(
            {
                "game_id": _stable_franchise_game_id(g),
                "home_id": hid,
                "away_id": aid,
                "home_goals": int(hg),
                "away_goals": int(ag),
                "overtime": bool(g.get("overtime") or g.get("went_ot") or g.get("ot")),
                "iso": str(g.get("iso") or ""),
                "status": "final",
                "completed": True,
                "is_final": True,
                "simmed": True,
            }
        )

    return dict(by_idx)
def _coerce_final_score(value: Any) -> int:
    """
    Backend score sanitizer.
    Game scores must be non-negative integers.
    """
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid game score value: {value!r}")

    if n < 0:
        raise ValueError(f"Negative game score value: {value!r}")

    return n


def _validate_final_game_result_payload(
    *,
    home_id: str,
    away_id: str,
    home_goals: Any,
    away_goals: Any,
    calendar_day: int,
) -> Tuple[int, int]:
    """
    Hard validator for a completed game.

    A completed hockey game must have:
    - two different teams
    - valid non-negative integer scores
    - no tie after final
    """
    hid = str(home_id or "").strip()
    aid = str(away_id or "").strip()

    if not hid or not aid:
        raise ValueError(f"Game result missing team id(s) on calendar day {calendar_day}.")

    if hid == aid:
        raise ValueError(f"Self-play game result detected on calendar day {calendar_day}: {hid}.")

    hg = _coerce_final_score(home_goals)
    ag = _coerce_final_score(away_goals)

    if hg == ag:
        raise ValueError(
            f"Final game result cannot be tied on calendar day {calendar_day}: {hid} {hg}, {aid} {ag}."
        )

    return hg, ag


def _saved_game_is_final(g: Dict[str, Any]) -> bool:
    """
    Saved backend result should only be considered final if explicitly final
    OR it came from the sim result store with valid final scores.

    This prevents schedule placeholders from becoming fake 0-0 finals.
    """
    if not isinstance(g, dict):
        return False

    status = str(
        g.get("status")
        or g.get("game_status")
        or g.get("state")
        or ""
    ).strip().lower()

    explicit_final = status in {
        "final",
        "completed",
        "complete",
        "played",
        "done",
        "simmed",
        "finished",
    }

    has_home = g.get("home_goals", None) is not None or g.get("home_score", None) is not None
    has_away = g.get("away_goals", None) is not None or g.get("away_score", None) is not None

    if not has_home or not has_away:
        return False

    try:
        hg = _coerce_final_score(g.get("home_goals", g.get("home_score")))
        ag = _coerce_final_score(g.get("away_goals", g.get("away_score")))
    except ValueError:
        return False

    # Placeholder scheduled games are often 0-0.
    # A true final can never be tied anyway.
    if hg == ag:
        return False

    return explicit_final or bool(g.get("simmed") or g.get("completed") or g.get("is_final"))


def _game_result_calendar_index(g: Dict[str, Any]) -> Optional[int]:
    """Calendar index from a saved game box (day 0 is valid ΓÇö never use `value or default` on day)."""
    if not isinstance(g, dict):
        return None
    v = g.get("day")
    if v is None:
        v = g.get("calendar_day")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _remaining_regular_games_count(session: FranchiseSession) -> int:
    """
    Count unplayed regular-season scheduled games remaining.
    Used to prevent playoffs from starting too early.
    """
    cal = getattr(session, "nhl_calendar", None) or []
    by_day = getattr(session, "by_day", None) or {}

    remaining = 0

    for day_idx, slots in (by_day or {}).items():
        try:
            di = int(day_idx)
        except (TypeError, ValueError):
            continue

        if di < 0 or di >= len(cal):
            continue

        row = cal[di] or {}
        seg = str(row.get("segment") or row.get("season_segment") or "")

        if seg != "regular":
            continue

        remaining += len(slots or [])

    return int(remaining)


def _completed_regular_games_count(session: FranchiseSession) -> int:
    """
    Count valid completed regular-season games from session.game_results.
    """
    cal = getattr(session, "nhl_calendar", None) or []
    count = 0

    for g in getattr(session, "game_results", None) or []:
        if not isinstance(g, dict):
            continue

        try:
            di = int(g.get("day", -1))
        except (TypeError, ValueError):
            continue

        if di < 0 or di >= len(cal):
            continue

        row = cal[di] or {}
        seg = str(row.get("segment") or row.get("season_segment") or "")

        if seg != "regular":
            continue

        if _saved_game_is_final(g):
            count += 1

    return int(count)


def _regular_season_is_truly_complete(session: FranchiseSession) -> bool:
    """
    True only when:
    - calendar cursor is past the regular season boundary
    - no regular-season slots remain in by_day
    """
    _sync_nhl_calendar_bounds(session)

    cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    last = int(getattr(session, "nhl_regular_season_last_index", 0) or 0)

    if cursor <= last:
        return False

    return _remaining_regular_games_count(session) == 0
def _game_results_by_calendar_day(session: FranchiseSession) -> Dict[int, List[Dict[str, Any]]]:
    """
    Full completed game payloads grouped by calendar index.

    Slots are cleared after sim, so this preserves past games.
    But it must only return real completed games.
    """
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for g in getattr(session, "game_results", None) or []:
        if not isinstance(g, dict):
            continue

        try:
            di = int(g.get("day", -1))
        except (TypeError, ValueError):
            continue

        if di < 0:
            continue

        if not _saved_game_is_final(g):
            continue

        out[di].append(g)

    return dict(out)


def _stable_franchise_game_id(g: Dict[str, Any]) -> str:
    """Stable recap key: prefer stored game_id, else day+matchup (older saves / edge cases)."""
    existing = str(g.get("game_id") or "").strip()
    if existing:
        return existing
    try:
        di = int(g.get("day", -1))
    except (TypeError, ValueError):
        di = -1
    return f"d{di}_{g.get('home_id')}_{g.get('away_id')}"


def _stable_franchise_game_id_from_row(hid: str, aid: str, calendar_day: int) -> str:
    return f"d{int(calendar_day)}_{hid}_{aid}"


def _ginfo_from_saved(session: FranchiseSession, saved: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a saved completed result into frontend game info.

    Important:
    - Only attaches scores if the game is truly final.
    - Never turns missing scores into 0-0.
    """
    hid = str(saved.get("home_id") or "")
    aid = str(saved.get("away_id") or "")
    ht = session.team_by_id.get(hid)
    at = session.team_by_id.get(aid)

    try:
        di = int(saved.get("day", -1))
    except (TypeError, ValueError):
        di = -1

    row = {
        "game_id": _stable_franchise_game_id(saved),
        "id": _stable_franchise_game_id(saved),
        "home_id": hid,
        "away_id": aid,
        "home_name": str(saved.get("home_name") or (_display_team(ht) if ht else hid)),
        "away_name": str(saved.get("away_name") or (_display_team(at) if at else aid)),
        "home_abbr": _team_abbr(ht, hid),
        "away_abbr": _team_abbr(at, aid),
        "day": di,
        "calendar_day": di,
        "iso": str(saved.get("iso") or ""),
        "date": str(saved.get("iso") or ""),
        "status": "scheduled",
        "completed": False,
        "is_final": False,
        "simmed": False,
    }

    if _saved_game_is_final(saved):
        try:
            hg, ag = _validate_final_game_result_payload(
                home_id=hid,
                away_id=aid,
                home_goals=saved.get("home_goals", saved.get("home_score")),
                away_goals=saved.get("away_goals", saved.get("away_score")),
                calendar_day=di,
            )
        except ValueError:
            return row

        row.update(
            {
                "home_goals": int(hg),
                "away_goals": int(ag),
                "home_score": int(hg),
                "away_score": int(ag),
                "overtime": bool(saved.get("overtime") or saved.get("went_ot") or saved.get("ot")),
                "status": "final",
                "completed": True,
                "is_final": True,
                "simmed": True,
            }
        )

    return row


def _nhl_calendar_full_with_slates(session: FranchiseSession) -> List[Dict[str, Any]]:
    """Full season calendar rows plus NHL slate + user focus + final scores for the franchise UI."""
    cal = getattr(session, "nhl_calendar", None) or []
    if not cal:
        return []
    by_day = getattr(session, "by_day", None) or {}
    uid = str(session.user_team_id)
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    results_by_idx = _results_by_calendar_index(session)
    saved_by_day = _game_results_by_calendar_day(session)

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(cal):
        d = dict(row)
        d["calendar_index"] = int(i)
        d["is_today_cursor"] = bool(i == cur)
        d["is_past"] = bool(i < cur)
        slots = list(by_day.get(i, []) or [])
        res_list = list(results_by_idx.get(i, []) or [])
        saved_list = list(saved_by_day.get(i, []) or [])
        games: List[Dict[str, Any]] = []

        for sl in slots:
            hid = _safe_slot_team_id(sl, "home_id")
            aid = _safe_slot_team_id(sl, "away_id")
            ht = session.team_by_id.get(hid)
            at = session.team_by_id.get(aid)
            ginfo: Dict[str, Any] = {
                "home_id": hid,
                "away_id": aid,
                "home_name": _display_team(ht) if ht else hid,
                "away_name": _display_team(at) if at else aid,
                "home_abbr": _team_abbr(ht, hid),
                "away_abbr": _team_abbr(at, aid),
            }
            for rg in res_list:
                if str(rg.get("home_id")) == hid and str(rg.get("away_id")) == aid:
                    ginfo["home_goals"] = int(rg.get("home_goals", 0) or 0)
                    ginfo["away_goals"] = int(rg.get("away_goals", 0) or 0)
                    ginfo["overtime"] = bool(rg.get("overtime"))
                    ginfo["game_id"] = str(rg.get("game_id") or "") or _stable_franchise_game_id_from_row(hid, aid, i)
                    break
            if "game_id" not in ginfo:
                ginfo["game_id"] = _stable_franchise_game_id_from_row(hid, aid, i)
            games.append(ginfo)

        if not games and saved_list:
            for sg in saved_list:
                games.append(_ginfo_from_saved(session, sg))

        d["has_slate"] = bool(slots or saved_list)
        d["games"] = games
        user_game: Optional[Dict[str, Any]] = None
        for g in games:
            hid = str(g.get("home_id") or "")
            aid = str(g.get("away_id") or "")
            if uid not in (hid, aid):
                continue
            is_home = hid == uid
            opp_id = aid if is_home else hid
            opp_name = str(g.get("away_name") if is_home else g.get("home_name") or opp_id)
            opp_abbr = str(g.get("away_abbr") if is_home else g.get("home_abbr") or opp_id)
            res_bits: Dict[str, Any] = {
                "home_away": "home" if is_home else "away",
                "opponent_id": opp_id,
                "opponent_name": opp_name,
                "opponent_abbr": opp_abbr,
            }
            res_bits["game_id"] = str(g.get("game_id") or "").strip() or _stable_franchise_game_id_from_row(hid, aid, i)
            if "home_goals" in g:
                hg = int(g["home_goals"])
                ag = int(g["away_goals"])
                ot = bool(g.get("overtime"))
                won = (hg > ag) if is_home else (ag > hg)
                user_sc = int(hg) if is_home else int(ag)
                opp_sc = int(ag) if is_home else int(hg)
                if won:
                    letter = "W"
                elif ot and not won:
                    letter = "OTL"
                else:
                    letter = "L"
                res_bits["record_line"] = f"{user_sc}-{opp_sc}{' OT' if ot else ''} ({letter})"
                res_bits["result_letter"] = letter
                res_bits["user_score"] = user_sc
                res_bits["opp_score"] = opp_sc
                res_bits["overtime"] = ot
            user_game = res_bits
            break
        d["user_game"] = user_game
        d["special_event"] = _special_event_overlay(session, d)
        out.append(d)
    return out


def _team_abbr(team: Any, tid: str) -> str:
    standard = {
        "anaheim ducks": "ANA",
        "boston bruins": "BOS",
        "buffalo sabres": "BUF",
        "calgary flames": "CGY",
        "carolina hurricanes": "CAR",
        "chicago blackhawks": "CHI",
        "colorado avalanche": "COL",
        "columbus blue jackets": "CBJ",
        "dallas stars": "DAL",
        "detroit red wings": "DET",
        "edmonton oilers": "EDM",
        "florida panthers": "FLA",
        "los angeles kings": "LAK",
        "minnesota wild": "MIN",
        "montreal canadiens": "MTL",
        "montr├⌐al canadiens": "MTL",
        "nashville predators": "NSH",
        "new jersey devils": "NJD",
        "new york islanders": "NYI",
        "new york rangers": "NYR",
        "ottawa senators": "OTT",
        "philadelphia flyers": "PHI",
        "pittsburgh penguins": "PIT",
        "seattle kraken": "SEA",
        "san jose sharks": "SJS",
        "st. louis blues": "STL",
        "st louis blues": "STL",
        "tampa bay lightning": "TBL",
        "toronto maple leafs": "TOR",
        "utah hockey club": "UTA",
        "vancouver canucks": "VAN",
        "vegas golden knights": "VGK",
        "washington capitals": "WSH",
        "winnipeg jets": "WPG",
    }

    raw_tid = str(tid or "").strip().upper()
    if raw_tid in standard.values():
        return raw_tid

    if team is not None:
        city = str(getattr(team, "city", "") or "").strip().lower()
        name = str(getattr(team, "name", "") or "").strip().lower()
        full = f"{city} {name}".strip()
        if full in standard:
            return standard[full]
        if name:
            # fallback by nickname token when city text is unusual
            by_name = {
                "ducks": "ANA",
                "bruins": "BOS",
                "sabres": "BUF",
                "flames": "CGY",
                "hurricanes": "CAR",
                "blackhawks": "CHI",
                "avalanche": "COL",
                "blue jackets": "CBJ",
                "stars": "DAL",
                "red wings": "DET",
                "oilers": "EDM",
                "panthers": "FLA",
                "kings": "LAK",
                "wild": "MIN",
                "canadiens": "MTL",
                "predators": "NSH",
                "devils": "NJD",
                "islanders": "NYI",
                "rangers": "NYR",
                "senators": "OTT",
                "flyers": "PHI",
                "penguins": "PIT",
                "kraken": "SEA",
                "sharks": "SJS",
                "blues": "STL",
                "lightning": "TBL",
                "maple leafs": "TOR",
                "hockey club": "UTA",
                "canucks": "VAN",
                "golden knights": "VGK",
                "capitals": "WSH",
                "jets": "WPG",
            }
            if name in by_name:
                return by_name[name]

    return raw_tid[:3] if raw_tid else "NHL"


def _city_lower(team: Any) -> str:
    return str(getattr(team, "city", "") or "").strip().lower()


def _is_canadian_franchise_team(team: Any) -> bool:
    if team is None:
        return False
    city = _city_lower(team)
    hints = ("toronto", "montr├⌐al", "montreal", "ottawa", "winnipeg", "edmonton", "calgary", "vancouver")
    if any(h in city for h in hints):
        return True
    nm = str(getattr(team, "name", "") or "").lower()
    for hint in ("maple leaf", "canadien", "senator", "jet", "oiler", "flame", "canuck"):
        if hint in nm:
            return True
    return False


def _standing_row_snapshot(session: FranchiseSession, tid: str, r: Any) -> Dict[str, Any]:
    tm = session.team_by_id.get(tid)
    return {
        "id": str(tid),
        "abbr": _team_abbr(tm, str(tid)),
        "name": str(getattr(r, "name", tid)),
    }


def _top_two_teams_overall(session: FranchiseSession) -> List[Dict[str, Any]]:
    if not getattr(session, "standings", None):
        return []
    rows: List[Tuple[int, str, Any]] = []
    for tid, r in session.standings.records.items():
        rows.append((int(getattr(r, "points", 0) or 0), str(tid), r))
    rows.sort(key=lambda x: (-x[0], x[1]))
    return [_standing_row_snapshot(session, tid, r) for _, tid, r in rows[:2]]


def _top_two_canadian_teams(session: FranchiseSession) -> List[Dict[str, Any]]:
    if not getattr(session, "standings", None):
        return []
    rows: List[Tuple[int, str, Any]] = []
    for tid, r in session.standings.records.items():
        tm = session.team_by_id.get(tid)
        if not _is_canadian_franchise_team(tm):
            continue
        rows.append((int(getattr(r, "points", 0) or 0), str(tid), r))
    rows.sort(key=lambda x: (-x[0], x[1]))
    if len(rows) >= 2:
        return [_standing_row_snapshot(session, tid, r) for _, tid, r in rows[:2]]
    return _top_two_teams_overall(session)


def _special_event_overlay(session: FranchiseSession, day: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Showcase matchups derived from current standings (same pass as calendar cells)."""
    tags = {str(x).lower() for x in (day.get("tags") or [])}
    tbd = {"id": "", "abbr": "?", "name": "TBD"}
    if "winter_classic" in tags:
        pair = _top_two_teams_overall(session)
        if len(pair) >= 2:
            return {"kind": "winter_classic", "title": "Winter Classic", "home": pair[0], "away": pair[1]}
        return {"kind": "winter_classic", "title": "Winter Classic", "home": tbd, "away": tbd}
    if "heritage_classic" in tags:
        pair = _top_two_canadian_teams(session)
        if len(pair) >= 2:
            return {"kind": "heritage_classic", "title": "Heritage Classic", "home": pair[0], "away": pair[1]}
        return {"kind": "heritage_classic", "title": "Heritage Classic", "home": tbd, "away": tbd}
    if "four_nations" in tags:
        return {
            "kind": "four_nations",
            "title": "4 Nations Face-Off",
            "nations": [
                {"code": "CAN", "label": "Canada"},
                {"code": "USA", "label": "United States"},
                {"code": "SWE", "label": "Sweden"},
                {"code": "FIN", "label": "Finland"},
            ],
        }
    return None


def _nhl_calendar_strip(session: FranchiseSession, *, before: int = 10, after: int = 28) -> List[Dict[str, Any]]:
    """Window of NHL dates around the cursor for the calendar screen."""
    cal = getattr(session, "nhl_calendar", None) or []
    if not cal:
        return []
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    lo = max(0, cur - int(before))
    hi = min(len(cal), cur + int(after))
    by_day = getattr(session, "by_day", None) or {}
    out: List[Dict[str, Any]] = []
    for i in range(lo, hi):
        d = dict(cal[i]) if i < len(cal) else {}
        d["calendar_index"] = i
        d["has_game"] = bool(by_day.get(i))
        d["is_today"] = i == cur
        out.append(d)
    return out


def _serialize_player_row(
    p: Any,
    *,
    include_ratings: bool = False,
    session: Optional[FranchiseSession] = None,
    _team: Optional[Any] = None,
) -> Dict[str, Any]:
    ident = getattr(p, "identity", None)
    ovr_f = getattr(p, "ovr", None)
    try:
        ov = float(ovr_f() if callable(ovr_f) else ovr_f)
    except Exception:
        ov = 0.0
    pid = str(getattr(p, "id", "") or "")
    hcm = int(getattr(ident, "height_cm", 0) or 0) if ident else 0
    row: Dict[str, Any] = {
        "player_id": pid,
        "name": str(getattr(ident, "name", None) or "?"),
        "position": str(getattr(getattr(ident, "position", None), "value", ident) if ident else "?"),
        "ovr": round(ov * 99, 1) if ov <= 1.5 else round(ov, 1),
        "morale": round(float(getattr(getattr(p, "psych", None), "morale", 0.5) or 0.5), 3),
        "age": int(getattr(ident, "age", 0) or 0),
        "nationality": str(getattr(ident, "birth_country", "") or ""),
        "height_cm": hcm,
        "height_display": height_cm_to_imperial(hcm) if hcm else "ΓÇö",
        "archetype": str(getattr(p, "archetype", "") or ""),
        "contract": {
            "salary": round(_player_cap_hit_millions(p), 3),
            "cap_hit": round(_player_cap_hit_millions(p), 3),
        },
    }
    gr = _get_live_injury_games_remaining(p)
    hstat = _get_player_health_status(p)
    if gr > 0 and hstat == "HEALTHY":
        hstat = "INJURED"
    injured = _is_player_live_injured(p)
    tier_guess = _get_live_injury_tier(p)
    if injured and tier_guess is None:
        tier_guess = "minor"
    tier = str(tier_guess or "").lower() if tier_guess else ""
    inj_label = _tier_human_label(tier) if injured else ""
    if gr > 0:
        avail = "Out"
    elif hstat == "DAY_TO_DAY":
        avail = "Day-to-day"
    else:
        avail = "Available"
    ret_est, ret_iso = ("", "")
    if session is not None and gr > 0:
        ret_est, ret_iso = _estimate_return_from_games_remaining(session, gr)
    elif gr > 0:
        ret_est = f"In {gr} games"
    disp_status = "HEALTHY"
    if injured:
        disp_status = "INJURED" if gr > 0 else hstat
    row.update(
        {
            "injury_status": disp_status,
            "health_status": disp_status,
            "injury": inj_label if injured else "",
            "injury_tier": tier if injured else "",
            "injury_type": tier if injured else "",
            "injury_games_remaining": int(gr),
            "games_remaining": int(gr),
            "is_injured": bool(injured),
            "availability_status": avail,
            "return_estimate": ret_est,
            "return_date": ret_iso,
        }
    )
    if include_ratings:
        row["rating_groups"] = _rating_groups_for_player(p)
    return row


def _rows_from_players_list(
    players: Any,
    *,
    include_ratings: bool = False,
    session: Optional[FranchiseSession] = None,
    team: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in players or []:
        if getattr(p, "retired", False):
            continue
        rows.append(_serialize_player_row(p, include_ratings=include_ratings, session=session, _team=team))
    rows.sort(key=lambda x: -float(x.get("ovr") or 0))
    return rows


def _serialize_development_leagues(blocks: Any) -> List[Dict[str, Any]]:
    """API-safe copy: league stores Player objects under each junior team."""
    out: List[Dict[str, Any]] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        teams_out: List[Dict[str, Any]] = []
        for tm in block.get("teams") or []:
            if not isinstance(tm, dict):
                continue
            teams_out.append(
                {
                    "team_id": tm.get("team_id"),
                    "name": tm.get("name"),
                    "players": _rows_from_players_list(tm.get("players")),
                }
            )
        out.append(
            {
                "league_code": block.get("league_code"),
                "league_name": block.get("league_name"),
                "teams": teams_out,
            }
        )
    return out


def _build_roster_browser(
    sim: Any,
    user_team_id: Optional[str] = None,
    franchise_session: Optional[FranchiseSession] = None,
) -> Dict[str, Any]:
    league = getattr(sim, "league", None)
    if league is None:
        return {
            "organizations": [],
            "free_agents": [],
            "overseas_free_agents": [],
            "development_leagues": [],
            "counts": {},
        }
    uid = str(user_team_id) if user_team_id is not None else ""
    orgs: List[Dict[str, Any]] = []
    for t in getattr(league, "teams", None) or []:
        raw_tid = getattr(t, "team_id", None)
        if raw_tid is None:
            raw_tid = getattr(t, "id", None)
        tid = str(raw_tid) if raw_tid is not None else ""
        is_user = bool(uid) and tid == uid
        orgs.append(
            {
                "team_id": tid,
                "name": _display_team(t),
                "nhl": _rows_from_players_list(
                    getattr(t, "roster", None),
                    include_ratings=is_user,
                    session=franchise_session,
                    team=t,
                ),
                "ahl": _rows_from_players_list(
                    getattr(t, "ahl_roster", None),
                    include_ratings=is_user,
                    session=franchise_session,
                    team=t,
                ),
                "echl": _rows_from_players_list(
                    getattr(t, "echl_roster", None),
                    include_ratings=is_user,
                    session=franchise_session,
                    team=t,
                ),
            }
        )
    try:
        from app.sim_engine.league_hierarchy_bootstrap import count_pool_players

        counts: Dict[str, Any] = dict(count_pool_players(league))
    except Exception:
        counts = {}
    return {
        "organizations": orgs,
        "free_agents": _rows_from_players_list(getattr(league, "free_agents", None), session=franchise_session)[
            :420
        ],
        "overseas_free_agents": _rows_from_players_list(
            getattr(league, "overseas_free_agents", None), session=franchise_session
        )[:260],
        "development_leagues": _serialize_development_leagues(getattr(league, "development_leagues", None)),
        "counts": counts,
    }


def build_draft_class_rankings(session: FranchiseSession, sim: Any) -> Dict[str, Any]:
    """Combined junior/prospect board for UI; ranks draft-age skaters in development leagues."""
    league = getattr(sim, "league", None)
    if league is None:
        return {"entries": [], "subtitle": "", "total": 0}
    prospects: List[Dict[str, Any]] = []
    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "")
        title = str(block.get("league_name") or "")
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if getattr(p, "retired", False):
                    continue
                ident = getattr(p, "identity", None)
                age = int(getattr(ident, "age", 99) or 99) if ident else 99
                if age > 20:
                    continue
                ovr_f = getattr(p, "ovr", None)
                try:
                    ov = float(ovr_f() if callable(ovr_f) else ovr_f)
                    ovr99 = round(ov * 99, 1) if ov <= 1.5 else round(ov, 1)
                except Exception:
                    ovr99 = 0.0
                pk = str(getattr(p, "id", "") or "")
                if not pk:
                    continue
                h = abs(hash(pk)) % 997
                scout = max(18.0, min(99.0, ovr99 + (h % 23) - 11))
                tier = ("A", "B", "C")[h % 3]
                prospects.append(
                    {
                        "key": pk,
                        "name": _name_str(p),
                        "position": _pos_str(p),
                        "age": age,
                        "true_ovr": ovr99,
                        "scout_grade": round(scout, 1),
                        "scout_tier": tier,
                        "league_code": code,
                        "league_name": title,
                        "_sort": ovr99,
                    }
                )
    prospects.sort(key=lambda x: -float(x["_sort"]))
    prev = dict(getattr(session, "draft_rank_prev", None) or {})
    entries: List[Dict[str, Any]] = []
    for i, row in enumerate(prospects[:320]):
        rank = i + 1
        key = str(row["key"])
        pr = int(prev.get(key, rank))
        delta = pr - rank
        if key not in prev:
            trend = "NEW"
        elif delta >= 2:
            trend = "UP"
        elif delta <= -2:
            trend = "DOWN"
        else:
            trend = "SAME"
        entries.append(
            {
                **{k: v for k, v in row.items() if k != "_sort"},
                "rank": rank,
                "rank_prev": pr,
                "rank_delta": abs(delta),
                "trend": trend,
            }
        )
    return {
        "entries": entries,
        "subtitle": f"Draft-age (Γëñ20) in dev leagues ┬╖ showing {len(entries)}",
        "total": len(prospects),
    }


def snapshot_draft_rank_prev(session: FranchiseSession, sim: Any) -> None:
    """Store current draft ranks so the next payload can show trend vs last advance."""
    board = build_draft_class_rankings(session, sim)
    entries = board.get("entries") or []
    session.draft_rank_prev = {str(e.get("key")): int(e.get("rank", i + 1)) for i, e in enumerate(entries) if e.get("key")}


def _normalize_storyline_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Unify franchise + SimEngine-style rows for the Storylines UI."""
    out: Dict[str, Any] = {}
    out["type"] = str(raw.get("type") or raw.get("tone") or "news")
    d = raw.get("date")
    if d is None or d == "":
        d = raw.get("calendar_iso") or raw.get("day") or ""
    out["date"] = d
    out["headline"] = str(raw.get("headline") or raw.get("title") or "").strip()
    out["team"] = str(raw.get("team") or raw.get("team_id") or raw.get("to_team_id") or "").strip()
    plist = raw.get("players")
    if isinstance(plist, list):
        out["players"] = [str(x) for x in plist if str(x).strip()]
    else:
        out["players"] = []
    out["priority"] = str(raw.get("priority") or "MEDIUM").upper()
    if raw.get("from_team_id") is not None:
        out["from_team_id"] = str(raw.get("from_team_id") or "")
    if raw.get("calendar_iso") is not None:
        out["calendar_iso"] = str(raw.get("calendar_iso") or "")
    if raw.get("tone") is not None:
        out["tone"] = raw.get("tone")
    if raw.get("details") is not None:
        out["details"] = str(raw.get("details") or "")
    if raw.get("team_id") is not None:
        out["team_id"] = str(raw.get("team_id") or "").strip()
    if raw.get("team_abbrev") is not None:
        out["team_abbrev"] = str(raw.get("team_abbrev") or "").strip()
    if raw.get("player_name") is not None:
        out["player_name"] = str(raw.get("player_name") or "").strip()
    if raw.get("summary") is not None:
        out["summary"] = str(raw.get("summary") or "").strip()
    if raw.get("text") is not None:
        out["text"] = str(raw.get("text") or "").strip()
    out["id"] = _stable_storyline_id(raw)
    action_opts = raw.get("action_options")
    if isinstance(action_opts, list):
        norm_opts: List[Dict[str, Any]] = []
        for idx, opt in enumerate(action_opts):
            if not isinstance(opt, dict):
                continue
            oid = str(opt.get("id") or f"opt_{idx}")
            norm_opts.append(
                {
                    "id": oid,
                    "label": str(opt.get("label") or oid.replace("_", " ").title()),
                    "effects": dict(opt.get("effects") or {}),
                    "effect_summary": str(opt.get("effect_summary") or "").strip(),
                }
            )
        if norm_opts:
            out["action_options"] = norm_opts
    if raw.get("cause") is not None:
        out["cause"] = str(raw.get("cause") or "").strip()
    if raw.get("effects") is not None:
        out["effects"] = dict(raw.get("effects") or {})
    if raw.get("effect_summary") is not None:
        out["effect_summary"] = str(raw.get("effect_summary") or "").strip()
    return out


def _stable_storyline_id(raw: Dict[str, Any]) -> str:
    explicit = str(raw.get("id") or "").strip()
    if explicit:
        return explicit
    parts = [
        str(raw.get("type") or raw.get("tone") or "storyline"),
        str(raw.get("calendar_iso") or raw.get("date") or raw.get("day") or ""),
        str(raw.get("team_id") or raw.get("team") or ""),
        str(raw.get("player_id") or raw.get("player_name") or ""),
        str(raw.get("headline") or raw.get("title") or raw.get("text") or raw.get("summary") or ""),
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8", "ignore")).hexdigest()[:12]
    return f"story_{digest}"


def _normalize_notification_payload(raw: Any, index: int) -> Dict[str, Any]:
    if isinstance(raw, dict):
        text = str(raw.get("text") or raw.get("headline") or raw.get("message") or raw.get("title") or "").strip()
        return {
            "id": str(raw.get("id") or f"notif_{index}_{uuid.uuid4().hex[:8]}"),
            "type": str(raw.get("type") or "news").lower(),
            "priority": str(raw.get("priority") or "LOW").upper(),
            "title": str(raw.get("title") or raw.get("headline") or "").strip(),
            "text": text,
            "date": raw.get("date") or raw.get("calendar_iso") or "",
            "team_id": str(raw.get("team_id") or raw.get("team") or "").strip(),
            "source": str(raw.get("source") or "franchise").strip(),
        }
    text = str(raw or "").strip()
    return {
        "id": f"notif_{index}_{uuid.uuid4().hex[:8]}",
        "type": "news",
        "priority": "LOW",
        "title": "",
        "text": text,
        "date": "",
        "team_id": "",
        "source": "franchise",
    }


def _get_player_health_status(player: Any) -> str:
    """Return INJURED | DAY_TO_DAY | HEALTHY from player.health if present."""
    h = getattr(player, "health", None)
    if h is None:
        return "HEALTHY"
    st: Any = None
    if isinstance(h, dict):
        st = h.get("injury_status")
    else:
        st = getattr(h, "injury_status", None)
    if st is None:
        return "HEALTHY"
    val = getattr(st, "value", None)
    raw = str(val if val is not None else getattr(st, "name", None) or st).lower().replace("-", "_")
    if "day_to_day" in raw or raw == "daytoday":
        return "DAY_TO_DAY"
    if raw in ("injured", "injury", "out"):
        return "INJURED"
    if raw in ("healthy", "health"):
        return "HEALTHY"
    if "injur" in raw and "healthy" not in raw:
        return "INJURED"
    return "HEALTHY"


def _get_live_injury_games_remaining(player: Any) -> int:
    return max(0, int(getattr(player, "_world_injury_games_remaining", 0) or 0))


def _get_live_injury_tier(player: Any) -> Optional[str]:
    t = getattr(player, "_world_injury_tier", None)
    if t is not None and str(t).strip():
        return str(t).strip().lower()
    return None


def _is_player_live_injured(player: Any) -> bool:
    if _get_live_injury_games_remaining(player) > 0:
        return True
    return _get_player_health_status(player) in ("INJURED", "DAY_TO_DAY")


def _find_latest_injury_log_for_player(session: FranchiseSession, player_id: str) -> Dict[str, Any]:
    pid = str(player_id or "")
    for inj in reversed(list(getattr(session, "injury_log_all", None) or [])):
        if not isinstance(inj, dict):
            continue
        if str(inj.get("player_id") or "") == pid:
            return dict(inj)
    return {}


def _estimate_return_from_games_remaining(session: FranchiseSession, games_remaining: int) -> Tuple[str, str]:
    gr = int(max(0, games_remaining))
    if gr <= 0:
        return "", ""
    estimate = f"In {gr} games"
    cal = getattr(session, "nhl_calendar", None) or []
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    counted = 0
    end_idx = cur
    for i in range(cur, min(len(cal), cur + 400)):
        row = cal[i]
        seg = str(row.get("segment") or "")
        if seg not in ("preseason", "regular", "playoffs"):
            continue
        if row.get("allows_games") is False:
            continue
        counted += 1
        if counted >= gr:
            end_idx = i
            break
    iso = ""
    if cal and 0 <= end_idx < len(cal):
        iso = str(cal[end_idx].get("iso") or "")
    return estimate, iso


def _tier_human_label(tier: Optional[str]) -> str:
    t = str(tier or "minor").lower()
    if t == "major":
        return "Significant injury"
    if t == "moderate":
        return "Moderate injury"
    return "Minor injury"


def _build_active_injury_row(session: FranchiseSession, player: Any, team: Any, team_id: str) -> Dict[str, Any]:
    pname = _name_str(player)
    pos = _pos_str(player)
    ab = _franchise_team_abbrev(team)
    gr = _get_live_injury_games_remaining(player)
    status = _get_player_health_status(player)
    if gr > 0 and status == "HEALTHY":
        status = "INJURED"
    tier_guess = _get_live_injury_tier(player)
    if tier_guess is None and _is_player_live_injured(player):
        tier_guess = "minor"
    tier = str(tier_guess or "minor").lower()
    meta = _find_latest_injury_log_for_player(session, str(getattr(player, "id", "") or ""))
    tid_s = str(team_id)
    pid_s = str(getattr(player, "id", "") or "")
    dkey = meta.get("calendar_day", meta.get("date", "unk"))
    stable_id = f"injury:{tid_s}:{pid_s}:{dkey}"
    cal_day = int(meta.get("calendar_day", meta.get("date", 0)) or 0)
    cal_iso = str(meta.get("calendar_iso") or "")
    if not cal_iso and cal_day >= 0:
        calrows = getattr(session, "nhl_calendar", None) or []
        if cal_day < len(calrows):
            cal_iso = str(calrows[cal_day].get("iso") or "")
    ret_est, ret_iso = _estimate_return_from_games_remaining(session, gr)
    inj_label = _tier_human_label(tier)
    desc = f"{pname} ({ab}): {inj_label}, {gr} games remaining."
    games_initial = int(meta.get("games_initial", meta.get("games", gr)) or gr)
    return {
        "id": stable_id,
        "player_id": pid_s,
        "player_name": pname,
        "team_id": tid_s,
        "team_abbr": ab,
        "team_abbrev": ab,
        "position": pos,
        "status": status,
        "injury_status": status,
        "health_status": status,
        "injury": inj_label,
        "injury_type": tier,
        "tier": tier,
        "severity": tier,
        "games_remaining": gr,
        "days_remaining": gr,
        "duration": f"{gr} games" if gr else "0 games",
        "return_estimate": ret_est,
        "return_date": ret_iso,
        "calendar_day": cal_day,
        "calendar_iso": cal_iso,
        "date": cal_iso or (str(cal_day) if cal_day else ""),
        "description": desc,
        "source": "live_player_state",
        "games_initial": games_initial,
    }


def _build_injuries_payload(session: FranchiseSession, *, limit: int = 200) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    utid = str(getattr(session, "user_team_id", "") or "")
    for tid, tm in (getattr(session, "team_by_id", None) or {}).items():
        tid_s = str(tid)
        for p in getattr(tm, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            if not _is_player_live_injured(p):
                continue
            rows.append(_build_active_injury_row(session, p, tm, tid_s))

    def _sort_key(r: Dict[str, Any]) -> Tuple[int, int, str]:
        user_first = 0 if utid and str(r.get("team_id") or "") == utid else 1
        gr = -int(r.get("games_remaining") or 0)
        name = str(r.get("player_name") or "")
        return (user_first, gr, name)

    rows.sort(key=_sort_key)
    return rows[: int(limit)]


def _build_injury_history_payload(session: FranchiseSession, *, limit: int = 80) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for inj in list(getattr(session, "injury_log_all", None) or [])[-int(limit) :]:
        if not isinstance(inj, dict):
            continue
        row = dict(inj)
        row["historical"] = True
        row["source"] = "injury_log"
        tid = str(inj.get("team_id") or "")
        pid = str(inj.get("player_id") or "")
        d = inj.get("date", inj.get("calendar_day", "unk"))
        row.setdefault("id", str(inj.get("id") or f"injlog:{tid}:{pid}:{d}"))
        ta = str(inj.get("team_abbr") or inj.get("team_abbrev") or "")
        row["team_abbr"] = ta
        row["team_abbrev"] = ta
        out.append(row)
    return out


def _storyline_dedupe_key(ev: Dict[str, Any]) -> str:
    return f"{ev.get('type')}|{ev.get('date')}|{str(ev.get('headline') or '')[:200]}"


def _merge_simengine_league_news_into_storylines(session: FranchiseSession) -> None:
    """One-time merge of latest SimEngine league season news_events into the franchise feed."""
    sim = getattr(session, "sim", None)
    if sim is None:
        return
    hist = list(getattr(sim, "league_history", None) or [])
    if not hist:
        return
    last = hist[-1]
    sig = (int(getattr(last, "year", 0) or 0), id(last))
    if getattr(session, "_merged_engine_news_sig", None) == sig:
        return
    setattr(session, "_merged_engine_news_sig", sig)
    nev = list(getattr(last, "news_events", None) or [])
    for raw in nev[-150:]:
        if not isinstance(raw, dict):
            continue
        _record_storyline(session, raw)


def _record_storyline(session: FranchiseSession, event: Dict[str, Any]) -> None:
    ev = _normalize_storyline_payload(event if isinstance(event, dict) else {})
    if not ev.get("headline"):
        return
    dq = getattr(session, "_storyline_dedupe", None)
    if dq is None:
        dq = []
        session._storyline_dedupe = dq
    dk = _storyline_dedupe_key(ev)
    if dk in dq:
        return
    dq.append(dk)
    if len(dq) > 500:
        session._storyline_dedupe = dq[-400:]
    if getattr(session, "storyline_events", None) is None:
        session.storyline_events = []
    session.storyline_events.append(ev)
    if len(session.storyline_events) > 400:
        session.storyline_events = session.storyline_events[-400:]


def _storyline_choices_payload(session: FranchiseSession) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in list(getattr(session, "pending_decisions", None) or []):
        kind = str(d.get("kind") or "")
        if kind in ("franchise_critical_notice",):
            continue
        opts = list(d.get("options") or [])
        if not opts:
            continue
        meta = dict(d.get("meta") or {})
        storyline_id = str(meta.get("storyline_id") or d.get("storyline_id") or d.get("id") or "")
        if not storyline_id:
            storyline_id = str(d.get("id") or "")
        action_options: List[Dict[str, Any]] = []
        for idx, opt in enumerate(opts):
            oid = str(opt.get("id") or f"opt_{idx}")
            action_options.append(
                {
                    "id": oid,
                    "label": str(opt.get("label") or oid.replace("_", " ").title()),
                    "effects": dict(opt.get("effects") or {}),
                    "effect_summary": str(opt.get("effect_summary") or "").strip(),
                }
            )
        rows.append(
            {
                "storyline_id": storyline_id,
                "decision_id": str(d.get("id") or ""),
                "kind": kind,
                "priority": str(d.get("priority") or "MEDIUM").upper(),
                "title": str(d.get("title") or "Storyline choice"),
                "description": str(d.get("description") or ""),
                "team_id": str(meta.get("team_id") or meta.get("team") or ""),
                "player_id": str(meta.get("player_id") or ""),
                "player_name": str(meta.get("player_name") or ""),
                "cause": str(meta.get("cause") or ""),
                "action_options": action_options,
            }
        )
    return rows


def _franchise_enqueue_critical_notice(
    session: FranchiseSession, *, title: str, description: str, source: str
) -> None:
    if any(
        str(d.get("kind") or "") == "franchise_critical_notice" and str((d.get("meta") or {}).get("source") or "") == source
        for d in (session.pending_decisions or [])
    ):
        return
    dec_id = f"dec_{uuid.uuid4().hex[:12]}"
    session.pending_decisions.insert(
        0,
        {
            "id": dec_id,
            "kind": "franchise_critical_notice",
            "priority": "CRITICAL",
            "title": title,
            "description": description,
            "options": [{"id": "ack", "label": "Acknowledge"}],
            "meta": {"source": source},
        },
    )


def _franchise_daily_league_tick(session: FranchiseSession, calendar_idx: int) -> None:
    """Waivers / trades / call-ups (SimEngine helpers) before the day's games ΓÇö mutates league rosters."""
    if int(getattr(session, "_last_socio_tick_idx", -99)) == int(calendar_idx):
        return
    sim = session.sim
    league = getattr(sim, "league", None)
    teams = list(getattr(league, "teams", None) or [])
    st = session.standings
    if not teams or st is None:
        return
    rng = sim.rng
    utid = str(session.user_team_id)
    max_d = max(40, int(getattr(session, "nhl_regular_season_last_index", 0) or 0))
    news_tmp: List[Dict[str, Any]] = []
    ctr: Dict[str, int] = {"trade_executions": 0, "waiver_claims": 0, "major_injuries": 0}
    try:
        sim._standings_sync_team_metrics(st, teams)
        sim._season_daily_socio_economics(rng, int(calendar_idx), max_d, st, teams, news_tmp, ctr)
    except Exception:
        return
    iso = ""
    cal = getattr(session, "nhl_calendar", None) or []
    if 0 <= int(calendar_idx) < len(cal):
        iso = str(cal[int(calendar_idx)].get("iso") or "")
    for ev in news_tmp:
        ev2 = dict(ev)
        ev2.setdefault("priority", "MEDIUM")
        ev2["date"] = int(ev2.get("date") or calendar_idx)
        ev2["calendar_iso"] = iso
        _record_storyline(session, ev2)
        if str(ev2.get("type")) == "trade" and utid:
            ft = str(ev2.get("from_team_id") or "")
            tt = str(ev2.get("team") or "")
            if utid in (ft, tt):
                _franchise_enqueue_critical_notice(
                    session,
                    title="League office: trade register",
                    description=str(ev2.get("headline") or "A trade involving your club was processed."),
                    source=f"trade_{calendar_idx}_{ft}_{tt}",
                )
    setattr(session, "_last_socio_tick_idx", int(calendar_idx))


def _franchise_fanout_player_storylines(session: FranchiseSession, calendar_idx: int, day_meta: Dict[str, Any]) -> None:
    from app.sim_engine.franchise.common import _franchise_fanout_player_storylines as _fanout  # noqa: WPS433

    _fanout(session, calendar_idx, day_meta)


def _maybe_roll_storyline_arc(session: FranchiseSession, day_meta: Dict[str, Any], rng: random.Random) -> None:
    """Rare narrative beats (normal + wacky) ΓÇö morale nudge on user club when applicable."""
    if rng.random() > 0.038:
        return
    iso = str(day_meta.get("iso") or "")
    utid = str(session.user_team_id)
    pool = [
        ("normal", "Player breakout buzz", "Scouts league-wide note a rising star on your depth chart."),
        ("normal", "Underdog surge", "National pundits spotlight your club's improved underlying numbers."),
        ("wacky", "Locker-room scuffle", "Practice intensity boiled over ΓÇö coaches reset the room."),
        ("wacky", "Social media storm", "A viral clip from the airport spun into a week of distraction."),
        ("wacky", "League discipline review", "DoPS is reviewing a borderline hit from last game."),
    ]
    tone, head, sub = rng.choice(pool)
    ev = {
        "type": "storyline",
        "tone": tone,
        "date": iso or "?",
        "headline": head,
        "team": utid,
        "players": [],
        "priority": "HIGH" if tone == "wacky" else "MEDIUM",
    }
    _record_storyline(session, ev)
    if tone == "wacky" and rng.random() < 0.55:
        user_tm = session.team_by_id.get(utid)
        if user_tm is not None:
            for p in getattr(user_tm, "roster", None) or []:
                if getattr(p, "psych", None) is None:
                    continue
                p.psych.role_satisfaction = _clamp(float(p.psych.role_satisfaction) - 0.02)
                break
    session.notifications.append(f"Storyline ({tone}): {head}")


def _simulate_franchise_slot(session: FranchiseSession, slot: Any) -> Tuple[Optional[str], Optional[str]]:
    """Simulate one scheduled league game. Returns (user_summary_line_or_none, league_line_or_none)."""
    sim = session.sim
    teams = list(sim.league.teams)
    r = sim.rng
    user_tid = str(session.user_team_id)

    # team_by_id is keyed by str(team_id); slots may carry int ids (dataclasses do not enforce types).
    hid = _safe_slot_team_id(slot, "home_id")
    aid = _safe_slot_team_id(slot, "away_id")
    home = session.team_by_id.get(hid)
    away = session.team_by_id.get(aid)
    if home is None or away is None:
        return None, None
    d = int(slot.day)
    cal = getattr(session, "nhl_calendar", None) or []
    cal_iso = ""
    if 0 <= int(d) < len(cal):
        cal_iso = str(cal[int(d)].get("iso") or "")

    h_goal = _goalie_availability_status(home)
    a_goal = _goalie_availability_status(away)
    if int(h_goal["total"]) <= 0 or int(a_goal["total"]) <= 0:
        _fr_dbg(f"goalie availability failure on day {d}: {hid} total={h_goal['total']} {aid} total={a_goal['total']}")
        _franchise_enqueue_critical_notice(
            session,
            title="Roster integrity issue",
            description="A scheduled game has no listed goalie on one side. Resolve roster integrity before advancing.",
            source=f"goalie-missing:{hid}:{aid}:{d}",
        )
        raise RuntimeError("Cannot simulate game without at least one goalie on each roster.")
    if bool(h_goal["forced_injured_start"]) or bool(a_goal["forced_injured_start"]):
        forced_team = hid if bool(h_goal["forced_injured_start"]) else aid
        forced_tm = home if forced_team == hid else away
        forced_name = _display_team(forced_tm)
        _fr_dbg(f"forced injured goalie start: day={d} team={forced_team} ({forced_name})")
        _record_storyline(
            session,
            {
                "type": "injury",
                "priority": "HIGH" if forced_team == user_tid else "MEDIUM",
                "headline": f"{forced_name} emergency goalie start",
                "team_id": forced_team,
                "team": forced_team,
                "calendar_iso": cal_iso,
                "date": int(d),
                "cause": "All healthy goalies unavailable due to injuries.",
                "effects": {"goalie_availability_delta": -1, "stability_delta": -2},
                "effect_summary": "Emergency start required; elevated goals-against volatility.",
            },
        )

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

        h_scale = max(0.93, min(1.07, hm * hc * hf * hmr)) * float(sim._roster_injury_depth_penalty(home))
        a_scale = max(0.93, min(1.07, am * ac * af * amr)) * float(sim._roster_injury_depth_penalty(away))

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
            light_mode=bool(getattr(session, "_light_game_stat_accumulation", False)),
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

        if getattr(session, "injuries_enabled", True):
            for tm in (home, away):
                ev = world_injuries.maybe_injure_roster_subset(
                    tm, r, session.chaos_index, max_checks=8
                )
                tid_inj = next((str(v) for v in (getattr(tm, "team_id", None), getattr(tm, "id", None)) if v is not None), "")
                abbrev = _franchise_team_abbrev(tm)
                slot_user_game = user_tid in (hid, aid)
                for label, tier, games, pid in ev:
                    _franchise_log_injury_and_ui(
                        session,
                        player_id=pid,
                        player_name=label,
                        team_id=tid_inj,
                        team_abbrev=abbrev,
                        tier=str(tier),
                        games=int(games),
                        injury_type=str(tier),
                        calendar_day=int(d),
                        calendar_iso=cal_iso,
                        game_day_injury=bool(slot_user_game and tid_inj.lower() == user_tid.lower()),
                    )
    else:
        _, nh = sim._identity_runner_strength_noise_factors(home)
        _, na = sim._identity_runner_strength_noise_factors(away)
        id_noise = 0.5 * (nh + na)
        h_inj = float(sim._roster_injury_depth_penalty(home))
        a_inj = float(sim._roster_injury_depth_penalty(away))
        hg, ag, ot = sim._simulate_game(
            r,
            home,
            away,
            session.strength_map,
            home_strength_scale=h_inj,
            away_strength_scale=a_inj,
            noise_scale=id_noise,
            light_mode=bool(getattr(session, "_light_game_stat_accumulation", False)),
        )

        if world_injuries is not None:
            for tm in (home, away):
                for pl in getattr(tm, "roster", None) or []:
                    if int(getattr(pl, "_world_injury_games_remaining", 0) or 0) > 0:
                        world_injuries.tick_games_missed(pl)
            if getattr(session, "injuries_enabled", True):
                for tm in (home, away):
                    ev = world_injuries.maybe_injure_roster_subset(
                        tm, r, session.chaos_index, max_checks=8
                    )
                    tid_inj = next((str(v) for v in (getattr(tm, "team_id", None), getattr(tm, "id", None)) if v is not None), "")
                    abbrev = _franchise_team_abbrev(tm)
                    slot_user_game = user_tid in (hid, aid)
                    for label, tier, games, pid in ev:
                        _franchise_log_injury_and_ui(
                            session,
                            player_id=pid,
                            player_name=label,
                            team_id=tid_inj,
                            team_abbrev=abbrev,
                            tier=str(tier),
                            games=int(games),
                            injury_type=str(tier),
                            calendar_day=int(d),
                            calendar_iso=cal_iso,
                            game_day_injury=bool(slot_user_game and tid_inj.lower() == user_tid.lower()),
                        )

        hg, ag = _validate_final_game_result_payload(
        home_id=hid,
        away_id=aid,
        home_goals=hg,
        away_goals=ag,
        calendar_day=d,
    )

    session.standings.record_game(slot.home_id, slot.away_id, hg, ag, overtime=ot)

    _accumulate_franchise_game_stats(
        session,
        home=home,
        away=away,
        hid=hid,
        aid=aid,
        hg=int(hg),
        ag=int(ag),
        ot=bool(ot),
        calendar_day=d,
        rng=r,
        calendar_iso=cal_iso,
    )
    

    hn = (_display_team(home) or "?")[:24]
    an = (_display_team(away) or "?")[:24]
    league_line = f"{hn} {int(hg)}-{int(ag)} {an}{' OT' if ot else ''}"

    user_line: Optional[str] = None
    if hid == user_tid or aid == user_tid:
        opp = away if hid == user_tid else home
        won = (hg > ag) if hid == user_tid else (ag > hg)
        wl = "W" if won else "L"
        gs = f"{hg}-{ag}"
        if ot:
            gs += " OT"
        user_line = f"{wl} vs {_display_team(opp)} ({gs}) ΓÇö calendar day {d}"

    return user_line, league_line


def _simulate_slots_for_day(
    session: FranchiseSession,
    calendar_day: int,
    slots: List[Any],
) -> Tuple[List[str], List[str]]:
    """
    Simulate every scheduled slot for one calendar day.

    After simulation, verify that every slot generated one real completed result.
    This prevents silent partial days where standings/games drift apart.
    """
    lines: List[str] = []
    league_lines: List[str] = []

    expected_keys: set = set()

    for slot in slots or []:
        hid = _safe_slot_team_id(slot, "home_id")
        aid = _safe_slot_team_id(slot, "away_id")

        if hid and aid:
            expected_keys.add((hid, aid))

        ul, ll = _simulate_franchise_slot(session, slot)

        if ul:
            lines.append(ul)

        if ll:
            league_lines.append(ll)

    # Verify result store has a valid final for every scheduled slot.
    saved_for_day = [
        g
        for g in (getattr(session, "game_results", None) or [])
        if isinstance(g, dict) and _game_result_calendar_index(g) == int(calendar_day)
    ]

    completed_keys: set = set()

    for g in saved_for_day:
        if not _saved_game_is_final(g):
            continue

        hid = str(g.get("home_id") or "")
        aid = str(g.get("away_id") or "")

        try:
            _validate_final_game_result_payload(
                home_id=hid,
                away_id=aid,
                home_goals=g.get("home_goals", g.get("home_score")),
                away_goals=g.get("away_goals", g.get("away_score")),
                calendar_day=int(calendar_day),
            )
        except ValueError:
            continue

        completed_keys.add((hid, aid))

    missing = sorted(expected_keys - completed_keys)

    if missing:
        raise RuntimeError(
            f"Game result integrity error on calendar day {calendar_day}: "
            f"{len(missing)} scheduled game(s) did not produce a valid final result. "
            f"First missing: {missing[0][0]} vs {missing[0][1]}"
        )

    return lines, league_lines


def _purge_retired_from_extra_pools(session: FranchiseSession, player: Any) -> None:
    league = getattr(session.sim, "league", None)
    if league is None:
        return
    for attr in ("free_agents", "overseas_free_agents"):
        lst = getattr(league, attr, None)
        if not lst:
            continue
        try:
            if player in lst:
                lst.remove(player)
        except Exception:
            pass
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            pls = tm.get("players")
            if isinstance(pls, list) and player in pls:
                try:
                    pls.remove(player)
                except Exception:
                    pass
    for tm in getattr(league, "teams", None) or []:
        for attr in ("ahl_roster", "echl_roster"):
            lst = getattr(tm, attr, None)
            if not lst:
                continue
            try:
                if player in lst:
                    lst.remove(player)
            except Exception:
                pass


def _depth_pool_progression_tick(session: FranchiseSession) -> None:
    """Periodic full progression pass on non-NHL depth (prospects, overseas, FA, minors)."""
    from app.sim_engine.progression import run_player_progression

    league = getattr(session.sim, "league", None)
    if league is None:
        return
    rng = session.sim.rng
    pool: List[Any] = []
    for p in getattr(league, "free_agents", None) or []:
        if not getattr(p, "retired", False):
            pool.append(p)
    for p in getattr(league, "overseas_free_agents", None) or []:
        if not getattr(p, "retired", False):
            pool.append(p)
    for tm in getattr(league, "teams", None) or []:
        for p in getattr(tm, "ahl_roster", None) or []:
            if not getattr(p, "retired", False):
                pool.append(p)
        for p in getattr(tm, "echl_roster", None) or []:
            if not getattr(p, "retired", False):
                pool.append(p)
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if not getattr(p, "retired", False):
                    pool.append(p)
    if not pool:
        return
    rng.shuffle(pool)
    for p in pool[: min(72, len(pool))]:
        try:
            _, retired = run_player_progression(p, rng)
            if retired:
                setattr(p, "retired", True)
                _purge_retired_from_extra_pools(session, p)
        except Exception:
            pass


def _finalize_regular_calendar_day(
    session: FranchiseSession,
    day_meta: Dict[str, Any],
    user_lines: List[str],
    league_lines: List[str],
    *,
    day_ordinal: int,
) -> None:
    """Timeline, GM prompts, and league-wide off-day development after a calendar slate completes."""
    session.calendar_days_finished = int(getattr(session, "calendar_days_finished", 0) or 0) + 1

    iso = str(day_meta.get("iso") or "")
    ui_phase = str(day_meta.get("ui_phase") or "")
    total_reg_days = int(getattr(session, "nhl_regular_season_last_index", 0) or 0) + 1
    day_label = f"{iso} ┬╖ {ui_phase} ┬╖ league day {int(day_ordinal)} / {total_reg_days}"
    session.timeline.append(day_label)
    if league_lines:
        cap = 10
        bits = league_lines[:cap]
        tail = len(league_lines) - len(bits)
        slate = " ┬╖ ".join(bits)
        if tail > 0:
            slate += f" ΓÇª +{tail} more"
        session.timeline.append(f"League: {slate}")
    for ln in user_lines[:6]:
        session.timeline.append(ln)
    utid = str(session.user_team_id)
    user_tm = session.team_by_id.get(utid) or session.team_by_id.get(session.user_team_id)
    uname = (_display_team(user_tm) or "Your club")[:28]
    if not user_lines:
        if league_lines:
            session.timeline.append(f"{uname}: no game today.")
        else:
            session.timeline.append("League: quiet day (no games on the calendar).")
    if session.standings:
        rr = session.standings.records.get(utid) or session.standings.records.get(session.user_team_id)
        if rr is not None:
            session.timeline.append(
                f"{uname} record: {getattr(rr, 'wins', 0)}-{getattr(rr, 'losses', 0)}-{getattr(rr, 'otl', 0)} "
                f"({getattr(rr, 'points', 0)} pts)"
            )
    if len(session.timeline) > 200:
        session.timeline = session.timeline[-200:]

    just_idx = int(session.calendar_cursor) - 1
    try:
        from app.sim_engine.franchise.storyline_engine import franchise_record_data_storylines  # noqa: WPS433

        franchise_record_data_storylines(session, just_idx, day_meta, rng=session.sim.rng)
    except Exception:
        pass
    _franchise_fanout_player_storylines(session, just_idx, day_meta)

    _maybe_enqueue_post_day_decisions(session, user_lines)
    try:
        from app.sim_engine.league_hierarchy_bootstrap import tick_extra_league_development

        tick_extra_league_development(session.sim, session.sim.rng)
    except Exception:
        pass
    if int(session.calendar_days_finished) % 5 == 0:
        _depth_pool_progression_tick(session)

    _maybe_enqueue_wjc_loan_decisions(session, day_meta)
    _maybe_enqueue_showcase_popups(session, day_meta)
    _maybe_roll_storyline_arc(session, day_meta, session.sim.rng)


def _split_preseason_from_regular_if_needed(session: FranchiseSession, day_meta: Dict[str, Any]) -> None:
    """At first regular-season day, snapshot preseason stats and reset regular-season counters."""
    if str(day_meta.get("segment") or "") != "regular":
        return
    if bool(getattr(session, "_regular_stats_split_done", False)):
        return
    try:
        # Keep preseason snapshots available for UI or later diagnostics.
        session.preseason_standings_snapshot = session.standings
    except Exception:
        pass
    try:
        session.preseason_player_stats_snapshot = dict(getattr(session, "player_season_stats", None) or {})
    except Exception:
        session.preseason_player_stats_snapshot = {}
    try:
        session.preseason_game_results_snapshot = list(getattr(session, "game_results", None) or [])
    except Exception:
        session.preseason_game_results_snapshot = []

    # Start fresh regular-season records.
    try:
        teams = list(getattr(getattr(session, "sim", None), "league", None).teams)
        session.standings = StandingsTable(teams)
    except Exception:
        pass
    session.player_season_stats = {}
    session.game_results = []
    session.timeline.append("REGULAR SEASON: preseason stats archived; regular-season records reset.")
    setattr(session, "_regular_stats_split_done", True)


def dismiss_franchise_popups(session: FranchiseSession, popup_ids: List[str]) -> None:
    """Remove dismissed popups from the pending queue (archive is unchanged)."""
    if not popup_ids:
        return
    drop = {str(x).strip() for x in popup_ids if str(x).strip()}
    session.pending_ui_popups = [p for p in (session.pending_ui_popups or []) if str(p.get("id") or "") not in drop]


def _append_showcase_popup(session: FranchiseSession, dedupe_key: str, payload: Dict[str, Any]) -> None:
    if dedupe_key in session.shown_event_keys:
        return
    session.shown_event_keys.add(dedupe_key)
    pid = f"pop_{uuid.uuid4().hex[:12]}"
    body = dict(payload)
    body["id"] = pid
    session.pending_ui_popups.append(body)
    arch = list(getattr(session, "showcase_archive", None) or [])
    arch.append(dict(body))
    session.showcase_archive = arch[-48:]


def _rng_for_event(session: FranchiseSession, label: str) -> random.Random:
    base = abs(hash(f"{session.session_id}|{label}|{int(session.season_calendar_year)}")) % (2**31 - 1)
    return random.Random(int(base) or 1)


def _wjc_calendar_dates(season_y: int) -> List[date]:
    """Dec 26 (season_y) through Jan 5 (season_y+1), inclusive."""
    out: List[date] = []
    d = date(season_y, 12, 26)
    end = date(season_y + 1, 1, 5)
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _wjc_day_index_for_iso(iso: str, season_y: int) -> Optional[int]:
    try:
        y, m, dd = (int(x) for x in iso.split("-"))
        cur = date(y, m, dd)
    except (TypeError, ValueError):
        return None
    for i, d in enumerate(_wjc_calendar_dates(season_y)):
        if d == cur:
            return i
    return None


def _wjc_country_for_birth(rng: random.Random, birth_country: str) -> str:
    bc = str(birth_country or "").strip().lower()
    pairs = [
        (("canada", "can"), "CAN"),
        (("united states", "u.s", "usa", "america"), "USA"),
        (("sweden", "sverige"), "SWE"),
        (("finland", "suomi"), "FIN"),
        (("czech", "czechia"), "CZE"),
        (("slovak", "slovakia"), "SVK"),
        (("germany", "deutsch"), "GER"),
        (("latvia", "latv"), "LAT"),
        (("russia", "╤Ç╨╛╤ü╤ü", "rossiya"), "RUS"),
        (("kazakh", "╥¢╨░╨╖╨░"), "KAZ"),
        (("denmark", "norway", "austria", "switzerland"), "GER"),
    ]
    for hints, code in pairs:
        if any(h in bc for h in hints):
            return code
    return ""


def _wjc_countries_meta() -> List[Tuple[str, str]]:
    """National programs only (no NHL clubs)."""
    return [
        ("CAN", "Canada"),
        ("USA", "United States"),
        ("RUS", "Russia"),
        ("FIN", "Finland"),
        ("SWE", "Sweden"),
        ("GER", "Germany"),
        ("CZE", "Czechia"),
        ("LAT", "Latvia"),
        ("KAZ", "Kazakhstan"),
    ]


def _wjc_country_label(code: str) -> str:
    for c, lab in _wjc_countries_meta():
        if c == code:
            return lab
    return str(code or "?")


def _wjc_pool_codes() -> List[str]:
    return [c for c, _ in _wjc_countries_meta()]


def _country_in_wjc_pool(code: str) -> bool:
    return str(code or "") in set(_wjc_pool_codes())


def _player_ovr01(p: Any) -> float:
    ovr_f = getattr(p, "ovr", None)
    try:
        ov = float(ovr_f() if callable(ovr_f) else ovr_f)
    except Exception:
        ov = 0.55
    if ov > 1.5:
        ov = ov / 99.0
    return float(ov)


def _collect_user_wjc_prospects(session: FranchiseSession, rng: random.Random) -> List[Dict[str, Any]]:
    """U20 on your AHL affiliate, plus NHL U20 only if the user loaned them to their WJC country."""
    out: List[Dict[str, Any]] = []
    ut = session.team_by_id.get(str(session.user_team_id))
    if ut is None:
        return out
    loans = getattr(session, "wjc_nhl_u20_loan", None) or {}

    def _row(p: Any, *, roster: str) -> None:
        if getattr(p, "retired", False):
            return
        ident = getattr(p, "identity", None)
        if ident is None:
            return
        age = int(getattr(ident, "age", 99) or 99)
        if age > 20:
            return
        pid = str(getattr(p, "id", "") or "")
        nm = str(getattr(ident, "name", None) or "?")
        bc = str(getattr(ident, "birth_country", "") or "")
        code = _wjc_country_for_birth(rng, bc)
        if not _country_in_wjc_pool(code):
            # If the country is not in this year's tournament field, player does not participate.
            return
        lab = _wjc_country_label(code)
        ov = _player_ovr01(p)
        cut = 0.62 + 0.08 * rng.random()
        made = bool(ov >= cut or rng.random() < 0.28)
        note = (
            f"Named to {lab} U20 national roster."
            if made
            else f"Released from {lab} U20 national camp before the tournament."
        )
        out.append(
            {
                "player_id": pid,
                "name": nm,
                "age": age,
                "nationality": bc,
                "wjc_country": code,
                "wjc_country_label": lab,
                "made_wjc_team": made,
                "note": note,
                "roster": roster,
            }
        )

    for p in getattr(ut, "ahl_roster", None) or []:
        _row(p, roster="AHL")

    for p in getattr(ut, "roster", None) or []:
        if not loans.get(str(getattr(p, "id", "") or ""), False):
            continue
        _row(p, roster="NHL (loaned)")

    out.sort(key=lambda x: (-int(x.get("made_wjc_team") or 0), str(x.get("roster") or ""), str(x.get("name") or "")))
    return out


def _rr_standings_from_slice(codes: List[str], label_by: Dict[str, str], rr_slice: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    st: Dict[str, Dict[str, int]] = {
        c: {"gp": 0, "w": 0, "otl": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0} for c in codes
    }
    for g in rr_slice:
        home = str(g["home"])
        away = str(g["away"])
        hg = int(g["home_goals"])
        ga = int(g["away_goals"])
        st[home]["gp"] += 1
        st[away]["gp"] += 1
        st[home]["gf"] += hg
        st[home]["ga"] += ga
        st[away]["gf"] += ga
        st[away]["ga"] += hg
        if hg > ga:
            st[home]["w"] += 1
            st[away]["l"] += 1
            st[home]["pts"] += 2
        else:
            st[away]["w"] += 1
            st[home]["l"] += 1
            st[away]["pts"] += 2

    def _row_key(c: str) -> Tuple[int, int, int, int]:
        s = st[c]
        return (-int(s["pts"]), -(int(s["w"]) - int(s["l"])), -(int(s["gf"]) - int(s["ga"])), -int(s["gf"]))

    ordered = sorted(codes, key=_row_key)
    rows: List[Dict[str, Any]] = []
    for rank, c in enumerate(ordered, start=1):
        s = st[c]
        rows.append(
            {
                "place": rank,
                "code": c,
                "label": label_by[c],
                "gp": s["gp"],
                "w": s["w"],
                "otl": s["otl"],
                "l": s["l"],
                "gf": s["gf"],
                "ga": s["ga"],
                "pts": s["pts"],
            }
        )
    return rows


def _simulate_wjc_national_bundle(rng: random.Random) -> Dict[str, Any]:
    """Full U20 worlds ΓÇö national teams only (deterministic from rng)."""
    countries = _wjc_countries_meta()
    codes = [c for c, _ in countries]
    label_by = {c: lab for c, lab in countries}

    rr_games: List[Dict[str, Any]] = []
    for i, hi in enumerate(codes):
        for aj in codes[i + 1 :]:
            home, away = (hi, aj) if rng.random() < 0.5 else (aj, hi)
            hg = rng.randint(1, 5)
            ga = rng.randint(1, 5)
            if hg == ga:
                ga = min(6, ga + 1)
                if hg == ga:
                    hg = max(1, hg - 1)
            rr_games.append(
                {
                    "home": home,
                    "away": away,
                    "home_goals": hg,
                    "away_goals": ga,
                    "home_label": label_by[home],
                    "away_label": label_by[away],
                }
            )

    st_full = _rr_standings_from_slice(codes, label_by, rr_games)
    code_order = [r["code"] for r in st_full]

    def _play_pair(a: str, b: str, label: str, lb: Dict[str, str]) -> Dict[str, Any]:
        home, away = (a, b) if rng.random() < 0.5 else (b, a)
        hg = rng.randint(2, 5)
        ga = rng.randint(1, 4)
        if hg == ga:
            hg = min(6, hg + 1)
        w = home if hg > ga else away
        l = away if hg > ga else home
        return {
            "round": label,
            "home": home,
            "away": away,
            "home_goals": hg,
            "away_goals": ga,
            "winner": w,
            "loser": l,
            "home_label": lb[home],
            "away_label": lb[away],
            "winner_label": lb[w],
            "loser_label": lb[l],
        }

    playoff_pool = code_order[:8] if len(code_order) >= 8 else list(code_order)
    while len(playoff_pool) < 8 and playoff_pool:
        playoff_pool.append(playoff_pool[-1])
    s1, s2, s3, s4, s5, s6, s7, s8 = (
        playoff_pool[0],
        playoff_pool[1],
        playoff_pool[2],
        playoff_pool[3],
        playoff_pool[4],
        playoff_pool[5],
        playoff_pool[6],
        playoff_pool[7],
    )
    qf = [
        _play_pair(s1, s8, "Quarterfinal", label_by),
        _play_pair(s2, s7, "Quarterfinal", label_by),
        _play_pair(s3, s6, "Quarterfinal", label_by),
        _play_pair(s4, s5, "Quarterfinal", label_by),
    ]
    w_qf = [g["winner"] for g in qf]
    sf = [_play_pair(w_qf[0], w_qf[1], "Semifinal", label_by), _play_pair(w_qf[2], w_qf[3], "Semifinal", label_by)]
    w_sf = [g["winner"] for g in sf]
    l_sf = [g["loser"] for g in sf]
    bronze = _play_pair(l_sf[0], l_sf[1], "Bronze", label_by)
    gold = _play_pair(w_sf[0], w_sf[1], "Gold Medal", label_by)
    medals = {
        "gold": gold["winner"],
        "silver": gold["loser"],
        "bronze": bronze["winner"],
        "fourth": bronze["loser"],
    }
    medal_labels = {k: label_by.get(v, v) for k, v in medals.items()}

    return {
        "countries": [{"code": c, "label": lab} for c, lab in countries],
        "rr_games": rr_games,
        "playoffs": {"quarterfinals": qf, "semifinals": sf, "bronze": bronze, "gold": gold},
        "medals": medals,
        "medal_labels": medal_labels,
    }


def _ensure_wjc_tournament_bundle(session: FranchiseSession) -> None:
    sy = int(session.season_calendar_year)
    b = getattr(session, "wjc_tournament_bundle", None)
    if isinstance(b, dict) and int(b.get("season_sy", -1)) == sy:
        return
    rng = _rng_for_event(session, f"wjc_bundle_{sy}")
    core = _simulate_wjc_national_bundle(rng)
    session.wjc_tournament_bundle = {"season_sy": sy, **core}


def _strip_wjc_live_pending(session: FranchiseSession) -> None:
    session.pending_ui_popups = [p for p in (session.pending_ui_popups or []) if not p.get("wjc_live")]


def _push_wjc_live_popup(session: FranchiseSession, payload: Dict[str, Any]) -> None:
    """Replace any previous WJC live overlay so bulk sims only surface the latest day."""
    _strip_wjc_live_pending(session)
    pid = f"pop_{uuid.uuid4().hex[:12]}"
    body = dict(payload)
    body["id"] = pid
    body["wjc_live"] = True
    session.pending_ui_popups.append(body)


def _wjc_live_tournament_payload(session: FranchiseSession, iso: str, d_idx: int, n_days: int) -> Dict[str, Any]:
    sy = int(session.season_calendar_year)
    _ensure_wjc_tournament_bundle(session)
    bundle = getattr(session, "wjc_tournament_bundle", None) or {}
    rr_all: List[Dict[str, Any]] = list(bundle.get("rr_games") or [])
    countries = list(bundle.get("countries") or [])
    codes = [str(c["code"]) for c in countries]
    label_by = {str(c["code"]): str(c["label"]) for c in countries}
    rng = _rng_for_event(session, f"wjc_prospects_{sy}")

    n_rr = len(rr_all)
    rr_through = min(n_rr, max(1, ((d_idx + 1) * n_rr + n_days - 1) // n_days))
    rr_slice = rr_all[:rr_through]
    standings = _rr_standings_from_slice(codes, label_by, rr_slice)

    po_all = bundle.get("playoffs") or {}
    po_out: Dict[str, Any] = {}
    if d_idx >= 7 and rr_through >= n_rr:
        po_out["quarterfinals"] = po_all.get("quarterfinals") or []
    if d_idx >= 8 and rr_through >= n_rr:
        po_out["semifinals"] = po_all.get("semifinals") or []
    if d_idx >= 9 and rr_through >= n_rr:
        po_out["bronze"] = po_all.get("bronze")
    if d_idx >= 10 and rr_through >= n_rr:
        po_out["gold"] = po_all.get("gold")

    complete = bool(d_idx >= 10 and rr_through >= n_rr)
    medals = bundle.get("medal_labels") if complete else {}
    user_prospects = _collect_user_wjc_prospects(session, rng)

    return {
        "kind": "wjc_tournament",
        "wjc_live": True,
        "wjc_phase": "complete" if complete else "live",
        "calendar_iso": iso,
        "wjc_day": d_idx + 1,
        "wjc_days_total": n_days,
        "title": f"World Juniors (U20) ΓÇö day {d_idx + 1} of {n_days}",
        "season_label": f"{sy}ΓÇô{sy + 1}",
        "countries": countries,
        "round_robin_games": rr_slice,
        "round_robin_total": n_rr,
        "standings": standings,
        "playoffs": po_out,
        "medal_labels": medals if complete else {},
        "medals_final": complete,
        "user_prospects": user_prospects,
    }


def _maybe_enqueue_wjc_loan_decisions(session: FranchiseSession, day_meta: Dict[str, Any]) -> None:
    """After Christmas Day, before the first WJC calendar date, offer NHL U20 loan releases (national teams)."""
    iso_done = str(day_meta.get("iso") or "")
    sy = int(session.season_calendar_year)
    if iso_done != f"{sy}-12-25":
        return
    if getattr(session, "wjc_loan_prompts_enqueued", False):
        return
    ut = session.team_by_id.get(str(session.user_team_id))
    if ut is None:
        session.wjc_loan_prompts_enqueued = True
        return
    offered = False
    for p in getattr(ut, "roster", None) or []:
        if getattr(p, "retired", False):
            continue
        ident = getattr(p, "identity", None)
        if ident is None:
            continue
        age = int(getattr(ident, "age", 99) or 99)
        if age > 20:
            continue
        pid = str(getattr(p, "id", "") or "")
        nm = str(getattr(ident, "name", None) or "?")
        bc = str(getattr(ident, "birth_country", "") or "")
        nat = _wjc_country_for_birth(session.sim.rng, bc)
        if not _country_in_wjc_pool(nat):
            continue
        nat_lab = _wjc_country_label(nat)
        storyline_id = f"story_wjc_loan_{pid}"
        dec_id = f"dec_{uuid.uuid4().hex[:12]}"
        session.pending_decisions.append(
            {
                "id": dec_id,
                "storyline_id": storyline_id,
                "kind": "wjc_u20_loan",
                "title": f"World Juniors ΓÇö {nm}",
                "description": (
                    f"{nm} ({age}) is U20-eligible for {nat_lab}. "
                    "Loan him to the national junior tournament roster (WJC recap only ΓÇö no NHL club in the IIHF bracket), "
                    "or keep him with your NHL club."
                ),
                "options": [
                    {
                        "id": "keep",
                        "label": "Keep on NHL roster",
                        "effects": {"chemistry_delta": 0, "prospect_exposure_delta": -1},
                        "effect_summary": "Retains NHL depth, limits international development reps.",
                    },
                    {
                        "id": "loan",
                        "label": f"Loan to {nat_lab} U20",
                        "effects": {"chemistry_delta": 1, "prospect_exposure_delta": 2},
                        "effect_summary": "Improves tournament exposure and confidence, temporarily reduces NHL depth.",
                    },
                ],
                "meta": {
                    "storyline_id": storyline_id,
                    "player_id": pid,
                    "player_name": nm,
                    "wjc_country": nat,
                    "wjc_country_label": nat_lab,
                    "team_id": str(session.user_team_id),
                    "cause": "National team requested U20 availability during World Juniors.",
                },
            }
        )
        offered = True
    session.wjc_loan_prompts_enqueued = True
    if offered:
        session.notifications.append("World Juniors: loan decisions needed for eligible U20 NHL players (Hub).")


def _simulate_showcase_score(rng: random.Random) -> Tuple[int, int, bool]:
    """Return (home_goals, away_goals, overtime) for an outdoor / exhibition tilt."""
    hg = rng.randint(2, 5)
    ag = rng.randint(2, 5)
    ot = rng.random() < 0.22
    if not ot and hg == ag:
        ag += rng.choice([-1, 1])
        ag = max(1, min(6, ag))
    if ot and hg == ag:
        hg += 1
    return hg, ag, ot


def _allstar_game_payload(session: FranchiseSession, rng: random.Random) -> Dict[str, Any]:
    pool: List[Tuple[float, str, str]] = []
    for tid, tm in session.team_by_id.items():
        for p in getattr(tm, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            ident = getattr(p, "identity", None)
            if ident is None:
                continue
            ovr_f = getattr(p, "ovr", None)
            try:
                ov = float(ovr_f() if callable(ovr_f) else ovr_f)
            except Exception:
                ov = 0.6
            if ov > 1.5:
                ov = ov / 99.0
            nm = str(getattr(ident, "name", None) or "?")
            pool.append((ov, str(tid), nm))
    pool.sort(key=lambda x: -x[0])
    top = pool[:24]
    rng.shuffle(top)
    team_a = [x for x in top[:12]]
    team_b = [x for x in top[12:24]]
    ha, hb = rng.randint(4, 8), rng.randint(3, 7)
    if rng.random() < 0.5:
        ha, hb = hb, ha
    ut = str(session.user_team_id)
    user_names_a = [nm for ov, tid, nm in team_a if tid == ut]
    user_names_b = [nm for ov, tid, nm in team_b if tid == ut]
    return {
        "kind": "allstar_game",
        "title": "NHL All-Star Game",
        "season_label": f"{session.season_calendar_year}ΓÇô{int(session.season_calendar_year) + 1}",
        "team_a_label": "Team Pacific / Metro",
        "team_b_label": "Team Atlantic / Central",
        "team_a_score": ha,
        "team_b_score": hb,
        "team_a": [{"name": nm, "is_user": tid == ut} for ov, tid, nm in team_a],
        "team_b": [{"name": nm, "is_user": tid == ut} for ov, tid, nm in team_b],
        "user_allstars": user_names_a + user_names_b,
    }


def _maybe_enqueue_showcase_popups(session: FranchiseSession, day_meta: Dict[str, Any]) -> None:
    iso = str(day_meta.get("iso") or "")
    tags = {str(x).lower() for x in (day_meta.get("tags") or [])}
    sy = int(session.season_calendar_year)
    y2 = sy + 1

    if "world_juniors" in tags:
        d_idx = _wjc_day_index_for_iso(iso, sy)
        if d_idx is not None:
            n_days = len(_wjc_calendar_dates(sy))
            pl = _wjc_live_tournament_payload(session, iso, d_idx, n_days)
            _push_wjc_live_popup(session, pl)
            if pl.get("wjc_phase") == "complete" and f"wjc_final_arch_{sy}" not in session.shown_event_keys:
                session.shown_event_keys.add(f"wjc_final_arch_{sy}")
                snap = {k: v for k, v in pl.items() if k not in ("wjc_live",)}
                arch = list(getattr(session, "showcase_archive", None) or [])
                arch.append(snap)
                session.showcase_archive = arch[-48:]

    if iso == f"{sy}-12-31" and "winter_classic" in tags:
        rk = f"winter_classic_{sy}"
        if rk not in session.shown_event_keys:
            rng = _rng_for_event(session, rk)
            ov = _special_event_overlay(session, day_meta) or {}
            home = ov.get("home") or {"abbr": "?", "name": "TBD"}
            away = ov.get("away") or {"abbr": "?", "name": "TBD"}
            hg, ag, ot = _simulate_showcase_score(rng)
            _append_showcase_popup(
                session,
                rk,
                {
                    "kind": "showcase_game",
                    "subkind": "winter_classic",
                    "title": str(ov.get("title") or "Winter Classic"),
                    "iso": iso,
                    "home": home,
                    "away": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "overtime": ot,
                },
            )

    if iso == f"{y2}-01-13" and "heritage_classic" in tags:
        rk = f"heritage_classic_{sy}"
        if rk not in session.shown_event_keys:
            rng = _rng_for_event(session, rk)
            ov = _special_event_overlay(session, day_meta) or {}
            home = ov.get("home") or {"abbr": "?", "name": "TBD"}
            away = ov.get("away") or {"abbr": "?", "name": "TBD"}
            hg, ag, ot = _simulate_showcase_score(rng)
            _append_showcase_popup(
                session,
                rk,
                {
                    "kind": "showcase_game",
                    "subkind": "heritage_classic",
                    "title": str(ov.get("title") or "Heritage Classic"),
                    "iso": iso,
                    "home": home,
                    "away": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "overtime": ot,
                },
            )

    if iso == f"{y2}-02-03" and "allstar_break" in tags:
        rk = f"allstar_game_{sy}"
        if rk not in session.shown_event_keys:
            rng = _rng_for_event(session, rk)
            _append_showcase_popup(session, rk, _allstar_game_payload(session, rng))

    if iso == f"{y2}-03-14" and "four_nations" in tags:
        rk = f"four_nations_{sy}"
        if rk not in session.shown_event_keys:
            rng = _rng_for_event(session, rk)
            teams = ["CAN", "USA", "SWE", "FIN"]
            rng.shuffle(teams)
            a, b = teams[0], teams[1]
            hg, ag, ot = _simulate_showcase_score(rng)
            _append_showcase_popup(
                session,
                rk,
                {
                    "kind": "showcase_game",
                    "subkind": "four_nations",
                    "title": "4 Nations Face-Off ΓÇö Final",
                    "iso": iso,
                    "home": {"abbr": a, "name": a, "id": ""},
                    "away": {"abbr": b, "name": b, "id": ""},
                    "home_goals": hg,
                    "away_goals": ag,
                    "overtime": ot,
                },
            )


def _maybe_enqueue_post_day_decisions(session: FranchiseSession, user_lines: List[str]) -> None:
    """Lightweight GM prompts derived from engine state (no extra full-season sim)."""
    sim = session.sim
    user_team = session.team_by_id.get(session.user_team_id)
    if user_team is None:
        return
    r = sim.rng

    # Injury prompts for user club (moderate + major; cap 2/day, same calendar day only)
    just_finished_idx = int(session.calendar_cursor) - 1
    decisions_added = 0
    for inj in reversed((session.injury_log_all or [])[-15:]):
        if int(inj.get("date", -1)) != just_finished_idx:
            continue
        if str(inj.get("team_id") or "") != str(session.user_team_id):
            continue
        tier = str(inj.get("tier") or "").lower()
        if tier not in ("major", "moderate"):
            continue
        pname = str(inj.get("player_name") or inj.get("player") or "Player")
        games = int(inj.get("games") or 0)
        storyline_id = f"story_injury_protocol_{str(inj.get('id') or pname).replace(' ', '_')}"
        dec_id = f"dec_{uuid.uuid4().hex[:12]}"
        session.pending_decisions.append(
            {
                "id": dec_id,
                "storyline_id": storyline_id,
                "kind": "injury_protocol",
                "priority": "CRITICAL" if tier == "major" else "HIGH",
                "title": "Medical staff report",
                "description": f"{pname} ΓÇö {tier} injury (~{games} games). Choose how you message the room.",
                "options": [
                    {
                        "id": "transparent",
                        "label": "Transparent update (builds trust)",
                        "effects": {"morale_delta": 2, "media_noise_delta": 1},
                        "effect_summary": "Boosts room trust, slightly increases media cycle.",
                    },
                    {
                        "id": "minimize",
                        "label": "Minimize publicly (reduces media noise)",
                        "effects": {"morale_delta": 1, "media_noise_delta": -1},
                        "effect_summary": "Keeps coverage quiet with a small trust bump.",
                    },
                    {
                        "id": "next_man",
                        "label": "Next-man-up rhetoric (pressure on depth)",
                        "effects": {"morale_delta": 0, "depth_pressure_delta": 2},
                        "effect_summary": "Signals urgency; depth players absorb extra pressure.",
                    },
                ],
                "meta": {
                    "storyline_id": storyline_id,
                    "injury": inj,
                    "player_name": pname,
                    "team_id": str(session.user_team_id),
                    "cause": f"{pname} suffered a {tier} injury (~{games} games).",
                },
            }
        )
        decisions_added += 1
        if decisions_added >= 2:
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
                storyline_id = f"story_ice_time_{str(nm).replace(' ', '_').lower()}_{just_finished_idx}"
                session.pending_decisions.append(
                    {
                        "id": dec_id,
                        "storyline_id": storyline_id,
                        "kind": "ice_time",
                        "priority": "MEDIUM",
                        "title": f"{nm} wants a larger role",
                        "description": "Agents and internal scouts disagree on fit. Your call.",
                        "options": [
                            {
                                "id": "promote",
                                "label": "Promote usage (+ morale short-term, fatigue risk)",
                                "effects": {"morale_delta": 2, "fatigue_delta": 2},
                                "effect_summary": "Immediate confidence boost with heavier workload risk.",
                            },
                            {
                                "id": "steady",
                                "label": "Hold structure (stable room)",
                                "effects": {"morale_delta": 1, "fatigue_delta": 0},
                                "effect_summary": "Keeps current deployment and room balance intact.",
                            },
                            {
                                "id": "bench_msg",
                                "label": "Send message with minutes cut (discipline)",
                                "effects": {"morale_delta": -2, "fatigue_delta": -1},
                                "effect_summary": "Lowers satisfaction but protects energy and hierarchy.",
                            },
                        ],
                        "meta": {
                            "storyline_id": storyline_id,
                            "player_name": nm,
                            "team_id": str(session.user_team_id),
                            "cause": f"{nm}'s camp requested a larger role.",
                        },
                    }
                )

    if not user_lines and r.random() < 0.12:
        storyline_id = f"story_trade_inquiry_{just_finished_idx}"
        dec_id = f"dec_{uuid.uuid4().hex[:12]}"
        session.pending_decisions.append(
            {
                "id": dec_id,
                "storyline_id": storyline_id,
                "kind": "trade_inquiry",
                "priority": "MEDIUM",
                "title": "Trade desk ping",
                "description": "Rival GM floats a futures-for-help concept. No names on paper yet.",
                "options": [
                    {
                        "id": "listen",
                        "label": "Stay open ΓÇö scouting will dig",
                        "effects": {"trade_activity_delta": 2, "asset_risk_delta": 1},
                        "effect_summary": "Increases market optionality with mild valuation risk.",
                    },
                    {
                        "id": "decline",
                        "label": "Decline politely",
                        "effects": {"trade_activity_delta": -1, "asset_risk_delta": -1},
                        "effect_summary": "Preserves assets and short-term stability.",
                    },
                    {
                        "id": "counter",
                        "label": "Counter with salary retention ask",
                        "effects": {"trade_activity_delta": 1, "cap_flex_delta": 1},
                        "effect_summary": "Keeps talks alive while targeting cap leverage.",
                    },
                ],
                "meta": {
                    "storyline_id": storyline_id,
                    "team_id": str(session.user_team_id),
                    "cause": "A rival GM floated a futures-for-help concept.",
                },
            }
        )


class _FranchiseLifecycleLogger:
    """Satisfies run_sim career pass: no console spam, optional capture later."""

    log_level = "normal"

    def emit(self, *_args: Any, **_kwargs: Any) -> None:
        return


def _strip_retired_from_nhl_rosters(teams: List[Any]) -> int:
    removed = 0
    for team in teams:
        roster = list(getattr(team, "roster", None) or [])
        kept = [p for p in roster if not getattr(p, "retired", False)]
        removed += len(roster) - len(kept)
        team.roster = kept
    return removed


def _franchise_nhl_age_and_phase_tick(session: FranchiseSession, teams: List[Any]) -> None:
    """One calendar year: Player.advance_year + career phase (mirrors universe roster pass)."""
    from app.sim_engine import engine as eng_mod
    from app.sim_engine.engine import assign_career_phase_from_age

    team_instability = max(0.28, min(0.62, 1.05 - float(session.chaos_index)))
    for team in teams:
        roster = getattr(team, "roster", None) or []
        dev_quality = float(getattr(team, "development_quality", 0.5))
        dev_mod = dev_quality - 0.5
        for player in roster:
            if getattr(player, "retired", False):
                continue
            advance_fn = getattr(player, "advance_year", None)
            if not callable(advance_fn):
                ident = getattr(player, "identity", None)
                if ident is not None and hasattr(ident, "age"):
                    try:
                        ident.age = int(getattr(ident, "age", 0)) + 1
                    except (TypeError, ValueError):
                        pass
            else:
                try:
                    ident = getattr(player, "identity", None)
                    age = int(getattr(ident, "age", getattr(player, "age", 25)) if ident else getattr(player, "age", 25))
                    age_damp = max(0.35, min(1.0, 1.0 - max(0.0, (age - 26)) / 10.0))
                    morale = float(getattr(getattr(player, "psych", None), "morale", 0.5) or 0.5)
                    inj = float(getattr(getattr(player, "health", None), "injury_risk_baseline", 0.1) or 0.1)
                    try:
                        sys_dev = float(eng_mod.team_system_development_modifier(team))
                    except Exception:
                        sys_dev = 0.0
                    advance_fn(
                        season_morale=morale,
                        season_injury_risk=inj,
                        team_instability=team_instability,
                        development_modifier=dev_mod * age_damp + sys_dev,
                    )
                except Exception:
                    pass
            try:
                assign_career_phase_from_age(player)
            except Exception:
                pass


def _run_franchise_season_end_progression(session: FranchiseSession) -> Dict[str, Any]:
    """
    After the regular-season calendar: NHL roster aging + the same progression stack as the
    universe runner (development pass ΓåÆ major career events ΓåÆ soft anti-inflation guard).
    """
    out: Dict[str, Any] = {"aged": True, "lifecycle": None, "retired_removed": 0}
    sim = session.sim
    league = getattr(sim, "league", None)
    teams = list(getattr(league, "teams", None) or [])
    if not teams or league is None:
        return out

    rng = sim.rng
    sy = int(session.season_calendar_year)

    try:
        setattr(
            league,
            "_tuning_context",
            {
                "chaos_index": float(session.chaos_index),
                "parity_index": 0.52,
                "league_health": 0.58,
                "active_era": "modern",
            },
        )
    except Exception:
        pass

    _franchise_nhl_age_and_phase_tick(session, teams)

    if getattr(rs, "_run_player_progression_pass", None):
        try:
            rs._run_player_progression_pass(teams, rng, None)
        except Exception:
            pass

    if getattr(rs, "_run_career_lifecycle_pass", None):
        try:
            out["lifecycle"] = rs._run_career_lifecycle_pass(
                teams,
                rng,
                _FranchiseLifecycleLogger(),
                league=league,
                state=None,
                season_year=sy,
            )
        except Exception:
            out["lifecycle"] = {"skipped": True}

    try:
        from app.sim_engine.engine import apply_league_ovr_soft_regression_if_needed

        apply_league_ovr_soft_regression_if_needed(teams, rng, avg_trigger=74.5)
    except Exception:
        pass

    out["retired_removed"] = int(_strip_retired_from_nhl_rosters(teams))
    out["season_calendar_year_unchanged"] = sy
    return out


def _auto_resolve_pending_decisions(session: FranchiseSession) -> None:
    """
    Bulk sim helper only.

    Manual Advance Day should not call this.
    This is intentionally conservative and logs every forced choice.
    """
    while getattr(session, "pending_decisions", None):
        d = session.pending_decisions[0]

        if not isinstance(d, dict):
            session.pending_decisions.pop(0)
            continue

        opts = d.get("options") or d.get("choices") or []

        if not opts:
            session.timeline.append(
                f"AUTO-RESOLVE: removed decision with no options ({d.get('id') or d.get('kind') or 'unknown'})."
            )
            session.pending_decisions.pop(0)
            continue

        first = opts[0] if isinstance(opts[0], dict) else {"id": str(opts[0])}
        choice_id = str(first.get("id") or first.get("choice_id") or "")

        if not choice_id:
            session.timeline.append(
                f"AUTO-RESOLVE: removed malformed decision ({d.get('id') or d.get('kind') or 'unknown'})."
            )
            session.pending_decisions.pop(0)
            continue

        session.timeline.append(
            f"AUTO-RESOLVE: {d.get('kind') or d.get('type') or 'decision'} "
            f"{d.get('id') or ''} -> {choice_id}"
        )

        apply_decision(session, str(d.get("id") or ""), choice_id)
def _apply_injury_decision_effect(
    session: FranchiseSession,
    decision: Dict[str, Any],
    choice: Dict[str, Any],
) -> Dict[str, Any]:
    user_team = session.team_by_id.get(str(session.user_team_id))
    meta = dict(decision.get("meta") or {})
    injury = dict(meta.get("injury") or {})
    player_name = str(meta.get("player_name") or injury.get("player_name") or injury.get("player") or "Player")
    player_id = str(meta.get("player_id") or injury.get("player_id") or "")
    choice_id = str(choice.get("id") or "")

    effects: Dict[str, Any] = {}

    if user_team is None:
        return effects

    target = _find_player_on_team_by_id_or_name(user_team, player_id=player_id, player_name=player_name)

    if choice_id == "transparent":
        changed = _nudge_team_room(user_team, morale=0.012, confidence=0.006, role_satisfaction=0.018)
        if target is not None:
            _nudge_player_psych(target, morale=0.025, confidence=0.01, role_satisfaction=0.03)
        effects.update({"room_trust_delta": 2, "media_noise_delta": 1, "players_affected": changed})

    elif choice_id == "minimize":
        changed = _nudge_team_room(user_team, morale=0.004, confidence=0.004, role_satisfaction=-0.004)
        effects.update({"room_trust_delta": 1, "media_noise_delta": -1, "players_affected": changed})

    elif choice_id == "next_man":
        changed = _nudge_team_room(user_team, morale=0.002, confidence=0.012, role_satisfaction=-0.008)
        setattr(user_team, "_depth_pressure", float(getattr(user_team, "_depth_pressure", 0.0) or 0.0) + 0.06)
        effects.update({"depth_pressure_delta": 2, "confidence_delta": 1, "players_affected": changed})

    elif choice_id == "call_up_player":
        setattr(user_team, "_needs_callup", True)
        setattr(user_team, "_depth_pressure", float(getattr(user_team, "_depth_pressure", 0.0) or 0.0) - 0.02)
        changed = _nudge_team_room(user_team, morale=0.004, confidence=0.004)
        effects.update({"callup_flag": 1, "depth_pressure_delta": -1, "players_affected": changed})

    elif choice_id == "shuffle_lines":
        setattr(user_team, "_lines_shuffled", True)
        changed = _nudge_team_room(user_team, morale=0.002, confidence=0.006, role_satisfaction=0.006)
        effects.update({"line_chemistry_volatility": 1, "players_affected": changed})

    elif choice_id == "play_short_roster":
        setattr(user_team, "_depth_pressure", float(getattr(user_team, "_depth_pressure", 0.0) or 0.0) + 0.10)
        changed = _nudge_team_room(user_team, morale=-0.004, confidence=-0.004, role_satisfaction=-0.012)
        effects.update({"depth_pressure_delta": 3, "fatigue_risk_delta": 2, "players_affected": changed})

    elif choice_id == "place_on_ir":
        setattr(user_team, "_ir_management_used", int(getattr(user_team, "_ir_management_used", 0) or 0) + 1)
        setattr(user_team, "_needs_callup", True)
        effects.update({"ir_used": 1, "callup_flag": 1, "cap_flexibility_delta": 1})

    return effects


def _apply_ice_time_decision_effect(
    session: FranchiseSession,
    decision: Dict[str, Any],
    choice: Dict[str, Any],
) -> Dict[str, Any]:
    user_team = session.team_by_id.get(str(session.user_team_id))
    meta = dict(decision.get("meta") or {})
    player_name = str(meta.get("player_name") or "")
    player_id = str(meta.get("player_id") or "")
    choice_id = str(choice.get("id") or "")

    effects: Dict[str, Any] = {}

    if user_team is None:
        return effects

    target = _find_player_on_team_by_id_or_name(user_team, player_id=player_id, player_name=player_name)

    if target is None:
        return effects

    if choice_id == "promote":
        _nudge_player_psych(target, morale=0.02, confidence=0.035, role_satisfaction=0.10)
        setattr(target, "_temporary_role_boost_games", 5)
        effects.update({"role_satisfaction_delta": 3, "confidence_delta": 2, "temporary_role_boost_games": 5})

    elif choice_id == "bench_msg":
        _nudge_player_psych(target, morale=-0.015, confidence=-0.012, role_satisfaction=-0.12)
        setattr(target, "_accountability_pressure_games", 4)
        effects.update({"role_satisfaction_delta": -3, "accountability_pressure_games": 4})

    else:
        _nudge_player_psych(target, morale=0.006, confidence=0.006, role_satisfaction=0.02)
        effects.update({"role_satisfaction_delta": 1})

    return effects


def _apply_generic_storyline_choice_effect(
    session: FranchiseSession,
    decision: Dict[str, Any],
    choice: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fallback for any future storyline choice.
    Reads choice.effects and converts them into visible player/team nudges.
    """
    user_team = session.team_by_id.get(str(session.user_team_id))
    meta = dict(decision.get("meta") or {})
    player_name = str(meta.get("player_name") or "")
    player_id = str(meta.get("player_id") or "")
    raw_effects = dict(choice.get("effects") or {})

    applied: Dict[str, Any] = {}

    if user_team is None:
        return applied

    target = _find_player_on_team_by_id_or_name(user_team, player_id=player_id, player_name=player_name)

    morale = float(raw_effects.get("morale_delta", raw_effects.get("morale", 0)) or 0) / 100.0
    confidence = float(raw_effects.get("confidence_delta", raw_effects.get("confidence", 0)) or 0) / 100.0
    role = float(raw_effects.get("role_satisfaction_delta", raw_effects.get("role", 0)) or 0) / 100.0

    if target is not None:
        _nudge_player_psych(target, morale=morale, confidence=confidence, role_satisfaction=role)
        applied["target_player_affected"] = 1
    elif any([morale, confidence, role]):
        changed = _nudge_team_room(
            user_team,
            morale=morale * 0.35,
            confidence=confidence * 0.35,
            role_satisfaction=role * 0.35,
        )
        applied["players_affected"] = changed

    for k, v in raw_effects.items():
        applied[k] = v

    return applied
def _pending_decision_snapshot(session: FranchiseSession) -> List[Dict[str, Any]]:
    """
    Small frontend-safe decision payload.
    Used when Advance Day stops before simming.
    """
    out: List[Dict[str, Any]] = []

    for index, d in enumerate(list(getattr(session, "pending_decisions", None) or [])):
        if not isinstance(d, dict):
            continue

        meta = dict(d.get("meta") or {})
        opts = list(d.get("options") or d.get("choices") or [])

        safe_options: List[Dict[str, Any]] = []
        for opt in opts:
            if not isinstance(opt, dict):
                continue

            safe_options.append(
                {
                    "id": str(opt.get("id") or opt.get("choice_id") or ""),
                    "label": str(opt.get("label") or opt.get("title") or opt.get("text") or "Choice"),
                    "description": str(opt.get("description") or opt.get("details") or ""),
                }
            )

        out.append(
            {
                "id": str(d.get("id") or f"decision:{index}"),
                "kind": str(d.get("kind") or d.get("type") or "decision"),
                "title": str(
                    d.get("title")
                    or d.get("headline")
                    or meta.get("title")
                    or "Decision Required"
                ),
                "message": str(
                    d.get("message")
                    or d.get("text")
                    or d.get("details")
                    or meta.get("message")
                    or "Resolve this item before advancing."
                ),
                "priority": str(d.get("priority") or meta.get("priority") or "HIGH").upper(),
                "calendar_day": int(
                    d.get("calendar_day")
                    or d.get("date")
                    or meta.get("calendar_day")
                    or getattr(session, "calendar_cursor", 0)
                    or 0
                ),
                "calendar_iso": str(
                    d.get("calendar_iso")
                    or d.get("calendarIso")
                    or meta.get("calendar_iso")
                    or _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
                    or ""
                ),
                "team_id": str(d.get("team_id") or meta.get("team_id") or ""),
                "player_id": str(d.get("player_id") or meta.get("player_id") or ""),
                "player_name": str(d.get("player_name") or meta.get("player_name") or ""),
                "options": safe_options,
                "meta": meta,
            }
        )

    return out


def _advance_blocked_result(
    session: FranchiseSession,
    *,
    reason: str,
    message: str,
) -> Dict[str, Any]:
    """
    Return a normal object instead of forcing the frontend to interpret a crash.
    """
    return {
        "status": "blocked",
        "mode": "day",
        "reason": str(reason or "blocked"),
        "message": str(message or "Resolve pending decisions before advancing."),
        "calendar_index": int(getattr(session, "calendar_cursor", 0) or 0),
        "iso": _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0)),
        "pending_decisions": _pending_decision_snapshot(session),
    }
def auto_resolve_franchise_decisions(session: FranchiseSession) -> None:
    """Public alias for API use before a single manual advance."""
    _auto_resolve_pending_decisions(session)


def _enter_postseason(session: FranchiseSession) -> Dict[str, Any]:
    """Regular season finished — open playoff-ready flow (bracket UI, then postseason)."""
    from app.sim_engine.franchise.offseason import _transition_to_playoff_ready

    return _transition_to_playoff_ready(session)


def advance_franchise_day(session: FranchiseSession) -> Dict[str, Any]:
    """
    Advance exactly one NHL calendar day.

    Sacred rules:
    - Do not auto-resolve user decisions here.
    - Do not silently mutate the schedule here.
    - If a daily tick creates a user-facing blocking decision, stop before games.
    - Validate schedule before simming.
    - Sim games once.
    - Clear only the current day after successful simulation.
    - Move cursor exactly once after successful simulation.
    """
    _ensure_session_event_lists(session)

    if getattr(session, "pending_decisions", None):
        return _advance_blocked_result(
            session,
            reason="pending_decisions",
            message="Resolve pending decisions before advancing.",
        )

    _sync_nhl_calendar_bounds(session)

    if session.phase == "complete":
        return {
            "status": "complete",
            "mode": "day",
            "message": "Season and playoffs finished. Start a new franchise to continue.",
            "calendar_index": int(getattr(session, "calendar_cursor", 0) or 0),
            "iso": _calendar_iso_for_day(
                session,
                int(getattr(session, "calendar_cursor", 0) or 0),
            ),
        }

    if session.phase == "regular" and int(session.calendar_cursor) > int(session.nhl_regular_season_last_index):
        if not _regular_season_is_truly_complete(session):
            remaining = _remaining_regular_games_count(session)
            raise RuntimeError(
                f"Regular season boundary reached but {remaining} regular-season game(s) remain unsimulated."
            )

        return _enter_postseason(session)

    cal = getattr(session, "nhl_calendar", None) or []

    if not cal:
        raise RuntimeError("Franchise session missing NHL calendar data.")

    idx = int(session.calendar_cursor)

    if idx < 0 or idx >= len(cal):
        raise RuntimeError(f"Calendar cursor out of range: {idx} / {len(cal)}.")

    day_meta = cal[idx]
    day_ordinal = idx + 1

    _split_preseason_from_regular_if_needed(session, day_meta)

    # 1. Daily league office/storyline/injury/trade/news tick.
    # If this creates decisions, stop BEFORE games are simulated.
    _franchise_daily_league_tick(session, idx)

    if getattr(session, "pending_decisions", None):
        return _advance_blocked_result(
            session,
            reason="daily_tick_decision",
            message="A league-office alert needs your attention before this calendar date can be simulated.",
        )

    # 2. Off-day injury check for the user's team.
    # This can create an injury popup/decision. If it does, stop before games.
    if (
        world_injuries is not None
        and getattr(session, "injuries_enabled", True)
        and not _team_plays_on_day(
            session.by_day,
            idx,
            str(session.user_team_id),
        )
    ):
        user_team = session.team_by_id.get(str(session.user_team_id))

        if user_team is not None:
            iso_row = str(day_meta.get("iso") or "")
            events = world_injuries.maybe_injure_roster_subset(
                user_team,
                session.sim.rng,
                session.chaos_index,
                max_checks=1,
                low_intensity=True,
            )

            tid_inj = str(
                getattr(user_team, "team_id", None)
                or getattr(user_team, "id", None)
                or ""
            )
            abbrev = _franchise_team_abbrev(user_team)

            for label, tier, games, pid in events:
                _franchise_log_injury_and_ui(
                    session,
                    player_id=pid,
                    player_name=label,
                    team_id=tid_inj,
                    team_abbrev=abbrev,
                    tier=str(tier),
                    games=int(games),
                    injury_type=str(tier),
                    calendar_day=int(idx),
                    calendar_iso=iso_row,
                )

    if getattr(session, "pending_decisions", None):
        return _advance_blocked_result(
            session,
            reason="injury_decision",
            message="An injury decision needs your attention before this calendar date can be simulated.",
        )

    # 3. Hard schedule validation.
    # Do not repair here. Runtime repair makes the calendar lie.
    day_schedule_errors = _validate_schedule_hard(session.by_day, cal, day_filter=idx)

    if day_schedule_errors:
        _fr_dbg(f"schedule hard-validation failed on day {idx}: {day_schedule_errors[0]}")
        raise RuntimeError(
            f"Schedule integrity error at {day_meta.get('iso') or idx}: {day_schedule_errors[0]}"
        )

    slots = list(session.by_day.get(idx, []) or [])

    # 4. User double-booking should already be impossible after group 1 fixes.
    # If it still happens, fail loudly instead of silently shifting the calendar.
    utid = str(session.user_team_id)

    user_slots = [
        sl
        for sl in slots
        if _safe_slot_team_id(sl, "home_id") == utid
        or _safe_slot_team_id(sl, "away_id") == utid
    ]

    if len(user_slots) > 1:
        raise RuntimeError(
            f"Schedule integrity error at {day_meta.get('iso') or idx}: user team has "
            f"{len(user_slots)} games on the same day. Fix schedule generation, not runtime advance."
        )

    # 5. Sim the actual day.
    user_lines, league_lines = _simulate_slots_for_day(session, idx, slots)

    # 6. Only after successful simulation do we clear the slate and move the cursor.
    session.by_day[idx] = []
    session.calendar_cursor = int(session.calendar_cursor) + 1

    _finalize_regular_calendar_day(
        session,
        day_meta,
        user_lines,
        league_lines,
        day_ordinal=day_ordinal,
    )

    return {
        "status": "ok",
        "mode": "day",
        "calendar_index": idx,
        "next_calendar_index": int(session.calendar_cursor),
        "iso": str(day_meta.get("iso") or ""),
        "user_game_summaries": user_lines,
        "league_game_summaries": league_lines,
        "games_simulated": int(len(slots)),
        "pending_decisions": _pending_decision_snapshot(session),
    }
def advance_franchise_one_game(session: FranchiseSession) -> Dict[str, Any]:
    """One real NHL calendar day (same as advance day ΓÇö game-by-game calendar progression removed)."""
    return advance_franchise_day(session)


def advance_franchise_bulk(
    session: FranchiseSession,
    *,
    mode: str = "day",
    count: int = 1,
    auto_resolve_decisions: bool = False,
) -> Dict[str, Any]:
    """
    Run multiple advance steps server-side.

    Manual/day advancement must not auto-resolve decisions.
    Full season/bulk sim may opt into auto-resolve by explicitly passing true.
    """
    raw = (mode or "day").strip().lower()

    if raw == "day":
        eff_mode, eff_count = "days", max(1, int(count))
    else:
        eff_mode, eff_count = raw, max(1, int(count))

    steps: List[Dict[str, Any]] = []
    guard = 0
    max_iter = 6000
    stopped: Optional[str] = None

    while guard < max_iter:
        guard += 1

        if auto_resolve_decisions and getattr(session, "pending_decisions", None):
            _auto_resolve_pending_decisions(session)

        if getattr(session, "pending_decisions", None):
            stopped = "pending_decisions"
            break

        if eff_mode == "days":
            step = advance_franchise_day(session)
        elif eff_mode == "games":
            step = advance_franchise_one_game(session)
        elif eff_mode == "season":
            step = advance_franchise_day(session)
        else:
            step = advance_franchise_day(session)

        steps.append(step)

        st = str(step.get("status") or "")

        if st == "blocked":
            stopped = str(step.get("reason") or "blocked")
            break

        if st != "ok":
            stopped = st
            break

        if eff_mode == "days":
            eff_count -= 1
            if eff_count <= 0:
                stopped = "count"
                break

        elif eff_mode == "games":
            eff_count -= 1
            if eff_count <= 0:
                stopped = "count"
                break

        elif eff_mode == "season":
            if session.phase != "regular":
                stopped = "phase"
                break

        else:
            stopped = "count"
            break

    if guard >= max_iter:
        stopped = "guard_limit"

    last = steps[-1] if steps else {
        "status": "blocked" if getattr(session, "pending_decisions", None) else "noop",
        "reason": "pending_decisions" if getattr(session, "pending_decisions", None) else "noop",
        "pending_decisions": _pending_decision_snapshot(session),
    }

    tail = steps[-20:] if len(steps) > 20 else steps

    return {
        "status": last.get("status", "noop"),
        "bulk": True,
        "steps_completed": len(steps),
        "stopped_reason": stopped,
        "last_step": last,
        "recent_steps": tail,
        "pending_decisions": _pending_decision_snapshot(session),
        "calendar_index": int(getattr(session, "calendar_cursor", 0) or 0),
        "iso": _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0)),
    }

def apply_storyline_choice(session: FranchiseSession, storyline_id: str, choice_id: str) -> None:
    """
    Resolve a choice by storyline id.

    Frontend story cards usually know storyline_id, not pending decision id.
    """
    sid = str(storyline_id or "").strip()
    cid = str(choice_id or "").strip()

    if not sid:
        raise ValueError("Storyline id is required.")

    if not cid:
        raise ValueError("Choice id is required.")

    for d in list(getattr(session, "pending_decisions", None) or []):
        if not isinstance(d, dict):
            continue

        meta = dict(d.get("meta") or {})
        dec_story_id = str(
            meta.get("storyline_id")
            or d.get("storyline_id")
            or d.get("id")
            or ""
        )

        if dec_story_id == sid:
            apply_decision(session, str(d.get("id") or ""), cid)
            return

    raise ValueError(f"Storyline choice target not found: {sid}")


def apply_decision(session: FranchiseSession, decision_id: str, choice_id: str) -> None:
    """
    Resolve a pending user decision and apply real effects.

    This now:
    - validates decision id
    - validates choice id
    - applies injury/ice-time/storyline effects
    - writes visible feedback to timeline/calendar/storylines/notifications
    - removes matching popups so the UI does not keep showing stale blockers
    """
    did = str(decision_id or "").strip()
    cid = str(choice_id or "").strip()

    if not did:
        raise ValueError("Decision id is required.")

    if not cid:
        raise ValueError("Choice id is required.")

    pending = list(getattr(session, "pending_decisions", None) or [])

    for i, d in enumerate(pending):
        if not isinstance(d, dict):
            continue

        if str(d.get("id") or "") != did:
            continue

        kind = str(d.get("kind") or d.get("type") or "decision")
        options = list(d.get("options") or d.get("choices") or [])
        chosen: Optional[Dict[str, Any]] = None

        for opt in options:
            if not isinstance(opt, dict):
                continue
            if str(opt.get("id") or opt.get("choice_id") or "") == cid:
                chosen = opt
                break

        if chosen is None:
            raise ValueError(f"Choice {cid!r} not found for decision {did!r}.")

        # Remove the decision first so retry loops do not double-apply it.
        session.pending_decisions.pop(i)

        effects: Dict[str, Any] = {}

        if kind in ("injury_protocol", "injury_decision"):
            effects.update(_apply_injury_decision_effect(session, d, chosen))

        elif kind == "ice_time":
            effects.update(_apply_ice_time_decision_effect(session, d, chosen))

        elif kind == "wjc_u20_loan":
            meta = dict(d.get("meta") or {})
            pid = str(meta.get("player_id") or "").strip()

            if pid:
                if not hasattr(session, "wjc_nhl_u20_loan") or session.wjc_nhl_u20_loan is None:
                    session.wjc_nhl_u20_loan = {}

                session.wjc_nhl_u20_loan[pid] = bool(cid == "loan")
                effects["wjc_loan"] = 1 if cid == "loan" else 0

        else:
            effects.update(_apply_generic_storyline_choice_effect(session, d, chosen))

        # Merge visible declared effects with actual applied effects.
        declared = dict(chosen.get("effects") or {})
        final_effects = {**declared, **effects}

        meta = dict(d.get("meta") or {})
        title = str(d.get("title") or d.get("headline") or "Decision Resolved")
        label = str(chosen.get("label") or cid)
        player_name = str(meta.get("player_name") or d.get("player_name") or "")

        headline = f"{title}: {label}"
        summary = f"You chose: {label}."
        if player_name:
            summary = f"{player_name} ΓÇö {summary}"

        if chosen.get("effect_summary"):
            summary += f" {chosen.get('effect_summary')}"

        _append_decision_feedback(
            session,
            decision=d,
            choice=chosen,
            headline=headline,
            summary=summary,
            priority=str(d.get("priority") or "MEDIUM").upper(),
            effects=final_effects,
        )

        # Clear stale UI popups tied to this decision/storyline.
        story_id = str(meta.get("storyline_id") or d.get("storyline_id") or "")
        next_popups: List[Dict[str, Any]] = []

        for popup in list(getattr(session, "pending_ui_popups", None) or []):
            if not isinstance(popup, dict):
                continue

            popup_decision_id = str(popup.get("decision_id") or popup.get("id") or "")
            popup_story_id = str(popup.get("storyline_id") or "")

            if popup_decision_id == did:
                continue

            if story_id and popup_story_id == story_id:
                continue

            next_popups.append(popup)

        session.pending_ui_popups = next_popups
        session.timeline.append(f"Decision ({kind}): {did} -> {cid}")

        return

    raise ValueError(f"Decision {did!r} not found.")
def _player_display_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    if ident is not None and getattr(ident, "name", None):
        return str(ident.name)
    return str(getattr(player, "name", None) or "Player")


def _find_player_on_team_by_id_or_name(team: Any, *, player_id: str = "", player_name: str = "") -> Optional[Any]:
    pid = str(player_id or "").strip()
    pname = str(player_name or "").strip().lower()

    for p in getattr(team, "roster", None) or []:
        if getattr(p, "retired", False):
            continue

        p_id = str(
            getattr(p, "player_id", None)
            or getattr(p, "id", None)
            or getattr(p, "uid", None)
            or ""
        ).strip()

        if pid and p_id and p_id == pid:
            return p

        pn = _player_display_name(p).strip().lower()

        if pname and pn == pname:
            return p

    return None


def _nudge_player_psych(
    player: Any,
    *,
    morale: float = 0.0,
    confidence: float = 0.0,
    role_satisfaction: float = 0.0,
) -> None:
    psych = getattr(player, "psych", None)

    if psych is None:
        return

    for attr, delta in (
        ("morale", morale),
        ("confidence", confidence),
        ("role_satisfaction", role_satisfaction),
    ):
        if not delta:
            continue

        try:
            cur = float(getattr(psych, attr, 0.5) or 0.5)
            setattr(psych, attr, _clamp(cur + float(delta)))
        except Exception:
            pass


def _nudge_team_room(
    team: Any,
    *,
    morale: float = 0.0,
    confidence: float = 0.0,
    role_satisfaction: float = 0.0,
    limit: int = 28,
) -> int:
    changed = 0

    for p in getattr(team, "roster", None) or []:
        if changed >= int(limit):
            break

        if getattr(p, "retired", False):
            continue

        if getattr(p, "psych", None) is None:
            continue

        _nudge_player_psych(
            p,
            morale=morale,
            confidence=confidence,
            role_satisfaction=role_satisfaction,
        )
        changed += 1

    return changed


def _append_decision_feedback(
    session: FranchiseSession,
    *,
    decision: Dict[str, Any],
    choice: Dict[str, Any],
    headline: str,
    summary: str,
    priority: str = "MEDIUM",
    effects: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Every resolved user choice should leave visible evidence:
    timeline + notification + storyline + calendar event.
    """
    _ensure_session_event_lists(session)

    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    iso = _calendar_iso_for_day(session, cur)
    did = str(decision.get("id") or uuid.uuid4().hex[:10])
    cid = str(choice.get("id") or "choice")
    kind = str(decision.get("kind") or decision.get("type") or "decision")
    meta = dict(decision.get("meta") or {})
    team_id = str(meta.get("team_id") or decision.get("team_id") or getattr(session, "user_team_id", "") or "")
    player_id = str(meta.get("player_id") or decision.get("player_id") or "")
    player_name = str(meta.get("player_name") or decision.get("player_name") or "")

    event_id = f"decision:{did}:{cid}"

    row = {
        "id": event_id,
        "kind": "decision_result",
        "type": "decision_result",
        "calendar_day": cur,
        "date": cur,
        "calendar_iso": iso,
        "title": headline,
        "headline": headline,
        "summary": summary,
        "description": summary,
        "priority": str(priority or "MEDIUM").upper(),
        "team_id": team_id,
        "player_id": player_id,
        "player_name": player_name,
        "decision_kind": kind,
        "choice_id": cid,
        "choice_label": str(choice.get("label") or cid),
        "effects": effects or dict(choice.get("effects") or {}),
        "effect_summary": str(choice.get("effect_summary") or choice.get("effectSummary") or ""),
        "surfaces": ["calendar", "storylines", "notifications", "timeline"],
    }

    _append_unique_dict_event(session.calendar_events, row)

    session.notifications.append(
        _normalized_notification(
            notification_id=f"notif:{event_id}",
            notification_type="decision_result",
            text=summary,
            priority=str(priority or "MEDIUM").upper(),
            calendar_day=cur,
            calendar_iso=iso,
            team_id=team_id,
            player_id=player_id,
            source="user_decision",
            extra={
                "decision_kind": kind,
                "choice_id": cid,
                "choice_label": str(choice.get("label") or cid),
            },
        )
    )

    _record_storyline(
        session,
        {
            "id": f"story:{event_id}",
            "type": "decision_result",
            "kind": "decision_result",
            "headline": headline,
            "details": summary,
            "cause": str(decision.get("description") or decision.get("message") or ""),
            "effects": effects or dict(choice.get("effects") or {}),
            "effect_summary": str(choice.get("effect_summary") or choice.get("effectSummary") or ""),
            "team": team_id,
            "team_id": team_id,
            "player_id": player_id,
            "player_name": player_name,
            "players": [player_name] if player_name else [],
            "priority": str(priority or "MEDIUM").upper(),
            "date": cur,
            "calendar_day": cur,
            "calendar_iso": iso,
            "surfaces": ["storylines", "calendar"],
        },
    )

    session.timeline.append(
        _normalized_timeline_event(
            event_id=f"timeline:{event_id}",
            event_type="decision_result",
            text=f"{headline}: {summary}",
            calendar_day=cur,
            calendar_iso=iso,
            team_id=team_id,
            player_id=player_id,
            priority=str(priority or "MEDIUM").upper(),
            extra={"choice_id": cid, "decision_kind": kind},
        )
    )
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


def _user_team_record_from_game_results(session: FranchiseSession) -> Dict[str, Any]:
    """
    User W-L-OTL and points derived only from game_results (same source as the calendar).
    Matches NHL-style counting: OTL is not a regulation loss; OT loss earns 1 point.
    """
    uid = str(session.user_team_id or "")
    out: Dict[str, Any] = {"gp": 0, "w": 0, "l": 0, "otl": 0, "pts": 0}
    if not uid:
        return out
    for g in getattr(session, "game_results", None) or []:
        hid = str(g.get("home_id") or "")
        aid = str(g.get("away_id") or "")
        if uid not in (hid, aid):
            continue
        try:
            hg = int(g.get("home_goals") or 0)
            ag = int(g.get("away_goals") or 0)
        except (TypeError, ValueError):
            continue
        if hg == ag:
            continue
        ot = bool(g.get("overtime"))
        is_home = hid == uid
        us = hg if is_home else ag
        them = ag if is_home else hg
        if us > them:
            out["w"] += 1
            out["pts"] += 2
        elif ot and us < them:
            out["otl"] += 1
            out["pts"] += 1
        else:
            out["l"] += 1
        out["gp"] += 1
    return out


def get_franchise_game_detail(session: FranchiseSession, game_id: str) -> Optional[Dict[str, Any]]:
    """Return a single saved box score (goals, skater lines, team shot totals) for the recap UI."""
    gid = str(game_id or "").strip()
    if not gid:
        return None
    for g in reversed(getattr(session, "game_results", None) or []):
        if str(g.get("game_id") or "").strip() == gid or _stable_franchise_game_id(g) == gid:
            out = dict(g)
            out["user_team_id"] = str(session.user_team_id)
            return out
    return None


def execute_trade_package(session: FranchiseSession, *, assets_by_team: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Apply a TradeHub-style package where each team column lists assets they acquire.
    Supported assets:
      - {"type":"player","id":"<player_id>","team":"<source_team_id>"}
      - {"type":"pick","id":"...","team":"<source_team_id>"} (logged only)
    """
    if not isinstance(assets_by_team, dict) or not assets_by_team:
        raise ValueError("Trade package is empty.")
    team_ids = [str(k) for k in assets_by_team.keys() if str(k)]
    if len(team_ids) < 2:
        raise ValueError("Trade package requires at least two teams.")

    moved: List[Dict[str, Any]] = []
    seen_player_ids: set[str] = set()

    def _find_team(tid: str) -> Any:
        tm = session.team_by_id.get(str(tid))
        if tm is None:
            raise ValueError(f"Unknown team in trade package: {tid}")
        return tm

    for acquiring_id, assets in assets_by_team.items():
        acq_tid = str(acquiring_id)
        _find_team(acq_tid)
        for raw in assets or []:
            if not isinstance(raw, dict):
                continue
            asset_type = str(raw.get("type") or "").lower()
            source_tid = str(raw.get("team") or "")
            if not source_tid or source_tid == acq_tid:
                continue
            _find_team(source_tid)
            if asset_type != "player":
                moved.append(
                    {
                        "asset_type": asset_type or "pick",
                        "asset_id": str(raw.get("id") or ""),
                        "source_team_id": source_tid,
                        "acquiring_team_id": acq_tid,
                        "applied": False,
                    }
                )
                continue
            pid = str(raw.get("id") or "")
            if not pid:
                continue
            if pid in seen_player_ids:
                raise ValueError(f"Duplicate player in trade package: {pid}")
            seen_player_ids.add(pid)

            source_team = _find_team(source_tid)
            acquiring_team = _find_team(acq_tid)
            src_roster = list(getattr(source_team, "roster", None) or [])
            idx = next((i for i, p in enumerate(src_roster) if str(getattr(p, "id", "") or "") == pid), -1)
            if idx < 0:
                raise ValueError(f"Player {pid} not found on source roster {source_tid}")
            player = src_roster.pop(idx)
            acq_roster = list(getattr(acquiring_team, "roster", None) or [])
            acq_roster.append(player)
            setattr(source_team, "roster", src_roster)
            setattr(acquiring_team, "roster", acq_roster)

            pname = _name_str(player)
            moved.append(
                {
                    "asset_type": "player",
                    "asset_id": pid,
                    "player_name": pname,
                    "source_team_id": source_tid,
                    "acquiring_team_id": acq_tid,
                    "applied": True,
                }
            )

    moved_players = [m for m in moved if m.get("asset_type") == "player" and m.get("applied")]
    if not moved_players:
        raise ValueError("No player movements were applied from this trade package.")

    headline_bits = []
    for m in moved_players[:4]:
        src = _display_team(session.team_by_id.get(m["source_team_id"])) if session.team_by_id.get(m["source_team_id"]) else m["source_team_id"]
        dst = _display_team(session.team_by_id.get(m["acquiring_team_id"])) if session.team_by_id.get(m["acquiring_team_id"]) else m["acquiring_team_id"]
        headline_bits.append(f"{m.get('player_name')}: {src} -> {dst}")
    headline = "TRADE EXECUTED: " + "; ".join(headline_bits)

    notif = _normalize_notification_payload(
        {
            "type": "trade",
            "priority": "HIGH",
            "title": "Trade Executed",
            "headline": headline,
            "text": headline,
            "source": "trade_hub",
        },
        index=len(session.notifications or []),
    )
    session.notifications.append(notif)
    _record_storyline(
        session,
        {
            "type": "trade",
            "priority": "HIGH",
            "headline": headline,
            "details": f"Moved {len(moved_players)} players via Trade Hub package.",
            "players": [str(m.get("player_name") or "") for m in moved_players],
            "team_id": str(session.user_team_id),
        },
    )
    trade_day = int(getattr(session, "calendar_cursor", 0) or 0)
    trade_iso = _calendar_iso_for_day(session, trade_day)

    session.timeline.append(
        _normalized_timeline_event(
            event_id=f"timeline:trade_hub:{trade_day}:{uuid.uuid4().hex[:8]}",
            event_type="trade",
            text=f"TRADE HUB: moved {len(moved_players)} player(s).",
            calendar_day=trade_day,
            calendar_iso=trade_iso,
            team_id=str(session.user_team_id),
            priority="HIGH",
        )
    )
    return {"moved_assets": moved, "headline": headline, "moved_players": len(moved_players)}

def _storyline_choices_payload(session: FranchiseSession) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for d in list(getattr(session, "pending_decisions", None) or []):
        if not isinstance(d, dict):
            continue

        meta = dict(d.get("meta") or {})
        sid = str(meta.get("storyline_id") or d.get("storyline_id") or d.get("id") or "")

        if not sid:
            continue

        options = []

        for opt in list(d.get("options") or d.get("choices") or []):
            if not isinstance(opt, dict):
                continue

            options.append(
                {
                    "id": str(opt.get("id") or opt.get("choice_id") or ""),
                    "label": str(opt.get("label") or opt.get("title") or opt.get("text") or "Choice"),
                    "effects": dict(opt.get("effects") or {}),
                    "effect_summary": str(opt.get("effect_summary") or opt.get("effectSummary") or ""),
                }
            )

        out.append(
            {
                "decision_id": str(d.get("id") or ""),
                "storyline_id": sid,
                "kind": str(d.get("kind") or d.get("type") or "decision"),
                "priority": str(d.get("priority") or "MEDIUM").upper(),
                "title": str(d.get("title") or d.get("headline") or "Decision Required"),
                "description": str(d.get("description") or d.get("message") or d.get("details") or ""),
                "action_options": options,
                "options": options,
                "meta": meta,
            }
        )

    return out
def build_state_payload(session: FranchiseSession) -> Dict[str, Any]:
    _sync_nhl_calendar_bounds(session)
    # Schedule cadence is smoothed in start_franchise only. Re-running _smooth_league_schedule
    # here made every GET /api/franchise/state take minutes (full league re-optimization).
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
            row = _serialize_player_row(p, include_ratings=True, session=session, _team=user_team)
            pid = row.get("player_id") or ""
            st = session.player_season_stats.get(pid)
            if st:
                row["season_stats"] = {
                    "gp": int(st.get("gp", 0)),
                    "g": int(st.get("g", 0)),
                    "a": int(st.get("a", 0)),
                    "pts": int(st.get("pts", 0)),
                    "sog": int(st.get("sog", 0)),
                    "pim": int(st.get("pim", 0)),
                    "hit": int(st.get("hit", st.get("hits", 0)) or 0),
                    "blk": int(st.get("blk", st.get("blocks", 0)) or 0),
                    "toi": round((int(st.get("toi_sec", 0) or 0) / max(1, int(st.get("gp", 0) or 1))) / 60.0, 1),
                    "ga": int(st.get("ga", 0)),
                    "w": int(st.get("w", 0)),
                    "l": int(st.get("l", 0)),
                    "otl": int(st.get("otl", 0)),
                }
            roster_rows.append(row)
        roster_rows.sort(key=lambda x: -float(x.get("ovr") or 0))

    cal_rec = _user_team_record_from_game_results(session)
    uid_s = str(session.user_team_id or "")

    standings_rows: List[Dict[str, Any]] = []
    if session.standings:
        for tid, r in session.standings.records.items():
            tid_s = str(tid)
            if uid_s and tid_s == uid_s and int(cal_rec.get("gp") or 0) > 0:
                standings_rows.append(
                    {
                        "team_id": tid_s,
                        "name": getattr(r, "name", tid),
                        "gp": int(cal_rec["gp"]),
                        "w": int(cal_rec["w"]),
                        "l": int(cal_rec["l"]),
                        "otl": int(cal_rec["otl"]),
                        "pts": int(cal_rec["pts"]),
                    }
                )
            else:
                standings_rows.append(
                    {
                        "team_id": tid_s,
                        "name": getattr(r, "name", tid),
                        "gp": getattr(r, "gp", 0),
                        "w": getattr(r, "wins", 0),
                        "l": getattr(r, "losses", 0),
                        "otl": getattr(r, "otl", 0),
                        "pts": getattr(r, "points", 0),
                    }
                )
        standings_rows.sort(key=lambda x: (-x["pts"], -(x["w"] - x["l"])))

    cap_info = _team_cap_snapshot(user_team, sim) if user_team is not None else {"salary_cap": 92.0, "cap_hit": 0.0, "cap_space": 92.0}
    cap_hint = str(getattr(user_team, "cap_pressure", "moderate") if user_team else "?")
    strat = str(getattr(user_team, "strategy", "balanced") if user_team else "?")

    day_display = "Off-season"
    prog = None
    nhl_today = _nhl_today_payload(session)
    nhl_strip = _nhl_calendar_strip(session)
    season_lbl = f"{session.season_calendar_year}ΓÇô{int(session.season_calendar_year) + 1}"
    if session.phase == "regular" and session.nhl_calendar:
        last = int(session.nhl_regular_season_last_index)
        cur = int(session.calendar_cursor)
        if cur <= last:
            cd = session.nhl_calendar[cur]
            wd = str(cd.get("weekday") or "").strip()
            day_display = (
                f"Next league day: {cd.get('iso', '')}"
                + (f" ({wd})" if wd else "")
                + f" ΓÇö {cd.get('ui_phase', '')}"
            )
            prog = f"{cur + 1} / {last + 1}"
        else:
            day_display = "Regular season complete ΓÇö advance for playoffs"
            prog = f"{last + 1} / {last + 1}"
    elif session.phase == "complete":
        day_display = f"Season complete ΓÇö Cup: {session.champion_id or '?'}"

    try:
        _merge_simengine_league_news_into_storylines(session)
    except Exception:
        pass

    notifications_raw = list(session.notifications[-56:])
    notifications_norm = [_normalize_notification_payload(n, i) for i, n in enumerate(notifications_raw)]
    storylines_norm = [_normalize_storyline_payload(ev if isinstance(ev, dict) else {"headline": str(ev or "")}) for ev in list(getattr(session, "storyline_events", None) or [])[-300:]]
    storyline_choices = _storyline_choices_payload(session)
    injuries_payload = _build_injuries_payload(session)
    injury_history_payload = _build_injury_history_payload(session)

    return {
        "session_id": session.session_id,
        "user_team_id": str(session.user_team_id),
        "phase": session.phase,
        "season_year": session.season_calendar_year,
        "games_per_team_schedule": int(getattr(session, "games_per_team_schedule", 82) or 82),
        "calendar_summary": day_display,
        "progress": prog,
        "nhl_season_label": season_lbl,
        "nhl_today": nhl_today,
        "nhl_calendar_strip": nhl_strip,
        "nhl_calendar_full": _nhl_calendar_full_with_slates(session),
        "season_anchor_events": season_anchor_event_markers(int(session.season_calendar_year)),
        "team": {
            "id": session.user_team_id,
            "name": _display_team(user_team) if user_team else session.user_team_id,
            "coach": session.head_coach_name,
            "coach_archetype": session.coach_archetype,
            "salary_cap": float(cap_info["salary_cap"]),
            "cap_hit": float(cap_info["cap_hit"]),
            "cap_space": float(cap_info["cap_space"]),
            "cap_limit": float(cap_info["salary_cap"]),
            "record": (
                {
                    "gp": int(cal_rec["gp"]),
                    "w": int(cal_rec["w"]),
                    "l": int(cal_rec["l"]),
                    "otl": int(cal_rec["otl"]),
                    "pts": int(cal_rec["pts"]),
                }
                if int(cal_rec.get("gp") or 0) > 0
                else (
                    {
                        "gp": getattr(rec, "gp", 0),
                        "w": getattr(rec, "wins", 0),
                        "l": getattr(rec, "losses", 0),
                        "otl": getattr(rec, "otl", 0),
                        "pts": getattr(rec, "points", 0),
                    }
                    if rec
                    else None
                )
            ),
            "cap_pressure": cap_hint,
            "strategy": strat,
        },
        "pending_decisions": list(session.pending_decisions),
        "pending_decisions": _pending_decision_snapshot(session),
                    "pendingDecisions": _pending_decision_snapshot(session),
                    "pending_ui_popups": list(getattr(session, "pending_ui_popups", None) or []),
                    "pendingUiPopups": list(getattr(session, "pending_ui_popups", None) or []),
        "storyline_choices": storyline_choices,
        "notifications": notifications_norm,
        "timeline": list(session.timeline[-80:]),
        "storyline_events": storylines_norm,
        "injuries": injuries_payload,
        "injury_history": injury_history_payload,
        "roster": roster_rows[:28],
        "calendar_events": list(getattr(session, "calendar_events", []) or []),
        "schedule_diagnostics": getattr(session, "schedule_diagnostics", {}) or {},
        "pending_decisions": list(getattr(session, "pending_decisions", []) or []),
        "roster_browser": _build_roster_browser(sim, str(session.user_team_id), franchise_session=session),
        "draft_class_rankings": build_draft_class_rankings(session, sim),
        "standings": standings_rows[:32],
        "stats_central": _build_stats_central_payload(session),
        "schedule_upcoming": _build_schedule_upcoming(session, limit=14),
        "flags": {
            "playoffs_done": session.playoffs_simulated,
            "can_advance": len(session.pending_decisions) == 0 and session.phase != "complete",
        },
        "pending_ui_popups": list(getattr(session, "pending_ui_popups", None) or []),
        "showcase_archive": list(getattr(session, "showcase_archive", None) or [])[-24:],
    }
    try:
        from app.sim_engine.franchise.offseason import build_offseason_state_extras

        extras = build_offseason_state_extras(session)
        extra_flags = dict(extras.pop("flags", {}) or {})
        payload.update(extras)
        payload["flags"] = {**payload.get("flags", {}), **extra_flags}
        payload["phase"] = str(session.phase)
        payload["season_phase"] = str(getattr(session, "season_phase", session.phase) or session.phase)
        payload["next_important_event"] = str(getattr(session, "next_important_event", "") or "")
        payload["playoff_payload"] = dict(getattr(session, "playoff_payload", None) or {})
    except Exception:
        pass
    return payload
