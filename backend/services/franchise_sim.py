"""
Interactive franchise day engine: advances the league calendar one day at a time
by reusing SimEngine's season simulation loop (per-game) without modifying SimEngine.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import math
import os
import random
import threading
import time
import uuid
from dataclasses import is_dataclass, replace
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from services.franchise_paths import ensure_simengine_path

ensure_simengine_path()
import run_sim as rs  # noqa: E402

_startup_log = logging.getLogger("uvicorn.error")
_TEAM_SUMMARY_CACHE: Optional[List[Dict[str, str]]] = None
# Per-session locks kept off FranchiseSession so pickle/save clones stay valid.
_DRAFT_RANKINGS_LOCKS: Dict[str, threading.Lock] = {}
_DRAFT_RANKINGS_LOCKS_GUARD = threading.Lock()


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
    clamp_height_cm_for_position,
)
from app.sim_engine.league import (
    compute_awards,
    generate_regular_season_schedule,
    simulate_playoffs,
)
from app.sim_engine.league.schedule_generator import GameSlot, _safe_team_id, _safe_id_str, _safe_slot_team_id
from app.sim_engine.league.standings import StandingsTable

from services.franchise_session import FranchiseSession
from services.nhl_season_calendar import (
    build_season_calendar,
    calendar_day_to_dict,
    current_nhl_season_start_year,
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

# Primary AHL affiliates for career / roster LG labeling (2024–26 map).
_NHL_AHL_AFFILIATE_BY_ABBR: Dict[str, str] = {
    "ANA": "San Diego Gulls",
    "BOS": "Providence Bruins",
    "BUF": "Rochester Americans",
    "CGY": "Calgary Wranglers",
    "CAR": "Chicago Wolves",
    "CHI": "Rockford IceHogs",
    "COL": "Colorado Eagles",
    "CBJ": "Cleveland Monsters",
    "DAL": "Texas Stars",
    "DET": "Grand Rapids Griffins",
    "EDM": "Bakersfield Condors",
    "FLA": "Charlotte Checkers",
    "LAK": "Ontario Reign",
    "MIN": "Iowa Wild",
    "MTL": "Laval Rocket",
    "NSH": "Milwaukee Admirals",
    "NJD": "Utica Comets",
    "NYI": "Bridgeport Islanders",
    "NYR": "Hartford Wolf Pack",
    "OTT": "Belleville Senators",
    "PHI": "Lehigh Valley Phantoms",
    "PIT": "Wilkes-Barre/Scranton Penguins",
    "SEA": "Coachella Valley Firebirds",
    "SJS": "San Jose Barracuda",
    "STL": "Springfield Thunderbirds",
    "TBL": "Syracuse Crunch",
    "TOR": "Toronto Marlies",
    "UTA": "Tucson Roadrunners",
    "VAN": "Abbotsford Canucks",
    "VGK": "Henderson Silver Knights",
    "WSH": "Hershey Bears",
    "WPG": "Manitoba Moose",
}


def _ahl_affiliate_display_name(team: Any) -> str:
    """Resolve the AHL club name for an NHL parent (e.g. Ottawa → Belleville Senators)."""
    if team is None:
        return "AHL Affiliate"
    for attr in ("ahl_team_name", "affiliate_name", "ahl_name", "farm_team_name"):
        raw = getattr(team, attr, None)
        if raw and str(raw).strip():
            return str(raw).strip()
    abbr = ""
    try:
        abbr = str(_franchise_team_abbrev(team) or "").upper()
    except Exception:
        abbr = ""
    if abbr in _NHL_AHL_AFFILIATE_BY_ABBR:
        return _NHL_AHL_AFFILIATE_BY_ABBR[abbr]
    city = str(getattr(team, "city", "") or "").strip()
    name = str(getattr(team, "name", "") or "").strip()
    # Fallback: keep parent nickname with an AHL tag rather than inventing a city.
    if name:
        return f"{city} {name} (AHL)".strip() if city else f"{name} (AHL)"
    return "AHL Affiliate"


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
        h, a = _team_ids_for_slot(sl)
        if h == tid or a == tid:
            return True
    return False


def _slot_team_ids_cached(slot: Any) -> Tuple[str, str]:
    """Normalize slot team ids once and memoize on slot objects when possible."""
    try:
        h = getattr(slot, "_norm_home_id", None)
        a = getattr(slot, "_norm_away_id", None)
        if isinstance(h, str) and isinstance(a, str):
            return h, a
    except Exception:
        pass
    h = _safe_slot_team_id(slot, "home_id")
    a = _safe_slot_team_id(slot, "away_id")
    try:
        setattr(slot, "_norm_home_id", h)
        setattr(slot, "_norm_away_id", a)
    except Exception:
        pass
    return h, a


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
    h, a = _slot_team_ids_cached(slot)
    return (h, a, bool(getattr(slot, "is_playoff", False)))


def _team_ids_for_slot(slot: Any) -> Tuple[str, str]:
    return _slot_team_ids_cached(slot)


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
    team_days_by_team: Optional[Dict[str, List[int]]] = None,
    team_days_set_by_team: Optional[Dict[str, set]] = None,
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

    if home_id == "" or away_id == "":
        return False

    if home_id == away_id:
        return False

    if team_days_set_by_team is not None:
        if target_day in (team_days_set_by_team.get(home_id) or set()):
            return False
    elif _team_plays_on_day(by_day, target_day, home_id):
        return False

    if team_days_set_by_team is not None:
        if target_day in (team_days_set_by_team.get(away_id) or set()):
            return False
    elif _team_plays_on_day(by_day, target_day, away_id):
        return False

    if _would_create_bad_cadence_for_slot(
        slot,
        target_day,
        by_day,
        old_day=old_day,
        max_games_in_4=3,
        max_games_in_7=4,
        team_days_by_team=team_days_by_team,
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
    for tid in sorted({_safe_id_str(t_home), _safe_id_str(t_away)}):
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
    team_days_by_team: Optional[Dict[str, List[int]]] = None,
    team_days_set_by_team: Optional[Dict[str, set]] = None,
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

    if team_days_set_by_team is not None:
        if target_day in (team_days_set_by_team.get(t1) or set()):
            return False
    elif team_days_by_team is not None:
        if target_day in set(int(x) for x in (team_days_by_team.get(t1, []) or [])):
            return False
    elif _team_plays_on_day(by_day, target_day, t1):
        return False
    if team_days_set_by_team is not None:
        if target_day in (team_days_set_by_team.get(t2) or set()):
            return False
    elif team_days_by_team is not None:
        if target_day in set(int(x) for x in (team_days_by_team.get(t2, []) or [])):
            return False
    elif _team_plays_on_day(by_day, target_day, t2):
        return False

    # HARD cadence rules: do not allow smoothing to create unrealistic stretches.
    if _would_create_bad_cadence_for_slot(
        slot,
        target_day,
        by_day,
        old_day=old_day,
        max_games_in_4=3,
        max_games_in_7=4,
        team_days_by_team=team_days_by_team,
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
    team_days_by_team: Optional[Dict[str, List[int]]] = None,
) -> bool:
    """
    Hard rejection used by the schedule smoother.
    Prevents the move from creating 4-in-4 or 5-in-7 type nonsense.
    """
    target_day = int(target_day)
    team_ids = [x for x in _team_ids_for_slot(slot) if x]

    old_day_i = int(old_day) if old_day is not None else None
    key = _slot_key(slot)
    for tid in team_ids:
        if team_days_by_team is not None:
            base = [int(d) for d in (team_days_by_team.get(tid, []) or [])]
            if old_day_i is not None:
                base = [d for d in base if d != old_day_i]
            days = sorted(set(base + [target_day]))
        else:
            days = []
            for d, slots in (by_day or {}).items():
                di = int(d)
                for sl in slots or []:
                    if old_day_i is not None and di == old_day_i and _slot_key(sl) == key:
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


def _repair_impossible_cadence(
    by_day: Dict[int, List[Any]],
    nhl_cal: List[Dict[str, Any]],
    *,
    max_moves: int = 240,
) -> Dict[int, List[Any]]:
    """Move slots that create 4-in-4 / 5-in-7 windows onto free eligible days."""
    fixed: Dict[int, List[Any]] = {int(k): list(v or []) for k, v in (by_day or {}).items()}
    eligible = [
        i
        for i, row in enumerate(nhl_cal or [])
        if str((row or {}).get("segment") or (row or {}).get("season_segment") or "") == "regular"
        and (row or {}).get("allows_games", (row or {}).get("allowsGames", True)) is not False
    ]
    if not eligible:
        return fixed

    moves = 0
    for _pass in range(10):
        if moves >= max_moves:
            break
        team_days = _build_team_game_days(fixed, nhl_cal=nhl_cal, regular_only=True)
        progressed = False
        for tid, days in sorted(team_days.items(), key=lambda kv: -len(kv[1])):
            bad = _team_has_impossible_cadence(days)
            if not bad:
                continue
            ds = sorted(int(x) for x in days)
            worst_start = None
            worst_count = 0
            lo, hi = ds[0], ds[-1]
            for start in range(lo, hi + 1):
                window = [d for d in ds if start <= d <= start + 6]
                if len(window) > worst_count:
                    worst_count = len(window)
                    worst_start = start
            if worst_start is None or worst_count < 5:
                for start in range(lo, hi + 1):
                    window = [d for d in ds if start <= d <= start + 3]
                    if len(window) >= 4:
                        worst_start = start
                        worst_count = len(window)
                        break
            if worst_start is None:
                continue
            window_days = [d for d in ds if worst_start <= d <= worst_start + 6]
            if len(window_days) < 4:
                continue
            # Try moving each game in the dense window until one succeeds.
            for move_day in list(reversed(window_days)):
                candidates = [
                    sl
                    for sl in (fixed.get(move_day, []) or [])
                    if _slot_has_team(sl, tid)
                ]
                if not candidates:
                    continue
                slot = candidates[0]
                team_days_map = _build_team_game_days(fixed, nhl_cal=nhl_cal, regular_only=True)
                placed = False
                for delta in range(1, 28):
                    for nd in (move_day + delta, move_day - delta):
                        if nd not in eligible:
                            continue
                        if not _can_place_slot_on_day(
                            slot,
                            nd,
                            fixed,
                            old_day=move_day,
                            eligible_set=set(eligible),
                            max_games_per_day=16,
                            nhl_cal=nhl_cal,
                            team_days_by_team=team_days_map,
                        ):
                            continue
                        _move_slot(fixed, slot, move_day, nd)
                        moves += 1
                        progressed = True
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if moves >= max_moves:
                break
        if not progressed:
            break
    return fixed


def _merge_abstract_schedule_to_by_day(
    by_abs: Dict[int, List[Any]],
    abstract_keys: List[int],
    day_map: Dict[int, int],
    nhl_cal: List[Dict[str, Any]],
) -> Tuple[Dict[int, List[Any]], List[Any]]:
    """Map abstract schedule days onto NHL calendar indices (mirrors franchise start)."""
    del nhl_cal  # reserved for future calendar-aware slot placement
    by_day: Dict[int, List[Any]] = defaultdict(list)
    schedule: List[Any] = []
    for old in abstract_keys:
        nid = int(day_map[int(old)])
        for slot in by_abs[old]:
            gs = GameSlot(day=nid, home_id=slot.home_id, away_id=slot.away_id, is_playoff=slot.is_playoff)
            by_day[nid].append(gs)
            schedule.append(gs)
    return dict(by_day), schedule


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
        # Second pass after denser regular-only mapping — clears residual last-day books.
        fixed = _repair_regular_day_conflicts(fixed, nhl_cal)
        _franchise_startup_stage("schedule finalize: conflict repair complete")
    except Exception as e:
        _startup_log.exception(
            "[franchise start] _repair_regular_day_conflicts failed; continuing with pre-repair slate."
        )
        repair_error = str(e)

    try:
        _franchise_startup_stage("schedule finalize: cadence repair")
        fixed = _repair_impossible_cadence(fixed, nhl_cal)
        _franchise_startup_stage("schedule finalize: cadence repair complete")
    except Exception as e:
        _startup_log.exception(
            "[franchise start] _repair_impossible_cadence failed; continuing with pre-cadence slate."
        )
        if not repair_error:
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
    team_days = _build_team_game_days(by_day, nhl_cal=nhl_cal, regular_only=True)
    team_days_set = {tid: set(days) for tid, days in team_days.items()}

    def _candidate_repair_days(old_day: int) -> List[int]:
        """
        Prefer close dates around the original slot, alternating future/past.
        Always fall through to the full regular calendar by distance so late-season
        conflicts (e.g. final day double-books after denser regular mapping) can move.
        """
        old_day = int(old_day)
        out: List[int] = []
        seen: set = set()

        for radius in range(1, 22):
            for cand in (old_day + radius, old_day - radius):
                if cand in seen:
                    continue
                seen.add(cand)
                if cand in regular_days:
                    out.append(cand)

        # Full-season fallback by distance (deduped).
        for cand in sorted(regular_days, key=lambda d: (abs(int(d) - old_day), int(d))):
            if cand not in seen and int(cand) != old_day:
                seen.add(cand)
                out.append(int(cand))

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
                team_days_by_team=team_days,
                team_days_set_by_team=team_days_set,
            ):
                return int(cand)

        # Last-resort: allow a slightly denser day (19) rather than leave a double-book.
        for cand in _candidate_repair_days(old_day):
            if _slot_would_be_legal_after_move(
                slot,
                cand,
                by_day,
                nhl_cal,
                old_day=old_day,
                max_games_per_day=20,
                team_days_by_team=team_days,
                team_days_set_by_team=team_days_set,
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
            # Incrementally update cached team-day indexes for cadence/occupancy checks.
            if d != int(target):
                for tid in (h, a):
                    if not tid:
                        continue
                    s = team_days_set.setdefault(tid, set())
                    if d in s:
                        s.discard(d)
                    s.add(int(target))
                    team_days[tid] = sorted(s)
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
            team_days_set = {tid: set(days) for tid, days in team_days.items()}
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
                            team_days_by_team=team_days,
                            team_days_set_by_team=team_days_set,
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
        from app.sim_engine.franchise.common import _same_calendar_day  # noqa: WPS433

        same_day_injury_popups = [
            p
            for p in session.pending_ui_popups
            if isinstance(p, dict)
            and str(p.get("kind") or "") == "injury"
            and _same_calendar_day(p, cur_date, calendar_iso)
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
    player_universe: str = "generated",
) -> FranchiseSession:
    ensure_simengine_path()
    from app.sim_engine.engine import SimEngine

    universe = str(player_universe or "generated").strip().lower()
    if universe not in ("generated", "real_nhl"):
        raise ValueError('player_universe must be "generated" or "real_nhl".')

    _franchise_startup_stage("SimEngine import complete; constructing engine")
    master = seed if seed is not None else random.randrange(1, 10**9)
    sim = SimEngine(
        seed=master,
        debug=False,
        populate_initial_rosters=(universe == "generated"),
    )
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

    season_y = (
        int(season_start_year)
        if season_start_year is not None
        else current_nhl_season_start_year()
    )
    try:
        from app.sim_engine.economy.cap_engine import apply_nhl_salary_cap_for_season

        apply_nhl_salary_cap_for_season(league, season_y)
    except Exception:
        pass

    if universe == "real_nhl":
        _franchise_startup_stage("preparing real NHL roster universe")
        try:
            from services.real_nhl_roster_importer import build_real_nhl_league_players
        except ImportError as e:
            raise ValueError(
                "Real NHL Players mode is unavailable on this install. "
                "Switch to Generated Players or restore real_nhl_roster_importer.py."
            ) from e
        try:
            build_real_nhl_league_players(
                teams=teams,
                league=league,
                rng=sim.rng,
                season_year=season_y,
            )
        except Exception as e:
            code = getattr(e, "code", None) or "REAL_NHL_ROSTER_IMPORT_FAILED"
            message = getattr(e, "message", None) or str(e)
            raise ValueError(f"{message} (retry or switch to Generated Players) [{code}]") from e
        _franchise_startup_stage("real NHL roster universe ready")

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
        player_universe=universe,
        preseason_applied=True,
        phase="preseason",
        season_phase="preseason",
    )
    try:
        from services.franchise_store import live_code_revision  # noqa: WPS433

        session.code_revision = str(live_code_revision() or "")
    except Exception:
        session.code_revision = ""
    session.schedule_diagnostics = schedule_diagnostics
    session.notifications = getattr(session, "notifications", None) or []
    session.timeline = getattr(session, "timeline", None) or []
    session.pending_ui_popups = getattr(session, "pending_ui_popups", None) or []
    session.calendar_events = getattr(session, "calendar_events", None) or []
    session.pending_decisions = getattr(session, "pending_decisions", None) or []
    _sync_session_phase_from_calendar(session)
    ensure_session_nhl_salary_cap(session)
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

        bootstrap_full_league_hierarchy(league, sim.rng, season_year=season_y)
        if universe == "real_nhl":
            from services.real_nhl_roster_importer import (
                enforce_opening_night_cap_compliance,
                trim_team_roster_to_nhl_limit,
            )

            for tm in teams:
                trim_team_roster_to_nhl_limit(tm)
            try:
                from services.brady_tkachuk_chaos import (
                    apply_brady_chaos_to_league,
                    inject_brady_storylines,
                )

                apply_brady_chaos_to_league(teams)
                inject_brady_storylines(session, team_abbr="OTT")
            except Exception as brady_err:
                session.notifications.append(f"Brady chaos skipped: {brady_err}")
        npl = len(getattr(league, "players", None) or [])
        session.notifications.append(
            f"League depth online ΓÇö NHL affiliates (AHL/ECHL), UFA pools, overseas, juniors (~{npl} player records)."
        )
    except Exception as e:
        session.notifications.append(f"League depth bootstrap skipped: {e}")
    try:
        n_contracts = _ensure_league_roster_contracts(league, season_y)
        session.financials_status = "ready" if n_contracts >= 0 else "partial"
        _franchise_startup_stage(f"roster contracts bootstrapped ({n_contracts} generated)")
        if universe == "real_nhl":
            from services.real_nhl_roster_importer import enforce_opening_night_cap_compliance

            cap_fix = enforce_opening_night_cap_compliance(league, season_y)
            if cap_fix.get("players_demoted"):
                session.notifications.append(
                    f"Opening-night cap trim: sent {cap_fix['players_demoted']} players to AHL."
                )
            if cap_fix.get("still_over"):
                session.notifications.append(
                    f"Cap notes: {len(cap_fix['still_over'])} clubs still over (NMC / LTIR-style pressure)."
                )
        from services.contract_economy import validate_franchise_cap_at_start
        cap_issues = validate_franchise_cap_at_start(league, season_y)
        if cap_issues:
            session.notifications.append(f"Cap compliance warnings: {len(cap_issues)} teams trimmed")
        from services.contract_economy import install_prospect_contract_hooks

        install_prospect_contract_hooks(league)
    except Exception as e:
        session.financials_status = "failed"
        session.notifications.append(f"Contract bootstrap incomplete: {e}")
    try:
        _sync_prospect_stats_to_calendar(session)
        # Defer initial draft board rank snapshot until first real board access.
        # This avoids paying draft ranking compute cost during franchise creation.
        session.draft_rank_prev = {}
    except Exception:
        pass
    # Warm rankings/HUD off the request thread so the first Draft Class open is a
    # cache hit. Ranking formulas are unchanged — this only precomputes the board.
    try:
        _schedule_draft_class_cache_warm(session)
    except Exception:
        pass
    _franchise_startup_stage("start_franchise complete; returning session")
    return session


def _schedule_draft_class_cache_warm(session: FranchiseSession) -> None:
    """Background single-flight warm of draft rankings + HUD after franchise start."""
    if getattr(session, "_draft_class_warm_scheduled", False):
        return
    try:
        session._draft_class_warm_scheduled = True
    except Exception:
        pass

    def _warm() -> None:
        try:
            sim = getattr(session, "sim", None)
            if sim is None:
                return
            board = get_cached_draft_class_rankings(session, sim)
            user_team = None
            try:
                user_team = (getattr(session, "team_by_id", None) or {}).get(
                    str(getattr(session, "user_team_id", "") or "")
                )
            except Exception:
                user_team = None
            get_cached_draft_class_hud(
                session,
                user_team,
                {},
                [],
                (board or {}).get("entries"),
            )
        except Exception:
            pass

    threading.Thread(target=_warm, name="draft-class-warm", daemon=True).start()


def _name_str(p: Any) -> str:
    ident = getattr(p, "identity", None)
    raw = str(getattr(ident, "name", None) or "?")
    try:
        from services.brady_tkachuk_chaos import display_name_with_cancer_tag, is_brady_tkachuk

        if is_brady_tkachuk(p):
            return display_name_with_cancer_tag(raw, p)
    except Exception:
        pass
    return raw


def _pos_str(p: Any) -> str:
    """Normalize position the same way SimEngine does (identity, then player.position)."""
    ident = getattr(p, "identity", None)
    pos = getattr(ident, "position", None) if ident else None
    if pos is None or str(getattr(pos, "value", pos) or "").strip() in ("", "?"):
        pos = getattr(p, "position", None)
    raw = str(getattr(pos, "value", pos) or "?").strip().upper()
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw or "?"


def _is_goalie_player(p: Any) -> bool:
    return _pos_str(p).upper() == "G"


def _player_cap_hit_millions(player: Any) -> float:
    from app.sim_engine.economy.cap_engine import player_cap_hit_millions as _cap_hit
    return _cap_hit(player)


def _team_cap_snapshot(team: Any, sim: Any, *, season_year: int | None = None) -> Dict[str, float]:
    from services.contract_economy import get_team_cap_snapshot_full, team_cap_snapshot_legacy_compat
    league = getattr(sim, "league", None)
    if season_year is None:
        season_year = (
            getattr(sim, "season_calendar_year", None)
            or getattr(league, "season_year", None)
            or getattr(league, "current_season_year", None)
        )
        try:
            season_year = int(season_year) if season_year is not None else None
        except (TypeError, ValueError):
            season_year = None
    snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
    return team_cap_snapshot_legacy_compat(snap)


# =============================================================================
# Roster contract bootstrap — every NHL player gets a real contract so cap
# math is the sum of actual deals, not a UI placeholder.
# =============================================================================

LEAGUE_MINIMUM_AAV_M = 0.775
CAP_SAFE_CORE_TOP_N = 6
CAP_SAFE_STAR_OVR = 88.0
BOOTSTRAP_CAP_HEADROOM_MIN_M = 2.5
BOOTSTRAP_CAP_HEADROOM_MAX_M = 9.0


def _resolve_league_salary_cap_m(league: Any, season_year: int | None = None) -> float:
    """Single source for league upper limit in millions."""
    sy = season_year
    if sy is None and league is not None:
        try:
            sy = int(getattr(league, "season_year", None) or getattr(league, "season_start_year", None) or 0) or None
        except Exception:
            sy = None
    if sy is not None:
        try:
            from app.sim_engine.economy.cap_engine import apply_nhl_salary_cap_for_season, nhl_upper_limit_millions

            if league is not None:
                apply_nhl_salary_cap_for_season(league, int(sy))
            # Session/calendar season → published NHL table wins over any stale
            # league.salary_cap_m (commonly still $88 on 2025+ saves).
            return float(nhl_upper_limit_millions(int(sy)))
        except Exception:
            pass
    if league is not None:
        try:
            from app.sim_engine.economy.cap_engine import _league_cap_bounds_millions

            bounds = _league_cap_bounds_millions(league, None, season_start_year=sy)
            upper = float(bounds.get("upper") or 0.0)
            if upper > 0:
                return upper
        except Exception:
            pass
        try:
            ctx = league.get_league_context() if hasattr(league, "get_league_context") else {}
            econ = (ctx or {}).get("economics") or {}
            raw = float(econ.get("salary_cap", 0) or 0)
            if raw > 0:
                return raw / 1_000_000.0 if raw > 250 else raw
        except Exception:
            pass
    raw = float(getattr(league, "salary_cap_m", 0) or getattr(league, "salary_cap", 0) or 0) if league is not None else 0.0
    if raw <= 0:
        try:
            from app.sim_engine.economy.cap_engine import nhl_upper_limit_millions

            return float(nhl_upper_limit_millions(sy if sy is not None else getattr(league, "season_year", 2025)))
        except Exception:
            return 95.5
    return raw / 1_000_000.0 if raw > 250 else raw


def _sync_session_phase_from_calendar(session: FranchiseSession) -> None:
    """Keep session.phase aligned with the NHL calendar segment (preseason/regular)."""
    phase = str(getattr(session, "phase", "") or "").lower()
    if phase in ("playoffs", "playoff_ready", "post_cup", "complete"):
        return
    if phase == "offseason" and getattr(session, "offseason_stage", None):
        return
    cal = list(getattr(session, "nhl_calendar", None) or [])
    if not cal:
        return
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    cur = max(0, min(cur, len(cal) - 1))
    row = cal[cur] if isinstance(cal[cur], dict) else {}
    seg = str(row.get("segment") or row.get("season_segment") or "").lower()
    if seg == "preseason":
        session.phase = "preseason"
        session.season_phase = "preseason"
    elif seg == "regular":
        if phase in ("", "preseason", "regular"):
            session.phase = "regular"
            session.season_phase = "regular"
    elif seg == "offseason" and phase in ("", "preseason", "regular", "offseason"):
        # Calendar offseason days before an explicit offseason pipeline starts.
        if not getattr(session, "offseason_stage", None):
            session.phase = "offseason"
            session.season_phase = "offseason"


def ensure_session_nhl_salary_cap(session: FranchiseSession) -> float:
    """Always stamp the live NHL upper limit for the session season onto the league."""
    sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
    league = getattr(getattr(session, "sim", None), "league", None)
    try:
        from app.sim_engine.economy.cap_engine import apply_nhl_salary_cap_for_season, nhl_upper_limit_millions

        if league is not None:
            apply_nhl_salary_cap_for_season(league, sy)
        return float(nhl_upper_limit_millions(sy))
    except Exception:
        return float(_resolve_league_salary_cap_m(league, season_year=sy))


def _team_nhl_payroll_m(team: Any) -> float:
    total = 0.0
    for p in _active_roster(team):
        hit = _player_cap_hit_millions(p)
        if hit > 0:
            total += hit
    return float(total)


def _set_player_contract_aav(player: Any, aav_m: float, season_year: int) -> None:
    """Update or create bootstrap contract with a cap-safe AAV (millions)."""
    aav = max(0.0, round(float(aav_m), 3))
    contract = getattr(player, "contract", None)
    if contract is None:
        age = _player_age_int(player)
        ovr = _player_ovr99(player)
        pos = _pos_str(player)
        seed = abs(hash(str(getattr(player, "id", "") or _name_str(player)))) & 0xFFFFFFFF
        _, years = _generate_contract_terms(ovr, age, pos, random.Random(seed))
        if age <= 23 and ovr < 82:
            contract_type = "RFA_BRIDGE"
            rights = "RFA"
        elif age < 27:
            contract_type = "STANDARD"
            rights = "RFA" if age < 25 else "UFA"
        else:
            contract_type = "STANDARD"
            rights = "UFA"
        contract = _GeneratedContract(
            aav_m=aav,
            years=max(1, int(years)),
            expiry_year=int(season_year) + max(1, int(years)),
            contract_type=contract_type,
            rights_status=rights,
            two_way=bool(ovr < 72 and age <= 25),
        )
        try:
            player.contract = contract
        except Exception:
            pass
    else:
        try:
            contract.cap_hit_m = aav
            contract.aav_m = aav
            contract.salary_m = aav
        except Exception:
            pass
    try:
        player.cap_hit_m = aav
    except Exception:
        pass


def _sync_team_cap_fields(team: Any, league: Any) -> None:
    """Persist payroll + cap space on the team object for serializers/UI."""
    from services.contract_economy import sync_team_cap_fields
    sync_team_cap_fields(team, league)


def _protected_core_player_ids(roster: List[Any]) -> set:
    if not roster:
        return set()
    by_ovr = sorted(roster, key=lambda p: -_player_ovr99(p))
    protected = {id(p) for p in by_ovr[:CAP_SAFE_CORE_TOP_N]}
    for p in roster:
        if _player_ovr99(p) >= CAP_SAFE_STAR_OVR:
            protected.add(id(p))
    return protected


def _rebalance_team_cap_compliance(
    team: Any,
    salary_cap_m: float,
    season_year: int,
    rng: random.Random,
    league: Any = None,
) -> bool:
    from services.contract_economy import rebalance_team_cap_at_bootstrap
    lg = league or getattr(team, "_bootstrap_league_ref", None)
    return rebalance_team_cap_at_bootstrap(team, lg, season_year, rng)


def _validate_team_cap_non_negative(
    team: Any,
    league: Any,
    season_year: int,
    rng: random.Random,
) -> Dict[str, float]:
    """Final guarantee: team cap space is never negative after bootstrap."""
    from services.real_nhl_contracts import is_real_nhl_contract

    cap_m = _resolve_league_salary_cap_m(league)
    payroll_m = _team_nhl_payroll_m(team)
    # Real NHL imports keep Spotrac AAVs; soft headroom / proportional trims would
    # destroy accurate contracts. Only hard-cap rebalance when truly over (rare).
    real_nhl_league = bool(getattr(league, "real_nhl_import_meta", None))
    if payroll_m > cap_m + 1e-6 and not real_nhl_league:
        _rebalance_team_cap_compliance(team, cap_m, season_year, rng, league=league)
        payroll_m = _team_nhl_payroll_m(team)
    if payroll_m > cap_m + 1e-6 and not real_nhl_league:
        roster = _active_roster(team)
        if roster and payroll_m > 0:
            from services.contract_economy import has_true_elc_contract
            factor = cap_m / payroll_m
            for p in roster:
                if has_true_elc_contract(p) or is_real_nhl_contract(p):
                    continue
                cur = _player_cap_hit_millions(p)
                _set_player_contract_aav(p, max(0.5, round(cur * factor, 3)), season_year)
            payroll_m = _team_nhl_payroll_m(team)

    if real_nhl_league:
        # Keep authentic payroll; sync fields and exit without inventing headroom.
        cap_space_m = max(0.0, cap_m - payroll_m)
        _sync_team_cap_fields(team, league)
        return {
            "salary_cap": round(cap_m, 3),
            "cap_hit": round(payroll_m, 3),
            "cap_space": round(cap_space_m, 3),
        }

    headroom_target = rng.uniform(BOOTSTRAP_CAP_HEADROOM_MIN_M, BOOTSTRAP_CAP_HEADROOM_MAX_M)
    target_payroll = max(0.0, cap_m - headroom_target)
    if payroll_m > target_payroll + 1e-6:
        from services.contract_economy import has_true_elc_contract

        excess = payroll_m - target_payroll
        roster = _active_roster(team)
        protected = _protected_core_player_ids(roster)
        trimmable = sorted(
            [
                p for p in roster
                if id(p) not in protected
                and not has_true_elc_contract(p)
                and not is_real_nhl_contract(p)
            ],
            key=_player_ovr99,
        )
        for p in trimmable:
            if excess <= 1e-6:
                break
            cur = _player_cap_hit_millions(p)
            cut = min(excess, max(0.0, cur - LEAGUE_MINIMUM_AAV_M))
            if cut <= 0.01:
                continue
            _set_player_contract_aav(p, round(cur - cut, 3), season_year)
            excess -= cut
        payroll_m = _team_nhl_payroll_m(team)

    cap_space_m = max(0.0, cap_m - payroll_m)
    _sync_team_cap_fields(team, league)
    return {
        "salary_cap": round(cap_m, 3),
        "cap_hit": round(payroll_m, 3),
        "cap_space": round(cap_space_m, 3),
    }


def _ensure_team_roster_contracts_cap_safe(
    team: Any,
    league: Any,
    season_year: int,
    rng: random.Random,
) -> int:
    """Assign NHL roster contracts while tracking remaining cap budget."""
    roster = _active_roster(team)
    if not roster:
        _sync_team_cap_fields(team, league)
        return 0

    cap_m = _resolve_league_salary_cap_m(league)
    generated = 0
    try:
        team._bootstrap_league_ref = league
    except Exception:
        pass

    # Leave in-season cap headroom — NHL clubs rarely open at a hard $0 ceiling.
    headroom_target = rng.uniform(BOOTSTRAP_CAP_HEADROOM_MIN_M, BOOTSTRAP_CAP_HEADROOM_MAX_M)
    payroll_ceiling = max(cap_m * 0.84, cap_m - headroom_target)
    remaining_budget = float(payroll_ceiling)

    ordered = sorted(roster, key=lambda p: (-_player_ovr99(p), -_player_age_int(p)))
    # Bootstrap contract valuation repeatedly queries team OVR/rank context.
    # Precompute once per team contract batch.
    try:
        roster_ovr = {id(p): _player_ovr99(p) for p in roster}
        roster_pos = {id(p): _pos_str(p) for p in roster}
        all_ovrs = sorted((float(v) for v in roster_ovr.values()), reverse=True)
        pos_ovrs: Dict[str, List[float]] = defaultdict(list)
        for p in roster:
            pos_ovrs[str(roster_pos.get(id(p), "") or "").upper()].append(float(roster_ovr.get(id(p), 0.0)))
        for k in list(pos_ovrs.keys()):
            pos_ovrs[k] = sorted(pos_ovrs[k], reverse=True)
        team._bootstrap_contract_ovr_cache = {
            "by_player_id": roster_ovr,
            "all_ovrs_desc": all_ovrs,
            "pos_ovrs_desc": pos_ovrs,
        }
    except Exception:
        pass

    for idx, p in enumerate(ordered):
        existing_hit = _player_cap_hit_millions(p)
        if existing_hit > 0 and getattr(p, "contract", None) is not None:
            remaining_budget = max(0.0, remaining_budget - existing_hit)
            continue

        slots_left = len(ordered) - idx
        min_reserve = LEAGUE_MINIMUM_AAV_M * max(0, slots_left - 1)
        max_aav = max(LEAGUE_MINIMUM_AAV_M, remaining_budget - min_reserve)

        seed = abs(hash(f"contract|{getattr(p, 'id', '') or _name_str(p)}|{season_year}")) & 0xFFFFFFFF
        if _ensure_player_contract(
            p, season_year, random.Random(seed), max_aav_m=max_aav,
            team=team, league=league, allow_bad=True,
        ):
            generated += 1
        hit = _player_cap_hit_millions(p)
        if hit <= 0:
            fallback = min(max_aav, max(LEAGUE_MINIMUM_AAV_M, remaining_budget))
            _set_player_contract_aav(p, fallback, season_year)
            hit = _player_cap_hit_millions(p)
            if hit > 0:
                generated += 1
        remaining_budget = max(0.0, remaining_budget - hit)

    try:
        team._bootstrap_contract_ovr_cache = None
    except Exception:
        pass
    _validate_team_cap_non_negative(team, league, season_year, rng)
    return generated


def _generate_contract_terms(ovr99: float, age: int, pos: str, rng: random.Random) -> Tuple[float, int]:
    """Backward-compatible wrapper — delegates to contract_economy valuation."""
    from types import SimpleNamespace
    from services.contract_economy import generate_contract_terms

    player = SimpleNamespace(
        ovr=lambda: float(ovr99) / 99.0 if float(ovr99) > 1.5 else float(ovr99),
        age=age,
        identity=SimpleNamespace(age=age, position=pos),
        position=pos,
        ratings={},
        season_stats={},
    )
    aav, years, _ = generate_contract_terms(player, None, None, rng, allow_bad=False)
    return aav, years


def _player_age_int(player: Any) -> int:
    ident = getattr(player, "identity", None)
    try:
        return int(getattr(ident, "age", 0) or getattr(player, "age", 0) or 27)
    except Exception:
        return 27


def _player_birth_ymd(player: Any) -> Optional[Tuple[int, int, int]]:
    """Return (year, month, day) when known."""
    raw = str(getattr(player, "birth_date", None) or getattr(player, "_birth_date", None) or "")[:10]
    if raw and raw.count("-") >= 2:
        try:
            y, m, d = [int(x) for x in raw.split("-")[:3]]
            return y, m, d
        except Exception:
            pass
    ident = getattr(player, "identity", None)
    if ident is None:
        return None
    try:
        y = int(getattr(ident, "birth_year", 0) or 0)
    except Exception:
        y = 0
    if y <= 1900:
        return None
    try:
        m = int(getattr(ident, "birth_month", 0) or 0)
    except Exception:
        m = 0
    try:
        d = int(getattr(ident, "birth_day", 0) or 0)
    except Exception:
        d = 0
    # Generated players often only have birth_year; mid-year birthday keeps Sept ages stable.
    if m <= 0:
        m = 7
    if d <= 0:
        d = 1
    return y, m, d


def _age_years_as_of(birth: Tuple[int, int, int], as_of_year: int, as_of_month: int = 9, as_of_day: int = 15) -> int:
    y, m, d = birth
    years = int(as_of_year) - int(y)
    if (int(as_of_month), int(as_of_day)) < (int(m), int(d)):
        years -= 1
    return max(15, min(55, years))


def sync_player_age_to_season(player: Any, season_year: int, *, as_of_month: int = 9, as_of_day: int = 15) -> int:
    """Authoritative age from birth date as of an as-of calendar date."""
    birth = _player_birth_ymd(player)
    ident = getattr(player, "identity", None)
    if birth is None:
        try:
            return int(getattr(ident, "age", 0) or getattr(player, "age", 0) or 27)
        except Exception:
            return 27
    age = _age_years_as_of(birth, season_year, as_of_month, as_of_day)
    if ident is not None and hasattr(ident, "age"):
        try:
            ident.age = age
        except Exception:
            pass
    try:
        setattr(player, "age", age)
    except Exception:
        pass
    return age


def session_age_as_of(session: Optional[FranchiseSession]) -> Tuple[int, int, int]:
    """(year, month, day) for DOB aging from the franchise NHL calendar.

    Priority:
    1. After year-end aging (season year not yet bumped) → next Sept 15
       so serialize cannot roll ages back via a stale June calendar cursor.
       Applies in playoffs / playoff_ready / offseason — not only ``offseason``.
    2. Live NHL calendar date (mid-season birthdays).
    3. Sept 15 of the current season year.
    """
    if session is None:
        return 2025, 9, 15
    sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
    # Year-end pin MUST beat the live calendar and MUST NOT depend on phase —
    # year-end runs into playoff_ready while the calendar is still April–June.
    if bool(getattr(session, "_year_end_progression_done", False)):
        return sy + 1, 9, 15
    iso = _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
    if iso:
        try:
            parts = str(iso).strip()[:10].split("-")
            if len(parts) == 3:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                    return y, m, d
        except Exception:
            pass
    return sy, 9, 15


def sync_player_age_to_session(player: Any, session: Optional[FranchiseSession]) -> int:
    y, m, d = session_age_as_of(session)
    return sync_player_age_to_season(player, y, as_of_month=m, as_of_day=d)


def resync_league_ages_to_session(session: Optional[FranchiseSession]) -> Dict[str, int]:
    """Force every league player age to match DOB vs the internal calendar.

    Returns counts for regression / diagnostics. Safe to call from FA desks,
    year-end ticks, and season rollover.
    """
    if session is None:
        return {"synced": 0, "missing_dob": 0, "skipped": 0}
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return {"synced": 0, "missing_dob": 0, "skipped": 0}
    synced = 0
    missing = 0
    skipped = 0
    for player in _iter_league_players_for_aging(league):
        if getattr(player, "retired", False):
            skipped += 1
            continue
        if _player_birth_ymd(player) is None:
            missing += 1
        try:
            sync_player_age_to_session(player, session)
            synced += 1
        except Exception:
            skipped += 1
    return {"synced": synced, "missing_dob": missing, "skipped": skipped}

def _iter_league_players_for_aging(league: Any) -> List[Any]:
    """NHL / AHL / ECHL / FA / prospects / juniors — everyone must age into and out of the game."""
    out: List[Any] = []
    seen: set = set()

    def _add(p: Any) -> None:
        if p is None or getattr(p, "retired", False):
            return
        pid = str(getattr(p, "id", "") or id(p))
        if pid in seen:
            return
        seen.add(pid)
        out.append(p)

    for team in list(getattr(league, "teams", None) or []):
        for attr in ("roster", "ahl_roster", "echl_roster", "prospect_pool"):
            for p in list(getattr(team, attr, None) or []):
                _add(p)
        for entry in list(getattr(team, "rfa_rights", None) or []):
            pref = entry.get("player_ref") if isinstance(entry, dict) else None
            _add(pref)
    for pool_attr in ("free_agents", "overseas_free_agents"):
        for p in list(getattr(league, pool_attr, None) or []):
            _add(p)
    for block in list(getattr(league, "development_leagues", None) or []):
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                _add(p)
    return out


def _player_ovr99(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    try:
        v = float(fn() if callable(fn) else fn or 0)
    except Exception:
        return 0.0
    return v * 99.0 if v <= 1.5 else v


class _GeneratedContract:
    """Lightweight contract record attached to players at bootstrap."""

    __slots__ = (
        "cap_hit_m", "aav_m", "salary_m", "years_remaining", "expiry_year",
        "contract_type", "rights_status", "no_trade_clause", "no_move_clause",
        "modified_no_trade_teams", "clauses", "two_way", "source",
    )

    def __init__(self, *, aav_m: float, years: int, expiry_year: int, contract_type: str,
                 rights_status: str, ntc: bool = False, nmc: bool = False, two_way: bool = False):
        self.cap_hit_m = float(aav_m)
        self.aav_m = float(aav_m)
        self.salary_m = float(aav_m)
        self.years_remaining = int(years)
        self.expiry_year = int(expiry_year)
        self.contract_type = str(contract_type)
        self.rights_status = str(rights_status)
        self.no_trade_clause = bool(ntc)
        self.no_move_clause = bool(nmc)
        self.modified_no_trade_teams = 0
        self.clauses = None
        self.two_way = bool(two_way)
        self.source = "bootstrap"

    def tick_year(self) -> None:
        self.years_remaining = max(0, int(self.years_remaining) - 1)


def _ensure_player_contract(
    player: Any,
    season_year: int,
    rng: Optional[random.Random] = None,
    *,
    max_aav_m: Optional[float] = None,
    team: Any = None,
    league: Any = None,
    allow_bad: bool = False,
) -> bool:
    """Attach a generated contract if the player has no usable cap hit. Returns True if generated."""
    from services.contract_economy import apply_contract_to_player, build_contract_for_player, hydrate_player_contract

    if getattr(player, "retired", False):
        return False
    hydrate_player_contract(player)
    if _player_cap_hit_millions(player) > 0 and getattr(player, "contract", None) is not None:
        return False
    if rng is None:
        seed = abs(hash(str(getattr(player, "id", "") or _name_str(player)))) & 0xFFFFFFFF
        rng = random.Random(seed)

    lg = league or getattr(team, "_bootstrap_league_ref", None)
    contract = build_contract_for_player(
        player, team, lg, season_year, rng,
        max_aav_m=max_aav_m, allow_bad=allow_bad,
    )
    apply_contract_to_player(player, contract, season_year)
    try:
        player.cap_hit_m = float(contract["cap_hit_m"])
    except Exception:
        pass
    return True


def _ensure_league_roster_contracts(league: Any, season_year: int) -> int:
    """Generate cap-compliant contracts for every NHL roster. Returns count generated."""
    generated = 0
    for team in getattr(league, "teams", None) or []:
        tid = str(getattr(team, "team_id", "") or getattr(team, "id", "") or "")
        seed = abs(hash(f"cap_bootstrap|{tid}|{season_year}")) & 0xFFFFFFFF
        team_rng = random.Random(seed)
        if getattr(team, "_contracts_bootstrapped", False):
            _validate_team_cap_non_negative(team, league, season_year, team_rng)
            generated += _ensure_team_affiliate_nhl_spcs(team, season_year, team_rng)
            continue
        generated += _ensure_team_roster_contracts_cap_safe(team, league, season_year, team_rng)
        generated += _ensure_team_affiliate_nhl_spcs(team, season_year, team_rng)
        try:
            team._contracts_bootstrapped = True
        except Exception:
            pass
    from services.contract_economy import fix_league_contract_truth
    fix_league_contract_truth(league)
    return generated


def _ensure_team_affiliate_nhl_spcs(
    team: Any,
    season_year: int,
    rng: random.Random,
    *,
    ahl_target: int = 20,
    echl_target: int = 5,
    org_slot_ceiling: int = 48,
) -> int:
    """Assign two-way NHL SPCs to unsigned AHL/ECHL depth so the 50-contract reserve fills.

    These deals count toward the 50 regardless of assignment. Cap snapshot still keys off
    the NHL active list, so minors AAV does not inflate upper-limit payroll here.
    """
    from services.contract_economy import (
        LEAGUE_MINIMUM_AAV_M,
        _count_team_contract_slots,
        uses_nhl_contract_slot,
    )

    generated = 0
    used = _count_team_contract_slots(team)
    room = max(0, int(org_slot_ceiling) - int(used))
    if room <= 0:
        return 0

    def _assign_pool(attr: str, target: int) -> int:
        nonlocal room, generated
        pool = list(getattr(team, attr, None) or [])
        already = sum(1 for p in pool if uses_nhl_contract_slot(p))
        need = max(0, min(int(target) - already, room))
        if need <= 0:
            return 0
        unsigned = [
            p for p in pool
            if not uses_nhl_contract_slot(p) and not bool(getattr(p, "retired", False))
        ]
        unsigned.sort(key=lambda p: (-_player_ovr99(p), _player_age_int(p)))
        made = 0
        for p in unsigned[:need]:
            aav = LEAGUE_MINIMUM_AAV_M
            years = 2 if _player_age_int(p) <= 24 else 1
            seed = abs(hash(f"aff_spc|{getattr(p, 'id', '')}|{season_year}")) & 0xFFFFFFFF
            _ = random.Random(seed)  # deterministic per player; keep for future jitter
            _set_player_contract_aav(p, aav, season_year)
            c = getattr(p, "contract", None)
            try:
                if isinstance(c, dict):
                    c["type"] = "STANDARD"
                    c["contract_type"] = "STANDARD"
                    c["aav_m"] = aav
                    c["cap_hit_m"] = aav
                    c["years"] = years
                    c["years_remaining"] = years
                    c["expiry_year"] = int(season_year) + years
                    c["two_way"] = True
                    c["is_nhl_spc"] = True
                    c["nhl_spc"] = True
                    c["standard_player_contract"] = True
                    c["rights_status"] = c.get("rights_status") or "RFA"
                    c["source"] = "affiliate_nhl_spc"
                    p.contract = c
                elif c is not None:
                    c.two_way = True
                    c.years = years
                    c.years_remaining = years
                    c.expiry_year = int(season_year) + years
                    # Force NHL SPC — never keep leftover AHL/ECHL type labels.
                    c.type = "STANDARD"
                    c.contract_type = "STANDARD"
                    c.aav_m = aav
                    c.cap_hit_m = aav
                    try:
                        c.is_nhl_spc = True
                        c.nhl_spc = True
                        c.standard_player_contract = True
                    except Exception:
                        pass
                else:
                    p.contract = {
                        "type": "STANDARD",
                        "contract_type": "STANDARD",
                        "aav_m": aav,
                        "cap_hit_m": aav,
                        "years": years,
                        "years_remaining": years,
                        "expiry_year": int(season_year) + years,
                        "two_way": True,
                        "is_nhl_spc": True,
                        "nhl_spc": True,
                        "standard_player_contract": True,
                        "rights_status": "RFA",
                        "source": "affiliate_nhl_spc",
                    }
                p.signed_status = "signed"
                p.in_minors = True
                p.roster_location = "ahl" if attr == "ahl_roster" else "echl"
                p.cap_hit_m = aav
                try:
                    p.is_nhl_spc = True
                except Exception:
                    pass
            except Exception:
                continue
            made += 1
            generated += 1
            room -= 1
            if room <= 0:
                break
        return made

    _assign_pool("ahl_roster", ahl_target)
    if room > 0:
        _assign_pool("echl_roster", echl_target)
    return generated


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
    """NHL active list — excludes retired / minors / buried (cap + lineup)."""
    try:
        from services.roster_compliance import iter_active_nhl_roster

        return iter_active_nhl_roster(team)
    except Exception:
        return [
            p for p in (getattr(team, "roster", None) or [])
            if not getattr(p, "retired", False)
            and not getattr(p, "in_minors", False)
            and not getattr(p, "is_buried", False)
            and not getattr(p, "buried", False)
        ]


def _skaters(team: Any) -> List[Any]:
    return [p for p in _active_roster(team) if _pos_str(p).upper() != "G"]


def _goalies(team: Any) -> List[Any]:
    return [p for p in _active_roster(team) if _is_goalie_player(p)]


def _available_goalies(team: Any) -> List[Any]:
    return [g for g in _goalies(team) if not _is_player_live_injured(g)]


def _goalies_on_nhl_roster_list(team: Any) -> List[Any]:
    """Any non-retired goalie still attached to the NHL roster list (IR / misflagged minors)."""
    out: List[Any] = []
    for p in list(getattr(team, "roster", None) or []):
        if getattr(p, "retired", False):
            continue
        if _is_goalie_player(p):
            out.append(p)
    return out


def _goalies_in_affiliates(team: Any) -> List[Any]:
    out: List[Any] = []
    for attr in ("ahl_roster", "echl_roster"):
        for p in list(getattr(team, attr, None) or []):
            if getattr(p, "retired", False):
                continue
            if _is_goalie_player(p):
                out.append(p)
    return out


def _emergency_call_up_goalie(team: Any) -> Optional[Any]:
    """Promote the best affiliate goalie onto the NHL roster so a game can be played."""
    if team is None:
        return None
    candidates = _goalies_in_affiliates(team)
    if not candidates:
        return None
    candidates.sort(key=lambda p: -_player_ovr99(p))
    pick = candidates[0]

    nhl = list(getattr(team, "roster", None) or [])
    ahl = list(getattr(team, "ahl_roster", None) or [])
    echl = list(getattr(team, "echl_roster", None) or [])
    demoted = None

    # Keep under 23 by demoting a weak non-NMC skater when needed.
    if len([p for p in nhl if not getattr(p, "retired", False) and not getattr(p, "in_minors", False)]) >= 23:
        demote_idx = None
        demote_score = 999.0
        for i, p in enumerate(nhl):
            if _is_goalie_player(p):
                continue
            c = getattr(p, "contract", None)
            if bool(getattr(c, "nmc", False) or getattr(c, "no_move_clause", False)):
                continue
            if getattr(p, "in_minors", False) or getattr(p, "is_buried", False):
                continue
            score = float(_player_ovr99(p))
            if score < demote_score:
                demote_score = score
                demote_idx = i
        if demote_idx is not None:
            demoted = nhl.pop(demote_idx)
            demoted.in_minors = True
            demoted.roster_location = "ahl"
            ahl.append(demoted)

    pid = str(getattr(pick, "id", "") or "")
    ahl = [p for p in ahl if str(getattr(p, "id", "")) != pid]
    echl = [p for p in echl if str(getattr(p, "id", "")) != pid]
    if not any(str(getattr(p, "id", "")) == pid for p in nhl):
        nhl.append(pick)
    pick.in_minors = False
    pick.is_buried = False
    if hasattr(pick, "buried"):
        pick.buried = False
    pick.roster_location = "nhl"
    setattr(team, "roster", nhl)
    setattr(team, "ahl_roster", ahl)
    setattr(team, "echl_roster", echl)
    return pick


def _reactivate_misflagged_nhl_goalie(team: Any) -> Optional[Any]:
    """Clear accidental minors/buried flags on a goalie who is already on the NHL list."""
    listed = _goalies_on_nhl_roster_list(team)
    if not listed:
        return None
    # Prefer a non-IR goalie when possible.
    prefer = [
        g for g in listed
        if not bool(getattr(g, "on_ir", False) or getattr(g, "is_ir", False) or getattr(g, "on_ltir", False))
    ]
    pool = prefer or listed
    pick = max(pool, key=_player_ovr99)
    for attr, val in (("in_minors", False), ("is_buried", False), ("buried", False), ("roster_location", "nhl")):
        try:
            setattr(pick, attr, val)
        except Exception:
            pass
    return pick


def _goalie_availability_status(team: Any) -> Dict[str, Any]:
    all_goalies = _goalies(team)
    healthy = _available_goalies(team)
    if all_goalies:
        return {
            "total": int(len(all_goalies)),
            "healthy": int(len(healthy)),
            "forced_injured_start": bool(all_goalies and not healthy),
        }

    # Engine dresses anyone on team.roster; franchise active-roster filters are stricter.
    listed = _goalies_on_nhl_roster_list(team)
    if listed:
        healthy_listed = [g for g in listed if not _is_player_live_injured(g)]
        return {
            "total": int(len(listed)),
            "healthy": int(len(healthy_listed)),
            "forced_injured_start": bool(not healthy_listed),
        }

    return {
        "total": 0,
        "healthy": 0,
        "forced_injured_start": False,
    }


def _ensure_goalie_for_game(team: Any) -> Dict[str, Any]:
    """Self-heal empty NHL goalie slots before a scheduled game."""
    if _goalies(team):
        return _goalie_availability_status(team)

    revived = _reactivate_misflagged_nhl_goalie(team)
    if _goalies(team):
        if revived is not None:
            _fr_dbg(
                f"reactivated misflagged NHL goalie {_name_str(revived)} "
                f"for {_display_team(team)}"
            )
        return _goalie_availability_status(team)

    called = _emergency_call_up_goalie(team)
    status = _goalie_availability_status(team)
    if called is not None and int(status.get("total") or 0) > 0:
        _fr_dbg(
            f"emergency goalie call-up {_name_str(called)} "
            f"for {_display_team(team)}"
        )
        return status
    return status


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


def _split_game_toi_sec(p: Any, total_sec: int, rng: random.Random) -> tuple[int, int, int]:
    """Split total TOI into EV/PP/PK for the season stats ledger."""
    total = max(0, int(total_sec))
    if total <= 0:
        return 0, 0, 0
    pos = _pos_str(p).upper()
    if pos == "G":
        return total, 0, 0
    usage = _player_role_usage_mult(p)
    role_raw = str(getattr(p, "line_role", None) or getattr(p, "role", None) or "").lower()
    pp_share = 0.0
    pk_share = 0.0
    if "pp" in role_raw or usage >= 1.5:
        pp_share = min(0.22, 0.08 + usage * 0.05) + rng.uniform(-0.02, 0.02)
    elif usage >= 1.0:
        pp_share = min(0.12, 0.04 + usage * 0.03)
    if pos == "D" or "pk" in role_raw or usage >= 1.2:
        pk_share = min(0.18, 0.05 + usage * 0.04) + rng.uniform(-0.01, 0.02)
    elif usage >= 0.9:
        pk_share = min(0.08, 0.02 + usage * 0.02)
    pp_share = max(0.0, min(0.28, pp_share))
    pk_share = max(0.0, min(0.22, pk_share))
    if pp_share + pk_share > 0.38:
        scale = 0.38 / (pp_share + pk_share)
        pp_share *= scale
        pk_share *= scale
    pp_sec = int(total * pp_share)
    pk_sec = int(total * pk_share)
    ev_sec = max(0, total - pp_sec - pk_sec)
    return ev_sec, pp_sec, pk_sec


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
            "primary_assists": 0,
            "secondary_assists": 0,
            "pts": 0,
            "sog": 0,
            "pim": 0,
            "hit": 0,
            "blk": 0,
            "toi_sec": 0,
            "ev_toi_sec": 0,
            "pp_toi_sec": 0,
            "pk_toi_sec": 0,
            "ga": 0,
            "w": 0,
            "l": 0,
            "otl": 0,
            "stat_authority": "session.player_season_stats",
        }
    row = reg[pid]
    row["name"] = _name_str(p)
    row["position"] = _pos_str(p)
    row["team_id"] = str(team_id)
    row["stat_authority"] = "session.player_season_stats"
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
        ev_toi, pp_toi, pk_toi = _split_game_toi_sec(p, toi, rng)
        pim = int(rng.choices([0, 2, 4, 6], weights=[0.56, 0.28, 0.12, 0.04], k=1)[0])
        if pos == "D":
            hit = int(rng.choices([0, 1, 2, 3, 4], weights=[0.17, 0.28, 0.30, 0.18, 0.07], k=1)[0] * max(0.7, min(1.5, usage)))
            blk = int(rng.choices([0, 1, 2, 3, 4], weights=[0.10, 0.26, 0.33, 0.21, 0.10], k=1)[0] * max(0.8, min(1.5, usage)))
        else:
            hit = int(rng.choices([0, 1, 2, 3], weights=[0.29, 0.37, 0.24, 0.10], k=1)[0] * max(0.7, min(1.4, usage)))
            blk = int(rng.choices([0, 1, 2], weights=[0.64, 0.29, 0.07], k=1)[0] * max(0.7, min(1.3, usage)))
        _stat_add(
            session,
            p,
            tid,
            gp=1,
            sog=sog,
            pim=pim,
            hit=hit,
            blk=blk,
            toi_sec=toi,
            ev_toi_sec=ev_toi,
            pp_toi_sec=pp_toi,
            pk_toi_sec=pk_toi,
        )
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
            "ev_toi_sec": ev_toi,
            "pp_toi_sec": pp_toi,
            "pk_toi_sec": pk_toi,
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


def _franchise_stat_scope(session: FranchiseSession, *, is_playoff: bool = False) -> str:
    if is_playoff:
        return "playoffs"
    phase = str(getattr(session, "phase", "") or getattr(session, "season_phase", "") or "").lower()
    if "preseason" in phase:
        return "preseason"
    if "playoff" in phase:
        return "playoffs"
    return "regular_season"


def _stable_franchise_game_id_for(
    session: FranchiseSession,
    *,
    calendar_day: int,
    hid: str,
    aid: str,
    stat_scope: str,
) -> str:
    season = int(getattr(session, "season_calendar_year", 0) or 0)
    raw = f"{season}:{stat_scope}:{int(calendar_day)}:{str(hid)}:{str(aid)}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:14]
    return f"g_{digest}"


def _processed_game_ids(session: FranchiseSession) -> set:
    processed = getattr(session, "processed_game_ids", None)
    if processed is None:
        processed = set()
        setattr(session, "processed_game_ids", processed)
    if isinstance(processed, list):
        processed = set(str(x) for x in processed if str(x or "").strip())
        setattr(session, "processed_game_ids", processed)
    return processed


def _find_game_result_by_id(session: FranchiseSession, game_id: str) -> Optional[Dict[str, Any]]:
    gid = str(game_id or "").strip()
    if not gid:
        return None
    for game in reversed(list(getattr(session, "game_results", None) or [])):
        if isinstance(game, dict) and str(game.get("game_id") or game.get("id") or "").strip() == gid:
            return game
    return None


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
    is_playoff: bool = False,
    home_b2b: bool = False,
    away_b2b: bool = False,
) -> Optional[Dict[str, Any]]:
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
    stat_scope = _franchise_stat_scope(session, is_playoff=bool(is_playoff))
    gid = _stable_franchise_game_id_for(
        session,
        calendar_day=int(calendar_day),
        hid=str(hid),
        aid=str(aid),
        stat_scope=stat_scope,
    )
    processed = _processed_game_ids(session)
    if gid in processed:
        return _find_game_result_by_id(session, gid)

    hg = _coerce_final_score(hg)
    ag = _coerce_final_score(ag)

    if hg == ag:
        raise RuntimeError(
            f"Stat accumulation refused tied final on day {calendar_day}: {hid} {hg}, {aid} {ag}"
        )

    light_stats = bool(getattr(session, "_light_game_stat_accumulation", False))
    # Same counting model for every club during bulk — do not force the user onto
    # the cooler event ledger while CPU teams use light concentration.
    stat_kw: Dict[str, Any] = {
        "build_game_payload": not light_stats,
        "calendar_day": int(calendar_day),
        "calendar_iso": str(calendar_iso or ""),
        "game_id": gid,
        "stat_scope": stat_scope,
        "is_playoff": bool(is_playoff),
        "home_b2b": bool(home_b2b),
        "away_b2b": bool(away_b2b),
    }
    try:
        import inspect

        sig = inspect.signature(sim.accumulate_unified_game_stats)
        if "light_mode" in sig.parameters:
            stat_kw["light_mode"] = light_stats
    except (TypeError, ValueError):
        pass

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
        **stat_kw,
    )

    if not isinstance(box, dict):
        if bool(getattr(session, "_light_game_stat_accumulation", False)):
            box = {
                "game_id": gid,
                "id": gid,
                "home_id": str(hid),
                "away_id": str(aid),
                "home_goals": int(hg),
                "away_goals": int(ag),
                "home_score": int(hg),
                "away_score": int(ag),
                "overtime": bool(ot),
                "ot": bool(ot),
                "day": int(calendar_day),
                "calendar_day": int(calendar_day),
                "iso": str(calendar_iso or ""),
                "calendar_iso": str(calendar_iso or ""),
                "stat_scope": stat_scope,
                "status": "final",
                "completed": True,
                "is_final": True,
                "simmed": True,
                "light_box": True,
                "scoring_events": [],
            }
        else:
            raise RuntimeError(
                f"Stat accumulation failed on calendar day {calendar_day}: SimEngine returned no game box."
            )

    box_hg = _coerce_final_score(box.get("home_goals", box.get("home_score", hg)))
    box_ag = _coerce_final_score(box.get("away_goals", box.get("away_score", ag)))

    if box_hg != int(hg) or box_ag != int(ag):
        # Never drop the HTTP connection over a boxscore re-roll. Trust the simulated final;
        # light bulk path especially used to re-sim events and diverge.
        _startup_log.warning(
            "Stat/game score coerce day=%s %s %s-%s -> box was %s-%s light=%s",
            calendar_day,
            hid,
            hg,
            ag,
            box_hg,
            box_ag,
            light_stats,
        )
        box["home_goals"] = int(hg)
        box["away_goals"] = int(ag)
        box["home_score"] = int(hg)
        box["away_score"] = int(ag)
        box["score_coerced"] = True

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
            "season_calendar_year": int(getattr(session, "season_calendar_year", 0) or 0),
            "stat_scope": stat_scope,
            "stat_authority": "session.player_season_stats",
            "status": "final",
            "completed": True,
            "is_final": True,
            "simmed": True,
        }
    )
    if not light_stats:
        for row in list((getattr(session, "player_season_stats", None) or {}).values()):
            if isinstance(row, dict):
                row.setdefault("stat_scope", stat_scope)
                row["stat_authority"] = "session.player_season_stats"

    session.game_results.append(box)
    processed.add(gid)
    if not light_stats:
        try:
            from app.sim_engine.franchise.storyline_coverage import ingest_game_box_storylines  # noqa: WPS433

            ingest_game_box_storylines(session, box)
        except Exception:
            pass

    if len(session.game_results) > 2400:
        session.game_results = session.game_results[-1800:]
    if not bool(getattr(session, "_light_game_stat_accumulation", False)):
        _bump_stats_revision(session)
    return box


def _bump_stats_revision(session: FranchiseSession) -> None:
    session._stats_revision = int(getattr(session, "_stats_revision", 0) or 0) + 1
    session._stats_central_cache = None


def _bump_prospect_revision(session: FranchiseSession) -> None:
    session._prospect_revision = int(getattr(session, "_prospect_revision", 0) or 0) + 1
    session._draft_class_detail_cache = None


def _purge_synthetic_universe_artifacts(session: FranchiseSession) -> bool:
    """
    Strip invented CF/xGF/PP fields from the failed universe-repair pass only.

    Do NOT strip legitimate light_strength CPU–CPU boxes — those now carry real
    SF/CF/xGF/PP from the light sim path and must feed Stats Central. An older
    purge wiped every light_box on Stats Central open, so team CF%/xGF% only
    reflected the handful of full-event games vs the user.
    """
    if bool(getattr(session, "_synthetic_universe_purged_v2", False)):
        return False

    cleared_games = 0
    synth_keys = (
        "home_shot_attempts",
        "away_shot_attempts",
        "home_cf",
        "away_cf",
        "home_ff",
        "away_ff",
        "home_xgf",
        "away_xgf",
        "home_xg",
        "away_xg",
        "home_ppo",
        "away_ppo",
        "home_opp_ppo",
        "away_opp_ppo",
        "home_pp_goals",
        "away_pp_goals",
        "home_ppga",
        "away_ppga",
        "universe_repaired",
    )
    for g in list(getattr(session, "game_results", None) or []):
        if not isinstance(g, dict):
            continue
        # Only the synthetic universe-repair pass — never light_strength CPU–CPU.
        if not g.get("universe_repaired"):
            continue
        for key in synth_keys:
            if key in g:
                g.pop(key, None)
        for key in ("home_xgf", "away_xgf", "home_xg", "away_xg"):
            g[key] = 0.0
        for key in (
            "home_shot_attempts",
            "away_shot_attempts",
            "home_cf",
            "away_cf",
            "home_ff",
            "away_ff",
            "home_ppo",
            "away_ppo",
            "home_pp_goals",
            "away_pp_goals",
            "home_ppga",
            "away_ppga",
            "home_opp_ppo",
            "away_opp_ppo",
        ):
            g[key] = 0
        cleared_games += 1

    try:
        setattr(session, "_synthetic_universe_purged_v2", True)
        setattr(session, "_synthetic_universe_purged_v1", True)
        setattr(session, "_universe_repair_v2", False)
    except Exception:
        pass

    if cleared_games:
        _bump_stats_revision(session)
        return True
    return False


def _backfill_missing_toi_from_game_boxes(session: FranchiseSession) -> bool:
    """
    Rebuild season toi_sec from stored game skater boxes when the ledger has
    counting/analytics (G/A/iXG) but TOI was never written (WAR collapses to 0).
    """
    if bool(getattr(session, "_toi_backfill_v1", False)):
        return False

    stats = getattr(session, "player_season_stats", None) or {}
    if not isinstance(stats, dict) or not stats:
        try:
            setattr(session, "_toi_backfill_v1", True)
        except Exception:
            pass
        return False

    skater_rows = [
        r
        for r in stats.values()
        if isinstance(r, dict)
        and str(r.get("stat_scope") or "regular_season") == "regular_season"
        and str(r.get("position") or "").upper() != "G"
        and int(r.get("gp", 0) or 0) >= 5
    ]
    if not skater_rows:
        try:
            setattr(session, "_toi_backfill_v1", True)
        except Exception:
            pass
        return False

    missing = [
        r
        for r in skater_rows
        if int(r.get("toi_sec", 0) or 0) < max(60, int(r.get("gp", 0) or 0) * 30)
    ]
    # Only repair when TOI is broadly broken (not a few healthy scratches).
    if len(missing) < max(8, int(len(skater_rows) * 0.35)):
        try:
            setattr(session, "_toi_backfill_v1", True)
        except Exception:
            pass
        return False

    games = [
        g
        for g in list(getattr(session, "game_results", None) or [])
        if isinstance(g, dict)
        and str(g.get("stat_scope") or "regular_season") == "regular_season"
        and not g.get("light_box")
        and str(g.get("stat_source") or "") != "light_strength"
    ]
    toi_from_boxes: Dict[str, int] = defaultdict(int)
    for g in games:
        for key in ("home_skaters", "away_skaters"):
            for row in list(g.get(key) or []):
                if not isinstance(row, dict):
                    continue
                pid = str(row.get("player_id") or row.get("id") or "")
                if not pid:
                    continue
                toi_from_boxes[pid] += max(0, int(row.get("toi_sec") or 0))

    if sum(1 for v in toi_from_boxes.values() if v >= 600) < 20:
        # Boxes don't carry usable TOI either — leave flag unset so a later
        # season with better boxes can still repair.
        return False

    repaired = 0
    for pid, toi in toi_from_boxes.items():
        row = stats.get(pid) or stats.get(str(pid))
        if not isinstance(row, dict) or toi <= 0:
            continue
        if int(row.get("toi_sec", 0) or 0) >= toi:
            continue
        row["toi_sec"] = int(toi)
        repaired += 1

    try:
        setattr(session, "_toi_backfill_v1", True)
    except Exception:
        pass
    if repaired:
        _bump_stats_revision(session)
        return True
    return False


def _backfill_player_analytics_from_game_boxes(session: FranchiseSession) -> bool:
    """
    Rebuild on-ice CF/xGF/goalie xGA from stored event game boxes when player
    ledgers were wiped (or never written) but team box analytics remain.

    Skater shares are TOI-weighted within each game so season CF%/xGF% reflect
    which matchups they played — not invented scoring.
    """
    if bool(getattr(session, "_analytics_backfill_v2", False)):
        return False

    stats = getattr(session, "player_season_stats", None) or {}
    if not isinstance(stats, dict) or not stats:
        try:
            setattr(session, "_analytics_backfill_v2", True)
        except Exception:
            pass
        return False

    sample = [
        r
        for r in stats.values()
        if isinstance(r, dict)
        and str(r.get("stat_scope") or "regular_season") == "regular_season"
        and int(r.get("gp", 0) or 0) >= 10
    ][:40]
    existing_cf = sum(float(r.get("cf", 0) or 0) + float(r.get("ca", 0) or 0) for r in sample)
    existing_xga = sum(
        float(r.get("goalie_xga", r.get("xga", 0)) or 0)
        for r in sample
        if str(r.get("position") or "").upper() == "G"
    )
    if existing_cf > 500 and existing_xga > 5:
        try:
            setattr(session, "_analytics_backfill_v2", True)
        except Exception:
            pass
        return False

    def _num(row: Dict[str, Any], *keys: str) -> float:
        for key in keys:
            if key in row and row.get(key) is not None:
                try:
                    return float(row.get(key) or 0)
                except (TypeError, ValueError):
                    continue
        return 0.0

    games = [
        g
        for g in list(getattr(session, "game_results", None) or [])
        if isinstance(g, dict)
        and str(g.get("stat_scope") or "regular_season") == "regular_season"
        and not g.get("light_box")
        and not g.get("universe_repaired")
        and str(g.get("stat_source") or "") != "light_strength"
    ]
    usable = 0
    for g in games:
        if _num(g, "home_shot_attempts", "home_cf") > 0 or _num(g, "home_xgf", "home_xg") > 0:
            usable += 1
    if usable < 20:
        try:
            setattr(session, "_analytics_backfill_v2", True)
        except Exception:
            pass
        return False

    # Reset possession ledgers before rebuild (keep G/A/SOG/SA).
    reset_keys = (
        "cf",
        "ca",
        "ff",
        "fa",
        "xgf",
        "xga",
        "ixg",
        "xa",
        "gf_on",
        "ga_on",
        "on_ice_shots_for",
        "on_ice_shots_against",
        "goalie_xga",
        "analytics_gp",
        "xgf_pct_sum",
        "xgf_pct_gp",
        "quality_starts",
        "bad_starts",
    )
    for row in stats.values():
        if not isinstance(row, dict):
            continue
        if str(row.get("stat_scope") or "regular_season") != "regular_season":
            continue
        for key in reset_keys:
            if key in row:
                row[key] = 0.0 if key not in ("analytics_gp", "xgf_pct_gp", "quality_starts", "bad_starts") else 0

    def _skater_ids(game: Dict[str, Any], side: str) -> List[Tuple[str, int]]:
        key = "home_skaters" if side == "home" else "away_skaters"
        rows = list(game.get(key) or [])
        out: List[Tuple[str, int]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            pid = str(r.get("player_id") or r.get("id") or "")
            if not pid:
                continue
            pos = str(r.get("position") or "").upper()
            if pos == "G":
                continue
            toi = int(r.get("toi_sec") or 0)
            out.append((pid, max(1, toi)))
        return out

    def _add(pid: str, **kwargs: Any) -> None:
        row = stats.get(pid) or stats.get(str(pid))
        if not isinstance(row, dict):
            for cand in stats.values():
                if not isinstance(cand, dict):
                    continue
                if str(cand.get("player_id") or cand.get("id") or "") == str(pid):
                    row = cand
                    break
        if not isinstance(row, dict):
            return
        for k, v in kwargs.items():
            if k in ("analytics_gp", "xgf_pct_gp", "quality_starts", "bad_starts"):
                row[k] = int(row.get(k, 0) or 0) + int(v)
            else:
                row[k] = float(row.get(k, 0) or 0) + float(v)

    for g in games:
        h_cf = _num(g, "home_shot_attempts", "home_cf")
        a_cf = _num(g, "away_shot_attempts", "away_cf")
        h_ff = _num(g, "home_ff", "home_fenwick")
        a_ff = _num(g, "away_ff", "away_fenwick")
        h_xgf = _num(g, "home_xgf", "home_xg")
        a_xgf = _num(g, "away_xgf", "away_xg")
        h_sf = _num(g, "home_shots", "home_sog")
        a_sf = _num(g, "away_shots", "away_sog")
        h_gf = _num(g, "home_goals", "home_score", "player_home_goals")
        a_gf = _num(g, "away_goals", "away_score", "player_away_goals")
        if h_cf <= 0 and a_cf <= 0 and h_xgf <= 0 and a_xgf <= 0:
            continue

        for side, cf, ca, ff, fa, xgf, xga, sf, sa, gf, ga in (
            ("home", h_cf, a_cf, h_ff, a_ff, h_xgf, a_xgf, h_sf, a_sf, h_gf, a_gf),
            ("away", a_cf, h_cf, a_ff, h_ff, a_xgf, h_xgf, a_sf, h_sf, a_gf, h_gf),
        ):
            skaters = _skater_ids(g, side)
            if not skaters:
                continue
            toi_sum = float(sum(t for _, t in skaters)) or float(len(skaters))
            game_xgf_pct = (xgf / (xgf + xga)) if (xgf + xga) > 0 else 0.5
            for pid, toi in skaters:
                share = float(toi) / toi_sum
                _add(
                    pid,
                    cf=cf * share,
                    ca=ca * share,
                    ff=ff * share,
                    fa=fa * share,
                    xgf=xgf * share,
                    xga=xga * share,
                    on_ice_shots_for=sf * share,
                    on_ice_shots_against=sa * share,
                    gf_on=gf * share,
                    ga_on=ga * share,
                    analytics_gp=1,
                    xgf_pct_sum=game_xgf_pct,
                    xgf_pct_gp=1,
                )

        # Goalie expected goals = opponent xGF while they started.
        for side, faced_xga, faced_sa, faced_ga in (
            ("home", a_xgf, a_sf, a_gf),
            ("away", h_xgf, h_sf, h_gf),
        ):
            gkey = "home_goalies" if side == "home" else "away_goalies"
            for gr in list(g.get(gkey) or []):
                if not isinstance(gr, dict) or not gr.get("starter"):
                    continue
                pid = str(gr.get("player_id") or gr.get("id") or "")
                if not pid:
                    continue
                sa = int(gr.get("shots_against") or faced_sa or 0)
                ga = int(gr.get("ga") or faced_ga or 0)
                saves = int(gr.get("saves") or max(0, sa - ga))
                _add(pid, goalie_xga=float(faced_xga), xga=float(faced_xga))
                if sa > 0:
                    sv = saves / float(sa)
                    if sv >= 0.915:
                        _add(pid, quality_starts=1)
                    elif sv < 0.885:
                        _add(pid, bad_starts=1)

    try:
        setattr(session, "_analytics_backfill_v2", True)
    except Exception:
        pass
    _bump_stats_revision(session)
    return True


def _build_team_analytics_rows(session: FranchiseSession) -> List[Dict[str, Any]]:
    from app.sim_engine.generation.player_analytics import aggregate_team_from_player_rows, enrich_team_game_result_row

    def _num(row: Dict[str, Any], *keys: str) -> float:
        for key in keys:
            if key in row and row.get(key) is not None:
                try:
                    return float(row.get(key) or 0)
                except (TypeError, ValueError):
                    continue
        return 0.0

    team_event_totals: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "cf": 0.0,
            "ca": 0.0,
            "ff": 0.0,
            "fa": 0.0,
            "xgf": 0.0,
            "xga": 0.0,
            "sf": 0.0,
            "sa": 0.0,
            "ppg": 0.0,
            "ppo": 0.0,
            "ppga": 0.0,
            "opp_ppo": 0.0,
            "event_games": 0.0,
            "shot_games": 0.0,
            "pp_games": 0.0,
            "light_games": 0.0,
            "full_event_games": 0.0,
        }
    )
    for game in list(getattr(session, "game_results", None) or []):
        if not isinstance(game, dict):
            continue
        if str(game.get("stat_scope") or "regular_season") != "regular_season":
            continue
        hid = str(game.get("home_id") or game.get("home_team_id") or "")
        aid = str(game.get("away_id") or game.get("away_team_id") or "")
        if not hid or not aid:
            continue
        h_cf = _num(game, "home_shot_attempts", "home_cf")
        a_cf = _num(game, "away_shot_attempts", "away_cf")
        h_ff = _num(game, "home_ff", "home_fenwick")
        a_ff = _num(game, "away_ff", "away_fenwick")
        h_xgf = _num(game, "home_xgf", "home_xg")
        a_xgf = _num(game, "away_xgf", "away_xg")
        h_sf = _num(game, "home_shots", "home_sog")
        a_sf = _num(game, "away_shots", "away_sog")
        h_ppg = _num(game, "home_pp_goals")
        a_ppg = _num(game, "away_pp_goals")
        h_ppo = _num(game, "home_ppo")
        a_ppo = _num(game, "away_ppo")
        h_ppga = _num(game, "home_ppga", "away_pp_goals")
        a_ppga = _num(game, "away_ppga", "home_pp_goals")
        h_opp_ppo = _num(game, "home_opp_ppo", "away_ppo")
        a_opp_ppo = _num(game, "away_opp_ppo", "home_ppo")

        ht = team_event_totals[hid]
        at = team_event_totals[aid]

        is_light = bool(game.get("light_box")) or str(game.get("stat_source") or "") == "light_strength"
        if is_light:
            ht["light_games"] += 1.0
            at["light_games"] += 1.0
        else:
            ht["full_event_games"] += 1.0
            at["full_event_games"] += 1.0

        # COUNTING: always take SOG when present — light CPU–CPU boxes have shots
        # but historically lacked CF/xGF; both paths must count toward season totals.
        if h_sf > 0 or a_sf > 0:
            ht["sf"] += h_sf
            ht["sa"] += a_sf
            at["sf"] += a_sf
            at["sa"] += h_sf
            ht["shot_games"] += 1.0
            at["shot_games"] += 1.0

        if h_ppo > 0 or a_ppo > 0 or h_ppg > 0 or a_ppg > 0:
            ht["ppg"] += h_ppg
            ht["ppo"] += h_ppo
            ht["ppga"] += h_ppga if h_ppga > 0 else a_ppg
            ht["opp_ppo"] += h_opp_ppo if h_opp_ppo > 0 else a_ppo
            at["ppg"] += a_ppg
            at["ppo"] += a_ppo
            at["ppga"] += a_ppga if a_ppga > 0 else h_ppg
            at["opp_ppo"] += a_opp_ppo if a_opp_ppo > 0 else h_ppo
            ht["pp_games"] += 1.0
            at["pp_games"] += 1.0

        # Include light_strength CF/xGF — CPU–CPU bulk games write these now.
        if h_cf > 0 or a_cf > 0 or h_xgf > 0 or a_xgf > 0:
            ht["cf"] += h_cf
            ht["ca"] += a_cf
            ht["ff"] += h_ff
            ht["fa"] += a_ff
            ht["xgf"] += h_xgf
            ht["xga"] += a_xgf
            ht["event_games"] += 1.0

            at["cf"] += a_cf
            at["ca"] += h_cf
            at["ff"] += a_ff
            at["fa"] += h_ff
            at["xgf"] += a_xgf
            at["xga"] += h_xgf
            at["event_games"] += 1.0

    rows_by_team: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in list((getattr(session, "player_season_stats", None) or {}).values()):
        if not isinstance(row, dict):
            continue
        tid = str(row.get("team_id") or "")
        if tid:
            rows_by_team[tid].append(row)

    out: List[Dict[str, Any]] = []
    for tid, team in (getattr(session, "team_by_id", None) or {}).items():
        tid_s = str(tid)
        players = rows_by_team.get(tid_s, [])
        agg = aggregate_team_from_player_rows(players, team_id=tid_s)
        rec = None
        if session.standings:
            rec = session.standings.records.get(tid_s) or session.standings.records.get(tid)
        gp = int(getattr(rec, "gp", 0) or 0)
        # Standings records use gf/ga (not goals_for/goals_against). Wrong names
        # forced player-ledger fallback → absurd DIFF (e.g. last place with +161).
        gf = int(getattr(rec, "gf", None) or getattr(rec, "goals_for", 0) or 0)
        ga = int(getattr(rec, "ga", None) or getattr(rec, "goals_against", 0) or 0)
        if gf <= 0 and gp <= 0:
            gf = int(agg.get("gf_player_sum", 0) or 0)
        if ga <= 0 and gp <= 0:
            ga = int(agg.get("ga_goalie_sum", 0) or 0)
        base = {
            "team_id": tid_s,
            "team_name": _display_team(team),
            "name": _display_team(team),
            "gf": gf,
            "ga": ga,
            "w": int(getattr(rec, "wins", 0) or 0),
            "l": int(getattr(rec, "losses", 0) or 0),
            "otl": int(getattr(rec, "otl", 0) or 0),
            "gp": gp,
            "points": int(getattr(rec, "points", 0) or 0),
            "goal_diff": int(gf - ga),
            "diff": int(gf - ga),
        }
        event_totals = dict(team_event_totals.get(tid_s) or {})
        event_games = int(event_totals.get("event_games", 0) or 0)
        shot_games = int(event_totals.get("shot_games", 0) or 0)
        pp_games = int(event_totals.get("pp_games", 0) or 0)
        light_games = int(event_totals.get("light_games", 0) or 0)
        full_event_games = int(event_totals.get("full_event_games", 0) or 0)

        # Player-ledger SOG/SA are complete for light + full games; box event SF
        # was historically only full-event games and must not clobber counting.
        player_sf = int(agg.get("sog", 0) or 0)
        player_sa = int(agg.get("shots_against_goalie_sum", 0) or 0)
        event_sf = int(event_totals.get("sf", 0) or 0)
        event_sa = int(event_totals.get("sa", 0) or 0)
        sf = player_sf if player_sf > 0 else event_sf
        sa = player_sa if player_sa > 0 else event_sa
        if shot_games >= max(1, int(gp * 0.85)) and event_sf >= int(player_sf * 0.85):
            sf = max(player_sf, event_sf)
            sa = max(player_sa, event_sa)

        event_base: Dict[str, Any] = {
            "sf": int(sf),
            "sa": int(sa),
            "shots_for": int(sf),
            "shots_against": int(sa),
            "team_shot_games": shot_games,
            "team_event_games": event_games,
            "team_light_games": light_games,
            "team_full_event_games": full_event_games,
        }
        # Possession (CF/FF/xGF): prefer team box totals. Skater ledgers store
        # on-ice shares (×5 unit) — never use raw sum as team xGF (CPU ~1600 vs user ~550).
        player_cf = float(agg.get("cf", 0) or 0)
        player_ca = float(agg.get("ca", 0) or 0)
        player_ff = float(agg.get("ff", 0) or 0)
        player_fa = float(agg.get("fa", 0) or 0)
        player_xgf = float(agg.get("xgf", 0) or 0)
        player_xga = float(agg.get("xga", 0) or 0)
        # De-scale on-ice unit inflation when falling back to player sums.
        _ON_ICE = 5.0
        if player_cf > 0 or player_xgf > 0:
            player_cf = player_cf / _ON_ICE
            player_ca = player_ca / _ON_ICE
            player_ff = player_ff / _ON_ICE
            player_fa = player_fa / _ON_ICE
            player_xgf = player_xgf / _ON_ICE
            player_xga = player_xga / _ON_ICE
        poss_coverage_ok = event_games >= max(1, int(gp * 0.5)) and (
            float(event_totals.get("cf", 0) or 0) + float(event_totals.get("xgf", 0) or 0) > 0
        )
        if poss_coverage_ok:
            event_base.update(
                {
                    "cf": int(event_totals.get("cf", 0) or 0),
                    "ca": int(event_totals.get("ca", 0) or 0),
                    "ff": int(event_totals.get("ff", 0) or 0),
                    "fa": int(event_totals.get("fa", 0) or 0),
                    "xgf": round(float(event_totals.get("xgf", 0) or 0), 4),
                    "xga": round(float(event_totals.get("xga", 0) or 0), 4),
                    "team_event_stats_source": "game_results",
                }
            )
        elif player_cf + player_ca > 0 or player_xgf + player_xga > 0:
            event_base.update(
                {
                    "cf": int(round(player_cf)),
                    "ca": int(round(player_ca)),
                    "ff": int(round(player_ff)),
                    "fa": int(round(player_fa)),
                    "xgf": round(player_xgf, 4),
                    "xga": round(player_xga, 4),
                    "team_event_stats_source": "player_season_stats_descaled",
                }
            )
        # PP counting: box totals when coverage is decent, else player PPG only.
        if pp_games >= max(1, int(gp * 0.5)):
            event_base["ppg"] = int(event_totals.get("ppg", 0) or 0)
            event_base["ppo"] = int(event_totals.get("ppo", 0) or 0)
            event_base["ppga"] = int(event_totals.get("ppga", 0) or 0)
            event_base["opp_ppo"] = int(event_totals.get("opp_ppo", 0) or 0)
        else:
            event_base["ppg"] = int(agg.get("ppg", 0) or 0)

        # Counting overlay wins over sparse analytics; keep standings W/L/GF/GA.
        merged = {**agg, **event_base, **base}
        cf = float(merged.get("cf", 0) or 0)
        ca = float(merged.get("ca", 0) or 0)
        ff = float(merged.get("ff", 0) or 0)
        fa = float(merged.get("fa", 0) or 0)
        xgf = float(merged.get("xgf", 0) or 0)
        xga = float(merged.get("xga", 0) or 0)
        sf = int(merged.get("sf", 0) or 0)
        sa = int(merged.get("sa", 0) or 0)
        merged["cf_pct"] = round(cf / (cf + ca), 4) if cf + ca > 0 else None
        merged["cf_pct_valid"] = bool(cf + ca > 0)
        merged["ff_pct"] = round(ff / (ff + fa), 4) if ff + fa > 0 else None
        merged["ff_pct_valid"] = bool(ff + fa > 0)
        merged["xgf_pct"] = round(xgf / (xgf + xga), 4) if xgf + xga > 0 else None
        merged["xgf_pct_valid"] = bool(xgf + xga > 0)
        merged["sf"] = sf
        merged["sa"] = sa
        merged["shots_for"] = sf
        merged["shots_against"] = sa
        merged["sf_pct"] = round(sf / (sf + sa), 4) if sf + sa > 0 else None
        # SH% from standings GF / skater SOG. SV% from goalie ledger (not standings
        # GA vs mixed SA — that produced .96 team SV% while starters sat at .907).
        sh_pct = (gf / float(sf)) if sf > 0 else None
        if sh_pct is not None:
            sh_pct = max(0.0, min(0.35, sh_pct))
        goalie_sa = int(agg.get("shots_against_goalie_sum", 0) or 0)
        goalie_saves = int(agg.get("saves_goalie_sum", 0) or 0)
        goalie_ga = int(agg.get("ga_goalie_sum", 0) or 0)
        if goalie_sa > 0 and goalie_saves >= 0:
            sv_pct = goalie_saves / float(goalie_sa)
            sa = goalie_sa
            merged["sa"] = sa
            merged["shots_against"] = sa
        elif sa > 0:
            sv_pct = ((sa - ga) / float(sa))
        else:
            sv_pct = None
        if sv_pct is not None:
            sv_pct = max(0.70, min(0.995, sv_pct))
        merged["sh_pct"] = round(sh_pct, 4) if sh_pct is not None else None
        merged["sv_pct"] = round(sv_pct, 4) if sv_pct is not None else None
        merged["ga_goalie_sum"] = goalie_ga
        merged["pdo"] = (
            round((float(sh_pct) + float(sv_pct)) * 100.0, 1)
            if sh_pct is not None and sv_pct is not None
            else None
        )
        merged["pdo_valid"] = bool(merged["pdo"] is not None)
        merged["corsi_for"] = int(cf)
        merged["corsi_against"] = int(ca)
        merged["shot_attempts_for"] = int(cf)
        merged["shot_attempts_against"] = int(ca)
        merged["fenwick_for"] = int(ff)
        merged["fenwick_against"] = int(fa)
        merged["expected_goals_for"] = round(xgf, 4)
        merged["expected_goals_against"] = round(xga, 4)
        ppo = int(merged.get("ppo", 0) or 0)
        ppg = int(merged.get("ppg", 0) or 0)
        opp_ppo = int(merged.get("opp_ppo", 0) or 0)
        ppga = int(merged.get("ppga", 0) or 0)
        merged["pp_pct"] = round(ppg / float(ppo), 4) if ppo > 0 else None
        merged["pk_pct"] = round(1.0 - (ppga / float(opp_ppo)), 4) if opp_ppo > 0 else None
        gp_analytics = max((int(p.get("analytics_gp", 0) or 0) for p in players), default=0)
        gp_play = max((int(p.get("gp", 0) or 0) for p in players if str(p.get("position", "")).upper() != "G"), default=0)
        merged["analytics_gp"] = gp_analytics
        merged["analytics_coverage_pct"] = round(gp_analytics / float(max(1, gp_play)), 4) if gp_play > 0 else 0.0
        out.append(enrich_team_game_result_row(merged))
    return out


def _league_weighted_shooting_metrics(session: FranchiseSession) -> Dict[str, Any]:
    rows = [
        row
        for row in list((getattr(session, "player_season_stats", None) or {}).values())
        if isinstance(row, dict) and str(row.get("stat_scope") or "regular_season") == "regular_season"
    ]
    total_g = total_sog = total_saves = total_sa = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        pos = str(row.get("position") or "").upper()
        if pos == "G":
            sa = int(row.get("shots_against", row.get("sa", 0)) or 0)
            sv = int(row.get("saves", 0) or 0)
            total_sa += sa
            total_saves += sv
        else:
            total_g += int(row.get("g", 0) or 0)
            total_sog += int(row.get("sog", 0) or 0)
    w_sh = (total_g / float(total_sog)) if total_sog > 0 else 0.0
    w_sv = (total_saves / float(total_sa)) if total_sa > 0 else 0.0
    games = [
        g for g in list(getattr(session, "game_results", None) or [])
        if isinstance(g, dict) and str(g.get("stat_scope") or "regular_season") == "regular_season"
    ]
    total_game_count = len(games)
    total_team_games = total_game_count * 2
    game_goals = sum(
        int(g.get("player_home_goals", g.get("hockey_home_goals", g.get("home_goals", g.get("home_score", 0)))) or 0)
        + int(g.get("player_away_goals", g.get("hockey_away_goals", g.get("away_goals", g.get("away_score", 0)))) or 0)
        for g in games
    )
    combined_gpg = game_goals / total_game_count if total_game_count else 0.0
    sog_per_team_game = total_sog / total_team_games if total_team_games else 0.0
    implied_combined = 2.0 * sog_per_team_game * w_sh if sog_per_team_game and w_sh else 0.0
    return {
        "total_goals": int(total_g),
        "total_sog": int(total_sog),
        "weighted_league_sh_pct": round(w_sh, 4),
        "total_saves": int(total_saves),
        "total_sa": int(total_sa),
        "weighted_league_sv_pct": round(w_sv, 4),
        "weighted_sh_plus_sv": round(w_sh + w_sv, 4),
        "total_games": int(total_game_count),
        "total_team_games": int(total_team_games),
        "combined_gpg_from_games": round(combined_gpg, 4),
        "sog_per_team_game": round(sog_per_team_game, 4),
        "scoring_identity_implied_combined_gpg": round(implied_combined, 4),
        "scoring_identity_delta": round(abs(combined_gpg - implied_combined), 4) if implied_combined else None,
    }


def _build_stats_central_payload(session: FranchiseSession) -> Dict[str, Any]:
    """
    StatsCentral payload via canonical player_analytics enrichment.
    """
    try:
        _purge_synthetic_universe_artifacts(session)
    except Exception:
        logging.getLogger(__name__).exception("Synthetic universe purge failed")

    try:
        _backfill_missing_toi_from_game_boxes(session)
    except Exception:
        logging.getLogger(__name__).exception("Player TOI backfill failed")

    try:
        _backfill_player_analytics_from_game_boxes(session)
    except Exception:
        logging.getLogger(__name__).exception("Player analytics backfill failed")

    try:
        from app.sim_engine.gameplay.game_analytics_ledger import league_assist_health_metrics
        from app.sim_engine.generation.player_analytics import (
            build_stats_central_player_payload,
            enrich_team_rows,
        )

        uid = str(getattr(session, "user_team_id", "") or "")
        team_by_player_id: Dict[str, str] = {}
        player_by_id: Dict[str, Any] = {}
        ovr_by_player_id: Dict[str, float] = {}
        try:
            from app.sim_engine.engine import career_ovr_0_100
        except Exception:
            career_ovr_0_100 = None  # type: ignore
        for tm in list(getattr(getattr(session, "sim", None), "league", None).teams or []):
            tid = str(getattr(tm, "team_id", None) or getattr(tm, "id", "") or "")
            if not tid:
                continue
            for pl in list(getattr(tm, "roster", None) or []):
                pid = str(getattr(pl, "id", "") or "")
                if not pid:
                    continue
                team_by_player_id[pid] = tid
                player_by_id[pid] = pl
                if career_ovr_0_100 is not None:
                    try:
                        ovr_by_player_id[pid] = float(career_ovr_0_100(pl))
                    except Exception:
                        pass
                else:
                    try:
                        ovr_by_player_id[pid] = float(
                            getattr(pl, "overall", None)
                            or getattr(pl, "ovr", None)
                            or getattr(pl, "effective_ovr", None)
                            or 0
                        )
                    except Exception:
                        pass

        rows: List[Dict[str, Any]] = []
        for src in list((getattr(session, "player_season_stats", None) or {}).values()):
            if not isinstance(src, dict):
                continue
            if str(src.get("stat_scope") or "regular_season") != "regular_season":
                continue
            # Copy — never mutate live ledger team_id (broke team GF vs standings after trades).
            row = dict(src)
            pid = str(row.get("player_id") or row.get("id") or "")
            player = player_by_id.get(pid)
            if player is not None:
                try:
                    from app.sim_engine.generation.player_headshots import merge_headshot_into_row

                    row = merge_headshot_into_row(row, player)
                except Exception:
                    pass
            live_tid = team_by_player_id.get(pid)
            if live_tid:
                row["current_team_id"] = live_tid
            ovr = ovr_by_player_id.get(pid)
            if ovr and ovr > 0:
                row["ovr"] = round(ovr, 1)
                row["overall"] = round(ovr, 1)
                row["effective_ovr"] = round(ovr, 1)
            rows.append(row)

        enriched = build_stats_central_player_payload(rows, user_team_id=uid, leader_limit=100)
        user_player_ids: set[str] = set()
        for tm in list(getattr(getattr(session, "sim", None), "league", None).teams or []):
            tid = str(getattr(tm, "team_id", None) or getattr(tm, "id", "") or "")
            if tid != uid:
                continue
            for pl in list(getattr(tm, "roster", None) or []):
                pid = str(getattr(pl, "id", "") or "")
                if pid:
                    user_player_ids.add(pid)
            break
        if user_player_ids:
            enriched["user_team_skaters"] = [
                r for r in list(enriched.get("skaters") or [])
                if str(r.get("player_id") or r.get("id") or "") in user_player_ids
            ]
            enriched["user_team_goalies"] = [
                r for r in list(enriched.get("goalies") or [])
                if str(r.get("player_id") or r.get("id") or "") in user_player_ids
            ]

        team_rows = enrich_team_rows(_build_team_analytics_rows(session))
        all_results = [g for g in list(getattr(session, "game_results", None) or []) if isinstance(g, dict)]
        integrity = _stats_integrity_payload(rows, all_results)
        goal_events: List[Dict[str, Any]] = []
        for g in all_results:
            for ev in g.get("scoring_events") or []:
                if isinstance(ev, dict):
                    goal_events.append(ev)
        integrity["assist_health"] = league_assist_health_metrics(goal_events)
        # Light boxes omit scoring_events — supplement with ledger assist rate.
        try:
            sk_g = sum(int(r.get("g", 0) or 0) for r in rows if str(r.get("position") or "").upper() != "G")
            sk_a = sum(int(r.get("a", 0) or 0) for r in rows if str(r.get("position") or "").upper() != "G")
            ah = dict(integrity.get("assist_health") or {})
            ah["ledger_assists_per_goal"] = round(sk_a / sk_g, 4) if sk_g > 0 else 0.0
            ah["ledger_skater_goals"] = int(sk_g)
            ah["ledger_skater_assists"] = int(sk_a)
            if int(ah.get("total_goals") or 0) < max(20, sk_g // 4):
                ah["source"] = "mixed_light_boxes"
            integrity["assist_health"] = ah
        except Exception:
            pass
        integrity["league_shooting"] = _league_weighted_shooting_metrics(session)
        enriched["integrity"] = integrity
        enriched["team_analytics"] = team_rows
        enriched["league_team_stats"] = team_rows
        enriched["teams"] = team_rows
        enriched["games"] = list(reversed(all_results[-100:]))
        enriched["calendar"] = enriched.get("calendar") or []
        enriched["leader_limit"] = 100
        enriched["players"] = enriched.get("skaters", [])
        enriched["goalies"] = enriched.get("goalies", [])
        enriched["user_leaders"] = enriched.get("user_team_skaters", [])[:20]
        enriched["goalie_leaders"] = enriched.get("league_goalies", [])[:20]
        enriched["stats_revision"] = int(getattr(session, "_stats_revision", 0) or 0)
        return enriched
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Stats Central enrichment failed; using fallback payload")

    # Fallback to basic ledger payload if enrichment unavailable
    all_results = [
        g for g in list(getattr(session, "game_results", None) or [])
        if isinstance(g, dict) and str(g.get("stat_scope") or "regular_season") == "regular_season"
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

    raw_rows = [
        r
        for r in list((getattr(session, "player_season_stats", None) or {}).values())
        if isinstance(r, dict) and str(r.get("stat_scope") or "regular_season") == "regular_season"
    ]
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
    # Explicit skater positions must never be reclassified via empty goalie keys
    # that the season ledger initializes on every row (ga/w/l/otl = 0).
    if pos in {"C", "LW", "RW", "D", "F", "W"}:
        return False

    sa = _stat_int(row, "shots_against", "sa", "goalie_shots_against")
    saves = _stat_int(row, "saves", "sv")
    return (sa > 0 or saves > 0) and not (_stat_int(row, "g") or _stat_int(row, "a") or _stat_int(row, "sog"))


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

    def _row_is_goalie(r: Dict[str, Any]) -> bool:
        if r.get("is_goalie") is True:
            return True
        pos = str(r.get("position") or r.get("pos") or "").strip().upper()
        return pos in {"G", "GOALIE", "GOALTENDER"}

    skaters = [r for r in rows if not _row_is_goalie(r)]
    goalies = [r for r in rows if _row_is_goalie(r)]

    total_player_goals = sum(int(r.get("g", 0) or 0) for r in skaters)

    total_box_goals = 0
    valid_games = 0
    light_games = 0
    full_event_games = 0
    light_with_cf = 0
    light_with_shots = 0

    for g in game_results or []:
        if not isinstance(g, dict):
            continue

        try:
            hg = int(round(float(g.get("player_home_goals", g.get("hockey_home_goals", g.get("home_goals", g.get("home_score", 0)))) or 0)))
            ag = int(round(float(g.get("player_away_goals", g.get("hockey_away_goals", g.get("away_goals", g.get("away_score", 0)))) or 0)))
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
        is_light = bool(g.get("light_box")) or str(g.get("stat_source") or "") == "light_strength"
        if is_light:
            light_games += 1
            try:
                hsf = float(g.get("home_shots", g.get("home_sog", 0)) or 0)
                asf = float(g.get("away_shots", g.get("away_sog", 0)) or 0)
            except (TypeError, ValueError):
                hsf = asf = 0.0
            if hsf > 0 or asf > 0:
                light_with_shots += 1
            try:
                hcf = float(g.get("home_shot_attempts", g.get("home_cf", 0)) or 0)
                acf = float(g.get("away_shot_attempts", g.get("away_cf", 0)) or 0)
                hx = float(g.get("home_xgf", g.get("home_xg", 0)) or 0)
                ax = float(g.get("away_xgf", g.get("away_xg", 0)) or 0)
            except (TypeError, ValueError):
                hcf = acf = hx = ax = 0.0
            if hcf > 0 or acf > 0 or hx > 0 or ax > 0:
                light_with_cf += 1
        else:
            full_event_games += 1

    if valid_games and abs(total_player_goals - total_box_goals) > 0:
        warnings.append(
            f"PLAYER_GOALS_MISMATCH: skater goals {total_player_goals} != game box goals {total_box_goals}."
        )

    top_pts = max([int(r.get("pts", 0) or 0) for r in skaters], default=0)
    top_gp = max([int(r.get("gp", 0) or 0) for r in skaters], default=0)

    if top_gp > 82:
        over = [
            f"{r.get('name') or r.get('player_id')}={int(r.get('gp', 0) or 0)}"
            for r in skaters
            if int(r.get("gp", 0) or 0) > 82
        ][:8]
        warnings.append(
            f"PLAYER_GP_OVER_82: max gp={top_gp} ({', '.join(over)}). "
            "Likely dual-roster or duplicate game processing."
        )

    if valid_games >= 300 and top_pts < 45:
        warnings.append(
            f"LOW_LEAGUE_SCORING: top scorer has only {top_pts} points after {valid_games} completed games."
        )

    if light_games >= 50 and light_with_shots < int(light_games * 0.5):
        warnings.append(
            f"LIGHT_BOX_SHOTS_MISSING: only {light_with_shots}/{light_games} CPU–CPU boxes have SOG."
        )
    if light_games >= 50 and light_with_cf < int(light_games * 0.5):
        warnings.append(
            f"LIGHT_BOX_CF_MISSING: only {light_with_cf}/{light_games} CPU–CPU boxes have CF/xGF "
            "(Stats Central team possession may under-count bulk games)."
        )

    return {
        "skater_rows": len(skaters),
        "goalie_rows": len(goalies),
        "valid_games_counted": int(valid_games),
        "light_games_counted": int(light_games),
        "full_event_games_counted": int(full_event_games),
        "light_games_with_shots": int(light_with_shots),
        "light_games_with_cf": int(light_with_cf),
        "cpu_cpu_games_included": bool(light_games == 0 or light_with_shots >= int(light_games * 0.5)),
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
    phase = str(getattr(session, "phase", "") or "")
    if not cal or phase not in ("regular", "preseason"):
        return {}
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    last = int(getattr(session, "nhl_regular_season_last_index", 0) or 0)
    if phase == "regular" and cur > last:
        return {"headline": "Regular season complete — advance for playoffs", "iso": "", "segment": "regular", "calendar_index": cur}
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


def _scheduled_regular_games_count(session: FranchiseSession) -> int:
    """Active schedule is the source of truth for expected regular-season games."""
    schedule = getattr(session, "schedule", None) or []
    if schedule:
        return int(len(schedule))
    # Fallback: sum all by_day slots that were originally scheduled (may be empty mid-season).
    by_day = getattr(session, "by_day", None) or {}
    return int(sum(len(slots or []) for slots in by_day.values()))


def _remaining_regular_games_count(session: FranchiseSession) -> int:
    """
    Count unplayed regular-season scheduled games remaining.
    Used to prevent playoffs from starting too early.

    Counts slots on regular-segment days AND any leftover slots still sitting on
    preseason-segment days (legacy mapping bug / mid-repair leftovers).
    """
    cal = getattr(session, "nhl_calendar", None) or []
    by_day = getattr(session, "by_day", None) or {}

    remaining = 0

    for day_idx, slots in (by_day or {}).items():
        try:
            di = int(day_idx)
        except (TypeError, ValueError):
            continue

        if not slots:
            continue

        if di < 0 or di >= len(cal):
            # Orphaned slots still count as remaining work.
            remaining += len(slots or [])
            continue

        row = cal[di] or {}
        seg = str(row.get("segment") or row.get("season_segment") or "")

        # Regular matchups must never be ignored just because they landed on a
        # preseason calendar index (historical mapping bug).
        if seg in ("regular", "preseason", ""):
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
            di = -1

        # Prefer explicit segment tags on the game payload when present.
        gseg = str(g.get("segment") or g.get("season_segment") or "").lower()
        if gseg and gseg not in ("regular", "reg", "rs"):
            continue

        if di >= 0 and di < len(cal):
            row = cal[di] or {}
            seg = str(row.get("segment") or row.get("season_segment") or "")
            # After the mapping fix, regular games live on regular days.
            # Still accept finals without a calendar row (light boxes).
            if seg and seg not in ("regular", "preseason", ""):
                continue

        if _saved_game_is_final(g):
            count += 1

    return int(count)


def _dump_remaining_regular_games(session: FranchiseSession, *, reason: str = "") -> List[Dict[str, Any]]:
    """Diagnostic dump used when playoff_ready is blocked by remaining games."""
    cal = getattr(session, "nhl_calendar", None) or []
    by_day = getattr(session, "by_day", None) or {}
    out: List[Dict[str, Any]] = []
    for day_idx, slots in sorted(by_day.items(), key=lambda kv: int(kv[0]) if str(kv[0]).lstrip("-").isdigit() else 0):
        try:
            di = int(day_idx)
        except (TypeError, ValueError):
            di = -1
        row = cal[di] if 0 <= di < len(cal) else {}
        seg = str((row or {}).get("segment") or "")
        iso = str((row or {}).get("iso") or "")
        for slot in slots or []:
            hid = str(getattr(slot, "home_id", getattr(slot, "home_team_id", "")) or "")
            aid = str(getattr(slot, "away_id", getattr(slot, "away_team_id", "")) or "")
            out.append(
                {
                    "day_index": di,
                    "iso": iso,
                    "segment": seg,
                    "home_id": hid,
                    "away_id": aid,
                    "completion_state": "unplayed_slot",
                    "season_calendar_year": int(getattr(session, "season_calendar_year", 0) or 0),
                    "not_selected_reason": reason
                    or (
                        "slot_on_preseason_segment"
                        if seg == "preseason"
                        else "slot_still_in_by_day"
                    ),
                }
            )
    return out


def _regular_season_completion_snapshot(session: FranchiseSession) -> Dict[str, Any]:
    scheduled = _scheduled_regular_games_count(session)
    completed = _completed_regular_games_count(session)
    remaining_slots = _remaining_regular_games_count(session)
    return {
        "scheduled_regular_games": int(scheduled),
        "completed_regular_games": int(completed),
        "remaining_regular_games": int(max(0, scheduled - completed)),
        "remaining_by_day_slots": int(remaining_slots),
        "calendar_cursor": int(getattr(session, "calendar_cursor", 0) or 0),
        "nhl_regular_season_last_index": int(getattr(session, "nhl_regular_season_last_index", 0) or 0),
        "phase": str(getattr(session, "phase", "") or ""),
    }


def _regular_season_is_truly_complete(session: FranchiseSession) -> bool:
    """
    True only when:
    - calendar cursor is past the regular season boundary
    - no regular-season slots remain in by_day
    - completed regular games >= scheduled regular games (schedule is source of truth)
    """
    _sync_nhl_calendar_bounds(session)

    cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    last = int(getattr(session, "nhl_regular_season_last_index", 0) or 0)

    if cursor <= last:
        return False

    remaining_slots = _remaining_regular_games_count(session)
    if remaining_slots > 0:
        return False

    scheduled = _scheduled_regular_games_count(session)
    completed = _completed_regular_games_count(session)
    if scheduled > 0 and completed < scheduled:
        if bool(getattr(session, "_audit_schedule_invariants", False)) or os.environ.get("NHL_FRANCHISE_AUDIT") == "1":
            dump = _dump_remaining_regular_games(session, reason="completed_lt_scheduled")
            snap = _regular_season_completion_snapshot(session)
            print(
                "[SCHEDULE INVARIANT] playoff_ready blocked: "
                f"completed={completed} scheduled={scheduled} remaining_slots={remaining_slots} "
                f"snapshot={snap} orphan_dump_n={len(dump)}"
            )
            if dump:
                print("[SCHEDULE INVARIANT] remaining slots sample:", dump[:50])
        return False

    return True
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


def _nhl_calendar_full_with_slates(
    session: FranchiseSession,
    *,
    cursor_window: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Full season calendar rows plus NHL slate + user focus + final scores for the franchise UI.

    cursor_window=(behind, ahead): lean payloads only ship days near the cursor so
    /advance responses stay small enough for the browser (full season is multi‑MB).
    """
    cal = getattr(session, "nhl_calendar", None) or []
    if not cal:
        return []
    by_day = getattr(session, "by_day", None) or {}
    uid = str(session.user_team_id)
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    results_by_idx = _results_by_calendar_index(session)
    saved_by_day = _game_results_by_calendar_day(session)

    n = len(cal)
    if cursor_window is not None:
        behind, ahead = int(cursor_window[0]), int(cursor_window[1])
        i_lo = max(0, cur - max(0, behind))
        i_hi = min(n, cur + max(0, ahead) + 1)
        day_indices = range(i_lo, i_hi)
    else:
        day_indices = range(n)

    out: List[Dict[str, Any]] = []
    for i in day_indices:
        row = cal[i]
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
    roster_kind: str = "",
) -> Dict[str, Any]:
    ident = getattr(p, "identity", None)
    if session is not None:
        try:
            sync_player_age_to_session(p, session)
        except Exception:
            pass
    ovr_f = getattr(p, "ovr", None)
    try:
        ov = float(ovr_f() if callable(ovr_f) else ovr_f)
    except Exception:
        ov = 0.0
    pid = str(getattr(p, "id", "") or "")
    pos_raw = getattr(ident, "position", None) if ident else None
    pos_str = str(getattr(pos_raw, "value", pos_raw) or "?")
    hcm = clamp_height_cm_for_position(getattr(ident, "height_cm", 0) if ident else 0, pos_str)
    wkg = int(getattr(ident, "weight_kg", 0) or 0) if ident else 0
    cap_hit_m = round(_player_cap_hit_millions(p), 3)
    display_name = str(getattr(ident, "name", None) or "?")
    name_tags: List[str] = []
    try:
        from services.brady_tkachuk_chaos import (
            display_name_with_cancer_tag,
            is_brady_tkachuk,
        )

        if is_brady_tkachuk(p):
            display_name = display_name_with_cancer_tag(display_name, p)
            name_tags = ["CANCER"]
    except Exception:
        pass
    row: Dict[str, Any] = {
        "player_id": pid,
        "name": display_name,
        "position": pos_str,
        "handedness": str(getattr(ident, "shoots", "") or "") if ident else "",
        "ovr": round(ov * 99, 1) if ov <= 1.5 else round(ov, 1),
        "age": int(getattr(ident, "age", 0) or 0),
        "birth_date": (
            str(getattr(p, "birth_date", None) or getattr(p, "_birth_date", None) or "")[:10]
            or (
                f"{int(getattr(ident, 'birth_year', 0) or 0):04d}-"
                f"{int(getattr(ident, 'birth_month', 1) or 1):02d}-"
                f"{int(getattr(ident, 'birth_day', 1) or 1):02d}"
                if ident is not None and int(getattr(ident, "birth_year", 0) or 0) > 1900
                else ""
            )
        ),
        "nationality": str(getattr(ident, "birth_country", "") or ""),
        "height_cm": hcm,
        "height_display": height_cm_to_imperial(hcm) if hcm else "—",
        "height": height_cm_to_imperial(hcm) if hcm else "—",
        "weight_kg": wkg,
        "weight": round(wkg * 2.20462) if wkg > 0 else 0,
        "archetype": str(getattr(p, "archetype", "") or ""),
        "contract": {
            "salary": cap_hit_m,
            "cap_hit": cap_hit_m,
        },
    }
    try:
        from app.sim_engine.generation.player_headshots import merge_headshot_into_row

        # One serialization seam for NHL photography and deterministic fallback
        # metadata. Optional fields keep older saves and API consumers valid.
        row = merge_headshot_into_row(row, p)
    except Exception:
        pass
    if name_tags:
        row["name_tags"] = list(name_tags)
        row["locker_room_cancer"] = True
        row["brady_tkachuk_chaos"] = True
        row["display_name_tag"] = "CANCER"
    # Normalized 0–100 psych + chemistry profile for UI contracts.
    try:
        from app.sim_engine.systems.chemistry import (  # noqa: WPS433
            ensure_player_chemistry_profile,
            safe_get_psych,
            coach_system_fit_score,
            usage_satisfaction_score,
        )

        psych01 = safe_get_psych(p)
        morale100 = int(round(float(psych01.get("morale", 0.5)) * 100.0))
        conf100 = int(round(float(psych01.get("confidence", 0.5)) * 100.0))
        role100 = int(round(float(psych01.get("role_satisfaction", 0.5)) * 100.0))
        coach_trust_raw = getattr(getattr(p, "psych", None), "coach_trust", None)
        if coach_trust_raw is None:
            coach100 = int(round((conf100 + role100) / 2.0))
        else:
            ct = float(coach_trust_raw or 0.5)
            coach100 = int(round(ct * 100.0 if ct <= 1.5 else ct))
        prof = dict(ensure_player_chemistry_profile(p) or {})
        prof["morale"] = morale100
        prof["confidence"] = conf100
        prof["role_satisfaction"] = role100
        prof["coach_trust"] = coach100
        prof["compete"] = int(prof.get("competitiveness", prof.get("compete", 50)) or 50)
        prof["adaptability"] = int(prof.get("adaptability", 50) or 50)
        prof["leadership"] = int(prof.get("leadership", 50) or 50)
        prof["coach_system_fit"] = int(round(coach_system_fit_score(p, _team)))
        prof["usage_satisfaction"] = int(round(usage_satisfaction_score(p)))
        row["morale"] = morale100
        row["confidence"] = conf100
        row["role_satisfaction"] = role100
        row["coach_trust"] = coach100
        row["chemistry_profile"] = prof
    except Exception:
        m_raw = float(getattr(getattr(p, "psych", None), "morale", 0.5) or 0.5)
        row["morale"] = int(round(m_raw * 100.0 if m_raw <= 1.5 else m_raw))
        row["confidence"] = 50
        row["role_satisfaction"] = 50
        row["coach_trust"] = 50
        row["chemistry_profile"] = None
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
    try:
        from app.sim_engine.franchise.conduct_incidents import (  # noqa: WPS433
            get_active_incident_for_player,
            player_eligible_to_dress,
            serialize_incident_for_ui,
        )

        eligible = bool(player_eligible_to_dress(p, session))
        row["conduct_eligible_to_play"] = eligible
        row["conduct_incident_id"] = str(getattr(p, "_conduct_incident_id", "") or "")
        row["conduct_dress_backlash_risk"] = float(getattr(p, "_conduct_dress_backlash_risk", 0) or 0)
        cgr = int(getattr(p, "_world_conduct_games_remaining", 0) or 0)
        row["conduct_games_remaining"] = cgr
        row["conduct_trade_restricted"] = bool(getattr(p, "_conduct_trade_restricted", False))
        if not eligible:
            row["availability_status"] = "Suspended" if str(getattr(p, "_world_conduct_status", "") or "") == "league_suspended" else "Leave"
            if cgr > 0 and session is not None:
                cret, ciso = _estimate_return_from_games_remaining(session, cgr)
                row["return_estimate"] = cret or row.get("return_estimate") or ""
                row["return_date"] = ciso or row.get("return_date") or ""
            elif cgr > 0:
                row["return_estimate"] = f"In {cgr} games"
        if session is not None and row.get("conduct_incident_id"):
            inc = get_active_incident_for_player(session, str(getattr(p, "id", "") or getattr(p, "player_id", "") or ""))
            if isinstance(inc, dict):
                row["conduct_incident"] = serialize_incident_for_ui(inc)
    except Exception:
        pass
    if include_ratings:
        row["rating_groups"] = _rating_groups_for_player(p)
        try:
            from app.sim_engine.entities.chapter_attributes import serialize_chapter_profile_for_api  # noqa: WPS433

            chapter_payload = serialize_chapter_profile_for_api(p)
            if chapter_payload:
                row["chapter_profile"] = chapter_payload
        except Exception:
            pass
    try:
        from app.sim_engine.franchise.storyline_conduct import (  # noqa: WPS433
            get_base_ovr_display,
            get_effective_ovr_display,
            serialize_ovr_modifiers_for_ui,
        )

        # Universal display OVR for every team (user + CPU): effective ratings blend.
        # Keep sim_ovr separately so callers can still see raw engine ovr().
        base_ovr = int(get_base_ovr_display(p))
        eff_ovr = int(get_effective_ovr_display(p))
        mods = serialize_ovr_modifiers_for_ui(p)
        row["sim_ovr"] = row.get("ovr")
        row["base_ovr"] = base_ovr
        row["effective_ovr"] = eff_ovr
        row["ovr"] = eff_ovr
        row["overall"] = eff_ovr
        row["ovr_modifiers"] = mods
        row["overall_drop"] = max(0, base_ovr - eff_ovr)
    except Exception:
        pass
    def _plain_draft(v: Any) -> Any:
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if callable(v):
            return None
        return str(v)

    draft_year = getattr(p, "draft_year", None)
    if draft_year is None and ident is not None:
        draft_year = getattr(ident, "draft_year", None)
    draft_round = getattr(p, "draft_round", None)
    if draft_round is None and ident is not None:
        draft_round = getattr(ident, "draft_round", None)
    draft_overall = getattr(p, "draft_overall_pick", None)
    if draft_overall is None and ident is not None:
        draft_overall = getattr(ident, "draft_pick", None)
    is_drafted = bool(getattr(p, "drafted", False) or draft_overall or draft_year)
    is_undrafted = bool(getattr(p, "undrafted", False)) and not is_drafted
    if is_drafted or is_undrafted or draft_year or draft_overall:
        row.update({
            "drafted": is_drafted,
            "undrafted": is_undrafted or (not is_drafted and bool(getattr(p, "undrafted", False))),
            "draft_year": _plain_draft(draft_year),
            "draft_round": _plain_draft(draft_round),
            "draft_overall_pick": _plain_draft(draft_overall),
            "draft_team_id": _plain_draft(getattr(p, "draft_team_id", None)),
            "drafted_by_team_id": _plain_draft(getattr(p, "drafted_by", None) or getattr(p, "draft_team_id", None)),
            "nhl_rights_team_id": _plain_draft(getattr(p, "nhl_rights_team_id", None)),
            "rights_team_id": _plain_draft(
                getattr(p, "rights_team_id", None) or getattr(p, "nhl_rights_team_id", None)
            ),
            "rights_status": _plain_draft(getattr(p, "rights_status", None)),
            "signed_status": _plain_draft(getattr(p, "signed_status", None)),
            "development_path": _plain_draft(getattr(p, "development_path", None)),
            "nhl_eta": _plain_draft(getattr(p, "nhl_eta", None)),
            "prospect_status": _plain_draft(getattr(p, "prospect_status", None)),
            "draft_profile_summary": _plain_draft(getattr(p, "draft_profile_summary", None)),
        })
    elif getattr(p, "undrafted", False):
        row["drafted"] = False
        row["undrafted"] = True
    # Rights / org status always (prospects + veterans).
    try:
        row["rights_type"] = str(getattr(p, "rights_type", "") or "") or None
        row["rights_expiry_year"] = getattr(p, "rights_expiry_year", None)
        row["organizational_status"] = str(getattr(p, "organizational_status", "") or "") or None
        row["signed_status"] = row.get("signed_status") or str(getattr(p, "signed_status", "") or "") or None
        row["rights_status"] = row.get("rights_status") or str(getattr(p, "rights_status", "") or "") or None
        row["elc_eligible"] = bool(getattr(p, "entry_level_contract_eligible", False))
        row["elc_slide_eligible"] = bool(getattr(p, "elc_slide_eligible", False))
        row["slide_games_threshold"] = getattr(p, "slide_games_threshold", None)
        row["roster_location"] = str(getattr(p, "roster_location", "") or "") or None
        row["in_minors"] = bool(getattr(p, "in_minors", False) or getattr(p, "is_buried", False))
        row["waiver_status"] = str(getattr(p, "waiver_status", "") or "") or None
        row["birth_city"] = str(getattr(ident, "birth_city", "") or "") if ident else None
        jersey = getattr(p, "jersey_number", None) or getattr(p, "number", None) or getattr(ident, "number", None)
        if jersey is not None:
            row["jersey_number"] = int(jersey) if str(jersey).isdigit() else jersey
    except Exception:
        pass
    # Potential (0–100 display) from real engine fields — never invent.
    try:
        ratings = getattr(p, "ratings", None) or {}
        pot_raw = None
        if isinstance(ratings, dict):
            pot_raw = ratings.get("dev_potential")
        if pot_raw is None:
            pot_raw = getattr(p, "potential", None)
        if pot_raw is not None:
            pot_f = float(pot_raw)
            pot99 = int(round(pot_f * 99.0)) if pot_f <= 1.5 else int(round(pot_f))
            row["potential"] = pot99
            row["potential_score"] = pot99
            row["dev_potential"] = pot99
    except Exception:
        pass
    # Fatigue from health state when present.
    try:
        health = getattr(p, "health", None)
        fat = getattr(health, "fatigue", None) if health is not None else getattr(p, "fatigue", None)
        if fat is not None:
            fat_f = float(fat)
            row["fatigue"] = int(round(fat_f * 100.0)) if fat_f <= 1.5 else int(round(fat_f))
    except Exception:
        pass
    # Full contract summary for dossier Contract tab.
    try:
        from services.contract_economy import (
            get_contract_display_summary,
            is_waiver_exempt,
            normalize_contract_payload,
        )

        season_y = int(getattr(session, "season_calendar_year", 2025) or 2025) if session is not None else None
        csum = get_contract_display_summary(p, season_y)
        cnorm = normalize_contract_payload(p)
        clause = csum.get("clause") if isinstance(csum.get("clause"), dict) else {}
        clause_label = "None"
        if clause.get("nmc"):
            clause_label = "NMC"
        elif clause.get("ntc"):
            clause_label = "NTC"
        elif clause.get("clause_type") and str(clause.get("clause_type")) != "None":
            clause_label = str(clause.get("clause_type"))
        expiry_year = cnorm.get("expiry_year")
        if expiry_year is None and csum.get("years_remaining") and season_y is not None:
            try:
                yr = int(csum.get("years_remaining") or 0)
                expiry_year = int(season_y) + yr if yr > 0 else None
            except Exception:
                expiry_year = None
        try:
            if expiry_year is not None and int(expiry_year) <= 0:
                expiry_year = None
        except Exception:
            expiry_year = None
        years_remaining = int(csum.get("years_remaining") or 0)
        row["contract"] = {
            "salary": round(float(csum.get("nhl_salary_m") or csum.get("aav_m") or row["contract"]["salary"] or 0), 3),
            "cap_hit": round(float(csum.get("cap_hit_m") or csum.get("aav_m") or row["contract"]["cap_hit"] or 0), 3),
            "aav": round(float(csum.get("aav_m") or 0), 3),
            "term": years_remaining,
            "years_remaining": years_remaining,
            "expiry_year": expiry_year,
            "expiry": str(expiry_year) if expiry_year else None,
            "type": str(csum.get("type") or "") or None,
            "contract_type": str(csum.get("type") or "") or None,
            "clause": clause_label,
            "two_way": bool(csum.get("two_way")),
            "is_entry_level": bool(csum.get("is_entry_level")),
            "signing_bonus_m": float(csum.get("signing_bonus_m") or 0),
            "performance_bonus_m": float(csum.get("performance_bonus_m") or 0),
            "minor_salary_m": float(csum.get("minor_salary_m") or 0),
            "start_year": cnorm.get("start_year") or cnorm.get("signed_year"),
            "rights_status": str(cnorm.get("rights_status") or getattr(p, "rights_status", "") or "") or None,
        }
        try:
            from app.sim_engine.franchise.player_agent_engine import agent_public_view  # noqa: WPS433

            row["contract"]["agent"] = agent_public_view(p, session)
        except Exception:
            pass
        try:
            row["waiver_exempt"] = bool(is_waiver_exempt(p, _team, getattr(getattr(session, "sim", None), "league", None) if session else None))
        except Exception:
            row["waiver_exempt"] = None
    except Exception:
        pass
    # Development snapshots — only real ledger / history entries.
    try:
        hist_raw = getattr(p, "development_history", None)
        hist_out: List[Dict[str, Any]] = []
        if isinstance(hist_raw, list):
            for entry in hist_raw[-12:]:
                if not isinstance(entry, dict):
                    continue
                ob = entry.get("ovr_before")
                oa = entry.get("ovr_after")
                try:
                    ob_f = float(ob) if ob is not None else None
                    oa_f = float(oa) if oa is not None else None
                    if ob_f is not None and ob_f <= 1.5:
                        ob_f = ob_f * 99.0
                    if oa_f is not None and oa_f <= 1.5:
                        oa_f = oa_f * 99.0
                except Exception:
                    ob_f, oa_f = None, None
                hist_out.append({
                    "season": entry.get("season"),
                    "ovr_before": int(round(ob_f)) if ob_f is not None else None,
                    "ovr_after": int(round(oa_f)) if oa_f is not None else None,
                    "delta": round(oa_f - ob_f, 1) if ob_f is not None and oa_f is not None else None,
                    "source_path": entry.get("source_path"),
                    "development_applied": bool(entry.get("development_applied")),
                })
        ledger = getattr(p, "development_ledger", None)
        if isinstance(ledger, dict) and ledger.get("season") is not None:
            row["development_ledger"] = {
                "season": ledger.get("season"),
                "ovr_before": ledger.get("ovr_before"),
                "ovr_after": ledger.get("ovr_after"),
                "source_path": ledger.get("source_path"),
                "development_applied": bool(ledger.get("development_applied")),
            }
        if hist_out:
            row["development_history"] = hist_out
        arch = str(getattr(p, "_dev_archetype", "") or getattr(p, "dev_type", "") or "")
        if arch:
            row["development_type"] = arch
        profile = getattr(p, "development_profile", None)
        if isinstance(profile, dict):
            row["development_profile"] = {
                "expected_ceiling": profile.get("expected_ceiling"),
                "maximum_ceiling": profile.get("maximum_ceiling"),
            }
    except Exception:
        pass
    # Awards / career stats when stored on the player.
    try:
        awards = getattr(p, "career_awards", None) or getattr(p, "awards_won", None)
        if isinstance(awards, list) and awards:
            row["career_awards"] = [
                (a if isinstance(a, (str, int, float)) else dict(a) if isinstance(a, dict) else str(a))
                for a in awards[:24]
            ]
        career = getattr(p, "career_stats", None)
        if isinstance(career, dict) and career:
            seasons = career.get("seasons") or career.get("by_season") or career.get("history")
            if isinstance(seasons, list) and seasons:
                row["career_seasons"] = [dict(s) for s in seasons if isinstance(s, dict)][-16:]
            totals = career.get("totals") or career.get("career") or career.get("nhl")
            if isinstance(totals, dict):
                row["career_totals"] = dict(totals)
    except Exception:
        pass
    if session is not None:
        try:
            nhl_compact = _compact_season_stats_for_player(session, pid)
            roster_l = str(roster_kind or "").lower()
            is_ahl = roster_l == "ahl"
            is_echl = roster_l == "echl"
            compact: Optional[Dict[str, Any]] = None

            if is_ahl:
                ahl_compact = _ahl_light_season_stats(
                    p, session=session, is_goalie=(pos_str == "G"), team=_team
                )
                # Preserve any true NHL games as a separate career split, then
                # show the AHL affiliate line as the live season summary.
                if nhl_compact and int(nhl_compact.get("gp", 0) or 0) > 0:
                    nhl_line = dict(nhl_compact)
                    nhl_line["league"] = "NHL"
                    nhl_line.pop("is_ahl_synthetic", None)
                    _merge_current_season_into_career_seasons(
                        row, nhl_line, session=session, team=_team
                    )
                if ahl_compact:
                    compact = ahl_compact
                    affiliate = str(ahl_compact.get("team_name") or ahl_compact.get("team") or "")
                    row["league"] = "AHL"
                    row["league_code"] = "AHL"
                    if affiliate:
                        row["team_name"] = affiliate
                        row["teamName"] = affiliate
                        row["affiliate_team_name"] = affiliate
                elif nhl_compact and int(nhl_compact.get("gp", 0) or 0) > 0:
                    compact = nhl_compact
            elif is_echl:
                row["league"] = "ECHL"
                row["league_code"] = "ECHL"
                if nhl_compact and int(nhl_compact.get("gp", 0) or 0) > 0:
                    compact = nhl_compact
            else:
                if nhl_compact and int(nhl_compact.get("gp", 0) or 0) > 0:
                    compact = nhl_compact

            if compact:
                row["season_stats"] = compact
                _merge_current_season_into_career_seasons(row, compact, session=session, team=_team)
        except Exception:
            pass
        try:
            start = getattr(p, "season_start_ovr", None)
            if start is None:
                start = getattr(p, "_season_start_ovr", None)
            cur = float(row.get("ovr") or row.get("effective_ovr") or row.get("base_ovr") or 0)
            if start is not None:
                start_f = float(start)
                if start_f <= 1.5:
                    start_f *= 99.0
                row["season_start_ovr"] = int(round(start_f))
                row["growth_delta"] = round(cur - start_f, 1)
                row["overall_delta"] = row["growth_delta"]
            else:
                accum = getattr(p, "_in_season_ovr_delta_accum", None)
                if accum is not None:
                    row["growth_delta"] = round(float(accum), 1)
                    row["overall_delta"] = row["growth_delta"]
        except Exception:
            pass
        # Resolve draft team display name when possible.
        try:
            tid = str(row.get("drafted_by_team_id") or row.get("draft_team_id") or "")
            if tid and session.team_by_id:
                tm = session.team_by_id.get(tid)
                if tm is None:
                    tid_l = tid.lower()
                    for _k, _tm in session.team_by_id.items():
                        abbrs = {
                            str(getattr(_tm, "id", "") or "").lower(),
                            str(getattr(_tm, "abbr", "") or "").lower(),
                            str(getattr(_tm, "abbreviation", "") or "").lower(),
                            str(_k).lower(),
                        }
                        if tid_l in abbrs:
                            tm = _tm
                            break
                if tm is not None:
                    row["drafted_by_team_name"] = _display_team(tm)
        except Exception:
            pass
    return row


def _ahl_light_season_stats(
    p: Any,
    *,
    session: Optional[FranchiseSession] = None,
    is_goalie: bool = False,
    team: Optional[Any] = None,
) -> Dict[str, Any]:
    """Light season line for AHL affiliate players.

    AHL games are not simulated player-by-player like the NHL schedule, so
    `session.player_season_stats` never has an entry for them. Reuse the same
    prospect-league statistical model that drives junior/NCAA/European stat
    lines (deterministic per player, calendar-advanced) under an "AHL"
    scoring profile so affiliate players show real basic totals instead of
    a blank stat line.
    """
    try:
        from app.sim_engine.generation.prospect_league_scoring import prospect_stats_for_api
    except Exception:
        return {}

    season_year = int(getattr(session, "season_calendar_year", 0) or 0) if session is not None else 0
    calendar_iso = None
    if session is not None:
        try:
            calendar_iso = _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
        except Exception:
            calendar_iso = None
    try:
        stats = prospect_stats_for_api(
            p,
            "AHL",
            calendar_iso=calendar_iso,
            season_year=season_year or None,
        )
    except Exception:
        return {}

    gp = int(stats.get("gp") or stats.get("games_played") or 0)
    if gp <= 0:
        return {}

    affiliate = _ahl_affiliate_display_name(team)
    out: Dict[str, Any] = {
        "gp": gp,
        "pim": int(stats.get("pim") or 0),
        "ppg": stats.get("ppg") or stats.get("points_per_game") or 0.0,
        "league": "AHL",
        "league_code": "AHL",
        "team": affiliate,
        "team_name": affiliate,
        "is_ahl_synthetic": True,
    }
    if is_goalie:
        out.update(
            {
                "wins": int(stats.get("wins") or 0),
                "losses": int(stats.get("losses") or 0),
                "otl": int(stats.get("ot_losses") or 0),
                "svPct": stats.get("save_pct"),
                "sv_pct": stats.get("save_pct"),
                "gaa": stats.get("gaa"),
                "shutouts": int(stats.get("shutouts") or 0),
            }
        )
    else:
        goals = int(stats.get("goals") or 0)
        assists = int(stats.get("assists") or 0)
        analytics = stats.get("analytics") if isinstance(stats.get("analytics"), dict) else {}
        pm = stats.get("plus_minus")
        if pm is None:
            pm = analytics.get("plus_minus")
        war = stats.get("war")
        out.update(
            {
                "g": goals,
                "a": assists,
                "pts": int(stats.get("points") or (goals + assists)),
                "plusMinus": int(pm) if pm is not None else 0,
                "plus_minus": int(pm) if pm is not None else 0,
            }
        )
        if war is not None:
            try:
                out["war"] = round(float(war), 2)
            except Exception:
                pass
    return out


def _compact_season_stats_for_player(session: FranchiseSession, player_id: str) -> Dict[str, Any]:
    """UI-facing season line for roster rows (skater + goalie fields)."""
    if not player_id:
        return {}
    st = dict((getattr(session, "player_season_stats", None) or {}).get(str(player_id)) or {})
    if not st:
        return {}
    gp = int(st.get("gp", 0) or 0)
    g = int(st.get("g", 0) or 0)
    a = int(st.get("a", 0) or 0)
    pts = int(st.get("pts", 0) or (g + a))
    toi_sec = int(st.get("toi_sec", 0) or 0)
    ev_toi_sec = int(st.get("ev_toi_sec") or st.get("even_strength_toi_sec") or 0)
    pp_toi_sec = int(st.get("pp_toi_sec") or st.get("power_play_toi_sec") or 0)
    pk_toi_sec = int(st.get("pk_toi_sec") or st.get("penalty_kill_toi_sec") or 0)
    if ev_toi_sec <= 0 and toi_sec > 0:
        ev_toi_sec = max(0, toi_sec - pp_toi_sec - pk_toi_sec)
    # Goalies: light bulk historically omitted toi_sec while accruing GA → absurd GAA.
    pos_u = str(st.get("position") or st.get("pos") or "").upper()
    if pos_u == "G" and gp > 0 and toi_sec < gp * 1800:
        toi_sec = gp * 3600
    sa = int(st.get("sa", st.get("shots_against", 0)) or 0)
    saves = int(st.get("saves", st.get("sv", 0)) or 0)
    ga = int(st.get("ga", 0) or 0)
    sv_pct = None
    if sa > 0:
        sv_pct = round(saves / float(sa), 3)
    elif st.get("sv_pct") is not None or st.get("save_pct") is not None:
        try:
            raw = float(st.get("sv_pct", st.get("save_pct")))
            sv_pct = round(raw / 100.0, 3) if raw > 1.5 else round(raw, 3)
        except Exception:
            sv_pct = None
    gaa = None
    if toi_sec > 0 and ga >= 0:
        gaa = round(ga * 3600.0 / float(toi_sec), 2)
    elif st.get("gaa") is not None:
        try:
            gaa = round(float(st.get("gaa")), 2)
        except Exception:
            gaa = None
    out: Dict[str, Any] = {
        "gp": gp,
        "g": g,
        "a": a,
        "pts": pts,
        "sog": int(st.get("sog", st.get("shots", 0)) or 0),
        "shots": int(st.get("sog", st.get("shots", 0)) or 0),
        "pim": int(st.get("pim", 0) or 0),
        "hit": int(st.get("hit", st.get("hits", 0)) or 0),
        "hits": int(st.get("hit", st.get("hits", 0)) or 0),
        "blk": int(st.get("blk", st.get("blocks", 0)) or 0),
        "blocks": int(st.get("blk", st.get("blocks", 0)) or 0),
        "toi": round((toi_sec / max(1, gp)) / 60.0, 1) if gp > 0 else 0.0,
        "toi_sec": toi_sec,
        "ev_toi_sec": ev_toi_sec,
        "pp_toi_sec": pp_toi_sec,
        "pk_toi_sec": pk_toi_sec,
        "even_strength_toi_sec": ev_toi_sec,
        "power_play_toi_sec": pp_toi_sec,
        "penalty_kill_toi_sec": pk_toi_sec,
        "plus_minus": int(
            st.get("plus_minus", st.get("pm"))
            if st.get("plus_minus", st.get("pm")) is not None
            else round(float(st.get("gf_on") or 0) - float(st.get("ga_on") or 0))
        ),
        "plusMinus": int(
            st.get("plus_minus", st.get("pm"))
            if st.get("plus_minus", st.get("pm")) is not None
            else round(float(st.get("gf_on") or 0) - float(st.get("ga_on") or 0))
        ),
        "ga": ga,
        "w": int(st.get("w", st.get("wins", 0)) or 0),
        "wins": int(st.get("w", st.get("wins", 0)) or 0),
        "l": int(st.get("l", st.get("losses", 0)) or 0),
        "losses": int(st.get("l", st.get("losses", 0)) or 0),
        "otl": int(st.get("otl", 0) or 0),
        "saves": saves,
        "shots_against": sa,
        "sa": sa,
        "ppg": round(pts / float(gp), 2) if gp > 0 else 0.0,
    }
    if sv_pct is not None:
        out["svPct"] = sv_pct
        out["sv_pct"] = sv_pct
        out["save_pct"] = sv_pct
    if gaa is not None:
        out["gaa"] = gaa
    # Possession / WAR for dossier Stats tab (light bulk now writes cf/xgf).
    try:
        cf = float(st.get("cf") or 0)
        ca = float(st.get("ca") or 0)
        xgf = float(st.get("xgf") or 0)
        xga = float(st.get("xga") or 0)
        if cf + ca > 0:
            out["cf"] = round(cf, 1)
            out["ca"] = round(ca, 1)
            out["cf_pct"] = round(cf / (cf + ca), 4)
            out["cfPct"] = round(100.0 * cf / (cf + ca), 1)
        if xgf + xga > 0:
            out["xgf"] = round(xgf, 2)
            out["xga"] = round(xga, 2)
            out["xgf_pct"] = round(xgf / (xgf + xga), 4)
            out["xgfPct"] = round(100.0 * xgf / (xgf + xga), 1)
        if gp > 0:
            from app.sim_engine.generation.player_analytics import enrich_player_row

            enriched = enrich_player_row(st)
            war_raw = enriched.get("war")
            if war_raw is not None:
                out["war"] = round(float(war_raw), 2)
                out["war_valid"] = bool(enriched.get("war_valid"))
    except Exception:
        pass
    return out


def _merge_current_season_into_career_seasons(
    row: Dict[str, Any],
    compact: Dict[str, Any],
    *,
    session: Optional[FranchiseSession] = None,
    team: Optional[Any] = None,
) -> None:
    """Fold this session's in-progress/just-archived season line into
    `row["career_seasons"]` so the current franchise-universe season shows up
    alongside NHL-import career history in the UI, instead of only living in
    the ephemeral `season_stats` field that resets every season rollover.
    """
    gp_now = int(compact.get("gp", 0) or 0)
    if gp_now <= 0:
        return
    try:
        cal_year = int(getattr(session, "season_calendar_year", 0) or 0)
    except Exception:
        cal_year = 0
    season_label = f"{cal_year}-{str(cal_year + 1)[-2:]}" if cal_year else ""
    league = str(compact.get("league") or compact.get("league_code") or "NHL").strip().upper() or "NHL"
    if league == "AHL":
        team_name = str(
            compact.get("team_name")
            or compact.get("team")
            or _ahl_affiliate_display_name(team)
            or ""
        )
    elif league == "ECHL":
        team_name = str(compact.get("team_name") or compact.get("team") or "ECHL Affiliate")
    else:
        team_name = _display_team(team) if team is not None else str(
            row.get("teamName") or row.get("team_name") or ""
        )
        league = "NHL"
    is_goalie_row = str(row.get("position") or "").upper() == "G"

    entry: Dict[str, Any] = {
        "season": season_label or "Current",
        "team": team_name or "—",
        "team_name": team_name or "—",
        "league": league,
        "gp": gp_now,
        "is_current_season": True,
    }
    if is_goalie_row:
        entry.update(
            {
                "wins": compact.get("wins"),
                "losses": compact.get("losses"),
                "otl": compact.get("otl"),
                "sv_pct": compact.get("svPct"),
                "gaa": compact.get("gaa"),
                "shutouts": compact.get("shutouts"),
            }
        )
    else:
        entry.update(
            {
                "g": compact.get("g"),
                "a": compact.get("a"),
                "pts": compact.get("pts"),
                "plus_minus": compact.get("plusMinus"),
                "pim": compact.get("pim"),
                "war": compact.get("war"),
            }
        )

    seasons = [dict(s) for s in (row.get("career_seasons") or []) if isinstance(s, dict)]
    # Dedup by season + league + team so NHL split and AHL split can coexist.
    dedup_key = (str(entry["season"]), str(entry["league"]), str(entry["team"]))
    seasons = [
        s
        for s in seasons
        if (
            str(s.get("season") or ""),
            str(s.get("league") or "NHL").upper(),
            str(s.get("team") or s.get("team_name") or ""),
        )
        != dedup_key
    ]
    seasons.append(entry)
    row["career_seasons"] = seasons[-16:]


def _rows_from_players_list(
    players: Any,
    *,
    include_ratings: bool = False,
    session: Optional[FranchiseSession] = None,
    team: Optional[Any] = None,
    roster_kind: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in players or []:
        if getattr(p, "retired", False):
            continue
        rows.append(
            _serialize_player_row(
                p, include_ratings=include_ratings, session=session, _team=team, roster_kind=roster_kind
            )
        )
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
                    roster_kind="ahl",
                ),
                "echl": _rows_from_players_list(
                    getattr(t, "echl_roster", None),
                    include_ratings=is_user,
                    session=franchise_session,
                    team=t,
                    roster_kind="echl",
                ),
                "prospects": _rows_from_players_list(
                    getattr(t, "prospect_pool", None) or getattr(t, "prospects", None),
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


_DRAFT_LEAGUE_DISPLAY: Dict[str, str] = {
    "CHL_OHL": "OHL",
    "CHL_WHL": "WHL",
    "CHL_QMJHL": "QMJHL",
    "USHL": "USHL",
    "NCAA": "NCAA",
    "EU_J_SHL": "Sweden J20",
    "EU_J_LIIGA": "Finland U20",
    "EU_J_DEL": "Germany DNL",
    "EU_J_SWISS": "Swiss Elite Jr",
    "EU_J_CZ": "Czechia U20",
    "EU_J_SK": "Slovakia U20",
    "EU_J_KHL_JR": "MHL",
    "EU_J_NOR": "Norway Jr",
    "EU_J_DEN": "Denmark U20",
    "EU_J_AUT": "Austria Jr",
}


def _draft_league_display(code: str, name: str = "") -> str:
    c = str(code or "").strip().upper()
    if c in _DRAFT_LEAGUE_DISPLAY:
        return _DRAFT_LEAGUE_DISPLAY[c]
    return c.replace("_", " ") if c else (str(name or "Junior")[:18])


# Class-relative letter tiers (percentile within draft class), top-pick floor at A-.
_DRAFT_TIER_BANDS: List[Tuple[float, str]] = [
    (0.985, "A+"), (0.96, "A"), (0.92, "A-"), (0.84, "B+"),
    (0.70, "B"), (0.52, "B-"), (0.32, "C+"), (0.0, "C"),
]

GENERATIONAL_GOALIE_SCORE_PCT = 0.985  # goalie must beat ~98.5% of class to escape penalty
GOALIE_RANK_PENALTY = 5.0  # score points removed from non-generational goalies


def _draft_potential99(p: Any, ovr99: float) -> float:
    """Resolve prospect expected ceiling on the 0–99 display scale via development profile."""
    try:
        from app.sim_engine.entities.player import display_rating, normalize_rating
        from app.sim_engine.progression.development import resolve_development_profile

        profile = resolve_development_profile(
            p,
            {
                "current_ovr_hint": normalize_rating(ovr99),
            },
        )
        return float(display_rating(profile.get("expected_ceiling")))
    except Exception:
        pass
    # Controlled fallback — no large tier-based OVR bumps.
    r = getattr(p, "ratings", None) or {}
    for k in ("dev_potential", "potential", "dev_ceiling"):
        try:
            v = float(r.get(k, 0) or 0)
            if v > 0:
                from app.sim_engine.entities.player import display_rating, normalize_rating

                return float(display_rating(normalize_rating(v)))
        except Exception:
            pass
    try:
        from app.sim_engine.entities.player import display_rating, normalize_rating

        base = normalize_rating(ovr99)
        return float(display_rating(min(0.92, base + 0.08)))
    except Exception:
        return float(min(92.0, float(ovr99) + 8.0))


def _draft_stock_reason(row: Dict[str, Any], signed_delta: int, *, goalie_penalized: bool) -> str:
    mode = str(row.get("stock_mode") or "")
    weekly = str(row.get("weekly_stock_reason") or "").strip()
    # Prefer compact weekly heat reasons when display is heat-driven.
    if mode == "weekly_heat" and weekly and weekly not in (
        "Quiet week on the scoresheet.",
        "Holding steady on the public board.",
        "Scouts are moving him up the board.",
        "Draft buzz cooling after recent tape.",
    ):
        return weekly[:72]
    if mode == "rank_change":
        if signed_delta >= 5:
            return f"Board jump +{signed_delta} spots"
        if signed_delta >= 1:
            return f"Up {signed_delta} on the public board"
        if signed_delta <= -5:
            return f"Board drop {signed_delta} spots"
        if signed_delta <= -1:
            return f"Down {abs(signed_delta)} on the public board"
    if goalie_penalized and signed_delta < 0:
        return "Goalie value adjustment"
    week_gp = int(row.get("week_gp") or 0)
    week_pts = int(row.get("week_points") or 0)
    if week_gp > 0 and abs(signed_delta) >= 1:
        return f"{week_pts}P / {week_gp} GP"
    if abs(signed_delta) <= 0:
        return "No games this week." if week_gp <= 0 else f"{week_pts}P / {week_gp} GP · holding"
    return "Weekly board movement"


_FRANCHISE_TIER_META: Dict[str, Tuple[str, int]] = {
    "franchise_swing": ("Franchise Swing", 1),
    "core_upside": ("Core Upside", 2),
    "debate_room": ("Debate Room", 3),
    "safe_depth": ("Safe Depth", 4),
    "mystery_box": ("Mystery Box", 5),
    "late_flyer": ("Late Flyer", 6),
    "unclassified": ("Unclassified", 999),
}

_TALENT_GRADE_SCORE: Dict[str, int] = {
    "A+": 6, "A": 5, "A-": 4, "B+": 3, "B": 2, "B-": 1, "C+": 0, "C": -1,
}


def _franchise_tier_payload(
    key: str,
    reason: str,
    confidence: int,
) -> Dict[str, Any]:
    label, order = _FRANCHISE_TIER_META.get(key, _FRANCHISE_TIER_META["unclassified"])
    return {
        "key": key,
        "label": label,
        "order": order,
        "confidence": int(max(0, min(100, confidence))),
        "reason": str(reason or ""),
        "source": "backend",
    }


def _compute_franchise_tier_object(row: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Classify prospect franchise tier from backend scouting/draft signals only."""
    pot = float(row.get("potential_score") or row.get("true_ovr") or 0)
    conf = float(row.get("scouting_confidence") or 50)
    talent = str(row.get("talent_grade") or row.get("scout_tier") or "C")
    talent_score = _TALENT_GRADE_SCORE.get(talent, 0)
    risk = str(row.get("risk") or "Medium")
    is_gem = bool(row.get("is_gem"))
    is_bust = bool(row.get("is_bust_risk") or row.get("boom_bust"))
    char = bool(row.get("character_concerns"))
    prod_adj = float(row.get("production_adjusted_score") or 0)

    if rank > 64 or (rank > 48 and pot < 70):
        return _franchise_tier_payload(
            "late_flyer",
            f"Outside early-round band at #{rank} with limited ceiling signal",
            int(min(85, conf)),
        )

    if conf < 58 and (is_gem or is_bust or char):
        return _franchise_tier_payload(
            "mystery_box",
            "High variance with incomplete scouting picture",
            max(40, int(100 - conf)),
        )

    if rank <= 5 and pot >= 78 and talent_score >= 4 and not is_bust:
        return _franchise_tier_payload(
            "franchise_swing",
            f"Top-{rank} profile with {talent} tools and {pot:.0f} potential",
            int(min(95, conf)),
        )
    if rank <= 3 and talent_score >= 3 and pot >= 74:
        return _franchise_tier_payload(
            "franchise_swing",
            f"Elite ceiling at #{rank} with translatable traits",
            int(min(90, conf)),
        )

    if rank <= 18 and pot >= 72 and talent_score >= 2 and not (is_bust and char):
        return _franchise_tier_payload(
            "core_upside",
            f"First-round upside at #{rank} — {talent} grade tools",
            int(min(88, conf)),
        )

    if (is_bust or char or risk == "High") and rank <= 32:
        return _franchise_tier_payload(
            "debate_room",
            "Scouts divided — risk flags competing with talent grade",
            70,
        )
    if conf < 65 and rank <= 25 and prod_adj < 0.35:
        return _franchise_tier_payload(
            "debate_room",
            "Needs more viewings before consensus tier",
            55,
        )

    if rank <= 45 and talent_score <= 2 and pot < 78 and not is_gem:
        return _franchise_tier_payload(
            "safe_depth",
            "Projectable floor with moderate ceiling",
            int(min(80, conf)),
        )
    if rank <= 64:
        return _franchise_tier_payload(
            "safe_depth",
            f"Organizational depth prospect at #{rank}",
            int(min(75, conf)),
        )

    return _franchise_tier_payload("unclassified", "Backend tier data unavailable", 0)


def _compute_draft_stock_object(
    row: Dict[str, Any],
    rank: int,
    *,
    key_in_prev: bool,
) -> Dict[str, Any]:
    """Normalized draft stock payload for UI."""
    trend = str(row.get("trend") or "SAME").upper()
    signed_delta = int(row.get("stock_delta") or row.get("stock_change") or 0)
    prev_rank_raw = row.get("previous_rank") if key_in_prev else row.get("rank_prev")
    prev_rank: Optional[int]
    try:
        prev_rank = int(prev_rank_raw) if prev_rank_raw is not None and key_in_prev else None
    except (TypeError, ValueError):
        prev_rank = None

    direction_map = {"UP": "UP", "DOWN": "DOWN", "SAME": "STABLE", "NEW": "NEW"}
    direction = direction_map.get(trend, "UNKNOWN")
    if not key_in_prev and trend != "NEW":
        direction = "NEW"
    label_defaults = {
        "UP": "Rising",
        "DOWN": "Falling",
        "STABLE": "Holding",
        "NEW": "New Entry",
        "UNKNOWN": "No Movement Data",
    }
    label = str(row.get("stock_label") or label_defaults.get(direction, "No Movement Data"))
    week_gp = int(row.get("week_gp") or 0)
    # Sample confidence — not ambient scouting %
    if week_gp <= 0 and str(row.get("stock_mode") or "") != "rank_change":
        stock_conf = 35
    elif week_gp <= 2:
        stock_conf = 48
    elif week_gp <= 4:
        stock_conf = 62
    else:
        stock_conf = 78
    if str(row.get("stock_mode") or "") == "rank_change":
        stock_conf = max(stock_conf, 70)
    updated = str(row.get("last_prospect_stat_update_date") or "")
    mode = str(row.get("stock_mode") or ("rank_change" if key_in_prev and abs(signed_delta) >= 1 and int(row.get("rank_delta") or 0) != 0 else "weekly_heat"))
    unit = str(row.get("stock_unit") or ("rank" if mode == "rank_change" else "heat"))

    return {
        "direction": direction,
        "delta_rank": signed_delta,  # display delta (rank spots OR heat — see stock_unit)
        "display_delta": signed_delta,
        "rank_delta": int(row.get("rank_delta") or 0),
        "stock_heat": int(row.get("stock_heat") or row.get("weekly_stock_delta") or 0),
        "stock_mode": mode,
        "stock_unit": unit,
        "previous_rank": prev_rank,
        "current_rank": rank,
        "label": label,
        "reason": str(row.get("stock_reason") or ""),
        "confidence": int(min(95, max(30, stock_conf))),
        "updated_at": updated,
        "source": "backend",
    }


def _effective_stock_direction(stock: Dict[str, Any]) -> str:
    """Classify movers using delta sign so labels and lists stay aligned."""
    direction = str(stock.get("direction") or "UNKNOWN").upper()
    try:
        delta = int(stock.get("delta_rank") or stock.get("display_delta") or 0)
    except (TypeError, ValueError):
        delta = 0
    if direction == "NEW":
        return "NEW"
    if delta > 0:
        return "UP"
    if delta < 0:
        return "DOWN"
    if direction in ("UP", "DOWN", "STABLE"):
        return direction
    return "STABLE"


def _build_stock_market_summary(
    entries: List[Dict[str, Any]],
    *,
    generated_at: str = "",
) -> Dict[str, Any]:
    risers: List[Dict[str, Any]] = []
    fallers: List[Dict[str, Any]] = []
    stable: List[Dict[str, Any]] = []
    new_entries: List[Dict[str, Any]] = []
    no_movement = 0

    for e in entries:
        stock = e.get("draft_stock") if isinstance(e.get("draft_stock"), dict) else {}
        direction = _effective_stock_direction(stock)
        item = {
            "key": str(e.get("key") or ""),
            "name": str(e.get("name") or ""),
            "rank": int(e.get("rank") or 0),
            "delta_rank": int(stock.get("delta_rank") or 0),
            "label": str(stock.get("label") or ""),
            "position": str(e.get("position") or ""),
        }
        if direction == "UP":
            risers.append(item)
        elif direction == "DOWN":
            fallers.append(item)
        elif direction == "NEW":
            new_entries.append(item)
        elif direction == "STABLE":
            stable.append(item)
        else:
            no_movement += 1

    # Prioritize top-of-board movers; within a rank band, sort by magnitude.
    risers.sort(
        key=lambda x: (
            int(x.get("rank") or 999),
            -abs(int(x.get("delta_rank") or 0)),
        )
    )
    fallers.sort(
        key=lambda x: (
            int(x.get("rank") or 999),
            abs(int(x.get("delta_rank") or 0)),
        )
    )

    return {
        "risers": risers[:12],
        "fallers": fallers[:12],
        "stable": stable[:8],
        "new_entries": new_entries[:8],
        "no_movement_count": no_movement,
        "generated_at": generated_at,
        "source": "backend",
    }


def _build_tier_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {k: 0 for k in _FRANCHISE_TIER_META}
    for e in entries:
        ft = e.get("franchise_tier") if isinstance(e.get("franchise_tier"), dict) else {}
        key = str(ft.get("key") or "unclassified")
        if key not in counts:
            key = "unclassified"
        counts[key] += 1
    out: Dict[str, Any] = {k: counts[k] for k in counts}
    out["source"] = "backend"
    return out


def _draft_country_iso(raw: str) -> str:
    """Map prospect nationality text/codes to ISO 3166-1 alpha-2 for flag APIs."""
    s = str(raw or "").strip()
    if not s:
        return ""
    if len(s) == 2 and s.isalpha():
        return s.upper()
    key = s.upper()
    mapping = {
        "CAN": "CA",
        "CANADA": "CA",
        "USA": "US",
        "US": "US",
        "UNITED STATES": "US",
        "UNITED STATES OF AMERICA": "US",
        "SWE": "SE",
        "SWEDEN": "SE",
        "FIN": "FI",
        "FINLAND": "FI",
        "RUS": "RU",
        "RUSSIA": "RU",
        "RUSSIAN FEDERATION": "RU",
        "CZE": "CZ",
        "CZECHIA": "CZ",
        "CZECH REPUBLIC": "CZ",
        "SVK": "SK",
        "SLOVAKIA": "SK",
        "SUI": "CH",
        "SWITZERLAND": "CH",
        "GER": "DE",
        "GERMANY": "DE",
        "NOR": "NO",
        "NORWAY": "NO",
        "DEN": "DK",
        "DENMARK": "DK",
        "LAT": "LV",
        "LATVIA": "LV",
        "AUT": "AT",
        "AUSTRIA": "AT",
        "KAZ": "KZ",
        "KAZAKHSTAN": "KZ",
        "BLR": "BY",
        "BELARUS": "BY",
        "UKR": "UA",
        "UKRAINE": "UA",
        "POL": "PL",
        "POLAND": "PL",
        "FRA": "FR",
        "FRANCE": "FR",
        "GBR": "GB",
        "UNITED KINGDOM": "GB",
        "GREAT BRITAIN": "GB",
        "JPN": "JP",
        "JAPAN": "JP",
    }
    if key in mapping:
        return mapping[key]
    title = s.title()
    title_map = {
        "Canada": "CA",
        "United States": "US",
        "United States Of America": "US",
        "Sweden": "SE",
        "Finland": "FI",
        "Russia": "RU",
        "Russian Federation": "RU",
        "Czechia": "CZ",
        "Czech Republic": "CZ",
        "Slovakia": "SK",
        "Switzerland": "CH",
        "Germany": "DE",
        "Norway": "NO",
        "Denmark": "DK",
        "Latvia": "LV",
        "Austria": "AT",
        "Kazakhstan": "KZ",
        "Belarus": "BY",
        "Ukraine": "UA",
        "Poland": "PL",
        "France": "FR",
        "United Kingdom": "GB",
        "Great Britain": "GB",
        "Japan": "JP",
    }
    return title_map.get(title, "")


def build_draft_class_rankings(session: FranchiseSession, sim: Any) -> Dict[str, Any]:
    """Draft board built from real development-league players: stats, nationality,
    league difficulty, goalie positional realism, and stock movement."""
    league = getattr(sim, "league", None)
    if league is None:
        return {"entries": [], "subtitle": "", "total": 0}
    # One-time repair: older franchises shaped #1 overalls into the low-50s while
    # "nhl_floor" depth sat higher. Bring pipeline stars up to realistic draft OVRs.
    try:
        if not getattr(session, "_draft_pipeline_ovr_repaired", False):
            from app.sim_engine.league_hierarchy_bootstrap import repair_undervalued_draft_pipeline_stars

            repair_undervalued_draft_pipeline_stars(
                league,
                getattr(sim, "rng", None),
            )
            setattr(session, "_draft_pipeline_ovr_repaired", True)
    except Exception:
        pass
    try:
        if not getattr(session, "_draft_age_reanchored", False):
            from app.sim_engine.league_hierarchy_bootstrap import (
                reanchor_generated_junior_dobs,
                set_spawn_as_of_year,
            )

            anchor = int(getattr(session, "season_calendar_year", 0) or 0)
            if anchor < 2000:
                anchor = int(session_age_as_of(session)[0])
            set_spawn_as_of_year(anchor)
            reanchor_generated_junior_dobs(league, anchor)
            setattr(session, "_draft_age_reanchored", True)
    except Exception:
        pass
    try:
        from app.sim_engine.generation.prospect_league_scoring import (
            prospect_stats_for_api,
            normalize_league_leader_board,
        )
        scoring_available = True
    except Exception:
        scoring_available = False

    rng = getattr(sim, "rng", None)
    calendar_iso = _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    stat_keys = (
        "gp", "games_played", "goals", "assists", "points", "ppg", "points_per_game",
        "wins", "losses", "ot_losses", "save_pct", "gaa", "shutouts", "pim",
        "shots", "plus_minus", "primary_points", "shooting_pct", "shot_rate",
        "xgf_pct", "cf_pct", "ff_pct", "war", "offensive_war", "defensive_war",
        "defensive_impact", "quality_of_competition", "quality_of_teammates",
        "gsax", "quality_starts", "analytics",
        "production_context", "translation_risk", "scoring_environment",
        "league_difficulty", "production_adjusted_score", "league_scoring_profile",
        "actual_stats", "projected_stats", "recent_form",
        "projected_gp", "projected_goals", "projected_assists", "projected_points", "projected_ppg",
        "stock_delta", "stock_label", "stock_trend", "stock_reason",
        "weekly_stock_delta", "weekly_stock_label", "weekly_stock_reason",
        "weekly_production_score", "weekly_analytics_score", "week_gp", "week_points",
        "last_prospect_stat_update_date", "prospect_games_simulated_to_date",
    )

    from services.draft_ranking_logic import (
        MIN_BOARD_GOALIES,
        backfill_draft_eligible_goalies,
        build_draft_rank_reason_codes,
        clean_team_name,
        collect_goalie_pipeline_stats,
        enrich_prospect_row_from_player,
        fix_prospect_league_team_row,
        normalize_league_code,
    )

    pipeline_stats = collect_goalie_pipeline_stats(league, age_max=20, pos_fn=_pos_str)
    if pipeline_stats["draft_eligible_dev_goalies"] < MIN_BOARD_GOALIES:
        backfill_draft_eligible_goalies(
            league,
            rng if rng is not None else random.Random(42),
            MIN_BOARD_GOALIES - pipeline_stats["draft_eligible_dev_goalies"],
        )
        pipeline_stats = collect_goalie_pipeline_stats(league, age_max=20, pos_fn=_pos_str)

    prospects: List[Dict[str, Any]] = []

    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "")
        title = str(block.get("league_name") or "")
        for tm in block.get("teams") or []:
            team_name = str(tm.get("name") or "")
            team_id = str(tm.get("team_id") or "")
            for p in tm.get("players") or []:
                if getattr(p, "retired", False):
                    continue
                # Already-drafted juniors stay on their clubs for rights/dev but
                # are not part of the *upcoming* draft class board.
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
                # Draft class matches bootstrap shaping cohort (17–20). Age 16
                # kids are next-year depth, not this year's board.
                if age < 17 or age > 20:
                    continue
                pk = str(getattr(p, "id", "") or "")
                if not pk:
                    continue
                ovr99 = round(_player_ovr99(p), 1)
                pot99 = round(_draft_potential99(p, ovr99), 1)
                pos = _pos_str(p)
                h = abs(hash(pk)) % 997

                # Scout confidence: grows with GP sample; lower early in the year.
                eu = code.startswith("EU_")
                confidence = max(35, min(96, 84 - (8 if eu else 0) + (h % 17) - 8))

                character_concerns = (
                    bool(getattr(p, "character_concerns", False))
                    or bool(getattr(p, "pipeline_bust", False))
                )
                boom_bust = bool(getattr(p, "pipeline_bust", False))
                is_transcendent = bool(getattr(p, "is_transcendent", False) or getattr(p, "transcendent_talent", False))

                league_parts = normalize_league_code(code, title)
                clean_team = clean_team_name(team_name, code, league_parts.get("league_display") or "")

                row: Dict[str, Any] = {
                    "key": pk,
                    "name": _name_str(p),
                    "position": pos,
                    "age": age,
                    "true_ovr": ovr99,
                    "potential_score": pot99,
                    "league_code": code,
                    "league_name": title,
                    "league": league_parts.get("league_display") or _draft_league_display(code, title),
                    "league_display": league_parts.get("league_display") or _draft_league_display(code, title),
                    "league_parent": league_parts.get("league_parent"),
                    "league_sub": league_parts.get("league_sub"),
                    "team_id": team_id,
                    "team_name": clean_team or team_name,
                    "team": clean_team or team_name,
                    "nationality": str(getattr(ident, "birth_country", "") or "") if ident else "",
                    "country": str(getattr(ident, "birth_country", "") or "") if ident else "",
                    "hometown": str(getattr(ident, "birth_city", "") or "") if ident else "",
                    "birth_city": str(getattr(ident, "birth_city", "") or "") if ident else "",
                    "handedness": "Left" if str(getattr(ident, "shoots", "") or "").upper().endswith("L") else "Right",
                    "scouting_confidence": confidence,
                    "character_concerns": character_concerns,
                    "is_bust_risk": bool(getattr(p, "pipeline_bust", False)),
                    "is_gem": bool(getattr(p, "pipeline_steal", False)),
                    "boom_bust": boom_bust,
                    "is_transcendent": is_transcendent,
                    "transcendent_talent": is_transcendent,
                    "generational_goalie": bool(getattr(p, "generational_goalie", False)),
                }
                if is_transcendent:
                    from services.draft_ranking_logic import pick_transcendent_backstory
                    from services.transcendent_origin_stories import attach_origin_story_to_player

                    story_pack = pick_transcendent_backstory(
                        rng,
                        key=str(getattr(p, "backstory_key", "") or ""),
                    )
                    origin = story_pack.get("origin_story") or {}
                    row.update({
                        "aura_tier": str(getattr(p, "aura_tier", "") or "gold"),
                        "draft_hype_tier": str(getattr(p, "draft_hype_tier", "") or "mythic"),
                        "tank_target": True,
                        "true_potential_score": 99.0,
                        "potential_score": 99.0,
                        "origin_story": origin,
                    })
                    attach_origin_story_to_player(p, story_pack)
                elif rng.random() < 0.015 and pos != "G":
                    from services.transcendent_origin_stories import pick_origin_story

                    flavor = pick_origin_story(rng, transcendent=False, seed_hint=h)
                    row["origin_story"] = flavor.get("origin_story")
                row["country_code"] = _draft_country_iso(row.get("nationality") or row.get("country") or "")
                hcm = clamp_height_cm_for_position(getattr(ident, "height_cm", 0) if ident else 0, row.get("position") or pos)
                if hcm:
                    row["height"] = height_cm_to_imperial(hcm)
                    row["height_display"] = height_cm_to_imperial(hcm)
                    row["height_cm"] = hcm
                wkg = int(getattr(ident, "weight_kg", 0) or 0) if ident else 0
                if wkg:
                    row["weight"] = round(wkg * 2.20462)

                enrich_prospect_row_from_player(p, row)

                if scoring_available:
                    try:
                        # Fast path for the full draft-age pool: reuse cached season
                        # lines without deriving full analytics for every junior.
                        # Full prospect_stats_for_api (analytics/war/etc.) runs only
                        # for the composed live board below.
                        from app.sim_engine.generation.prospect_league_scoring import (
                            ensure_prospect_season_stats,
                        )

                        actual = ensure_prospect_season_stats(
                            p,
                            code,
                            rng=rng,
                            calendar_iso=calendar_iso,
                            season_year=season_year,
                        )
                        gp_now = int(actual.get("gp") or actual.get("games_played") or 0)
                        pts_now = int(actual.get("points") or 0)
                        ppg_now = actual.get("ppg")
                        if ppg_now is None and gp_now > 0:
                            ppg_now = round(pts_now / float(gp_now), 3)
                        for sk, val in (
                            ("gp", gp_now),
                            ("games_played", gp_now),
                            ("goals", actual.get("goals")),
                            ("assists", actual.get("assists")),
                            ("points", pts_now),
                            ("ppg", ppg_now),
                            ("points_per_game", ppg_now),
                            ("production_adjusted_score", actual.get("production_adjusted_score")),
                            ("wins", actual.get("wins")),
                            ("losses", actual.get("losses")),
                            ("ot_losses", actual.get("ot_losses")),
                            ("save_pct", actual.get("save_pct")),
                            ("gaa", actual.get("gaa")),
                            ("shutouts", actual.get("shutouts")),
                            ("stock_delta", actual.get("stock_delta")),
                            ("stock_label", actual.get("stock_label")),
                            ("stock_trend", actual.get("stock_trend")),
                            ("stock_reason", actual.get("stock_reason")),
                            ("weekly_stock_delta", actual.get("weekly_stock_delta")),
                            ("weekly_stock_label", actual.get("weekly_stock_label")),
                            ("weekly_stock_reason", actual.get("weekly_stock_reason")),
                        ):
                            if val is not None:
                                row[sk] = val
                        if gp_now > 0:
                            confidence = max(
                                confidence,
                                min(92, 38 + int(gp_now * 1.15) + (h % 9)),
                            )
                        else:
                            confidence = min(confidence, 52)
                        row["scouting_confidence"] = int(confidence)
                    except Exception:
                        row["scouting_confidence"] = int(confidence)
                else:
                    row["scouting_confidence"] = int(confidence)
                # League/team display cleanup is UI-only — run on the composed board
                # (~320), not the full draft-age pool (thousands).
                row["_player"] = p
                prospects.append(row)

    if scoring_available:
        by_league: Dict[str, List[Dict[str, Any]]] = {}
        for row in prospects:
            by_league.setdefault(str(row.get("league_code") or ""), []).append(row)
        for code, group in by_league.items():
            try:
                normalize_league_leader_board(group, code, rng=rng)
            except Exception:
                pass

    from services.draft_ranking_logic import (
        apply_goalie_class_rank_caps,
        apply_goalie_ranking_adjustments,
        apply_hard_ranking_floor_pass,
        apply_potential_band_enforcement,
        enforce_goalie_scatter_final,
        build_potential_intel,
        calculate_prospect_eta,
        compose_live_draft_board,
        compute_ceiling_visibility,
        compute_consensus_potential_evaluation,
        compute_enhanced_draft_score,
        compute_prospect_outcome_band,
        log_draft_class_audit,
        scouting_confidence_for_entry,
        transcendent_storyline_event,
    )

    goalie_class_strength = str(getattr(league, "goalie_class_strength", "normal") or "normal")
    draft_class_depth = str(getattr(league, "draft_class_depth", "average") or "average")

    # --- Draft score: ability + market consensus + production + size translation ---
    wjc_boosts = getattr(session, "wjc_draft_score_boosts", None) or {}
    for row in prospects:
        row["true_potential_score"] = float(row.get("true_potential_score") or row.get("potential_score") or 0)
        row["consensus_potential_score"] = compute_consensus_potential_evaluation(row)
        row["_score"] = compute_enhanced_draft_score(row)
        wjc_key = str(row.get("key") or "")
        if wjc_key and wjc_key in wjc_boosts:
            row["_score"] = float(row["_score"]) + float(wjc_boosts[wjc_key])
        row.update(compute_prospect_outcome_band(row))
        row["draft_rank_reason_codes"] = build_draft_rank_reason_codes(row)

    scores_sorted = sorted((float(r["_score"]) for r in prospects), reverse=True)
    n = len(scores_sorted)
    generational_cut = scores_sorted[max(0, min(n - 1, int(n * (1.0 - GENERATIONAL_GOALIE_SCORE_PCT))))] if n else 0.0
    goalie_boost = float(getattr(league, "goalie_class_boost", 0.0) or 0.0)
    apply_goalie_ranking_adjustments(
        prospects,
        goalie_class_boost=goalie_boost,
        generational_cut=generational_cut,
        goalie_class_strength=goalie_class_strength,
    )

    prospects.sort(key=lambda x: -float(x["_score"]))
    apply_hard_ranking_floor_pass(prospects)
    apply_goalie_class_rank_caps(prospects, goalie_class_strength=goalie_class_strength)
    apply_hard_ranking_floor_pass(prospects)
    apply_potential_band_enforcement(prospects, band_size=32)

    ranking_pool_goalies = sum(1 for r in prospects if str(r.get("position") or "").upper() == "G")
    pipeline_stats["goalies_entering_ranking_pool"] = ranking_pool_goalies

    board_prospects = compose_live_draft_board(prospects)
    board_prospects = [fix_prospect_league_team_row(row) for row in board_prospects]
    # Full analytics payload only for the live board (~320), not the whole junior pool.
    if scoring_available:
        for row in board_prospects:
            p = row.get("_player")
            if p is None:
                continue
            code = str(row.get("league_code") or "")
            try:
                stats = prospect_stats_for_api(
                    p,
                    code,
                    rng=rng,
                    calendar_iso=calendar_iso,
                    season_year=season_year,
                )
                for sk in stat_keys:
                    if stats.get(sk) is not None:
                        row[sk] = stats[sk]
            except Exception:
                pass
    apply_goalie_class_rank_caps(
        board_prospects,
        goalie_class_strength=goalie_class_strength,
        preserve_order=True,
    )
    apply_potential_band_enforcement(board_prospects, band_size=32)
    # Final positional guarantee: non-franchise goalies out of Round 1 and scattered. Must be
    # the last ordering step so the potential-band pass above can't re-promote a strong goalie.
    enforce_goalie_scatter_final(board_prospects, goalie_class_strength=goalie_class_strength)
    # Top board slots must carry realistic current ability (not raw ~50 junior OVR
    # promoted only by early-season PPG). Reshape live players in place.
    try:
        from app.sim_engine.league_hierarchy_bootstrap import ensure_board_prospect_ovr_floors

        ensure_board_prospect_ovr_floors(
            board_prospects,
            rng=rng if rng is not None else random.Random(42),
        )
        for row in board_prospects:
            p = row.get("_player")
            if p is not None and row.get("_ovr_floor_repaired"):
                try:
                    enrich_prospect_row_from_player(p, row)
                except Exception:
                    pass
            # Never leak live player objects into the API payload / cache.
            row.pop("_player", None)
            row.pop("_ovr_floor_repaired", None)
        for row in prospects:
            row.pop("_player", None)
    except Exception:
        for row in board_prospects:
            row.pop("_player", None)
            row.pop("_ovr_floor_repaired", None)
        for row in prospects:
            row.pop("_player", None)

    pipeline_stats["goalies_on_live_board"] = sum(
        1 for r in board_prospects if str(r.get("position") or "").upper() == "G"
    )
    goalie_ranks = [
        i + 1 for i, r in enumerate(board_prospects) if str(r.get("position") or "").upper() == "G"
    ]
    pipeline_stats["goalie_class_strength"] = goalie_class_strength
    pipeline_stats["top_goalie_rank"] = min(goalie_ranks) if goalie_ranks else 0
    pipeline_stats["goalies_top32_on_board"] = sum(
        1 for r in board_prospects[:32] if str(r.get("position") or "").upper() == "G"
    )
    pipeline_stats["goalies_top10_on_board"] = sum(
        1 for r in board_prospects[:10] if str(r.get("position") or "").upper() == "G"
    )

    # --- First-round potential floor + rank-aware outcome bands ---
    # A first-round pick is, by definition, a projected NHLer: no prospect is ever
    # drafted early expecting a bust, so lift the *displayed* ceiling for the top band
    # (≈86 at #1 sliding to ≈70 at #32) and recompute the outcome band with rank present
    # so the reliable floor follows the (now rank-aware) ceiling.
    for _rank_i, _brow in enumerate(board_prospects):
        _rank = _rank_i + 1
        _brow["rank"] = _rank
        if _rank <= 32:
            _min_ceiling = 86.0 - (_rank - 1) * (86.0 - 70.0) / 31.0
            if float(_brow.get("potential_score") or 0) < _min_ceiling:
                _brow["potential_score"] = round(_min_ceiling, 1)
                if float(_brow.get("true_potential_score") or 0) < _min_ceiling:
                    _brow["true_potential_score"] = round(_min_ceiling, 1)
                try:
                    _brow["consensus_potential_score"] = compute_consensus_potential_evaluation(_brow)
                except Exception:
                    pass
        try:
            _brow.update(compute_prospect_outcome_band(_brow))
        except Exception:
            pass

    # --- Class-relative tier grades (1st overall can never be C-grade) ---
    total = len(prospects)
    max_score = float(board_prospects[0]["_score"]) if board_prospects else 1.0
    min_score = float(board_prospects[-1]["_score"]) if board_prospects else 0.0
    spread = max(1e-6, max_score - min_score)
    for i, row in enumerate(board_prospects):
        pct = 1.0 - (i / max(1, total - 1)) if total > 1 else 1.0
        tier = "C"
        for cut, label in _DRAFT_TIER_BANDS:
            if pct >= cut:
                tier = label
                break
        if i == 0 and tier not in ("A+", "A", "A-"):
            tier = "A-"
        row["scout_tier"] = tier
        row["talent_grade"] = tier
        norm = (float(row["_score"]) - min_score) / spread
        row["scout_grade"] = round(40.0 + norm * 58.0, 1)

    # --- Class strength tier from top-end quality + depth profile ---
    top_pot = [float(r.get("true_potential_score") or r.get("potential_score") or 0) for r in board_prospects[:10]]
    avg_top = sum(top_pot) / max(1, len(top_pot))
    if avg_top >= 82:
        class_strength = "Generational class"
    elif avg_top >= 76:
        class_strength = "Elite class"
    elif avg_top >= 70:
        class_strength = "Strong class"
    elif avg_top >= 63:
        class_strength = "Average class"
    else:
        class_strength = "Weak class"
    depth_suffix = {
        "elite": " · Deep class",
        "strong": " · Good depth",
        "average": "",
        "weak": " · Thin depth",
    }.get(draft_class_depth, "")
    class_strength = f"{class_strength}{depth_suffix}"

    # --- Weekly stock movement (rank change since last week snapshot) ---
    prev = dict(getattr(session, "draft_rank_prev", None) or {})
    preseason = dict(getattr(session, "draft_preseason_rank", None) or prev)
    midseason = dict(getattr(session, "draft_midseason_rank", None) or {})
    entries: List[Dict[str, Any]] = []
    for i, row in enumerate(board_prospects):
        rank = i + 1
        key = str(row["key"])
        pr = int(prev.get(key, rank))
        ps = int(preseason.get(key, pr))
        ms = int(midseason.get(key) or 0) or None
        rank_signed_delta = pr - rank  # positive = moved up on board (weekly, non-cumulative)

        weekly_heat = int(row.get("weekly_stock_delta") or 0)
        # Keep units honest: rank spots when the board moved; otherwise weekly heat.
        # Never add heat onto a rank jump (that inflated +N past real spots moved).
        if key not in prev:
            signed_delta = max(-6, min(6, weekly_heat))
            stock_mode = "weekly_heat" if weekly_heat != 0 else "none"
            stock_unit = "heat"
        elif abs(rank_signed_delta) >= 1:
            signed_delta = rank_signed_delta
            stock_mode = "rank_change"
            stock_unit = "rank"
        else:
            signed_delta = max(-6, min(6, weekly_heat))
            stock_mode = "weekly_heat" if signed_delta != 0 else "none"
            stock_unit = "heat"

        if key not in prev:
            trend = "NEW"
            stock_label = str(row.get("weekly_stock_label") or row.get("stock_label") or "New")
        elif signed_delta >= 5:
            trend, stock_label = "UP", "Rocketing"
        elif signed_delta >= 3:
            trend, stock_label = "UP", "Rising"
        elif signed_delta >= 1:
            trend, stock_label = "UP", "Trending Up"
        elif signed_delta <= -5:
            trend, stock_label = "DOWN", "Crashing"
        elif signed_delta <= -3:
            trend, stock_label = "DOWN", "Falling"
        elif signed_delta <= -1:
            trend, stock_label = "DOWN", "Slipping"
        else:
            trend, stock_label = "SAME", "Holding"

        # Always rebuild reason from the FINAL displayed delta (not a stale weekly string).
        stock_reason = _draft_stock_reason(
            {**row, "stock_mode": stock_mode, "stock_unit": stock_unit},
            signed_delta,
            goalie_penalized=bool(row.get("_goalie_penalized")),
        )
        row["stock_mode"] = stock_mode
        row["stock_unit"] = stock_unit
        row["stock_heat"] = weekly_heat
        row["rank_delta"] = rank_signed_delta if key in prev else 0
        row["stock_label"] = stock_label
        row["stock_reason"] = stock_reason
        row["trend"] = trend
        row["stock_delta"] = signed_delta
        row["stock_change"] = signed_delta

        goalie_penalized = bool(row.get("_goalie_penalized"))
        base_conf = float(row.get("scouting_confidence") or 50)
        conf = scouting_confidence_for_entry(row, session, base_conf=base_conf)
        # The displayed scout estimate is a FOGGED read of the prospect's real
        # ceiling: centre on true potential, then apply a persistent per-prospect
        # evaluation miss (seed_key) so scouts disagree and can be wrong — rather
        # than a widening band that always straddles the hidden centre. (Ranking /
        # band decisions use the observable consensus_potential_score elsewhere.)
        display_center = float(
            row.get("true_potential_score")
            or row.get("potential_score")
            or row.get("consensus_potential_score")
            or 0
        )
        pot_seed = str(row.get("key") or row.get("name") or "")
        # Ceiling readability is gated by draft position: obvious for early picks, fading
        # to a vague range and then vanishing (floor-only) the deeper the prospect sits.
        # Only the user's OWN dedicated scouting overlay (not the ambient games-played
        # confidence) can re-open a late prospect's ceiling.
        scout_overlay_pct = 0.0
        _scouting_state = getattr(session, "scouting_state", None) or {}
        _scout_prospects = _scouting_state.get("prospects") if isinstance(_scouting_state, dict) else {}
        if isinstance(_scout_prospects, dict):
            _ov = _scout_prospects.get(str(row.get("key") or ""))
            if isinstance(_ov, dict) and _ov.get("scouted_percentage") is not None:
                try:
                    scout_overlay_pct = float(_ov["scouted_percentage"])
                except (TypeError, ValueError):
                    scout_overlay_pct = 0.0
        _cv = compute_ceiling_visibility(rank, scout_overlay_pct)
        ceiling_visibility = float(_cv["visibility"])
        ceiling_state = str(_cv["state"])
        ceiling_hidden = bool(_cv["ceiling_hidden"])
        # Lower visibility widens the scout's ceiling band (less certainty on upside).
        ceiling_conf = conf * (0.5 + 0.5 * ceiling_visibility)
        intel = build_potential_intel(
            display_center, ceiling_conf, overlay_pct=ceiling_conf, include_true=False, seed_key=pot_seed,
        )
        entry = {k: v for k, v in row.items() if not k.startswith("_")}
        # Strip hidden truth fields before public serialization.
        raw_true_ovr = entry.pop("true_ovr", None)
        entry.pop("true_potential_score", None)
        key_in_prev = key in prev
        # Fog current ability — exact true_ovr only when scouting intends a reveal.
        try:
            t_ovr = float(raw_true_ovr if raw_true_ovr is not None else row.get("true_ovr") or 0)
        except (TypeError, ValueError):
            t_ovr = 0.0
        ovr_gap = max(2.0, (100.0 - conf) * 0.2)
        ovr_bias = ((int(conf) % 11) - 5) * 0.15
        ovr_lo = max(40.0, t_ovr - ovr_gap + ovr_bias)
        ovr_hi = min(95.0, t_ovr + ovr_gap * 0.45 + ovr_bias)
        ovr_est = round((ovr_lo + ovr_hi) / 2.0, 1)
        ovr_revealed = bool(row.get("ovr_revealed")) or conf >= 88.0
        entry.update(
            {
                "rank": rank,
                "rank_prev": pr,
                "previous_rank": pr,
                "preseason_rank": ps,
                "midseason_rank": ms,
                "rank_change": rank_signed_delta if key_in_prev else 0,
                "rank_delta": rank_signed_delta if key_in_prev else 0,
                "stock_delta": signed_delta,
                "stock_change": signed_delta,
                "stock_heat": weekly_heat,
                "stock_mode": stock_mode,
                "stock_unit": stock_unit,
                "production_score": row.get("production_score"),
                "analytics_score": row.get("analytics_score"),
                "stock_direction": "up" if signed_delta > 0 else ("down" if signed_delta < 0 else "flat"),
                "stock_label": stock_label,
                "stock_trend": str(row.get("stock_trend") or stock_label),
                "stock_reason": stock_reason,
                "trend": trend,
                "risk": "High" if row.get("is_bust_risk") or row.get("character_concerns") else (
                    "Medium" if float(row.get("scouting_confidence") or 70) < 65 else "Low"
                ),
                "scouting_confidence": round(float(conf), 1),
                "public_scouting_confidence": round(float(base_conf), 1),
                "intel_label": intel["intel_label"],
                "potential_range": intel["potential_range"],
                "ceiling_range": intel.get("ceiling_range") or intel["potential_range"],
                "expected_ceiling_estimate": intel.get("expected_ceiling_estimate", intel["potential_score"]),
                "potential_score": intel["potential_score"],  # estimated ceiling (NOT true)
                "uncertainty": intel.get("uncertainty"),
                "ceiling_visibility": round(ceiling_visibility, 3),
                "ceiling_state": ceiling_state,
                "ceiling_hidden": ceiling_hidden,
                "scouted_percentage": round(scout_overlay_pct, 1),
                "ovr_revealed": ovr_revealed,
                # Exact ability only when revealed; otherwise omit misleading true_ovr key.
                "current_ovr_estimate": ovr_est,
                "current_ovr_range": [round(ovr_lo, 1), round(ovr_hi, 1)],
                "eta": calculate_prospect_eta(row, final_rank=rank),
            }
        )
        if ovr_revealed:
            entry["true_ovr"] = round(t_ovr, 1)
        else:
            entry.pop("true_ovr", None)
        if ceiling_hidden:
            # Late/low-attention: strip every graded ceiling/floor number from the public
            # board. The user projects upside from production, age, size and attributes.
            entry["potential_range"] = None
            entry["ceiling_range"] = None
            entry["expected_ceiling_estimate"] = None
            entry["potential_score"] = None
            entry["floor_score"] = None
            entry["ceiling_score"] = None
            entry["outcome_band"] = None
            entry["ceiling_hint"] = None
            if scout_overlay_pct < 20.0:
                # Ambient GP confidence must not look like a finished scouting file.
                entry["scouting_confidence"] = min(float(entry.get("scouting_confidence") or 42), 42.0)
                entry["intel_label"] = "Limited"
        entry["draft_stock"] = _compute_draft_stock_object(entry, rank, key_in_prev=key_in_prev)
        entry["franchise_tier"] = _compute_franchise_tier_object(entry, rank)
        # Gem-finder WAR after ceiling fog is known — overproduction lights up without leaking potential.
        try:
            from services.draft_prospect_profile import _analytics_from_row_stats

            derived = _analytics_from_row_stats(entry)
            if derived:
                existing = entry.get("analytics") if isinstance(entry.get("analytics"), dict) else {}
                merged = dict(existing or {})
                merged.update(derived)
                entry["analytics"] = merged
                for k in ("war", "offensive_war", "defensive_war", "xgf_pct", "cf_pct", "shot_rate", "primary_points", "toi"):
                    if derived.get(k) is not None:
                        entry[k] = derived[k]
        except Exception:
            pass
        entries.append(entry)

    transcendent_rows = [e for e in entries if e.get("is_transcendent")]
    if transcendent_rows:
        _record_storyline(session, transcendent_storyline_event(transcendent_rows[0]))
        session.transcendent_draft_prospect_id = str(transcendent_rows[0].get("key") or "")
        from services.transcendent_tank_behavior import refresh_transcendent_tank_pressure

        refresh_transcendent_tank_pressure(session, sim, transcendent_present=True)

    # Public board must not ship hidden-truth / engine-private keys (matches entry-draft strip).
    try:
        from services.draft_ranking_logic import PUBLIC_FORBIDDEN_KEYS

        cleaned_entries: List[Dict[str, Any]] = []
        for e in entries:
            cleaned_entries.append(
                {
                    k: v
                    for k, v in e.items()
                    if not (isinstance(k, str) and (k.startswith("_") or k in PUBLIC_FORBIDDEN_KEYS))
                }
            )
        entries = cleaned_entries
    except Exception:
        pass

    audit = log_draft_class_audit(entries, label=str(getattr(league, "draft_class_strength", "")))

    cal_iso = _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
    stock_summary = _build_stock_market_summary(entries, generated_at=str(cal_iso or "")[:10])
    tier_summary = _build_tier_summary(entries)

    return {
        "entries": entries,
        "class_strength": class_strength,
        "subtitle": f"{class_strength} ┬╖ draft-age (Γëñ20) in dev leagues ┬╖ showing {len(entries)}",
        "total": total,
        "stock_market_summary": stock_summary,
        "tier_summary": tier_summary,
        "draft_audit": audit,
        "goalie_class_strength": str(getattr(league, "goalie_class_strength", "normal") or "normal"),
        "draft_class_depth": draft_class_depth,
        "goalie_pipeline": pipeline_stats,
    }


def _draft_stock_week_key(session: FranchiseSession) -> str:
    try:
        from app.sim_engine.generation.prospect_league_scoring import _iso_week_key

        iso = _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
        return _iso_week_key(iso)
    except Exception:
        return ""


def snapshot_draft_rank_prev(session: FranchiseSession, sim: Any, *, force: bool = False) -> None:
    """Store draft ranks once per ISO week so stock reflects weekly movement, not cumulative drift."""
    from services.franchise_entry_draft import append_stock_history_snapshot

    week_key = _draft_stock_week_key(session)
    if not force and week_key and getattr(session, "draft_rank_snapshot_week", "") == week_key:
        return

    board = build_draft_class_rankings(session, sim)
    entries = board.get("entries") or []
    ranks = {str(e.get("key")): int(e.get("rank", i + 1)) for i, e in enumerate(entries) if e.get("key")}
    cal_iso = _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
    date_label = str(cal_iso)[:10] if cal_iso else ""
    ui_phase = "midseason"
    if getattr(session, "phase", "") == "offseason":
        ui_phase = "final_ranking"
    elif force:
        ui_phase = "season_end"
    # Seed a preseason trail point the first time we snapshot so charts have a left anchor.
    if not getattr(session, "draft_preseason_rank", None):
        session.draft_preseason_rank = dict(ranks)
        for e in entries[:200]:
            append_stock_history_snapshot(
                session, e, event_source="preseason", date_label=date_label or "Preseason"
            )
    # Capture midseason ranks once we cross the calendar midpoint of the regular season.
    mid_map = getattr(session, "draft_midseason_rank", None)
    if not isinstance(mid_map, dict):
        mid_map = {}
    try:
        cur = int(getattr(session, "calendar_cursor", 0) or 0)
        last_reg = int(getattr(session, "nhl_regular_season_last_index", 0) or 0)
        if last_reg > 0 and cur >= max(1, last_reg // 2) and not mid_map:
            mid_map = dict(ranks)
            session.draft_midseason_rank = mid_map
            ui_phase = "midseason"
            date_label = date_label or "Midseason"
            # One-shot mid-season development-promise morale enforcement
            if not getattr(session, "midseason_contract_ledger", None):
                try:
                    from services.elc_year_end_ledger import run_midseason_contract_ledger

                    sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
                    session.midseason_contract_ledger = run_midseason_contract_ledger(
                        session, season_year=sy
                    )
                except Exception:
                    pass
    except Exception:
        pass
    for e in entries[:200]:
        append_stock_history_snapshot(session, e, event_source=ui_phase, date_label=date_label)
    session.draft_rank_prev = ranks
    if week_key:
        session.draft_rank_snapshot_week = week_key
    if not getattr(session, "draft_preseason_rank", None):
        session.draft_preseason_rank = dict(ranks)


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
    for key in (
        "storyline_id",
        "stable_key",
        "category",
        "trade_category",
        "trade_id",
        "reason_codes",
        "reason_text",
        "execution",
        "severity",
        "short_summary",
        "description",
        "player_id",
        "player_position",
        "player_overall",
        "team_name",
        "evidence",
        "requires_action",
        "status",
        "source",
        "heat",
        "credibility",
        "repeat_count",
        "escalated_from",
        "related_team_ids",
        "related_player_ids",
        "games_remaining",
        "games_initial",
        "return_date",
        "return_estimate",
        "overall_before",
        "overall_after",
        "overall_delta",
        "base_overall",
        "effective_overall",
        "impact_reason",
        "follow_up",
        "arc_status",
        "legal_severity",
        "event_type",
        "cause_type",
        "cause_event_id",
        "culprit_player_id",
        "affected_player_ids",
        "source_label",
        "user_visible_explanation",
        "recovery_conditions",
        "resolution_condition",
        "resolution_reason",
        "culprit_player_name",
        # Conduct state machine channels (allegation ≠ guilt)
        "incident_id",
        "eligible_to_play",
        "team_can_override",
        "allegation_note",
        "information_status",
        "legal_status",
        "league_status",
        "team_status",
        "conduct_model",
        "dress_backlash_risk",
        "incident_family",
        "evidence_confidence",
        "resolution",
        "kind",
        "arc_tier",
        "calendar_day",
        "team_abbr",
        "from_team_name",
        "to_team_name",
        "from_team_abbrev",
        "to_team_abbrev",
        "related_teams",
        "teams",
        # Narrative universe
        "arc_id",
        "beat_id",
        "beat_index",
        "arc_phase",
        "knowledge_type",
        "narrative_angle",
        "reporter_id",
        "reporter_name",
        "outlet_id",
        "outlet_name",
        "world_event_id",
        "body",
        "knowledge_layers",
        "public_knowledge_level",
        "gm_knows_more",
        "visibility",
        "market_key",
        "market_tone",
        "market_descriptor",
        "breaking_level",
        "press_conference_id",
    ):
        if raw.get(key) is not None:
            out[key] = raw.get(key)
    if raw.get("body") is not None and not out.get("summary"):
        out["summary"] = str(raw.get("body") or "").strip()
    if raw.get("requires_action") is not None:
        out["requires_action"] = bool(raw.get("requires_action"))
    if raw.get("eligible_to_play") is not None:
        out["eligible_to_play"] = bool(raw.get("eligible_to_play"))
    if raw.get("team_can_override") is not None:
        out["team_can_override"] = bool(raw.get("team_can_override"))
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
    sk = str(ev.get("stable_key") or "").strip()
    if sk:
        return sk
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
    raw = event if isinstance(event, dict) else {}
    try:
        from app.sim_engine.franchise.storyline_engine import enrich_storyline_for_narrative_universe  # noqa: WPS433

        raw = enrich_storyline_for_narrative_universe(session, raw)
    except Exception:
        pass
    ev = _normalize_storyline_payload(raw)
    if not ev.get("headline"):
        return
    dq = getattr(session, "_storyline_dedupe", None)
    if dq is None:
        dq = []
        session._storyline_dedupe = dq
    dq_set = getattr(session, "_storyline_dedupe_set", None)
    if dq_set is None:
        dq_set = set(dq)
        session._storyline_dedupe_set = dq_set
    dk = _storyline_dedupe_key(ev)
    if dk in dq_set:
        return
    dq_set.add(dk)
    dq.append(dk)
    if len(dq) > 500:
        trimmed = dq[-400:]
        session._storyline_dedupe = trimmed
        session._storyline_dedupe_set = set(trimmed)
    if getattr(session, "storyline_events", None) is None:
        session.storyline_events = []
    session.storyline_events.append(ev)
    if len(session.storyline_events) > 400:
        session.storyline_events = session.storyline_events[-400:]


def _team_record_snapshot_for_direction(session: FranchiseSession, team_id: str) -> Dict[str, float]:
    st = getattr(session, "standings", None)
    if st is None:
        return {"gp": 0.0, "pts_pct": 0.0}
    rec = None
    try:
        if hasattr(st, "find_record"):
            rec = st.find_record(str(team_id))
        if rec is None:
            rec = (getattr(st, "records", None) or {}).get(str(team_id))
    except Exception:
        rec = None
    if rec is None:
        return {"gp": 0.0, "pts_pct": 0.0}
    gp = float(getattr(rec, "gp", 0) or 0)
    pts = float(getattr(rec, "pts", getattr(rec, "points", 0)) or 0)
    return {"gp": gp, "pts_pct": (pts / max(1.0, gp * 2.0)) if gp > 0 else 0.0}


def _cpu_direction_state(
    *,
    team: Any,
    team_id: str,
    record: Dict[str, float],
    deadline_phase: float,
    pressure: float,
    prev_profile: Dict[str, Any],
    calendar_idx: int,
) -> Tuple[str, str, float]:
    window = str(getattr(team, "gm_window", getattr(team, "window", "balanced")) or "balanced").lower()
    pts_pct = float(record.get("pts_pct", 0.0) or 0.0)
    gp = int(record.get("gp", 0) or 0)
    streak_bias = float(getattr(team, "point_pct", pts_pct) or pts_pct)
    quality_hint = max(0.0, min(1.0, (streak_bias + pts_pct) * 0.5))
    reason = "BASELINE_REASSESSMENT"
    confidence = 0.58 + min(0.27, float(gp) / 170.0)
    state = "HOLDING"

    if pressure >= 0.84:
        state = "CAP_CORRECTION"
        reason = "CAP_PRESSURE"
    elif quality_hint >= 0.69 and deadline_phase > 0.55 and window in ("contender", "retool"):
        state = "ALL_IN_CONTENDER"
        reason = "OWNERSHIP_WIN_NOW"
        confidence += 0.1
    elif quality_hint >= 0.62 and window in ("contender", "emerging", "retool"):
        state = "PLAYOFF_BUYER" if deadline_phase > 0.28 else "CONTENDER"
        reason = "UNEXPECTED_CONTENTION" if window != "contender" else "PLAYOFF_PUSH"
    elif quality_hint <= 0.41 and window in ("rebuild", "tank", "declining"):
        state = "DEEP_REBUILD" if quality_hint <= 0.35 else "REBUILDING"
        reason = "FAILED_EXPECTATIONS"
    elif quality_hint <= 0.44 and deadline_phase > 0.4:
        state = "SELLER"
        reason = "PLAYOFF_ODDS_COLLAPSE"
    elif window in ("retool", "balanced"):
        state = "COMPETITIVE_RETOOL"
        reason = "AGE_TIMELINE_BALANCE"

    prev_state = str(prev_profile.get("team_direction") or "")
    prev_conf = float(prev_profile.get("direction_confidence", 0.0) or 0.0)
    changed_recently_at = int(prev_profile.get("direction_last_changed_day", -999) or -999)
    cur_day = int(calendar_idx)
    # Hysteresis: avoid noisy daily flips unless confidence clearly improved.
    if prev_state and prev_state != state and (cur_day - changed_recently_at) < 6 and confidence <= (prev_conf + 0.09):
        return prev_state, str(prev_profile.get("direction_change_reason") or "STABILITY_HOLD"), max(prev_conf, confidence - 0.06)
    return state, reason, max(0.25, min(0.98, confidence))


def _default_cpu_ideology(team: Any) -> Dict[str, float]:
    window = str(getattr(team, "gm_window", getattr(team, "window", "balanced")) or "balanced").lower()
    strategy = str(getattr(team, "gm_strategy", "balanced") or "balanced").lower()
    aggression = 0.55 if "aggressive" in strategy or "win" in strategy else 0.42
    patience = 0.58 if "patient" in strategy or "develop" in strategy else 0.45
    future = 0.68 if window in ("rebuild", "tank", "declining") else 0.38
    veteran = 0.62 if window in ("contender",) else 0.4
    return {
        "aggression": aggression,
        "patience": patience,
        "future_asset_preference": future,
        "veteran_preference": veteran,
        "prospect_protection": 0.55 + (0.15 if future > 0.55 else 0.0),
        "draft_pick_protection": 0.52 + (0.12 if future > 0.55 else 0.0),
        "trade_frequency_preference": 0.48 + (aggression - 0.5) * 0.4,
        "risk_tolerance": 0.45 + (aggression - 0.5) * 0.5,
        "contract_term_tolerance": 0.5,
        "goaltending_investment": 0.5,
        "best_player_available_bias": 0.55,
        "positional_need_draft_bias": 0.45,
    }


def _evolve_cpu_ideology(
    *,
    prev: Dict[str, Any],
    direction: str,
    change_reason: str,
    record: Dict[str, float],
    pressure: float,
    calendar_idx: int,
) -> Tuple[Dict[str, float], Optional[str]]:
    ideology = dict(prev.get("ideology") or {})
    if not ideology:
        ideology = {}
    last_change = int(prev.get("ideology_last_changed_day", -999) or -999)
    # Annual / major-event cooldown (~45 days).
    if (calendar_idx - last_change) < 45 and ideology:
        return ideology, None
    pts_pct = float(record.get("pts_pct", 0.0) or 0.0)
    story: Optional[str] = None
    deltas: List[Tuple[str, float]] = []
    if change_reason == "FAILED_EXPECTATIONS" and direction in ("REBUILDING", "DEEP_REBUILD"):
        deltas = [("future_asset_preference", 0.06), ("patience", 0.05), ("aggression", -0.04)]
        story = "Organization commits to a longer youth timeline."
    elif change_reason == "PLAYOFF_ODDS_COLLAPSE" and direction == "SELLER":
        deltas = [("future_asset_preference", 0.05), ("veteran_preference", -0.04)]
        story = "Management pivots toward future assets after playoff odds collapsed."
    elif change_reason in ("PLAYOFF_PUSH", "OWNERSHIP_WIN_NOW") and direction in ("CONTENDER", "ALL_IN_CONTENDER", "PLAYOFF_BUYER"):
        deltas = [("aggression", 0.05), ("draft_pick_protection", -0.04)]
        story = "Management becomes more aggressive in pursuit of a playoff push."
    elif change_reason == "CAP_PRESSURE" or pressure >= 0.84:
        deltas = [("contract_term_tolerance", -0.06), ("future_asset_preference", 0.04)]
        story = "Team prioritizes longer-term flexibility after cap pressure."
    elif pts_pct >= 0.62 and direction in ("CONTENDER", "PLAYOFF_BUYER") and float(record.get("gp", 0) or 0) >= 40:
        deltas = [("patience", -0.03), ("risk_tolerance", 0.04)]
    if not deltas:
        return ideology, None
    for key, delta in deltas[:2]:
        base = float(ideology.get(key, 0.5) or 0.5)
        ideology[key] = round(max(0.15, min(0.9, base + delta)), 3)
    return ideology, story


def _refresh_cpu_franchise_profiles(session: FranchiseSession, *, calendar_idx: int, force: bool = False) -> None:
    scheduler = getattr(session, "cpu_scheduler_state", None)
    if not isinstance(scheduler, dict):
        scheduler = {}
        session.cpu_scheduler_state = scheduler
    profiles = getattr(session, "cpu_franchise_profiles", None)
    if not isinstance(profiles, dict):
        profiles = {}
        session.cpu_franchise_profiles = profiles

    last_strategic_day = int(scheduler.get("last_strategic_day", -999) or -999)
    if not force and (calendar_idx - last_strategic_day) < 7:
        return

    max_d = max(40, int(getattr(session, "nhl_regular_season_last_index", 0) or 0))
    md = max(40, int(max(120, max_d) * 0.56))
    deadline_phase = max(0.0, min(1.0, (float(calendar_idx) - float(md)) / max(20.0, float(max_d) * 0.2)))
    league = getattr(getattr(session, "sim", None), "league", None)
    hist = list(getattr(league, "trade_history", None) or [])
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    completed_cpu = sum(
        1
        for row in hist
        if isinstance(row, dict)
        and int(row.get("season_year") or season_year) == season_year
        and not bool(row.get("user_involved"))
    )

    for tid, tm in (getattr(session, "team_by_id", None) or {}).items():
        tid_s = str(tid)
        if tid_s == str(getattr(session, "user_team_id", "") or ""):
            continue
        prev = dict(profiles.get(tid_s) or {})
        cap_pressure = float(getattr(tm, "cap_pressure", 0.0) or 0.0)
        rec = _team_record_snapshot_for_direction(session, tid_s)
        direction, change_reason, confidence = _cpu_direction_state(
            team=tm,
            team_id=tid_s,
            record=rec,
            deadline_phase=deadline_phase,
            pressure=cap_pressure,
            prev_profile=prev,
            calendar_idx=calendar_idx,
        )
        changed = direction != str(prev.get("team_direction") or "")
        ideology = dict(prev.get("ideology") or _default_cpu_ideology(tm))
        for k, v in _default_cpu_ideology(tm).items():
            ideology.setdefault(k, v)
        evolved, ideo_story = _evolve_cpu_ideology(
            prev={**prev, "ideology": ideology},
            direction=direction,
            change_reason=change_reason if changed else "",
            record=rec,
            pressure=cap_pressure,
            calendar_idx=calendar_idx,
        )
        ideology = evolved
        profile = {
            "team_id": tid_s,
            "team_direction": direction,
            "direction_confidence": round(confidence, 3),
            "direction_last_changed_day": int(calendar_idx) if changed else int(prev.get("direction_last_changed_day", calendar_idx) or calendar_idx),
            "direction_change_reason": change_reason if changed else str(prev.get("direction_change_reason") or change_reason),
            "competitive_window": str(getattr(tm, "gm_window", getattr(tm, "window", "balanced")) or "balanced").lower(),
            "management_aggression": str(getattr(tm, "gm_strategy", "balanced") or "balanced").lower(),
            "cap_pressure": round(cap_pressure, 3),
            "deadline_urgency": round(deadline_phase, 3),
            "playoff_points_pct": round(float(rec.get("pts_pct", 0.0) or 0.0), 3),
            "trade_activity_level": "high" if deadline_phase > 0.72 else "medium" if deadline_phase > 0.35 else "low",
            "last_profile_refresh_day": int(calendar_idx),
            "season_year": season_year,
            "league_cpu_trade_count": int(completed_cpu),
            "ideology": ideology,
            "ideology_last_changed_day": int(calendar_idx) if ideo_story else int(prev.get("ideology_last_changed_day", -999) or -999),
            "ideology_change_reason": ideo_story or str(prev.get("ideology_change_reason") or ""),
        }
        profiles[tid_s] = profile
        try:
            setattr(tm, "_cpu_direction_state", direction)
            setattr(tm, "_cpu_direction_confidence", confidence)
            setattr(tm, "_cpu_ideology", ideology)
        except Exception:
            pass
        if ideo_story and changed:
            _record_storyline(
                session,
                {
                    "type": "management",
                    "date": int(calendar_idx),
                    "headline": ideo_story,
                    "team": tid_s,
                    "priority": "MEDIUM",
                    "cause_type": "ideology_shift",
                    "cause": ideo_story,
                },
            )
    scheduler["last_strategic_day"] = int(calendar_idx)


def _cpu_trade_asset_lines(
    ev: Dict[str, Any],
    session: Optional[FranchiseSession] = None,
) -> Tuple[List[str], List[str]]:
    ex = dict(ev.get("execution") or {})
    moved = list(ex.get("moved_assets") or [])
    hist = dict(ex.get("history_record") or {})
    if not moved:
        moved = list(hist.get("moved_players") or []) + list(hist.get("moved_picks") or [])
    from_team = str(ev.get("from_team_id") or "")
    to_team = str(ev.get("to_team_id") or ev.get("team") or "")
    to_lines: List[str] = []
    from_lines: List[str] = []
    for asset in moved:
        if not isinstance(asset, dict):
            continue
        src = str(asset.get("source_team_id") or "")
        dst = str(asset.get("acquiring_team_id") or "")
        at = str(asset.get("asset_type") or asset.get("type") or "").lower()
        if at in ("player", ""):
            label = str(asset.get("player_name") or asset.get("display_name") or asset.get("asset_id") or "Player")
        else:
            yr = asset.get("year")
            rnd = asset.get("round")
            label = (
                str(asset.get("display_name") or "")
                or (f"{yr} Round {rnd}" if yr and rnd else f"Pick {asset.get('asset_id') or '?'}")
            )
            orig_abbr = _trade_popup_team_abbr(session, asset.get("original_team_id"))
            if orig_abbr and orig_abbr not in label:
                label = f"{label} ({orig_abbr})"
        try:
            retained = float(asset.get("retained_pct") or 0)
        except (TypeError, ValueError):
            retained = 0.0
        if at in ("player", "") and retained > 0:
            label = f"{label} ({retained:g}% retained)"
        if dst == to_team and src == from_team:
            to_lines.append(label)
        elif dst == from_team and src == to_team:
            from_lines.append(label)
        elif dst == to_team:
            to_lines.append(label)
        elif dst == from_team:
            from_lines.append(label)
    # Fallback to proposer labels when execution payload is sparse.
    if not to_lines:
        to_lines = [str(x) for x in list(ev.get("outgoing") or []) if str(x).strip()][:10]
    if not from_lines:
        from_lines = [str(x) for x in list(ev.get("incoming") or []) if str(x).strip()][:10]
    return to_lines[:10], from_lines[:10]


def _trade_popup_resolve_player(session: Optional[FranchiseSession], team_id: str, player_id: str) -> Optional[Any]:
    if session is None or not player_id:
        return None
    pid = str(player_id)
    tid = str(team_id or "")
    if tid:
        found = _fan_resolve_player(session, tid, pid)
        if found is not None:
            return found
    for tm in (getattr(session, "team_by_id", None) or {}).values():
        for p in getattr(tm, "roster", None) or []:
            if str(getattr(p, "id", "") or "") == pid:
                return p
    return None


def _trade_popup_archetype_label(player: Any) -> str:
    raw = str(getattr(player, "archetype", "") or "").strip()
    if not raw:
        return ""
    return raw.replace("_", " ").replace("-", " ").title()


def _trade_popup_contract_bits(player: Any) -> Tuple[Optional[float], Optional[int]]:
    cap_hit = None
    years_left = None
    try:
        cap_hit = float(_player_cap_hit_millions(player))
    except Exception:
        cap_hit = None
    contract = getattr(player, "contract", None)
    for attr in ("years_remaining", "years_left", "term_years", "years"):
        raw = getattr(contract, attr, None) if contract is not None else None
        if raw is None:
            raw = getattr(player, attr, None)
        try:
            if raw is not None:
                years_left = max(0, int(raw))
                break
        except Exception:
            continue
    return cap_hit, years_left


def _trade_popup_ovr(player: Any) -> Optional[int]:
    try:
        from app.sim_engine.franchise.storyline_conduct import get_effective_ovr_display  # noqa: WPS433

        return int(get_effective_ovr_display(player))
    except Exception:
        pass
    ovr_f = getattr(player, "ovr", None)
    try:
        ov = float(ovr_f() if callable(ovr_f) else ovr_f)
        if ov <= 1.5:
            ov *= 99.0
        return int(round(ov))
    except Exception:
        return None


def _trade_popup_age(player: Any) -> Optional[int]:
    ident = getattr(player, "identity", None)
    for src in (ident, player):
        if src is None:
            continue
        for attr in ("age", "player_age"):
            raw = getattr(src, attr, None)
            try:
                if raw is not None:
                    age = int(raw)
                    if 15 <= age <= 55:
                        return age
            except Exception:
                continue
    return None


def _trade_popup_season_stats(session: Optional[FranchiseSession], player_id: str) -> Dict[str, Any]:
    if session is None or not player_id:
        return {}
    row = dict((getattr(session, "player_season_stats", None) or {}).get(str(player_id)) or {})
    if not row:
        return {}
    gp = int(row.get("gp") or 0)
    g = int(row.get("g") or 0)
    a = int(row.get("a") or 0)
    pts = int(row.get("pts") or (g + a))
    xgf_pct = None
    war = None
    xgf_sample = float(row.get("xgf") or 0) + float(row.get("xga") or 0)
    xgf_gp = int(row.get("xgf_pct_gp") or 0)
    has_xgf_sample = xgf_sample > 0 or xgf_gp > 0
    try:
        from app.sim_engine.gameplay.game_analytics_ledger import season_xgf_pct_from_row  # noqa: WPS433
        from app.sim_engine.generation.player_analytics import enrich_player_row  # noqa: WPS433

        enriched = enrich_player_row(row)
        if has_xgf_sample:
            raw_xgf = enriched.get("xgf_pct")
            if raw_xgf is None:
                raw_xgf = season_xgf_pct_from_row(row)
            if raw_xgf is not None:
                xf = float(raw_xgf)
                xgf_pct = round(xf * 100.0, 1) if xf <= 1.5 else round(xf, 1)
        war_raw = enriched.get("war")
        # Prefer computed WAR whenever the player has played; enrichment always derives it.
        if war_raw is not None and gp > 0:
            war = round(float(war_raw), 1)
    except Exception:
        # Fallback: derive xGF% from ledger totals when enrichment is unavailable.
        try:
            if xgf_sample > 0:
                xgf_pct = round((float(row.get("xgf") or 0) / xgf_sample) * 100.0, 1)
            elif xgf_gp > 0:
                xgf_pct = round((float(row.get("xgf_pct_sum") or 0) / float(xgf_gp)) * 100.0, 1)
            elif row.get("xgf_pct") is not None:
                xf = float(row.get("xgf_pct"))
                xgf_pct = round(xf * 100.0, 1) if xf <= 1.5 else round(xf, 1)
        except Exception:
            xgf_pct = None
        try:
            if row.get("war") is not None and gp > 0:
                war = round(float(row.get("war")), 1)
        except Exception:
            war = None
    return {
        "gp": gp,
        "g": g,
        "a": a,
        "pts": pts,
        "xgf_pct": xgf_pct,
        "war": war,
    }


def _resolve_session_team(session: Optional[FranchiseSession], team_id: Any) -> Optional[Any]:
    if session is None or team_id is None or team_id == "":
        return None
    by_id = getattr(session, "team_by_id", None) or {}
    tid = str(team_id)
    tm = by_id.get(tid) or by_id.get(team_id)
    if tm is not None:
        return tm
    if tid.isdigit():
        try:
            tm = by_id.get(int(tid))
            if tm is not None:
                return tm
        except Exception:
            pass
    # Last-resort scan: numeric vs string id mismatches in older saves.
    for key, team in by_id.items():
        if str(key) == tid:
            return team
    return None


def _trade_popup_team_abbr(session: Optional[FranchiseSession], team_id: Any) -> str:
    """Club abbreviation for wire copy; never leaks a bare numeric club id."""
    tid = str(team_id or "").strip()
    if not tid:
        return ""
    tm = _resolve_session_team(session, tid)
    if tm is None:
        return "" if tid.isdigit() else tid
    abbr = _franchise_team_abbrev(tm)
    if not abbr or abbr == "?" or abbr.isdigit():
        abbr = _team_abbr(tm, tid)
    if not abbr or abbr.isdigit():
        return "" if tid.isdigit() else tid
    return abbr


def _trade_popup_value_lookup(ex: Dict[str, Any]) -> Dict[str, float]:
    """Map asset id -> current trade value from execution value_breakdown."""
    out: Dict[str, float] = {}
    vb = dict(ex.get("value_breakdown") or {})
    for side in vb.values():
        if not isinstance(side, dict):
            continue
        for bucket in ("incoming", "outgoing"):
            for row in list(side.get(bucket) or []):
                if not isinstance(row, dict):
                    continue
                aid = str(row.get("asset_id") or row.get("id") or "")
                total = row.get("total", row.get("trade_value"))
                if not aid or total is None:
                    continue
                try:
                    out[aid] = float(total)
                except Exception:
                    continue
    return out


def _trade_popup_team_value(ex: Dict[str, Any], team_id: str) -> Optional[float]:
    vb = dict(ex.get("value_breakdown") or {})
    side = vb.get(str(team_id)) or vb.get(team_id)
    if side is None and str(team_id).isdigit():
        try:
            side = vb.get(int(team_id))
        except Exception:
            side = None
    if isinstance(side, dict) and side.get("incoming_total") is not None:
        try:
            return round(float(side.get("incoming_total")), 1)
        except Exception:
            pass
    scores = dict(ex.get("history_record") or {}).get("value_scores") or {}
    raw = scores.get(str(team_id), scores.get(team_id))
    if raw is None and str(team_id).isdigit():
        try:
            raw = scores.get(int(team_id))
        except Exception:
            raw = None
    if raw is not None:
        try:
            return round(float(raw), 1)
        except Exception:
            return None
    return None


def _structured_trade_assets_from_execution(
    ev: Dict[str, Any],
    session: Optional[FranchiseSession] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compact structured assets for Trade Wire popup consumers."""
    ex = dict(ev.get("execution") or {})
    hist = dict(ex.get("history_record") or {})
    moved = [m for m in list(ex.get("moved_assets") or []) if isinstance(m, dict)]
    if not moved:
        moved = [
            m
            for m in (list(hist.get("moved_players") or []) + list(hist.get("moved_picks") or []))
            if isinstance(m, dict)
        ]
    # Also fold in any picks that landed only under moved_picks.
    seen_ids = {
        str(m.get("asset_id") or m.get("pick_id") or m.get("player_id") or "")
        for m in moved
    }
    for m in list(hist.get("moved_picks") or []):
        if not isinstance(m, dict):
            continue
        pid = str(m.get("asset_id") or m.get("pick_id") or "")
        if pid and pid not in seen_ids:
            moved.append(m)
            seen_ids.add(pid)

    from_team = str(ev.get("from_team_id") or "")
    to_team = str(ev.get("to_team_id") or ev.get("team") or "")
    to_assets: List[Dict[str, Any]] = []
    from_assets: List[Dict[str, Any]] = []
    value_by_id = _trade_popup_value_lookup(ex)
    league = getattr(session, "league", None) if session is not None else None
    pick_reg = dict(getattr(league, "draft_pick_registry", None) or {}) if league is not None else {}

    def _push(bucket: List[Dict[str, Any]], asset: Dict[str, Any]) -> None:
        at = str(asset.get("asset_type") or asset.get("type") or "").lower()
        retained = asset.get("retained_pct")
        if at in ("player", "") and (asset.get("player_name") or asset.get("player_id") or at == "player"):
            pid = str(asset.get("asset_id") or asset.get("player_id") or "")
            acq = str(asset.get("acquiring_team_id") or "")
            player = _trade_popup_resolve_player(session, acq, pid)
            if player is None:
                player = _trade_popup_resolve_player(session, str(asset.get("source_team_id") or ""), pid)
            pos = str(asset.get("position") or "")
            ovr = None
            archetype = ""
            cap_hit = None
            years_left = None
            age = None
            if player is not None:
                pos = pos or _pos_str(player)
                ovr = _trade_popup_ovr(player)
                archetype = _trade_popup_archetype_label(player)
                cap_hit, years_left = _trade_popup_contract_bits(player)
                age = _trade_popup_age(player)
            stats = _trade_popup_season_stats(session, pid)
            tv = value_by_id.get(pid)
            bucket.append(
                {
                    "asset_type": "player",
                    "player_id": pid,
                    "display_name": str(
                        asset.get("player_name")
                        or (_name_str(player) if player is not None else "")
                        or asset.get("asset_id")
                        or "Player"
                    ),
                    "position": pos,
                    "archetype": archetype,
                    "role_line": " | ".join([p for p in (pos, archetype) if p]),
                    "age": age,
                    "ovr": ovr,
                    "cap_hit_m": round(float(cap_hit), 2) if cap_hit is not None else None,
                    "years_left": years_left,
                    "retained_salary": retained if retained else None,
                    "trade_value": round(float(tv), 1) if tv is not None else None,
                    "season_stats": stats,
                }
            )
        elif at in ("pick", "draft_pick"):
            pick_id = str(asset.get("asset_id") or asset.get("pick_id") or "")
            reg_row = dict(pick_reg.get(pick_id) or {})
            year = asset.get("year") if asset.get("year") is not None else reg_row.get("year")
            rnd = asset.get("round") if asset.get("round") is not None else reg_row.get("round")
            orig = str(
                asset.get("original_team_id")
                or reg_row.get("original_team_id")
                or ""
            )
            tv = value_by_id.get(pick_id)
            orig_abbr = _trade_popup_team_abbr(session, orig)
            display = str(asset.get("display_name") or "").strip()
            if not display:
                display = f"{year} Round {rnd}" if year and rnd else f"Pick {pick_id or '?'}"
            if orig_abbr and orig_abbr not in display:
                display = f"{display} ({orig_abbr})"
            bucket.append(
                {
                    "asset_type": "draft_pick",
                    "pick_id": pick_id,
                    "year": year,
                    "round": rnd,
                    "original_team_id": orig,
                    "display_name": display,
                    "trade_value": round(float(tv), 1) if tv is not None else None,
                }
            )
        else:
            bucket.append(
                {
                    "asset_type": at or "asset",
                    "display_name": str(asset.get("player_name") or asset.get("display_name") or asset.get("asset_id") or "Asset"),
                }
            )

    for asset in moved:
        src = str(asset.get("source_team_id") or "")
        dst = str(asset.get("acquiring_team_id") or "")
        if dst == to_team:
            _push(to_assets, asset)
        elif dst == from_team:
            _push(from_assets, asset)
        elif src == from_team:
            _push(to_assets, asset)
        elif src == to_team:
            _push(from_assets, asset)

    if not to_assets and not from_assets:
        for name in list(ev.get("outgoing") or []) or list(ev.get("players") or [])[:6]:
            to_assets.append({"asset_type": "player", "display_name": str(name)})
        for name in list(ev.get("incoming") or []):
            from_assets.append({"asset_type": "player", "display_name": str(name)})
    return to_assets[:12], from_assets[:12]


def _reason_text_from_codes(codes: List[str], fallback: str = "") -> str:
    mapping = {
        "TOP_SIX_SCORING_NEED": "Added top-six scoring for the playoff push.",
        "MIDDLE_SIX_SCORING_NEED": "Added middle-six scoring depth.",
        "CENTRE_DEPTH_NEED": "Addressed a need at centre.",
        "BOTTOM_SIX_DEPTH": "Bolstered bottom-six depth.",
        "TOP_PAIR_DEFENCE_NEED": "Upgraded the top defence pair.",
        "SECOND_PAIR_DEFENCE_NEED": "Added second-pair defence.",
        "DEFENSIVE_DEPTH_NEED": "Added defensive depth.",
        "PUCK_MOVING_DEFENCE_NEED": "Added puck-moving defence.",
        "STARTING_GOALIE_NEED": "Addressed starting goaltending.",
        "BACKUP_GOALIE_NEED": "Added goaltending depth.",
        "GOALTENDING_INSURANCE": "Added goaltending insurance.",
        "INJURY_REPLACEMENT": "Filled a hole created by injury.",
        "PLAYOFF_DEPTH": "Added playoff depth.",
        "DEADLINE_RENTAL": "Acquired a low-cost rental before the deadline.",
        "LONG_TERM_CORE_TARGET": "Targeted a longer-term core piece.",
        "CAP_EFFICIENT_UPGRADE": "Found a cap-efficient upgrade.",
        "REBUILDING_FUTURES": "Moved a veteran for future assets.",
        "PENDING_UFA_SALE": "Moved an expiring veteran for future assets.",
        "CAP_RELIEF": "Cleared cap space.",
        "CAP_COMPLIANCE": "Cleared cap space for compliance.",
        "ROSTER_SURPLUS": "Moved surplus roster depth.",
        "GOALTENDER_SURPLUS": "Moved an extra goaltender.",
        "PROSPECT_BLOCKED": "Opened a path for a younger player.",
        "AGING_VETERAN": "Moved a veteran who no longer fit the timeline.",
        "TIMELINE_MISMATCH": "Exchanged assets that better fit each timeline.",
        "POSITIONAL_SWAP": "Exchanged positional surplus for a better roster fit.",
        "ROSTER_BALANCE": "Rebalanced the roster.",
        "AGE_TIMELINE_SWAP": "Swapped assets across age timelines.",
        "SIMILAR_VALUE_DIFFERENT_NEED": "Swapped similar-value assets for different needs.",
        "PICK_VALUE_REALLOCATION": "Reallocated draft capital.",
        "DRAFT_TRADE_UP": "Moved up to select a priority organizational target.",
        "DRAFT_TRADE_DOWN": "Moved down while remaining inside the same prospect tier.",
        "EXTRA_PICK_ACCUMULATION": "Accumulated additional draft picks.",
        "DRAFT_CAPITAL_RECOVERY": "Recovered draft capital in the deal.",
        "PLAYOFF_ODDS_COLLAPSE": "Sold after playoff odds collapsed.",
        "DEEP_REBUILD_ASSET_SALE": "Moved a veteran during a deep rebuild.",
        "STAR_ACQUISITION": "Acquired a high-impact roster piece.",
        "YOUNG_PLAYER_TARGET": "Acquired a younger NHL-ready piece.",
        "RETOOLING_SWAP": "Completed a retooling roster swap.",
    }
    texts: List[str] = []
    for code in codes:
        text = mapping.get(str(code or "").upper())
        if text and text not in texts:
            texts.append(text)
    if texts:
        return " ".join(texts)
    return fallback or "Roster management trade."


def _trade_category_label(category: str) -> str:
    raw = str(category or "league_trade").replace("_", " ").strip()
    if not raw:
        return "League Trade"
    return raw.title()


def build_cpu_trade_transaction_event(
    session: FranchiseSession,
    ev: Dict[str, Any],
    *,
    calendar_idx: int,
    iso: str = "",
) -> Optional[Dict[str, Any]]:
    """Shared post-commit event builder for regular-season and draft-day CPU trades."""
    trade_id = str(ev.get("trade_id") or (ev.get("execution") or {}).get("trade_id") or "")
    if not trade_id:
        trade_id = f"cpu_trade:{calendar_idx}:{hashlib.sha1(str(ev.get('headline') or ev).encode('utf-8', 'ignore')).hexdigest()[:10]}"
    seen = getattr(session, "cpu_trade_event_seen_ids", None)
    if not isinstance(seen, set):
        seen = set(list(seen or []))
        session.cpu_trade_event_seen_ids = seen
    if trade_id in seen:
        return None
    seen.add(trade_id)

    to_team = str(ev.get("team") or ev.get("to_team_id") or "")
    from_team = str(ev.get("from_team_id") or "")
    to_tm = _resolve_session_team(session, to_team)
    from_tm = _resolve_session_team(session, from_team)
    to_abbr = _franchise_team_abbrev(to_tm) if to_tm is not None else (to_team or "?")
    from_abbr = _franchise_team_abbrev(from_tm) if from_tm is not None else (from_team or "?")
    # Never leave bare numeric club ids in the wire copy.
    if to_abbr.isdigit() and to_tm is not None:
        to_abbr = _team_abbr(to_tm, to_team) or to_abbr
    if from_abbr.isdigit() and from_tm is not None:
        from_abbr = _team_abbr(from_tm, from_team) or from_abbr
    to_name = _display_team(to_tm) if to_tm is not None else to_team
    from_name = _display_team(from_tm) if from_tm is not None else from_team

    to_assets, from_assets = _structured_trade_assets_from_execution(ev, session=session)
    to_lines, from_lines = _cpu_trade_asset_lines(ev, session=session)
    if not to_lines:
        to_lines = [str(a.get("display_name") or "") for a in to_assets if a.get("display_name")]
    if not from_lines:
        from_lines = [str(a.get("display_name") or "") for a in from_assets if a.get("display_name")]

    reason_codes = [str(c) for c in list(ev.get("reason_codes") or []) if str(c).strip()]
    reason_text = str(ev.get("reason_text") or "").strip() or _reason_text_from_codes(reason_codes)
    # Expand short single-line reasons with secondary codes when available.
    if reason_codes and len(reason_text.split()) < 12:
        expanded = _reason_text_from_codes(reason_codes, fallback=reason_text)
        if expanded:
            reason_text = expanded
    category = str(ev.get("trade_category") or "league_trade")
    importance = str(ev.get("importance") or ("major" if "major" in category else "standard"))
    profiles = dict(getattr(session, "cpu_franchise_profiles", None) or {})
    team_directions = {
        from_team: str((profiles.get(from_team) or {}).get("team_direction") or ""),
        to_team: str((profiles.get(to_team) or {}).get("team_direction") or ""),
    }
    ex = dict(ev.get("execution") or {})
    to_value = _trade_popup_team_value(ex, to_team)
    from_value = _trade_popup_team_value(ex, from_team)
    season_y = int(getattr(session, "season_calendar_year", 0) or 0)
    season_label = f"{season_y}-{(season_y + 1) % 100:02d}" if season_y else ""

    left = ", ".join(to_lines[:3]) or "assets"
    right = ", ".join(from_lines[:3]) or "assets"
    summary_line = f"{to_abbr} acquires {left} from {from_abbr} for {right}."

    details = [
        f"{to_abbr} receives: {', '.join(to_lines) if to_lines else 'Assets listed in trade log.'}",
        f"{from_abbr} receives: {', '.join(from_lines) if from_lines else 'Assets listed in trade log.'}",
        f"Context: {reason_text}",
    ]
    return {
        "id": f"cpu_trade_popup:{trade_id}",
        "kind": "storyline",
        "type": "trade",
        "event_type": "CPU_TRADE",
        "priority": "HIGH",
        "title": "Trade Completed",
        "headline": "Trade Completed",
        "summary": summary_line,
        "story_report": reason_text,
        "franchise_impact": "",
        "effect_summary": reason_text,
        "source_label": "League Trade Wire",
        "cause_type": "trade",
        "cause": reason_text,
        "team_id": to_team,
        "from_team_id": from_team,
        "team_abbrev": to_abbr,
        "team_abbr": to_abbr,
        "calendar_day": int(calendar_idx),
        "calendar_iso": str(iso or ""),
        "date": str(iso or calendar_idx),
        "season": season_y,
        "season_label": season_label,
        "trade_id": trade_id,
        "transaction_id": trade_id,
        "trade_category": category,
        "trade_type_label": _trade_category_label(category),
        "importance": importance,
        "primary_reason": reason_codes[0] if reason_codes else "",
        "reason_codes": reason_codes[:6],
        "secondary_reasons": reason_codes[1:6],
        "reason_text": reason_text,
        "team_directions": team_directions,
        "trade_value": {
            "left_team_id": to_team,
            "right_team_id": from_team,
            "left_value": to_value,
            "right_value": from_value,
            "label": "Trade Value",
        },
        "teams": [
            {
                "team_id": to_team,
                "abbreviation": to_abbr,
                "display_name": to_name,
                "acquired_assets": to_assets,
                "trade_value": to_value,
            },
            {
                "team_id": from_team,
                "abbreviation": from_abbr,
                "display_name": from_name,
                "acquired_assets": from_assets,
                "trade_value": from_value,
            },
        ],
        "is_user_team": False,
        "details": "\n".join(details),
        "draft_context": bool(ev.get("draft_context")),
        "theme": "info",
        "icon": "⇄",
    }


def _max_moved_player_trade_value(ev: Dict[str, Any]) -> float:
    """Highest player asset trade value in a CPU trade event (picks ignored)."""
    best = 0.0
    ex = dict(ev.get("execution") or {}) if isinstance(ev, dict) else {}
    vb = dict(ex.get("value_breakdown") or {})
    for side in vb.values():
        if not isinstance(side, dict):
            continue
        for bucket in ("incoming", "outgoing"):
            for row in list(side.get(bucket) or []):
                if not isinstance(row, dict):
                    continue
                at = str(row.get("asset_type") or row.get("type") or "").strip().lower()
                if at in ("pick", "draft_pick", "draftpick"):
                    continue
                # Player rows carry total/trade_value; skip empty shells.
                if at not in ("player", "") and not (
                    row.get("player_id") or row.get("player_name") or row.get("name")
                ):
                    continue
                raw = row.get("total", row.get("trade_value"))
                if raw is None:
                    continue
                try:
                    best = max(best, float(raw))
                except (TypeError, ValueError):
                    continue
    if best > 0:
        return best
    # Fallback: moved player assets + value lookup map.
    lookup = _trade_popup_value_lookup(ex)
    moved = [m for m in list(ex.get("moved_assets") or []) if isinstance(m, dict)]
    hist = dict(ex.get("history_record") or {})
    if not moved:
        moved = [m for m in list(hist.get("moved_players") or []) if isinstance(m, dict)]
    for m in moved:
        at = str(m.get("asset_type") or m.get("type") or "player").lower()
        if at in ("pick", "draft_pick", "draftpick"):
            continue
        pid = str(m.get("asset_id") or m.get("player_id") or "")
        raw = m.get("trade_value")
        if raw is None and pid:
            raw = lookup.get(pid)
        if raw is None:
            continue
        try:
            best = max(best, float(raw))
        except (TypeError, ValueError):
            continue
    return best


def _enqueue_cpu_trade_popup(session: FranchiseSession, ev: Dict[str, Any], *, calendar_idx: int, iso: str) -> None:
    """Queue a trade wire popup only when a moved player's trade value exceeds 70.

    Quieter depth / pick-heavy swaps stay in the showcase archive without a modal.
    """
    max_player_value = _max_moved_player_trade_value(ev)
    if max_player_value <= 70.0:
        try:
            popup = build_cpu_trade_transaction_event(session, ev, calendar_idx=calendar_idx, iso=iso)
            if popup:
                arch = list(getattr(session, "showcase_archive", None) or [])
                arch.append(dict(popup))
                session.showcase_archive = arch[-64:]
        except Exception:
            pass
        return
    popup = build_cpu_trade_transaction_event(session, ev, calendar_idx=calendar_idx, iso=iso)
    if not popup:
        return
    try:
        popup["max_player_trade_value"] = round(float(max_player_value), 1)
    except Exception:
        pass
    _append_unique_dict_event(session.pending_ui_popups, popup)
    # Never drop undismissed trade popups; hard ceiling only for pathological growth.
    if len(session.pending_ui_popups) > 500:
        session.pending_ui_popups = session.pending_ui_popups[:500]


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
    # Bulk calendar advance: socio/trades dominate wall time. Keep features by
    # running on a throttle (every 5th day during multi-day/season sims).
    bulk = bool(getattr(session, "_bulk_calendar_advance", False))
    if bulk and not bool(getattr(session, "_bulk_run_socio_economics", False)):
        last_run = int(getattr(session, "_bulk_socio_last_idx", -99) or -99)
        if (int(calendar_idx) - last_run) < 8:
            setattr(session, "_last_socio_tick_idx", int(calendar_idx))
            return
        setattr(session, "_bulk_socio_last_idx", int(calendar_idx))
    # Legacy audit path: skip socio only when explicitly forced off — light-mode
    # bulk (fast game sim) must still hit the throttle above.
    if bool(getattr(session, "_defer_prospect_sync", False)) and bool(
        getattr(session, "_skip_bulk_socio_economics", False)
    ):
        if not bool(getattr(session, "_bulk_run_socio_economics", False)):
            setattr(session, "_last_socio_tick_idx", int(calendar_idx))
            return
    sim = session.sim
    league = getattr(sim, "league", None)
    teams = list(getattr(league, "teams", None) or [])
    st = session.standings
    if not teams or st is None:
        return
    _refresh_cpu_franchise_profiles(session, calendar_idx=int(calendar_idx), force=False)
    try:
        setattr(league, "cpu_franchise_profiles", dict(getattr(session, "cpu_franchise_profiles", None) or {}))
        setattr(league, "cpu_scheduler_state", dict(getattr(session, "cpu_scheduler_state", None) or {}))
        # So CPU trade execution can retarget season-stat team_id for traded players.
        setattr(league, "player_season_stats", getattr(session, "player_season_stats", None))
        sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
        from app.sim_engine.trades.trade_pick_registry import upcoming_draft_year

        setattr(league, "season_year", sy)
        setattr(league, "current_season", sy)
        setattr(league, "draft_year", upcoming_draft_year(sy))
        setattr(league, "season_is_calendar", True)
    except Exception:
        pass
    rng = sim.rng
    utid = str(session.user_team_id)
    try:
        setattr(league, "_franchise_user_team_id", utid)
    except Exception:
        pass
    max_d = max(40, int(getattr(session, "nhl_regular_season_last_index", 0) or 0))
    news_tmp: List[Dict[str, Any]] = []
    ctr: Dict[str, int] = {"trade_executions": 0, "waiver_claims": 0, "major_injuries": 0}
    socio_ok = False
    try:
        from services.transcendent_tank_behavior import apply_tank_daily_behavior

        apply_tank_daily_behavior(session, sim, teams, rng, news_tmp, ctr)
        sim._season_daily_socio_economics(rng, int(calendar_idx), max_d, st, teams, news_tmp, ctr)
        socio_ok = True
    except Exception:
        socio_ok = False
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
        if str(ev2.get("type")) == "trade":
            _enqueue_cpu_trade_popup(session, ev2, calendar_idx=int(calendar_idx), iso=iso)
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
    if not socio_ok:
        logger = logging.getLogger(__name__)
        logger.warning("Daily socio-economics tick failed at calendar day %s", int(calendar_idx))
    # Canonical chemistry progression: daily familiarity / room tick for user team.
    try:
        from app.sim_engine.systems.chemistry import apply_daily_chemistry_tick  # noqa: WPS433

        user_team = session.team_by_id.get(utid) if utid else None
        if user_team is not None:
            apply_daily_chemistry_tick(user_team, session=session, rng=rng)
    except Exception:
        pass
    setattr(session, "_last_socio_tick_idx", int(calendar_idx))


def _franchise_fanout_player_storylines(session: FranchiseSession, calendar_idx: int, day_meta: Dict[str, Any]) -> None:
    from app.sim_engine.franchise.common import _franchise_fanout_player_storylines as _fanout  # noqa: WPS433

    _fanout(session, calendar_idx, day_meta)


def _maybe_roll_storyline_arc(session: FranchiseSession, day_meta: Dict[str, Any], rng: random.Random) -> None:
    """Disabled — data-driven storylines come from storyline_engine + game ledger."""
    return


def _attach_franchise_saved_lineups(
    session: FranchiseSession,
    home: Any,
    away: Any,
    *,
    home_id: str,
    away_id: str,
    user_tid: str,
) -> None:
    """Attach Edit Lines / PP / PK payloads onto the user team for game deployment."""
    for tm in (home, away):
        if tm is None:
            continue
        setattr(tm, "_franchise_saved_lines", None)
        setattr(tm, "_franchise_saved_lines_bundle", None)
        setattr(tm, "_franchise_saved_pp", None)
        setattr(tm, "_franchise_saved_pk", None)
        setattr(tm, "_franchise_deployed_lineup", None)
        setattr(tm, "_user_scratched_ids", set())

    lines_root = getattr(session, "lines", None)
    if not isinstance(lines_root, dict):
        return

    def _payload_for(unit_key: str) -> Optional[Any]:
        block = lines_root.get(unit_key)
        if not isinstance(block, dict):
            return None
        inner = block.get("lines")
        return inner if inner is not None else block

    even = _payload_for("even_strength")
    pp = _payload_for("power_play")
    pk = _payload_for("penalty_kill")
    if even is None and pp is None and pk is None:
        return

    ut = str(user_tid or "")
    target = None
    if ut and ut == str(home_id):
        target = home
    elif ut and ut == str(away_id):
        target = away
    if target is None:
        return
    if isinstance(even, dict) and (even.get("forwards") or even.get("defense") or even.get("goalies")):
        setattr(target, "_franchise_saved_lines", even)
        setattr(target, "_franchise_saved_lines_bundle", lines_root.get("even_strength"))
    if pp is not None:
        setattr(target, "_franchise_saved_pp", pp)
    if pk is not None:
        setattr(target, "_franchise_saved_pk", pk)


def _clear_franchise_saved_lineups(home: Any, away: Any) -> None:
    for tm in (home, away):
        if tm is None:
            continue
        setattr(tm, "_franchise_saved_lines", None)
        setattr(tm, "_franchise_saved_lines_bundle", None)
        setattr(tm, "_franchise_saved_pp", None)
        setattr(tm, "_franchise_saved_pk", None)
        setattr(tm, "_franchise_deployed_lineup", None)
        setattr(tm, "_user_scratched_ids", set())


def _simulate_franchise_slot(session: FranchiseSession, slot: Any) -> Tuple[Optional[str], Optional[str]]:
    """Simulate one scheduled league game. Returns (user_summary_line_or_none, league_line_or_none)."""
    sim = session.sim
    teams = list(sim.league.teams)
    r = sim.rng
    user_tid = str(session.user_team_id)
    bulk_light = bool(getattr(session, "_bulk_calendar_advance", False)) and bool(
        getattr(session, "_light_game_stat_accumulation", False)
    )

    # team_by_id is keyed by str(team_id); slots may carry int ids (dataclasses do not enforce types).
    hid = _safe_slot_team_id(slot, "home_id")
    aid = _safe_slot_team_id(slot, "away_id")
    is_user_game = user_tid in (hid, aid)
    cpu_only_bulk = bulk_light and not is_user_game
    home = session.team_by_id.get(hid)
    away = session.team_by_id.get(aid)
    if home is None or away is None:
        return None, None
    d = int(slot.day)
    cal = getattr(session, "nhl_calendar", None) or []
    cal_iso = ""
    if 0 <= int(d) < len(cal):
        cal_iso = str(cal[int(d)].get("iso") or "")
    is_playoff_slot = bool(getattr(slot, "is_playoff", False))
    stat_scope = _franchise_stat_scope(session, is_playoff=is_playoff_slot)
    stable_gid = _stable_franchise_game_id_for(
        session,
        calendar_day=int(d),
        hid=hid,
        aid=aid,
        stat_scope=stat_scope,
    )
    if stable_gid in _processed_game_ids(session):
        saved = _find_game_result_by_id(session, stable_gid)
        if isinstance(saved, dict):
            shg = int(saved.get("home_goals", saved.get("home_score", 0)) or 0)
            sag = int(saved.get("away_goals", saved.get("away_score", 0)) or 0)
            sot = bool(saved.get("overtime", saved.get("ot", False)))
            hn = (_display_team(home) or "?")[:24]
            an = (_display_team(away) or "?")[:24]
            league_line = f"{hn} {shg}-{sag} {an}{' OT' if sot else ''}"
            user_line: Optional[str] = None
            if hid == user_tid or aid == user_tid:
                opp = away if hid == user_tid else home
                won = (shg > sag) if hid == user_tid else (sag > shg)
                gs = f"{shg}-{sag}" + (" OT" if sot else "")
                user_line = f"{'W' if won else 'L'} vs {_display_team(opp)} ({gs}) - calendar day {d}"
            return user_line, league_line

    h_goal = _ensure_goalie_for_game(home)
    a_goal = _ensure_goalie_for_game(away)
    if int(h_goal["total"]) <= 0 or int(a_goal["total"]) <= 0:
        _fr_dbg(f"goalie availability failure on day {d}: {hid} total={h_goal['total']} {aid} total={a_goal['total']}")
        missing = []
        if int(h_goal["total"]) <= 0:
            missing.append(_display_team(home) or hid)
        if int(a_goal["total"]) <= 0:
            missing.append(_display_team(away) or aid)
        _franchise_enqueue_critical_notice(
            session,
            title="Roster integrity issue",
            description=(
                "A scheduled game has no listed goalie on one side "
                f"({', '.join(missing)}). Resolve roster integrity before advancing."
            ),
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

    def _team_b2b(team_id: str) -> bool:
        play_days = getattr(session, "play_days", None) or {}
        days = play_days.get(team_id, set())
        if not days or world_calendar is None:
            return int(d) - 1 in days
        return bool(world_calendar.is_back_to_back(days, d))

    hb2b = _team_b2b(hid)
    ab2b = _team_b2b(aid)

    try:
        if not cpu_only_bulk:
            from app.sim_engine.franchise.storyline_stat_bridge import (  # noqa: WPS433
                prime_franchise_game_stat_modifiers,
            )

            prime_franchise_game_stat_modifiers(
                session,
                sim,
                hid,
                aid,
                game_meta={
                    "game_id": stable_gid,
                    "calendar_day": d,
                    "calendar_iso": cal_iso,
                },
            )
    except Exception:
        pass

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
        try:
            if not cpu_only_bulk:
                from app.sim_engine.franchise.storyline_coverage import apply_matchup_to_scales  # noqa: WPS433

                h_scale, a_scale, _ = apply_matchup_to_scales(
                    session,
                    hid,
                    aid,
                    h_scale,
                    a_scale,
                    {"game_id": f"{hid}_{aid}_{d}", "is_playoff": is_playoff_slot},
                )
        except Exception:
            pass

        base_noise = 1.0 + 0.22 * (session.chaos_index - 0.5)
        nh = world_chemistry.chemistry_chaos_dampen(home, base_noise)
        na = world_chemistry.chemistry_chaos_dampen(away, base_noise)
        _, ih = sim._identity_runner_strength_noise_factors(home)
        _, ia = sim._identity_runner_strength_noise_factors(away)
        noise_scale = 0.5 * (nh + na) * (0.5 * (ih + ia))

        if not cpu_only_bulk:
            world_fatigue.tick_roster_fatigue_for_game(home, r, hb2b, session.schedule, d, hid)
            world_fatigue.tick_roster_fatigue_for_game(away, r, ab2b, session.schedule, d, aid)
            _attach_franchise_saved_lineups(
                session, home, away, home_id=hid, away_id=aid, user_tid=user_tid
            )
        # Bulk/light season accumulation must use one counting model for every club.
        # Forcing the user onto full event sim while CPU–CPU stays light systematically
        # under-scored the controlled team vs the rest of the league.
        use_light = bool(getattr(session, "_light_game_stat_accumulation", False))
        try:
            hg, ag, ot = sim._simulate_game(
                r,
                home,
                away,
                session.strength_map,
                home_strength_scale=h_scale,
                away_strength_scale=a_scale,
                noise_scale=noise_scale,
                is_playoff=is_playoff_slot,
                calendar_day=int(d),
                home_b2b=hb2b,
                away_b2b=ab2b,
                light_mode=use_light,
            )
        finally:
            _clear_franchise_saved_lineups(home, away)

        world_momentum.update_momentum_after_game(home, hg, ag, r)
        world_momentum.update_momentum_after_game(away, ag, hg, r)
        if not cpu_only_bulk:
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
            tid_tm = str(getattr(tm, "team_id", None) or getattr(tm, "id", "") or "")
            for pl in getattr(tm, "roster", None) or []:
                if int(getattr(pl, "_world_injury_games_remaining", 0) or 0) > 0:
                    world_injuries.tick_games_missed(pl)
                if cpu_only_bulk:
                    continue
                has_conduct = bool(getattr(pl, "_conduct_incident_id", None)) or int(
                    getattr(pl, "_world_conduct_games_remaining", 0) or 0
                ) > 0
                if has_conduct:
                    try:
                        from app.sim_engine.franchise.conduct_incidents import (  # noqa: WPS433
                            apply_dress_backlash,
                            player_eligible_to_dress,
                            tick_incident_games,
                        )

                        tick_incident_games(session, pl)
                        if player_eligible_to_dress(pl, session) and float(
                            getattr(pl, "_conduct_dress_backlash_risk", 0) or 0
                        ) >= 0.08:
                            apply_dress_backlash(session, team_id=tid_tm, player=pl)
                    except Exception:
                        try:
                            from app.sim_engine.franchise.storyline_conduct import (  # noqa: WPS433
                                tick_conduct_games_missed,
                            )

                            tick_conduct_games_missed(pl)
                        except Exception:
                            pass

        if getattr(session, "injuries_enabled", True) and not cpu_only_bulk:
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
        try:
            from app.sim_engine.franchise.storyline_coverage import apply_matchup_to_scales  # noqa: WPS433

            h_inj, a_inj, _ = apply_matchup_to_scales(
                session,
                hid,
                aid,
                h_inj,
                a_inj,
                {"game_id": f"{hid}_{aid}_{d}", "is_playoff": is_playoff_slot},
            )
        except Exception:
            pass
        _attach_franchise_saved_lineups(
            session, home, away, home_id=hid, away_id=aid, user_tid=user_tid
        )
        try:
            hg, ag, ot = sim._simulate_game(
                r,
                home,
                away,
                session.strength_map,
                home_strength_scale=h_inj,
                away_strength_scale=a_inj,
                noise_scale=id_noise,
                is_playoff=is_playoff_slot,
                calendar_day=int(d),
                home_b2b=hb2b,
                away_b2b=ab2b,
                light_mode=bool(getattr(session, "_light_game_stat_accumulation", False)),
            )
        finally:
            _clear_franchise_saved_lineups(home, away)

        if world_injuries is not None:
            for tm in (home, away):
                tid_tm = str(getattr(tm, "team_id", None) or getattr(tm, "id", "") or "")
                for pl in getattr(tm, "roster", None) or []:
                    if int(getattr(pl, "_world_injury_games_remaining", 0) or 0) > 0:
                        world_injuries.tick_games_missed(pl)
                    has_conduct = bool(getattr(pl, "_conduct_incident_id", None)) or int(
                        getattr(pl, "_world_conduct_games_remaining", 0) or 0
                    ) > 0
                    if has_conduct:
                        try:
                            from app.sim_engine.franchise.conduct_incidents import (  # noqa: WPS433
                                apply_dress_backlash,
                                player_eligible_to_dress,
                                tick_incident_games,
                            )

                            tick_incident_games(session, pl)
                            if player_eligible_to_dress(pl, session) and float(
                                getattr(pl, "_conduct_dress_backlash_risk", 0) or 0
                            ) >= 0.08:
                                apply_dress_backlash(session, team_id=tid_tm, player=pl)
                        except Exception:
                            try:
                                from app.sim_engine.franchise.storyline_conduct import (  # noqa: WPS433
                                    tick_conduct_games_missed,
                                )

                                tick_conduct_games_missed(pl)
                            except Exception:
                                pass
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

    box = _accumulate_franchise_game_stats(
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
        is_playoff=is_playoff_slot,
        home_b2b=hb2b,
        away_b2b=ab2b,
    )
    if not isinstance(box, dict):
        box = {}

    session.standings.record_game(
        slot.home_id,
        slot.away_id,
        hg,
        ag,
        overtime=ot,
        shootout=bool(box.get("shootout", False)),
        stats_home_goals=int(box.get("player_home_goals", box.get("home_goals", hg)) or 0),
        stats_away_goals=int(box.get("player_away_goals", box.get("away_goals", ag)) or 0),
    )
    

    if cpu_only_bulk:
        try:
            from app.sim_engine.franchise.storyline_stat_bridge import (  # noqa: WPS433
                clear_franchise_game_stat_modifiers,
            )

            clear_franchise_game_stat_modifiers(sim)
        except Exception:
            pass
        return None, None

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

    try:
        from app.sim_engine.franchise.storyline_stat_bridge import (  # noqa: WPS433
            clear_franchise_game_stat_modifiers,
        )

        clear_franchise_game_stat_modifiers(sim)
    except Exception:
        pass

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
    bulk_light = bool(getattr(session, "_bulk_calendar_advance", False)) and bool(
        getattr(session, "_light_game_stat_accumulation", False)
    )

    expected_keys: set = set()
    completed_keys: set = set()
    for slot in slots or []:
        hid = _safe_slot_team_id(slot, "home_id")
        aid = _safe_slot_team_id(slot, "away_id")

        if hid and aid:
            expected_keys.add((hid, aid))

        ul, ll = _simulate_franchise_slot(session, slot)

        if hid and aid:
            completed_keys.add((hid, aid))

        if ul:
            lines.append(ul)

        if ll:
            league_lines.append(ll)

    if bulk_light:
        missing = sorted(expected_keys - completed_keys)
        if missing:
            raise RuntimeError(
                f"Game result integrity error on calendar day {calendar_day}: "
                f"{len(missing)} scheduled game(s) did not produce a valid final result. "
                f"First missing: {missing[0][0]} vs {missing[0][1]}"
            )
        return lines, league_lines

    # Verify result store has a valid final for every scheduled slot.
    all_results = getattr(session, "game_results", None) or []
    saved_for_day = [
        g
        for g in all_results
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


PROSPECT_SYNC_THROTTLE_DAYS = 7


def _prospect_sync_should_run(session: FranchiseSession, *, force: bool = False) -> bool:
    """Return True when draft-age prospect stats should advance for this calendar step."""
    if force:
        return True
    if bool(getattr(session, "_defer_prospect_sync", False)):
        return False
    last_idx = getattr(session, "_prospect_sync_throttle_index", None)
    if last_idx is None:
        return True
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    throttle = int(getattr(session, "_prospect_sync_throttle_days", PROSPECT_SYNC_THROTTLE_DAYS) or PROSPECT_SYNC_THROTTLE_DAYS)
    return (cur - int(last_idx)) >= max(1, throttle)


def ensure_prospect_stats_current_for_scouting(session: FranchiseSession) -> None:
    """Bring prospect stats current for scouting/draft UI without always force-syncing.

    Force only when the board is meaningfully behind the calendar (or never synced).
    Same-day / within-throttle opens reuse the last sync — huge win after bulk season.
    """
    iso = _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
    last_iso = str(getattr(session, "_prospect_stats_synced_iso", "") or "")
    if last_iso and iso and last_iso == str(iso):
        return
    # If we synced within the normal throttle window, don't re-run the full pool.
    last_idx = getattr(session, "_prospect_sync_throttle_index", None)
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    throttle = int(
        getattr(session, "_prospect_sync_throttle_days", PROSPECT_SYNC_THROTTLE_DAYS)
        or PROSPECT_SYNC_THROTTLE_DAYS
    )
    if last_idx is not None and (cur - int(last_idx)) < max(1, throttle) and last_iso:
        return
    _sync_prospect_stats_to_calendar(session, force=True)


def _sync_prospect_stats_to_calendar(session: FranchiseSession, *, force: bool = False) -> int:
    """Advance draft-age prospect league stats to the franchise calendar date (delta only)."""
    sim = getattr(session, "sim", None)
    if sim is None:
        return 0
    # One-shot: force a full delta pass so early-season boards drop leaked
    # prior-year GP after the season-year / calendar repair.
    if not bool(getattr(session, "_prospect_stale_gp_repair_v1", False)):
        try:
            session._prospect_stats_synced_iso = ""
            session._prospect_sync_rows = None
            session._prospect_stale_gp_repair_v1 = True
            force = True
        except Exception:
            pass
    # Existing franchises that already generated Y2 without the draft-class roll
    # still need junior aging + undrafted depth (inject-only left ages stuck at 16
    # with NHL-starter OVRs).
    try:
        sy = int(getattr(session, "season_calendar_year", 0) or 0)
        if sy and int(getattr(session, "_draft_class_roll_year", 0) or 0) != sy:
            from services.franchise_offseason import _roll_development_league_draft_class

            _roll_development_league_draft_class(session, sy)
            session._draft_class_roll_year = sy
    except Exception:
        pass
    # One-shot heal for saves that already injected 16yos at ~0.72–0.86 OVR.
    try:
        from services.franchise_offseason import _retune_inflated_underage_prospects

        _retune_inflated_underage_prospects(session)
    except Exception:
        pass
    if not _prospect_sync_should_run(session, force=force):
        return 0
    iso = _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
    if not iso:
        return 0
    last_iso = str(getattr(session, "_prospect_stats_synced_iso", "") or "")
    needs_scoring_retune = not bool(getattr(session, "_prospect_retune_v4_applied", False))
    try:
        from app.sim_engine.generation.prospect_league_scoring import advance_all_development_league_stats

        cache_key = (
            f"{int(getattr(session, 'season_calendar_year', 2025) or 2025)}"
            f"|{len(getattr(getattr(sim, 'league', None), 'development_leagues', []) or [])}"
            f"|ahl_echl_v1"
        )
        cached_key = str(getattr(session, "_prospect_sync_cache_key", "") or "")
        rows = getattr(session, "_prospect_sync_rows", None)
        if cached_key != cache_key or not isinstance(rows, list):
            rows = []
            league = getattr(sim, "league", None)
            for block in getattr(league, "development_leagues", None) or []:
                code = str(block.get("league_code") or "")
                for tm in block.get("teams") or []:
                    for p in tm.get("players") or []:
                        if getattr(p, "retired", False):
                            continue
                        ident = getattr(p, "identity", None)
                        age = int(getattr(ident, "age", 99) or 99) if ident else 99
                        if age <= 20:
                            rows.append((p, code))
            # Affiliate minors advance on the same calendar with AHL/ECHL scoring profiles.
            for tm in getattr(league, "teams", None) or []:
                for p in getattr(tm, "ahl_roster", None) or []:
                    if getattr(p, "retired", False):
                        continue
                    ident = getattr(p, "identity", None)
                    age = (
                        int(getattr(ident, "age", 99) or 99)
                        if ident
                        else int(getattr(p, "age", 99) or 99)
                    )
                    if age <= 23:
                        rows.append((p, "AHL"))
                for p in getattr(tm, "echl_roster", None) or []:
                    if getattr(p, "retired", False):
                        continue
                    ident = getattr(p, "identity", None)
                    age = (
                        int(getattr(ident, "age", 99) or 99)
                        if ident
                        else int(getattr(p, "age", 99) or 99)
                    )
                    if age <= 23:
                        rows.append((p, "ECHL"))
            session._prospect_sync_rows = rows
            session._prospect_sync_cache_key = cache_key

        # Always run advance so scoring retunes can apply even when calendar ISO is unchanged.
        n = advance_all_development_league_stats(
            sim,
            iso,
            season_year=int(getattr(session, "season_calendar_year", 2025) or 2025),
            rng=getattr(sim, "rng", None),
            prospect_rows=rows,
        )
        session._prospect_stats_synced_iso = str(iso)
        session._prospect_sync_throttle_index = int(getattr(session, "calendar_cursor", 0) or 0)
        if last_iso != str(iso) or needs_scoring_retune:
            session._prospect_retune_v4_applied = True
            if bool(getattr(session, "_defer_payload_invalidation", False)):
                session._pending_prospect_revision_bump = True
            else:
                invalidate_session_payload_caches(session, reason="prospect_stats")
                _bump_prospect_revision(session)
        return int(n)
    except Exception:
        return 0


def _depth_pool_progression_tick(session: FranchiseSession) -> None:
    """Periodic progression pass on non-NHL depth (one main development result per season)."""
    from app.sim_engine.progression import run_player_progression

    league = getattr(session.sim, "league", None)
    if league is None:
        return
    rng = session.sim.rng
    season_id = int(getattr(session, "season_calendar_year", 2025) or 2025)
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
    fa_ids = {
        str(getattr(p, "id", "") or "")
        for pool_attr in ("free_agents", "overseas_free_agents")
        for p in getattr(league, pool_attr, None) or []
    }
    rng.shuffle(pool)
    for p in pool[: min(72, len(pool))]:
        try:
            setattr(p, "_active_dev_season", season_id)
            _, retired = run_player_progression(
                p, rng, season_id=season_id, source_path="depth_pool"
            )
            if retired:
                setattr(p, "retired", True)
                _purge_retired_from_extra_pools(session, p)
            elif str(getattr(p, "id", "") or "") in fa_ids:
                # Free-agent-only: a rare, small late-career craft gain so older FAs are not
                # locked into pure decline. Once per season, does not touch rostered players.
                _maybe_free_agent_late_career_gain(p, rng, season_id)
                # Progression may have moved OVR — refresh market stock/asking at this checkpoint.
                _recompute_free_agent_stock(p, session)
        except Exception:
            pass


# Late-career "craft" attributes that can still sharpen after age 32 (mental game,
# positioning, consistency) — the parts of a player that age well.
_FA_LATE_CRAFT_SKATER = (
    "iqm_hockey_iq", "iqm_game_sense", "iqm_awareness", "iqm_composure", "iqm_consistency",
    "def_body_positioning", "def_defensive_iq", "off_offensive_awareness", "def_faceoffs",
)
_FA_LATE_CRAFT_GOALIE = (
    "g_positioning", "g_rebound_control_g", "iqm_composure", "iqm_consistency", "iqm_awareness",
)


def _maybe_free_agent_late_career_gain(p: Any, rng: random.Random, season_id: int) -> None:
    """Uncommon, small late-career craft improvement for unsigned players aged 32+.

    Guarded to once per season per player. Bumps a handful of mental/positioning attributes
    (which age well) by +1 so a veteran who ages gracefully can still tick up in OVR despite
    normal decline. Rare and small; recomputes OVR through the existing pipeline."""
    if _player_age_int(p) < 32:
        return
    if int(getattr(p, "_fa_latecareer_season", -1) or -1) == int(season_id):
        return
    setattr(p, "_fa_latecareer_season", int(season_id))
    if rng.random() >= 0.15:  # ~15% of veteran FAs per season
        return
    ratings = getattr(p, "ratings", None)
    if not isinstance(ratings, dict) or not ratings:
        return
    pool = _FA_LATE_CRAFT_GOALIE if _pos_str(p) == "G" else _FA_LATE_CRAFT_SKATER
    craft_keys = [k for k in pool if k in ratings]
    if not craft_keys:
        return
    rng.shuffle(craft_keys)
    for key in craft_keys[:5]:
        cur = int(float(ratings.get(key, 50) or 50))
        ratings[key] = min(92, cur + 1)
    try:
        from app.sim_engine.entities.player import persist_recomputed_ovr

        persist_recomputed_ovr(p)
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
    light_bulk = bool(getattr(session, "_light_game_stat_accumulation", False))

    iso = str(day_meta.get("iso") or "")
    ui_phase = str(day_meta.get("ui_phase") or "")
    total_reg_days = int(getattr(session, "nhl_regular_season_last_index", 0) or 0) + 1
    day_label = f"{iso} ┬╖ {ui_phase} ┬╖ league day {int(day_ordinal)} / {total_reg_days}"
    if not light_bulk:
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
    bulk = bool(getattr(session, "_bulk_calendar_advance", False))
    if not light_bulk and not bulk:
        try:
            from app.sim_engine.franchise.storyline_engine import franchise_record_data_storylines  # noqa: WPS433

            franchise_record_data_storylines(session, just_idx, day_meta, rng=session.sim.rng)
        except Exception:
            pass
        try:
            from app.sim_engine.franchise.storyline_engine import franchise_cause_storyline_daily_pass  # noqa: WPS433

            franchise_cause_storyline_daily_pass(session, just_idx, day_meta, rng=session.sim.rng)
        except Exception:
            pass
        try:
            _decay_all_fan_heat(session, just_idx)
        except Exception:
            pass
        _franchise_fanout_player_storylines(session, just_idx, day_meta)
        try:
            from services.trade_demand_engine import process_trade_demand_day

            process_trade_demand_day(session, just_idx, day_meta)
        except Exception:
            pass
        try:
            from app.sim_engine.franchise.common import _franchise_tick_conduct_and_resolve  # noqa: WPS433

            _franchise_tick_conduct_and_resolve(session, just_idx, day_meta)
        except Exception:
            pass

        _maybe_enqueue_post_day_decisions(session, user_lines)
    elif not light_bulk and bulk:
        # Legacy path (non-light bulk) — narrative throttled every 3rd day.
        if int(session.calendar_days_finished) % 3 == 0:
            try:
                from app.sim_engine.franchise.storyline_engine import franchise_record_data_storylines  # noqa: WPS433

                franchise_record_data_storylines(session, just_idx, day_meta, rng=session.sim.rng)
            except Exception:
                pass
            try:
                from app.sim_engine.franchise.storyline_engine import franchise_cause_storyline_daily_pass  # noqa: WPS433

                franchise_cause_storyline_daily_pass(session, just_idx, day_meta, rng=session.sim.rng)
            except Exception:
                pass
            _maybe_enqueue_post_day_decisions(session, user_lines)
    # Bulk light advance: skip narrative/trade-demand passes (ledger stats unchanged).
    try:
        from app.sim_engine.league_hierarchy_bootstrap import tick_extra_league_development

        # Thousands of minor-league rating ticks — defer during bulk, catch up once at end.
        if not bulk:
            tick_extra_league_development(session.sim, session.sim.rng)
    except Exception:
        pass
    try:
        # Prospect stats catch up once at bulk end (see advance_franchise_bulk finally).
        if not bulk:
            _sync_prospect_stats_to_calendar(session)
    except Exception:
        pass
    if not bulk and int(session.calendar_days_finished) % 5 == 0:
        _depth_pool_progression_tick(session)

    if not bulk and int(session.calendar_days_finished) % 8 == 0:
        try:
            _nhl_in_season_development_tick(session)
        except Exception:
            pass

    if not bulk and int(session.calendar_days_finished) % 30 == 0:
        try:
            from services.contract_economy import run_cpu_in_season_free_agency

            run_cpu_in_season_free_agency(session)
        except Exception:
            pass

    # Living FA market: during bulk, accumulate days and tick once at bulk end (see advance_franchise_bulk).
    try:
        if bool(getattr(session, "free_agency_open", False)):
            if bulk:
                session._bulk_fa_days_pending = int(getattr(session, "_bulk_fa_days_pending", 0) or 0) + 1
            else:
                from services.fa_market_engine import tick_free_agency_market
                from services.franchise_offseason import _open_free_agency

                tick = tick_free_agency_market(session, days=1)
                session._last_fa_market_tick = tick
                _open_free_agency(session, force=False)
        elif not light_bulk and not bulk and str(getattr(session, "phase", "") or "") == "regular":
            # Sparse in-season FA chatter between the heavier 30-day waves
            if int(session.calendar_days_finished) % 5 == 0:
                league = getattr(session.sim, "league", None)
                pool = list(getattr(league, "free_agents", None) or []) if league else []
                if pool:
                    from services.fa_market_engine import ensure_fa_market_book, tick_free_agency_market

                    ensure_fa_market_book(session)
                    tick_free_agency_market(
                        session,
                        days=1,
                        max_signings_per_day=1,
                        max_offers_per_day=4,
                    )
    except Exception:
        pass

    _maybe_enqueue_wjc_loan_decisions(session, day_meta)
    _maybe_enqueue_showcase_popups(session, day_meta)
    _maybe_roll_storyline_arc(session, day_meta, session.sim.rng)


def _split_preseason_from_regular_if_needed(session: FranchiseSession, day_meta: Dict[str, Any]) -> None:
    """At first regular-season day, snapshot preseason stats and reset regular-season counters."""
    if str(day_meta.get("segment") or "") != "regular":
        return
    # Always flip phase once the calendar is on a regular day — bulk/day sims used
    # to leave phase stuck on "preseason", which disabled REG SEASON and blocked
    # the normal playoff handoff.
    if str(getattr(session, "phase", "") or "") == "preseason":
        session.phase = "regular"
        session.season_phase = "regular"
        session.next_important_event = ""
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
    session.processed_game_ids = set()
    session.timeline.append("REGULAR SEASON: preseason stats archived; regular-season records reset.")
    setattr(session, "_regular_stats_split_done", True)
    try:
        _snapshot_season_start_ovrs(session)
    except Exception:
        pass


def _snapshot_season_start_ovrs(session: FranchiseSession) -> None:
    """Freeze display OVR at regular-season open for cumulative growth / roster +/-."""
    try:
        from app.sim_engine.franchise.storyline_conduct import get_base_ovr_display
    except Exception:
        get_base_ovr_display = None  # type: ignore
    league = getattr(getattr(session, "sim", None), "league", None)
    for tm in list(getattr(league, "teams", None) or []):
        pools = (
            getattr(tm, "roster", None) or [],
            getattr(tm, "ahl_roster", None) or [],
            getattr(tm, "echl_roster", None) or [],
        )
        for pool in pools:
            for p in pool:
                if p is None or getattr(p, "retired", False):
                    continue
                try:
                    if get_base_ovr_display is not None:
                        ov = int(get_base_ovr_display(p))
                    else:
                        ov = int(round(_player_ovr99(p)))
                    setattr(p, "season_start_ovr", ov)
                    setattr(p, "_season_start_ovr", ov)
                    setattr(p, "_in_season_growth_spent_01", 0.0)
                    setattr(p, "_in_season_ovr_delta_accum", 0.0)
                except Exception:
                    continue


def _nhl_in_season_development_tick(session: FranchiseSession) -> int:
    """Periodic real OVR drift for NHL (and depth) rosters during the regular season."""
    if str(getattr(session, "phase", "") or "") not in ("regular", "playoffs", ""):
        # Allow during regular; skip pure offseason.
        phase = str(getattr(session, "phase", "") or "")
        if phase in ("offseason", "preseason", "post_cup"):
            return 0
    if not bool(getattr(session, "_regular_stats_split_done", False)):
        return 0

    from app.sim_engine.progression.development import apply_in_season_development_pulse

    sim = getattr(session, "sim", None)
    league = getattr(sim, "league", None)
    if league is None:
        return 0
    rng = getattr(sim, "rng", None) or random.Random()
    moved = 0

    try:
        from services.franchise_offseason import _dev_stamp_season_production
    except Exception:
        _dev_stamp_season_production = None  # type: ignore

    for tm in list(getattr(league, "teams", None) or []):
        for p in list(getattr(tm, "roster", None) or []):
            if p is None or getattr(p, "retired", False):
                continue
            # Backfill season_start if save started mid-season.
            if getattr(p, "season_start_ovr", None) is None:
                try:
                    from app.sim_engine.franchise.storyline_conduct import get_base_ovr_display

                    ov = int(get_base_ovr_display(p))
                    accum = float(getattr(p, "_in_season_ovr_delta_accum", 0.0) or 0.0)
                    start = int(round(ov - accum))
                    setattr(p, "season_start_ovr", start)
                    setattr(p, "_season_start_ovr", start)
                except Exception:
                    pass
            try:
                if _dev_stamp_season_production is not None:
                    _dev_stamp_season_production(session, p)
            except Exception:
                pass
            try:
                d = apply_in_season_development_pulse(p, rng)
                if abs(float(d or 0)) >= 0.15:
                    moved += 1
            except Exception:
                continue

    if moved:
        try:
            session._cached_roster_browser_payload = None
        except Exception:
            pass
    return moved


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
        (("switzerland", "swiss"), "SUI"),
        (("denmark", "danish"), "DEN"),
        (("latvia", "latv"), "LAT"),
    ]
    for hints, code in pairs:
        if any(h in bc for h in hints):
            return code
    return ""


def _wjc_countries_meta() -> List[Tuple[str, str]]:
    """IIHF World Juniors field — national U20 programs only."""
    return [
        ("CAN", "Canada"),
        ("CZE", "Czechia"),
        ("DEN", "Denmark"),
        ("FIN", "Finland"),
        ("GER", "Germany"),
        ("LAT", "Latvia"),
        ("SVK", "Slovakia"),
        ("SWE", "Sweden"),
        ("SUI", "Switzerland"),
        ("USA", "United States"),
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


def _wjc_user_team_abbr(session: FranchiseSession) -> str:
    ut = session.team_by_id.get(str(session.user_team_id))
    if ut is None:
        return ""
    for attr in ("abbreviation", "abbr"):
        val = getattr(ut, attr, None)
        if val:
            return str(val).upper()[:8]
    name = str(getattr(ut, "name", "") or "")
    return name[:3].upper() if name else ""


def _wjc_find_roster_player(session: FranchiseSession, player_id: str) -> Any:
    pid = str(player_id or "")
    ut = session.team_by_id.get(str(session.user_team_id))
    if ut is None:
        return None
    for roster_name in ("ahl_roster", "roster"):
        for p in getattr(ut, roster_name, None) or []:
            if str(getattr(p, "id", "") or "") == pid:
                return p
    return None


def _wjc_enrich_prospect_row(
    session: FranchiseSession,
    row: Dict[str, Any],
    by_key: Dict[str, Dict[str, Any]],
    rank_by_key: Dict[str, int],
    ut_abbr: str,
) -> Dict[str, Any]:
    pid = str(row.get("player_id") or "")
    entry = by_key.get(pid)
    player = _wjc_find_roster_player(session, pid)
    try:
        before_rank = int(row.get("stock_rank_before")) if row.get("stock_rank_before") is not None else None
    except (TypeError, ValueError):
        before_rank = None
    ovr01 = _player_ovr01(player) if player else max(
        0.45,
        min(0.9, 1.0 - ((before_rank if before_rank is not None else 120) / 250.0)),
    )
    pos = str(row.get("position") or (_pos_str(player) if player else "F"))[:3].upper()

    if entry:
        rank = int(entry.get("rank") or rank_by_key.get(pid) or 999)
        row.update(
            {
                "draft_prospect_id": pid,
                "prospect_classification": "draft_eligible",
                "stock_rank_before": rank,
                "stock_rank_after": rank,
                "stock_delta": 0,
                "position": str(entry.get("position") or pos)[:3].upper(),
                "age": int(entry.get("age") or row.get("age") or 19),
                "nationality": str(entry.get("nationality") or row.get("nationality") or ""),
                "ovr": max(0.45, min(0.95, float(entry.get("true_ovr") or ovr01 * 99.0) / 99.0)),
                "junior_league": str(entry.get("league") or entry.get("league_name") or ""),
                "junior_team": str(entry.get("team") or entry.get("team_name") or ""),
                "junior_gp": int(entry.get("gp") or entry.get("games_played") or 0),
                "junior_g": int(entry.get("goals") or entry.get("g") or 0),
                "junior_a": int(entry.get("assists") or entry.get("a") or 0),
                "junior_pts": int(entry.get("points") or entry.get("pts") or 0),
                "scouting_confidence": entry.get("scouting_confidence"),
            }
        )
    elif bool(row.get("is_user_prospect")):
        row.update(
            {
                "draft_prospect_id": None,
                "prospect_classification": "drafted_user",
                "stock_rank_before": None,
                "stock_rank_after": None,
                "stock_delta": None,
                "owner_team_abbr": ut_abbr or "YOU",
                "position": pos,
                "ovr": ovr01,
            }
        )
    return row


def _wjc_hud_event_extras(session: FranchiseSession, now_iso: str, season_year: int) -> Dict[str, Any]:
    sy = int(season_year)
    evaluated = getattr(session, "wjc_stock_evaluated_seasons", None) or set()
    bundle = getattr(session, "wjc_tournament_bundle", None) or {}
    if sy in evaluated or bool(bundle.get("stock_evaluated")):
        return {
            "status": "complete",
            "phase": "complete",
            "display_override": "COMPLETE",
            "archive_available": True,
        }
    d_idx = _wjc_day_index_for_iso(str(now_iso or "")[:10], sy)
    if d_idx is not None:
        return {
            "status": "live",
            "phase": "live",
            "display_override": "LIVE",
            "wjc_day": d_idx + 1,
            "archive_available": False,
        }
    arch = list(getattr(session, "showcase_archive", None) or [])
    has_arch = any(
        isinstance(a, dict) and (a.get("kind") == "wjc_tournament" or a.get("wjc_phase"))
        for a in arch
    )
    return {
        "status": "upcoming",
        "phase": "upcoming",
        "archive_available": has_arch,
    }


def _persist_wjc_stock_to_draft_class(
    session: FranchiseSession,
    sim: Any,
    prospects_stocked: List[Dict[str, Any]],
    *,
    season_sy: int,
    cal_iso: str = "",
) -> Dict[str, Dict[str, int]]:
    """Apply WJC evaluation to persistent draft rankings exactly once per season."""
    evaluated = getattr(session, "wjc_stock_evaluated_seasons", None)
    if not isinstance(evaluated, set):
        session.wjc_stock_evaluated_seasons = set()
        evaluated = session.wjc_stock_evaluated_seasons
    if int(season_sy) in evaluated:
        return {}

    bundle = getattr(session, "wjc_tournament_bundle", None) or {}
    if bool(bundle.get("stock_evaluated")):
        evaluated.add(int(season_sy))
        return {}

    invalidate_session_payload_caches(session, reason="wjc_pre")
    board_before = build_draft_class_rankings(session, sim)
    entries_before = {
        str(e.get("key")): int(e.get("rank") or 0)
        for e in (board_before.get("entries") or [])
        if isinstance(e, dict) and e.get("key")
    }

    boosts = dict(getattr(session, "wjc_draft_score_boosts", None) or {})
    rank_changes: Dict[str, Dict[str, int]] = {}

    for p in prospects_stocked:
        if str(p.get("prospect_classification") or "") != "draft_eligible":
            continue
        if bool(p.get("is_npc")):
            continue
        key = str(p.get("draft_prospect_id") or p.get("player_id") or "")
        if not key or key not in entries_before:
            continue
        try:
            stock_delta = int(p.get("stock_delta") or 0)
        except (TypeError, ValueError):
            stock_delta = 0
        if stock_delta == 0:
            continue
        rank_before = int(entries_before.get(key) or p.get("stock_rank_before") or 999)
        boosts[key] = float(boosts.get(key, 0.0)) + float(stock_delta) * 0.35
        rank_changes[key] = {"before": rank_before, "stock_delta": stock_delta}

    if not rank_changes:
        evaluated.add(int(season_sy))
        bundle["stock_evaluated"] = True
        session.wjc_tournament_bundle = bundle
        return {}

    session.wjc_draft_score_boosts = boosts
    evaluated.add(int(season_sy))
    session.wjc_stock_evaluated_seasons = evaluated
    bundle["stock_evaluated"] = True
    session.wjc_tournament_bundle = bundle

    invalidate_session_payload_caches(session, reason="wjc_post")
    board_after = build_draft_class_rankings(session, sim)
    entries_after = {
        str(e.get("key")): int(e.get("rank") or 0)
        for e in (board_after.get("entries") or [])
        if isinstance(e, dict) and e.get("key")
    }

    from services.franchise_entry_draft import append_stock_history_snapshot

    date_label = str(cal_iso)[:10] if cal_iso else ""

    for key, info in rank_changes.items():
        before = int(info["before"])
        after = int(entries_after.get(key) or before)
        delta = before - after
        info["after"] = after
        info["delta"] = delta
        entry_row = next(
            (e for e in (board_after.get("entries") or []) if str(e.get("key")) == key),
            None,
        )
        if entry_row:
            hist_row = dict(entry_row)
            hist_row["rank"] = after
            if delta > 0:
                hist_row["stock_label"] = "Rocketing" if delta >= 5 else "Rising"
                hist_row["stock_reason"] = f"World Juniors breakout — up {delta} spots"
            elif delta < 0:
                hist_row["stock_label"] = "Crashing" if delta <= -5 else "Falling"
                hist_row["stock_reason"] = f"World Juniors setback — down {abs(delta)} spots"
            else:
                hist_row["stock_label"] = "Holding"
                hist_row["stock_reason"] = "World Juniors — flat tournament impact"
            append_stock_history_snapshot(
                session,
                hist_row,
                event_source="WORLD_JUNIORS",
                date_label=date_label,
            )

    bundle["prospects_stock_snapshot"] = list(prospects_stocked)
    session.wjc_tournament_bundle = bundle
    return rank_changes


def _wjc_sync_stock_display_after_persist(
    session: FranchiseSession,
    sim: Any,
    prospects_stocked: List[Dict[str, Any]],
    rank_changes: Dict[str, Dict[str, int]],
) -> List[Dict[str, Any]]:
    board = get_cached_draft_class_rankings(session, sim)
    entries = {
        str(e.get("key")): e for e in (board.get("entries") or []) if isinstance(e, dict) and e.get("key")
    }
    out: List[Dict[str, Any]] = []
    for p in prospects_stocked:
        row = dict(p)
        key = str(row.get("draft_prospect_id") or row.get("player_id") or "")
        ch = rank_changes.get(key) or {}
        if ch:
            row["stock_rank_before"] = int(ch.get("before") or row.get("stock_rank_before") or 0)
            row["stock_rank_after"] = int(ch.get("after") or row.get("stock_rank_after") or 0)
            row["stock_delta"] = int(ch.get("delta") or row.get("stock_delta") or 0)
        elif str(row.get("prospect_classification") or "") == "draft_eligible" and key in entries:
            e = entries[key]
            row["stock_rank_after"] = int(e.get("rank") or row.get("stock_rank_after") or 0)
            try:
                before = int(row.get("stock_rank_before") or row["stock_rank_after"])
                row["stock_delta"] = before - int(row["stock_rank_after"])
            except (TypeError, ValueError):
                pass
        out.append(row)
    return out


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


def _collect_wjc_tournament_prospects(
    session: FranchiseSession,
    rng: random.Random,
    codes: List[str],
    label_by: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Place draft-eligible prospects on national rosters for the U20 tournament."""
    by_code: Dict[str, List[Dict[str, Any]]] = {c: [] for c in codes}
    all_rows: List[Dict[str, Any]] = []
    seen_ids: set = set()

    sim = getattr(session, "sim", None)
    try:
        board = build_draft_class_rankings(session, sim) if sim is not None else {"entries": []}
    except Exception:
        board = {"entries": []}
    board_entries = board.get("entries") or []
    by_key = {
        str(e.get("key")): e
        for e in board_entries
        if isinstance(e, dict) and e.get("key")
    }
    rank_by_key = {
        str(e.get("key")): int(e.get("rank") or i + 1)
        for i, e in enumerate(board_entries)
        if isinstance(e, dict) and e.get("key")
    }
    ut_abbr = _wjc_user_team_abbr(session)

    for p in _collect_user_wjc_prospects(session, rng):
        if not p.get("made_wjc_team"):
            continue
        c = str(p.get("wjc_country") or "")
        if c not in by_code:
            continue
        pid = str(p.get("player_id") or "")
        if not pid or pid in seen_ids:
            continue
        seen_ids.add(pid)
        row = {
            "player_id": pid,
            "name": str(p.get("name") or "?"),
            "wjc_country": c,
            "wjc_country_label": str(p.get("wjc_country_label") or label_by.get(c, c)),
            "position": "F",
            "age": int(p.get("age") or 19),
            "nationality": str(p.get("nationality") or ""),
            "is_user_prospect": True,
            "roster": str(p.get("roster") or ""),
        }
        row = _wjc_enrich_prospect_row(session, row, by_key, rank_by_key, ut_abbr)
        by_code[c].append(row)
        all_rows.append(row)

    for entry in board_entries:
        if not isinstance(entry, dict):
            continue
        age = int(entry.get("age") or 99)
        if age > 20:
            continue
        nat = str(
            entry.get("nationality")
            or entry.get("country")
            or entry.get("birth_country")
            or ""
        )
        code = _wjc_country_for_birth(rng, nat)
        if not code or code not in by_code:
            continue
        pid = str(entry.get("key") or entry.get("player_id") or entry.get("id") or "")
        if not pid or pid in seen_ids:
            continue
        seen_ids.add(pid)
        stock = entry.get("draft_stock") if isinstance(entry.get("draft_stock"), dict) else {}
        rank = int(entry.get("rank") or stock.get("current_rank") or 999)
        row = {
            "player_id": pid,
            "draft_prospect_id": pid,
            "prospect_classification": "draft_eligible",
            "name": str(entry.get("name") or "?"),
            "wjc_country": code,
            "wjc_country_label": label_by.get(code, code),
            "position": str(entry.get("position") or "F")[:3].upper(),
            "age": age,
            "nationality": nat,
            "is_user_prospect": False,
            "stock_rank_before": rank,
            "stock_rank_after": rank,
            "stock_delta": 0,
            "ovr": max(0.45, min(0.95, 1.0 - (rank / 250.0))),
            "junior_league": str(entry.get("league") or entry.get("league_name") or ""),
            "junior_team": str(entry.get("team") or entry.get("team_name") or ""),
            "junior_gp": int(entry.get("gp") or entry.get("games_played") or 0),
            "junior_g": int(entry.get("goals") or entry.get("g") or 0),
            "junior_a": int(entry.get("assists") or entry.get("a") or 0),
            "junior_pts": int(entry.get("points") or entry.get("pts") or 0),
            "scouting_confidence": entry.get("scouting_confidence"),
        }
        by_code[code].append(row)
        all_rows.append(row)

    first_names = ["Alex", "Marcus", "Erik", "Liam", "Noah", "Owen", "Kai", "Mika", "Joonas", "Ivan"]
    last_names = ["Smith", "Johnson", "Karlsson", "Mueller", "Novak", "Silva", "Berg", "Petrov", "Lee", "Costa"]
    for c in codes:
        while len(by_code[c]) < 14:
            idx = len(by_code[c])
            pid = f"wjc_npc_{c}_{idx}"
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            nm = f"{rng.choice(first_names)} {rng.choice(last_names)}"
            rank = 40 + rng.randint(0, 160)
            row = {
                "player_id": pid,
                "draft_prospect_id": None,
                "prospect_classification": "tournament_npc",
                "name": nm,
                "wjc_country": c,
                "wjc_country_label": label_by.get(c, c),
                "position": rng.choice(["F", "F", "F", "D", "D", "G"]),
                "age": rng.randint(18, 20),
                "nationality": label_by.get(c, c),
                "is_user_prospect": False,
                "stock_rank_before": rank,
                "stock_rank_after": rank,
                "stock_delta": 0,
                "ovr": max(0.45, min(0.9, 1.0 - (rank / 250.0))),
                "is_npc": True,
            }
            by_code[c].append(row)
            all_rows.append(row)

    for c in codes:
        by_code[c].sort(key=lambda x: -float(x.get("ovr") or 0))
    return all_rows, by_code


def _wjc_skater_pool(prospects_by_code: Dict[str, List[Dict[str, Any]]], team_code: str) -> List[Dict[str, Any]]:
    return [
        p
        for p in prospects_by_code.get(team_code, [])
        if str(p.get("position") or "F").upper() != "G"
    ]


def _wjc_distribute_team_scoring(
    rng: random.Random,
    skaters: List[Dict[str, Any]],
    goals: int,
    *,
    team_won: bool,
) -> List[Dict[str, Any]]:
    pool = skaters[:10] if skaters else []
    if not pool or goals <= 0:
        return []

    weights = [max(0.12, float(s.get("ovr") or 0.5)) for s in pool]
    tallies: Dict[str, Dict[str, int]] = {
        str(s["player_id"]): {"g": 0, "a": 0, "sog": 0, "plus_minus": 0} for s in pool
    }

    for _ in range(goals):
        scorer = rng.choices(pool, weights=weights, k=1)[0]
        sid = str(scorer["player_id"])
        tallies[sid]["g"] += 1
        if rng.random() < 0.82:
            assister = rng.choices(pool, weights=weights, k=1)[0]
            tallies[str(assister["player_id"])]["a"] += 1

    for s in pool:
        sid = str(s["player_id"])
        tallies[sid]["sog"] = rng.randint(0, 5) + tallies[sid]["g"] * 2
        tallies[sid]["plus_minus"] = rng.randint(0, 2) if team_won else -rng.randint(0, 2)

    lines: List[Dict[str, Any]] = []
    for s in pool:
        sid = str(s["player_id"])
        t = tallies[sid]
        pts = int(t["g"]) + int(t["a"])
        if pts <= 0 and t["sog"] <= 1:
            continue
        lines.append(
            {
                "player_id": sid,
                "name": str(s.get("name") or "?"),
                "wjc_country": str(s.get("wjc_country") or ""),
                "position": str(s.get("position") or "F"),
                "g": int(t["g"]),
                "a": int(t["a"]),
                "pts": pts,
                "sog": int(t["sog"]),
                "plus_minus": int(t["plus_minus"]),
                "is_user_prospect": bool(s.get("is_user_prospect")),
            }
        )
    lines.sort(key=lambda r: (-int(r.get("pts", 0)), -int(r.get("g", 0))))
    return lines


def _build_wjc_game_box_score(
    rng: random.Random,
    game: Dict[str, Any],
    prospects_by_code: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    home = str(game.get("home") or "")
    away = str(game.get("away") or "")
    hg = int(game.get("home_goals") or 0)
    ag = int(game.get("away_goals") or 0)
    home_won = hg > ag
    home_lines = _wjc_distribute_team_scoring(
        rng, _wjc_skater_pool(prospects_by_code, home), hg, team_won=home_won
    )
    away_lines = _wjc_distribute_team_scoring(
        rng, _wjc_skater_pool(prospects_by_code, away), ag, team_won=not home_won
    )
    return {
        **game,
        "box_score": {
            "home": home_lines,
            "away": away_lines,
        },
    }


def _aggregate_wjc_player_stats(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg: Dict[str, Dict[str, Any]] = {}
    for g in games:
        box = g.get("box_score") if isinstance(g.get("box_score"), dict) else {}
        for side in ("home", "away"):
            for row in box.get(side) or []:
                if not isinstance(row, dict):
                    continue
                pid = str(row.get("player_id") or row.get("name") or "")
                if not pid:
                    continue
                cur = agg.get(pid)
                if cur is None:
                    cur = {
                        "player_id": pid,
                        "name": str(row.get("name") or "?"),
                        "wjc_country": str(row.get("wjc_country") or ""),
                        "position": str(row.get("position") or "F"),
                        "gp": 0,
                        "g": 0,
                        "a": 0,
                        "pts": 0,
                        "sog": 0,
                        "plus_minus": 0,
                        "is_user_prospect": bool(row.get("is_user_prospect")),
                    }
                    agg[pid] = cur
                cur["gp"] += 1
                cur["g"] += int(row.get("g") or 0)
                cur["a"] += int(row.get("a") or 0)
                cur["pts"] += int(row.get("pts") or 0)
                cur["sog"] += int(row.get("sog") or 0)
                cur["plus_minus"] += int(row.get("plus_minus") or 0)
    rows = list(agg.values())
    rows.sort(
        key=lambda r: (
            -int(r.get("pts", 0) or 0),
            -int(r.get("g", 0) or 0),
            -int(r.get("a", 0) or 0),
            str(r.get("name") or ""),
        )
    )
    return rows


def _apply_wjc_stock_after(
    prospects: List[Dict[str, Any]],
    player_stats: List[Dict[str, Any]],
    standings: List[Dict[str, Any]],
    medal_labels: Dict[str, Any],
    *,
    day_multiplier: float = 1.0,
) -> List[Dict[str, Any]]:
    stats_by_id = {str(r.get("player_id") or ""): r for r in player_stats}
    standing_by_code = {str(r.get("code") or ""): r for r in standings}
    medal_map = {str(v or "").lower(): k for k, v in (medal_labels or {}).items()}

    out: List[Dict[str, Any]] = []
    for p in prospects:
        row = dict(p)
        if str(row.get("prospect_classification") or "") == "drafted_user":
            row["stock_rank_after"] = None
            row["stock_delta"] = None
            out.append(row)
            continue
        pid = str(row.get("player_id") or "")
        before = row.get("stock_rank_before")
        try:
            base_rank = int(before) if before is not None else None
        except (TypeError, ValueError):
            base_rank = None
        if base_rank is None:
            if bool(row.get("is_npc")) or str(row.get("prospect_classification") or "") == "tournament_npc":
                try:
                    base_rank = int(before) if before is not None else 120
                except (TypeError, ValueError):
                    base_rank = 120
            else:
                out.append(row)
                continue
        st = stats_by_id.get(pid) or {}
        pts = int(st.get("pts") or 0)
        goals = int(st.get("g") or 0)
        plus = int(st.get("plus_minus") or 0)
        code = str(row.get("wjc_country") or "")
        nat_label = str(row.get("wjc_country_label") or "").lower()
        team_row = standing_by_code.get(code) or {}
        team_wins = int(team_row.get("w") or 0)
        team_losses = int(team_row.get("l") or 0)

        delta = 0
        delta -= goals * 3
        delta -= max(0, pts - goals) * 2
        delta -= max(0, plus) * 1
        delta -= team_wins * 2
        delta += team_losses * 2
        if nat_label and medal_map.get(nat_label) == "gold":
            delta -= 18
        elif nat_label and medal_map.get(nat_label) == "silver":
            delta -= 12
        elif nat_label and medal_map.get(nat_label) == "bronze":
            delta -= 8

        delta = int(round(delta * max(1.0, float(day_multiplier))))
        after = max(1, base_rank + delta)
        row["stock_rank_after"] = after
        row["stock_delta"] = base_rank - after
        row["tournament_pts"] = pts
        row["tournament_g"] = goals
        row["tournament_gp"] = int(st.get("gp") or 0)
        row["team_wins"] = team_wins
        row["team_losses"] = team_losses
        out.append(row)
    out.sort(key=lambda x: (-int(x.get("stock_delta") or 0), str(x.get("name") or "")))
    return out


def _all_wjc_games_from_bundle(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    games: List[Dict[str, Any]] = list(bundle.get("rr_games") or [])
    po = bundle.get("playoffs") or {}
    for key in ("quarterfinals", "semifinals"):
        games.extend(list(po.get(key) or []))
    for key in ("bronze", "gold"):
        g = po.get(key)
        if isinstance(g, dict):
            games.append(g)
    return games


def _wjc_build_rr_schedule(codes: List[str], rng: random.Random) -> List[List[Tuple[str, str]]]:
    """Round-robin pairings bucketed so each nation plays at most once per tournament day."""
    pairings: List[Tuple[str, str]] = []
    for i, hi in enumerate(codes):
        for aj in codes[i + 1 :]:
            pairings.append((hi, aj))
    rng.shuffle(pairings)

    days: List[List[Tuple[str, str]]] = []
    remaining = list(pairings)
    while remaining:
        used: set = set()
        day: List[Tuple[str, str]] = []
        still: List[Tuple[str, str]] = []
        for a, b in remaining:
            if a in used or b in used:
                still.append((a, b))
            else:
                day.append((a, b))
                used.add(a)
                used.add(b)
        if not day and remaining:
            a, b = remaining[0]
            day.append((a, b))
            still = remaining[1:]
        days.append(day)
        remaining = still
    return days


def _simulate_wjc_national_bundle(session: FranchiseSession, rng: random.Random) -> Dict[str, Any]:
    """Full U20 worlds — national teams only (deterministic from rng)."""
    countries = _wjc_countries_meta()
    codes = [c for c, _ in countries]
    label_by = {c: lab for c, lab in countries}

    rr_games: List[Dict[str, Any]] = []
    rr_days = _wjc_build_rr_schedule(codes, rng)
    for day_idx, day_pairings in enumerate(rr_days, start=1):
        for hi, aj in day_pairings:
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
                    "round": "Preliminary Round",
                    "game_day": day_idx,
                    "is_playoff": False,
                }
            )

    rr_days_total = len(rr_days)
    st_full = _rr_standings_from_slice(codes, label_by, rr_games)
    code_order = [r["code"] for r in st_full]

    def _play_pair(a: str, b: str, label: str, lb: Dict[str, str], *, game_day: int) -> Dict[str, Any]:
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
            "game_day": game_day,
            "is_playoff": True,
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
    po_day = rr_days_total + 1
    qf = [
        _play_pair(s1, s8, "Quarterfinal", label_by, game_day=po_day),
        _play_pair(s2, s7, "Quarterfinal", label_by, game_day=po_day),
        _play_pair(s3, s6, "Quarterfinal", label_by, game_day=po_day),
        _play_pair(s4, s5, "Quarterfinal", label_by, game_day=po_day),
    ]
    w_qf = [g["winner"] for g in qf]
    sf = [
        _play_pair(w_qf[0], w_qf[1], "Semifinal", label_by, game_day=po_day + 1),
        _play_pair(w_qf[2], w_qf[3], "Semifinal", label_by, game_day=po_day + 1),
    ]
    w_sf = [g["winner"] for g in sf]
    l_sf = [g["loser"] for g in sf]
    bronze = _play_pair(l_sf[0], l_sf[1], "Bronze", label_by, game_day=po_day + 2)
    gold = _play_pair(w_sf[0], w_sf[1], "Gold Medal", label_by, game_day=po_day + 2)
    medals = {
        "gold": gold["winner"],
        "silver": gold["loser"],
        "bronze": bronze["winner"],
        "fourth": bronze["loser"],
    }
    medal_labels = {k: label_by.get(v, v) for k, v in medals.items()}

    all_prospects, prospects_by_code = _collect_wjc_tournament_prospects(session, rng, codes, label_by)
    game_rng = random.Random(rng.randint(0, 2**31 - 1))
    rr_games = [_build_wjc_game_box_score(game_rng, g, prospects_by_code) for g in rr_games]
    qf = [_build_wjc_game_box_score(game_rng, g, prospects_by_code) for g in qf]
    sf = [_build_wjc_game_box_score(game_rng, g, prospects_by_code) for g in sf]
    bronze = _build_wjc_game_box_score(game_rng, bronze, prospects_by_code)
    gold = _build_wjc_game_box_score(game_rng, gold, prospects_by_code)

    return {
        "countries": [{"code": c, "label": lab} for c, lab in countries],
        "rr_games": rr_games,
        "playoffs": {"quarterfinals": qf, "semifinals": sf, "bronze": bronze, "gold": gold},
        "medals": medals,
        "medal_labels": medal_labels,
        "tournament_prospects": all_prospects,
        "prospects_by_country": prospects_by_code,
        "rr_days_total": rr_days_total,
        "wjc_format_version": 2,
    }


def _ensure_wjc_tournament_bundle(session: FranchiseSession) -> None:
    sy = int(session.season_calendar_year)
    b = getattr(session, "wjc_tournament_bundle", None)
    if (
        isinstance(b, dict)
        and int(b.get("season_sy", -1)) == sy
        and int(b.get("wjc_format_version", 0)) >= 2
        and isinstance(b.get("tournament_prospects"), list)
        and b.get("tournament_prospects")
    ):
        return
    rng = _rng_for_event(session, f"wjc_bundle_{sy}")
    core = _simulate_wjc_national_bundle(session, rng)
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
    rr_days_total = int(bundle.get("rr_days_total") or 9)
    current_day = d_idx + 1
    rr_slice = [g for g in rr_all if int(g.get("game_day") or 0) <= current_day]
    standings = _rr_standings_from_slice(codes, label_by, rr_slice)

    po_all = bundle.get("playoffs") or {}
    po_out: Dict[str, Any] = {}
    if current_day > rr_days_total:
        po_out["quarterfinals"] = po_all.get("quarterfinals") or []
    if current_day > rr_days_total + 1:
        po_out["semifinals"] = po_all.get("semifinals") or []
    if current_day > rr_days_total + 2 and po_all.get("bronze"):
        po_out["bronze"] = po_all.get("bronze")
    if current_day > rr_days_total + 2 and po_all.get("gold"):
        po_out["gold"] = po_all.get("gold")

    complete = bool(current_day > rr_days_total + 2 and po_all.get("gold"))
    medals = bundle.get("medal_labels") if complete else {}
    user_prospects = _collect_user_wjc_prospects(session, rng)

    all_games = _all_wjc_games_from_bundle(bundle)
    visible_games = [g for g in all_games if int(g.get("game_day") or 999) <= current_day]
    games_today = [g for g in visible_games if int(g.get("game_day") or 0) == current_day]
    player_stats = _aggregate_wjc_player_stats(visible_games)
    tournament_prospects = list(bundle.get("tournament_prospects") or [])
    evaluated = getattr(session, "wjc_stock_evaluated_seasons", None) or set()
    if complete and int(sy) in evaluated:
        snap = bundle.get("prospects_stock_snapshot")
        if isinstance(snap, list) and snap:
            prospects_stocked = list(snap)
        else:
            prospects_stocked = _apply_wjc_stock_after(
                tournament_prospects,
                player_stats,
                standings,
                medals if complete else {},
                day_multiplier=1.35,
            )
    else:
        prospects_stocked = _apply_wjc_stock_after(
            tournament_prospects,
            player_stats,
            standings,
            medals if complete else {},
            day_multiplier=1.35,
        )
        if complete:
            sim = getattr(session, "sim", None)
            rank_changes = _persist_wjc_stock_to_draft_class(
                session,
                sim,
                prospects_stocked,
                season_sy=sy,
                cal_iso=iso,
            )
            if rank_changes:
                prospects_stocked = _wjc_sync_stock_display_after_persist(
                    session, sim, prospects_stocked, rank_changes
                )
                bundle = getattr(session, "wjc_tournament_bundle", None) or {}
                bundle["prospects_stock_snapshot"] = list(prospects_stocked)
                session.wjc_tournament_bundle = bundle

    return {
        "kind": "wjc_tournament",
        "wjc_live": True,
        "wjc_phase": "complete" if complete else "live",
        "calendar_iso": iso,
        "wjc_day": d_idx + 1,
        "wjc_days_total": n_days,
        "title": f"World Juniors (U20) — day {d_idx + 1} of {n_days}",
        "season_label": f"{sy}–{sy + 1}",
        "countries": countries,
        "round_robin_games": rr_slice,
        "round_robin_total": n_rr,
        "standings": standings,
        "playoffs": po_out,
        "medal_labels": medals if complete else {},
        "medals_final": complete,
        "user_prospects": user_prospects,
        "tournament_prospects": prospects_stocked,
        "player_stats": player_stats,
        "all_games": visible_games,
        "games_today": games_today,
        "all_games_total": len(all_games),
        "rr_days_total": rr_days_total,
    }


def _build_wjc_client_payload(session: FranchiseSession) -> Optional[Dict[str, Any]]:
    """Persistent WJC snapshot for Calendar / Hub menus (survives popup dismiss)."""
    sy = int(session.season_calendar_year)
    now_iso = (
        _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
        or _today_iso(session)
        or str(getattr(session, "current_date", "") or "")
    )
    now_iso = str(now_iso or "")[:10]
    nations = [{"code": c, "label": lab} for c, lab in _wjc_countries_meta()]
    n_days = len(_wjc_calendar_dates(sy))
    bundle = getattr(session, "wjc_tournament_bundle", None)
    # Drop prior-year bundles so medal_labels from last WJC cannot resurrect
    # a finished desk in September of the new season.
    if isinstance(bundle, dict) and int(bundle.get("season_sy", -1)) != sy:
        session.wjc_tournament_bundle = None
        bundle = None
    has_bundle = (
        isinstance(bundle, dict)
        and int(bundle.get("season_sy", -1)) == sy
        and isinstance(bundle.get("tournament_prospects"), list)
        and bool(bundle.get("tournament_prospects"))
    )
    d_idx = _wjc_day_index_for_iso(now_iso, sy) if now_iso else None

    if has_bundle:
        if d_idx is not None:
            return _wjc_live_tournament_payload(session, now_iso, d_idx, n_days)
        evaluated = getattr(session, "wjc_stock_evaluated_seasons", None) or set()
        last = _wjc_calendar_dates(sy)[-1]
        past_window = bool(now_iso and now_iso >= last.isoformat())
        if past_window and (
            int(sy) in evaluated or bool(bundle.get("stock_evaluated")) or bool(bundle.get("medal_labels"))
        ):
            return _wjc_live_tournament_payload(session, last.isoformat(), n_days - 1, n_days)
        # Premature complete bundle (year-rollover leak before Dec 26) — wipe so
        # stock/medals cannot stick and the desk rebuilds when WJC actually opens.
        if bool(bundle.get("medal_labels")) or bool(bundle.get("stock_evaluated")) or int(sy) in evaluated:
            session.wjc_tournament_bundle = None
            try:
                evaluated.discard(int(sy))
                session.wjc_stock_evaluated_seasons = evaluated
            except Exception:
                pass
            has_bundle = False
        else:
            # Bundle pre-built before first calendar day — nations + prospect pool only.
            rng = _rng_for_event(session, f"wjc_prospects_{sy}")
            return {
                "kind": "wjc_tournament",
                "wjc_live": False,
                "wjc_phase": "upcoming",
                "calendar_iso": now_iso,
                "wjc_day": None,
                "wjc_days_total": n_days,
                "title": "World Juniors (U20)",
                "season_label": f"{sy}–{sy + 1}",
                "countries": list(bundle.get("countries") or nations),
                "round_robin_games": [],
                "round_robin_total": len(bundle.get("rr_games") or []),
                "standings": [],
                "playoffs": {},
                "medal_labels": {},
                "medals_final": False,
                "user_prospects": _collect_user_wjc_prospects(session, rng),
                "tournament_prospects": list(bundle.get("tournament_prospects") or []),
                "player_stats": [],
                "all_games": [],
                "games_today": [],
                "all_games_total": len(_all_wjc_games_from_bundle(bundle)),
                "rr_days_total": int(bundle.get("rr_days_total") or 9),
            }

    return {
        "kind": "wjc_tournament",
        "wjc_live": False,
        "wjc_phase": "upcoming",
        "calendar_iso": now_iso,
        "wjc_day": None,
        "wjc_days_total": n_days,
        "title": "World Juniors (U20)",
        "season_label": f"{sy}–{sy + 1}",
        "countries": nations,
        "round_robin_games": [],
        "round_robin_total": 0,
        "standings": [],
        "playoffs": {},
        "medal_labels": {},
        "medals_final": False,
        "user_prospects": [],
        "tournament_prospects": [],
        "player_stats": [],
        "all_games": [],
        "games_today": [],
        "all_games_total": 0,
        "rr_days_total": 9,
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
    """One calendar year of aging for the whole league (rosters, FA, minors, juniors).

    Age is synced from birth date to Sept 15 of the upcoming season year so players
    like Sanderson (2002-07-08) are 25 on 2027-09-15 — not permanently stuck a year young.
    """
    from app.sim_engine import engine as eng_mod
    from app.sim_engine.engine import assign_career_phase_from_age
    from app.sim_engine.progression.potential import ensure_development_ledger

    season_id = int(getattr(session, "season_calendar_year", 2025) or 2025)
    # End of season SY → next season's Sept 15 age (SY+1).
    age_as_of_year = season_id + 1
    league = getattr(getattr(session, "sim", None), "league", None)
    players = _iter_league_players_for_aging(league) if league is not None else []
    if not players:
        # Fallback: NHL rosters only
        for team in teams:
            players.extend(list(getattr(team, "roster", None) or []))

    team_by_player: Dict[Any, Any] = {}
    for team in teams:
        for p in getattr(team, "roster", None) or []:
            team_by_player[id(p)] = team

    for player in players:
        if getattr(player, "retired", False):
            continue
        try:
            setattr(player, "_active_dev_season", season_id)
        except Exception:
            pass
        ledger = ensure_development_ledger(player, season_id)
        if ledger.get("aging_applied"):
            # Still re-sync DOB age in case a prior tick used integer-only +1 and drifted.
            try:
                sync_player_age_to_season(player, age_as_of_year)
            except Exception:
                pass
            continue

        age_before = _player_age_int(player)
        try:
            sync_player_age_to_season(player, age_as_of_year)
        except Exception:
            advance_fn = getattr(player, "advance_year", None)
            if callable(advance_fn):
                try:
                    advance_fn(apply_peak_decline=False)
                except Exception:
                    pass
            else:
                # Last resort — prefer DOB sync via session pin when possible.
                try:
                    sync_player_age_to_session(player, session)
                except Exception:
                    ident = getattr(player, "identity", None)
                    if ident is not None and hasattr(ident, "age"):
                        try:
                            ident.age = int(getattr(ident, "age", 0)) + 1
                        except (TypeError, ValueError):
                            pass

        age_after = _player_age_int(player)
        # Soft aging decline: older players sometimes lose a touch of overall.
        try:
            if age_after >= 29 and age_after > age_before:
                from app.sim_engine.entities.player import persist_recomputed_ovr

                team = team_by_player.get(id(player))
                if team is not None:
                    try:
                        sys_dev = float(eng_mod.team_system_development_modifier(team))
                    except Exception:
                        sys_dev = 0.0
                else:
                    sys_dev = 0.0
                # Extra cliff chance rises with age; not every vet declines every year.
                cliff_p = 0.22 + max(0, age_after - 31) * 0.06
                rng = getattr(getattr(session, "sim", None), "rng", None)
                roll = float(rng.random()) if rng is not None else 0.5
                if roll < cliff_p:
                    ratings = getattr(player, "ratings", None)
                    if isinstance(ratings, dict) and ratings:
                        haircut = 0.6 + max(0, age_after - 32) * 0.15
                        for k, v in list(ratings.items()):
                            kl = str(k).lower()
                            if kl in ("dev_potential", "dev_ceiling", "potential", "overall", "ovr"):
                                continue
                            try:
                                ratings[k] = max(35.0, float(v) - haircut)
                            except Exception:
                                pass
                        try:
                            persist_recomputed_ovr(player)
                        except Exception:
                            pass
                    _ = sys_dev  # reserved for future team-context dampening
        except Exception:
            pass

        ledger["aging_applied"] = True
        try:
            assign_career_phase_from_age(player)
        except Exception:
            pass
        try:
            from app.sim_engine.entities.player import persist_recomputed_ovr

            persist_recomputed_ovr(player)
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
    # Pin ages to next Sept 15 BEFORE any session_age_as_of / serialize path can
    # read the still-live April–June calendar (flag is normally set by the caller
    # after this returns — set it here so mid-progression resync cannot undo ages).
    try:
        setattr(session, "_year_end_progression_done", True)
    except Exception:
        pass
    try:
        out["age_resync"] = resync_league_ages_to_session(session)
    except Exception:
        out["age_resync"] = {"synced": 0}

    # Wire season production into growth before the roster development pass.
    try:
        from services.franchise_offseason import _dev_stamp_season_production

        for tm in teams:
            for pl in getattr(tm, "roster", None) or []:
                _dev_stamp_season_production(session, pl)
    except Exception:
        pass

    # ELC Schedule A/B payouts + development-promise morale from season stats.
    try:
        from services.elc_year_end_ledger import run_year_end_contract_ledger

        out["contract_ledger"] = run_year_end_contract_ledger(session, season_year=sy)
    except Exception:
        out["contract_ledger"] = {"ok": False}

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

        # 74.5 was permanently "on" and clawed back fresh development years.
        apply_league_ovr_soft_regression_if_needed(teams, rng, avg_trigger=78.0)
    except Exception:
        pass

    out["retired_removed"] = int(_strip_retired_from_nhl_rosters(teams))
    # Season calendar year advances only in generate_next_season (authoritative transition).
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

        did = str(d.get("id") or "").strip()
        if not did:
            session.timeline.append(
                f"AUTO-RESOLVE: removed decision with missing id ({d.get('kind') or d.get('type') or 'unknown'})."
            )
            session.pending_decisions.pop(0)
            continue

        session.timeline.append(
            f"AUTO-RESOLVE: {d.get('kind') or d.get('type') or 'decision'} "
            f"{did} -> {choice_id}"
        )

        apply_decision(session, did, choice_id)
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
        called = _call_up_best_ahl_spc(user_team)
        setattr(user_team, "_needs_callup", not bool(called.get("ok")))
        setattr(user_team, "_depth_pressure", float(getattr(user_team, "_depth_pressure", 0.0) or 0.0) - 0.02)
        changed = _nudge_team_room(user_team, morale=0.004, confidence=0.004)
        effects.update({
            "callup_flag": 1 if called.get("ok") else 0,
            "callup": called,
            "depth_pressure_delta": -1,
            "players_affected": changed,
        })
        if called.get("ok"):
            effects["headline"] = called.get("headline") or "Called up affiliate depth"

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


def _call_up_best_ahl_spc(team: Any) -> Dict[str, Any]:
    """Validate-then-commit AHL→NHL call-up (23-man aware). Rolls back on failure."""
    from services.contract_economy import uses_nhl_contract_slot

    if team is None:
        return {"ok": False, "reason": "no_team"}

    nhl_snap = list(getattr(team, "roster", None) or [])
    ahl_snap = list(getattr(team, "ahl_roster", None) or [])
    nhl = list(nhl_snap)
    ahl = list(ahl_snap)
    demoted = None

    if len(nhl) >= 23:
        demote_idx = None
        demote_score = 999.0
        for i, p in enumerate(nhl):
            c = getattr(p, "contract", None)
            if bool(getattr(c, "nmc", False) or getattr(c, "no_move_clause", False)):
                continue
            score = float(_player_ovr99(p))
            if score < demote_score:
                demote_score = score
                demote_idx = i
        if demote_idx is None:
            return {"ok": False, "reason": "nhl_roster_full_no_demotion"}
        demoted = nhl.pop(demote_idx)
        ahl.append(demoted)

    eligible = [p for p in ahl if uses_nhl_contract_slot(p)]
    if demoted is not None:
        # Demoted player is temporarily on AHL list; do not call them back up.
        eligible = [
            p for p in eligible
            if str(getattr(p, "id", "")) != str(getattr(demoted, "id", ""))
        ]
    if not eligible:
        return {"ok": False, "reason": "no_spc_affiliate"}
    eligible.sort(key=lambda p: -_player_ovr99(p))
    pick = eligible[0]
    ahl = [p for p in ahl if str(getattr(p, "id", "")) != str(getattr(pick, "id", ""))]
    if len(nhl) >= 23:
        return {"ok": False, "reason": "nhl_roster_full_after_plan"}

    # Commit only after both lists are valid.
    try:
        if demoted is not None:
            demoted.in_minors = True
            demoted.roster_location = "ahl"
            demoted.is_buried = bool(getattr(demoted, "is_buried", False))
        pick.in_minors = False
        pick.is_buried = False
        pick.roster_location = "nhl"
        nhl.append(pick)
        setattr(team, "roster", nhl)
        setattr(team, "ahl_roster", ahl)
    except Exception as exc:
        setattr(team, "roster", nhl_snap)
        setattr(team, "ahl_roster", ahl_snap)
        return {"ok": False, "reason": f"call_up_rollback:{exc}"}

    return {
        "ok": True,
        "player_id": str(getattr(pick, "id", "")),
        "name": _name_str(pick),
        "headline": f"Called up {_name_str(pick)} from AHL",
        "demoted_player_id": str(getattr(demoted, "id", "")) if demoted is not None else "",
    }


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


def _apply_legal_conduct_decision_effect(
    session: FranchiseSession,
    decision: Dict[str, Any],
    choice: Dict[str, Any],
) -> Dict[str, Any]:
    """Wire GM legal popup choices into the conduct incident state machine."""
    from app.sim_engine.franchise.conduct_incidents import apply_gm_conduct_choice  # noqa: WPS433

    meta = dict(decision.get("meta") or {})
    cid = str(choice.get("id") or choice.get("choice_id") or "")
    incident_id = str(meta.get("incident_id") or meta.get("storyline_id") or decision.get("id") or "")
    tid = str(meta.get("team_id") or session.user_team_id or "")
    team = session.team_by_id.get(tid) or session.team_by_id.get(str(session.user_team_id))
    player = None
    if team is not None:
        player = _find_player_on_team_by_id_or_name(
            team,
            player_id=str(meta.get("player_id") or ""),
            player_name=str(meta.get("player_name") or ""),
        )
    # If incident_id missing, try player-linked active incident.
    if not incident_id and player is not None:
        incident_id = str(getattr(player, "_conduct_incident_id", "") or "")

    result = apply_gm_conduct_choice(
        session,
        incident_id=incident_id,
        choice_id=cid,
        player=player,
        statement_tone=str(choice.get("statement_tone") or meta.get("statement_tone") or ""),
        rng=getattr(getattr(session, "sim", None), "rng", None),
    )
    if not result.get("ok"):
        # Still acknowledge choice so the UI unblocks; surface soft failure.
        return {
            "conduct_ok": 0,
            "reason": str(result.get("reason") or "incident_not_found"),
            "effect_summary": "No matching conduct incident found — choice recorded without state change.",
        }

    summary = str(result.get("effect_summary") or choice.get("effect_summary") or "")
    if summary:
        choice["effect_summary"] = summary
    out = {
        "conduct_ok": 1,
        "incident_id": result.get("incident_id"),
        "eligible_to_play": result.get("eligible_to_play"),
        "status": result.get("status"),
        "effect_summary": summary,
    }
    org = result.get("org") or {}
    if org:
        out["owner_confidence"] = org.get("owner_confidence")
        out["fan_approval"] = org.get("fan_approval")
        out["media_heat"] = org.get("media_heat")
        out["sponsor_confidence"] = org.get("sponsor_confidence")
        out["revenue_modifier"] = org.get("revenue_modifier")
    if result.get("trade_market_restricted"):
        out["trade_market_restricted"] = 1
        try:
            if player is not None:
                setattr(player, "_conduct_trade_restricted", True)
        except Exception:
            pass
    return out


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

    try:
        from app.sim_engine.systems.chemistry import apply_storyline_chemistry_effect  # noqa: WPS433

        chem = apply_storyline_chemistry_effect(session, decision, choice, target=target)
        if chem:
            applied["chemistry"] = chem
    except Exception:
        pass

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
    from services.franchise_offseason import _transition_to_playoff_ready

    try:
        # Force a season-end board trail point so value trajectory reaches "Current".
        snapshot_draft_rank_prev(session, session.sim, force=True)
    except Exception:
        pass
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
    _sync_session_phase_from_calendar(session)
    ensure_session_nhl_salary_cap(session)

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
            snap = _regular_season_completion_snapshot(session)
            remaining = int(snap.get("remaining_by_day_slots", 0) or 0)
            dump = _dump_remaining_regular_games(
                session,
                reason="boundary_reached_with_incomplete_schedule",
            )
            raise RuntimeError(
                "Regular season boundary reached but schedule is incomplete: "
                f"scheduled={snap.get('scheduled_regular_games')} "
                f"completed={snap.get('completed_regular_games')} "
                f"remaining_slots={remaining} "
                f"orphan_dump_n={len(dump)}. "
                f"sample={dump[:12]}"
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
        if bool(getattr(session, "_bulk_auto_resolve_injuries", False)):
            _auto_resolve_pending_decisions(session)
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
        if bool(getattr(session, "_bulk_auto_resolve_injuries", False)):
            _auto_resolve_pending_decisions(session)
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
    try:
        if not bool(getattr(session, "_defer_payload_invalidation", False)):
            from services.franchise_scouting import apply_passive_scouting_progress
            apply_passive_scouting_progress(session)
    except Exception:
        pass
    if not bool(getattr(session, "_defer_payload_invalidation", False)):
        invalidate_session_payload_caches(session, reason="advance_day")

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

    prior_defer = bool(getattr(session, "_defer_prospect_sync", False))
    prior_light = bool(getattr(session, "_light_game_stat_accumulation", False))
    prior_bulk_inj = bool(getattr(session, "_bulk_auto_resolve_injuries", False))
    prior_defer_inv = bool(getattr(session, "_defer_payload_invalidation", False))
    prior_bulk_cal = bool(getattr(session, "_bulk_calendar_advance", False))
    session._defer_prospect_sync = True
    # Light accumulation is the designed bulk path: strength-based scores + allocated
    # stats (same counting model for every club). Full event sim is reserved for
    # single-day manual advance where the user may inspect box scores.
    use_light_bulk = eff_mode in ("season", "days")
    session._light_game_stat_accumulation = bool(use_light_bulk)
    session._bulk_calendar_advance = True
    session._bulk_fa_days_pending = 0
    session._bulk_auto_resolve_injuries = bool(auto_resolve_decisions)
    session._defer_payload_invalidation = True
    session._bulk_finalize_start_days = int(getattr(session, "calendar_days_finished", 0) or 0)
    try:
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

            if bool(getattr(session, "_audit_schedule_invariants", False)) and (len(steps) % 25 == 0):
                print(
                    f"[bulk] steps={len(steps)} phase={getattr(session, 'phase', None)} "
                    f"cursor={getattr(session, 'calendar_cursor', None)} "
                    f"games={len(getattr(session, 'game_results', []) or [])} "
                    f"status={step.get('status')}",
                    flush=True,
                )

            st = str(step.get("status") or "")

            if st == "blocked":
                if auto_resolve_decisions and getattr(session, "pending_decisions", None):
                    _auto_resolve_pending_decisions(session)
                    if not getattr(session, "pending_decisions", None):
                        continue
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
                # From camp, keep advancing until regular season is finished.
                # Once playoffs/offseason start, stop so the user gets the cinematic.
                ph = str(getattr(session, "phase", "") or "")
                if ph not in ("regular", "preseason"):
                    stopped = "phase"
                    break
                if ph == "regular" and bool(getattr(session, "regular_season_complete", False)):
                    stopped = "regular_complete"
                    break

            else:
                stopped = "count"
                break

        if guard >= max_iter:
            stopped = "guard_limit"
    finally:
        session._defer_prospect_sync = prior_defer
        session._light_game_stat_accumulation = prior_light
        session._bulk_auto_resolve_injuries = prior_bulk_inj
        session._defer_payload_invalidation = prior_defer_inv
        session._bulk_calendar_advance = prior_bulk_cal
        if steps and str(steps[-1].get("status") or "") == "ok":
            try:
                start_days = int(getattr(session, "_bulk_finalize_start_days", 0) or 0)
                end_days = int(getattr(session, "calendar_days_finished", 0) or 0)
                for day_n in range(start_days + 1, end_days + 1):
                    if day_n % 8 == 0:
                        try:
                            _nhl_in_season_development_tick(session)
                        except Exception:
                            pass
                    if day_n % 5 == 0:
                        _depth_pool_progression_tick(session)
            except Exception:
                pass
            try:
                # Catch-up minors development once after bulk (deferred daily ticks).
                from app.sim_engine.league_hierarchy_bootstrap import tick_extra_league_development

                tick_extra_league_development(session.sim, session.sim.rng)
            except Exception:
                pass
            try:
                pending_fa = int(getattr(session, "_bulk_fa_days_pending", 0) or 0)
                if pending_fa > 0 and bool(getattr(session, "free_agency_open", False)):
                    from services.fa_market_engine import tick_free_agency_market
                    from services.franchise_offseason import _open_free_agency

                    tick = tick_free_agency_market(session, days=pending_fa)
                    session._last_fa_market_tick = tick
                    _open_free_agency(session, force=False)
                session._bulk_fa_days_pending = 0
            except Exception:
                session._bulk_fa_days_pending = 0
            try:
                _sync_prospect_stats_to_calendar(session, force=True)
            except Exception:
                pass
            try:
                stat_scope = _franchise_stat_scope(session, is_playoff=False)
                for row in (getattr(session, "player_season_stats", None) or {}).values():
                    if isinstance(row, dict):
                        row.setdefault("stat_scope", stat_scope)
                        row["stat_authority"] = "session.player_season_stats"
            except Exception:
                pass
            if bool(getattr(session, "_pending_prospect_revision_bump", False)):
                session._pending_prospect_revision_bump = False
                _bump_prospect_revision(session)
            try:
                from services.franchise_scouting import apply_passive_scouting_progress

                steps_n = int(len(steps))
                if steps_n > 0:
                    apply_passive_scouting_progress(session, days=steps_n)
            except Exception:
                pass
        if not prior_defer_inv:
            invalidate_session_payload_caches(session, reason="bulk_complete")
            _bump_stats_revision(session)

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

def _storyline_ids_for_decision(decision: Dict[str, Any]) -> set:
    meta = dict(decision.get("meta") or {})
    ids: set = set()
    for val in (meta.get("storyline_id"), decision.get("storyline_id"), decision.get("id")):
        s = str(val or "").strip()
        if not s:
            continue
        ids.add(s)
        if s.startswith("dec_"):
            ids.add(s[4:])
    return ids


def _find_storyline_event(session: FranchiseSession, storyline_id: str) -> Optional[Dict[str, Any]]:
    sid = str(storyline_id or "").strip()
    if not sid:
        return None
    candidates = {sid}
    if sid.startswith("dec_"):
        candidates.add(sid[4:])
    for ev in reversed(list(getattr(session, "storyline_events", None) or [])):
        if not isinstance(ev, dict):
            continue
        eids = {
            str(ev.get("storyline_id") or ""),
            str(ev.get("id") or ""),
            str(ev.get("stable_key") or ""),
        }
        if candidates & {x for x in eids if x}:
            return ev
    return None


def _apply_storyline_event_choice(
    session: FranchiseSession,
    storyline_id: str,
    choice_id: str,
) -> bool:
    """Resolve a choice directly from a storyline event when no pending decision exists."""
    ev = _find_storyline_event(session, storyline_id)
    if not ev:
        return False

    opts = list(ev.get("action_options") or [])
    chosen: Optional[Dict[str, Any]] = None
    for opt in opts:
        if not isinstance(opt, dict):
            continue
        if str(opt.get("id") or opt.get("choice_id") or "") == choice_id:
            chosen = opt
            break
    if chosen is None:
        raise ValueError(f"Choice {choice_id!r} not found for storyline {storyline_id!r}.")

    sid = str(ev.get("storyline_id") or ev.get("id") or storyline_id)
    decision = {
        "id": f"dec_{sid}",
        "storyline_id": sid,
        "kind": "storyline_event",
        "priority": str(ev.get("priority") or "MEDIUM"),
        "title": str(ev.get("headline") or ev.get("title") or "Storyline choice"),
        "meta": {
            "storyline_id": sid,
            "team_id": str(ev.get("team_id") or ""),
            "player_id": str(ev.get("player_id") or ""),
            "player_name": str(ev.get("player_name") or ""),
            "cause": str(ev.get("cause") or ""),
        },
    }

    effects: Dict[str, Any] = {}
    try:
        from app.sim_engine.franchise.storyline_engine import _apply_storyline_effects  # noqa: WPS433

        _apply_storyline_effects(
            session,
            str(ev.get("team_id") or ""),
            str(ev.get("player_id") or ""),
            dict(chosen.get("effects") or {}),
        )
        effects.update(dict(chosen.get("effects") or {}))
    except Exception:
        effects.update(_apply_generic_storyline_choice_effect(session, decision, chosen))

    label = str(chosen.get("label") or choice_id)
    eff = dict(chosen.get("effects") or {})
    press_id = str(ev.get("press_conference_id") or "")
    headline = f"{decision['title']}: {label}"
    summary = f"You chose: {label}."
    if press_id and eff.get("question_id") and eff.get("response_id"):
        try:
            from app.sim_engine.franchise.storyline_engine import apply_press_conference_response  # noqa: WPS433

            press_result = apply_press_conference_response(
                session,
                press_id,
                str(eff.get("question_id") or ""),
                str(eff.get("response_id") or ""),
            )
            headline = str(press_result.get("headline") or headline)
            summary = f"Press conference response: {label}."
        except Exception:
            pass
    if chosen.get("effect_summary"):
        summary += f" {chosen.get('effect_summary')}"

    _append_decision_feedback(
        session,
        decision=decision,
        choice=chosen,
        headline=headline,
        summary=summary,
        priority=str(ev.get("priority") or "MEDIUM"),
        effects=effects,
    )

    ev["requires_action"] = False
    ev["action_options"] = []
    ev["status"] = "resolved"

    next_popups: List[Dict[str, Any]] = []
    for popup in list(getattr(session, "pending_ui_popups", None) or []):
        if not isinstance(popup, dict):
            continue
        popup_story_id = str(popup.get("storyline_id") or popup.get("id") or "")
        if popup_story_id and popup_story_id in {sid, storyline_id}:
            continue
        next_popups.append(popup)
    session.pending_ui_popups = next_popups
    session.timeline.append(f"Storyline choice: {sid} -> {choice_id}")
    return True


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

    match_ids = {sid}
    if sid.startswith("dec_"):
        match_ids.add(sid[4:])

    for d in list(getattr(session, "pending_decisions", None) or []):
        if not isinstance(d, dict):
            continue

        if not (_storyline_ids_for_decision(d) & match_ids):
            continue

        apply_decision(session, str(d.get("id") or ""), cid)
        return

    if _apply_storyline_event_choice(session, sid, cid):
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

        elif kind == "legal_storyline_decision":
            effects.update(_apply_legal_conduct_decision_effect(session, d, chosen))

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
    global _TEAM_SUMMARY_CACHE
    if isinstance(_TEAM_SUMMARY_CACHE, list) and _TEAM_SUMMARY_CACHE:
        return list(_TEAM_SUMMARY_CACHE)
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
    _TEAM_SUMMARY_CACHE = list(out)
    return list(out)


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


def _safe_points_pct(record: Dict[str, Any]) -> float:
    try:
        gp = float(record.get("gp", 0) or 0)
        pts = float(record.get("pts", 0) or 0)
        if gp <= 0:
            return 0.0
        return max(0.0, min(1.0, pts / max(1.0, gp * 2.0)))
    except Exception:
        return 0.0


def _roster_strength_score(roster_rows: List[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for row in roster_rows[:18]:
        try:
            ovr = float(row.get("ovr", 0) or 0)
            if ovr > 0:
                vals.append(ovr)
        except Exception:
            continue
    if not vals:
        return 0.0
    return round(sum(vals) / max(1, len(vals)), 2)


def _safe_ovr_from_row(row: Dict[str, Any]) -> float:
    try:
        ovr = float(row.get("ovr", 0) or 0)
    except Exception:
        ovr = 0.0
    if ovr <= 1.5:
        ovr *= 99.0
    return max(0.0, min(99.0, ovr))


def _player_ovr_99_scale(player: Any) -> float:
    """0–99 display scale for roster players (not sim _ovr_weight curve)."""
    try:
        fn = getattr(player, "ovr", None)
        o = float(fn() if callable(fn) else fn or 0)
    except Exception:
        return 0.0
    if o <= 1.5:
        o *= 99.0
    return max(0.0, min(99.0, round(o, 1)))


def _safe_age_from_row(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("age", 0) or 0)
    except Exception:
        return 0.0


def _pos_bucket(pos_raw: Any) -> str:
    pos = str(pos_raw or "").upper()
    if pos in ("LW", "RW", "W", "F", "C"):
        return "F"
    if pos in ("D", "LD", "RD"):
        return "D"
    if pos in ("G",):
        return "G"
    return "F"


def _team_component_scores(roster_rows: List[Dict[str, Any]], window: str = "") -> Dict[str, float]:
    forwards: List[float] = []
    defense: List[float] = []
    goalies: List[float] = []
    all_ovr: List[float] = []
    ages: List[float] = []

    for row in roster_rows:
        ovr = _safe_ovr_from_row(row)
        if ovr <= 0:
            continue
        all_ovr.append(ovr)
        age = _safe_age_from_row(row)
        if age > 0:
            ages.append(age)
        bucket = _pos_bucket(row.get("position"))
        if bucket == "F":
            forwards.append(ovr)
        elif bucket == "D":
            defense.append(ovr)
        elif bucket == "G":
            goalies.append(ovr)

    forwards.sort(reverse=True)
    defense.sort(reverse=True)
    goalies.sort(reverse=True)
    all_ovr.sort(reverse=True)

    top6_f = (sum(forwards[:6]) / max(1, min(6, len(forwards)))) if forwards else 0.0
    top4_d = (sum(defense[:4]) / max(1, min(4, len(defense)))) if defense else 0.0
    starter_g = goalies[0] if goalies else 0.0
    core12 = all_ovr[:12]
    depth_pool = all_ovr[12:] if len(all_ovr) > 12 else []
    depth_raw = (sum(depth_pool[:10]) / max(1, min(10, len(depth_pool)))) if depth_pool else (sum(core12) / max(1, len(core12)) if core12 else 0.0)
    elite_count = sum(1 for v in all_ovr if v >= 87)
    avg_age = (sum(ages) / len(ages)) if ages else 0.0

    elite_bonus = min(4.0, elite_count * 1.15)
    age_window_mod = 0.0
    w = str(window or "").lower()
    if w in ("contender", "playoff", "win_now"):
        if 26.0 <= avg_age <= 31.0:
            age_window_mod += 1.4
        elif avg_age < 24.0:
            age_window_mod -= 0.8
    elif w in ("rebuild", "rebuilding", "tanking", "tank"):
        if avg_age <= 25.0:
            age_window_mod += 0.6
        elif avg_age >= 30.0:
            age_window_mod -= 0.8

    strength = (
        top6_f * 0.36
        + top4_d * 0.28
        + starter_g * 0.20
        + depth_raw * 0.12
        + elite_bonus
        + age_window_mod
    )

    return {
        "strength": round(max(0.0, min(99.0, strength)), 2),
        "top6_forwards": round(top6_f, 2),
        "top4_defense": round(top4_d, 2),
        "starter_goalie": round(starter_g, 2),
        "depth_score": round(depth_raw, 2),
        "elite_count": float(elite_count),
        "elite_bonus": round(elite_bonus, 2),
        "avg_age": round(avg_age, 2),
        "age_window_mod": round(age_window_mod, 2),
    }


def _projected_points(cal_record: Dict[str, Any]) -> float:
    try:
        gp = float(cal_record.get("gp", 0) or 0)
        pts = float(cal_record.get("pts", 0) or 0)
        if gp <= 0:
            return 0.0
        return max(0.0, (pts / gp) * 82.0)
    except Exception:
        return 0.0


def _user_rank_snapshot(session: FranchiseSession) -> Dict[str, int]:
    uid = str(getattr(session, "user_team_id", "") or "")
    rows: List[Dict[str, Any]] = []
    if getattr(session, "standings", None):
        for tid, r in (session.standings.records or {}).items():
            rows.append(
                {
                    "team_id": str(tid),
                    "pts": int(getattr(r, "points", 0) or 0),
                    "w": int(getattr(r, "wins", 0) or 0),
                }
            )
    if not rows:
        return {"league_rank": 0}
    rows.sort(key=lambda x: (-x["pts"], -x["w"]))
    for i, row in enumerate(rows, start=1):
        if str(row.get("team_id")) == uid:
            return {"league_rank": i}
    return {"league_rank": 0}


def _recent_form_points_pct(session: FranchiseSession) -> float:
    uid = str(getattr(session, "user_team_id", "") or "")
    return _recent_form_points_pct_for_team(session, uid) if uid else 0.0


def _recent_form_points_pct_for_team(session: FranchiseSession, team_id: str) -> float:
    tid = str(team_id or "")
    if not tid:
        return 0.0
    recent: List[int] = []
    for g in reversed(list(getattr(session, "game_results", None) or [])):
        hid = str(g.get("home_id") or "")
        aid = str(g.get("away_id") or "")
        if tid not in (hid, aid):
            continue
        try:
            hg = int(g.get("home_goals") or 0)
            ag = int(g.get("away_goals") or 0)
        except (TypeError, ValueError):
            continue
        if hg == ag:
            continue
        ot = bool(g.get("overtime"))
        is_home = hid == tid
        us = hg if is_home else ag
        them = ag if is_home else hg
        pts = 2 if us > them else 1 if ot else 0
        recent.append(pts)
        if len(recent) >= 10:
            break
    if not recent:
        return 0.0
    return max(0.0, min(1.0, float(sum(recent)) / (2.0 * len(recent))))


def _team_cal_record_for_id(session: FranchiseSession, team_id: str) -> Dict[str, Any]:
    tid = str(team_id or "")
    uid = str(getattr(session, "user_team_id", "") or "")
    if tid and tid == uid:
        cal_rec = _user_team_record_from_game_results(session)
        if int(cal_rec.get("gp") or 0) > 0:
            return cal_rec
    standings = getattr(session, "standings", None)
    if standings is not None:
        rec = None
        if hasattr(standings, "find_record"):
            rec = standings.find_record(tid)
        if rec is None:
            rec = (getattr(standings, "records", None) or {}).get(tid)
        if rec is not None:
            return {
                "gp": int(getattr(rec, "gp", 0) or 0),
                "w": int(getattr(rec, "wins", 0) or 0),
                "l": int(getattr(rec, "losses", 0) or 0),
                "otl": int(getattr(rec, "otl", 0) or 0),
                "pts": int(getattr(rec, "points", 0) or 0),
            }
    return {"gp": 0, "w": 0, "l": 0, "otl": 0, "pts": 0}


def _roster_rows_for_playoff_outlook(team: Any, session: FranchiseSession) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in list(getattr(team, "roster", None) or []):
        if getattr(p, "retired", False):
            continue
        ovr = _player_ovr_99_scale(p)
        if ovr <= 0:
            continue
        rows.append(
            {
                "ovr": ovr,
                "position": _pos_str(p),
                "age": getattr(p, "age", 0),
                "is_injured": bool(_is_player_live_injured(p)),
                "injury_games_remaining": int(_get_live_injury_games_remaining(p)),
                "injury_tier": str(_get_live_injury_tier(p) or ""),
            }
        )
    return rows


def _playoff_score_from_pace(p_pct: float, proj_pts: float) -> float:
    if p_pct >= 0.68:
        pace_score = 97.0
    elif p_pct >= 0.64:
        pace_score = 91.0
    elif p_pct >= 0.60:
        pace_score = 82.0
    elif p_pct >= 0.56:
        pace_score = 72.0
    elif p_pct >= 0.52:
        pace_score = 60.0
    elif p_pct >= 0.48:
        pace_score = 48.0
    elif p_pct >= 0.44:
        pace_score = 36.0
    elif p_pct >= 0.40:
        pace_score = 24.0
    elif p_pct >= 0.36:
        pace_score = 14.0
    else:
        pace_score = 7.0

    if proj_pts >= 105.0:
        proj_score = 95.0
    elif proj_pts >= 100.0:
        proj_score = 86.0
    elif proj_pts >= 95.0:
        proj_score = 76.0
    elif proj_pts >= 90.0:
        proj_score = 64.0
    elif proj_pts >= 85.0:
        proj_score = 50.0
    elif proj_pts >= 80.0:
        proj_score = 38.0
    elif proj_pts >= 75.0:
        proj_score = 26.0
    else:
        proj_score = 12.0

    return 0.6 * pace_score + 0.4 * proj_score


def _strength_to_playoff_component(strength: float) -> float:
    s = float(strength or 0.0)
    if s <= 0:
        return 40.0
    return max(5.0, min(98.0, (s - 55.0) * 2.2 + 10.0))


def _compute_injury_impact_from_team(team: Any) -> float:
    impact = 0.0
    for p in list(getattr(team, "roster", None) or []):
        if not _is_player_live_injured(p):
            continue
        ovr = _player_ovr_99_scale(p)
        gr = int(_get_live_injury_games_remaining(p))
        tier = str(_get_live_injury_tier(p) or "minor").lower()
        pos = _pos_bucket(_pos_str(p))
        tier_mult = {"major": 1.0, "moderate": 0.75, "minor": 0.45}.get(tier, 0.55)
        duration_mult = min(1.0, gr / 20.0) if gr > 0 else 0.5
        if pos == "G":
            base_pen = min(8.0, ovr * 0.08)
        elif ovr >= 86.0:
            base_pen = min(6.0, ovr * 0.055)
        elif ovr >= 80.0:
            base_pen = min(4.0, ovr * 0.035)
        else:
            base_pen = min(2.5, ovr * 0.02)
        impact += base_pen * tier_mult * (0.55 + 0.45 * duration_mult)
    return round(min(18.0, impact), 2)


def _standings_playoff_component(
    session: FranchiseSession,
    team_id: str,
    cal_record: Dict[str, Any],
    p_pct: float,
    proj_pts: float,
) -> tuple[float, Dict[str, Any]]:
    tid = str(team_id or "")
    score = _playoff_score_from_pace(p_pct, proj_pts)
    gp = int(cal_record.get("gp") or 0)
    context: Dict[str, Any] = {
        "points_pct": round(p_pct, 3),
        "projected_points": round(proj_pts, 1),
        "games_played": gp,
        "games_remaining": max(0, 82 - gp) if gp > 0 else 82,
        "league_rank": None,
        "conference_rank": None,
        "division_rank": None,
        "wildcard_rank": None,
        "playoff_status": None,
        "is_playoff_team": False,
        "is_bubble": False,
    }

    standings = getattr(session, "standings", None)
    if standings is None:
        return score, context

    try:
        ranks = standings.rank_maps() if hasattr(standings, "rank_maps") else {}
        status_map = standings.playoff_status_map() if hasattr(standings, "playoff_status_map") else {}
        status = dict(status_map.get(tid) or {})
        context.update(
            {
                "league_rank": ranks.get("league", {}).get(tid),
                "conference_rank": ranks.get("conference", {}).get(tid),
                "division_rank": ranks.get("division", {}).get(tid),
                "wildcard_rank": status.get("wildcard_rank"),
                "playoff_status": status.get("playoff_status"),
                "is_playoff_team": bool(status.get("is_playoff_team")),
                "is_bubble": bool(status.get("is_bubble")),
            }
        )

        if status.get("is_playoff_team"):
            conf_rank = int(status.get("conference_rank") or 99)
            div_rank = int(status.get("division_rank") or 99)
            if div_rank <= 1:
                score = max(score, 92.0)
            elif conf_rank <= 4:
                score = max(score, 86.0)
            else:
                score = max(score, 78.0)
        elif status.get("is_bubble"):
            wc = int(status.get("wildcard_rank") or 5)
            score = max(score, max(22.0, 52.0 - (wc - 3) * 7.0))
        elif status.get("is_eliminated"):
            score = min(score, 8.0)
        elif str(status.get("playoff_status") or "") == "longshot":
            score = min(score, 18.0)

        league_rank = int(ranks.get("league", {}).get(tid) or 32)
        gp = int(cal_record.get("gp") or 0)
        rank_weight = min(1.0, gp / 24.0) if gp > 0 else 0.4
        if league_rank <= 8:
            score = max(score, 78.0 + 10.0 * rank_weight)
        elif league_rank <= 16:
            score = max(score, 58.0 + 12.0 * rank_weight)
        elif league_rank >= 28:
            score = min(score, 18.0)
    except Exception:
        pass

    return max(0.0, min(99.0, score)), context


def _contention_label_from_outlook(
    odds: int,
    health_adj: float,
    direction: str,
    p_pct: float,
    strength: float,
) -> str:
    d = str(direction or "").lower()
    if odds >= 90 and health_adj >= 86.0:
        return "CUP THREAT"
    if odds >= 75:
        return "CONTENDER"
    if odds >= 55:
        return "PLAYOFF"
    if odds >= 25:
        return "BUBBLE"
    if odds < 10 and p_pct < 0.42:
        return "TANKING"
    if odds < 25:
        if d in ("rebuild", "rebuilding", "emerging"):
            return "REBUILDER"
        if d in ("seller", "declining", "tank", "tanking"):
            return "SELLER" if d in ("seller", "declining") else "TANKING"
        if strength < 76.0 and p_pct < 0.46:
            return "REBUILDER"
        return "SELLER"
    return "BUBBLE"


def compute_team_playoff_outlook(session: FranchiseSession, team: Any) -> Dict[str, Any]:
    """
    Franchise-wide playoff odds + outlook for trade hub / team cards.
    Weights: standings 55%, strength 25%, health 12%, recent form 8%.
    """
    tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
    direction = str(
        getattr(team, "gm_window", None) or getattr(team, "window", None) or "unknown"
    ).strip().lower()
    cal_record = _team_cal_record_for_id(session, tid)
    roster_rows = _roster_rows_for_playoff_outlook(team, session)
    comp = _team_component_scores(roster_rows, window=direction)
    strength = float(comp.get("strength", 0.0))
    injury_impact = _compute_injury_impact_from_team(team)
    health_adjusted = round(max(0.0, min(99.0, strength - injury_impact)), 2)
    p_pct = _safe_points_pct(cal_record)
    proj_pts = _projected_points(cal_record)
    points_pace = round(float(cal_record.get("pts", 0) or 0) / max(1.0, float(cal_record.get("gp", 0) or 0)), 3)
    recent_form = _recent_form_points_pct_for_team(session, tid)

    standings_score, standings_context = _standings_playoff_component(
        session, tid, cal_record, p_pct, proj_pts,
    )
    gp = int(cal_record.get("gp") or 0)
    gp_blend = min(1.0, gp / 22.0) if gp > 0 else 0.45
    standings_score = standings_score * gp_blend + 46.0 * (1.0 - gp_blend)

    strength_score = _strength_to_playoff_component(strength)
    health_score = _strength_to_playoff_component(health_adjusted)
    form_score = recent_form * 100.0

    raw_odds = (
        standings_score * 0.55
        + strength_score * 0.25
        + health_score * 0.12
        + form_score * 0.08
    )

    # Strong roster / weak record → cap blind contender labels via odds ceiling.
    if strength >= 86.0 and p_pct < 0.50:
        raw_odds = min(raw_odds, 68.0)
    elif strength < 74.0 and p_pct >= 0.56:
        raw_odds = min(raw_odds, 62.0)

    odds = int(max(0, min(99, round(raw_odds * 1.08 + 5))))
    label = _contention_label_from_outlook(odds, health_adjusted, direction, p_pct, strength)

    standings_context.update(
        {
            "offense_rating": comp.get("top6_forwards"),
            "defense_rating": comp.get("top4_defense"),
            "goalie_rating": comp.get("starter_goalie"),
            "team_strength": strength,
            "recent_form_pct": round(recent_form, 3),
            "team_direction": direction,
        }
    )

    return {
        "playoff_odds": odds,
        "playoff_pct": odds,
        "outlook_label": label,
        "contention_label": label,
        "team_status": label,
        "health_adjusted_rating": health_adjusted,
        "injury_impact": injury_impact,
        "points_pace": points_pace,
        "projected_points": round(proj_pts, 1),
        "standings_context": standings_context,
    }


def _team_status_payload(
    session: FranchiseSession,
    user_team: Any,
    cal_record: Dict[str, Any],
    roster_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    window = str(
        getattr(user_team, "gm_window", None)
        or getattr(user_team, "window", None)
        or ""
    ).strip().lower()
    comp = _team_component_scores(roster_rows, window=window)
    strength = float(comp.get("strength", 0.0))
    top6_f = float(comp.get("top6_forwards", 0.0))
    top4_d = float(comp.get("top4_defense", 0.0))
    starter_g = float(comp.get("starter_goalie", 0.0))
    elite_count = int(comp.get("elite_count", 0.0))
    p_pct = _safe_points_pct(cal_record)
    proj_pts = _projected_points(cal_record)
    recent_form = _recent_form_points_pct(session)
    rank = int(_user_rank_snapshot(session).get("league_rank", 0) or 0)
    # Record/roster driven baseline with composition checks.
    contender_window = window in ("contender", "playoff", "win_now")
    if (
        (p_pct >= 0.62 or proj_pts >= 102.0)
        and strength >= 84.0
        and elite_count >= 1
        and starter_g >= 77.0
        and contender_window
    ):
        base = {
            "key": "cup_contender",
            "label": "Cup Contender",
            "confidence": 0.86,
            "reason": "High pace, strong weighted roster composition, and contender window.",
        }
    elif (
        (p_pct >= 0.54 or proj_pts >= 89.0)
        and strength >= 80.0
        and top4_d >= 77.0
        and starter_g >= 74.0
    ):
        base = {
            "key": "playoff_contender",
            "label": "Playoff Contender",
            "confidence": 0.79,
            "reason": "Competitive pace with stable top-4 defense and goaltending baseline.",
        }
    elif (
        p_pct < 0.38
        and strength < 74.0
        and elite_count <= 1
        and (rank == 0 or rank >= 26)
    ):
        base = {
            "key": "tanking",
            "label": "Tanking",
            "confidence": 0.82,
            "reason": "Low pace, weak roster composition, and bottom-table profile.",
        }
    elif (
        (p_pct < 0.46 or strength < 76.0 or proj_pts < 82.0)
        and window in ("rebuild", "rebuilding", "declining", "emerging")
    ):
        base = {
            "key": "rebuilding",
            "label": "Rebuilding",
            "confidence": 0.76,
            "reason": "Weak current profile with development/future-focused window.",
        }
    else:
        base = {
            "key": "middling",
            "label": "Middling",
            "confidence": 0.7,
            "reason": "Middle-tier pace and roster composition without clear contender/tank indicators.",
        }

    # Window influences only where it is directionally safe; it cannot promote to Cup Contender by itself.
    if window in ("rebuild", "rebuilding", "declining"):
        if base["key"] in ("cup_contender", "playoff_contender"):
            return {
                "key": "middling",
                "label": "Middling",
                "confidence": 0.7,
                "reason": f"Window={window} moderates contender status despite record profile.",
            }
        return {
            **base,
            "reason": f"{base['reason']} Window={window}.",
        }
    if window in ("tank", "tanking"):
        return {
            "key": "tanking",
            "label": "Tanking",
            "confidence": max(0.75, float(base.get("confidence", 0.7))),
            "reason": f"Window={window} with performance profile considered.",
        }
    if window in ("contender", "playoff"):
        if base["key"] == "cup_contender" and not (p_pct >= 0.60 and strength >= 82):
            return {
                "key": "playoff_contender",
                "label": "Playoff Contender",
                "confidence": 0.76,
                "reason": "Contender window but profile below Cup tier threshold.",
            }
        return {
            **base,
            "reason": f"{base['reason']} Window={window}.",
        }
    return {
        **base,
        "metrics": {
            "points_pct": round(p_pct, 3),
            "projected_points": round(proj_pts, 1),
            "league_rank": rank,
            "recent_form_pct": round(recent_form, 3),
            "team_strength": round(strength, 2),
            "top6_forwards": round(top6_f, 2),
            "top4_defense": round(top4_d, 2),
            "starting_goalie": round(starter_g, 2),
            "elite_count": elite_count,
            "window": window,
        },
    }


def _event_days_payload(
    event_key: str,
    label: str,
    now_iso: str,
    season_year: int,
) -> Dict[str, Any]:
    now_day = None
    try:
        if now_iso:
            now_day = date.fromisoformat(str(now_iso)[:10])
    except Exception:
        now_day = None

    markers = season_anchor_event_markers(int(season_year))
    row = next((m for m in markers if str(m.get("key")) == event_key), None)
    fallback_dates = {
        "wjc_start": f"{int(season_year)}-12-26",
        "draft_lottery": f"{int(season_year) + 1}-05-07",
        "draft": f"{int(season_year) + 1}-06-27",
    }
    raw_date = str((row or {}).get("date") or fallback_dates.get(event_key, ""))
    event_day = None
    try:
        if raw_date:
            event_day = date.fromisoformat(raw_date[:10])
    except Exception:
        event_day = None

    if now_day and event_day and event_day < now_day:
        next_markers = season_anchor_event_markers(int(season_year) + 1)
        next_row = next((m for m in next_markers if str(m.get("key")) == event_key), None)
        next_date = str((next_row or {}).get("date") or "")
        try:
            if next_date:
                event_day = date.fromisoformat(next_date[:10])
                raw_date = next_date
        except Exception:
            pass

    if not now_day or not event_day:
        return {"label": label, "date": raw_date or "", "days_until": None, "display": "—"}
    delta = (event_day - now_day).days
    if delta < 0:
        display = "PASSED"
    elif delta == 0:
        display = "TODAY"
    else:
        display = f"{delta} DAYS"
    return {"label": label, "date": raw_date, "days_until": int(delta), "display": display}


def _build_draft_class_hud_payload(
    session: FranchiseSession,
    user_team: Any,
    cal_record: Dict[str, Any],
    roster_rows: List[Dict[str, Any]],
    draft_entries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    now_iso = (
        _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
        or _today_iso(session)
        or str(getattr(session, "current_date", "") or "")
    )
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    team_status = _team_status_payload(session, user_team, cal_record, roster_rows)
    wjc_event = _event_days_payload("wjc_start", "WJC", now_iso, season_year)
    wjc_event.update(_wjc_hud_event_extras(session, now_iso, season_year))
    if wjc_event.get("display_override"):
        wjc_event["display"] = str(wjc_event["display_override"])
    payload: Dict[str, Any] = {
        "team_status": team_status,
        "events": {
            "wjc": wjc_event,
            "lottery": _event_days_payload("draft_lottery", "Lottery", now_iso, season_year),
            "draft": _event_days_payload("draft", "Draft", now_iso, season_year),
        },
    }
    if draft_entries is not None:
        try:
            from services.draft_prospect_profile import build_prospect_profiles_by_id

            payload["prospect_profiles_by_id"] = build_prospect_profiles_by_id(
                draft_entries,
                roster_rows=roster_rows,
                team_status=team_status,
            )
            hist = getattr(session, "draft_stock_history", None) or {}
            for pid, prof in (payload.get("prospect_profiles_by_id") or {}).items():
                rows = hist.get(str(pid))
                synthesized = list(prof.get("rankHistory") or prof.get("stock_history") or [])
                live = list(rows) if isinstance(rows, list) else []
                # Prefer a live trail whenever it has at least two samples; frontend
                # compresses flat tails so early rises still fill the chart.
                if len(live) >= 2:
                    prof["stock_history"] = live
                    prof["rankHistory"] = live
                elif live and not synthesized:
                    prof["stock_history"] = live
                    prof["rankHistory"] = live
                # else keep synthesized Preseason/Midseason/Current checkpoints.
        except Exception:
            payload["prospect_profiles_by_id"] = {}
    return payload


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


# ---------------------------------------------------------------------------
# Trade Hub Fan Heat / Fan Reaction — persistent franchise system
# ---------------------------------------------------------------------------

def _clamp_fan_score(value: float, low: int = 0, high: int = 100) -> int:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 50.0
    return int(max(low, min(high, round(v))))


def _get_team_fan_profile(session: FranchiseSession, team_id: str) -> Dict[str, Any]:
    return _ensure_team_fan_profile(session, team_id)


def _ensure_team_fan_profile(session: FranchiseSession, team_id: str) -> Dict[str, Any]:
    profiles = getattr(session, "fan_profiles", None)
    if not isinstance(profiles, dict):
        profiles = {}
        session.fan_profiles = profiles
    tid = str(team_id or "")
    existing = profiles.get(tid)
    if isinstance(existing, dict) and existing.get("fan_confidence") is not None:
        for k, default in (
            ("fan_patience", 55.0),
            ("fan_trust_in_gm", 58.0),
            ("recent_trade_heat", 0.0),
            ("season_backlash_events", []),
            ("trade_reaction_history", []),
            ("last_major_trade_reaction", None),
            ("last_decay_week", -1),
        ):
            if k not in existing:
                existing[k] = default if not isinstance(default, list) else list(default)
        return existing

    team = session.team_by_id.get(tid)
    outlook: Dict[str, Any] = {}
    try:
        outlook = compute_team_playoff_outlook(session, team) if team is not None else {}
    except Exception:
        outlook = {}
    odds = int(outlook.get("playoff_odds", 50) or 50)
    direction = str(outlook.get("team_direction", getattr(team, "gm_window", "unknown")) or "unknown").lower()
    base_conf = _clamp_fan_score(42 + odds * 0.36, 45, 78)
    if direction in ("rebuild", "rebuilding", "tanking", "seller"):
        patience = 62.0
        trust = 54.0
    elif direction in ("contender", "playoff", "cup threat"):
        patience = 46.0
        trust = 60.0
    else:
        patience = 55.0
        trust = 58.0
    profiles[tid] = {
        "fan_confidence": float(base_conf),
        "fan_patience": float(patience),
        "fan_trust_in_gm": float(trust),
        "recent_trade_heat": 0.0,
        "season_backlash_events": [],
        "trade_reaction_history": [],
        "last_major_trade_reaction": None,
        "last_decay_week": -1,
    }
    return profiles[tid]


def _fan_player_ovr99(player: Any) -> float:
    from app.sim_engine.engine import _franchise_player_ovr99  # noqa: WPS433

    return _franchise_player_ovr99(player)


def _compute_player_fan_attachment(
    player: Any,
    team: Any,
    session: FranchiseSession,
) -> float:
    from app.sim_engine.engine import _franchise_player_fan_attachment  # noqa: WPS433

    pid = str(getattr(player, "id", "") or "")
    stats = dict(getattr(session, "player_season_stats", None) or {}).get(pid) or {}
    return _franchise_player_fan_attachment(player, team, session_stats=stats)


def _fan_category_from_score(score: int) -> str:
    s = int(score)
    if s >= 80:
        return "Fan Favorite Move"
    if s >= 60:
        return "Supportive"
    if s >= 40:
        return "Mixed"
    if s >= 20:
        return "Backlash Risk"
    return "PR Disaster"


def _fan_heat_label_from_heat(heat: int) -> str:
    h = int(heat)
    if h >= 75:
        return "Furious"
    if h >= 55:
        return "Backlash"
    if h >= 30:
        return "Uneasy"
    return "Calm"


def _fan_resolve_player(session: FranchiseSession, team_id: str, pid: str) -> Optional[Any]:
    team = session.team_by_id.get(str(team_id))
    if team is None:
        return None
    for p in getattr(team, "roster", None) or []:
        if str(getattr(p, "id", "") or "") == str(pid):
            return p
    return None


def _fan_split_trade_assets(
    assets_by_team: Dict[str, Any],
    user_team_id: str,
    partner_team_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    utid = str(user_team_id or "")
    outgoing: List[Dict[str, Any]] = []
    incoming: List[Dict[str, Any]] = []
    for tid, assets in (assets_by_team or {}).items():
        for raw in assets or []:
            if not isinstance(raw, dict):
                continue
            src = str(raw.get("team") or tid or "")
            row = dict(raw)
            row.setdefault("team", src)
            if src == utid:
                outgoing.append(row)
            elif utid and src != utid:
                incoming.append(row)
            elif partner_team_id and str(tid) == str(partner_team_id):
                incoming.append(row)
    return outgoing, incoming


def _fan_is_division_rival(session: FranchiseSession, user_team: Any, partner_id: str) -> bool:
    if user_team is None or not partner_id:
        return False
    partner = session.team_by_id.get(str(partner_id))
    if partner is None:
        return False
    udiv = str(getattr(user_team, "division", "") or "")
    pdiv = str(getattr(partner, "division", "") or "")
    return bool(udiv and pdiv and udiv == pdiv)


def _fan_outgoing_player_impact(
    session: FranchiseSession,
    team: Any,
    player: Any,
    asset: Dict[str, Any],
    outlook: Dict[str, Any],
    tolerance: Dict[str, float],
    *,
    rival: bool,
) -> Tuple[float, List[str]]:
    factors: List[str] = []
    ovr = _fan_player_ovr99(player)
    ident = getattr(player, "identity", None)
    age = int(getattr(ident, "age", getattr(player, "age", 26)) or 26)
    attach = _compute_player_fan_attachment(player, team, session)
    loss = attach * 0.22
    if ovr >= 88:
        loss += 18.0 * float(tolerance.get("sell_star_penalty", 1.2))
        factors.append("Star player")
    elif ovr >= 84:
        loss += 12.0
        factors.append("Fan favorite")
    elif ovr >= 80:
        loss += 7.0
    if age <= 24 and ovr >= 78:
        loss += 9.0 * float(tolerance.get("sell_youth_penalty", 1.25))
        factors.append("Young core")
    if age >= 30 and ovr >= 80 and outlook.get("team_direction") in ("rebuild", "rebuilding", "tanking", "seller"):
        loss *= float(tolerance.get("sell_veteran_relief", 0.85))
    if bool(getattr(player, "is_captain", False)) or str(getattr(player, "captaincy", "") or "").upper() in ("C", "CAPTAIN"):
        loss += 14.0
        factors.append("Captain")
    pst = getattr(player, "_franchise_storyline_state", None) or {}
    if bool(pst.get("was_recently_shopped")):
        loss += 3.0
    if rival and attach >= 55:
        loss += 4.0
        factors.append("Rival")
    return loss, factors


def _fan_incoming_player_impact(
    session: FranchiseSession,
    team: Any,
    player: Any,
    outlook: Dict[str, Any],
    tolerance: Dict[str, float],
) -> Tuple[float, List[str]]:
    factors: List[str] = []
    ovr = _fan_player_ovr99(player)
    ident = getattr(player, "identity", None)
    age = int(getattr(ident, "age", getattr(player, "age", 26)) or 26)
    gain = 0.0
    direction = str(outlook.get("team_direction", "") or "").lower()
    if ovr >= 88:
        gain += 16.0 * float(tolerance.get("buy_star_bonus", 1.1))
        factors.append("Star In")
    elif ovr >= 82:
        gain += 8.0
    if age <= 24 and ovr >= 78:
        gain += 7.0 * float(tolerance.get("buy_prospect_bonus", 1.0))
        factors.append("Prospect In")
    if direction in ("contender", "playoff") and age >= 28 and ovr >= 84:
        gain += 5.0
        factors.append("Playoff rental")
    if direction in ("rebuild", "rebuilding", "tanking") and age >= 32 and ovr >= 80:
        gain -= 6.0
        factors.append("Aging contract")
    return gain, factors


def _fan_outgoing_pick_impact(
    asset: Dict[str, Any],
    outlook: Dict[str, Any],
    tolerance: Dict[str, float],
) -> Tuple[float, List[str]]:
    factors: List[str] = []
    rnd = int(asset.get("round") or asset.get("pick_round") or 0)
    loss = 0.0
    direction = str(outlook.get("team_direction", "") or "").lower()
    if rnd == 1:
        loss = 12.0
        factors.append("1st Pick")
        if direction in ("rebuild", "rebuilding", "tanking", "seller"):
            loss += 6.0
    elif rnd == 2:
        loss = 4.0
        factors.append("2nd Pick")
    elif rnd >= 3:
        loss = 1.5
    return loss, factors


def _fan_incoming_pick_impact(asset: Dict[str, Any], outlook: Dict[str, Any]) -> Tuple[float, List[str]]:
    factors: List[str] = []
    rnd = int(asset.get("round") or asset.get("pick_round") or 0)
    direction = str(outlook.get("team_direction", "") or "").lower()
    gain = 0.0
    if rnd == 1:
        gain = 10.0
        factors.append("1st In")
        if direction in ("rebuild", "rebuilding", "tanking"):
            gain += 4.0
    elif rnd == 2:
        gain = 3.0
    return gain, factors


def _fan_compute_effects(score: int, heat: int, factors: List[str], profile: Dict[str, Any]) -> Dict[str, float]:
    backlash_count = len(list(profile.get("season_backlash_events") or []))
    if heat >= 75:
        return {
            "fan_confidence_delta": -12.0 - min(3.0, backlash_count * 0.5),
            "owner_patience_delta": -7.0 - min(2.0, backlash_count * 0.4),
            "gm_trust_delta": -10.0,
            "team_morale_delta": -3.5,
            "attendance_delta": -2.0,
            "pressure_delta": 8.0,
        }
    if heat >= 55:
        return {
            "fan_confidence_delta": -7.0,
            "owner_patience_delta": -4.0,
            "gm_trust_delta": -6.0,
            "team_morale_delta": -2.0,
            "attendance_delta": -1.0,
            "pressure_delta": 5.0,
        }
    if score >= 80:
        return {
            "fan_confidence_delta": 8.0,
            "owner_patience_delta": 4.0,
            "gm_trust_delta": 6.0,
            "team_morale_delta": 2.5,
            "attendance_delta": 1.5,
            "pressure_delta": -3.0,
        }
    if score >= 65:
        return {
            "fan_confidence_delta": 3.0,
            "owner_patience_delta": 1.0,
            "gm_trust_delta": 2.0,
            "team_morale_delta": 1.0,
            "attendance_delta": 0.5,
            "pressure_delta": -1.0,
        }
    if "Weak return" in factors or heat >= 40:
        return {
            "fan_confidence_delta": -3.0,
            "owner_patience_delta": -1.0,
            "gm_trust_delta": -2.0,
            "team_morale_delta": -0.5,
            "attendance_delta": 0.0,
            "pressure_delta": 2.0,
        }
    return {
        "fan_confidence_delta": 0.0,
        "owner_patience_delta": 0.0,
        "gm_trust_delta": 0.0,
        "team_morale_delta": 0.0,
        "attendance_delta": 0.0,
        "pressure_delta": 0.0,
    }


def _compute_trade_fan_reaction(
    session: FranchiseSession,
    user_team_id: str,
    assets_by_team: Dict[str, Any],
    evaluation_result: Optional[Dict[str, Any]] = None,
    *,
    completed: bool = False,
    partner_team_id: Optional[str] = None,
    trade_id: Optional[str] = None,
) -> Dict[str, Any]:
    utid = str(user_team_id or getattr(session, "user_team_id", "") or "")
    empty = {
        "fan_reaction_score": 50,
        "fan_heat": 50,
        "fan_category": "Mixed",
        "fan_heat_label": "Uneasy",
        "fan_factors": [],
        "fan_effects": _fan_compute_effects(50, 50, [], {}),
        "fan_storyline_type": None,
        "fan_headline": "",
        "fan_summary": "",
        "should_persist": False,
    }
    if not utid:
        return empty

    profile = _ensure_team_fan_profile(session, utid)
    team = session.team_by_id.get(utid)
    outlook: Dict[str, Any] = {}
    try:
        outlook = compute_team_playoff_outlook(session, team) if team is not None else {}
    except Exception:
        outlook = {}

    from app.sim_engine.engine import _franchise_team_window_fan_tolerance  # noqa: WPS433

    tolerance = _franchise_team_window_fan_tolerance(team) if team is not None else {}
    outgoing, incoming = _fan_split_trade_assets(assets_by_team, utid, partner_team_id)
    rival = _fan_is_division_rival(session, team, str(partner_team_id or ""))

    score = float(profile.get("fan_confidence", 55.0))
    factors: List[str] = []

    for asset in outgoing:
        atype = str(asset.get("type") or "").lower()
        if atype == "pick":
            loss, facs = _fan_outgoing_pick_impact(asset, outlook, tolerance)
            score -= loss
            factors.extend(facs)
            continue
        pid = str(asset.get("id") or "")
        pl = _fan_resolve_player(session, utid, pid)
        if pl is None:
            continue
        loss, facs = _fan_outgoing_player_impact(
            session, team, pl, asset, outlook, tolerance, rival=rival,
        )
        score -= loss
        factors.extend(facs)

    for asset in incoming:
        atype = str(asset.get("type") or "").lower()
        if atype == "pick":
            gain, facs = _fan_incoming_pick_impact(asset, outlook)
            score += gain
            factors.extend(facs)
            continue
        pid = str(asset.get("id") or "")
        src = str(asset.get("team") or partner_team_id or "")
        pl = _fan_resolve_player(session, src, pid)
        if pl is None:
            for tm in session.team_by_id.values():
                pl = _fan_resolve_player(session, str(getattr(tm, "team_id", getattr(tm, "id", ""))), pid)
                if pl is not None:
                    break
        if pl is None:
            continue
        gain, facs = _fan_incoming_player_impact(session, team, pl, outlook, tolerance)
        score += gain
        factors.extend(facs)

    if evaluation_result:
        user_net = float((evaluation_result.get("asset_breakdown") or {}).get("user", {}).get("net", 0) or 0)
        if evaluation_result.get("accepted") and user_net >= -2:
            score += 4.0
            factors.append("Fair deal")
        elif user_net < -10:
            score -= 8.0
            factors.append("Weak return")
        elif user_net < -6:
            score -= 4.0

    cal = getattr(session, "nhl_calendar", None) or []
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    if cur > 0 and outlook.get("recent_form_pct") is not None:
        form = float(outlook.get("recent_form_pct") or 0.5)
        if form < 0.38:
            score -= 3.0
        elif form >= 0.58:
            score += 2.0

    score_i = _clamp_fan_score(score, 8, 98)
    recent_heat = float(profile.get("recent_trade_heat", 0) or 0)
    fan_heat = _clamp_fan_score((100 - score_i) + recent_heat * 0.15)
    fan_factors = list(dict.fromkeys(factors))[:8]
    fan_effects = _fan_compute_effects(score_i, fan_heat, fan_factors, profile)
    category = _fan_category_from_score(score_i)
    heat_label = _fan_heat_label_from_heat(fan_heat)

    storyline_type = None
    headline = ""
    summary = ""
    if fan_heat >= 75:
        storyline_type = "FAN_BACKLASH"
        if "Captain" in fan_factors:
            headline = "Fans erupt after captain is traded."
            summary = "Captaincy move shocks the fanbase."
        elif "Star player" in fan_factors:
            storyline_type = "STAR_TRADED_SHOCK"
            headline = "Star traded — fanbase in uproar."
            summary = "Market questions the franchise direction."
        else:
            headline = "Backlash building after unpopular trade."
            summary = "Social channels turn on the front office."
    elif fan_heat >= 55:
        storyline_type = "FAN_BACKLASH"
        headline = "Fanbase split after blockbuster move."
        summary = "Supporters and critics clash online."
    elif score_i >= 80:
        storyline_type = "FANS_LOVE_THE_MOVE"
        headline = "Supporters praise bold hockey move."
        summary = "Ticket talk trends positive after the deal."
    elif len(profile.get("season_backlash_events") or []) >= 2 and fan_heat >= 45:
        storyline_type = "OWNER_CONCERNED"
        headline = "Ownership watching fan unrest closely."
        summary = "Another controversial move tests patience."

    iso = str(cal[cur].get("iso") or "") if 0 <= cur < len(cal) else str(cur)

    return {
        "fan_reaction_score": score_i,
        "fan_heat": fan_heat,
        "fan_category": category,
        "fan_heat_label": heat_label,
        "fan_factors": fan_factors,
        "fan_effects": fan_effects,
        "fan_storyline_type": storyline_type,
        "fan_headline": headline or category,
        "fan_summary": summary or f"Fans rate this move: {category}.",
        "should_persist": bool(completed),
        "trade_id": str(trade_id or ""),
        "calendar_day": cur,
        "calendar_iso": iso,
    }


def _tr_player_display_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    return str(getattr(player, "name", None) or getattr(ident, "name", None) or getattr(player, "id", "") or "Player")


def _tr_need_to_chip(label: str) -> str:
    low = str(label or "").lower()
    if "forward" in low and "top" in low:
        return "TOP 6"
    if "defense" in low or "defence" in low:
        return "DEFENSE"
    if "goalie" in low or "goaltend" in low:
        return "GOALIE"
    if "pick" in low:
        return "PICKS"
    if "prospect" in low:
        return "PROSPECT"
    if "cap" in low:
        return "CAP"
    if "rental" in low:
        return "RENTAL"
    if "depth" in low:
        return "DEPTH"
    tok = str(label or "").upper().replace("-", " ").strip()
    return tok[:12] if tok else ""


def _tr_format_asset_label(a: Dict[str, Any]) -> str:
    if not isinstance(a, dict):
        return ""
    nm = str(a.get("name") or "").strip()
    asset_type = str(a.get("type") or "").lower()
    if asset_type == "pick" or "-round" in nm.lower():
        yr = a.get("year", "")
        rnd = a.get("round", "")
        if yr and rnd:
            return f"{yr} R{rnd}"
        raw = nm or str(a.get("asset_id") or a.get("pick_id") or "")
        low = raw.lower()
        if "-round" in low:
            head, tail = low.split("-round", 1)
            rnd_part = tail.split("-", 1)[0]
            if head.isdigit() and rnd_part.isdigit():
                return f"{head} R{rnd_part}"
        return raw
    return nm


def _tr_asset_labels_from_breakdown(
    evaluation: Dict[str, Any],
    *,
    side: str = "user",
    direction: str,
    limit: int = 4,
) -> List[str]:
    bd = dict((evaluation.get("asset_breakdown") or {}).get(side) or {})
    labels: List[str] = []
    for a in safe_list_tr(bd.get(direction)):
        lbl = _tr_format_asset_label(a)
        if lbl:
            labels.append(lbl)
    return labels[:limit]


def _tr_player_labels_from_breakdown(
    evaluation: Dict[str, Any],
    *,
    side: str = "user",
    direction: str,
    limit: int = 4,
) -> List[str]:
    bd = dict((evaluation.get("asset_breakdown") or {}).get(side) or {})
    labels: List[str] = []
    for a in safe_list_tr(bd.get(direction)):
        if not isinstance(a, dict):
            continue
        if str(a.get("type") or "").lower() != "player":
            continue
        lbl = _tr_format_asset_label(a)
        if lbl:
            labels.append(lbl)
    return labels[:limit]


def _tr_dedupe_labels(labels: List[str], limit: int = 4) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for lbl in labels:
        key = str(lbl or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(str(lbl).strip())
    return out[:limit]


def _tr_clause_blocked_players(evaluation: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    clause_imp = evaluation.get("clause_impact") or {}
    if isinstance(clause_imp, dict):
        for msgs in clause_imp.values():
            for m in safe_list_tr(msgs):
                part = str(m).split(":", 1)[0].strip()
                if part:
                    names.append(part)
    return list(dict.fromkeys(names))[:3]


def _tr_partner_shopping_targets(
    session: FranchiseSession,
    partner_team_id: str,
    user_team_id: str,
    needs_impact: Dict[str, Any],
    *,
    limit: int = 3,
) -> List[str]:
    user_team = session.team_by_id.get(str(user_team_id or ""))
    if user_team is None:
        return []
    priority = [str(n).lower() for n in safe_list_tr(needs_impact.get("priority_needs"))]
    need_def = any("defense" in n or "defence" in n for n in priority)
    need_fwd = any("forward" in n or "top-line" in n or "top line" in n for n in priority)
    need_g = any("goalie" in n or "goaltend" in n for n in priority)
    ranked: List[Tuple[float, str]] = []
    for p in getattr(user_team, "roster", None) or []:
        if getattr(p, "retired", False):
            continue
        pos = str(
            getattr(p, "position", None)
            or getattr(getattr(p, "identity", None), "position", None)
            or ""
        ).upper()
        ovr = float(_fan_player_ovr99(p))
        match = not priority
        if need_g and pos == "G":
            match = True
        elif need_def and pos in ("D", "LD", "RD"):
            match = True
        elif need_fwd and pos in ("C", "LW", "RW", "W", "F"):
            match = True
        if match:
            ranked.append((ovr, _tr_player_display_name(p)))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in ranked[:limit]]


def _tr_balance_label(user_out: float, user_in: float) -> Tuple[str, int, str]:
    left = float(user_out or 0)
    right = float(user_in or 0)
    diff = abs(left - right)
    total = max(left + right, 1.0)
    ratio = diff / total
    if left <= 0 and right <= 0:
        return "EVEN", 50, "No assets in package."
    if ratio <= 0.08:
        return "EVEN", 50, "Package is near fair."
    score = 50
    if left > right:
        score = int(_clamp_fan_score(50 + ratio * 100, 55, 95))
        if ratio >= 0.35:
            return "HEAVY", score, "User side gives more."
        if ratio >= 0.18:
            return "YOU PAY", score, "User side gives more."
        return "LIGHT", score, "User side gives slightly more."
    score = int(_clamp_fan_score(50 - ratio * 100, 5, 45))
    if ratio >= 0.35:
        return "STEAL", score, "User side receives more."
    if ratio >= 0.18:
        return "THEY PAY", score, "User side receives more."
    return "CLOSE", score, "Package is near fair."


def _tr_why_block(
    evaluation: Dict[str, Any],
    *,
    user_team_id: str = "",
    partner_team_id: str = "",
) -> Dict[str, Any]:
    blocking = list(evaluation.get("rejection_reasons") or [])
    warnings = list(evaluation.get("warnings") or [])
    blob = " ".join(blocking + warnings).lower()
    verdict = str(evaluation.get("verdict") or "").lower()
    accepted = bool(evaluation.get("accepted"))
    can_execute = bool(evaluation.get("can_execute"))

    out_names = _tr_asset_labels_from_breakdown(evaluation, side="user", direction="outgoing", limit=3)
    in_names = _tr_asset_labels_from_breakdown(evaluation, side="user", direction="incoming", limit=3)
    blocked = _tr_clause_blocked_players(evaluation)
    players = list(dict.fromkeys(blocked + out_names + in_names))[:4]

    if accepted and can_execute:
        summary = "Deal can execute."
        if out_names and in_names:
            summary = f"{out_names[0]} for {in_names[0]} works."
        elif out_names:
            summary = f"Moving {out_names[0]} is approved."
        chips = players[:3] if players else ["FIT", "CAP OK"]
        return {"primary_code": "ACCEPTED", "summary": summary, "chips": chips, "players": players, "source": "backend"}

    chips: List[str] = []
    if blocked:
        chips.extend(blocked[:2])
    if "cap" in blob or "salary" in blob or verdict == "cap_illegal":
        if "CAP" not in chips:
            chips.append("CAP")
    if "clause" in blob or "nmc" in blob or "ntc" in blob or verdict == "ntc_nmc_conflict":
        if blocked:
            chips = blocked[:2] + [c for c in chips if c not in blocked]
        elif "CLAUSE" not in chips:
            chips.append("CLAUSE")
    if "pick" in blob and ("own" in blob or "registry" in blob):
        chips.append("PICK")
    if "roster" in blob or "slot" in blob or "maximum" in blob:
        chips.append("ROSTER")
    if "value" in blob or "overpay" in blob or verdict in ("trade_value_too_low", "rejected"):
        if "VALUE" not in chips:
            chips.append("VALUE")
    if not chips and players:
        chips = players[:3]
    if not chips and verdict == "asset_not_owned":
        chips.append("ASSET")
    if not chips:
        chips.append("VALUE" if not accepted else "CAP")

    summary = "Trade blocked."
    if blocked:
        summary = f"Clause blocks {blocked[0]}."
    elif "cap" in blob or verdict == "cap_illegal":
        summary = "Cap does not fit."
    elif "clause" in blob or "nmc" in blob or "ntc" in blob:
        summary = "Clause blocks player."
    elif "pick" in blob:
        summary = "Pick ownership issue."
    elif "roster" in blob:
        summary = "Roster limit failed."
    elif not accepted or verdict in ("rejected", "trade_value_too_low"):
        if out_names and not in_names:
            summary = f"Need better return for {out_names[0]}."
        elif in_names:
            summary = f"Package built around {in_names[0]} rejected."
        else:
            summary = "Need better return."
    elif not can_execute:
        summary = "Trade blocked."

    primary = chips[0] if chips else "BLOCKED"
    if players and primary in ("VALUE", "CLAUSE", "CAP") and players[0] not in chips:
        chips = [players[0]] + [c for c in chips if c != players[0]][:2]
    return {
        "primary_code": primary,
        "summary": summary,
        "chips": chips[:3],
        "players": players,
        "source": "backend",
    }


def _tr_team_wants_block(
    session: FranchiseSession,
    evaluation: Dict[str, Any],
    partner_team_id: str,
    *,
    user_team_id: str = "",
) -> Dict[str, Any]:
    immersion = dict(evaluation.get("immersion") or {})
    partner = session.team_by_id.get(str(partner_team_id)) if partner_team_id else None
    needs_impact = dict((evaluation.get("team_needs_impact") or {}).get(str(partner_team_id)) or {})
    chips: List[str] = []
    for src in (
        immersion.get("partner_needs"),
        needs_impact.get("priority_needs"),
        immersion.get("partner_values"),
    ):
        for item in safe_list_tr(src):
            chip = _tr_need_to_chip(str(item))
            if chip and chip not in chips:
                chips.append(chip)

    offered = _tr_asset_labels_from_breakdown(evaluation, side="user", direction="outgoing", limit=3)
    shopping = _tr_partner_shopping_targets(
        session,
        partner_team_id,
        user_team_id,
        needs_impact,
        limit=4,
    )
    players = _tr_dedupe_labels(shopping, 4)

    window = str(immersion.get("partner_window") or getattr(partner, "gm_window", "") or "").lower()
    summary = "Balancing roster needs."
    if window in ("rebuild", "rebuilding", "tanking", "seller"):
        summary = "Selling for futures."
    elif window in ("contender", "playoff", "cup threat"):
        summary = "Buying NHL help."
    if shopping:
        summary = f"Wants {shopping[0]} from your roster."
    elif offered:
        summary = f"Likes {offered[0]} in this deal."
    elif any(c == "DEFENSE" for c in chips):
        summary = "Needs defensive help."
    elif any(c == "GOALIE" for c in chips):
        summary = "Needs goaltending."
    elif any(c == "TOP 6" for c in chips):
        summary = "Needs scoring help."
    elif not players and not chips:
        summary = "No clear ask."
    return {
        "summary": summary,
        "chips": chips[:4],
        "players": players,
        "source": "backend",
    }


def safe_list_tr(val: Any) -> List[Any]:
    return list(val) if isinstance(val, list) else []


def _tr_untouchables_block(session: FranchiseSession, partner_team_id: str) -> Dict[str, Any]:
    team = session.team_by_id.get(str(partner_team_id or ""))
    if team is None:
        return {
            "summary": "No protected core found.",
            "players": [],
            "chips": [],
            "source": "backend",
        }
    try:
        from app.sim_engine.trades.trade_rules import _clause_summary  # noqa: WPS433
    except Exception:
        _clause_summary = lambda _p: {}  # type: ignore

    chips: List[str] = []
    ranked: List[Tuple[float, str]] = []
    for p in getattr(team, "roster", None) or []:
        if getattr(p, "retired", False):
            continue
        ovr = float(_fan_player_ovr99(p))
        ident = getattr(p, "identity", None)
        age = int(getattr(ident, "age", getattr(p, "age", 26)) or 26)
        clause = _clause_summary(p) if callable(_clause_summary) else {}
        name = _tr_player_display_name(p)
        protected = False
        if clause.get("nmc"):
            chips.append("NMC")
            protected = True
        if clause.get("ntc"):
            chips.append("NTC")
            protected = True
        if bool(getattr(p, "is_captain", False)) or str(getattr(p, "captaincy", "") or "").upper() in ("C", "CAPTAIN"):
            chips.append("CAPTAIN")
            protected = True
        if ovr >= 88:
            chips.append("FRANCHISE")
            protected = True
        elif age <= 24 and ovr >= 82:
            chips.append("YOUNG CORE")
            protected = True
        if protected:
            ranked.append((ovr, name))

    ranked.sort(key=lambda x: x[0], reverse=True)
    players = [n for _, n in ranked[:4]]
    chip_uniq = list(dict.fromkeys(chips))[:4]
    summary = "Core pieces protected." if players else "No protected core found."
    return {
        "summary": summary,
        "players": players,
        "chips": chip_uniq,
        "source": "backend",
    }


def _tr_gm_interest_block(evaluation: Dict[str, Any], partner_team_id: str) -> Dict[str, Any]:
    raw = float((evaluation.get("interest_level") or {}).get(str(partner_team_id)) or 0)
    score = int(_clamp_fan_score(raw * 100, 0, 100)) if raw else 0
    if score >= 66:
        label = "HIGH"
    elif score >= 38:
        label = "MED"
    elif score > 0:
        label = "LOW"
    else:
        label = "LOW"
        score = 24
    reasons: List[str] = []
    needs = dict((evaluation.get("team_needs_impact") or {}).get(str(partner_team_id)) or {})
    if needs.get("fills_need"):
        reasons.append("Need fit")
    strengthens = safe_list_tr(needs.get("strengthens"))
    if strengthens:
        reasons.append(str(strengthens[0])[:18])
    user_bd = dict((evaluation.get("asset_breakdown") or {}).get("user") or {})
    offered_players = _tr_player_labels_from_breakdown(evaluation, side="user", direction="outgoing", limit=1)
    if offered_players:
        reasons.insert(0, offered_players[0])
    partner_net = float((evaluation.get("asset_breakdown") or {}).get("partner", {}).get("net", 0) or 0)
    if partner_net >= -2:
        reasons.append("Fair value")
    cap_imp = dict((evaluation.get("cap_impact") or {}).get(str(partner_team_id)) or {})
    if cap_imp and cap_imp.get("after_usable") is not None and float(cap_imp.get("after_usable", 0)) >= 0:
        reasons.append("Cap works")
    for r in safe_list_tr(evaluation.get("rejection_reasons")):
        rs = str(r)
        if "fit" in rs.lower() and "Need fit" not in reasons:
            reasons.append("Need fit")
            break
    if not reasons:
        if label == "HIGH":
            reasons.append("Strong interest")
        elif label == "MED":
            reasons.append("Open to deal")
        else:
            reasons.append("Low appetite")
    return {"label": label, "score": score, "reasons": _tr_dedupe_labels(reasons, 2), "source": "backend"}


def _tr_cap_after_block(evaluation: Dict[str, Any], user_team_id: str) -> Dict[str, Any]:
    cap_row = dict((evaluation.get("cap_impact") or {}).get(str(user_team_id)) or {})
    after = cap_row.get("after_usable")
    before = cap_row.get("before_usable")
    delta = cap_row.get("delta")
    try:
        after_f = float(after) if after is not None else None
    except (TypeError, ValueError):
        after_f = None
    try:
        before_f = float(before) if before is not None else None
    except (TypeError, ValueError):
        before_f = None
    try:
        delta_f = float(delta) if delta is not None else None
    except (TypeError, ValueError):
        delta_f = None
    if delta_f is not None:
        sign = "+" if delta_f >= 0 else "-"
        label = f"{sign}${abs(delta_f):.2f}M"
        tone = "good" if delta_f >= 0 else "bad"
    else:
        label = "—"
        tone = "neutral"
    if after_f is not None and tone == "neutral":
        tone = "good" if after_f >= 0 else "bad"
    return {
        "before_space_m": round(before_f, 3) if before_f is not None else None,
        "projected_space_m": round(after_f, 3) if after_f is not None else None,
        "delta_m": round(delta_f, 3) if delta_f is not None else None,
        "label": label,
        "tone": tone,
    }


def _tr_block_detail(
    evaluation: Dict[str, Any],
    *,
    user_team_id: str = "",
    partner_team_id: str = "",
) -> Dict[str, Any]:
    """Human-facing primary block / reject detail for Trade Hub Propose footer."""
    why = _tr_why_block(
        evaluation,
        user_team_id=user_team_id,
        partner_team_id=partner_team_id,
    )
    reasons = [str(r) for r in (evaluation.get("rejection_reasons") or []) if r]
    warnings = [str(w) for w in (evaluation.get("warnings") or []) if w]
    blob = " ".join(reasons + warnings).lower()
    accepted = bool(evaluation.get("accepted"))
    can_execute = bool(evaluation.get("can_execute"))
    verdict = str(evaluation.get("verdict") or "").lower()
    message = reasons[0] if reasons else str(why.get("summary") or "")
    code = str(why.get("primary_code") or "BLOCKED")
    if not can_execute:
        if "cap" in blob or verdict == "cap_illegal" or code == "CAP":
            code = "CAP"
        elif "clause" in blob or "ntc" in blob or "nmc" in blob:
            code = "CLAUSE"
        elif "roster" in blob or "slot" in blob:
            code = "ROSTER"
        elif "pick" in blob:
            code = "PICK"
        else:
            code = code if code not in ("VALUE", "ACCEPTED") else "RULES"
    elif not accepted:
        if "value" in blob or verdict in ("trade_value_too_low", "rejected") or code == "VALUE":
            code = "VALUE"
        else:
            code = "REJECTED"

    unblock_hint = ""
    if code == "CAP":
        unblock_hint = "Retain salary or move salary the other way."
    elif code == "CLAUSE":
        unblock_hint = "Ask the player to waive their NTC/NMC."
    elif code == "VALUE":
        unblock_hint = "Add a pick or swap for a lesser asset."
    elif code == "ROSTER":
        unblock_hint = "Clear a roster slot first."
    elif code == "PICK":
        unblock_hint = "Fix pick ownership or choose another pick."

    partner_cap = _tr_cap_after_block(evaluation, partner_team_id) if partner_team_id else {}
    partner_after = partner_cap.get("projected_space_m")
    if code == "CAP" and partner_after is not None and float(partner_after) < 0:
        message = (
            f"{message} Partner would be ${abs(float(partner_after)):.2f}M under."
            if message
            else f"Partner would be ${abs(float(partner_after)):.2f}M under the cap."
        )
        if not unblock_hint:
            unblock_hint = "Retain salary or include salary going out."

    return {
        "code": code,
        "message": message or str(why.get("summary") or "Trade blocked."),
        "unblock_hint": unblock_hint,
        "partner_cap_after_m": partner_after,
        "partner_cap_delta_m": partner_cap.get("delta_m"),
        "chips": list(why.get("chips") or [])[:3],
        "summary": why.get("summary"),
    }


def _tr_fan_backlash_block(fan_reaction: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    fr = dict(fan_reaction or {})
    heat = int(fr.get("fan_heat") or max(0, 100 - int(fr.get("fan_reaction_score") or 50)))
    label = str(fr.get("fan_heat_label") or _fan_heat_label_from_heat(heat)).upper()
    reasons = [str(f) for f in safe_list_tr(fr.get("fan_factors"))[:3]]
    return {"label": label, "score": heat, "reasons": reasons, "source": "backend"}


def _tr_result_block(evaluation: Dict[str, Any]) -> Tuple[str, str]:
    accepted = bool(evaluation.get("accepted"))
    can_execute = bool(evaluation.get("can_execute"))
    verdict = str(evaluation.get("verdict") or "").lower()
    reasons_blob = " ".join(str(r) for r in (evaluation.get("rejection_reasons") or [])).lower()
    if accepted and can_execute:
        if verdict == "needs_adjustment":
            return "CLOSE", "close"
        return "ACCEPTED", "accepted"
    if not can_execute:
        return "BLOCKED", "blocked"
    user_bd = dict((evaluation.get("asset_breakdown") or {}).get("user") or {})
    try:
        user_net = float(user_bd.get("net") or 0)
    except (TypeError, ValueError):
        user_net = 0.0
    if (
        verdict == "trade_value_too_low"
        or "overpay" in reasons_blob
        or "value" in reasons_blob
        or user_net < -10
    ):
        return "OVERPAY", "overpay"
    return "REJECTED", "rejected"


def build_trade_review_payload(
    session: FranchiseSession,
    evaluation: Dict[str, Any],
    assets_by_team: Dict[str, Any],
    *,
    partner_team_id: Optional[str] = None,
    user_team_id: Optional[str] = None,
    fan_reaction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Backend trade review intelligence for Trade Hub review screen."""
    utid = str(user_team_id or getattr(session, "user_team_id", "") or "")
    partner = str(partner_team_id or "")
    if not partner:
        for tid in (assets_by_team or {}).keys():
            if str(tid) != utid:
                partner = str(tid)
                break
    user_bd = dict((evaluation.get("asset_breakdown") or {}).get("user") or {})
    user_out = float(user_bd.get("outgoing_total") or 0)
    user_in = float(user_bd.get("incoming_total") or 0)
    bal_label, bal_score, bal_summary = _tr_balance_label(user_out, user_in)
    result_label, result_tone = _tr_result_block(evaluation)
    if fan_reaction is None:
        try:
            fan_reaction = preview_trade_fan_reaction(
                session,
                dict(assets_by_team or {}),
                evaluation,
                partner_team_id=partner or None,
            )
        except Exception:
            fan_reaction = {}
    block_detail = _tr_block_detail(
        evaluation,
        user_team_id=utid,
        partner_team_id=partner,
    )
    return {
        "result_label": result_label,
        "result_tone": result_tone,
        "why": _tr_why_block(evaluation, user_team_id=utid, partner_team_id=partner),
        "block_detail": block_detail,
        "team_wants": _tr_team_wants_block(session, evaluation, partner, user_team_id=utid),
        "untouchables": _tr_untouchables_block(session, partner),
        "gm_interest": _tr_gm_interest_block(evaluation, partner),
        "trade_balance": {
            "label": bal_label,
            "score": bal_score,
            "summary": bal_summary,
            "source": "backend",
            "user_out": round(user_out, 1),
            "user_in": round(user_in, 1),
            "gap": round(abs(user_out - user_in), 1),
            "net": round(user_in - user_out, 1),
        },
        "fan_backlash": _tr_fan_backlash_block(fan_reaction),
        "cap_after": _tr_cap_after_block(evaluation, utid),
        "cap_after_by_team": {
            tid: _tr_cap_after_block(evaluation, tid)
            for tid in (utid, partner)
            if tid
        },
    }


def preview_trade_fan_reaction(
    session: FranchiseSession,
    assets_by_team: Dict[str, Any],
    evaluation_result: Optional[Dict[str, Any]] = None,
    *,
    partner_team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Non-persistent fan preview for Trade Hub evaluation."""
    utid = str(getattr(session, "user_team_id", "") or "")
    partner = partner_team_id
    if not partner:
        for tid in (assets_by_team or {}).keys():
            if str(tid) != utid:
                partner = str(tid)
                break
    result = _compute_trade_fan_reaction(
        session,
        utid,
        dict(assets_by_team or {}),
        evaluation_result,
        completed=False,
        partner_team_id=partner,
    )
    return result


_FAN_LEGACY_MAJOR_VERDICTS = frozenset({
    "Franchise-Altering Win",
    "Franchise-Altering Loss",
    "Pick Regret",
    "Prospect Jackpot",
    "Disaster Confirmed",
    "Fans Were Right",
    "Fans Overreacted",
    "Rebuild Masterstroke",
    "Rental Gamble Failed",
})


def _fan_ordinal(n: int) -> str:
    n = int(n)
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _fan_find_player_anywhere(session: FranchiseSession, pid: str) -> Tuple[Optional[Any], str]:
    target = str(pid or "")
    if not target:
        return None, ""
    for tm in session.team_by_id.values():
        tid = str(getattr(tm, "team_id", getattr(tm, "id", "")) or "")
        for roster_name in ("roster", "ahl_roster", "prospect_roster"):
            for p in getattr(tm, roster_name, None) or []:
                if str(getattr(p, "id", "") or "") == target:
                    return p, tid
    return None, ""


def _fan_player_stats_dict(session: FranchiseSession, pid: str) -> Dict[str, Any]:
    raw = dict(getattr(session, "player_season_stats", None) or {}).get(str(pid or "")) or {}
    pts = int(raw.get("points", raw.get("pts", 0)) or 0)
    return {
        "games": int(raw.get("games", raw.get("gp", 0)) or 0),
        "goals": int(raw.get("goals", raw.get("g", 0)) or 0),
        "assists": int(raw.get("assists", raw.get("a", 0)) or 0),
        "points": pts,
        "plus_minus": int(raw.get("plus_minus", raw.get("pm", 0)) or 0),
        "wins": int(raw.get("wins", raw.get("w", 0)) or 0),
        "save_pct": float(raw.get("save_pct", raw.get("sv_pct", 0)) or 0),
        "gaa": float(raw.get("gaa", 0) or 0),
    }


def _fan_player_role_label(player: Any) -> str:
    ovr = _fan_player_ovr99(player)
    pos = str(
        getattr(player, "position", None)
        or getattr(getattr(player, "identity", None), "position", None)
        or ""
    ).upper()
    ident = getattr(player, "identity", None)
    age = int(getattr(ident, "age", getattr(player, "age", 26)) or 26)
    pot_raw = float(getattr(player, "potential", 0) or getattr(ident, "potential", 0) or 0)
    pot = pot_raw * 99 if pot_raw <= 1.5 else pot_raw
    if pos == "G":
        if ovr >= 82:
            return "starter"
        if ovr >= 72:
            return "backup"
        return "prospect"
    if age <= 23 and ovr < 76 and pot >= 76:
        return "prospect"
    if pos in ("LW", "RW", "C", "F", "W"):
        if ovr >= 84:
            return "top_six"
        if ovr >= 76:
            return "middle_six"
        return "depth"
    if ovr >= 82:
        return "top_four"
    if ovr >= 74:
        return "third_pair"
    return "depth"


def _fan_attachment_label(score: float) -> str:
    s = float(score or 0)
    if s >= 85:
        return "beloved"
    if s >= 70:
        return "fan favorite"
    if s >= 50:
        return "popular"
    if s >= 30:
        return "neutral"
    return "unknown"


def _fan_build_team_context_snapshot(session: FranchiseSession, team_id: str) -> Dict[str, Any]:
    tid = str(team_id or "")
    team = session.team_by_id.get(tid)
    outlook: Dict[str, Any] = {}
    try:
        outlook = compute_team_playoff_outlook(session, team) if team is not None else {}
    except Exception:
        outlook = {}
    cal_rec = _team_cal_record_for_id(session, tid)
    gp = int(cal_rec.get("gp") or 0)
    pts = int(cal_rec.get("pts") or 0)
    gd = int(outlook.get("goal_differential") or 0)
    if gp > 0 and not gd:
        gd = int(outlook.get("goals_for", 0) or 0) - int(outlook.get("goals_against", 0) or 0)
    cap_space = 0.0
    try:
        if team is not None:
            cap_snap = _team_cap_snapshot(team, session.sim)
            cap_space = float(cap_snap.get("cap_space") or cap_snap.get("usable_cap_space") or 0)
    except Exception:
        cap_space = 0.0
    rank = None
    standings = getattr(session, "standings", None)
    if standings is not None and hasattr(standings, "rank_maps"):
        try:
            rank = (standings.rank_maps().get("league") or {}).get(tid)
        except Exception:
            rank = None
    pts_pct = float(outlook.get("points_pace") or outlook.get("points_pct") or 0)
    if pts_pct <= 0 and gp > 0:
        pts_pct = _safe_points_pct(cal_rec)
    return {
        "playoff_odds": int(outlook.get("playoff_odds", 50) or 50),
        "record": dict(cal_rec),
        "standings_rank": rank,
        "team_status": str(outlook.get("team_status") or outlook.get("outlook_label") or outlook.get("team_direction") or ""),
        "goal_differential": gd,
        "points_pct": pts_pct,
        "cap_space": cap_space,
        "season_day": int(getattr(session, "calendar_cursor", 0) or 0),
    }


def _fan_snapshot_player(
    session: FranchiseSession,
    player: Any,
    *,
    team_from: str,
    team_to: str,
) -> Dict[str, Any]:
    pid = str(getattr(player, "id", "") or "")
    ident = getattr(player, "identity", None)
    age = int(getattr(ident, "age", getattr(player, "age", 26)) or 26)
    ovr = int(round(_fan_player_ovr99(player)))
    pot_raw = float(getattr(player, "potential", 0) or getattr(ident, "potential", 0) or 0)
    potential = int(round(pot_raw * 99 if pot_raw <= 1.5 else pot_raw))
    team_from_s = str(team_from or "")
    from_team = session.team_by_id.get(team_from_s)
    attach = _compute_player_fan_attachment(player, from_team, session)
    contract = getattr(player, "contract", None)
    aav = float(getattr(contract, "cap_hit_m", 0) or getattr(contract, "aav_m", 0) or 0)
    years = int(getattr(contract, "years_remaining", 0) or 0)
    cap = str(getattr(player, "captaincy", "") or "")
    is_captain = bool(getattr(player, "is_captain", False)) or cap.upper() in ("C", "CAPTAIN")
    drafted_by = str(
        getattr(player, "draft_team_id", None)
        or getattr(player, "origin_team_id", None)
        or getattr(player, "drafted_by", None)
        or ""
    )
    homegrown = bool(drafted_by and drafted_by == team_from_s)
    pos = str(
        getattr(player, "position", None)
        or getattr(ident, "position", None)
        or getattr(player, "pos", None)
        or ""
    )
    role = _fan_player_role_label(player)
    return {
        "asset_type": "player",
        "player_id": pid,
        "name": str(getattr(player, "name", None) or getattr(ident, "name", None) or pid),
        "position": pos,
        "age_at_trade": age,
        "team_from": team_from_s,
        "team_to": str(team_to or ""),
        "ovr_at_trade": ovr,
        "potential_at_trade": potential,
        "role_at_trade": role,
        "line_role_at_trade": role,
        "contract_aav": aav,
        "contract_years_left": years,
        "is_captain": is_captain,
        "captaincy": cap,
        "homegrown": homegrown,
        "drafted_by_team": drafted_by,
        "years_with_team": int(getattr(player, "years_with_team", 0) or getattr(player, "tenure_years", 0) or 0),
        "stats_at_trade": _fan_player_stats_dict(session, pid),
        "attachment_score_at_trade": float(attach),
        "fan_label_at_trade": _fan_attachment_label(attach),
    }


def _fan_snapshot_pick(
    session: FranchiseSession,
    asset: Dict[str, Any],
    *,
    owner_before: str,
    owner_after: str,
) -> Dict[str, Any]:
    pick_id = str(asset.get("id") or asset.get("pick_id") or "")
    league = getattr(session.sim, "league", None)
    row: Dict[str, Any] = {}
    try:
        from app.sim_engine.trades.trade_pick_registry import get_pick_by_id  # noqa: WPS433

        row = dict(get_pick_by_id(league, pick_id) or {})
    except Exception:
        row = {}
    rnd = int(row.get("round") or asset.get("round") or asset.get("pick_round") or 0)
    year = int(row.get("year") or asset.get("year") or 0)
    orig_tid = str(row.get("original_team_id") or asset.get("original_team_id") or owner_before or "")
    orig_team = session.team_by_id.get(orig_tid)
    outlook: Dict[str, Any] = {}
    try:
        outlook = compute_team_playoff_outlook(session, orig_team) if orig_team is not None else {}
    except Exception:
        outlook = {}
    resolved = bool(row.get("resolved"))
    prospect_id = row.get("selected_prospect_id") if resolved else None
    overall = row.get("overall_pick") or row.get("pick_number")
    sel_name = None
    sel_ovr = None
    sel_pot = None
    if prospect_id:
        pl, _ = _fan_find_player_anywhere(session, str(prospect_id))
        if pl is not None:
            sel_name = str(getattr(pl, "name", "") or "")
            sel_ovr = int(round(_fan_player_ovr99(pl)))
            pr = float(getattr(pl, "potential", 0) or 0)
            sel_pot = int(round(pr * 99 if pr <= 1.5 else pr))
    return {
        "asset_type": "pick",
        "pick_id": pick_id,
        "round": rnd,
        "year": year,
        "original_team_id": orig_tid,
        "owner_before": str(owner_before or ""),
        "owner_after": str(owner_after or ""),
        "projected_value_at_trade": float(asset.get("value") or asset.get("projected_value") or 0),
        "projected_pick_range": str(asset.get("projected_range") or asset.get("projected_pick_range") or asset.get("projected_slot") or ""),
        "original_team_points_pct_at_trade": float(outlook.get("points_pace") or outlook.get("points_pct") or 0),
        "original_team_playoff_odds_at_trade": int(outlook.get("playoff_odds", 50) or 50),
        "protection": row.get("protection") or asset.get("protection"),
        "conditions": row.get("conditions") or asset.get("conditions"),
        "became_pick_number": int(overall) if overall is not None and resolved else None,
        "selected_player_id": str(prospect_id) if prospect_id else None,
        "selected_player_name": sel_name,
        "selected_player_ovr_at_draft": sel_ovr,
        "selected_player_potential_at_draft": sel_pot,
    }


def _fan_upgrade_legacy_entry(entry: Dict[str, Any], session: FranchiseSession, utid: str) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    init = int(entry.get("initial_fan_reaction") or entry.get("fan_reaction_score") or 50)
    heat = int(entry.get("initial_fan_heat") or entry.get("fan_heat") or max(0, 100 - init))
    entry.setdefault("trade_id", entry.get("trade_id") or "")
    entry.setdefault("date", entry.get("date") or "")
    entry.setdefault("user_team_id", utid)
    entry.setdefault("partner_team_id", entry.get("partner_team_id") or "")
    entry.setdefault("initial_fan_reaction", init)
    entry.setdefault("initial_fan_heat", heat)
    entry.setdefault("current_fan_reaction", int(entry.get("current_fan_reaction") or init))
    entry.setdefault("current_fan_heat", int(entry.get("current_fan_heat") or max(0, 100 - int(entry.get("current_fan_reaction") or init))))
    entry.setdefault("initial_verdict", str(entry.get("initial_verdict") or entry.get("verdict") or "Too Early"))
    entry.setdefault("current_verdict", str(entry.get("current_verdict") or entry.get("verdict") or "Too Early"))
    entry.setdefault("verdict", entry.get("current_verdict"))
    entry.setdefault("outgoing_assets_snapshot", list(entry.get("outgoing_assets_snapshot") or []))
    entry.setdefault("incoming_assets_snapshot", list(entry.get("incoming_assets_snapshot") or []))
    entry.setdefault("team_context_at_trade", dict(entry.get("team_context_at_trade") or {}))
    entry.setdefault("review_notes", list(entry.get("review_notes") or []))
    entry.setdefault("fan_factors", list(entry.get("fan_factors") or entry.get("factors") or []))
    entry.setdefault("legacy_score_delta", int(entry.get("legacy_score_delta") or 0))
    entry.setdefault("last_reviewed_date", entry.get("last_reviewed_date"))
    entry.setdefault("review_stage", str(entry.get("review_stage") or "immediate"))
    if entry.get("_trade_day") is None:
        day = entry.get("date")
        entry["_trade_day"] = int(day) if isinstance(day, int) else int(getattr(session, "calendar_cursor", 0) or 0)
    if "next_review_date" not in entry:
        entry["next_review_date"] = int(entry.get("_trade_day") or 0) + 30
    return entry


def _fan_build_legacy_record(
    session: FranchiseSession,
    fan_result: Dict[str, Any],
    trade_context: Dict[str, Any],
    utid: str,
) -> Dict[str, Any]:
    partner_id = str(trade_context.get("partner_team_id") or "")
    assets_by_team = dict(trade_context.get("assets_by_team") or {})
    outgoing, incoming = _fan_split_trade_assets(assets_by_team, utid, partner_id)
    out_snaps: List[Dict[str, Any]] = []
    in_snaps: List[Dict[str, Any]] = []

    for asset in outgoing:
        atype = str(asset.get("type") or "").lower()
        if atype == "pick":
            out_snaps.append(_fan_snapshot_pick(session, asset, owner_before=utid, owner_after=partner_id))
            continue
        pid = str(asset.get("id") or "")
        pl = _fan_resolve_player(session, utid, pid)
        if pl is None:
            continue
        out_snaps.append(_fan_snapshot_player(session, pl, team_from=utid, team_to=partner_id))

    for asset in incoming:
        atype = str(asset.get("type") or "").lower()
        if atype == "pick":
            in_snaps.append(_fan_snapshot_pick(session, asset, owner_before=partner_id, owner_after=utid))
            continue
        pid = str(asset.get("id") or "")
        src = str(asset.get("team") or partner_id or "")
        pl = _fan_resolve_player(session, src, pid)
        if pl is None:
            pl, _ = _fan_find_player_anywhere(session, pid)
        if pl is None:
            continue
        in_snaps.append(_fan_snapshot_player(session, pl, team_from=src, team_to=utid))

    trade_day = int(fan_result.get("calendar_day") or getattr(session, "calendar_cursor", 0) or 0)
    init_score = int(fan_result.get("fan_reaction_score") or 50)
    init_heat = int(fan_result.get("fan_heat") or max(0, 100 - init_score))
    return {
        "trade_id": fan_result.get("trade_id"),
        "date": fan_result.get("calendar_iso") or trade_day,
        "_trade_day": trade_day,
        "user_team_id": utid,
        "partner_team_id": partner_id,
        "initial_fan_reaction": init_score,
        "initial_fan_heat": init_heat,
        "current_fan_reaction": init_score,
        "current_fan_heat": init_heat,
        "initial_verdict": "Too Early",
        "current_verdict": "Too Early",
        "verdict": "Too Early",
        "outgoing_assets_snapshot": out_snaps,
        "incoming_assets_snapshot": in_snaps,
        "outgoing_assets_summary": str(trade_context.get("outgoing_summary") or ""),
        "incoming_assets_summary": str(trade_context.get("incoming_summary") or ""),
        "team_context_at_trade": _fan_build_team_context_snapshot(session, utid),
        "review_notes": [],
        "fan_factors": list(fan_result.get("fan_factors") or [])[:8],
        "legacy_score_delta": 0,
        "last_reviewed_date": None,
        "review_stage": "immediate",
        "next_review_date": trade_day + 30,
    }


def _fan_pick_registry_row(session: FranchiseSession, pick_id: str) -> Dict[str, Any]:
    league = getattr(session.sim, "league", None)
    if league is None or not pick_id:
        return {}
    try:
        from app.sim_engine.trades.trade_pick_registry import get_pick_by_id  # noqa: WPS433

        return dict(get_pick_by_id(league, pick_id) or {})
    except Exception:
        return {}


def _fan_refresh_pick_snapshot(session: FranchiseSession, snap: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(snap or {})
    row = _fan_pick_registry_row(session, str(updated.get("pick_id") or ""))
    if not row or not row.get("resolved"):
        return updated
    prospect_id = row.get("selected_prospect_id")
    overall = row.get("overall_pick") or row.get("pick_number")
    updated["became_pick_number"] = int(overall) if overall is not None else updated.get("became_pick_number")
    if prospect_id:
        updated["selected_player_id"] = str(prospect_id)
        pl, _ = _fan_find_player_anywhere(session, str(prospect_id))
        if pl is not None:
            updated["selected_player_name"] = str(getattr(pl, "name", "") or "")
            updated["selected_player_ovr_at_draft"] = int(round(_fan_player_ovr99(pl)))
            pr = float(getattr(pl, "potential", 0) or 0)
            updated["selected_player_potential_at_draft"] = int(round(pr * 99 if pr <= 1.5 else pr))
    return updated


def _fan_review_player_outcome(
    session: FranchiseSession,
    snap: Dict[str, Any],
    direction: str,
) -> Tuple[float, List[str], List[str]]:
    notes: List[str] = []
    labels: List[str] = []
    pid = str(snap.get("player_id") or "")
    player, cur_team = _fan_find_player_anywhere(session, pid)
    trade_ovr = float(snap.get("ovr_at_trade") or 0)
    role = str(snap.get("role_at_trade") or "")
    utid = str(snap.get("team_to") if direction == "incoming" else snap.get("team_from") or "")

    if player is None:
        if direction == "incoming":
            notes.append("Incoming player no longer on roster.")
            return -6.0, notes, ["Incoming Player Declined"]
        notes.append("Asset outcome unavailable.")
        return 0.0, notes, []

    cur_ovr = _fan_player_ovr99(player)
    ovr_delta = cur_ovr - trade_ovr
    stats = _fan_player_stats_dict(session, pid)
    trade_stats = dict(snap.get("stats_at_trade") or {})
    pts_delta = int(stats.get("points") or 0) - int(trade_stats.get("points") or 0)
    cur_role = _fan_player_role_label(player)
    delta = 0.0

    if direction == "incoming":
        if ovr_delta >= 6:
            delta += 14.0 if role == "prospect" else 10.0
            notes.append(f"Acquired {'prospect' if role == 'prospect' else 'player'} gained +{int(ovr_delta)} OVR.")
            labels.append("Incoming Prospect Breakout" if role == "prospect" else "Incoming Star Hit")
        elif ovr_delta >= 3:
            delta += 6.0
            notes.append(f"Acquired player gained +{int(ovr_delta)} OVR.")
        elif ovr_delta <= -5:
            delta -= 10.0
            notes.append(f"Incoming player lost {int(-ovr_delta)} OVR.")
            labels.append("Incoming Player Declined")
        elif ovr_delta <= -2:
            delta -= 5.0
        if cur_role in ("top_six", "top_four", "starter") and role in ("prospect", "depth", "middle_six", "third_pair", "backup"):
            delta += 6.0
            notes.append(f"Acquired player became a {cur_role.replace('_', '-')} regular.")
        if pts_delta >= 25:
            delta += 4.0
            notes.append("Incoming production surged after the trade.")
        if str(cur_team) != utid and utid:
            delta -= 5.0
            notes.append("Incoming player left quickly.")
        aav = float(snap.get("contract_aav") or 0)
        if aav >= 7 and ovr_delta >= 2:
            labels.append("Contract Aged Well")
            delta += 3.0
        elif aav >= 7 and ovr_delta <= -3:
            labels.append("Contract Aged Poorly")
            delta -= 4.0
        if role == "prospect" and cur_ovr >= 72:
            labels.append("Prospect Became NHLer")
            delta += 6.0
        elif role == "prospect" and ovr_delta <= -2:
            labels.append("Prospect Stalled")
            delta -= 7.0
    else:
        if snap.get("is_captain") and ovr_delta <= -3:
            delta += 10.0
            notes.append("Captain's decline softens backlash.")
            labels.append("Outgoing Captain Declined")
        elif ovr_delta >= 6:
            delta -= 15.0 if trade_ovr >= 84 else -12.0
            notes.append(f"Outgoing star gained +{int(ovr_delta)} OVR elsewhere.")
            labels.append("Outgoing Star Dominated")
        elif ovr_delta >= 3:
            delta -= 9.0
            notes.append(f"Traded player thriving (+{int(ovr_delta)} OVR).")
            labels.append("Outgoing Star Dominated")
        elif ovr_delta <= -5:
            delta += 10.0
            notes.append(f"Traded player declined ({int(ovr_delta)} OVR).")
        elif ovr_delta <= -2:
            delta += 6.0
        if pts_delta >= 30 and trade_ovr >= 82:
            delta -= 7.0
            notes.append(f"Outgoing star scored {int(stats.get('points') or 0)} points.")
        if ovr_delta >= 4 and direction == "outgoing":
            labels.append("Weak Return Confirmed")

    return delta, notes, labels


def _fan_review_pick_outcome(
    session: FranchiseSession,
    snap: Dict[str, Any],
    direction: str,
) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
    updated = _fan_refresh_pick_snapshot(session, snap)
    overall = updated.get("became_pick_number")
    notes: List[str] = []
    labels: List[str] = []
    if overall is None:
        return 0.0, notes, labels, updated
    overall_i = int(overall)
    rnd = int(updated.get("round") or 1)
    delta = 0.0
    if direction == "incoming":
        if overall_i <= 5:
            delta += 16.0
            notes.append(f"Acquired pick became {_fan_ordinal(overall_i)} overall.")
            labels.append("Prospect Jackpot")
        elif overall_i <= 12:
            delta += 10.0
            notes.append(f"Acquired 1st landed at #{overall_i}.")
        elif overall_i >= 28 and rnd == 1:
            delta += 6.0
            notes.append("Acquired pick became late 1st-round value.")
        pot = int(updated.get("selected_player_potential_at_draft") or 0)
        if pot >= 82:
            delta += 4.0
            notes.append("Acquired pick selected a high-potential prospect.")
    else:
        if overall_i <= 5:
            delta -= 20.0
            notes.append(f"Traded pick became {_fan_ordinal(overall_i)} overall.")
            labels.append("Pick Regret")
        elif overall_i <= 10:
            delta -= 14.0
            notes.append(f"Moved pick became #{overall_i} overall.")
            labels.append("Pick Regret")
        elif overall_i >= 26:
            delta += 8.0
            notes.append(f"Traded-away pick landed #{overall_i} — softer than feared.")
        elif overall_i >= 20:
            delta += 4.0
        pot = int(updated.get("selected_player_potential_at_draft") or 0)
        if pot >= 84 and overall_i <= 8:
            delta -= 6.0
            notes.append("Traded-away pick became a star prospect.")
    return delta, notes, labels, updated


def _fan_review_team_outcome(
    session: FranchiseSession,
    entry: Dict[str, Any],
    utid: str,
) -> Tuple[float, List[str], List[str]]:
    ctx = dict(entry.get("team_context_at_trade") or {})
    cur = _fan_build_team_context_snapshot(session, utid)
    notes: List[str] = []
    labels: List[str] = []
    delta = 0.0
    init_pts = float(ctx.get("points_pct") or 0)
    cur_pts = float(cur.get("points_pct") or 0)
    init_odds = int(ctx.get("playoff_odds") or 50)
    cur_odds = int(cur.get("playoff_odds") or 50)
    pts_delta = cur_pts - init_pts
    odds_delta = cur_odds - init_odds
    init_gd = int(ctx.get("goal_differential") or 0)
    cur_gd = int(cur.get("goal_differential") or 0)

    if odds_delta >= 12 or pts_delta >= 0.05:
        delta += 10.0
        notes.append("Team points percentage rose after the trade.")
        labels.append("Team Improved")
    elif odds_delta <= -12 or pts_delta <= -0.05:
        delta -= 12.0
        notes.append("Team collapsed after controversial trade.")
        labels.append("Team Collapsed")

    if cur_gd - init_gd >= 15:
        delta += 4.0
    elif cur_gd - init_gd <= -15:
        delta -= 5.0

    direction = str(ctx.get("team_status") or "").lower()
    if direction in ("rebuild", "rebuilding", "tanking", "seller") and odds_delta <= -5:
        labels.append("Rebuild Stalled")
        delta -= 4.0
    elif direction in ("rebuild", "rebuilding", "tanking", "seller") and odds_delta >= 5:
        labels.append("Rebuild Accelerated")
        delta += 5.0

    if direction in ("contender", "playoff", "cup threat"):
        if odds_delta >= 8:
            labels.append("Playoff Push Worked")
            delta += 6.0
        elif odds_delta <= -10:
            labels.append("Playoff Push Failed")
            delta -= 8.0

    champ = str(getattr(session, "stanley_cup_winner", "") or getattr(session, "champion_id", "") or "")
    if champ and champ == utid and bool(getattr(session, "playoffs_simulated", False)):
        delta += 20.0
        notes.append("Stanley Cup win followed the trade.")
        labels.append("Franchise-Altering Win")
        labels.append("Cup Window Helped")
    elif direction in ("contender", "playoff", "cup threat") and odds_delta <= -15:
        labels.append("Cup Window Hurt")

    return delta, notes, labels


def _fan_legacy_unresolved_assets(entry: Dict[str, Any]) -> bool:
    for snap in list(entry.get("incoming_assets_snapshot") or []) + list(entry.get("outgoing_assets_snapshot") or []):
        if str(snap.get("asset_type")) == "pick" and snap.get("became_pick_number") is None:
            return True
        if str(snap.get("asset_type")) == "player" and str(snap.get("role_at_trade") or "") == "prospect":
            return True
    return False


def _fan_compute_legacy_review(
    session: FranchiseSession,
    entry: Dict[str, Any],
    utid: str,
) -> Tuple[int, List[str], List[str]]:
    init = int(entry.get("initial_fan_reaction") or 50)
    score = float(init)
    notes: List[str] = []
    labels: List[str] = []
    factors = list(entry.get("fan_factors") or [])

    incoming_snaps = list(entry.get("incoming_assets_snapshot") or [])
    outgoing_snaps = list(entry.get("outgoing_assets_snapshot") or [])
    for idx, snap in enumerate(incoming_snaps):
        if str(snap.get("asset_type")) == "pick":
            d, n, l, upd = _fan_review_pick_outcome(session, snap, "incoming")
            incoming_snaps[idx] = upd
        else:
            d, n, l = _fan_review_player_outcome(session, snap, "incoming")
        score += d
        notes.extend(n)
        labels.extend(l)
    for idx, snap in enumerate(outgoing_snaps):
        if str(snap.get("asset_type")) == "pick":
            d, n, l, upd = _fan_review_pick_outcome(session, snap, "outgoing")
            outgoing_snaps[idx] = upd
        else:
            d, n, l = _fan_review_player_outcome(session, snap, "outgoing")
        score += d
        notes.extend(n)
        labels.extend(l)
    entry["incoming_assets_snapshot"] = incoming_snaps
    entry["outgoing_assets_snapshot"] = outgoing_snaps

    td, tn, tl = _fan_review_team_outcome(session, entry, utid)
    score += td
    notes.extend(tn)
    labels.extend(tl)

    if "Playoff rental" in factors:
        rental_gone = False
        for snap in incoming_snaps:
            if str(snap.get("asset_type")) != "player":
                continue
            pl, cur_team = _fan_find_player_anywhere(session, str(snap.get("player_id") or ""))
            if pl is None or str(cur_team) != utid:
                rental_gone = True
                break
        if rental_gone and int(_fan_build_team_context_snapshot(session, utid).get("playoff_odds") or 50) < 45:
            score -= 12.0
            notes.append("Rental left in free agency.")
            labels.append("Rental Gamble Failed")

    if "Rebuild Accelerated" in labels and ("Incoming Prospect Breakout" in labels or "Prospect Jackpot" in labels):
        labels.append("Rebuild Masterstroke")

    init_heat = int(entry.get("initial_fan_heat") or max(0, 100 - init))
    if init_heat >= 55 and score > init + 8:
        notes.append("Fans now view the move more favorably.")
    elif init_heat >= 55 and score < init - 8:
        notes.append("Fans were right to hate this trade.")
    elif score >= init + 6:
        notes.append("The trade aged better than expected.")

    labels = list(dict.fromkeys(labels))
    entry["_legacy_labels"] = labels
    return _clamp_fan_score(score, 5, 98), notes[:8], labels


def _fan_resolve_legacy_verdict(
    entry: Dict[str, Any],
    init: int,
    cur: int,
    init_heat: int,
    labels: List[str],
) -> str:
    stage = str(entry.get("review_stage") or "immediate")
    unresolved = _fan_legacy_unresolved_assets(entry)
    delta = cur - init
    label_set = set(labels or [])

    if unresolved and stage in ("immediate", "30_day"):
        return "Too Early"
    if "Franchise-Altering Win" in label_set or (delta >= 18 and "Team Improved" in label_set):
        return "Franchise-Altering Win"
    if "Franchise-Altering Loss" in label_set or ("Team Collapsed" in label_set and delta <= -15):
        return "Franchise-Altering Loss"
    if "Pick Regret" in label_set:
        return "Pick Regret"
    if "Prospect Jackpot" in label_set:
        return "Prospect Jackpot"
    if "Rebuild Masterstroke" in label_set or ("Rebuild Accelerated" in label_set and delta >= 8):
        return "Rebuild Masterstroke"
    if "Rental Gamble Failed" in label_set:
        return "Rental Gamble Failed"
    if init_heat >= 55 and delta >= 10:
        return "Fans Overreacted"
    if init_heat >= 55 and delta <= -10:
        return "Fans Were Right"
    if init_heat >= 55 and delta <= -6 and ("Outgoing Star Dominated" in label_set or "Weak Return Confirmed" in label_set):
        return "Disaster Confirmed"
    if delta >= 8 and init_heat >= 45:
        return "Aging Better"
    if delta >= 6:
        return "Smart Move"
    if delta <= -12:
        return "Disaster Confirmed"
    if "Outgoing Captain Declined" in label_set and delta >= 0:
        return "Painful But Necessary"
    if "Team Improved" in label_set and "Outgoing Star Dominated" in label_set:
        return "Win-Win Trade"
    if delta <= -8 and init_heat >= 50:
        return "Still Hurts"
    return str(entry.get("current_verdict") or "Too Early")


def _fan_advance_legacy_stage(session: FranchiseSession, entry: Dict[str, Any], calendar_idx: int) -> Tuple[str, int]:
    stage = str(entry.get("review_stage") or "immediate")
    trade_day = int(entry.get("_trade_day") or 0)
    last_reg = int(getattr(session, "nhl_regular_season_last_index", 192) or 192)
    playoffs_done = bool(getattr(session, "playoffs_simulated", False))
    if stage == "immediate":
        return "30_day", trade_day + 30
    if stage == "30_day":
        return "trade_deadline_or_regular_season_end", max(last_reg, calendar_idx + 14)
    if stage == "trade_deadline_or_regular_season_end":
        if playoffs_done:
            return "post_playoffs", calendar_idx + 14
        return "post_playoffs", last_reg + 21
    if stage == "post_playoffs":
        return "draft_resolution", calendar_idx + 45
    if stage == "draft_resolution":
        return "one_year_later", trade_day + 250
    if stage == "one_year_later":
        return "future_season", calendar_idx + 120
    return "future_season", calendar_idx + 180


def _fan_legacy_due_for_review(session: FranchiseSession, entry: Dict[str, Any], calendar_idx: int) -> bool:
    if calendar_idx >= int(entry.get("next_review_date") or 0):
        return True
    for snap in list(entry.get("incoming_assets_snapshot") or []) + list(entry.get("outgoing_assets_snapshot") or []):
        if str(snap.get("asset_type")) != "pick" or snap.get("became_pick_number") is not None:
            continue
        refreshed = _fan_refresh_pick_snapshot(session, snap)
        if refreshed.get("became_pick_number") is not None:
            return True
    return False


def _fan_legacy_change_is_notifiable(old_verdict: str, new_verdict: str, old_reaction: int, new_reaction: int) -> bool:
    if old_verdict == new_verdict and abs(new_reaction - old_reaction) < 8:
        return False
    if new_verdict in _FAN_LEGACY_MAJOR_VERDICTS:
        return abs(new_reaction - old_reaction) >= 5 or old_verdict != new_verdict
    if old_verdict != new_verdict:
        return abs(new_reaction - old_reaction) >= 8
    return abs(new_reaction - old_reaction) >= 12


def _fan_emit_legacy_notification(
    session: FranchiseSession,
    entry: Dict[str, Any],
    old_verdict: str,
    notes: List[str],
) -> None:
    verdict = str(entry.get("current_verdict") or "Too Early")
    delta = int(entry.get("legacy_score_delta") or 0)
    headlines = {
        "Fans Overreacted": "Fans warming up after prospect breakout.",
        "Pick Regret": "Moving that first-round pick now looks costly.",
        "Outgoing Captain Declined": "Captain's decline softens backlash.",
        "Rental Gamble Failed": "Deadline gamble failed after early playoff exit.",
        "Incoming Star Hit": "Acquired star helped fuel playoff run.",
        "Aging Better": "Trade aging better than expected.",
        "Fans Were Right": "Early backlash looks justified in hindsight.",
        "Prospect Jackpot": "Acquired prospect changed the narrative.",
        "Disaster Confirmed": "Trade looks worse with every passing month.",
        "Franchise-Altering Win": "Stanley Cup run rewrote this trade's legacy.",
        "Smart Move": "Fans now view the move as smart.",
    }
    headline = headlines.get(verdict) or (notes[0] if notes else f"Fan reaction shifted ({delta:+d}).")
    text = f"Trade Revisited: {headline}"
    if notes:
        text = f"{text} {' · '.join(notes[:3])}"
    notif = _normalize_notification_payload(
        {
            "type": "trade_legacy",
            "priority": "MEDIUM" if abs(delta) < 12 else "HIGH",
            "title": "Trade Revisited",
            "headline": text[:120],
            "text": text[:220],
            "source": "trade_hub_legacy",
            "trade_id": entry.get("trade_id"),
            "old_verdict": old_verdict,
            "new_verdict": verdict,
            "fan_reaction_delta": delta,
            "review_notes": list(notes[:3]),
        },
        index=len(getattr(session, "notifications", None) or []),
    )
    session.notifications = getattr(session, "notifications", None) or []
    session.notifications.append(notif)
    trade_day = int(entry.get("_trade_day") or getattr(session, "calendar_cursor", 0) or 0)
    session.timeline.append(
        _normalized_timeline_event(
            event_id=f"timeline:trade_legacy:{entry.get('trade_id') or trade_day}:{uuid.uuid4().hex[:8]}",
            event_type="trade_legacy",
            text=text[:160],
            calendar_day=trade_day,
            calendar_iso=_calendar_iso_for_day(session, trade_day),
            team_id=str(entry.get("user_team_id") or session.user_team_id or ""),
            priority="MEDIUM",
            extra={
                "trade_id": entry.get("trade_id"),
                "old_verdict": old_verdict,
                "new_verdict": verdict,
                "fan_reaction_delta": delta,
                "review_notes": list(notes[:3]),
            },
        )
    )


def _apply_completed_trade_fan_effects(
    session: FranchiseSession,
    user_team_id: str,
    fan_result: Dict[str, Any],
    trade_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist fan consequences after a completed user trade only."""
    utid = str(user_team_id or "")
    if utid != str(getattr(session, "user_team_id", "") or ""):
        return fan_result
    if not fan_result.get("should_persist"):
        return fan_result
    profile = _ensure_team_fan_profile(session, utid)
    eff = dict(fan_result.get("fan_effects") or {})

    profile["fan_confidence"] = float(
        _clamp_fan_score(float(profile.get("fan_confidence", 55)) + float(eff.get("fan_confidence_delta", 0)))
    )
    profile["fan_patience"] = float(
        _clamp_fan_score(float(profile.get("fan_patience", 55)) + float(eff.get("owner_patience_delta", 0)))
    )
    profile["fan_trust_in_gm"] = float(
        _clamp_fan_score(float(profile.get("fan_trust_in_gm", 58)) + float(eff.get("gm_trust_delta", 0)))
    )
    profile["recent_trade_heat"] = float(
        _clamp_fan_score(max(float(profile.get("recent_trade_heat", 0)), float(fan_result.get("fan_heat", 0)) * 0.88))
    )

    if int(fan_result.get("fan_heat", 0)) >= 45:
        profile["season_backlash_events"] = list(profile.get("season_backlash_events") or []) + [
            {
                "trade_id": fan_result.get("trade_id"),
                "fan_heat": fan_result.get("fan_heat"),
                "day": fan_result.get("calendar_day"),
            }
        ]
        profile["season_backlash_events"] = profile["season_backlash_events"][-12:]

    history_entry = _fan_build_legacy_record(session, fan_result, trade_context, utid)
    profile["trade_reaction_history"] = list(profile.get("trade_reaction_history") or []) + [history_entry]
    profile["trade_reaction_history"] = profile["trade_reaction_history"][-24:]
    if int(fan_result.get("fan_heat", 0)) >= 55 or int(fan_result.get("fan_reaction_score", 50)) >= 80:
        profile["last_major_trade_reaction"] = dict(fan_result)

    team = session.team_by_id.get(utid)
    st = getattr(team, "state", None) if team is not None else None
    if st is not None:
        try:
            if hasattr(st, "team_morale"):
                st.team_morale = max(0.0, min(1.0, float(getattr(st, "team_morale", 0.5)) + float(eff.get("team_morale_delta", 0)) * 0.01))
            if hasattr(st, "organizational_pressure"):
                st.organizational_pressure = max(
                    0.0,
                    min(1.0, float(getattr(st, "organizational_pressure", 0.5)) + float(eff.get("pressure_delta", 0)) * 0.01),
                )
        except (TypeError, ValueError):
            pass
        if hasattr(st, "clamp"):
            try:
                st.clamp()
            except Exception:
                pass

    storyline_type = fan_result.get("fan_storyline_type")
    if storyline_type:
        sl = {
            "type": str(storyline_type).lower(),
            "category": "trade",
            "cause_type": str(storyline_type),
            "priority": "HIGH" if int(fan_result.get("fan_heat", 0)) >= 75 else "MEDIUM",
            "tone": "negative" if int(fan_result.get("fan_heat", 0)) >= 55 else "positive",
            "headline": str(fan_result.get("fan_headline") or ""),
            "description": str(fan_result.get("fan_summary") or ""),
            "summary": str(fan_result.get("fan_summary") or ""),
            "team_id": utid,
            "effects": {
                "fan_confidence": float(eff.get("fan_confidence_delta", 0)),
                "owner_patience": float(eff.get("owner_patience_delta", 0)),
                "team_morale": float(eff.get("team_morale_delta", 0)),
                "media_pressure": float(max(0.0, eff.get("pressure_delta", 0))),
            },
            "fan_reaction_score": fan_result.get("fan_reaction_score"),
            "fan_heat": fan_result.get("fan_heat"),
            "fan_factors": list(fan_result.get("fan_factors") or [])[:4],
        }
        _record_storyline(session, sl)

    return fan_result


def apply_completed_trade_fan_reaction(
    session: FranchiseSession,
    user_team_id: str,
    assets_by_team: Dict[str, Any],
    evaluation_result: Optional[Dict[str, Any]],
    trade_context: Dict[str, Any],
) -> Dict[str, Any]:
    partner = trade_context.get("partner_team_id")
    fan_result = _compute_trade_fan_reaction(
        session,
        user_team_id,
        dict(assets_by_team or {}),
        evaluation_result,
        completed=True,
        partner_team_id=str(partner or ""),
        trade_id=str(trade_context.get("trade_id") or ""),
    )
    enriched_ctx = dict(trade_context or {})
    enriched_ctx.setdefault("assets_by_team", dict(assets_by_team or {}))
    enriched_ctx["evaluation"] = evaluation_result
    return _apply_completed_trade_fan_effects(session, user_team_id, fan_result, enriched_ctx)


def _decay_fan_heat_for_team(session: FranchiseSession, team_id: str, calendar_idx: int) -> None:
    profile = _ensure_team_fan_profile(session, team_id)
    week = int(calendar_idx // 7)
    if int(profile.get("last_decay_week", -1)) == week:
        return
    profile["last_decay_week"] = week
    heat = float(profile.get("recent_trade_heat", 0) or 0)
    profile["recent_trade_heat"] = float(_clamp_fan_score(max(0.0, heat - 4.0)))

    team = session.team_by_id.get(str(team_id))
    baseline = 55.0
    try:
        outlook = compute_team_playoff_outlook(session, team) if team is not None else {}
        baseline = float(_clamp_fan_score(40 + int(outlook.get("playoff_odds", 50) or 50) * 0.35))
        form = float(outlook.get("recent_form_pct") or 0.5)
        if form >= 0.58:
            profile["recent_trade_heat"] = float(_clamp_fan_score(float(profile["recent_trade_heat"]) - 2.0))
        elif form < 0.38:
            profile["recent_trade_heat"] = float(_clamp_fan_score(float(profile["recent_trade_heat"]) + 1.0))
    except Exception:
        pass

    conf = float(profile.get("fan_confidence", 55))
    trust = float(profile.get("fan_trust_in_gm", 58))
    if conf < baseline:
        profile["fan_confidence"] = float(min(baseline, conf + 1.5))
    elif conf > baseline + 8:
        profile["fan_confidence"] = float(max(baseline + 4, conf - 0.8))
    if float(profile.get("recent_trade_heat", 0)) <= 8:
        if trust < 58:
            profile["fan_trust_in_gm"] = float(min(58.0, trust + 1.0))


def _process_trade_fan_legacy_reviews(session: FranchiseSession, calendar_idx: int) -> None:
    utid = str(getattr(session, "user_team_id", "") or "")
    if not utid:
        return
    profile = _ensure_team_fan_profile(session, utid)
    history = list(profile.get("trade_reaction_history") or [])
    if not history:
        return
    changed = False
    for entry in history:
        if not isinstance(entry, dict):
            continue
        try:
            _fan_upgrade_legacy_entry(entry, session, utid)
            if not _fan_legacy_due_for_review(session, entry, calendar_idx):
                continue
            init = int(entry.get("initial_fan_reaction") or 50)
            init_heat = int(entry.get("initial_fan_heat") or max(0, 100 - init))
            old_verdict = str(entry.get("current_verdict") or entry.get("verdict") or "Too Early")
            old_reaction = int(entry.get("current_fan_reaction") or init)
            new_reaction, notes, labels = _fan_compute_legacy_review(session, entry, utid)
            new_verdict = _fan_resolve_legacy_verdict(entry, init, new_reaction, init_heat, labels)
            entry["current_fan_reaction"] = new_reaction
            entry["current_fan_heat"] = max(0, 100 - new_reaction)
            entry["legacy_score_delta"] = new_reaction - init
            entry["current_verdict"] = new_verdict
            entry["verdict"] = new_verdict
            merged_notes = list(entry.get("review_notes") or []) + list(notes or [])
            entry["review_notes"] = merged_notes[-12:]
            entry["last_reviewed_date"] = calendar_idx
            new_stage, new_nxt = _fan_advance_legacy_stage(session, entry, calendar_idx)
            entry["review_stage"] = new_stage
            entry["next_review_date"] = new_nxt
            if _fan_legacy_change_is_notifiable(old_verdict, new_verdict, old_reaction, new_reaction):
                _fan_emit_legacy_notification(session, entry, old_verdict, notes[:3])
            changed = True
        except Exception:
            continue
    if changed:
        profile["trade_reaction_history"] = history


def _decay_all_fan_heat(session: FranchiseSession, calendar_idx: int) -> None:
    """Weekly fan heat decay + legacy trade reviews."""
    utid = str(getattr(session, "user_team_id", "") or "")
    if utid:
        _decay_fan_heat_for_team(session, utid, calendar_idx)
        _process_trade_fan_legacy_reviews(session, calendar_idx)
    season = int(getattr(session, "season_calendar_year", 2025) or 2025)
    if int(calendar_idx) == int(getattr(session, "nhl_regular_season_last_index", 192) or 192):
        for tid, prof in dict(getattr(session, "fan_profiles", None) or {}).items():
            if isinstance(prof, dict):
                prof["season_backlash_events"] = []
                prof["recent_trade_heat"] = float(_clamp_fan_score(float(prof.get("recent_trade_heat", 0)) * 0.5))


def _record_legacy_trade_hub_notifications(
    session: FranchiseSession,
    exec_result: Dict[str, Any],
    ctx: Dict[str, Any],
) -> None:
    """Storyline + notification hooks after a validated trade execution."""
    utid = str(getattr(session, "user_team_id", "") or "")
    moved_assets = [
        m for m in (exec_result.get("moved_assets") or [])
        if m.get("applied", True) is not False
    ]
    moved_players = [
        m for m in moved_assets
        if str(m.get("asset_type") or m.get("type") or "player").lower() == "player"
    ]
    if not moved_assets:
        return
    user_involved = any(
        utid and utid in (str(m.get("source_team_id") or ""), str(m.get("acquiring_team_id") or ""))
        for m in moved_assets
    )
    if not user_involved:
        return

    trade_day = int(getattr(session, "calendar_cursor", 0) or 0)
    try:
        from app.sim_engine.franchise.storyline_engine import (  # noqa: WPS433
            migrate_session_storyline_state,
            record_decision_event,
            resolve_culprit_traded_storylines,
        )

        migrate_session_storyline_state(session)
        record_decision_event(
            session,
            {
                "event_type": "PLAYER_TRADED",
                "team_id": utid,
                "player_ids": [str(m.get("asset_id") or m.get("id") or "") for m in moved_players],
                "trade_id": str(exec_result.get("trade_id") or f"trade_exec_{trade_day}"),
                "severity": "medium",
            },
        )
        resolve_culprit_traded_storylines(session, moved_players)
    except Exception:
        pass

    history = dict(exec_result.get("history_record") or {})
    assets_by_team = dict(history.get("assets_by_team") or {})
    evaluation = dict(exec_result.get("evaluation") or {})
    partner_id = ""
    for tid in assets_by_team.keys():
        if str(tid) != utid:
            partner_id = str(tid)
            break
    outgoing_names: List[str] = []
    incoming_names: List[str] = []
    for ac in assets_by_team.get(utid, []) or []:
        if str(ac.get("type") or "").lower() == "player":
            pl = _fan_resolve_player(session, utid, str(ac.get("id") or ""))
            outgoing_names.append(str(getattr(pl, "name", None) or ac.get("id") or "Player"))
        elif str(ac.get("type") or "").lower() == "pick":
            outgoing_names.append(f"{ac.get('year', '?')} R{ac.get('round', '?')}")
    for tid, acs in assets_by_team.items():
        if str(tid) == utid:
            continue
        for ac in acs or []:
            if str(ac.get("type") or "").lower() == "player":
                pl = _fan_resolve_player(session, str(tid), str(ac.get("id") or ""))
                incoming_names.append(str(getattr(pl, "name", None) or ac.get("id") or "Player"))
            elif str(ac.get("type") or "").lower() == "pick":
                incoming_names.append(f"{ac.get('year', '?')} R{ac.get('round', '?')}")

    fan_result: Dict[str, Any] = {}
    try:
        fan_result = apply_completed_trade_fan_reaction(
            session,
            utid,
            assets_by_team,
            evaluation,
            {
                "trade_id": exec_result.get("trade_id"),
                "partner_team_id": partner_id,
                "outgoing_summary": ", ".join(outgoing_names[:4]),
                "incoming_summary": ", ".join(incoming_names[:4]),
                "assets_by_team": assets_by_team,
            },
        )
        exec_result["fan_reaction"] = fan_result
    except Exception:
        fan_result = {}

    headline = str(exec_result.get("headline") or "")
    if not headline:
        headline_bits = []
        for m in moved_players[:4]:
            src = (
                _display_team(session.team_by_id.get(m["source_team_id"]))
                if session.team_by_id.get(m.get("source_team_id"))
                else m.get("source_team_id")
            )
            dst = (
                _display_team(session.team_by_id.get(m["acquiring_team_id"]))
                if session.team_by_id.get(m.get("acquiring_team_id"))
                else m.get("acquiring_team_id")
            )
            headline_bits.append(f"{m.get('player_name')}: {src} -> {dst}")
        headline = "TRADE EXECUTED: " + ("; ".join(headline_bits) if headline_bits else "Assets moved")

    fan_headline = str(fan_result.get("fan_headline") or "").strip()
    fan_summary = str(fan_result.get("fan_summary") or "").strip()
    notif_text = fan_headline or headline
    if fan_summary and fan_summary not in notif_text:
        notif_text = f"{notif_text} — {fan_summary}"

    notif = _normalize_notification_payload(
        {
            "type": "trade",
            "priority": "HIGH" if int(fan_result.get("fan_heat", 0) or 0) >= 55 else "MEDIUM",
            "title": "Trade Executed",
            "headline": notif_text[:120],
            "text": notif_text[:220],
            "source": "trade_hub",
            "fan_reaction_score": fan_result.get("fan_reaction_score"),
            "fan_heat": fan_result.get("fan_heat"),
            "fan_category": fan_result.get("fan_category"),
            "fan_heat_label": fan_result.get("fan_heat_label"),
            "fan_factors": list(fan_result.get("fan_factors") or [])[:3],
            "fan_effects": dict(fan_result.get("fan_effects") or {}),
        },
        index=len(session.notifications or []),
    )
    session.notifications.append(notif)
    _record_storyline(
        session,
        {
            "type": "trade",
            "priority": "HIGH",
            "headline": notif_text[:120],
            "details": fan_summary or f"Moved {len(moved_players)} players via Trade Hub package.",
            "players": [str(m.get("player_name") or "") for m in moved_players],
            "team_id": utid,
            "fan_reaction_score": fan_result.get("fan_reaction_score"),
            "fan_heat": fan_result.get("fan_heat"),
        },
    )
    trade_iso = _calendar_iso_for_day(session, trade_day)
    timeline_bits = [fan_headline] if fan_headline else [f"TRADE HUB: moved {len(moved_assets)} asset(s)."]
    session.timeline.append(
        _normalized_timeline_event(
            event_id=f"timeline:trade_hub:{trade_day}:{uuid.uuid4().hex[:8]}",
            event_type="trade",
            text=timeline_bits[0][:160],
            calendar_day=trade_day,
            calendar_iso=trade_iso,
            team_id=utid,
            priority="HIGH",
        )
    )


def execute_trade_package(session: FranchiseSession, *, assets_by_team: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Deprecated alias — delegates to trade_service.execute_franchise_trade for full validation.

    Previously moved NHL players only with no picks, retention, cap, clause, or AI checks.
    """
    from services.trade_service import execute_franchise_trade

    exec_result = execute_franchise_trade(
        session,
        assets_by_team=dict(assets_by_team or {}),
        record_notifications_fn=_record_legacy_trade_hub_notifications,
    )
    invalidate_session_payload_caches(session, reason="trade_exec")
    moved_players = [
        m for m in (exec_result.get("moved_players") or exec_result.get("moved_assets") or [])
        if m.get("applied", True)
    ]
    headline = str(exec_result.get("headline") or "")
    return {
        "moved_assets": moved_players,
        "headline": headline,
        "moved_players": len(moved_players),
        "trade_id": exec_result.get("trade_id"),
        "execution": exec_result,
    }

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


def _user_stanley_cup_count(session: FranchiseSession) -> int:
    uid = str(session.user_team_id or "")
    if not uid:
        return 0
    count = 0
    for entry in getattr(session, "timeline", None) or []:
        text = str(entry or "")
        if "stanley_cup_winner" in text and f"champion={uid}" in text:
            count += 1
    return count


def _build_gm_world_payload(session: FranchiseSession) -> Dict[str, Any]:
    """Minimal GM office overview — no skill progression fields."""
    user_team = session.team_by_id.get(str(session.user_team_id or ""))
    cal_rec = _user_team_record_from_game_results(session)
    strategy = str(getattr(user_team, "strategy", "balanced") or "balanced").lower() if user_team else "balanced"
    window = str(getattr(user_team, "gm_window", None) or getattr(user_team, "window", "") or "").lower()
    if not window:
        if "rebuild" in strategy:
            window = "rebuild"
        elif "contend" in strategy or "win" in strategy or "all-in" in strategy:
            window = "contender"
        else:
            window = "balanced"

    gp = int(cal_rec.get("gp") or 0)
    points_pct = float(cal_rec.get("pts", 0)) / max(1, gp * 2) if gp > 0 else 0.0
    cap_space = float(getattr(user_team, "cap_space", 0) or 0) if user_team else 0.0

    return {
        "stanley_cups": _user_stanley_cup_count(session),
        "team_context": {
            "record": dict(cal_rec),
            "strategy": strategy,
            "window": window,
            "cap_space": round(cap_space, 2),
            "points_pct": round(points_pct, 3),
            "season": f"{session.season_calendar_year}–{int(session.season_calendar_year) + 1}",
            "phase": str(session.phase),
        },
    }


def build_state_payload(session: FranchiseSession, *, include_heavy: bool = False, crisis_tick: bool = False) -> Dict[str, Any]:
    from services.perf_profiler import span

    with span("state.build", heavy=bool(include_heavy)):
        return _build_state_payload_impl(session, include_heavy=include_heavy, crisis_tick=crisis_tick)


def _build_state_payload_impl(session: FranchiseSession, *, include_heavy: bool = False, crisis_tick: bool = False) -> Dict[str, Any]:
    _sync_nhl_calendar_bounds(session)
    _sync_session_phase_from_calendar(session)
    ensure_session_nhl_salary_cap(session)
    # Schedule cadence is smoothed in start_franchise only. Re-running _smooth_league_schedule
    # here made every GET /api/franchise/state take minutes (full league re-optimization).
    sim = session.sim
    user_team = session.team_by_id.get(str(session.user_team_id))
    if user_team is None:
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

    ensure_session_nhl_salary_cap(session)
    sy_cap = int(getattr(session, "season_calendar_year", 2025) or 2025)
    league_for_cap = getattr(sim, "league", None)
    cap_snapshot_full: Dict[str, Any] = {}
    if user_team is not None:
        try:
            from services.contract_economy import get_team_cap_snapshot_full, sync_team_cap_fields

            cap_snapshot_full = sync_team_cap_fields(
                user_team,
                league_for_cap,
                sim,
                season_year=sy_cap,
                calendar_cursor=int(getattr(session, "calendar_cursor", 0) or 0),
                regular_season_last_index=int(getattr(session, "nhl_regular_season_last_index", 192) or 192),
            )
            if not cap_snapshot_full:
                cap_snapshot_full = get_team_cap_snapshot_full(
                    user_team,
                    league_for_cap,
                    sim,
                    season_year=sy_cap,
                    calendar_cursor=int(getattr(session, "calendar_cursor", 0) or 0),
                    regular_season_last_index=int(getattr(session, "nhl_regular_season_last_index", 192) or 192),
                )
        except Exception:
            cap_snapshot_full = {}

    if cap_snapshot_full:
        cap_info = {
            "salary_cap": float(cap_snapshot_full.get("upper_limit_m") or 95.5),
            "cap_hit": float(cap_snapshot_full.get("total_cap_hit_m") or 0.0),
            "cap_space": float(cap_snapshot_full.get("usable_cap_space_m") or 0.0),
        }
    elif user_team is not None:
        cap_info = _team_cap_snapshot(user_team, sim, season_year=sy_cap)
    else:
        cap_info = {"salary_cap": 95.5, "cap_hit": 0.0, "cap_space": 95.5}
    cap_hint = str(getattr(user_team, "cap_pressure", "moderate") if user_team else "?")
    strat = str(getattr(user_team, "strategy", "balanced") if user_team else "?")

    day_display = "Off-season"
    prog = None
    nhl_today = _nhl_today_payload(session)
    nhl_strip = _nhl_calendar_strip(session)
    season_lbl = f"{session.season_calendar_year}–{int(session.season_calendar_year) + 1}"
    if session.phase in ("regular", "preseason") and session.nhl_calendar:
        last = int(session.nhl_regular_season_last_index)
        cur = int(session.calendar_cursor)
        if cur < len(session.nhl_calendar) and (session.phase == "preseason" or cur <= last):
            cd = session.nhl_calendar[min(cur, len(session.nhl_calendar) - 1)]
            wd = str(cd.get("weekday") or "").strip()
            phase_label = str(cd.get("ui_phase") or ("Training Camp" if session.phase == "preseason" else ""))
            day_display = (
                f"{cd.get('iso', '')}"
                + (f" ({wd})" if wd else "")
                + (f" — {phase_label}" if phase_label else "")
            )
            if session.phase == "regular" and cur <= last:
                prog = f"{cur + 1} / {last + 1}"
            elif session.phase == "preseason":
                prog = f"Camp · day {cur + 1}"
        elif session.phase == "regular":
            day_display = "Regular season complete — advance for playoffs"
            prog = f"{last + 1} / {last + 1}"
    elif session.phase == "complete":
        day_display = f"Season complete — Cup: {session.champion_id or '?'}"

    try:
        _merge_simengine_league_news_into_storylines(session)
    except Exception:
        pass

    notifications_raw = list(session.notifications[-56:])
    notifications_norm = [_normalize_notification_payload(n, i) for i, n in enumerate(notifications_raw)]
    storylines_norm = [_normalize_storyline_payload(ev if isinstance(ev, dict) else {"headline": str(ev or "")}) for ev in list(getattr(session, "storyline_events", None) or [])[-120:]]
    storyline_choices = _storyline_choices_payload(session)
    try:
        from app.sim_engine.franchise.storyline_engine import build_narrative_universe_payload  # noqa: WPS433

        narrative_universe = build_narrative_universe_payload(session)
    except Exception:
        narrative_universe = {}
    try:
        from services.trade_demand_engine import (  # noqa: WPS433
            build_trade_demand_crisis_payload,
            get_trade_deadline_context,
        )

        trade_demand_crisis = build_trade_demand_crisis_payload(session, tick_timers=crisis_tick)
        trade_deadline = get_trade_deadline_context(session)
    except Exception:
        trade_demand_crisis = None
        trade_deadline = {}
    injuries_payload = _build_injuries_payload(session)
    injury_history_payload = _build_injury_history_payload(session)

    draft_class_board: Dict[str, Any] = {}
    if include_heavy:
        draft_class_board = get_cached_draft_class_rankings(session, sim)

    phase_now = str(getattr(session, "phase", "") or "")
    postseason_lean = phase_now in (
        "playoffs",
        "playoff_ready",
        "post_cup",
        "offseason",
        "preseason",
    )
    # Full calendar is multi‑MB once scores pile up. Lean advance payloads use a
    # near-cursor window; postseason omits it entirely (UI uses schedule_upcoming).
    if postseason_lean:
        nhl_calendar_full: List[Dict[str, Any]] = []
    elif include_heavy:
        nhl_calendar_full = _nhl_calendar_full_with_slates(session)
    else:
        # Advance/day response: tighter window — Calendar screen can refetch full.
        nhl_calendar_full = _nhl_calendar_full_with_slates(session, cursor_window=(21, 14))

    # Build once — both keys must reference the SAME list so GameUIContext cannot
    # double-concat snake+camel into a 2x pending queue / 2x wire cost.
    pending_snap = _pending_decision_snapshot(session)

    payload = {
        "session_id": session.session_id,
        "user_team_id": str(session.user_team_id),
        "phase": session.phase,
        "season_year": session.season_calendar_year,
        "player_universe": str(getattr(session, "player_universe", None) or "generated"),
        "games_per_team_schedule": int(getattr(session, "games_per_team_schedule", 82) or 82),
        "calendar_summary": day_display,
        "progress": prog,
        "nhl_season_label": season_lbl,
        "nhl_today": nhl_today,
        "nhl_calendar_strip": nhl_strip,
        "nhl_calendar_full": nhl_calendar_full,
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
            "cap_snapshot": cap_snapshot_full,
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
            **(
                lambda _fp: {
                    "fan_profile": _fp,
                    "fan_morale": _fp.get("fan_confidence"),
                    "fan_satisfaction": _fp.get("fan_confidence"),
                }
            )(_get_team_fan_profile(session, str(session.user_team_id or ""))),
        },
        "pending_decisions": pending_snap,
        "pendingDecisions": pending_snap,
        # Oldest-first pending queue: never hide older undismissed items behind a newest-only tail.
        "pending_ui_popups": list(getattr(session, "pending_ui_popups", None) or [])[:64],
        "storyline_choices": storyline_choices,
        "notifications": notifications_norm,
        "timeline": list(session.timeline[-80:]),
        "storyline_events": storylines_norm,
        "active_storylines": len(storylines_norm),
        "narrative_universe": narrative_universe,
        "trade_demand_crisis": trade_demand_crisis,
        "trade_deadline": trade_deadline,
        "trade_stability_roster_flags": dict(getattr(session, "trade_stability_roster_flags", None) or {}),
        "conduct_org_pressure": dict(getattr(session, "_conduct_org_pressure", None) or {}),
        "injuries": injuries_payload,
        "injury_history": injury_history_payload,
        "roster": roster_rows[:28],
        "calendar_events": list(getattr(session, "calendar_events", []) or [])[-48:],
        "schedule_diagnostics": getattr(session, "schedule_diagnostics", {}) or {},
        # Full NHL slate (32) — calendar standings snapshot needs every club + abbrevs.
        "standings": standings_rows[:32],
        "stats_revision": int(getattr(session, "_stats_revision", 0) or 0),
        "prospect_revision": int(getattr(session, "_prospect_revision", 0) or 0),
        "cpu_franchise_profiles": dict(getattr(session, "cpu_franchise_profiles", None) or {}),
        "schedule_upcoming": _build_schedule_upcoming(session, limit=14),
        "flags": {
            "playoffs_done": session.playoffs_simulated,
            "can_advance": len(session.pending_decisions) == 0 and session.phase != "complete",
        },
        "showcase_archive": list(getattr(session, "showcase_archive", None) or [])[-24:],
        "lines": dict(getattr(session, "lines", None) or {}),
        "financials_status": str(getattr(session, "financials_status", "") or ""),
        "wjc_nations": [{"code": c, "label": lab} for c, lab in _wjc_countries_meta()],
    }
    try:
        payload["wjc_tournament"] = _build_wjc_client_payload(session)
    except Exception:
        payload["wjc_tournament"] = None

    if include_heavy:
        payload["roster_browser"] = get_cached_roster_browser(session, sim, str(session.user_team_id))
        payload["draft_class_rankings"] = draft_class_board
        payload["draft_class_hud"] = get_cached_draft_class_hud(
            session,
            user_team,
            cal_rec,
            roster_rows,
            draft_class_board.get("entries"),
        )
    payload["gm_world"] = _build_gm_world_payload(session)
    try:
        from services.league_operations import (
            build_franchise_pulse,
            get_cached_league_operations_payload,
            slim_league_operations_for_state,
        )

        league_ops = get_cached_league_operations_payload(session)
        payload["franchise_pulse"] = build_franchise_pulse(session, league_ops)
        # Lean state: drop the full 32-team revenue table (dedicated route has it).
        # Heavy / League Ops screen still gets the full table via /league-operations.
        if include_heavy:
            payload["league_operations"] = league_ops
        else:
            payload["league_operations"] = slim_league_operations_for_state(league_ops)
    except Exception:
        payload["league_operations"] = {}
        payload["franchise_pulse"] = {}
    try:
        from services.franchise_offseason import build_offseason_state_extras

        extras = build_offseason_state_extras(session)
        extra_flags = dict(extras.pop("flags", {}) or {})
        payload.update(extras)
        payload["flags"] = {**payload.get("flags", {}), **extra_flags}
        payload["phase"] = str(session.phase)
        payload["season_phase"] = str(getattr(session, "season_phase", session.phase) or session.phase)
        payload["next_important_event"] = str(getattr(session, "next_important_event", "") or "")
        payload["playoff_payload"] = dict(getattr(session, "playoff_payload", None) or {})
        if getattr(session, "playoff_live", None):
            try:
                from services.franchise_playoffs import slim_live_for_client

                slim = slim_live_for_client(session.playoff_live)
            except Exception:
                slim = dict(session.playoff_live)
            payload["playoff_live"] = slim
            # Avoid shipping a second full copy of every series game log.
            payload["playoff_payload"]["live_state"] = slim
            for key in ("series", "series_list", "first_round", "first_round_matchups", "matchups"):
                payload["playoff_payload"].pop(key, None)
    except Exception:
        # Still expose playoff state during playoffs even if offseason extras fail.
        try:
            payload["phase"] = str(session.phase)
            payload["season_phase"] = str(getattr(session, "season_phase", session.phase) or session.phase)
            payload["playoff_payload"] = dict(getattr(session, "playoff_payload", None) or {})
            if getattr(session, "playoff_live", None):
                try:
                    from services.franchise_playoffs import slim_live_for_client

                    payload["playoff_live"] = slim_live_for_client(session.playoff_live)
                except Exception:
                    payload["playoff_live"] = dict(session.playoff_live)
        except Exception:
            pass
    try:
        from services.franchise_scouting import _ensure_scouting_state, DEFAULT_SCOUTING_BUDGET

        ss = _ensure_scouting_state(session)
        payload["scouting_state"] = {
            "budget": float(ss.get("budget") or DEFAULT_SCOUTING_BUDGET),
            "used_budget": float(ss.get("used_budget") or 0.0),
            "watchlist": list(ss.get("watchlist") or []),
            "active_deployments": list(ss.get("active_deployments") or []),
            "prospects": dict(ss.get("prospects") or {}),
        }
    except Exception:
        payload["scouting_state"] = {}
    try:
        import os
        from app.sim_engine.franchise.storyline_engine import build_storyline_debug_payload  # noqa: WPS433

        if os.environ.get("NODE_ENV", "") == "development" or os.environ.get("NHL_FRANCHISE_DEBUG", "0") == "1":
            payload["storyline_debug"] = build_storyline_debug_payload(session)
    except Exception:
        pass
    try:
        from services.transcendent_tank_behavior import check_dev_league_generation_version

        league_obj = getattr(sim, "league", None)
        gen_check = check_dev_league_generation_version(league_obj) if league_obj is not None else {}
        payload["dev_league_generation"] = gen_check
        if gen_check.get("needs_rebootstrap"):
            warn = str(gen_check.get("warning") or "")
            if warn:
                payload.setdefault("warnings", [])
                if warn not in payload["warnings"]:
                    payload["warnings"].append(warn)
                _startup_log.warning("DEV_LEAGUE_REBOOTSTRAP_NEEDED: %s", gen_check)
    except Exception:
        pass

    # Postseason UI does not need the full NHL calendar — drops ~300KB+ per response.
    phase_now = str(payload.get("phase") or session.phase or "")
    if phase_now in ("playoffs", "playoff_ready", "post_cup", "offseason", "preseason"):
        payload.pop("nhl_calendar_full", None)
        # Calendar ledger is season-sized and unused after Cup — keep a tiny tail.
        payload["calendar_events"] = list(payload.get("calendar_events") or [])[-12:]
        payload["pending_ui_popups"] = list(payload.get("pending_ui_popups") or [])[:64]
        payload.pop("pendingUiPopups", None)
        # Prefer slim live already applied above; keep awards ceremony-sized.
        if isinstance(payload.get("awards"), dict):
            try:
                from services.franchise_offseason import slim_awards_payload_for_client

                payload["awards"] = slim_awards_payload_for_client(payload.get("awards"))
            except Exception:
                pass
    return payload


def build_state_payload_safe(session: FranchiseSession, *, include_heavy: bool = False, crisis_tick: bool = False) -> Dict[str, Any]:
    """Never fail Cup/offseason advances on a state-serialize bug — return a lean shell."""
    from services.json_safe import json_safe
    from services.perf_profiler import span

    with span("state.build_safe", heavy=bool(include_heavy)):
        try:
            return json_safe(build_state_payload(session, include_heavy=include_heavy, crisis_tick=crisis_tick))
        except Exception as exc:  # noqa: BLE001
            try:
                from services.franchise_offseason import slim_awards_payload_for_client

                awards = slim_awards_payload_for_client(getattr(session, "awards_payload", None))
            except Exception:
                awards = {}
            return json_safe({
                "session_id": getattr(session, "session_id", None),
                "user_team_id": str(getattr(session, "user_team_id", "") or ""),
                "phase": str(getattr(session, "phase", "") or ""),
                "season_phase": str(getattr(session, "season_phase", getattr(session, "phase", "")) or ""),
                "offseason_stage": getattr(session, "offseason_stage", None),
                "next_important_event": str(getattr(session, "next_important_event", "") or ""),
                "champion_id": getattr(session, "champion_id", None),
                "stanley_cup_winner": getattr(session, "stanley_cup_winner", None) or getattr(session, "champion_id", None),
                "playoffs_done": bool(getattr(session, "playoffs_simulated", False)),
                "awards": awards,
                "flags": {
                    "playoffs_done": bool(getattr(session, "playoffs_simulated", False)),
                    "can_continue_offseason": str(getattr(session, "phase", "")) in ("post_cup", "offseason"),
                    "can_enter_playoffs": str(getattr(session, "phase", "")) == "playoff_ready",
                },
                "state_build_error": str(exc),
            })


def save_franchise_lines(session: FranchiseSession, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and persist user lineups (even-strength / PP / PK) on the session.

    Stores whatever units the UI sends under payload["unit_type"] (default
    "even_strength"). Validation warns on unknown players, duplicates, and
    position mismatches but only rejects malformed payloads.
    """
    if not isinstance(payload, dict):
        raise ValueError("Lineup payload must be an object")
    unit_type = str(payload.get("unit_type") or "even_strength")
    lines = payload.get("lines")
    if not isinstance(lines, (dict, list)):
        raise ValueError("Lineup payload missing 'lines'")

    user_team = session.team_by_id.get(str(session.user_team_id))
    roster_by_id: Dict[str, Any] = {}
    for p in getattr(user_team, "roster", None) or []:
        pid = str(getattr(p, "id", "") or "")
        if pid:
            roster_by_id[pid] = p

    warnings: List[str] = []
    seen: Dict[str, str] = {}

    def _check_slot(slot_name: str, pid: Any) -> None:
        spid = str(pid or "")
        if not spid:
            return
        player = roster_by_id.get(spid)
        if player is None:
            warnings.append(f"{slot_name}: player {spid} is not on the NHL roster")
            return
        pos = _pos_str(player).upper()
        slot_u = slot_name.split(":")[-1].strip().upper()
        if slot_u in ("STARTER", "BACKUP", "THIRD") and pos != "G":
            warnings.append(f"{slot_name}: {_name_str(player)} is not a goalie")
        if slot_u in ("LW", "C", "RW", "LD", "RD") and pos == "G":
            warnings.append(f"{slot_name}: {_name_str(player)} is a goalie in a skater slot")
        elif slot_u in ("LD", "RD") and pos not in ("D", "LD", "RD"):
            warnings.append(f"{slot_name}: {_name_str(player)} ({pos}) placed on defense")
        elif slot_u == "C" and pos in ("LW", "RW"):
            warnings.append(f"{slot_name}: {_name_str(player)} is a winger at center")
        if spid in seen:
            warnings.append(f"{_name_str(player) if player else spid} assigned to both {seen[spid]} and {slot_name}")
        else:
            seen[spid] = slot_name

    def _walk_units(units: Any, prefix: str) -> None:
        if isinstance(units, list):
            for unit in units:
                if not isinstance(unit, dict):
                    continue
                label = str(unit.get("name") or unit.get("id") or prefix)
                for slot, pid in (unit.get("slots") or {}).items():
                    _check_slot(f"{label}:{slot}", pid)
        elif isinstance(units, dict):
            for slot, pid in units.items():
                _check_slot(f"{prefix}:{slot}", pid)

    if isinstance(lines, dict):
        for group_name, units in lines.items():
            _walk_units(units, str(group_name))
    else:
        _walk_units(lines, unit_type)

    if not isinstance(session.lines, dict):
        session.lines = {}
    prev_saved = dict(session.lines.get(unit_type) or {})
    try:
        from app.sim_engine.franchise.storyline_engine import record_lineup_save_decisions  # noqa: WPS433

        record_lineup_save_decisions(
            session,
            new_lines=lines,
            unit_type=unit_type,
            previous_lines=prev_saved.get("lines"),
        )
    except Exception:
        pass
    session.lines[unit_type] = {
        "lines": lines,
        "warnings": warnings,
        "last_saved": _today_iso(session),
        "source": "user",
    }
    return {"ok": True, "unit_type": unit_type, "warnings": warnings, "lines": session.lines}


def _today_iso(session: FranchiseSession) -> str:
    try:
        cal = getattr(session, "nhl_calendar", None) or []
        cur = int(getattr(session, "calendar_cursor", 0) or 0)
        if 0 <= cur < len(cal):
            return str(cal[cur].get("iso") or cal[cur].get("date") or "")
    except Exception:
        pass
    return ""


# --- Post-split helpers (state caches, league teams, contract aliases) ---

def invalidate_session_payload_caches(session: FranchiseSession, reason: str = "") -> None:
    """Drop cached read-model payloads after mutating session state."""
    # A single draft selection does NOT change any prospect's consensus ranking or
    # score — it only marks one player drafted, and availability is filtered
    # downstream against drafted_prospect_ids. Rebuilding the entire draft board on
    # every pick was the dominant per-pick cost (build_draft_class_rankings rescans
    # and re-scores the whole development pool). Keep the rankings cache valid across
    # the live draft; every other reason (prospect_stats, game_stats, season_reset,
    # scouting, trades, …) still refreshes it, so the board is never stale between
    # drafts or after any evaluation-changing event.
    #
    # Also: do NOT bump `_prospect_revision` on draft_pick. Rankings/HUD caches are
    # keyed on that revision, so bumping it silently defeated the rankings-cache
    # keep-alive above and forced a full rebuild after every selection.
    if reason == "draft_pick":
        # Selection only mutates rights / drafted set. Clear roster/scouting views
        # that list undrafted prospects, but leave rankings + trade market warm.
        session._cached_scouting_prospects_payload = None
        session._cached_roster_browser_payload = None
        session.draft_payload = None
        return

    if reason != "draft_pick":
        session._draft_rankings_cache_state = "dirty"
    session._cached_trade_assets_payload = None
    session._cached_scouting_prospects_payload = None
    session._cached_scouting_world_payload = None
    session._cached_roster_browser_payload = None
    session._cached_league_operations = None
    session._cached_league_operations_key = None
    session._cached_lean_state_payload = None
    session._cached_lean_state_key = None
    if reason in (
        "trade_exec",
        "game_stats",
        "season_reset",
        "player_stats",
        # Day advance may include CPU trades and/or games; always bust stats +
        # roster_browser caches so lean /state merges do not keep stale teams.
        "advance_day",
        "bulk_complete",
    ):
        _bump_stats_revision(session)
    if reason in (
        "prospect_stats",
        "wjc_pre",
        "wjc_post",
        "scouting_meta",
        "scouting_action",
        "draft_combine",
        "combine_meeting",
    ):
        _bump_prospect_revision(session)
    if _fr_dbg_enabled():
        try:
            from app.sim_engine.trades.trade_pick_registry import audit_pick_registry_integrity

            league = getattr(getattr(session, "sim", None), "league", None)
            if league is not None:
                audit = audit_pick_registry_integrity(
                    league,
                    start_year=int(getattr(session, "season_calendar_year", 2025) or 2025) + 1,
                    years_ahead=4,
                    rounds=7,
                )
                if not audit.get("ok"):
                    first = (audit.get("errors") or ["unknown registry issue"])[0]
                    _fr_dbg(
                        f"pick registry integrity warning after cache invalidation ({reason or 'unspecified'}): {first}"
                    )
        except Exception as exc:
            _fr_dbg(f"pick registry audit failed while invalidating caches: {exc}")


_STATS_CENTRAL_SANITIZER_VERSION = 3  # v3: early-season war_valid (gp>=3, toi>=45)


def get_cached_stats_central_payload(session: FranchiseSession) -> Dict[str, Any]:
    rev = int(getattr(session, "_stats_revision", 0) or 0)
    cached = getattr(session, "_stats_central_cache", None)
    if (
        isinstance(cached, dict)
        and int(cached.get("revision", -1)) == rev
        and int(cached.get("sanitizer", 0) or 0) == _STATS_CENTRAL_SANITIZER_VERSION
    ):
        payload = cached.get("payload")
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["cache_hit"] = True
            return payload
    payload = _build_stats_central_payload(session)
    # Re-read revision — repair may bump it during payload build.
    rev = int(getattr(session, "_stats_revision", 0) or 0)
    session._stats_central_cache = {
        "revision": rev,
        "sanitizer": _STATS_CENTRAL_SANITIZER_VERSION,
        "payload": payload,
    }
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["cache_hit"] = False
    return payload


def get_cached_draft_class_detail_payload(session: FranchiseSession, sim: Any) -> Dict[str, Any]:
    try:
        ensure_prospect_stats_current_for_scouting(session)
    except Exception:
        pass
    rev = int(getattr(session, "_prospect_revision", 0) or 0)
    cached = getattr(session, "_draft_class_detail_cache", None)
    if isinstance(cached, dict) and int(cached.get("revision", -1)) == rev:
        payload = cached.get("payload")
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["cache_hit"] = True
            return payload
    board = get_cached_draft_class_rankings(session, sim)
    hud = _build_draft_class_hud_payload(
        session,
        session.team_by_id.get(str(session.user_team_id)),
        _user_team_record_from_game_results(session),
        [],
        draft_entries=board.get("entries"),
    )
    # Re-read after sync/retune may bump prospect revision.
    rev = int(getattr(session, "_prospect_revision", 0) or 0)
    payload = {"draft_class_rankings": board, "draft_class_hud": hud, "prospect_revision": rev}
    session._draft_class_detail_cache = {"revision": rev, "payload": payload}
    payload["cache_hit"] = False
    return payload


def get_cached_draft_class_rankings(session: FranchiseSession, sim: Any) -> Dict[str, Any]:
    # Key the board cache on the prospect revision (which only bumps when the prospect
    # pool/stats/scouting actually change) rather than the broad payload-dirty flag. This
    # stops unrelated gameplay (day advances, games, trades, cap moves) from forcing a full
    # draft-board rebuild every time the Draft Class screen is opened.
    league = getattr(sim, "league", None) if sim is not None else None
    # Bust stale boards built before top-slot OVR floors / attribute-lift removal.
    if not getattr(session, "_draft_board_ovr_floors_v3", False):
        session._cached_draft_class_rankings = None
        session._cached_draft_class_hud_payload = None
        session._draft_class_detail_cache = None
        session._prospect_revision = int(getattr(session, "_prospect_revision", 0) or 0) + 1
        setattr(session, "_draft_board_ovr_floors_v3", True)
        setattr(session, "_draft_pipeline_ovr_repaired", False)
    try:
        if league is not None and not getattr(session, "_draft_pipeline_ovr_repaired", False):
            from app.sim_engine.league_hierarchy_bootstrap import repair_undervalued_draft_pipeline_stars

            repaired = repair_undervalued_draft_pipeline_stars(
                league,
                getattr(sim, "rng", None),
            )
            setattr(session, "_draft_pipeline_ovr_repaired", True)
            if repaired:
                session._prospect_revision = int(getattr(session, "_prospect_revision", 0) or 0) + 1
                session._cached_draft_class_rankings = None
                session._cached_draft_class_hud_payload = None
                session._draft_class_detail_cache = None
    except Exception:
        setattr(session, "_draft_pipeline_ovr_repaired", True)
    try:
        if league is not None and not getattr(session, "_draft_age_reanchored", False):
            from app.sim_engine.league_hierarchy_bootstrap import (
                reanchor_generated_junior_dobs,
                set_spawn_as_of_year,
            )

            anchor = int(getattr(session, "season_calendar_year", 0) or 0)
            if anchor < 2000:
                anchor = int(session_age_as_of(session)[0])
            set_spawn_as_of_year(anchor)
            fixed = reanchor_generated_junior_dobs(league, anchor)
            setattr(session, "_draft_age_reanchored", True)
            if fixed:
                session._prospect_revision = int(getattr(session, "_prospect_revision", 0) or 0) + 1
                session._cached_draft_class_rankings = None
                session._cached_draft_class_hud_payload = None
                session._draft_class_detail_cache = None
    except Exception:
        setattr(session, "_draft_age_reanchored", True)

    rev = int(getattr(session, "_prospect_revision", 0) or 0)
    cached = getattr(session, "_cached_draft_class_rankings", None)
    cached_rev = int(getattr(session, "_cached_draft_class_rankings_rev", -1) or -1)
    if isinstance(cached, dict) and cached and cached_rev == rev:
        return cached

    # Single-flight: Draft Class opens heavy + scouting/prospects in parallel; without
    # a lock both miss cache and rebuild the entire board (~20–30s) twice.
    sid = str(getattr(session, "session_id", "") or id(session))
    with _DRAFT_RANKINGS_LOCKS_GUARD:
        lock = _DRAFT_RANKINGS_LOCKS.get(sid)
        if lock is None:
            lock = threading.Lock()
            _DRAFT_RANKINGS_LOCKS[sid] = lock
    with lock:
        rev = int(getattr(session, "_prospect_revision", 0) or 0)
        cached = getattr(session, "_cached_draft_class_rankings", None)
        cached_rev = int(getattr(session, "_cached_draft_class_rankings_rev", -1) or -1)
        if isinstance(cached, dict) and cached and cached_rev == rev:
            return cached
        payload = build_draft_class_rankings(session, sim)
        session._cached_draft_class_rankings = payload
        session._cached_draft_class_rankings_rev = int(getattr(session, "_prospect_revision", 0) or 0)
        session._draft_rankings_cache_state = "valid"
        return payload


def get_cached_draft_class_hud(
    session: FranchiseSession,
    user_team: Any,
    cal_rec: Any,
    roster_rows: List[Dict[str, Any]],
    draft_entries: Any,
) -> Dict[str, Any]:
    """Revision-keyed cache for the (expensive) HUD prospect profiles so the ~320-profile
    build isn't repeated on every Draft Class open. Rebuilds only when prospects change."""
    rev = int(getattr(session, "_prospect_revision", 0) or 0)
    cached = getattr(session, "_cached_draft_class_hud_payload", None)
    cached_rev = int(getattr(session, "_cached_draft_class_hud_rev", -1) or -1)
    if isinstance(cached, dict) and cached and cached_rev == rev:
        return cached
    payload = _build_draft_class_hud_payload(
        session, user_team, cal_rec, roster_rows, draft_entries=draft_entries
    )
    session._cached_draft_class_hud_payload = payload
    session._cached_draft_class_hud_rev = int(getattr(session, "_prospect_revision", 0) or 0)
    return payload


def get_cached_roster_browser(session: FranchiseSession, sim: Any, user_team_id: str) -> Dict[str, Any]:
    cached = getattr(session, "_cached_roster_browser_payload", None)
    if isinstance(cached, dict) and cached:
        return cached
    payload = _build_roster_browser(sim, user_team_id, franchise_session=session)
    session._cached_roster_browser_payload = payload
    return payload


def player_cap_hit_millions(player: Any) -> float:
    return _player_cap_hit_millions(player)


def _contract_years_remaining(player: Any) -> int:
    c = getattr(player, "contract", None)
    for obj in (player, c):
        if obj is None:
            continue
        for key in ("years_remaining", "term_remaining", "remaining_years", "term"):
            v = getattr(obj, key, None)
            if v is not None:
                try:
                    return max(0, int(v))
                except (TypeError, ValueError):
                    pass
    return 1


def _build_standings_rows(session: FranchiseSession) -> List[Dict[str, Any]]:
    """Lightweight standings for calendar/hub — includes abbrev + GF/GA/PP/PK when available."""
    standings_rows: List[Dict[str, Any]] = []
    uid_s = str(session.user_team_id or "")
    cal_rec = _user_team_record_from_game_results(session)
    team_by_id = getattr(session, "team_by_id", None) or {}
    rich_by_tid: Dict[str, Dict[str, Any]] = {}
    try:
        league_teams, _ = _build_league_teams_payload(session)
        for row in league_teams or []:
            if isinstance(row, dict):
                rich_by_tid[str(row.get("team_id") or "")] = row
    except Exception:
        rich_by_tid = {}

    if session.standings:
        for tid, r in session.standings.records.items():
            tid_s = str(tid)
            tm = team_by_id.get(tid) or team_by_id.get(tid_s)
            rich = rich_by_tid.get(tid_s) or {}
            abbrev = str(rich.get("abbrev") or rich.get("abbr") or "").strip().upper()
            if not abbrev and tm is not None:
                abbrev = _franchise_team_abbrev(tm)
            if not abbrev:
                abbrev = _franchise_team_abbrev(
                    type(
                        "_StandingsAbbrevProxy",
                        (),
                        {
                            "name": getattr(r, "name", tid),
                            "city": getattr(tm, "city", "") if tm is not None else "",
                            "abbr": None,
                            "code": None,
                            "abbreviation": None,
                            "short_name": None,
                            "team_id": tid_s,
                            "id": tid_s,
                        },
                    )()
                )

            if uid_s and tid_s == uid_s and int(cal_rec.get("gp") or 0) > 0:
                base = {
                    "team_id": tid_s,
                    "name": getattr(r, "name", tid),
                    "gp": int(cal_rec["gp"]),
                    "w": int(cal_rec["w"]),
                    "l": int(cal_rec["l"]),
                    "otl": int(cal_rec["otl"]),
                    "pts": int(cal_rec["pts"]),
                }
            else:
                base = {
                    "team_id": tid_s,
                    "name": getattr(r, "name", tid),
                    "gp": getattr(r, "gp", 0),
                    "w": getattr(r, "wins", 0),
                    "l": getattr(r, "losses", 0),
                    "otl": getattr(r, "otl", 0),
                    "pts": getattr(r, "points", 0),
                }

            base["abbrev"] = abbrev
            base["abbr"] = abbrev
            base["abbreviation"] = abbrev
            if tm is not None:
                base["division"] = str(getattr(tm, "division", None) or rich.get("division") or "")
                base["conference"] = str(getattr(tm, "conference", None) or rich.get("conference") or "")
            gf = rich.get("gf", rich.get("goals_for"))
            ga = rich.get("ga", rich.get("goals_against"))
            if gf is not None:
                base["gf"] = int(gf or 0)
                base["goals_for"] = int(gf or 0)
            if ga is not None:
                base["ga"] = int(ga or 0)
                base["goals_against"] = int(ga or 0)
            if rich.get("pp_pct") is not None:
                base["pp_pct"] = float(rich["pp_pct"])
                base["power_play_pct"] = float(rich["pp_pct"])
            if rich.get("pk_pct") is not None:
                base["pk_pct"] = float(rich["pk_pct"])
                base["penalty_kill_pct"] = float(rich["pk_pct"])
            standings_rows.append(base)
        standings_rows.sort(key=lambda x: (-x["pts"], -(x["w"] - x["l"])))
        for idx, row in enumerate(standings_rows, start=1):
            row["rank"] = idx
            row["league_rank"] = idx
    return standings_rows


def ensure_player_financials(player: Any, league: Any, season_y: int, team: Any = None) -> None:
    """Best-effort contract bootstrap before trade evaluation."""
    _ensure_player_contract(player, int(season_y or 2025))


def _serialize_player_trade_block(
    player: Any,
    *,
    source_team: Any,
    acquiring_team: Any,
    league: Any,
    session: Any,
    trade_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pid = str(getattr(player, "id", "") or "")
    ctx = dict(trade_context or {})
    if session is not None and not ctx:
        max_d = max(40, int(getattr(session, "nhl_regular_season_last_index", 192) or 192))
        md = max(40, int(max(120, max_d) * 0.56))
        cursor = int(getattr(session, "calendar_cursor", 0) or 0)
        ctx = {
            "season_year": int(getattr(session, "season_calendar_year", 2025) or 2025),
            "calendar_cursor": cursor,
            "regular_season_last_index": max_d,
            "deadline_phase": max(0.0, min(1.0, (float(cursor) - float(md)) / max(20.0, float(max_d) * 0.2))),
            "team_by_id": dict(getattr(session, "team_by_id", None) or {}),
        }
    tradeable = True
    trade_block_reason = ""
    clause_label = "None"
    approved: List[str] = []
    loc = "nhl"
    acq_id = str(
        getattr(acquiring_team, "team_id", None)
        or getattr(acquiring_team, "id", "")
        or ""
    )
    valuation: Dict[str, Any] = {}
    try:
        from services.franchise_paths import ensure_simengine_path

        ensure_simengine_path()
        from app.sim_engine.trades.trade_asset import (
            player_holds_nhl_spc,
            player_is_tradeable_draft_rights,
            player_trade_roster_location,
        )
        from app.sim_engine.trades.trade_rules import _clause_summary
        from app.sim_engine.trades.trade_value import evaluate_player_asset_value

        loc = player_trade_roster_location(source_team, pid)
        clause = _clause_summary(player)
        clause_label = str(clause.get("label") or "None")
        approved = list(clause.get("approved_destinations") or [])
        acq_id = str(
            getattr(acquiring_team, "team_id", None)
            or getattr(acquiring_team, "id", "")
            or acq_id
        )
        if loc == "prospect" and not player_holds_nhl_spc(player) and player_is_tradeable_draft_rights(player):
            pass  # unsigned draft rights are tradeable assets
        elif loc in ("ahl", "echl", "prospect") and not player_holds_nhl_spc(player):
            tradeable = False
            trade_block_reason = "Affiliate-only contract — NHL SPC required to trade"
        elif not loc and not player_holds_nhl_spc(player):
            tradeable = False
            trade_block_reason = "Player must be under NHL organizational control"
        elif clause.get("nmc"):
            tradeable = False
            trade_block_reason = "No-movement clause"
        elif clause.get("ntc"):
            waivers = {}
            if session is not None:
                waivers = dict(getattr(session, "ntc_waivers", None) or {})
            waiver = waivers.get(pid) or waivers.get(f"{pid}->{acq_id}")
            waived_ok = (
                isinstance(waiver, dict)
                and bool(waiver.get("accepted"))
                and (
                    not waiver.get("destination_team_id")
                    or str(waiver.get("destination_team_id")) == str(acq_id)
                )
            )
            if waived_ok:
                tradeable = True
                trade_block_reason = ""
                ctx = dict(ctx)
                ctx["ntc_waived"] = True
                ctx["ntc_value_penalty_pct"] = float(waiver.get("value_penalty_pct") or 0.08)
                ctx["ntc_waivers"] = waivers
            else:
                tradeable = False
                trade_block_reason = "No-trade clause — ask player to waive"
        elif clause.get("mntc", 0) > 0:
            can_to_partner = bool(approved) and acq_id in approved
            waivers = dict(getattr(session, "ntc_waivers", None) or {}) if session is not None else {}
            waiver = waivers.get(pid) or waivers.get(f"{pid}->{acq_id}")
            waived_ok = (
                isinstance(waiver, dict)
                and bool(waiver.get("accepted"))
                and (
                    not waiver.get("destination_team_id")
                    or str(waiver.get("destination_team_id")) == str(acq_id)
                )
            )
            tradeable = can_to_partner or waived_ok
            if not tradeable:
                trade_block_reason = "Modified no-trade clause — ask player to waive"
            elif waived_ok and not can_to_partner:
                ctx = dict(ctx)
                ctx["ntc_waived"] = True
                ctx["ntc_value_penalty_pct"] = float(waiver.get("value_penalty_pct") or 0.08)
                ctx["ntc_waivers"] = waivers
        valuation = evaluate_player_asset_value(
            player, source_team, acquiring_team, league, context=ctx,
        )
    except Exception:
        valuation = {}

    requires_ntc_waive = bool(clause_label in ("NTC", "M-NTC") and not tradeable and trade_block_reason)
    ntc_waived = False
    ntc_waiver_reason = ""
    if session is not None:
        waivers = dict(getattr(session, "ntc_waivers", None) or {})
        w = waivers.get(pid) or waivers.get(f"{pid}->{acq_id}")
        if isinstance(w, dict) and bool(w.get("accepted")):
            if not w.get("destination_team_id") or str(w.get("destination_team_id")) == str(acq_id):
                ntc_waived = True
                ntc_waiver_reason = str(w.get("reason") or "")
                requires_ntc_waive = False

    row = {
        "player_id": pid,
        "name": _name_str(player),
        "position": _pos_str(player),
        "ovr": round(float(_ovr_weight(player) * 99.0), 1),
        "cap_hit": round(_player_cap_hit_millions(player), 3),
        "contract_years_remaining": _contract_years_remaining(player),
        "source_team_id": str(getattr(source_team, "team_id", "") or ""),
        "clause_label": clause_label,
        "tradeable": tradeable,
        "trade_block_reason": trade_block_reason,
        "requires_ntc_waive": requires_ntc_waive or (clause_label == "NTC" and not ntc_waived),
        "ntc_waived": ntc_waived,
        "ntc_waiver_reason": ntc_waiver_reason,
        "approved_trade_teams": approved,
        "approved_trade_team_ids": approved,
        "can_trade_to_partner": tradeable if acq_id else None,
        "assignment_level": loc or "nhl",
        "org_level": loc or "nhl",
    }
    if valuation:
        row.update({
            "trade_value": valuation.get("total"),
            "value_tier": valuation.get("value_tier"),
            "breakdown": valuation.get("components") or valuation.get("breakdown") or {},
            "explain": list(valuation.get("explain") or []),
            "risk_flags": list(valuation.get("risk_flags") or []),
            "contract_flags": list(valuation.get("contract_flags") or []),
            "cap_impact": valuation.get("cap_impact"),
        })
    return row


def _empty_league_team_row(tid: str) -> Dict[str, Any]:
    return {
        "team_id": tid,
        "id": tid,
        "gp": 0,
        "w": 0,
        "l": 0,
        "otl": 0,
        "pts": 0,
        "gf": 0,
        "ga": 0,
        "sf": 0,
        "sa": 0,
        "ppg": 0,
        "ppo": 0,
        "ppga": 0,
        "opp_ppo": 0,
        "xgf_pct_sum": 0.0,
        "xgf_pct_gp": 0,
    }


def _build_league_teams_payload(
    session: FranchiseSession,
    *,
    team_totals: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Full league team analytics rows + lightweight directory for Stats Central."""
    rows_by_tid: Dict[str, Dict[str, Any]] = {}

    for tid_raw, tm in (getattr(session, "team_by_id", None) or {}).items():
        tid = str(tid_raw)
        row = _empty_league_team_row(tid)
        row["name"] = _display_team(tm)
        abbr = _franchise_team_abbrev(tm)
        row["abbrev"] = abbr
        row["abbr"] = abbr
        rows_by_tid[tid] = row

    st = getattr(session, "standings", None)
    if st is not None:
        for tid_raw, rec in (getattr(st, "records", None) or {}).items():
            tid = str(tid_raw)
            row = rows_by_tid.setdefault(tid, _empty_league_team_row(tid))
            if not row.get("name"):
                row["name"] = str(getattr(rec, "name", tid) or tid)
            row["gp"] = int(getattr(rec, "gp", 0) or 0)
            row["w"] = int(getattr(rec, "wins", 0) or 0)
            row["l"] = int(getattr(rec, "losses", 0) or 0)
            row["otl"] = int(getattr(rec, "otl", 0) or 0)
            row["pts"] = int(getattr(rec, "points", 0) or 0)
            row["wins"] = row["w"]
            row["losses"] = row["l"]
            row["points"] = row["pts"]

    for g in getattr(session, "game_results", None) or []:
        if not isinstance(g, dict):
            continue
        hid = str(g.get("home_id") or "")
        aid = str(g.get("away_id") or "")
        if not hid or not aid:
            continue
        try:
            hg = int(g.get("player_home_goals", g.get("hockey_home_goals", g.get("home_goals", g.get("home_score", 0)))) or 0)
            ag = int(g.get("player_away_goals", g.get("hockey_away_goals", g.get("away_goals", g.get("away_score", 0)))) or 0)
        except (TypeError, ValueError):
            continue
        if hg < 0 or ag < 0 or hg == ag:
            continue

        home_shots = int(g.get("home_shots", g.get("home_sog", 0)) or 0)
        away_shots = int(g.get("away_shots", g.get("away_sog", 0)) or 0)
        home_cf = int(g.get("home_shot_attempts", g.get("home_cf", home_shots)) or 0)
        away_cf = int(g.get("away_shot_attempts", g.get("away_cf", away_shots)) or 0)
        home_ff = int(g.get("home_ff", g.get("home_fenwick", home_shots)) or 0)
        away_ff = int(g.get("away_ff", g.get("away_fenwick", away_shots)) or 0)
        home_xgf = float(g.get("home_xgf", g.get("home_xg", 0)) or 0)
        away_xgf = float(g.get("away_xgf", g.get("away_xg", 0)) or 0)

        for tid, gf, ga, sf, sa, cf, ca, ff, fa, xgf, xga, ppg, ppo, ppga, opp_ppo in (
            (
                hid,
                hg,
                ag,
                home_shots,
                away_shots,
                home_cf,
                away_cf,
                home_ff,
                away_ff,
                home_xgf,
                away_xgf,
                int(g.get("home_pp_goals", 0) or 0),
                int(g.get("home_ppo", 0) or 0),
                int(g.get("home_ppga", g.get("away_pp_goals", 0)) or 0),
                int(g.get("home_opp_ppo", g.get("away_ppo", 0)) or 0),
            ),
            (
                aid,
                ag,
                hg,
                away_shots,
                home_shots,
                away_cf,
                home_cf,
                away_ff,
                home_ff,
                away_xgf,
                home_xgf,
                int(g.get("away_pp_goals", 0) or 0),
                int(g.get("away_ppo", 0) or 0),
                int(g.get("away_ppga", g.get("home_pp_goals", 0)) or 0),
                int(g.get("away_opp_ppo", g.get("home_ppo", 0)) or 0),
            ),
        ):
            row = rows_by_tid.setdefault(tid, _empty_league_team_row(tid))
            row["gf"] += gf
            row["ga"] += ga
            row["sf"] += sf
            row["sa"] += sa
            row["cf"] = int(row.get("cf", 0) or 0) + int(cf)
            row["ca"] = int(row.get("ca", 0) or 0) + int(ca)
            row["ff"] = int(row.get("ff", 0) or 0) + int(ff)
            row["fa"] = int(row.get("fa", 0) or 0) + int(fa)
            row["xgf"] = round(float(row.get("xgf", 0) or 0) + float(xgf), 4)
            row["xga"] = round(float(row.get("xga", 0) or 0) + float(xga), 4)
            row["ppg"] += ppg
            row["ppo"] += ppo
            row["ppga"] += ppga
            row["opp_ppo"] += opp_ppo

    if team_totals:
        for tid, ttot in team_totals.items():
            row = rows_by_tid.setdefault(str(tid), _empty_league_team_row(str(tid)))
            if not row.get("gf"):
                row["gf"] = int(ttot.get("goals", 0) or 0)

    league_teams: List[Dict[str, Any]] = []
    for row in rows_by_tid.values():
        gf = int(row.get("gf", 0) or 0)
        ga = int(row.get("ga", 0) or 0)
        sf = int(row.get("sf", 0) or 0)
        sa = int(row.get("sa", 0) or 0)
        cf = int(row.get("cf", 0) or 0)
        ca = int(row.get("ca", 0) or 0)
        ff = int(row.get("ff", 0) or 0)
        fa = int(row.get("fa", 0) or 0)
        xgf = float(row.get("xgf", 0.0) or 0.0)
        xga = float(row.get("xga", 0.0) or 0.0)
        ppg = int(row.get("ppg", 0) or 0)
        ppo = int(row.get("ppo", 0) or 0)
        ppga = int(row.get("ppga", 0) or 0)
        opp_ppo = int(row.get("opp_ppo", 0) or 0)

        row["goals_for"] = gf
        row["goals_against"] = ga
        row["goal_diff"] = gf - ga
        row["shots_for"] = sf
        row["shots_against"] = sa
        row["corsi_for"] = cf
        row["corsi_against"] = ca
        row["shot_attempts_for"] = cf
        row["shot_attempts_against"] = ca
        row["fenwick_for"] = ff
        row["fenwick_against"] = fa
        row["expected_goals_for"] = round(xgf, 4)
        row["expected_goals_against"] = round(xga, 4)

        if ppo > 0:
            row["pp_pct"] = ppg / float(ppo)
            row["power_play_pct"] = row["pp_pct"]
        if opp_ppo > 0:
            row["pk_pct"] = 1.0 - (ppga / float(opp_ppo))
            row["penalty_kill_pct"] = row["pk_pct"]
        if cf + ca > 0:
            row["cf_pct"] = cf / float(cf + ca)
            row["corsi_pct"] = row["cf_pct"]
        if ff + fa > 0:
            row["ff_pct"] = ff / float(ff + fa)
            row["fenwick_pct"] = row["ff_pct"]
        if xgf + xga > 0:
            row["xgf_pct"] = xgf / float(xgf + xga)
        else:
            row["xgf_pct"] = None
        sh_pct = gf / float(sf) if sf > 0 else 0.0
        sv_pct = (sa - ga) / float(sa) if sa > 0 else 0.0
        row["sh_pct"] = sh_pct if sf > 0 else None
        row["shooting_pct"] = row["sh_pct"]
        row["sv_pct"] = sv_pct if sa > 0 else None
        row["save_pct"] = row["sv_pct"]
        row["pdo"] = round((sh_pct + sv_pct) * 100.0, 1) if sf > 0 and sa > 0 else None
        row["pdo_valid"] = bool(sf > 0 and sa > 0)

        league_teams.append(row)

    league_teams.sort(
        key=lambda r: (
            -int(r.get("pts", 0) or 0),
            -int(r.get("goal_diff", 0) or 0),
            -int(r.get("gf", 0) or 0),
        )
    )
    for idx, row in enumerate(league_teams, start=1):
        row["league_rank"] = idx

    teams_directory = [
        {
            "team_id": str(r.get("team_id") or ""),
            "id": str(r.get("team_id") or ""),
            "name": str(r.get("name") or ""),
            "abbrev": str(r.get("abbrev") or r.get("abbr") or ""),
        }
        for r in league_teams
    ]
    return league_teams, teams_directory



# --- Post-split public API (offseason + api_bridge) ---

def continue_franchise_offseason(session, *, from_stage: str | None = None):
    from services.franchise_offseason import continue_offseason
    return continue_offseason(session, from_stage=from_stage)


def reopen_franchise_offseason_stage(session, stage: str):
    from services.franchise_offseason import reopen_offseason_stage
    return reopen_offseason_stage(session, stage)


def generate_franchise_next_season(session):
    from services.franchise_offseason import generate_next_season
    return generate_next_season(session)


def advance_season_phase(session, target=None):
    from services.franchise_offseason import advance_season_phase as _fn
    return _fn(session, target)


def get_franchise_chemistry_report(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.systems.chemistry import build_public_chemistry_report

    return build_public_chemistry_report(session)


def enter_franchise_playoffs(session: FranchiseSession) -> Dict[str, Any]:
    """Start interactive live playoffs (bracket hub). Does not instantly crown a Cup winner."""
    from services.franchise_playoffs import handle_playoff_action, get_playoff_hub_payload

    # If already finished, stay idempotent.
    if session.playoffs_simulated:
        from services.franchise_offseason import complete_playoffs

        return complete_playoffs(session)

    result = handle_playoff_action(session, "enter")
    return {
        "status": "playoffs",
        "season_phase": "playoffs",
        "playoff": get_playoff_hub_payload(session),
        "live": result.get("live"),
    }


def get_cached_trade_assets_payload(session: FranchiseSession, *, force: bool = False) -> Dict[str, Any]:
    """Return trade assets; rebuild when formula version changes (no new save needed)."""
    from services.trade_service import TRADE_ASSETS_CACHE_VERSION, build_trade_assets_payload

    cached = getattr(session, "_cached_trade_assets_payload", None)
    cached_ver = None
    if isinstance(cached, dict):
        cached_ver = cached.get("formula_version")
    session_ver = getattr(session, "_trade_value_formula_version", None)

    # SimEngine edits are outside uvicorn's default watch path — reload valuation
    # module when our watched cache version advances so existing saves pick up the curve.
    need_rebuild = (
        force
        or not isinstance(cached, dict)
        or not cached
        or int(cached_ver or -1) != int(TRADE_ASSETS_CACHE_VERSION)
        or int(session_ver or -1) != int(TRADE_ASSETS_CACHE_VERSION)
    )
    if need_rebuild:
        try:
            import importlib

            from app.sim_engine.trades import trade_value as tv_mod

            importlib.reload(tv_mod)
        except Exception:
            pass
        payload = build_trade_assets_payload(session)
        if not isinstance(payload, dict):
            payload = {"teams": {}}
        payload["formula_version"] = int(TRADE_ASSETS_CACHE_VERSION)
        session._cached_trade_assets_payload = payload
        session._trade_value_formula_version = int(TRADE_ASSETS_CACHE_VERSION)
        return payload
    return cached


def _contract_clause_label(c: Any) -> str:
    if c is None:
        return "None"
    if getattr(c, "no_move_clause", False):
        return "NMC"
    if getattr(c, "no_trade_clause", False):
        return "NTC"
    if int(getattr(c, "modified_no_trade_teams", 0) or 0) > 0:
        return "M-NTC"
    return "None"


def _build_contract_ledger_row(p: Any, session: FranchiseSession, season_year: int) -> Dict[str, Any]:
    c = getattr(p, "contract", None)
    aav = _player_cap_hit_millions(p)
    yrs = _contract_years_remaining(p)
    age = _player_age_int(p)
    ovr = round(_player_ovr99(p))
    rights = str(getattr(c, "rights_status", "") or ("RFA" if age < 27 else "UFA")).upper()
    expiry_year = int(getattr(c, "expiry_year", 0) or (season_year + yrs))

    tags: List[str] = []
    if yrs <= 1:
        tags.append("Expiring")
    # Bad contract = meaningful overpay vs a two-slope fair-value curve
    # (mid-class slope + elite premium), with age/term risk. A $10M elite
    # is fine; a declining veteran on long expensive term is not.
    fair_aav = 1.0 + max(0.0, ovr - 58.0) * 0.16 + max(0.0, ovr - 82.0) * 0.45
    if age >= 35:
        fair_aav *= 0.72
    elif age >= 32:
        fair_aav *= 0.84
    elif age <= 24 and ovr >= 78:
        fair_aav *= 1.10
    term_risk = 1.0
    if yrs >= 5 and age >= 30:
        term_risk = 1.22
    elif yrs >= 4 and age >= 32:
        term_risk = 1.32
    elif yrs <= 1:
        term_risk = 0.92
    if aav >= 2.0 and (aav - fair_aav) >= 1.25 and (aav / max(0.75, fair_aav)) * term_risk >= 1.35:
        tags.append("Bad Contract")
    if 0 < aav < 1.5 and ovr >= 74:
        tags.append("Cheap Deal")
    if str(getattr(c, "contract_type", "") or "").upper() == "ELC":
        tags.append("ELC")

    pid = str(getattr(p, "id", "") or "")
    return {
        "playerId": pid,
        "player_id": pid,
        "id": pid,
        "name": _name_str(p),
        "position": _pos_str(p),
        "age": age,
        "overall": ovr,
        "ovr": ovr,
        "aav": round(aav, 3),
        "capHit": round(aav, 3),
        "cap_hit": round(aav, 3),
        "yearsRemaining": yrs,
        "years_remaining": yrs,
        "expiryYear": expiry_year,
        "expiry_year": expiry_year,
        "expiryStatus": rights,
        "expiry_status": rights,
        "contractType": str(getattr(c, "contract_type", "STANDARD") or "STANDARD"),
        "clauseLabel": _contract_clause_label(c),
        "tags": tags,
        "contract": {
            "cap_hit": round(aav, 3),
            "aav": round(aav, 3),
            "years_remaining": yrs,
            "expiry_year": expiry_year,
            "ntc": bool(getattr(c, "no_trade_clause", False)),
            "nmc": bool(getattr(c, "no_move_clause", False)),
            "two_way": bool(getattr(c, "two_way", False)),
        },
    }


def _free_agent_asking_terms(ovr: float, age: int, rng: random.Random, position: str = "C") -> Tuple[float, int]:
    """Asking price scales off what a team would pay, plus a market premium."""
    aav, years = _generate_contract_terms(ovr, age, position or "C", rng)
    ask = max(LEAGUE_MINIMUM_AAV_M, round(aav * rng.uniform(1.0, 1.18), 3))
    return ask, years


# ---------------------------------------------------------------------------
# Free-agent in-season lifecycle: active assignments, deterministic season
# statistics, and market stock. Values persist on the player object and only
# change at simulation checkpoints (see _recompute_free_agent_stock, called from
# _depth_pool_progression_tick) — never regenerated per API request.
# ---------------------------------------------------------------------------

# Relative quality vs NHL (1.0) and scoring environment per external league.
_FA_LEAGUE_INFO: Dict[str, Dict[str, Any]] = {
    "KHL": {"level": "pro_euro", "strength": 0.82, "gp": 62},
    "SHL": {"level": "pro_euro", "strength": 0.78, "gp": 52},
    "Liiga": {"level": "pro_euro", "strength": 0.74, "gp": 60},
    "NL": {"level": "pro_euro", "strength": 0.72, "gp": 50},
    "DEL": {"level": "pro_euro", "strength": 0.68, "gp": 52},
    "Czech Extraliga": {"level": "pro_euro", "strength": 0.70, "gp": 52},
    "AHL": {"level": "minor_pro", "strength": 0.75, "gp": 72},
}
_FA_DEFAULT_LEAGUE = {"level": "minor_pro", "strength": 0.70, "gp": 62}


def _fa_season_progress(session: Any) -> float:
    """Fraction of the current league year elapsed (0..1), used to scale season totals."""
    last = int(getattr(session, "nhl_regular_season_last_index", 192) or 192)
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    if last <= 0:
        return 1.0
    return max(0.0, min(1.0, cur / float(last)))


def _fa_seed(pid: str, season_year: int, salt: str = "") -> int:
    return abs(hash(f"fa|{salt}|{pid}|{season_year}")) & 0xFFFFFFFF


def _fa_current_assignment(p: Any, season_year: int) -> Dict[str, Any]:
    """Resolve the live competition (league/team/level) an unsigned player is featuring in.

    Reads the existing _franchise_assignment meta first; only falls back to a deterministic
    pick among leagues that already exist in the backend (never invents league names)."""
    meta = getattr(p, "_franchise_assignment", None) or {}
    pid = str(getattr(p, "id", "") or "")
    ovr = round(_player_ovr99(p))
    rng = random.Random(_fa_seed(pid, season_year, "league"))
    code = str(meta.get("overseas_league") or "").strip()
    if code not in _FA_LEAGUE_INFO:
        if meta.get("overseas"):
            code = rng.choice(["KHL", "SHL", "Liiga", "NL", "DEL", "Czech Extraliga"])
        elif ovr >= 80:
            code = rng.choice(["SHL", "KHL", "Liiga", "NL"])
        elif ovr >= 70:
            code = rng.choice(["AHL", "DEL", "Liiga", "Czech Extraliga"])
        else:
            code = "AHL"
    info = _FA_LEAGUE_INFO.get(code, _FA_DEFAULT_LEAGUE)
    team = meta.get("club") or None
    return {
        "current_league": code,
        "current_team": team,
        "league_level": info["level"],
        "strength": float(info["strength"]),
        "gp_base": int(info["gp"]),
    }


def _fa_role_projection(position: str, ovr: float) -> str:
    pos = str(position or "").upper()
    if pos == "G":
        return "Starter" if ovr >= 80 else ("1B / tandem" if ovr >= 72 else "Backup")
    if pos == "D":
        if ovr >= 82:
            return "Top-pair D"
        if ovr >= 74:
            return "Middle-pair D"
        return "Bottom-pair D"
    if ovr >= 83:
        return "Top-six forward"
    if ovr >= 74:
        return "Middle-six forward"
    return "Bottom-six forward"


def _fa_stat_projection(p: Any, season_year: int, assign: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic full-season stat line from real attributes, role and league strength."""
    pid = str(getattr(p, "id", "") or "")
    pos = _pos_str(p)
    ovr = round(_player_ovr99(p))
    age = _player_age_int(p)
    rng = random.Random(_fa_seed(pid, season_year, "stats"))
    strength = float(assign.get("strength", 0.7))
    gp_base = int(assign.get("gp_base", 62))

    talent = max(0.0, min(1.0, (ovr - 45) / 45.0))
    # A weaker league inflates a given talent's raw production.
    league_boost = 1.0 + (1.0 - strength) * 0.55
    availability = rng.uniform(0.80, 1.0) * (1.0 if age < 33 else 0.9)
    gp = max(1, int(round(gp_base * availability)))

    if pos == "G":
        starts = max(1, int(round(gp * rng.uniform(0.85, 1.0))))
        sv_pct = round(0.885 + talent * 0.045 + rng.uniform(-0.006, 0.006), 3)
        sv_pct = max(0.86, min(0.945, sv_pct))
        gaa = round(max(1.6, 3.9 - talent * 1.9 + rng.uniform(-0.2, 0.25)), 2)
        wins = int(round(starts * (0.40 + talent * 0.25) * rng.uniform(0.85, 1.1)))
        wins = max(0, min(starts, wins))
        shutouts = int(round((talent * 6.0) * rng.uniform(0.5, 1.1)))
        return {
            "is_goalie": True,
            "gp": gp,
            "starts": starts,
            "wins": wins,
            "save_pct": sv_pct,
            "gaa": gaa,
            "shutouts": max(0, shutouts),
        }

    base_ppg = 0.12 + talent * 1.02
    ppg = base_ppg * league_boost * rng.uniform(0.88, 1.14)
    pts = int(round(ppg * gp))
    goal_share = 0.30 if pos == "D" else 0.42
    goals = int(round(pts * goal_share * rng.uniform(0.9, 1.12)))
    goals = max(0, min(pts, goals))
    assists = max(0, pts - goals)
    shots = int(round(goals / 0.108)) if goals else int(round(gp * (0.6 + talent)))
    toi = round((11.0 + talent * 10.0) + (1.5 if pos == "D" else 0.0), 1)
    plus_minus = int(round((talent - 0.5) * 20 * rng.uniform(0.6, 1.2)))
    return {
        "is_goalie": False,
        "gp": gp,
        "goals": goals,
        "assists": assists,
        "points": pts,
        "ppg": round(pts / gp, 2) if gp else 0.0,
        "shots": max(0, shots),
        "toi": toi,
        "plus_minus": plus_minus,
    }


def _scale_stat_line(full: Dict[str, Any], gp_full: int, gp_now: int) -> Dict[str, Any]:
    """Scale full-season counting totals down to games played so far (rates unchanged)."""
    if gp_full <= 0:
        return dict(full)
    ratio = max(0.0, min(1.0, gp_now / float(gp_full)))
    counting = {"gp", "starts", "wins", "shutouts", "goals", "assists", "points", "shots"}
    out: Dict[str, Any] = {}
    for k, v in full.items():
        if k in counting and isinstance(v, (int, float)):
            out[k] = int(round(v * ratio))
        else:
            out[k] = v
    if not full.get("is_goalie"):
        gp = out.get("gp", 0)
        out["ppg"] = round(out.get("points", 0) / gp, 2) if gp else 0.0
    return out


def _ensure_free_agent_ledger(p: Any, session: Any, *, persist: bool = True) -> Dict[str, Any]:
    """Return a per-season stat projection for the player; roll last year into 'previous'.

    Deterministic: the same season year always yields the same full-season projection.
    When ``persist`` is False (read/render path) the computed ledger is NOT written back to
    the player — screen requests never mutate simulation state. The checkpoint tick calls it
    with ``persist=True`` so the previous-season rollover is recorded once per season."""
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    ledger = getattr(p, "_fa_season_ledger", None)
    if isinstance(ledger, dict) and int(ledger.get("season") or 0) == season_year:
        return ledger
    prev = None
    if isinstance(ledger, dict) and ledger.get("full"):
        prev = {
            "season": ledger.get("season"),
            "league": ledger.get("current_league"),
            "line": ledger.get("full"),
        }
    assign = _fa_current_assignment(p, season_year)
    full = _fa_stat_projection(p, season_year, assign)
    new_ledger = {
        "season": season_year,
        "current_league": assign["current_league"],
        "current_team": assign["current_team"],
        "league_level": assign["league_level"],
        "gp_full": int(full.get("gp", 0) or 0),
        "full": full,
        "previous": prev if prev is not None else (ledger or {}).get("previous"),
    }
    if persist:
        setattr(p, "_fa_season_ledger", new_ledger)
    return new_ledger


def _free_agent_stat_view(p: Any, session: Any) -> Dict[str, Any]:
    """Current (games-so-far) and previous-season stat lines for a free agent (read-only)."""
    ledger = _ensure_free_agent_ledger(p, session, persist=False)
    gp_full = int(ledger.get("gp_full", 0) or 0)
    progress = _fa_season_progress(session)
    gp_now = max(0, int(round(gp_full * progress)))
    current = _scale_stat_line(ledger.get("full") or {}, gp_full, gp_now)
    prev = ledger.get("previous") or None
    prev_line = (prev or {}).get("line") if isinstance(prev, dict) else None
    return {
        "current_league": ledger.get("current_league"),
        "current_team": ledger.get("current_team"),
        "league_level": ledger.get("league_level"),
        "season_stats": current,
        "previous_season_stats": prev_line,
        "previous_season_league": (prev or {}).get("league") if isinstance(prev, dict) else None,
    }


def _fa_performance_score(stat_line: Dict[str, Any], ovr: float, strength: float) -> float:
    """0..1 score of production relative to what this OVR is expected to post in this league."""
    talent = max(0.05, min(1.0, (ovr - 45) / 45.0))
    if stat_line.get("is_goalie"):
        sv = float(stat_line.get("save_pct", 0.9) or 0.9)
        expected = 0.885 + talent * 0.045
        return max(0.0, min(1.0, 0.5 + (sv - expected) * 18.0))
    ppg = float(stat_line.get("ppg", 0.0) or 0.0)
    league_boost = 1.0 + (1.0 - strength) * 0.55
    expected = (0.12 + talent * 1.02) * league_boost
    if expected <= 0:
        return 0.5
    return max(0.0, min(1.0, 0.5 * (ppg / expected)))


def _fa_leverage(ovr: float, age: int) -> float:
    """0..1 negotiating leverage. Low-OVR players unsigned all year have little pull, so
    their demands sit near the league minimum on short deals; only genuine NHL talent
    (higher OVR) commands a real premium."""
    lev = (ovr - 66.0) / 20.0  # ovr<=66 -> 0, ovr>=86 -> 1
    lev = max(0.0, min(1.0, lev))
    if age >= 33:
        lev *= 0.8  # aging players have even less leverage
    return lev


def _recompute_free_agent_stock(p: Any, session: Any, *, persist: bool = True) -> Dict[str, Any]:
    """Compute market stock from the latest OVR + season production.

    Persisted (and compared to the prior snapshot) only at simulation checkpoints
    (depth-pool progression tick). On the read/render path ``persist`` is False, so the
    screen never mutates state — a transient, deterministic stock view is returned."""
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    pid = str(getattr(p, "id", "") or "")
    ovr = round(_player_ovr99(p))
    age = _player_age_int(p)
    pos = _pos_str(p)
    ledger = _ensure_free_agent_ledger(p, session, persist=persist)
    strength = float(_FA_LEAGUE_INFO.get(ledger.get("current_league"), _FA_DEFAULT_LEAGUE)["strength"])
    full = ledger.get("full") or {}
    perf = _fa_performance_score(full, ovr, strength)

    prior = getattr(p, "_fa_stock", None)
    prior = prior if isinstance(prior, dict) else {}
    prev_ovr = float(prior.get("_ovr_snapshot", ovr))
    dev_delta = round(ovr - prev_ovr, 1)

    market_value = float(LEAGUE_MINIMUM_AAV_M)
    try:
        from services.contract_economy import compute_market_value

        market_value = float(compute_market_value(p, getattr(getattr(session, "sim", None), "league", None)))
    except Exception:
        market_value = round(max(LEAGUE_MINIMUM_AAV_M, (ovr - 58) * 0.14) * (0.85 + perf * 0.5), 3)
    market_value = max(LEAGUE_MINIMUM_AAV_M, round(market_value, 3))
    prev_market = float(prior.get("current_market_value", market_value))

    # Leverage compresses the ask toward the league minimum for low-OVR / no-leverage FAs
    # and shortens their term (little bargaining power → short, cheap deals).
    lev = _fa_leverage(float(ovr), age)
    ask_rng = random.Random(_fa_seed(pid, season_year, "ask"))
    base_ask, base_term = _free_agent_asking_terms(float(ovr), age, ask_rng, pos)
    # Anchor to live market value so depth FAs ask near the minimum, not stale $3M.
    anchor = max(LEAGUE_MINIMUM_AAV_M, min(float(base_ask), market_value * 1.08))
    premium = anchor * (0.9 + perf * 0.35) - LEAGUE_MINIMUM_AAV_M
    asking = round(LEAGUE_MINIMUM_AAV_M + max(0.0, premium) * (0.2 + 0.8 * lev), 3)
    if ovr < 75:
        asking = min(asking, LEAGUE_MINIMUM_AAV_M + 0.95 + max(0.0, ovr - 65.0) * 0.08)
    asking = max(LEAGUE_MINIMUM_AAV_M, asking)
    term = max(1, int(round(1 + lev * (base_term - 1))))
    prev_ask = float(prior.get("asking_aav", asking))

    interest = int(round(2 + perf * 6 + max(0.0, (ovr - 78)) * 0.4))
    if pos == "G":
        interest = max(1, interest - 1)

    change = round(market_value - prev_market, 3)
    breakout = bool(dev_delta >= 2.5 and perf >= 0.72 and age <= 26)
    if breakout:
        direction = "breakout"
    elif change > 0.15 or dev_delta >= 1.0:
        direction = "rising"
    elif change < -0.15 or dev_delta <= -1.0:
        direction = "falling"
    else:
        direction = "stable"

    reason = _fa_stock_reason(direction, ledger, full, perf, dev_delta, age, pos, interest)

    stock = {
        "stock_direction": direction,
        "stock_change": change,
        "stock_reason": reason,
        "previous_market_value": round(prev_market, 3),
        "current_market_value": market_value,
        "previous_asking_aav": round(prev_ask, 3),
        "asking_aav": asking,
        "asking_term": term,
        "development_delta": dev_delta,
        "season_performance_score": round(perf, 3),
        "market_interest_count": max(0, interest),
        "_ovr_snapshot": ovr,
    }
    if persist:
        setattr(p, "_fa_stock", stock)
    return stock


def _fa_stock_reason(
    direction: str,
    ledger: Dict[str, Any],
    line: Dict[str, Any],
    perf: float,
    dev_delta: float,
    age: int,
    pos: str,
    interest: int,
) -> str:
    league = str(ledger.get("current_league") or "").strip()
    if direction == "breakout":
        return f"Breakout year in {league}" if league else "Breakout season"
    if direction == "falling":
        if age >= 33:
            return "Age-related decline"
        if line.get("is_goalie"):
            return "Weak goalie results"
        if perf < 0.4:
            return "Production down this year"
        return "Reduced role"
    if direction == "rising":
        if line.get("is_goalie"):
            return "Goalie workload increased"
        if perf >= 0.7:
            return f"Top-six production in {league}" if league else "Strong production"
        if dev_delta >= 1.0:
            return "Attributes trending up"
        return "Market interest rising" if interest >= 5 else "Stock rising"
    # stable
    if age >= 33:
        return "Veteran holding steady"
    return "Development steady"


def _free_agent_stock_view(p: Any, session: Any) -> Dict[str, Any]:
    """Live ask from current valuation; direction still compares to persisted prior."""
    stock = _recompute_free_agent_stock(p, session, persist=False)
    out = {k: v for k, v in stock.items() if not k.startswith("_")}
    return out


def _fa_list_stat(season_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal stat block for the list view (GP + primary column only)."""
    s = season_stats or {}
    if s.get("is_goalie"):
        return {"is_goalie": True, "gp": s.get("gp", 0), "save_pct": s.get("save_pct")}
    return {"is_goalie": False, "gp": s.get("gp", 0), "points": s.get("points", 0)}


def _build_free_agent_row(p: Any, season_year: int, session: Any = None, *, detail: bool = False) -> Dict[str, Any]:
    """Build a free-agent row.

    detail=False (default, used for the 320-row list) returns only the columns the list
    renders — keeping the contract-office payload small. detail=True returns the full
    projected stat lines, previous season and every stock field for the detail panel."""
    pid = str(getattr(p, "id", "") or "")
    age = _player_age_int(p)
    ovr = round(_player_ovr99(p))
    pos = _pos_str(p)
    ident = getattr(p, "identity", None)
    meta = getattr(p, "_franchise_assignment", None) or {}
    pot = getattr(p, "ratings", None) or {}
    potential = int(float(pot.get("dev_potential", 0) or 0)) or None

    row: Dict[str, Any] = {
        "id": pid,
        "playerId": pid,
        "name": _name_str(p),
        "age": age,
        "position": pos,
        "overall": ovr,
        "ovr": ovr,
        "potential": potential,
        "nationality": str(getattr(ident, "birth_country", "") or ""),
        "status": "UFA",
        "ufaOrRfa": "UFA",
        "expiry_status": "UFA",
        "role": _fa_role_projection(pos, ovr),
        "availability_to_sign": True,
        "nhl_transfer_eligible": True,
    }
    try:
        from app.sim_engine.generation.player_headshots import merge_headshot_into_row

        row = merge_headshot_into_row(row, p)
    except Exception:
        pass

    if session is None:
        # Legacy path (no session): deterministic asking terms only.
        rng = random.Random(abs(hash(f"fa-ask|{pid}|{season_year}")) & 0xFFFFFFFF)
        ask, term = _free_agent_asking_terms(float(ovr), age, rng, pos)
        row.update({
            "previous_team": str(meta.get("club") or meta.get("overseas_league") or "Unsigned"),
            "askingAav": ask, "asking_aav": ask, "askingTerm": term, "asking_term": term,
        })
        return row

    stats = _free_agent_stat_view(p, session)
    stock = _free_agent_stock_view(p, session)
    cur_team = stats.get("current_team") or meta.get("club")
    cur_league = stats.get("current_league")
    ask = float(stock.get("asking_aav") or LEAGUE_MINIMUM_AAV_M)
    term = int(stock.get("asking_term") or 1)
    season_stats = stats.get("season_stats") or {}

    if not detail:
        # Slim list row: identity + columns + trend/ask only.
        row.update({
            "current_league": cur_league,
            "current_team": cur_team,
            "previous_team": str(cur_team or cur_league or "Unsigned"),
            "games_played": season_stats.get("gp", 0),
            "season_stats": _fa_list_stat(season_stats),
            "stat_projected": True,
            "askingAav": ask, "asking_aav": ask, "askingTerm": term, "asking_term": term,
            "stock_direction": stock.get("stock_direction"),
            "stock_change": stock.get("stock_change"),
            "player_id": pid,
        })
        return row

    # Full detail row for the detail panel.
    row.update({
        "current_league": cur_league,
        "current_team": cur_team,
        "league_level": stats.get("league_level"),
        "season": season_year,
        "stat_scope": "season_to_date",
        "stat_projected": True,
        "stat_source": "projected",
        "last_game_date": _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0)),
        "season_stats": season_stats,
        "previous_season_stats": stats.get("previous_season_stats"),
        "previous_season_league": stats.get("previous_season_league"),
        "games_played": season_stats.get("gp", 0),
        "previous_team": str(cur_team or cur_league or "Unsigned"),
        "askingAav": ask, "asking_aav": ask, "expected_aav": ask,
        "askingTerm": term, "asking_term": term,
        "tags": (["Veteran"] if age >= 32 else []) + (["Overseas"] if meta.get("overseas") else []),
        "fit": "Depth add" if ovr < 76 else ("Roster upgrade" if ovr < 84 else "Impact signing"),
        "risk": _fa_risk_summary(age, stock, season_stats),
        **stock,
    })
    return row


def _fa_risk_summary(age: int, stock: Dict[str, Any], line: Dict[str, Any]) -> str:
    if age >= 34:
        return "Age — short-term only"
    if stock.get("stock_direction") == "falling":
        return "Trending down"
    perf = float(stock.get("season_performance_score") or 0.5)
    if perf < 0.4:
        return "Underperforming role"
    if age >= 30:
        return "Medium"
    return "Low"


def get_contract_office(session: FranchiseSession) -> Dict[str, Any]:
    """Full contract ledger + cap snapshot for the Cap Ledger UI."""
    from services.contract_economy import build_contract_office
    return build_contract_office(session)


def execute_franchise_draft_pick(
    session: FranchiseSession,
    *,
    player_id: str,
    drafting_team_id: Optional[str] = None,
    pick_round: int = 1,
    pick_overall: int = 1,
    pick_number: Optional[int] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one franchise Entry Draft pick.

    Live source of truth: backend/services/franchise_entry_draft.py.
    SimEngine/app/sim_engine/franchise/ mirror is not used for live Entry Draft API.
    """
    from services.franchise_entry_draft import execute_user_draft_pick

    _ = pick_round, pick_overall, pick_number, drafting_team_id
    return execute_user_draft_pick(session, str(player_id), request_id=request_id)


def resolve_player_meeting(session: FranchiseSession, interaction_id: str, choice_id: str) -> Dict[str, Any]:
    """Resolve player-initiated universe meeting or storyline interaction."""
    from app.sim_engine.franchise.storyline_engine import resolve_player_meeting_interaction

    result = resolve_player_meeting_interaction(session, str(interaction_id), str(choice_id))
    invalidate_session_payload_caches(session, "player_meeting_resolve")
    return result


def start_player_meeting(session: FranchiseSession, player_id: str, interaction_type: str) -> Dict[str, Any]:
    """GM-initiated player meeting."""
    from app.sim_engine.franchise.storyline_engine import start_gm_player_meeting

    result = start_gm_player_meeting(session, str(player_id), str(interaction_type))
    invalidate_session_payload_caches(session, "player_meeting_start")
    return result


def advance_player_meeting(session: FranchiseSession, meeting_id: str, choice_id: str) -> Dict[str, Any]:
    """Resolve an in-progress GM-initiated meeting."""
    from app.sim_engine.franchise.storyline_engine import resolve_gm_player_meeting

    result = resolve_gm_player_meeting(session, str(meeting_id), str(choice_id))
    invalidate_session_payload_caches(session, "player_meeting_advance")
    return result


def get_player_meeting_detail_payload(session: FranchiseSession, player_id: str) -> Dict[str, Any]:
    from app.sim_engine.franchise.storyline_engine import get_player_meeting_detail

    return get_player_meeting_detail(session, str(player_id))
