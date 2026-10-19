# app/sim_engine/world/fatigue.py
"""Player fatigue [0,100]: games, B2B, travel; rest decay; performance and injury coupling."""

from __future__ import annotations

import random
from typing import Any

KEY = "_world_fatigue"


def get_fatigue(player: Any) -> float:
    v = getattr(player, KEY, None)
    if v is None:
        setattr(player, KEY, 12.0)
        return 12.0
    return max(0.0, min(100.0, float(v)))


def set_fatigue(player: Any, v: float) -> None:
    setattr(player, KEY, max(0.0, min(100.0, float(v))))


def performance_factor(player: Any) -> float:
    """Slight drag when gassed."""
    f = get_fatigue(player)
    # 0 fatigue -> ~1.0 ; 100 -> ~0.91 (the drop steepens once he's really gassed)
    return max(0.91, 1.0 - 0.00055 * f - 0.0008 * max(0.0, f - 55.0))


def injury_risk_from_fatigue(player: Any) -> float:
    f = get_fatigue(player)
    return min(0.55, 0.08 + 0.0033 * f)


def add_game_load(
    player: Any,
    rng: random.Random,
    *,
    back_to_back: bool,
    travel_bonus: float = 0.0,
    ice_time_proxy: float = 1.0,
    durability_fatigue_mult: float = 1.0,
) -> None:
    from app.sim_engine.world import durability as world_durability

    world_durability.init_player_durability(player)
    base = 2.4 + 1.1 * float(ice_time_proxy)
    if back_to_back:
        base += 5.5
    base += float(travel_bonus) * 18.0
    base *= float(durability_fatigue_mult)
    base *= 0.92 + 0.10 * rng.random()
    set_fatigue(player, get_fatigue(player) + base)


def rest_decay(player: Any, off_days: int, low_usage: bool = False) -> None:
    if off_days <= 0:
        return
    f = get_fatigue(player)
    rec = 3.8 * off_days * (1.25 if low_usage else 1.0)
    set_fatigue(player, max(0.0, f - rec))


def rest_roster(team: Any, off_days: int, rng: random.Random) -> None:
    roster = getattr(team, "roster", None) or []
    for p in roster:
        if getattr(p, "retired", False):
            continue
        if _is_sidelined(p):
            continue
        low = rng.random() < 0.35
        rest_decay(p, off_days, low_usage=low)


def team_avg_fatigue(team: Any) -> float:
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    if not roster:
        return 0.0
    return sum(get_fatigue(p) for p in roster) / len(roster)


def goalie_workload_drag(team: Any) -> float:
    """Drag from riding the current starter: long start streaks and a heavy season load."""
    st = getattr(team, "_gm_goalie_start_state", None)
    if not isinstance(st, dict):
        return 0.0
    recs = [r for k, r in st.items() if k != "_team" and isinstance(r, dict)]
    if not recs:
        return 0.0
    cur = max(recs, key=lambda r: int(r.get("last_day", -1) or -1))
    consec = int(cur.get("consecutive", 0) or 0)
    gp = int(cur.get("gp", 0) or 0)
    b2b = int(cur.get("b2b_starts", 0) or 0)
    drag = max(0, consec - 5) * 0.0035 + max(0, gp - 55) * 0.0006 + max(0, b2b - 6) * 0.0008
    return min(0.035, drag)


def team_fatigue_strength_factor(team: Any) -> float:
    """Aggregate team drag from fatigue and goalie workload (~0.92–1.0)."""
    avg = team_avg_fatigue(team)
    skaters = max(0.95, 1.0 - 0.00035 * avg - 0.0004 * max(0.0, avg - 45.0))
    return skaters * (1.0 - goalie_workload_drag(team))


def _is_sidelined(player: Any) -> bool:
    gl = int(getattr(player, "_world_injury_games_remaining", 0) or 0)
    if gl > 0:
        return True
    h = getattr(player, "health", None)
    if h is not None:
        st = getattr(h, "injury_status", None)
        if st is not None and getattr(st, "name", str(st)) != "HEALTHY":
            return True
    return False


def tick_roster_fatigue_for_game(
    team: Any,
    rng: random.Random,
    b2b_team: bool,
    schedule: Any,
    day: int,
    team_id: str,
    load_mult: float = 1.0,
) -> None:
    """``load_mult`` scales the game's load (coach tactics: fast pace / heavy forecheck > 1)."""
    from app.sim_engine.world import calendar as world_calendar
    from app.sim_engine.world import durability as world_durability

    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    if not roster:
        return
    travel = world_calendar.travel_fatigue_bonus(team_id, day, schedule) if schedule else 0.0
    k = max(6, min(len(roster), 12))
    picked = rng.sample(roster, k=min(k, len(roster)))
    for p in roster:
        if _is_sidelined(p):
            continue
        if getattr(p, "_rest_day", None) == day:
            continue  # coach rest day: no load
        if p not in picked:
            add_game_load(
                p,
                rng,
                back_to_back=False,
                travel_bonus=travel * 0.25,
                ice_time_proxy=0.55,
                durability_fatigue_mult=world_durability.fatigue_buildup_multiplier(p) * float(load_mult),
            )
        else:
            ice = 0.85 + 0.25 * rng.random()
            add_game_load(
                p,
                rng,
                back_to_back=b2b_team,
                travel_bonus=travel,
                ice_time_proxy=ice,
                durability_fatigue_mult=world_durability.fatigue_buildup_multiplier(p) * float(load_mult),
            )
