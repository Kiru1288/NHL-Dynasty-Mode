# app/sim_engine/world/fatigue.py
"""Player fatigue [0,100]: games, B2B, travel; rest decay; performance and injury coupling."""

from __future__ import annotations

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
    # 0 fatigue -> ~1.0 ; 100 -> ~0.94
    return max(0.94, 1.0 - 0.00055 * f)


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


def team_fatigue_strength_factor(team: Any) -> float:
    """Aggregate team drag from fatigue (~0.97–1.0)."""
    avg = team_avg_fatigue(team)
    return max(0.965, 1.0 - 0.00035 * avg)


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


def tick_roster_fatigue_for_game(team: Any, rng: random.Random, b2b_team: bool, schedule: Any, day: int, team_id: str) -> None:
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
        if p not in picked:
            add_game_load(
                p,
                rng,
                back_to_back=False,
                travel_bonus=travel * 0.25,
                ice_time_proxy=0.55,
                durability_fatigue_mult=world_durability.fatigue_buildup_multiplier(p),
            )
        else:
            ice = 0.85 + 0.25 * rng.random()
            add_game_load(
                p,
                rng,
                back_to_back=b2b_team,
                travel_bonus=travel,
                ice_time_proxy=ice,
                durability_fatigue_mult=world_durability.fatigue_buildup_multiplier(p),
            )
