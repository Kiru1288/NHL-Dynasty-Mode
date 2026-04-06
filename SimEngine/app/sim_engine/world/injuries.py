# app/sim_engine/world/injuries.py
"""Contextual injuries: fatigue, durability, workload, league chaos; tiers + games missed."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.world import durability as world_durability
from app.sim_engine.world import fatigue as world_fatigue

GAMES_KEY = "_world_injury_games_remaining"
TIER_KEY = "_world_injury_tier"
EVENTS_KEY = "_world_injury_event_count"


def _healthy_enough(player: Any) -> bool:
    gl = int(getattr(player, GAMES_KEY, 0) or 0)
    if gl > 0:
        return False
    h = getattr(player, "health", None)
    if h is not None:
        st = getattr(h, "injury_status", None)
        if st is not None:
            name = getattr(st, "name", str(st))
            if name != "HEALTHY":
                return False
    return True


def workload_proxy(player: Any) -> float:
    gp = int(getattr(player, "games_played", 0) or 0)
    return min(1.0, gp / 82.0)


def injury_roll_weight(player: Any, chaos_index: float) -> float:
    world_durability.init_player_durability(player)
    f = world_fatigue.get_fatigue(player)
    d = world_durability.get_durability(player)
    w = workload_proxy(player)
    chaos = max(0.15, min(0.95, float(chaos_index)))
    # High fatigue / workload / chaos up; durability down
    score = (
        0.22 * (f / 100.0)
        + 0.18 * w
        + 0.12 * chaos
        + 0.28 * world_durability.injury_chance_multiplier(player)
        + world_fatigue.injury_risk_from_fatigue(player) * 0.35
    )
    return max(0.04, score)


def _set_player_injured(player: Any, tier: str, games: int) -> None:
    setattr(player, GAMES_KEY, int(max(1, games)))
    setattr(player, TIER_KEY, tier)
    cnt = int(getattr(player, EVENTS_KEY, 0) or 0) + 1
    setattr(player, EVENTS_KEY, cnt)
    h = getattr(player, "health", None)
    if h is not None:
        try:
            from app.sim_engine.entities.player import InjuryStatus

            if tier == "major":
                h.injury_status = InjuryStatus.INJURED
            elif tier == "moderate":
                h.injury_status = InjuryStatus.INJURED
            else:
                h.injury_status = InjuryStatus.DAY_TO_DAY
        except Exception:
            pass
        hist = getattr(h, "injury_history", None)
        if isinstance(hist, list):
            hist.append({"tier": tier, "games": int(games)})


def clear_if_recovered(player: Any) -> None:
    gl = int(getattr(player, GAMES_KEY, 0) or 0)
    if gl > 0:
        return
    setattr(player, TIER_KEY, None)
    h = getattr(player, "health", None)
    if h is not None:
        try:
            from app.sim_engine.entities.player import InjuryStatus

            h.injury_status = InjuryStatus.HEALTHY
        except Exception:
            pass


def tick_games_missed(player: Any) -> None:
    gl = int(getattr(player, GAMES_KEY, 0) or 0)
    if gl <= 0:
        clear_if_recovered(player)
        return
    setattr(player, GAMES_KEY, gl - 1)
    if int(getattr(player, GAMES_KEY, 0) or 0) <= 0:
        clear_if_recovered(player)


def maybe_injure_roster_subset(
    team: Any,
    rng: Any,
    chaos_index: float,
    max_checks: int = 8,
) -> List[Tuple[str, str, int]]:
    """Returns list of (player_label, tier, games) for logging."""
    roster = [
        p
        for p in (getattr(team, "roster", None) or [])
        if not getattr(p, "retired", False) and _healthy_enough(p)
    ]
    if not roster:
        return []
    rng.shuffle(roster)
    out: List[Tuple[str, str, int]] = []
    checks = min(max_checks, len(roster))
    for p in roster[:checks]:
        w = injury_roll_weight(p, chaos_index)
        p_inj = min(0.20, 0.026 + w * 0.055)
        if rng.random() >= p_inj:
            continue
        roll = rng.random()
        if roll < 0.62:
            tier, games = "minor", rng.randint(1, 4)
        elif roll < 0.92:
            tier, games = "moderate", rng.randint(5, 18)
        else:
            tier, games = "major", rng.randint(20, 55)
        games = int(games * (0.75 + 0.35 * world_durability.get_durability(p)))
        games = max(1, games)
        _set_player_injured(p, tier, games)
        label = str(
            getattr(p, "name", None)
            or getattr(getattr(p, "identity", None), "name", None)
            or "Player"
        )
        out.append((label, tier, int(getattr(p, GAMES_KEY, games))))
    return out


def is_world_injured(player: Any) -> bool:
    return int(getattr(player, GAMES_KEY, 0) or 0) > 0
