# app/sim_engine/world/injuries.py
"""Contextual injuries: fatigue, durability, workload, league chaos; tiers + games missed.

Important:
- This module stores injury duration as TEAM GAMES MISSED, not calendar days.
- tick_games_missed(player) should be called only when that player's team plays
  and the injured player misses that game.
- If tick_games_missed is called every calendar day, long injuries will recover
  way too fast and injured players can still finish with unrealistic GP totals.

Frontend/backend compatibility:
- Keeps GAMES_KEY = "_world_injury_games_remaining"
- Keeps TIER_KEY = "_world_injury_tier"
- Keeps EVENTS_KEY = "_world_injury_event_count"
- Keeps maybe_injure_roster_subset(...) return shape:
  List[Tuple[player_label, tier, games, player_id]]
- Keeps is_world_injured(player)
- Keeps tick_games_missed(player)
- Keeps clear_if_recovered(player)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.world import durability as world_durability
from app.sim_engine.world import fatigue as world_fatigue


# ---------------------------------------------------------------------------
# Public storage keys
# ---------------------------------------------------------------------------

GAMES_KEY = "_world_injury_games_remaining"
TIER_KEY = "_world_injury_tier"
EVENTS_KEY = "_world_injury_event_count"

# Extra internal metadata keys. Safe to expose if the backend wants richer data.
INJURY_LABEL_KEY = "_world_injury_label"
INJURY_SOURCE_KEY = "_world_injury_source"
INJURY_SEVERITY_KEY = "_world_injury_severity"
INJURY_ORIGINAL_GAMES_KEY = "_world_injury_original_games"
INJURY_LAST_LOG_KEY = "_world_injury_last_log"
INJURY_DAY_TO_DAY_KEY = "_world_day_to_day"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _player_label(player: Any) -> str:
    return str(
        getattr(player, "name", None)
        or getattr(getattr(player, "identity", None), "name", None)
        or getattr(player, "full_name", None)
        or getattr(player, "display_name", None)
        or "Player"
    )


def _player_id(player: Any) -> str:
    return str(
        getattr(player, "id", None)
        or getattr(player, "player_id", None)
        or getattr(getattr(player, "identity", None), "id", None)
        or ""
    )


def _get_health(player: Any) -> Any:
    return getattr(player, "health", None)


def _set_health_status(player: Any, tier: Optional[str]) -> None:
    """
    Set the engine's canonical health status if the Player model supports it.

    We do this defensively because some custom player objects may not have the
    same entity classes.
    """
    health = _get_health(player)
    if health is None:
        return

    try:
        from app.sim_engine.entities.player import InjuryStatus

        if tier is None:
            health.injury_status = InjuryStatus.HEALTHY
        elif tier == "minor":
            health.injury_status = InjuryStatus.DAY_TO_DAY
        else:
            health.injury_status = InjuryStatus.INJURED
    except Exception:
        # Fallback for plain objects/enums/strings.
        try:
            if tier is None:
                health.injury_status = "HEALTHY"
            elif tier == "minor":
                health.injury_status = "DAY_TO_DAY"
            else:
                health.injury_status = "INJURED"
        except Exception:
            pass


def _health_status_name(player: Any) -> str:
    health = _get_health(player)
    if health is None:
        return "UNKNOWN"

    status = getattr(health, "injury_status", None)
    if status is None:
        return "UNKNOWN"

    return str(getattr(status, "name", status))


def _append_health_history(player: Any, row: Dict[str, Any]) -> None:
    health = _get_health(player)
    if health is None:
        return

    hist = getattr(health, "injury_history", None)
    if isinstance(hist, list):
        hist.append(dict(row))


def _normalize_tier(tier: Any) -> str:
    t = str(tier or "").strip().lower()

    if t in {"day_to_day", "day-to-day", "dtd", "minor"}:
        return "minor"
    if t in {"moderate", "medium"}:
        return "moderate"
    if t in {"major", "severe", "long_term", "long-term"}:
        return "major"

    return "minor"


def _tier_human_label(tier: Optional[str]) -> str:
    if tier == "major":
        return "Major injury"
    if tier == "moderate":
        return "Moderate injury"
    if tier == "minor":
        return "Day-to-day injury"
    return "Healthy"


def _tier_severity(tier: str) -> int:
    if tier == "major":
        return 3
    if tier == "moderate":
        return 2
    return 1


# ---------------------------------------------------------------------------
# Public injury state helpers
# ---------------------------------------------------------------------------


def injury_games_remaining(player: Any) -> int:
    return max(0, _safe_int(getattr(player, GAMES_KEY, 0), 0))


def injury_tier(player: Any) -> Optional[str]:
    tier = getattr(player, TIER_KEY, None)
    if tier is None:
        return None

    tier = str(tier).strip().lower()
    if not tier or tier == "none":
        return None

    return _normalize_tier(tier)


def injury_event_count(player: Any) -> int:
    return max(0, _safe_int(getattr(player, EVENTS_KEY, 0), 0))


def injury_status_label(player: Any) -> str:
    gl = injury_games_remaining(player)
    tier = injury_tier(player)

    if gl <= 0 or tier is None:
        return "Healthy"

    if tier == "minor":
        if gl <= 1:
            return "Day-to-day"
        return f"Day-to-day · {gl} games"

    if tier == "moderate":
        return f"Out {gl} games"

    return f"Long-term injury · {gl} games"


def is_world_injured(player: Any) -> bool:
    """
    Canonical injury availability check.

    The game sim/lineup builder should use this to exclude players from games.
    """
    return injury_games_remaining(player) > 0


def player_available_for_game(player: Any) -> bool:
    """
    Preferred helper for simulation lineup selection.

    Use this wherever the sim chooses dressed players, scorers, goalies, etc.
    """
    if getattr(player, "retired", False):
        return False

    if is_world_injured(player):
        return False

    health_name = _health_status_name(player)
    if health_name not in {"UNKNOWN", "HEALTHY"}:
        return False

    return True


def _healthy_enough(player: Any) -> bool:
    """
    Internal injury-roll eligibility.

    A player is healthy enough to suffer a new injury only if they are not
    already injured and not already marked injured by the canonical health object.
    """
    if is_world_injured(player):
        return False

    health_name = _health_status_name(player)
    if health_name not in {"UNKNOWN", "HEALTHY"}:
        return False

    return True


def injury_payload_for_player(player: Any) -> Dict[str, Any]:
    """
    Optional helper for backend payloads. Existing code does not have to use this,
    but it gives a stable shape if you want richer injury reports later.
    """
    tier = injury_tier(player)
    games = injury_games_remaining(player)

    return {
        "player_id": _player_id(player),
        "player_name": _player_label(player),
        "is_injured": games > 0,
        "tier": tier,
        "tier_label": _tier_human_label(tier),
        "games_remaining": games,
        "original_games": max(0, _safe_int(getattr(player, INJURY_ORIGINAL_GAMES_KEY, games), games)),
        "event_count": injury_event_count(player),
        "status": injury_status_label(player),
        "health_status": _health_status_name(player),
        "source": getattr(player, INJURY_SOURCE_KEY, None),
        "severity": getattr(player, INJURY_SEVERITY_KEY, None),
    }


# ---------------------------------------------------------------------------
# Risk model
# ---------------------------------------------------------------------------


def workload_proxy(player: Any) -> float:
    """
    Workload proxy used for injury risk.

    This is intentionally conservative. It should influence risk, not dominate it.
    """
    gp = _safe_int(getattr(player, "games_played", 0), 0)

    # Some player stat models store GP in season_stats.
    stats = getattr(player, "season_stats", None)
    if gp <= 0 and stats is not None:
        gp = _safe_int(getattr(stats, "games_played", 0), 0)

    return _clamp(gp / 82.0, 0.0, 1.0)


def _position_risk_modifier(player: Any) -> float:
    """
    Light positional flavor. Goalies and heavy-minute defenders are slightly
    different injury profiles, but do not let position explode risk.
    """
    pos = str(
        getattr(player, "position", None)
        or getattr(player, "pos", None)
        or getattr(getattr(player, "identity", None), "position", None)
        or ""
    ).upper()

    if pos in {"G", "GOALIE"}:
        return 0.92

    if pos in {"D", "LD", "RD", "DEFENSE", "DEFENSEMAN"}:
        return 1.06

    return 1.0


def _age_risk_modifier(player: Any) -> float:
    age = _safe_float(getattr(player, "age", 0), 0.0)

    if age <= 0:
        return 1.0
    if age < 22:
        return 0.94
    if age <= 29:
        return 1.0
    if age <= 33:
        return 1.06
    if age <= 36:
        return 1.12
    return 1.18


def injury_roll_weight(player: Any, chaos_index: float) -> float:
    """
    Returns a contextual injury risk weight.

    Higher means more likely to get hurt.

    Inputs:
    - fatigue
    - workload
    - league chaos
    - durability
    - fatigue-specific injury risk
    - light age/position modifiers
    """
    world_durability.init_player_durability(player)

    fatigue = _safe_float(world_fatigue.get_fatigue(player), 0.0)
    durability = _clamp(world_durability.get_durability(player), 0.0, 1.0)
    workload = workload_proxy(player)
    chaos = _clamp(float(chaos_index), 0.05, 1.0)

    durability_risk = _safe_float(world_durability.injury_chance_multiplier(player), 1.0)
    fatigue_risk = _safe_float(world_fatigue.injury_risk_from_fatigue(player), 0.0)

    score = (
        0.18 * (fatigue / 100.0)
        + 0.14 * workload
        + 0.10 * chaos
        + 0.22 * durability_risk
        + 0.30 * fatigue_risk
        + 0.12 * (1.0 - durability)
    )

    score *= _position_risk_modifier(player)
    score *= _age_risk_modifier(player)

    return _clamp(score, 0.025, 1.25)


def _injury_probability(player: Any, chaos_index: float, low_intensity: bool) -> float:
    """
    Per-player check probability when selected for injury evaluation.

    The previous version could be a little too swingy. This keeps injuries
    meaningful without nuking the league every week.
    """
    weight = injury_roll_weight(player, chaos_index)

    # Normal game-level injury check. This assumes maybe_injure_roster_subset()
    # is called in game/day simulation, not 300 times per day.
    p_inj = 0.010 + weight * 0.026

    if low_intensity:
        p_inj *= 0.35

    return _clamp(p_inj, 0.001, 0.090)


def _roll_tier(player: Any, rng: Any, chaos_index: float) -> Tuple[str, int]:
    """
    Roll injury tier and base games missed.

    Games returned are TEAM GAMES missed.
    """
    fatigue = _safe_float(world_fatigue.get_fatigue(player), 0.0)
    durability = _clamp(world_durability.get_durability(player), 0.0, 1.0)
    chaos = _clamp(float(chaos_index), 0.05, 1.0)

    roll = rng.random()

    # Worse fatigue/chaos slightly pushes injuries up the severity ladder.
    severity_push = (
        0.06 * _clamp(fatigue / 100.0, 0.0, 1.0)
        + 0.05 * chaos
        + 0.05 * (1.0 - durability)
    )

    minor_cut = 0.68 - severity_push
    moderate_cut = 0.93 - severity_push * 0.5

    if roll < minor_cut:
        tier = "minor"
        games = rng.randint(1, 4)
    elif roll < moderate_cut:
        tier = "moderate"
        games = rng.randint(5, 18)
    else:
        tier = "major"
        games = rng.randint(20, 55)

    games = _apply_duration_modifiers(player, tier, games, chaos_index)
    return tier, games


def _apply_duration_modifiers(player: Any, tier: str, base_games: int, chaos_index: float) -> int:
    """
    Apply durability/fatigue/age modifiers to injury length.

    Higher durability shortens injuries.
    Higher fatigue/older age/chaos can lengthen them a bit.

    The key fix from the old model is preserved:
    durability must REDUCE games missed, not increase it.
    """
    world_durability.init_player_durability(player)

    durability = _clamp(world_durability.get_durability(player), 0.0, 1.0)
    fatigue = _clamp(_safe_float(world_fatigue.get_fatigue(player), 0.0) / 100.0, 0.0, 1.0)
    chaos = _clamp(float(chaos_index), 0.05, 1.0)

    age = _safe_float(getattr(player, "age", 0), 0.0)

    # Durability: 0.0 = longer, 1.0 = shorter.
    durability_mult = 1.24 - 0.48 * durability

    # Fatigue and chaos can slightly extend recovery.
    fatigue_mult = 1.00 + 0.18 * fatigue
    chaos_mult = 0.96 + 0.12 * chaos

    age_mult = 1.0
    if age >= 34:
        age_mult = 1.10
    elif age >= 30:
        age_mult = 1.05
    elif 0 < age <= 22:
        age_mult = 0.96

    games = int(round(float(base_games) * durability_mult * fatigue_mult * chaos_mult * age_mult))

    if tier == "minor":
        return max(1, min(games, 6))
    if tier == "moderate":
        return max(4, min(games, 24))
    return max(16, min(games, 70))


# ---------------------------------------------------------------------------
# Injury state mutation
# ---------------------------------------------------------------------------


def _set_player_injured(
    player: Any,
    tier: str,
    games: int,
    *,
    source: str = "game",
    label: Optional[str] = None,
) -> None:
    """
    Apply injury state to a player.

    games = TEAM GAMES missed, not calendar days.
    """
    tier = _normalize_tier(tier)
    games = int(max(1, games))

    setattr(player, GAMES_KEY, games)
    setattr(player, TIER_KEY, tier)
    setattr(player, INJURY_ORIGINAL_GAMES_KEY, games)
    setattr(player, INJURY_SOURCE_KEY, source)
    setattr(player, INJURY_SEVERITY_KEY, _tier_severity(tier))
    setattr(player, INJURY_LABEL_KEY, label or _tier_human_label(tier))
    setattr(player, INJURY_DAY_TO_DAY_KEY, tier == "minor")

    cnt = injury_event_count(player) + 1
    setattr(player, EVENTS_KEY, cnt)

    _set_health_status(player, tier)

    row = {
        "player_id": _player_id(player),
        "player_name": _player_label(player),
        "tier": tier,
        "tier_label": _tier_human_label(tier),
        "games": games,
        "games_remaining": games,
        "source": source,
        "event_count": cnt,
    }

    setattr(player, INJURY_LAST_LOG_KEY, row)
    _append_health_history(player, row)


def force_injury(
    player: Any,
    tier: str,
    games: int,
    *,
    source: str = "manual",
    label: Optional[str] = None,
) -> None:
    """
    Public helper for storylines/admin/dev tools.
    Keeps the same injury state used by random injuries.
    """
    _set_player_injured(player, tier, games, source=source, label=label)


def clear_injury(player: Any) -> None:
    """
    Fully clear world injury state.
    """
    setattr(player, GAMES_KEY, 0)
    setattr(player, TIER_KEY, None)
    setattr(player, INJURY_LABEL_KEY, None)
    setattr(player, INJURY_SOURCE_KEY, None)
    setattr(player, INJURY_SEVERITY_KEY, None)
    setattr(player, INJURY_ORIGINAL_GAMES_KEY, 0)
    setattr(player, INJURY_DAY_TO_DAY_KEY, False)

    _set_health_status(player, None)


def clear_if_recovered(player: Any) -> None:
    """
    Clear status once games remaining reaches zero.
    """
    if injury_games_remaining(player) > 0:
        return

    clear_injury(player)


def tick_games_missed(player: Any) -> None:
    """
    Decrement injury by one TEAM GAME MISSED.

    This should be called when:
    - the player's team plays a game
    - the player is injured
    - the player does NOT appear in that game

    Do NOT call this once per calendar day unless your calendar has exactly one
    team game per day, because that will make injuries recover too fast.
    """
    games_left = injury_games_remaining(player)

    if games_left <= 0:
        clear_if_recovered(player)
        return

    games_left -= 1
    setattr(player, GAMES_KEY, games_left)

    if games_left <= 0:
        clear_if_recovered(player)


def tick_calendar_day(player: Any) -> None:
    """
    Calendar-day tick intentionally does NOT reduce games remaining.

    This exists so code can safely call a day tick without accidentally turning
    a 41-game injury into a 41-calendar-day injury.

    If your sim currently calls tick_games_missed() every day, switch it to this
    for daily maintenance and only call tick_games_missed() after the player's
    team actually plays and the player misses the game.
    """
    if injury_games_remaining(player) <= 0:
        clear_if_recovered(player)


def mark_player_missed_team_game(player: Any) -> None:
    """
    Clear semantic alias for callers.

    Use this after lineup selection confirms the player was unavailable and
    missed an actual team game.
    """
    tick_games_missed(player)


def mark_player_played_game(player: Any) -> None:
    """
    Safety hook.

    If a player somehow plays while still marked injured, do NOT decrement the
    injury counter here. That would hide the bug. The correct fix is to prevent
    injured players from being selected in the first place.
    """
    return


# ---------------------------------------------------------------------------
# Main random injury function
# ---------------------------------------------------------------------------


def maybe_injure_roster_subset(
    team: Any,
    rng: Any,
    chaos_index: float,
    max_checks: int = 8,
    *,
    low_intensity: bool = False,
) -> List[Tuple[str, str, int, str]]:
    """
    Potentially injure a subset of a team's roster.

    Returns:
        List of (player_label, tier, games, player_id)

    This return shape is preserved for existing backend/frontend logging.
    """
    roster = [
        player
        for player in (getattr(team, "roster", None) or [])
        if not getattr(player, "retired", False)
        and _healthy_enough(player)
    ]

    if not roster:
        return []

    rng.shuffle(roster)

    out: List[Tuple[str, str, int, str]] = []

    checks = max(0, min(int(max_checks), len(roster)))

    for player in roster[:checks]:
        p_inj = _injury_probability(player, chaos_index, low_intensity)

        if rng.random() >= p_inj:
            continue

        tier, games = _roll_tier(player, rng, chaos_index)

        _set_player_injured(
            player,
            tier,
            games,
            source="game_low_intensity" if low_intensity else "game",
        )

        out.append(
            (
                _player_label(player),
                tier,
                injury_games_remaining(player),
                _player_id(player),
            )
        )

    return out


# ---------------------------------------------------------------------------
# Roster/team helpers
# ---------------------------------------------------------------------------


def active_roster_players(team: Any) -> List[Any]:
    """
    Players eligible to appear in a game.
    """
    return [
        player
        for player in (getattr(team, "roster", None) or [])
        if player_available_for_game(player)
    ]


def injured_roster_players(team: Any) -> List[Any]:
    """
    Players currently unavailable because of world injury state.
    """
    return [
        player
        for player in (getattr(team, "roster", None) or [])
        if is_world_injured(player)
    ]


def tick_team_injuries_after_game(team: Any, dressed_player_ids: Optional[List[str]] = None) -> None:
    """
    Decrement injuries after a team game.

    This is the safest function for the sim to call after each team game.

    Args:
        team:
            Team object with roster.
        dressed_player_ids:
            Optional IDs of players who actually played. If provided, an injured
            player who somehow appears in dressed_player_ids will NOT have their
            injury decremented, because that means the lineup code allowed an
            injured player to play and should be fixed.

    Behavior:
        - injured players who did not dress lose one game from games_remaining
        - healthy players are cleared/normalized
    """
    dressed = set(str(x) for x in (dressed_player_ids or []))

    for player in (getattr(team, "roster", None) or []):
        if not is_world_injured(player):
            clear_if_recovered(player)
            continue

        pid = _player_id(player)

        if dressed and pid in dressed:
            # Do not hide the lineup bug by ticking down an injury while the
            # player also gets credited with a game played.
            continue

        tick_games_missed(player)


def tick_team_injuries_calendar_day(team: Any) -> None:
    """
    Safe daily maintenance. Does not reduce games remaining.

    Use this for daily league advancement if the team did not play.
    """
    for player in (getattr(team, "roster", None) or []):
        tick_calendar_day(player)


def team_injury_report(team: Any) -> List[Dict[str, Any]]:
    """
    Rich injury report for backend/UI if needed.
    """
    rows = []

    for player in (getattr(team, "roster", None) or []):
        if is_world_injured(player):
            rows.append(injury_payload_for_player(player))

    rows.sort(
        key=lambda row: (
            -int(row.get("severity") or 0),
            -int(row.get("games_remaining") or 0),
            str(row.get("player_name") or ""),
        )
    )

    return rows