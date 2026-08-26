"""Bridge franchise storyline / universe modifiers into live engine.py stat allocation."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _player_id(player: Any) -> str:
    return str(getattr(player, "id", "") or "")


def _roster_modifiers(team: Any) -> Dict[str, Dict[str, float]]:
    from app.sim_engine.franchise.storyline_conduct import get_player_stat_allocation_modifiers  # noqa: WPS433

    out: Dict[str, Dict[str, float]] = {}
    for player in getattr(team, "roster", None) or []:
        pid = _player_id(player)
        if not pid:
            continue
        mods = get_player_stat_allocation_modifiers(player)
        if mods:
            out[pid] = mods
    return out


def build_franchise_game_stat_context(
    session: Any,
    home_team_id: str,
    away_team_id: str,
    game_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build per-player stat allocation modifiers for one matchup.

    Prefers narrative-universe V3 when available; otherwise maps active
    storyline conduct readiness modifiers onto stat fingerprints.
    """
    meta = dict(game_meta or {})
    try:
        from app.sim_engine.franchise.storyline_engine import build_universe_matchup_context  # noqa: WPS433

        return build_universe_matchup_context(session, home_team_id, away_team_id, meta)
    except (ImportError, AttributeError):
        pass

    team_by_id = getattr(session, "team_by_id", None) or {}
    home = team_by_id.get(str(home_team_id))
    away = team_by_id.get(str(away_team_id))
    home_mods = _roster_modifiers(home) if home is not None else {}
    away_mods = _roster_modifiers(away) if away is not None else {}
    base_id = str(meta.get("game_id") or f"conduct_{home_team_id}_{away_team_id}_{meta.get('calendar_day', 0)}")
    return {
        "id": base_id,
        "home_team_id": str(home_team_id),
        "away_team_id": str(away_team_id),
        "home": {"player_modifiers": home_mods},
        "away": {"player_modifiers": away_mods},
        "home_win_probability_delta": 0.0,
    }


def prime_franchise_game_stat_modifiers(
    session: Any,
    sim: Any,
    home_team_id: str,
    away_team_id: str,
    *,
    game_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach storyline/universe modifiers to the live SimEngine for one game."""
    context = build_franchise_game_stat_context(session, home_team_id, away_team_id, game_meta)
    home_mods = dict((context.get("home") or {}).get("player_modifiers") or {})
    away_mods = dict((context.get("away") or {}).get("player_modifiers") or {})
    fn = getattr(sim, "set_franchise_game_stat_modifiers", None)
    if callable(fn):
        fn(
            home_player_modifiers=home_mods,
            away_player_modifiers=away_mods,
            home_win_probability_delta=float(context.get("home_win_probability_delta", 0) or 0),
        )
    return context


def clear_franchise_game_stat_modifiers(sim: Any) -> None:
    fn = getattr(sim, "clear_franchise_game_stat_modifiers", None)
    if callable(fn):
        fn()


def apply_franchise_postgame_storyline(
    session: Any,
    team_id: str,
    game_result: Dict[str, Any],
    rng: Any = None,
) -> None:
    """Optional universe postgame tick when V3 helpers exist."""
    try:
        from app.sim_engine.franchise.storyline_engine import apply_universe_postgame  # noqa: WPS433
    except (ImportError, AttributeError):
        return
    try:
        apply_universe_postgame(session, team_id, game_result, rng=rng)
    except Exception:
        return
