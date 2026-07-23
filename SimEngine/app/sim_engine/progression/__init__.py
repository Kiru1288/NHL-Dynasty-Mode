# app/sim_engine/progression/__init__.py
"""
Master progression pipeline: development, potential, regression, role, retirement.

Seasonal order (one of each per season via development_ledger):
  performance context → developmental attribute changes → breakout/bust drift
  → age progression/decline (caller) → injury effects → attribute clamp → OVR recompute
"""

from typing import Any, Optional, Tuple

from app.sim_engine.progression import aging_curves
from app.sim_engine.progression import development
from app.sim_engine.progression import potential
from app.sim_engine.progression import regression
from app.sim_engine.progression import retirement
from app.sim_engine.progression import role_changes
from app.sim_engine.progression.development import (
    prime_development_environment_for_rosters,
    resolve_development_profile,
)
from app.sim_engine.progression.potential import apply_potential_drift, ensure_development_ledger


def run_player_progression(
    player: Any,
    rng: Any,
    season_id: Optional[Any] = None,
    *,
    source_path: str = "run_player_progression",
) -> Tuple[Any, bool]:
    """
    Run full progression lifecycle in order:
    1. Development (young player growth) — once per season
    2. Potential update (breakout/stagnate/bust) — once per season
    3. Regression (injury/morale wear; age cliffs are career lifecycle AGING DECLINE)
    4. Role update
    5. Retirement check

    Returns (player, retired). If retired is True, caller should set player.retired and remove from league.
    """
    sid = season_id
    if sid is None:
        sid = getattr(player, "_active_dev_season", None)
    if sid is None:
        sid = getattr(player, "last_development_year", None) or "default"
    try:
        setattr(player, "_active_dev_season", sid)
        setattr(player, "_dev_source_path", source_path)
    except Exception:
        pass

    ledger = ensure_development_ledger(player, sid)
    # Ensure profile exists before growth math.
    try:
        resolve_development_profile(player)
    except Exception:
        pass

    development.apply_player_development(player, rng)
    potential.update_player_potential(player, rng)
    regression.apply_regression(player, rng)
    try:
        from app.sim_engine.entities.player import persist_recomputed_ovr

        persist_recomputed_ovr(player)
    except Exception:
        pass
    role_changes.update_player_role(player)
    retired = retirement.should_player_retire(player, rng)
    if ledger.get("source_path") in (None, "", "near_ceiling", "no_ratings", "outside_window"):
        ledger["source_path"] = source_path
    return (player, retired)


__all__ = [
    "run_player_progression",
    "aging_curves",
    "development",
    "potential",
    "regression",
    "retirement",
    "role_changes",
    "prime_development_environment_for_rosters",
    "resolve_development_profile",
    "apply_potential_drift",
    "ensure_development_ledger",
]
