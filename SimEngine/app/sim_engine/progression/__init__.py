# app/sim_engine/progression/__init__.py
"""
Master progression pipeline: development, potential, regression, role, retirement.
"""

from typing import Any, Tuple

from app.sim_engine.progression import aging_curves
from app.sim_engine.progression import development
from app.sim_engine.progression import potential
from app.sim_engine.progression import regression
from app.sim_engine.progression import retirement
from app.sim_engine.progression import role_changes
from app.sim_engine.progression.development import prime_development_environment_for_rosters


def run_player_progression(player: Any, rng: Any) -> Tuple[Any, bool]:
    """
    Run full progression lifecycle in order:
    1. Development (young player growth)
    2. Potential update (breakout/stagnate/bust)
    3. Regression (injury/morale wear; age cliffs are career lifecycle AGING DECLINE)
    4. Role update
    5. Retirement check

    Returns (player, retired). If retired is True, caller should set player.retired and remove from league.
    """
    development.apply_player_development(player, rng)
    potential.update_player_potential(player, rng)
    regression.apply_regression(player, rng)
    role_changes.update_player_role(player)
    retired = retirement.should_player_retire(player, rng)
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
]
