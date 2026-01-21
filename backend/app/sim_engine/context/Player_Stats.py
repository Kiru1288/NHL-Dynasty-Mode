"""
NHL DYNASTY MODE — Player_Stats.py
---------------------------------
Season-level player stat generation & performance evaluation.

RESPONSIBILITY:
- Generate per-season stats for player objects
- Convert performance → rating changes (20–99 scale)
- Store season stats for logging

DOES NOT:
- Handle contracts
- Handle morale
- Handle injuries
- Handle aging (handled in Player.advance_year)
"""

from __future__ import annotations

import random
from typing import Dict, Any

from app.sim_engine.context.League_Stats import LeagueStats
from app.sim_engine.entities.player import (
    OFFENSE_KEYS,
    PASSING_KEYS,
    DEFENSE_KEYS,
    SKATING_KEYS,
    IQ_KEYS,
)

# ============================================================
# CONSTANTS
# ============================================================

RATING_MIN = 20
RATING_MAX = 99


def clamp_rating(x: float) -> int:
    return int(max(RATING_MIN, min(RATING_MAX, round(x))))


# ============================================================
# PLAYER STATS ENGINE
# ============================================================

class PlayerStatsEngine:
    """
    Generates season stats and applies performance-based rating changes.
    """

    def __init__(self, league: LeagueStats, seed: int | None = None) -> None:
        self.league = league
        self.rng = random.Random(seed)

    # --------------------------------------------------------
    # PUBLIC ENTRY
    # --------------------------------------------------------

    def simulate_player_season(self, player: Any, season: int) -> Dict[str, Any]:
        """
        Simulate one season of performance and apply rating deltas.
        """

        role = self._determine_role(player)

        # -------------------------
        # Generate stats
        # -------------------------
        goals = self.league.sample_metric("skater", "goals", tier=role, rng=self.rng)
        assists = self.league.sample_metric("skater", "assists", tier=role, rng=self.rng)
        war = self.league.sample_metric("skater", "war", tier=role, rng=self.rng)
        xgf_pct = self.league.sample_metric("skater", "xgf_pct", tier=role, rng=self.rng)

        points = goals + assists

        # -------------------------
        # Percentiles (0–100)
        # -------------------------
        war_pct = self.league.value_to_percentile("skater", "war", war)
        pts_pct = self.league.value_to_percentile("skater", "points", points)
        xgf_pctile = self.league.value_to_percentile("skater", "xgf_pct", xgf_pct)

        # -------------------------
        # Performance score (0–100)
        # -------------------------
        performance_score = (
            0.45 * war_pct +
            0.35 * pts_pct +
            0.20 * xgf_pctile
        )

        expected = self._expected_performance(player, role)
        delta = performance_score - expected

        # -------------------------
        # Apply rating changes
        # -------------------------
        self._apply_rating_changes(player, delta)

        # -------------------------
        # Store season stats
        # -------------------------
        stat_line = {
            "season": season,
            "role": role,
            "goals": int(goals),
            "assists": int(assists),
            "points": int(points),
            "war": round(war, 2),
            "xgf_pct": round(xgf_pct, 3),
            "performance_score": round(performance_score, 2),
            "expected_score": expected,
            "delta": round(delta, 2),
        }

        if not hasattr(player, "season_stats"):
            player.season_stats = {}

        player.season_stats[season] = stat_line
        return stat_line

    # ========================================================
    # INTERNAL LOGIC
    # ========================================================

    def _determine_role(self, player: Any) -> str:
        """
        Determine usage tier based on OVR (0–100).
        """
        ovr = player.ovr()

        if ovr >= 88:
            return "elite"
        if ovr >= 80:
            return "top_line"
        if ovr >= 72:
            return "middle_six"
        if ovr >= 65:
            return "bottom_six"
        return "depth"

    def _expected_performance(self, player: Any, role: str) -> float:
        """
        Expected performance score by role (0–100).
        """
        base = {
            "elite": 82,
            "top_line": 72,
            "middle_six": 60,
            "bottom_six": 52,
            "depth": 45,
        }.get(role, 55)

        if player.age <= 21:
            base -= 5
        elif player.age >= 32:
            base -= 6

        return base

    def _apply_rating_changes(self, player: Any, delta: float) -> None:
        """
        Apply performance delta to player ratings.
        """

        # Convert performance → rating points
        # ±15 performance ≈ ±3 rating
        rating_delta = delta * 0.20
        rating_delta = max(-4.0, min(4.0, rating_delta))

        # (KEYS, WEIGHT) pairs — NOT A DICT
        groups = [
            (OFFENSE_KEYS, 0.30),
            (PASSING_KEYS, 0.20),
            (SKATING_KEYS, 0.20),
            (DEFENSE_KEYS, 0.20),
            (IQ_KEYS, 0.10),
        ]

        for keys, weight in groups:
            for k in keys:
                player.ratings[k] = clamp_rating(
                    player.ratings[k] + rating_delta * weight
                )
