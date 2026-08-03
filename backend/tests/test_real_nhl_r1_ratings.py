"""Unit tests for Real NHL R2 usage-aware rating helpers (no live network)."""

from __future__ import annotations

from services.real_nhl_roster_importer import (
    infer_skater_profile,
    target_ovr_from_goalie_stats,
    target_ovr_from_skater_stats,
)


def test_elite_forward_rates_as_star():
    ovr, note = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 78,
            "points": 121,
            "goals": 37,
            "assists": 84,
            "evPoints": 75,
            "ppPoints": 46,
            "pointsPerGame": 1.55,
            "timeOnIcePerGame": 1271,
            "shootingPct": 0.14,
            "team_toi_rank_pct": 0.95,
        },
        position_code="R",
        age=30,
    )
    assert ovr >= 0.88
    assert note.startswith("r2_")


def test_top_pair_low_scoring_d_beats_pp_specialist():
    shutdown, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 82,
            "points": 28,
            "goals": 5,
            "assists": 23,
            "evPoints": 24,
            "ppPoints": 4,
            "pointsPerGame": 0.34,
            "timeOnIcePerGame": 1500,
            "hits": 120,
            "blockedShots": 180,
            "team_toi_rank_pct": 0.92,
        },
        position_code="D",
        age=28,
    )
    pp_only, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 75,
            "points": 40,
            "goals": 10,
            "assists": 30,
            "evPoints": 12,
            "ppPoints": 28,
            "pointsPerGame": 0.53,
            "timeOnIcePerGame": 780,
            "hits": 20,
            "blockedShots": 30,
            "team_toi_rank_pct": 0.35,
        },
        position_code="D",
        age=27,
    )
    assert shutdown >= pp_only - 0.02


def test_depth_forward_rates_below_star():
    ovr, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 70,
            "points": 18,
            "goals": 6,
            "assists": 12,
            "evPoints": 16,
            "ppPoints": 2,
            "pointsPerGame": 0.26,
            "timeOnIcePerGame": 720,
            "shootingPct": 0.09,
            "team_toi_rank_pct": 0.20,
        },
        position_code="C",
        age=27,
    )
    assert 0.60 <= ovr <= 0.78


def test_infer_profiles():
    assert (
        infer_skater_profile(
            {
                "gamesPlayed": 80,
                "goals": 50,
                "assists": 30,
                "points": 80,
                "ppPoints": 20,
                "shootingPct": 0.16,
                "hits": 40,
            },
            position_code="L",
        )
        == "sniper"
    )
    assert (
        infer_skater_profile(
            {
                "gamesPlayed": 82,
                "goals": 8,
                "assists": 55,
                "points": 63,
                "ppPoints": 30,
                "hits": 40,
                "blockedShots": 50,
            },
            position_code="D",
        )
        == "offensive_d"
    )


def test_elite_goalie_rates_high():
    ovr, note = target_ovr_from_goalie_stats(
        {
            "gamesPlayed": 63,
            "gamesStarted": 62,
            "savePct": 0.925,
            "goalsAgainstAverage": 2.00,
            "wins": 47,
            "shotsAgainst": 1664,
            "team_start_rank_pct": 1.0,
        },
        age=31,
    )
    assert ovr >= 0.86
    assert note.startswith("r2_g_")


def test_backup_goalie_rates_lower():
    ovr, _ = target_ovr_from_goalie_stats(
        {
            "gamesPlayed": 18,
            "gamesStarted": 12,
            "savePct": 0.898,
            "goalsAgainstAverage": 3.15,
            "wins": 5,
            "shotsAgainst": 480,
            "team_start_rank_pct": 0.0,
        },
        age=28,
    )
    assert ovr < 0.82
