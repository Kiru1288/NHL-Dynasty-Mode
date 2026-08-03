"""Forward OVR: scorers above FO/two-way inflation; defensive F not buried."""

from __future__ import annotations

from services.real_nhl_roster_importer import target_ovr_from_skater_stats


def test_stutzle_outrates_hischier_band():
    """Real-ish 2024-25: Stützle (~0.96 PPG) should clear Hischier (~0.92 FO monster)."""
    stutzle, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 82,
            "points": 79,
            "goals": 24,
            "assists": 55,
            "evPoints": 52,
            "ppPoints": 27,
            "pointsPerGame": 0.96,
            "timeOnIcePerGame": 1200,
            "shootingPct": 0.12,
            "faceoffWinPct": 0.478,
            "team_toi_rank_pct": 0.88,
            "hits": 55,
            "blockedShots": 30,
            "shPoints": 0,
            "plusMinus": 5,
        },
        position_code="C",
        age=23,
        analytics={
            "games_played": 82,
            "icetime": 98000,
            "gameScore": 81,
            "onIce_xGoalsPercentage": 0.54,
            "offIce_xGoalsPercentage": 0.48,
            "onIce_corsiPercentage": 0.52,
            "offIce_corsiPercentage": 0.48,
        },
    )
    hischier, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 75,
            "points": 69,
            "goals": 27,
            "assists": 42,
            "evPoints": 48,
            "ppPoints": 21,
            "pointsPerGame": 0.92,
            "timeOnIcePerGame": 1200,
            "shootingPct": 0.14,
            "faceoffWinPct": 0.555,
            "team_toi_rank_pct": 0.92,
            "hits": 60,
            "blockedShots": 55,
            "shPoints": 4,
            "plusMinus": 12,
        },
        position_code="C",
        age=26,
        analytics={
            "games_played": 75,
            "icetime": 90000,
            "gameScore": 82,
            "onIce_xGoalsPercentage": 0.56,
            "offIce_xGoalsPercentage": 0.51,
            "onIce_corsiPercentage": 0.55,
            "offIce_corsiPercentage": 0.50,
        },
    )
    assert stutzle >= 0.885
    assert hischier <= 0.895
    assert stutzle >= hischier


def test_batherson_tier_not_stuck_mid_80s():
    ovr, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 82,
            "points": 68,
            "goals": 26,
            "assists": 42,
            "evPoints": 48,
            "ppPoints": 20,
            "pointsPerGame": 0.83,
            "timeOnIcePerGame": 1080,
            "shootingPct": 0.13,
            "team_toi_rank_pct": 0.72,
            "hits": 40,
            "blockedShots": 20,
        },
        position_code="R",
        age=27,
        analytics={
            "games_played": 82,
            "icetime": 88000,
            "gameScore": 62,
            "onIce_xGoalsPercentage": 0.50,
            "offIce_xGoalsPercentage": 0.50,
            "onIce_corsiPercentage": 0.50,
            "offIce_corsiPercentage": 0.50,
        },
    )
    assert 0.86 <= ovr <= 0.90


def test_defensive_forward_not_buried_at_74():
    ovr, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 78,
            "points": 28,
            "goals": 10,
            "assists": 18,
            "evPoints": 24,
            "ppPoints": 2,
            "pointsPerGame": 0.36,
            "timeOnIcePerGame": 960,
            "shootingPct": 0.10,
            "faceoffWinPct": 0.53,
            "team_toi_rank_pct": 0.45,
            "hits": 120,
            "blockedShots": 70,
            "shPoints": 5,
            "plusMinus": 8,
        },
        position_code="C",
        age=28,
        analytics={
            "games_played": 78,
            "icetime": 75000,
            "gameScore": 35,
            "onIce_xGoalsPercentage": 0.54,
            "offIce_xGoalsPercentage": 0.46,
            "onIce_corsiPercentage": 0.53,
            "offIce_corsiPercentage": 0.46,
        },
    )
    assert 0.78 <= ovr <= 0.845


def test_elite_scorer_still_clears_two_way():
    elite, _ = target_ovr_from_skater_stats(
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
        position_code="C",
        age=28,
    )
    assert elite >= 0.92
