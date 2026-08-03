"""Tests for MoneyPuck R3 analytics blend (defensive D lift)."""

from __future__ import annotations

from services.real_nhl_analytics import analytics_impact_score, parse_moneypuck_skaters_csv
from services.real_nhl_roster_importer import infer_skater_profile, target_ovr_from_skater_stats


def test_parse_moneypuck_prefers_5on5():
    csv_text = """playerId,season,name,team,position,situation,games_played,icetime,gameScore,onIce_xGoalsPercentage,offIce_xGoalsPercentage,onIce_corsiPercentage,offIce_corsiPercentage,OnIce_F_xGoals,OnIce_A_xGoals,I_F_xGoals
8481606,2025,Jordan Spence,OTT,D,all,73,82019,55.1,0.61,0.50,0.62,0.50,1,1,1
8481606,2025,Jordan Spence,OTT,D,5on5,73,73169,55.1,0.62,0.49,0.60,0.50,1,1,1
"""
    parsed = parse_moneypuck_skaters_csv(csv_text)
    assert 8481606 in parsed
    assert parsed[8481606]["situation"] == "5on5"
    assert parsed[8481606]["onIce_xGoalsPercentage"] == 0.62


def test_spence_like_d_lifts_with_analytics():
    boxcars = {
        "gamesPlayed": 73,
        "points": 28,
        "goals": 6,
        "assists": 22,
        "evPoints": 24,
        "ppPoints": 4,
        "pointsPerGame": 0.38,
        "timeOnIcePerGame": 1120,
        "hits": 40,
        "blockedShots": 90,
        "team_toi_rank_pct": 0.55,
    }
    low_only, _ = target_ovr_from_skater_stats(boxcars, position_code="D", age=24)
    high_anal = {
        "games_played": 73,
        "icetime": 73169,
        "gameScore": 55.1,
        "onIce_xGoalsPercentage": 0.62,
        "offIce_xGoalsPercentage": 0.49,
        "onIce_corsiPercentage": 0.60,
        "offIce_corsiPercentage": 0.50,
    }
    with_anal, note = target_ovr_from_skater_stats(
        boxcars, position_code="D", age=24, analytics=high_anal
    )
    assert with_anal >= low_only
    assert with_anal >= 0.80  # should not sit in the low 70s
    assert note.startswith("r3|")
    # Strong possession still rates as a legitimate top-4 / top-pair D.
    assert with_anal >= 0.82


def test_poor_possession_d_not_inflated():
    boxcars = {
        "gamesPlayed": 70,
        "points": 20,
        "goals": 3,
        "assists": 17,
        "evPoints": 18,
        "ppPoints": 2,
        "pointsPerGame": 0.29,
        "timeOnIcePerGame": 1100,
        "hits": 50,
        "blockedShots": 80,
        "team_toi_rank_pct": 0.40,
    }
    anal = {
        "games_played": 70,
        "icetime": 70000,
        "gameScore": 10.0,
        "onIce_xGoalsPercentage": 0.43,
        "offIce_xGoalsPercentage": 0.52,
        "onIce_corsiPercentage": 0.44,
        "offIce_corsiPercentage": 0.51,
    }
    ovr, _ = target_ovr_from_skater_stats(boxcars, position_code="D", age=27, analytics=anal)
    assert ovr < 0.82


def test_defensive_d_profile_from_xgf():
    stats = {
        "gamesPlayed": 73,
        "goals": 6,
        "assists": 22,
        "points": 28,
        "ppPoints": 4,
        "hits": 40,
        "blockedShots": 90,
        "mp_onIce_xGoalsPercentage": 0.62,
    }
    assert infer_skater_profile(stats, position_code="D") == "defensive_d"


def test_analytics_impact_score_spence_band():
    score, note = analytics_impact_score(
        {
            "games_played": 73,
            "icetime": 73169,
            "gameScore": 55.1,
            "onIce_xGoalsPercentage": 0.62,
            "offIce_xGoalsPercentage": 0.49,
            "onIce_corsiPercentage": 0.60,
        }
    )
    assert score >= 0.75
    assert "mp_xgf=" in note
