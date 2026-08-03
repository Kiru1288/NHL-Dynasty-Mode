"""Goalie OVR calibration: GSAx + NHL counting stats + top-10 guide."""

from __future__ import annotations

from services.real_nhl_analytics import (
    goalie_analytics_impact_score,
    parse_moneypuck_goalies_csv,
)
from services.real_nhl_roster_importer import target_ovr_from_goalie_stats


def _hellebuyck_box():
    return {
        "gamesPlayed": 63,
        "gamesStarted": 63,
        "savePct": 0.925,
        "goalsAgainstAverage": 2.00,
        "wins": 47,
        "losses": 12,
        "otLosses": 3,
        "shotsAgainst": 1664,
        "shutouts": 8,
        "team_start_rank_pct": 1.0,
    }


def _hellebuyck_mp():
    return {
        "games_played": 63,
        "xGoals": 164.6,
        "goals": 125.0,
        "gsax": 39.6,
        "highDangerxGoals": 55.6,
        "highDangerGoals": 38.0,
        "hd_gsax": 17.6,
        "kind": "goalie",
    }


def test_parse_goalie_csv_prefers_all_situations():
    csv_text = """playerId,season,name,team,position,situation,games_played,icetime,xGoals,goals,highDangerxGoals,highDangerGoals,ongoal
8476945,2024,Connor Hellebuyck,WPG,G,5on5,63,3000,100,80,40,30,1000
8476945,2024,Connor Hellebuyck,WPG,G,all,63,3600,164.6,125,55.6,38,1600
"""
    parsed = parse_moneypuck_goalies_csv(csv_text)
    assert 8476945 in parsed
    assert parsed[8476945]["situation"] == "all"
    assert abs(parsed[8476945]["gsax"] - 39.6) < 0.05


def test_hellebuyck_tier_low_90s():
    ovr, note = target_ovr_from_goalie_stats(
        _hellebuyck_box(), age=32, analytics=_hellebuyck_mp()
    )
    assert ovr >= 0.90
    assert ovr <= 0.94
    assert "r3g|" in note


def test_negative_gsax_workhorse_not_elite():
    """Swayman-like: heavy starts but below expected — capped below true elites."""
    box = {
        "gamesPlayed": 58,
        "gamesStarted": 58,
        "savePct": 0.905,
        "goalsAgainstAverage": 3.00,
        "wins": 22,
        "losses": 29,
        "otLosses": 5,
        "shotsAgainst": 1700,
        "shutouts": 2,
        "team_start_rank_pct": 1.0,
    }
    anal = {
        "games_played": 58,
        "gsax": -9.1,
        "xGoals": 166.9,
        "goals": 176.0,
        "hd_gsax": 18.2,
        "highDangerxGoals": 59.2,
        "highDangerGoals": 41.0,
        "kind": "goalie",
    }
    ovr, _ = target_ovr_from_goalie_stats(box, age=26, analytics=anal)
    helle, _ = target_ovr_from_goalie_stats(
        _hellebuyck_box(), age=32, analytics=_hellebuyck_mp()
    )
    assert 0.80 <= ovr <= 0.85
    assert ovr < helle - 0.04


def test_cup_starter_gsax_lifts_above_backup_cluster():
    """Adin Hill / Thompson type — positive GSAx starters should clear the 81 pile."""
    box = {
        "gamesPlayed": 50,
        "gamesStarted": 48,
        "savePct": 0.910,
        "goalsAgainstAverage": 2.50,
        "wins": 28,
        "losses": 16,
        "otLosses": 4,
        "shotsAgainst": 1400,
        "shutouts": 3,
        "team_start_rank_pct": 0.95,
    }
    anal = {
        "games_played": 50,
        "gsax": 14.5,
        "xGoals": 135.5,
        "goals": 121.0,
        "hd_gsax": 7.5,
        "kind": "goalie",
    }
    ovr, _ = target_ovr_from_goalie_stats(box, age=29, analytics=anal)
    assert 0.83 <= ovr <= 0.89


def test_small_sample_elite_sv_capped():
    """Stolarz-type: tiny sample + elite SV% cannot leapfrog Vezina workhorses."""
    box = {
        "gamesPlayed": 34,
        "gamesStarted": 34,
        "savePct": 0.926,
        "goalsAgainstAverage": 2.10,
        "wins": 20,
        "losses": 9,
        "otLosses": 3,
        "shotsAgainst": 900,
        "shutouts": 3,
        "team_start_rank_pct": 0.7,
    }
    anal = {
        "games_played": 34,
        "gsax": 25.8,
        "xGoals": 96.8,
        "goals": 71.0,
        "hd_gsax": 14.6,
        "kind": "goalie",
    }
    ovr, _ = target_ovr_from_goalie_stats(box, age=31, analytics=anal)
    assert ovr <= 0.875
    assert ovr >= 0.84


def test_true_backup_stays_separated():
    ovr, _ = target_ovr_from_goalie_stats(
        {
            "gamesPlayed": 18,
            "gamesStarted": 12,
            "savePct": 0.898,
            "goalsAgainstAverage": 3.15,
            "wins": 5,
            "losses": 8,
            "otLosses": 2,
            "shotsAgainst": 480,
            "team_start_rank_pct": 0.0,
        },
        age=28,
    )
    assert ovr < 0.80


def test_goalie_analytics_score_hellebuyck():
    score, note = goalie_analytics_impact_score(_hellebuyck_mp())
    assert score >= 0.85
    assert "mp_gsax=" in note
