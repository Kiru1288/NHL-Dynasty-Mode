"""Calibration: top-line OVRs + attribute-derived overall (same path as generated)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SimEngine"))

from app.sim_engine.engine import build_role_shaped_ratings  # noqa: E402
from app.sim_engine.entities.player import (  # noqa: E402
    BackstoryType,
    BackstoryUpbringing,
    DevResources,
    IdentityBio,
    Player,
    Position,
    PressureLevel,
    Shoots,
    SupportLevel,
    UpbringingType,
    persist_recomputed_ovr,
)
from services.real_nhl_roster_importer import (  # noqa: E402
    align_attribute_ovr_to_target,
    target_ovr_from_goalie_stats,
    target_ovr_from_skater_stats,
)


def test_batherson_tier_target_in_high_80s():
    ovr, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 82,
            "points": 74,
            "goals": 26,
            "assists": 48,
            "evPoints": 50,
            "ppPoints": 24,
            "pointsPerGame": 0.90,
            "timeOnIcePerGame": 1080,
            "shootingPct": 0.13,
            "team_toi_rank_pct": 0.75,
            "hits": 40,
            "blockedShots": 20,
        },
        position_code="R",
        age=26,
    )
    assert 0.87 <= ovr <= 0.91


def test_stutzle_tier_target_in_high_80s():
    ovr, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 82,
            "points": 82,
            "goals": 24,
            "assists": 58,
            "evPoints": 55,
            "ppPoints": 27,
            "pointsPerGame": 1.00,
            "timeOnIcePerGame": 1200,
            "shootingPct": 0.12,
            "faceoffWinPct": 0.48,
            "team_toi_rank_pct": 0.85,
            "hits": 50,
            "blockedShots": 25,
        },
        position_code="C",
        age=23,
    )
    assert 0.875 <= ovr <= 0.92


def test_top_line_below_superstar_ceiling():
    top, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 82,
            "points": 78,
            "goals": 35,
            "assists": 43,
            "evPoints": 52,
            "ppPoints": 26,
            "pointsPerGame": 0.95,
            "timeOnIcePerGame": 1140,
            "shootingPct": 0.14,
            "team_toi_rank_pct": 0.88,
            "hits": 250,
            "blockedShots": 40,
        },
        position_code="L",
        age=25,
    )
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
    assert top < elite
    assert elite >= 0.92


def test_starter_goalie_not_above_top_line_skater():
    skater, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 82,
            "points": 74,
            "goals": 26,
            "assists": 48,
            "evPoints": 50,
            "ppPoints": 24,
            "pointsPerGame": 0.90,
            "timeOnIcePerGame": 1080,
            "shootingPct": 0.13,
            "team_toi_rank_pct": 0.75,
        },
        position_code="R",
        age=26,
    )
    goalie, _ = target_ovr_from_goalie_stats(
        {
            "gamesPlayed": 60,
            "gamesStarted": 58,
            "savePct": 0.912,
            "goalsAgainstAverage": 2.55,
            "wins": 30,
            "losses": 20,
            "otLosses": 6,
            "shotsAgainst": 1800,
            "team_start_rank_pct": 1.0,
        },
        age=28,
        analytics={
            "games_played": 60,
            "gsax": 12.0,
            "xGoals": 150.0,
            "goals": 138.0,
            "hd_gsax": 8.0,
            "kind": "goalie",
        },
    )
    assert 0.82 <= goalie <= 0.90
    assert goalie <= skater + 0.02



def test_displayed_ovr_comes_from_attribute_card():
    """Same rule as generated players: OVR is compute_ovr(ratings), then nudged to target."""
    target = 0.88
    rng = random.Random(7)
    ratings = build_role_shaped_ratings(
        position=Position.C,
        target_ovr=target,
        rng=rng,
        profile="playmaker",
    )
    identity = IdentityBio(
        name="Test Forward",
        age=25,
        birth_year=2000,
        birth_country="Canada",
        birth_city="Ottawa",
        height_cm=183,
        weight_kg=90,
        position=Position.C,
        shoots=Shoots.L,
        draft_year=2018,
        draft_round=1,
        draft_pick=18,
    )
    backstory = BackstoryUpbringing(
        backstory=BackstoryType.PRODIGY,
        upbringing=UpbringingType.STABLE_MIDDLE_CLASS,
        family_support=SupportLevel.MEDIUM,
        early_pressure=PressureLevel.MODERATE,
        dev_resources=DevResources.LOCAL,
    )
    player = Player(
        identity=identity,
        backstory=backstory,
        ratings=ratings,
        rng_seed=7,
        pool_context="nhl",
    )
    final = align_attribute_ovr_to_target(player, target, rounds=12)
    assert abs(final - target) <= 0.02
    recomputed = persist_recomputed_ovr(player)
    assert abs(recomputed - final) <= 0.004
    assert float(player.ratings.get("skg_speed", 0) or 0) >= 70
    assert float(player.ratings.get("off_wrist_shot_accuracy", 0) or 0) >= 60


def test_mid_sample_hot_ppg_does_not_mint_elite_target():
    hot, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 54,
            "points": 57,
            "goals": 19,
            "assists": 38,
            "evPoints": 40,
            "ppPoints": 14,
            "pointsPerGame": 1.06,
            "timeOnIcePerGame": 1140,
            "shootingPct": 0.14,
            "team_toi_rank_pct": 0.87,
            "hits": 20,
            "blockedShots": 15,
        },
        position_code="C",
        age=30,
    )
    full, _ = target_ovr_from_skater_stats(
        {
            "gamesPlayed": 79,
            "points": 81,
            "goals": 39,
            "assists": 42,
            "evPoints": 51,
            "ppPoints": 30,
            "pointsPerGame": 1.03,
            "timeOnIcePerGame": 1230,
            "shootingPct": 0.15,
            "team_toi_rank_pct": 0.89,
            "hits": 40,
            "blockedShots": 30,
        },
        position_code="C",
        age=29,
    )
    assert hot < 0.90
    assert full >= 0.90
    assert full - hot >= 0.015


def test_align_closes_large_target_gap():
    target = 0.91
    rng = random.Random(11)
    ratings = build_role_shaped_ratings(
        position=Position.C,
        target_ovr=0.78,
        rng=rng,
        profile="two_way",
    )
    identity = IdentityBio(
        name="Gap Closer",
        age=28,
        birth_year=1997,
        birth_country="Canada",
        birth_city="Toronto",
        height_cm=185,
        weight_kg=92,
        position=Position.C,
        shoots=Shoots.R,
        draft_year=2015,
        draft_round=1,
        draft_pick=2,
    )
    backstory = BackstoryUpbringing(
        backstory=BackstoryType.PRODIGY,
        upbringing=UpbringingType.STABLE_MIDDLE_CLASS,
        family_support=SupportLevel.MEDIUM,
        early_pressure=PressureLevel.MODERATE,
        dev_resources=DevResources.LOCAL,
    )
    player = Player(
        identity=identity,
        backstory=backstory,
        ratings=ratings,
        rng_seed=11,
        pool_context="nhl",
    )
    final = align_attribute_ovr_to_target(player, target, rounds=40)
    assert abs(final - target) <= 0.02


