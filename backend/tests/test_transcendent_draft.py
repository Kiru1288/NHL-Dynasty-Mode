"""A transcendent draft talent is 99 overall on draft day and stays there."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine"), str(ROOT / "backend" / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.sim_engine.entities.player import display_rating, player_current_ovr_01  # noqa: E402
from app.sim_engine.engine import _career_aging_decline_try_v3  # noqa: E402
from app.sim_engine.league_hierarchy_bootstrap import _shape_draft_class_pipeline  # noqa: E402
from app.sim_engine.progression.development import (  # noqa: E402
    apply_live_overall_step,
    season_growth_context,
)
from app.sim_engine.progression.regression import apply_regression  # noqa: E402
from test_chapter_attributes import _make_player  # noqa: E402


def _ovr(player) -> int:
    return int(round(display_rating(player_current_ovr_01(player))))


def _pot(player) -> int:
    raw = player.ratings.get("dev_potential")
    return int(round(float(raw)))


def _draft_class():
    players = []
    for i in range(16):
        player = _make_player("C" if i % 5 else "D", seed=700 + i)
        player.id = f"draft-{i}"
        player.identity.age = 18
        player.identity.birth_year = 2008
        player.identity.name = f"Draft Prospect {i}"
        player.drafted = False
        player.nhl_rights_team_id = None
        player.rights_team_id = None
        players.append(player)
    league = SimpleNamespace(
        season_year=2025,
        development_leagues=[
            {
                "league_code": "CHL_OHL",
                "teams": [{"team_id": "OHL_1", "name": "Ottawa 67s", "players": players}],
            }
        ],
    )
    return league, players


def test_a_forced_transcendent_is_99_overall_from_the_draft():
    os.environ["NHL_FORCE_TRANSCENDENT"] = "1"
    try:
        league, players = _draft_class()
        _shape_draft_class_pipeline(league, random.Random(1), draft_year=2026)
    finally:
        os.environ.pop("NHL_FORCE_TRANSCENDENT", None)

    found = [p for p in players if getattr(p, "is_transcendent", False)]
    assert len(found) == 1
    star = found[0]
    assert _ovr(star) == 99, _ovr(star)
    assert _pot(star) == 99
    assert str(getattr(star, "pipeline_tier", "")) == "transcendent"


def test_a_transcendent_stays_99_through_a_bad_year_and_old_age():
    os.environ["NHL_FORCE_TRANSCENDENT"] = "1"
    try:
        league, players = _draft_class()
        _shape_draft_class_pipeline(league, random.Random(2), draft_year=2026)
    finally:
        os.environ.pop("NHL_FORCE_TRANSCENDENT", None)
    star = next(p for p in players if getattr(p, "is_transcendent", False))
    star.identity.age = 38
    star.psych.morale = 0.1
    star.games_played = 70
    star._dev_season_points_surplus = -80
    star.season_start_ovr = 99

    assert season_growth_context(star, bucket="nhl")["gain"] == 0.0
    assert apply_live_overall_step(star, season_id=2026, bucket="nhl", day=1) == 0
    apply_regression(star, random.Random(3))
    assert _career_aging_decline_try_v3(star, random.Random(4), league) is None
    assert _ovr(star) == 99
    assert _pot(star) == 99


def test_the_public_board_shows_him_at_99_with_a_full_file():
    from services.draft_ranking_logic import (
        _character_concerns_flag,
        compute_consensus_potential_evaluation,
        pin_transcendent_first,
        stamp_transcendent_public_entry,
    )

    row = {
        "key": "star",
        "is_transcendent": True,
        "true_ovr": 99,
        "potential_score": 99,
        "character_concerns": True,
        "character_score": 30,
        "scout_confidence": "Low",
        "scout_tags": ["Elite Potential"],
    }
    assert compute_consensus_potential_evaluation(row) == 99.0
    assert _character_concerns_flag(row) is False
    stamp_transcendent_public_entry(row)
    assert row["current_ovr_range"] == [99.0, 99.0]
    assert row["potential_score"] == 99.0
    assert row["scouted_percentage"] == 100.0
    assert row["public_scouting_confidence"] == 100.0
    assert row["character_concerns"] is False
    assert row["scout_confidence"] == "High"
    assert row["scout_projection"] == "Can't Miss"
    board = [{"key": "other", "_score": 200}, row]
    pin_transcendent_first(board)
    assert board[0]["key"] == "star"


def test_the_public_board_shows_him_at_99_with_a_full_file():
    from services.draft_ranking_logic import (
        _character_concerns_flag,
        compute_consensus_potential_evaluation,
        pin_transcendent_first,
        stamp_transcendent_public_entry,
    )

    row = {
        "key": "star",
        "is_transcendent": True,
        "true_ovr": 99,
        "potential_score": 99,
        "character_concerns": True,
        "character_score": 30,
        "scout_confidence": "Low",
        "scout_tags": ["Elite Potential"],
    }
    assert compute_consensus_potential_evaluation(row) == 99.0
    assert _character_concerns_flag(row) is False
    stamp_transcendent_public_entry(row)
    assert row["current_ovr_range"] == [99.0, 99.0]
    assert row["potential_score"] == 99.0
    assert row["scouted_percentage"] == 100.0
    assert row["public_scouting_confidence"] == 100.0
    assert row["character_concerns"] is False
    assert row["scout_confidence"] == "High"
    assert row["scout_projection"] == "Can't Miss"
    board = [{"key": "other", "_score": 200}, row]
    pin_transcendent_first(board)
    assert board[0]["key"] == "star"
