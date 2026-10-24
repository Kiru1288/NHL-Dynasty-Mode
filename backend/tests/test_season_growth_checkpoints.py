"""Per-player NHL in-season checkpoints and potential season caps."""

import random
from types import SimpleNamespace

from app.sim_engine.progression.development import (
    ensure_nhl_checkpoint_marks,
    player_nhl_checkpoints_due,
)
from app.sim_engine.progression.potential import nudge_shown_potential
from services.franchise_sim import _league_games_played


def test_per_player_checkpoint_marks_are_stable_and_spread():
    player = SimpleNamespace(
        id="p-1",
        gp=0,
        games_played=0,
        _nhl_checkpoint_done=[],
    )
    marks_a = ensure_nhl_checkpoint_marks(player, 2026)
    marks_b = ensure_nhl_checkpoint_marks(player, 2026)
    assert marks_a == marks_b
    assert len(marks_a) == 3
    assert marks_a[0] < marks_a[1] < marks_a[2]
    assert 16 <= marks_a[0] <= 24
    assert 37 <= marks_a[1] <= 45
    assert 57 <= marks_a[2] <= 65


def test_checkpoint_due_uses_player_gp_not_league():
    player = SimpleNamespace(
        id="p-2",
        gp=0,
        games_played=0,
        _nhl_checkpoint_done=[],
    )
    marks = ensure_nhl_checkpoint_marks(player, 2026)
    player.gp = marks[0]
    player.games_played = marks[0]
    due = player_nhl_checkpoints_due(player, 2026)
    assert due == [marks[0]]


def test_games_played_comes_from_the_standings():
    session = SimpleNamespace(
        standings=SimpleNamespace(records={"5": SimpleNamespace(gp=82)}),
        sim=SimpleNamespace(league=SimpleNamespace(teams=[SimpleNamespace(gp=0)])),
    )
    assert _league_games_played(session) == 82


def test_shown_potential_cannot_jump_more_than_three_in_a_season():
    player = SimpleNamespace(potential=0.80, ratings={"dev_potential": 80})
    rng = random.Random(2)
    total = 0.0
    for _ in range(10):
        total += nudge_shown_potential(
            player,
            rng,
            ovr_delta=1.4,
            winning=True,
            production_gap=0.2,
        )
    assert 0 < total <= 3
