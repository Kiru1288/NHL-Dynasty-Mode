"""Draft class ages should be first-year 17–18s, not a 19–20 overager pile."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("NHL_FRANCHISE_DEBUG", "1")

from app.sim_engine.entities.player import Position  # noqa: E402
from app.sim_engine.league_hierarchy_bootstrap import (  # noqa: E402
    _pick_spawn_age,
    _shape_draft_class_pipeline,
    _spawn_player,
    reanchor_generated_junior_dobs,
    set_spawn_as_of_year,
)


def test_spawn_birth_year_tracks_season_not_2025():
    set_spawn_as_of_year(2026)
    rng = random.Random(11)
    player = _spawn_player(
        rng,
        pos=Position.C,
        ovr_lo=0.40,
        ovr_hi=0.48,
        age_lo=18,
        age_hi=18,
        used_names=set(),
        league_players=[],
        pool_context="junior",
        league_code="CHL_OHL",
    )
    ident = player.identity
    assert ident.age == 18
    assert ident.birth_year == 2008
    assert ident.birth_year != 2025 - 18
    assert getattr(player, "_dob_anchor_year") == 2026


def test_chl_age_weights_are_mostly_first_year():
    rng = random.Random(42)
    ages = [
        _pick_spawn_age(rng, 16, 20, pool_context="junior", league_code="CHL_OHL")
        for _ in range(400)
    ]
    first_year = sum(1 for a in ages if a <= 18)
    overagers = sum(1 for a in ages if a >= 19)
    assert first_year > overagers * 2
    assert ages.count(18) > ages.count(20)


def test_reanchor_restores_eighteen_year_olds_from_2025_formula():
    ident = SimpleNamespace(name="Kid", age=19, birth_year=2007, position=Position.C)
    player = SimpleNamespace(identity=ident, age=19, real_nhl_import=False)
    league = SimpleNamespace(
        development_leagues=[{"teams": [{"players": [player]}]}],
        _junior_dobs_reanchored=False,
    )
    fixed = reanchor_generated_junior_dobs(league, 2026)
    assert fixed == 1
    assert ident.birth_year == 2008
    assert ident.age == 18
    assert reanchor_generated_junior_dobs(league, 2026) == 0


def test_pipeline_stars_are_first_year_eligible():
    rng = random.Random(7)
    players = []
    for i, age in enumerate([18] * 40 + [19] * 40 + [20] * 40):
        pos = Position.C if i % 8 else Position.G
        ident = SimpleNamespace(name=f"P{i}", age=age, position=pos)
        p = SimpleNamespace(
            identity=ident,
            drafted=False,
            nhl_rights_team_id=None,
            rights_team_id=None,
            drafted_by=None,
            ratings={"skating": 0.4},
            pipeline_tier="pool",
        )
        players.append(p)
    league = SimpleNamespace(
        development_leagues=[{"league_code": "CHL_OHL", "teams": [{"players": players}]}],
    )
    _shape_draft_class_pipeline(league, rng)
    stars = [
        p
        for p in players
        if str(getattr(p, "pipeline_tier", "")).lower() in ("transcendent", "franchise", "elite", "top")
        and getattr(p.identity, "position", None) != Position.G
    ]
    assert stars
    first_year = sum(1 for p in stars if p.identity.age <= 18)
    assert first_year >= int(len(stars) * 0.7)
