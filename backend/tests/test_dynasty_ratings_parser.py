"""Tests for dynasty_ratings.txt parser."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("NHL_FRANCHISE_DEBUG", "1")

from services.dynasty_ratings_parser import (  # noqa: E402
    apply_overall_patches,
    load_dynasty_ratings_registry,
    normalize_player_name,
    parse_dynasty_ratings,
)


def test_normalize_player_name_strips_accents():
    assert normalize_player_name("Aatu Jämsen") == normalize_player_name("Aatu Jamsen")


def test_parse_sample_block():
    sample = """
=== BOSTON BRUINS ===

-- NHL FORWARDS --
Pastrnak: 95 ovr, 93 cha, 96 off, 84 def, 91 tra, 90 men, 76 phy, 96 pot

-- NHL GOALIES --
Swayman: 92 ovr, 94 glove, 91 blocker, 85 stick, 94 pot
"""
    reg = parse_dynasty_ratings(sample)
    assert reg.parse_stats["parsed"] == 2
    sk = reg.match_player("BOS", "nhl", "David Pastrnak", last_name="Pastrnak")
    assert sk is not None
    assert sk.chapters["overall"] == 95
    assert sk.chapters["offence"] == 96
    g = reg.match_player("BOS", "nhl", "Jeremy Swayman", last_name="Swayman")
    assert g is not None
    assert g.chapters["glove"] == 94


def test_patch_overrides_overall():
    sample = """
=== LOS ANGELES KINGS ===
-- NHL FORWARDS --
Trevor Moore: 83 ovr, 82 cha, 78 off, 85 def, 82 tra, 80 men, 74 phy, 76 pot
"""
    reg = parse_dynasty_ratings(sample)
    patches = "Trevor Moore\n83 → 85\n"
    n = apply_overall_patches(reg, patches)
    assert n >= 1
    entry = reg.match_player("LAK", "nhl", "Trevor Moore", last_name="Moore")
    assert entry is not None
    assert entry.chapters["overall"] == 85
    assert entry.patched is True


def test_full_file_loads():
    reg = load_dynasty_ratings_registry()
    assert reg.parse_stats["parsed"] >= 2000
    assert reg.parse_stats["teams"] == 32


def test_makar_siblings_do_not_collide_on_colorado():
    sample = """
=== COLORADO AVALANCHE ===
-- NHL DEFENCE --
Cale Makar: 98 ovr, 86 cha, 97 off, 88 def, 98 tra, 88 men, 76 phy, 98 pot
-- AHL: COLORADO EAGLES --
Taylor Makar (LW/C): 77 ovr, 76 cha, 74 off, 70 def, 76 tra, 74 men, 74 phy, 78 pot
"""
    reg = parse_dynasty_ratings(sample)
    cale = reg.match_player("COL", "nhl", "Cale Makar", last_name="Makar")
    taylor = reg.match_player("COL", "nhl", "Taylor Makar", last_name="Makar")
    assert cale is not None and cale.chapters["overall"] == 98
    assert taylor is not None and taylor.chapters["overall"] == 77
    assert taylor.level == "ahl"
    assert reg.match_player("COL", "nhl", "Makar", last_name="Makar") is None


def test_spawn_ottawa_affiliated_prospect():
    import random

    from services.dynasty_ratings_parser import spawn_player_from_dynasty_entry

    reg = load_dynasty_ratings_registry()
    prospects = reg.entries_for_team("OTT", "prospect")
    ahl = reg.entries_for_team("OTT", "ahl")
    assert len(prospects) >= 20
    assert any(e.raw_name == "Logan Hensler" for e in prospects)
    assert any(e.raw_name == "Ryan Suzuki" for e in ahl)

    entry = next(e for e in prospects if e.raw_name == "Logan Hensler")
    player = spawn_player_from_dynasty_entry(
        entry,
        rng=random.Random(1),
        pool_context="prospect",
        used_names=set(),
        league_players=[],
        as_of_year=2026,
    )
    from app.sim_engine.entities.player import display_rating

    assert player.identity.name == "Logan Hensler"
    assert getattr(player, "dynasty_ratings_import", False) is True
    assert getattr(player, "player_bio_import", False) is True
    assert str(getattr(player, "birth_date", "") or "").startswith("2006-10-14")
    shown = display_rating(player.ovr())
    assert 72 <= shown <= 78, shown


def test_apply_dynasty_entry_keeps_chapter_overall():
    import random

    from app.sim_engine.entities.player import display_rating
    from services.dynasty_ratings_parser import (
        apply_dynasty_entry_to_player,
        spawn_player_from_dynasty_entry,
    )

    reg = load_dynasty_ratings_registry()
    stutzle = next(e for e in reg.entries_for_team("OTT", "nhl") if e.raw_name == "Tim Stutzle")
    player = spawn_player_from_dynasty_entry(
        stutzle,
        rng=random.Random(2),
        pool_context="ahl",
        used_names=set(),
        league_players=[],
        as_of_year=2026,
    )
    apply_dynasty_entry_to_player(player, stutzle, seed=2)
    shown = display_rating(player.ovr())
    assert 91 <= shown <= 96, shown


if __name__ == "__main__":
    test_normalize_player_name_strips_accents()
    test_parse_sample_block()
    test_patch_overrides_overall()
    test_full_file_loads()
    test_spawn_ottawa_affiliated_prospect()
    test_apply_dynasty_entry_keeps_chapter_overall()
    print("ok")
