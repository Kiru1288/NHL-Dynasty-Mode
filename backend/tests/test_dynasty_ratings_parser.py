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


if __name__ == "__main__":
    test_normalize_player_name_strips_accents()
    test_parse_sample_block()
    test_patch_overrides_overall()
    test_full_file_loads()
    print("ok")
