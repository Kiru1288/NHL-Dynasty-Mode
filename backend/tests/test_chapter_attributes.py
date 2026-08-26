"""
Chapter-Based Player Rating System foundation tests.
Run: python backend/tests/test_chapter_attributes.py
"""
from __future__ import annotations

import copy
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("NHL_FRANCHISE_DEBUG", "1")

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]

from app.sim_engine.entities.chapter_attributes import (  # noqa: E402
    SCHEMA_VERSION,
    aggregate_chapter_score,
    build_attribute_profile,
    detect_emergent_tendencies,
    ensure_player_attribute_profile,
    generate_hidden_profile,
    get_visible_chapter_ids,
    legacy_ratings_from_hidden,
    player_type_for_position,
    profile_deepcopy,
    sync_legacy_ratings_from_profile,
)
from app.sim_engine.entities.player import (  # noqa: E402
    ATTRIBUTE_KEYS,
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
)
from app.sim_engine.engine import SimEngine  # noqa: E402


def _make_player(position: str = "RW", seed: int = 4242) -> Player:
    pos_enum = Position(position) if position in ("C", "LW", "RW", "D", "G") else Position.RW
    ident = IdentityBio(
        name=f"Test {position}",
        age=24,
        birth_year=2001,
        birth_country="CAN",
        birth_city="Testville",
        height_cm=185,
        weight_kg=88,
        position=pos_enum,
        shoots=Shoots.R,
        draft_year=2019,
        draft_round=1,
        draft_pick=10,
    )
    backstory = BackstoryUpbringing(
        backstory=BackstoryType.PRODIGY,
        upbringing=UpbringingType.STABLE_MIDDLE_CLASS,
        family_support=SupportLevel.HIGH,
        early_pressure=PressureLevel.MODERATE,
        dev_resources=DevResources.ELITE,
    )
    return Player(
        identity=ident,
        backstory=backstory,
        ratings={},
        rng_seed=seed,
        enforce_floor_on_init=False,
    )


def test_1_same_offence_different_players() -> None:
    chapters = {"offence": 90, "overall": 88}
    scorer = build_attribute_profile(
        chapters,
        position="RW",
        tendencies={"scorer": 1.0},
        seed=101,
    )
    playmaker = build_attribute_profile(
        chapters,
        position="RW",
        tendencies={"playmaker": 1.0},
        seed=101,
    )
    h_s = scorer["hidden"]
    h_p = playmaker["hidden"]
    off_s = aggregate_chapter_score("offence", h_s, player_type="skater")
    off_p = aggregate_chapter_score("offence", h_p, player_type="skater")
    assert off_s is not None and abs(off_s - 90) <= 3.0
    assert off_p is not None and abs(off_p - 90) <= 3.0
    assert h_s["finishing"] > h_p["finishing"] + 4
    assert h_p["playmaking"] > h_s["playmaking"] + 4
    tend_s = detect_emergent_tendencies(h_s)
    tend_p = detect_emergent_tendencies(h_p)
    assert tend_s.get("scorer", 0) > tend_p.get("scorer", 0)
    assert tend_p.get("playmaker", 0) > tend_s.get("playmaker", 0)


def test_2_specialist_physical_outlier() -> None:
    chapters = {
        "overall": 77,
        "offence": 72,
        "defence": 80,
        "transition": 71,
        "mental": 86,
        "physical": 98,
        "character": 74,
        "potential": 70,
    }
    profile = build_attribute_profile(chapters, position="D", seed=202)
    assert profile["chapters"]["physical"] == 98
    assert profile["chapters"]["overall"] == 77
    assert profile["derived_chapters"]["physical"] >= 95
    assert profile["derived_chapters"]["offence"] <= 76


def test_3_two_way_star() -> None:
    chapters = {"offence": 92, "defence": 91, "transition": 86, "mental": 88, "overall": 90}
    profile = build_attribute_profile(chapters, position="C", seed=303)
    assert profile["derived_chapters"]["offence"] >= 88
    assert profile["derived_chapters"]["defence"] >= 87


def test_4_weakness_preserved() -> None:
    chapters = {
        "overall": 80,
        "offence": 88,
        "defence": 84,
        "transition": 55,
        "mental": 82,
        "physical": 78,
        "character": 80,
        "potential": 75,
    }
    profile = build_attribute_profile(chapters, position="LW", seed=404, preserve_weaknesses=True)
    assert profile["derived_chapters"]["transition"] <= 60
    assert profile["derived_chapters"]["offence"] >= 84


def test_5_determinism() -> None:
    chapters = {"offence": 87, "defence": 83, "overall": 84}
    a = generate_hidden_profile(chapters, position="RW", seed=505)
    b = generate_hidden_profile(chapters, position="RW", seed=505)
    assert a == b
    c = generate_hidden_profile(chapters, position="RW", seed=506)
    assert a != c


def test_6_save_load_roundtrip() -> None:
    player = _make_player("C", seed=606)
    profile = build_attribute_profile(
        {"overall": 85, "offence": 88, "defence": 79, "transition": 82, "mental": 84, "physical": 80, "character": 83, "potential": 86},
        position="C",
        tendencies={"playmaker": 0.7},
        seed=606,
    )
    player.attribute_profile = profile
    blob = pickle.dumps(player)
    loaded = pickle.loads(blob)
    assert loaded.attribute_profile["chapters"] == player.attribute_profile["chapters"]
    assert loaded.attribute_profile["hidden"] == player.attribute_profile["hidden"]


def test_7_legacy_compatibility_sim_engine() -> None:
    player = _make_player("RW", seed=707)
    profile = build_attribute_profile(
        {"overall": 86, "offence": 90, "defence": 72, "transition": 84, "mental": 80, "physical": 78, "character": 82, "potential": 85},
        position="RW",
        tendencies={"scorer": 1.0},
        seed=707,
    )
    player.attribute_profile = profile
    projected = sync_legacy_ratings_from_profile(player)
    assert projected
    assert all(k in ATTRIBUTE_KEYS for k in projected)
    sim = SimEngine()
    sim.set_franchise_game_stat_modifiers(
        home_player_modifiers={"": {}},
        away_player_modifiers={},
    )
    w = sim._gm_shot_volume_weight(player)
    assert w > 0
    ovr = player.ovr()
    assert 0.0 < ovr <= 1.0


def test_8_goalie_separate_schema() -> None:
    assert player_type_for_position("G") == "goalie"
    goalie_ids = get_visible_chapter_ids("goalie")
    assert "glove" in goalie_ids
    assert "offence" not in goalie_ids
    profile = build_attribute_profile(
        {"overall": 84, "glove": 88, "blocker": 86, "stick": 80, "potential": 82},
        position="G",
        player_type="goalie",
        seed=808,
    )
    assert profile["player_type"] == "goalie"
    assert "glove_reaction" in profile["hidden"]
    assert "finishing" not in profile["hidden"]


def test_schema_version_export() -> None:
    from app.sim_engine.entities.chapter_attributes import chapter_schema_export

    exported = chapter_schema_export()
    assert exported["schema_version"] == SCHEMA_VERSION
    assert len(exported["skater_chapters"]) >= 7
    assert len(exported["goalie_chapters"]) >= 4


if __name__ == "__main__":
    tests = [
        test_1_same_offence_different_players,
        test_2_specialist_physical_outlier,
        test_3_two_way_star,
        test_4_weakness_preserved,
        test_5_determinism,
        test_6_save_load_roundtrip,
        test_7_legacy_compatibility_sim_engine,
        test_8_goalie_separate_schema,
        test_schema_version_export,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"All {len(tests)} chapter attribute tests passed.")
