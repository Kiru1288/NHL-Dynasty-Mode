"""Tests for emergent prospect identity (body maturation + play-driven archetype)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "SimEngine"
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

import pytest  # noqa: E402


class _Ident:
    def __init__(self, *, age=17, height_cm=180, weight_kg=75, position="C"):
        self.age = age
        self.height_cm = height_cm
        self.weight_kg = weight_kg
        self.position = position


def _player(**kwargs):
    defaults = {
        "identity": _Ident(),
        "ratings": {
            "off_shot_accuracy": 62,
            "off_shot_power": 58,
            "pm_passing": 55,
            "def_positioning": 54,
            "skg_speed": 57,
            "phy_strength": 52,
        },
        "gp": 0,
        "games_played": 0,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_apply_yearly_body_maturation_adds_weight_for_teens():
    from app.sim_engine.generation.prospect_identity import apply_yearly_body_maturation

    p = _player(identity=_Ident(age=16, height_cm=182, weight_kg=72))
    changed = apply_yearly_body_maturation(p, __import__("random").Random(42))
    assert changed is True
    assert p.identity.weight_kg >= 72


def test_commit_identity_from_goal_scoring_stats():
    from app.sim_engine.generation.prospect_identity import commit_prospect_identity

    p = _player(
        ratings={
            "off_shot_accuracy": 70,
            "off_shot_power": 68,
            "pm_passing": 52,
            "def_positioning": 50,
            "skg_speed": 58,
            "phy_strength": 55,
        }
    )
    stats = {"gp": 34, "goals": 28, "assists": 10, "points": 38}
    ok = commit_prospect_identity(p, stats=stats, min_gp=8)
    assert ok is True
    assert "sniper" in str(p.playstyle).lower() or p.archetype == "SNIPER"


def test_commit_identity_from_playmaking_stats():
    from app.sim_engine.generation.prospect_identity import commit_prospect_identity

    p = _player(
        ratings={
            "off_shot_accuracy": 55,
            "off_shot_power": 52,
            "pm_passing": 72,
            "pm_vision": 70,
            "def_positioning": 54,
            "skg_speed": 60,
            "phy_strength": 50,
        }
    )
    stats = {"gp": 40, "goals": 8, "assists": 42, "points": 50}
    ok = commit_prospect_identity(p, stats=stats, min_gp=8)
    assert ok is True
    assert "playmaker" in str(p.playstyle).lower() or p.archetype == "PLAYMAKER"


def test_commit_skips_without_enough_games():
    from app.sim_engine.generation.prospect_identity import refresh_player_identity

    p = _player()
    p._identity_committed = True
    result = refresh_player_identity(p, stats={"gp": 4, "goals": 3, "assists": 1, "points": 4})
    assert result.get("changed") is False


def test_archetype_can_shift_after_body_and_stats_change():
    from app.sim_engine.generation.prospect_identity import refresh_player_identity

    p = _player(
        identity=_Ident(age=19, height_cm=182, weight_kg=76),
        ratings={
            "off_shot_accuracy": 58,
            "off_shot_power": 56,
            "pm_passing": 54,
            "def_positioning": 56,
            "skg_speed": 57,
            "phy_strength": 52,
        },
    )
    p.playstyle = "playmaker"
    p.archetype = "PLAYMAKER"
    p._identity_committed = True

    p.identity.height_cm = 193
    p.identity.weight_kg = 98
    p._body_maturation_changed = True

    stats = {"gp": 36, "goals": 22, "assists": 11, "points": 33}
    result = refresh_player_identity(p, stats=stats, min_gp=8)
    assert result.get("changed") is True
    assert result.get("archetype_changed") is True or "power" in str(p.playstyle).lower() or p.archetype == "POWER_FORWARD"


def test_young_adult_still_gains_weight():
    from app.sim_engine.generation.prospect_identity import apply_yearly_body_maturation

    p = _player(identity=_Ident(age=23, height_cm=188, weight_kg=86))
    before = p.identity.weight_kg
    changed = apply_yearly_body_maturation(p, __import__("random").Random(7))
    assert changed is True
    assert p.identity.weight_kg >= before
