"""Tests for Real NHL 23-man opening roster trim → AHL overflow."""

from __future__ import annotations

from types import SimpleNamespace

from services.real_nhl_roster_importer import (
    NHL_OPENING_ROSTER_MAX,
    select_opening_nhl_roster,
    trim_team_roster_to_nhl_limit,
)


def _p(name: str, pos: str, ovr99: float) -> SimpleNamespace:
    return SimpleNamespace(
        identity=SimpleNamespace(name=name, position=pos),
        position=pos,
        ovr=lambda o=ovr99 / 99.0: o,
        real_nhl_import=True,
        in_minors=False,
        is_buried=False,
        buried=False,
        roster_location="nhl",
        context=SimpleNamespace(current_team_id="T1"),
    )


def test_select_opening_roster_caps_at_23_with_position_floors():
    players = []
    for i in range(16):
        players.append(_p(f"F{i}", "C" if i % 3 == 0 else ("LW" if i % 3 == 1 else "RW"), 90 - i))
    for i in range(10):
        players.append(_p(f"D{i}", "D", 88 - i))
    for i in range(4):
        players.append(_p(f"G{i}", "G", 85 - i))

    keep, overflow = select_opening_nhl_roster(players)
    assert len(keep) == NHL_OPENING_ROSTER_MAX
    assert len(overflow) == len(players) - NHL_OPENING_ROSTER_MAX
    g = sum(1 for p in keep if p.position == "G")
    d = sum(1 for p in keep if p.position == "D")
    f = sum(1 for p in keep if p.position in ("C", "LW", "RW"))
    assert g >= 2
    assert d >= 6
    assert f >= 12
    # Best goalies kept
    assert any(p.identity.name == "G0" for p in keep)


def test_trim_moves_overflow_to_ahl_roster():
    team = SimpleNamespace(
        team_id="T1",
        id="T1",
        city="Test",
        name="Test",
        roster=[_p(f"P{i}", "C" if i < 18 else ("D" if i < 28 else "G"), 90 - i) for i in range(32)],
        ahl_roster=[],
    )
    info = trim_team_roster_to_nhl_limit(team)
    assert info["nhl"] == 23
    assert info["sent_to_ahl"] == 9
    assert len(team.roster) == 23
    assert len(team.ahl_roster) == 9
    assert all(not getattr(p, "in_minors", False) for p in team.roster)
    assert all(getattr(p, "in_minors", False) for p in team.ahl_roster)
    assert all(getattr(p, "roster_location", "") == "ahl" for p in team.ahl_roster)
