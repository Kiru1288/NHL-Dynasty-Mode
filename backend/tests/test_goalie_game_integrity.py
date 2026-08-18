"""Advance Day must not hard-fail when NHL goalie slots can be self-healed."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from services import franchise_sim as fs  # noqa: E402


def _player(pid, pos="G", *, in_minors=False, ovr=70.0, nmc=False):
    return types.SimpleNamespace(
        id=pid,
        name=pid,
        position=types.SimpleNamespace(value=pos),
        identity=types.SimpleNamespace(position=types.SimpleNamespace(value=pos), name=pid),
        retired=False,
        is_buried=False,
        buried=False,
        in_minors=in_minors,
        roster_location="ahl" if in_minors else "nhl",
        on_ir=False,
        on_ltir=False,
        is_ir=False,
        is_ltir=False,
        ovr=ovr,
        contract=types.SimpleNamespace(nmc=nmc, no_move_clause=nmc),
        _world_injury_games_remaining=0,
        health=None,
        injury_status=None,
        status=None,
    )


class GoalieGameIntegrityTests(unittest.TestCase):
    def test_pos_str_falls_back_to_player_position(self):
        p = types.SimpleNamespace(
            identity=types.SimpleNamespace(position=None),
            position=types.SimpleNamespace(value="G"),
        )
        self.assertEqual(fs._pos_str(p), "G")

    def test_ensure_reactivates_misflagged_nhl_goalie(self):
        g = _player("g1", in_minors=True, ovr=72)
        team = types.SimpleNamespace(
            roster=[g],
            ahl_roster=[],
            echl_roster=[],
            team_id="T1",
            id="T1",
            name="Test",
            abbrev="TST",
        )
        status = fs._ensure_goalie_for_game(team)
        self.assertGreater(int(status["total"]), 0)
        self.assertFalse(bool(g.in_minors))
        self.assertEqual(getattr(g, "roster_location", ""), "nhl")

    def test_ensure_calls_up_affiliate_goalie(self):
        sk = _player("s1", pos="C", ovr=60)
        g = _player("g1", in_minors=True, ovr=68)
        team = types.SimpleNamespace(
            roster=[sk],
            ahl_roster=[g],
            echl_roster=[],
            team_id="T2",
            id="T2",
            name="Test2",
            abbrev="TS2",
        )
        status = fs._ensure_goalie_for_game(team)
        self.assertGreater(int(status["total"]), 0)
        self.assertIn(g, team.roster)
        self.assertFalse(bool(g.in_minors))
        self.assertNotIn(g, team.ahl_roster)


if __name__ == "__main__":
    unittest.main()
