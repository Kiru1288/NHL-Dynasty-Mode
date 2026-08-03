"""Regression tests for shared roster compliance (active roster + gate)."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _player(pid, pos="C", *, buried=False, on_ir=False, on_ltir=False, retired=False):
    return types.SimpleNamespace(
        id=pid,
        name=pid,
        position=types.SimpleNamespace(value=pos),
        identity=types.SimpleNamespace(position=types.SimpleNamespace(value=pos), name=pid),
        retired=retired,
        is_buried=buried,
        buried=buried,
        in_minors=buried,
        on_ir=on_ir,
        on_ltir=on_ltir,
        is_ir=on_ir,
        is_ltir=on_ltir,
        contract=types.SimpleNamespace(
            years_remaining=2,
            aav_m=1.0,
            cap_hit_m=1.0,
            is_nhl_spc=True,
            type="STANDARD",
            contract_type="STANDARD",
        ),
        signed_status="signed",
    )


def _team(roster=None, ahl=None):
    return types.SimpleNamespace(
        team_id="OTT",
        id="OTT",
        roster=list(roster or []),
        ahl_roster=list(ahl or []),
        echl_roster=[],
        prospect_pool=[],
        injured_reserve=[],
    )


class RosterComplianceUnitTests(unittest.TestCase):
    def test_position_enum_unwrap(self):
        from app.sim_engine.entities.player import Position
        from services.roster_compliance import position_code, position_bucket

        self.assertEqual(position_code(types.SimpleNamespace(position=Position.C)), "C")
        self.assertEqual(position_code(types.SimpleNamespace(position=Position.G)), "G")
        self.assertEqual(position_bucket(types.SimpleNamespace(position="LW")), "F")
        self.assertEqual(position_bucket(types.SimpleNamespace(position="LD")), "D")

    def test_buried_and_ir_excluded_from_active(self):
        from services.roster_compliance import summarize_team_roster_capacity

        active = [_player(f"a{i}", "C") for i in range(10)]
        active += [_player(f"d{i}", "D") for i in range(6)]
        active += [_player(f"g{i}", "G") for i in range(2)]
        buried = [_player("b1", "C", buried=True), _player("b2", "D", buried=True)]
        ir = [_player("ir1", "C", on_ir=True)]
        ltir = [_player("lt1", "D", on_ltir=True)]
        team = _team(active + buried + ir + ltir)
        cap = summarize_team_roster_capacity(team)
        self.assertEqual(cap["nhl_count"], 18)
        self.assertEqual(cap["forwards"], 10)
        self.assertEqual(cap["defense"], 6)
        self.assertEqual(cap["goalies"], 2)
        self.assertEqual(cap["buried_count"], 2)
        self.assertEqual(cap["ir_count"], 1)
        self.assertEqual(cap["ltir_count"], 1)
        self.assertEqual(cap["raw_roster_count"], 22)

    def test_standard_gate_blocks_under_minimums(self):
        from services.roster_compliance import evaluate_roster_compliance

        # 15 active: 9F / 5D / 1G — all below standard mins
        roster = [_player(f"f{i}", "C") for i in range(9)]
        roster += [_player(f"d{i}", "D") for i in range(5)]
        roster += [_player("g1", "G")]
        team = _team(roster)
        evaluation = evaluate_roster_compliance(
            team,
            cap_snap={"usable_cap_space_m": 5.0, "total_cap_hit_m": 80.0},
        )
        codes = {b["code"] for b in evaluation["blocking"]}
        self.assertIn("roster_min", codes)
        self.assertIn("forward_depth", codes)
        self.assertIn("defense_depth", codes)
        self.assertIn("goalie_depth", codes)
        self.assertFalse(evaluation["valid"])

    def test_standard_gate_passes_compliant_roster(self):
        from services.roster_compliance import evaluate_roster_compliance

        roster = [_player(f"f{i}", "C" if i % 3 == 0 else ("LW" if i % 3 == 1 else "RW")) for i in range(12)]
        roster += [_player(f"d{i}", "D") for i in range(6)]
        roster += [_player(f"g{i}", "G") for i in range(2)]
        # Extra buried must not inflate active count past 23 or block.
        roster += [_player(f"b{i}", "C", buried=True) for i in range(5)]
        team = _team(roster)
        evaluation = evaluate_roster_compliance(
            team,
            cap_snap={"usable_cap_space_m": 2.5, "total_cap_hit_m": 85.0},
        )
        self.assertEqual(evaluation["nhl_roster_count"], 20)
        self.assertTrue(evaluation["valid"], evaluation["blocking_reasons"])

    def test_cap_check_failure_blocks(self):
        from services.roster_compliance import evaluate_roster_compliance

        roster = [_player(f"f{i}", "C") for i in range(12)]
        roster += [_player(f"d{i}", "D") for i in range(6)]
        roster += [_player(f"g{i}", "G") for i in range(2)]
        team = _team(roster)
        evaluation = evaluate_roster_compliance(team, cap_error="snapshot exploded")
        codes = {b["code"] for b in evaluation["blocking"]}
        self.assertIn("cap_check_failed", codes)
        self.assertFalse(evaluation["valid"])

    def test_over_cap_blocks(self):
        from services.roster_compliance import evaluate_roster_compliance

        roster = [_player(f"f{i}", "C") for i in range(12)]
        roster += [_player(f"d{i}", "D") for i in range(6)]
        roster += [_player(f"g{i}", "G") for i in range(2)]
        team = _team(roster)
        evaluation = evaluate_roster_compliance(
            team,
            cap_snap={"usable_cap_space_m": -1.25, "total_cap_hit_m": 90.0},
        )
        codes = {b["code"] for b in evaluation["blocking"]}
        self.assertIn("cap_over", codes)

    def test_roster_max_uses_active_not_raw(self):
        from services.roster_compliance import evaluate_roster_compliance

        roster = [_player(f"f{i}", "C") for i in range(13)]
        roster += [_player(f"d{i}", "D") for i in range(7)]
        roster += [_player(f"g{i}", "G") for i in range(3)]  # 23 active
        roster += [_player(f"b{i}", "C", buried=True) for i in range(8)]  # raw 31
        team = _team(roster)
        evaluation = evaluate_roster_compliance(
            team,
            cap_snap={"usable_cap_space_m": 1.0, "total_cap_hit_m": 80.0},
        )
        self.assertEqual(evaluation["nhl_roster_count"], 23)
        self.assertNotIn("roster_max", {b["code"] for b in evaluation["blocking"]})


if __name__ == "__main__":
    unittest.main()
