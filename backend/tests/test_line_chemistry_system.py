"""End-to-end tests for line/pair chemistry report + roster serialization."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.sim_engine.systems.chemistry import (  # noqa: E402
    _canonical_player_id,
    _canonical_player_id_from_string,
    _pair_index_key,
    _resolve_line_units_from_payload,
    build_pair_index,
    build_public_chemistry_report,
    calculate_forward_line_chemistry,
    calculate_pair_chemistry,
    calculate_team_chemistry_report,
    ensure_player_chemistry_profile,
)
from services import franchise_sim  # noqa: E402


def _player(
    pid: str,
    *,
    name: str,
    position: str = "C",
    personality: str = "balanced",
    playstyle: str = "balanced",
    morale: float = 0.72,
    confidence: float = 0.68,
    familiarity_with: dict | None = None,
) -> types.SimpleNamespace:
    prof = {
        "personality": personality,
        "playstyle": playstyle,
        "ego": 48,
        "adaptability": 62,
        "coachability": 64,
        "defensive_buy_in": 58,
        "leadership": 55,
        "temperament": 60,
        "competitiveness": 66,
        "loyalty": 58,
        "work_ethic": 61,
        "pressure_response": 63,
        "room_presence": 57,
        "_materialized": True,
    }
    p = types.SimpleNamespace(
        id=pid,
        player_id=pid,
        identity=types.SimpleNamespace(name=name, position=position, age=26),
        position=position,
        psych=types.SimpleNamespace(
            morale=morale,
            confidence=confidence,
            role_satisfaction=0.70,
            coach_trust=0.66,
        ),
        traits=types.SimpleNamespace(
            ego=0.45,
            competitiveness=0.70,
            coachability=0.64,
            work_ethic=0.61,
            leadership=0.55,
            volatility=0.35,
            adaptability=0.62,
            loyalty=0.58,
        ),
        chemistry_profile=dict(prof),
        chemistry_relationships=dict(familiarity_with or {}),
        retired=False,
    )
    ensure_player_chemistry_profile(p)
    return p


class TestLineChemistrySystem(unittest.TestCase):
    def test_canonical_player_id_normalizes_numeric_ids(self):
        self.assertEqual(_canonical_player_id_from_string("8471214"), "NHL_8471214")
        self.assertEqual(_canonical_player_id_from_string("NHL_8471214"), "NHL_8471214")
        p = _player("8471214", name="Test")
        self.assertEqual(_canonical_player_id(p), "NHL_8471214")

    def test_pair_scores_span_wide_range(self):
        playmaker = _player("NHL_1", name="PM", position="C", personality="leader", playstyle="playmaker", morale=0.88, confidence=0.86)
        sniper = _player(
            "NHL_2",
            name="SN",
            position="RW",
            personality="young_skilled",
            playstyle="sniper",
            morale=0.84,
            confidence=0.82,
            familiarity_with={"NHL_1": 92.0},
        )
        ego_a = _player("NHL_3", name="Star A", position="LW", personality="high_ego_star", playstyle="sniper", morale=0.42, confidence=0.40)
        ego_b = _player("NHL_4", name="Star B", position="C", personality="high_ego_star", playstyle="sniper", morale=0.38, confidence=0.36)

        good = calculate_pair_chemistry(playmaker, sniper, context={"slot_a": "C", "slot_b": "RW"})
        bad = calculate_pair_chemistry(ego_a, ego_b, context={"slot_a": "LW", "slot_b": "C"})

        self.assertGreaterEqual(good["chemistry"], 75, good)
        self.assertGreaterEqual(good["familiarity"], 85, good)
        self.assertLessEqual(bad["chemistry"], 58, bad)
        self.assertGreater(good["chemistry"] - bad["chemistry"], 12)

    def test_forward_line_report_exposes_pair_links(self):
        line = [
            _player("NHL_10", name="LW", position="LW", playstyle="sniper"),
            _player("NHL_11", name="C", position="C", playstyle="playmaker"),
            _player("NHL_12", name="RW", position="RW", playstyle="power_forward"),
        ]
        report = calculate_forward_line_chemistry(line)
        self.assertEqual(len(report.get("pair_links") or []), 3)
        keys = {
            _pair_index_key(row.get("player_a_id"), row.get("player_b_id"))
            for row in report["pair_links"]
        }
        self.assertIn(_pair_index_key("NHL_10", "NHL_11"), keys)
        self.assertIn(_pair_index_key("NHL_11", "NHL_12"), keys)
        self.assertIn(_pair_index_key("NHL_10", "NHL_12"), keys)

    def test_saved_lines_resolve_with_numeric_slot_ids(self):
        lw = _player("NHL_100", name="Left", position="LW")
        c = _player("NHL_101", name="Center", position="C")
        rw = _player("NHL_102", name="Right", position="RW")
        team = types.SimpleNamespace(roster=[lw, c, rw])
        payload = {
            "forwards": [
                {
                    "id": "f1",
                    "slots": {"LW": "100", "C": "101", "RW": "102"},
                }
            ],
            "defense": [],
            "goalies": [],
        }
        resolved = _resolve_line_units_from_payload(team, payload)
        self.assertEqual(len(resolved["forward_lines"]), 1)
        self.assertEqual(len(resolved["forward_lines"][0]), 3)
        self.assertEqual(_canonical_player_id(resolved["forward_lines"][0][0]), "NHL_100")

    def test_team_report_builds_pair_index_for_saved_lines(self):
        lw = _player("NHL_201", name="Zed", position="LW", playstyle="sniper")
        c = _player("NHL_202", name="Coz", position="C", playstyle="playmaker", familiarity_with={"NHL_201": 80.0})
        rw = _player("NHL_203", name="Gir", position="RW", playstyle="power_forward")
        ld = _player("NHL_204", name="LD", position="LD", playstyle="shutdown")
        rd = _player("NHL_205", name="RD", position="RD", playstyle="puck_mover")
        team = types.SimpleNamespace(roster=[lw, c, rw, ld, rd], coach=types.SimpleNamespace(system="balanced"))
        session = types.SimpleNamespace(
            user_team_id="ott",
            team_by_id={"ott": team},
            lines={
                "even_strength": {
                    "lines": {
                        "forwards": [
                            {"id": "f1", "slots": {"LW": "NHL_201", "C": "NHL_202", "RW": "NHL_203"}},
                            {"id": "f2", "slots": {"LW": "", "C": "", "RW": ""}},
                            {"id": "f3", "slots": {"LW": "", "C": "", "RW": ""}},
                            {"id": "f4", "slots": {"LW": "", "C": "", "RW": ""}},
                        ],
                        "defense": [
                            {"id": "d1", "slots": {"LD": "NHL_204", "RD": "NHL_205"}},
                        ],
                        "goalies": [{"id": "g1", "slots": {"Starter": "", "Backup": ""}}],
                    }
                }
            },
            chaos_index=0.35,
            storyline_events=[],
            nhl_today={"iso": "2026-01-01"},
        )
        rep = calculate_team_chemistry_report(team, session=session)
        public = build_public_chemistry_report(session)

        self.assertTrue(rep.get("pair_index"))
        self.assertTrue(public.get("pair_index"))
        key = _pair_index_key("NHL_201", "NHL_202")
        self.assertIn(key, rep["pair_index"])
        self.assertIn(key, public["pair_index"])
        scores = sorted({row["chemistry"] for row in rep["pair_index"].values()})
        self.assertGreaterEqual(len(scores), 2, scores)
        self.assertGreaterEqual(max(scores) - min(scores), 3, scores)
        self.assertNotEqual(scores[0], 65)

    def test_roster_serialization_includes_chemistry_contract(self):
        a = _player("NHL_301", name="A", position="LW", playstyle="playmaker", personality="leader")
        b = _player("NHL_302", name="B", position="C", playstyle="sniper", personality="young_skilled")
        a.chemistry_relationships = {"NHL_302": 77.0}
        row = franchise_sim._serialize_player_row(a, include_ratings=True, session=None, _team=None)
        self.assertEqual(row.get("id"), "NHL_301")
        self.assertEqual(row.get("player_id"), "NHL_301")
        self.assertEqual(row.get("personality"), "leader")
        self.assertEqual(row.get("playstyle"), "playmaker")
        prof = row.get("chemistry_profile") or {}
        self.assertEqual(prof.get("personality"), "leader")
        self.assertEqual(prof.get("playstyle"), "playmaker")
        self.assertEqual((row.get("chemistry_relationships") or {}).get("NHL_302"), 77.0)

    def test_pair_index_lookup_is_order_independent(self):
        rows = [
            {"player_a_id": "NHL_9", "player_b_id": "NHL_8", "chemistry": 71},
            {"player_a_id": "NHL_2", "player_b_id": "NHL_1", "chemistry": 83},
        ]
        index = build_pair_index(rows)
        self.assertEqual(index[_pair_index_key("NHL_8", "NHL_9")]["chemistry"], 71)
        self.assertEqual(index[_pair_index_key("NHL_1", "NHL_2")]["chemistry"], 83)

    def test_empty_chemistry_profile_infers_from_archetype(self):
        p = types.SimpleNamespace(
            id="NHL_8482116",
            archetype="PLAYMAKER",
            identity=types.SimpleNamespace(name="Playmaker", position="C", age=24),
            traits=types.SimpleNamespace(
                leadership=0.7,
                ego=0.45,
                work_ethic=0.6,
                volatility=0.3,
                adaptability=0.62,
                competitiveness=0.7,
                loyalty=0.55,
                coachability=0.64,
            ),
            psych=types.SimpleNamespace(morale=0.72, confidence=0.7, role_satisfaction=0.68),
            chemistry_profile={},
            chemistry_relationships={},
            retired=False,
        )
        prof = ensure_player_chemistry_profile(p)
        self.assertEqual(prof.get("playstyle"), "playmaker")
        self.assertTrue(prof.get("_materialized"))
        self.assertNotEqual(prof.get("personality"), "balanced")

    def test_stub_two_way_profile_is_rebuilt_from_archetype(self):
        p = types.SimpleNamespace(
            id="NHL_8480208",
            archetype="SNIPER",
            identity=types.SimpleNamespace(name="Sniper", position="RW", age=26),
            traits=types.SimpleNamespace(
                leadership=0.4, ego=0.5, work_ethic=0.55, volatility=0.3,
                adaptability=0.5, competitiveness=0.6, loyalty=0.5, coachability=0.5,
            ),
            psych=types.SimpleNamespace(morale=0.5, confidence=0.5, role_satisfaction=0.5),
            chemistry_profile={"personality": "balanced", "playstyle": "two_way", "_materialized": True},
            chemistry_relationships={},
            retired=False,
        )
        prof = ensure_player_chemistry_profile(p)
        self.assertEqual(prof.get("playstyle"), "sniper")


if __name__ == "__main__":
    unittest.main()
