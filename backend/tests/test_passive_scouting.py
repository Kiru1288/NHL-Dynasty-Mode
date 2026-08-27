"""Passive dedicated-file growth for assigned scouts and regional deployments."""

import unittest
from unittest.mock import MagicMock, patch

from services.franchise_scouting import (
    PASSIVE_ASSIGNED_DAILY,
    apply_passive_scouting_progress,
    apply_scouting_command,
)
from services.franchise_session import FranchiseSession


def _minimal_session(**kwargs) -> FranchiseSession:
    sim = MagicMock()
    sim.league = MagicMock()
    sim.league.development_leagues = []
    sim.league.teams = []
    sim.rng = MagicMock()

    session = FranchiseSession(
        session_id="test-passive-scout",
        sim=sim,
        user_team_id="T1",
        head_coach_name="Coach",
        coach_archetype="balanced",
        team_ids=["T1"],
        season_calendar_year=2025,
        calendar_cursor=120,
        phase="regular",
        **kwargs,
    )
    session.scouting_state = {
        "budget": 2_500_000,
        "used_budget": 0.0,
        "prospects": {},
        "watchlist": [],
        "assignments": [],
        "active_deployments": [],
    }
    return session


class PassiveScoutingTests(unittest.TestCase):
    def test_assigned_scout_gains_passive_coverage_daily(self):
        session = _minimal_session()
        board = {
            "entries": [
                {
                    "key": "P1",
                    "name": "Test Prospect",
                    "rank": 40,
                    "country": "Canada",
                    "region": "North America",
                    "scouting_confidence": 62,
                }
            ]
        }
        session.scouting_state["prospects"]["P1"] = {
            "assigned_scout": "Regional Scout NA",
            "scouted_percentage": 5.0,
        }

        with patch("services.franchise_scouting._draft_entries", return_value=board["entries"]):
            with patch("services.franchise_scouting.invalidate_session_payload_caches"):
                changed = apply_passive_scouting_progress(session)

        self.assertTrue(changed)
        pct = session.scouting_state["prospects"]["P1"]["scouted_percentage"]
        self.assertGreater(pct, 5.0)
        self.assertAlmostEqual(pct, 5.0 + PASSIVE_ASSIGNED_DAILY, places=1)

    def test_assign_meta_kickstarts_dedicated_file(self):
        session = _minimal_session()
        board = {
            "entries": [
                {
                    "key": "P2",
                    "name": "Another",
                    "rank": 12,
                    "country": "Sweden",
                    "region": "Scandinavia",
                    "scouting_confidence": 70,
                }
            ]
        }
        with patch("services.franchise_scouting._draft_entries", return_value=board["entries"]):
            with patch("services.franchise_scouting.invalidate_session_payload_caches"):
                result = apply_scouting_command(
                    session,
                    {
                        "meta_only": True,
                        "prospect_id": "P2",
                        "meta_patch": {"assigned_scout": "Regional Scout EU"},
                    },
                    "focus",
                )

        self.assertTrue(result.get("ok"))
        overlay = session.scouting_state["prospects"]["P2"]
        self.assertEqual(overlay.get("assigned_scout"), "Regional Scout EU")
        self.assertGreaterEqual(float(overlay.get("scouted_percentage") or 0), 3.0)

    def test_region_sweep_registers_active_deployment(self):
        session = _minimal_session()
        entries = [
            {"key": "P3", "name": "Canadian", "rank": 80, "country": "Canada", "region": "North America"},
            {"key": "P4", "name": "American", "rank": 90, "country": "United States", "region": "North America"},
        ]
        with patch("services.franchise_scouting._draft_entries", return_value=entries):
            with patch("services.franchise_scouting.invalidate_session_payload_caches"):
                result = apply_scouting_command(
                    session,
                    {
                        "action": "region_sweep",
                        "target_type": "country",
                        "target_id": "canada",
                        "country_id": "canada",
                        "intensity": "normal",
                    },
                    "focus",
                )

        self.assertTrue(result.get("ok"))
        deployments = session.scouting_state.get("active_deployments") or []
        self.assertTrue(any(d.get("kind") == "country" and "canada" in str(d.get("id")).lower() for d in deployments))
        self.assertGreaterEqual(
            float(session.scouting_state["prospects"]["P3"]["scouted_percentage"]),
            6.0,
        )


if __name__ == "__main__":
    unittest.main()
