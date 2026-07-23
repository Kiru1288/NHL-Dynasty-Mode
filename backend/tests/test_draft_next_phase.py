"""Save/load and shared draft engine coverage for mid-draft restoration."""

from __future__ import annotations

import types
import unittest


class DraftSnapshotTests(unittest.TestCase):
    def _session(self):
        p1 = types.SimpleNamespace(
            id="p1",
            name="One",
            drafted=True,
            current_team_id="OHL_A",
            current_league_id="OHL",
            nhl_rights_team_id="TOR",
            rights_status="exclusive_rights",
            rights_type="chl_exclusive",
            rights_expiry_year=2028,
            organizational_status="unsigned_drafted",
            signed_status="unsigned",
            draft_overall_pick=3,
            development_path="Junior",
            elc_slide_eligible=True,
            elc_slide_years_remaining=1,
        )
        team = types.SimpleNamespace(
            team_id="TOR",
            prospect_pool=[p1],
            reserve_list=[{
                "player_id": "p1",
                "rights_team_id": "TOR",
                "signed_status": "unsigned",
                "rights_expiry_year": 2028,
            }],
        )
        league = types.SimpleNamespace(
            players_by_id={"p1": p1},
            development_leagues=[{
                "league_code": "OHL",
                "teams": [{"team_id": "OHL_A", "players": [p1]}],
            }],
            draft_pick_registry={
                "2026-R1-BOS": {
                    "pick_id": "2026-R1-BOS",
                    "current_owner_team_id": "TOR",
                    "original_team_id": "BOS",
                    "resolved": False,
                },
                "2026-R1-TOR": {
                    "pick_id": "2026-R1-TOR",
                    "current_owner_team_id": "TOR",
                    "original_team_id": "TOR",
                    "resolved": True,
                    "selected_prospect_id": "p1",
                },
            },
            teams=[team],
        )
        session = types.SimpleNamespace(
            draft_state={
                "draft_started": True,
                "draft_completed": False,
                "overall_pick": 4,
                "current_team_id": "TOR",
                "is_user_pick": True,
                "drafted_prospect_ids": ["p1"],
                "completed_picks": [{"overall_pick": 3, "prospect_id": "p1", "team_id": "TOR"}],
                "draft_order": [
                    {"overall_pick": 3, "team_id": "TOR", "pick_id": "2026-R1-TOR", "resolved": True},
                    {"overall_pick": 4, "team_id": "TOR", "pick_id": "2026-R1-BOS", "original_owner_team_id": "BOS"},
                ],
                "trade_offers": [{"from_team_id": "MTL", "offer_text": "test"}],
            },
            draft_completed=False,
            draft_payload={"stale": True},
            team_by_id={"TOR": team},
            user_team_id="TOR",
            season_calendar_year=2025,
            sim=types.SimpleNamespace(league=league),
        )
        return session, p1, league

    def test_snapshot_and_restore_user_pick_moment(self):
        from services.draft_state_snapshot import (
            assert_draft_restored_identically,
            restore_draft_moment,
            snapshot_draft_moment,
        )

        session, p1, league = self._session()
        snap = snapshot_draft_moment(session)
        self.assertEqual(snap["moment"], "user_pick")
        self.assertEqual(len(snap["drafted_rights"]), 1)
        self.assertTrue(snap["pending_trade_offers"])

        # Mutate live state then restore
        session.draft_state["overall_pick"] = 99
        session.draft_state["drafted_prospect_ids"] = []
        p1.nhl_rights_team_id = None
        league.draft_pick_registry.clear()

        out = restore_draft_moment(session, snap)
        self.assertTrue(out["ok"])
        assert_draft_restored_identically(snap, session)
        self.assertEqual(p1.nhl_rights_team_id, "TOR")
        self.assertIn("2026-R1-BOS", league.draft_pick_registry)

    def test_board_scouting_uncertainty_differs_by_team(self):
        from services.draft_board_engine import team_scouting_estimate

        a = team_scouting_estimate(
            team_id="AAA", prospect_id="p9", true_ovr=70, true_potential=85, public_rank=5, scouting_quality=55
        )
        b = team_scouting_estimate(
            team_id="BBB", prospect_id="p9", true_ovr=70, true_potential=85, public_rank=5, scouting_quality=90
        )
        self.assertNotEqual(a["scouted_overall_estimate"], b["scouted_overall_estimate"])
        self.assertGreater(b["scouting_confidence"], a["scouting_confidence"])

    def test_signing_decision_can_decline_ncaa(self):
        from services.draft_signing_engine import evaluate_elc_signing_decision

        player = types.SimpleNamespace(
            id="n1",
            age=19,
            development_path="NCAA",
            nhl_readiness=60,
            rights_expiry_year=2030,
            ncaa_commitment=True,
            willingness_to_sign=None,
            org_relationship=0.5,
            elc_slide_eligible=True,
            elc_slide_years_remaining=1,
        )
        team = types.SimpleNamespace(team_id="BOS")
        decision = evaluate_elc_signing_decision(player, team, season_year=2026)
        self.assertFalse(decision["accepted"])

    def test_unsigned_development_tick(self):
        from services.unsigned_prospect_development import develop_unsigned_prospect

        player = types.SimpleNamespace(
            id="d1",
            age=18,
            signed_status="unsigned",
            development_path="Junior",
            current_league_id="OHL",
            overall=62.0,
            potential_score=82.0,
            nhl_readiness=58.0,
            ice_time_quality=0.7,
            coaching_quality=0.6,
            ppg=0.9,
            elc_slide_eligible=True,
            elc_slide_years_remaining=1,
        )
        res = develop_unsigned_prospect(player, season_year=2026)
        self.assertTrue(res["ok"])
        self.assertGreater(player.overall, 62.0)


if __name__ == "__main__":
    unittest.main()
