"""Tests for draft rights affiliation and reservation model."""

from __future__ import annotations

import types
import unittest


class DraftRightsEngineTests(unittest.TestCase):
    def test_apply_draft_rights_keeps_development_affiliation(self):
        from services.draft_rights_engine import apply_draft_rights

        player = types.SimpleNamespace(
            id="p1",
            name="Test Prospect",
            age=18,
            team_id="OHL_TEAM",
        )
        block = {"league_code": "OHL"}
        tm = {"team_id": "OHL_TEAM", "name": "London"}
        rights = apply_draft_rights(
            player,
            nhl_team_id="TOR",
            draft_year=2026,
            pick_meta={"round": 1, "pick_in_round": 5, "overall_pick": 5},
            block=block,
            tm=tm,
            entry={"league_code": "OHL", "age": 18},
        )
        self.assertEqual(player.current_team_id, "OHL_TEAM")
        self.assertEqual(player.current_league_id, "OHL")
        self.assertEqual(player.team_id, "OHL_TEAM")
        self.assertEqual(player.nhl_rights_team_id, "TOR")
        self.assertEqual(player.rights_team_id, "TOR")
        self.assertEqual(player.rights_type, "chl_exclusive")
        self.assertEqual(player.rights_status, "exclusive_rights")
        self.assertEqual(player.signed_status, "unsigned")
        self.assertEqual(player.organizational_status, "unsigned_drafted")
        self.assertEqual(rights["rights_expiry_year"], 2028)

    def test_ncaa_and_europe_rights_types(self):
        from services.draft_rights_engine import build_draft_rights_fields

        ncaa = build_draft_rights_fields(team_id="BOS", draft_year=2026, entry={"league_code": "NCAA"})
        self.assertEqual(ncaa["rights_type"], "ncaa_college")
        self.assertEqual(ncaa["rights_status"], "college_rights")

        eu = build_draft_rights_fields(team_id="BOS", draft_year=2026, entry={"league_code": "EU_SHL"})
        self.assertEqual(eu["rights_type"], "european_exclusive")

    def test_assign_does_not_remove_from_dev_roster(self):
        from services.franchise_entry_draft import _assign_drafted_prospect

        player = types.SimpleNamespace(id="p2", name="Keep Me", age=19)
        tm = {"team_id": "NCAA_BU", "players": [player]}
        block = {"league_code": "NCAA"}
        team = types.SimpleNamespace(
            team_id="MTL",
            prospect_pool=[],
            prospects=[],
            reserve_list=[],
        )
        session = types.SimpleNamespace(
            team_by_id={"MTL": team},
            season_calendar_year=2025,
            draft_stock_history={},
            draft_results_archive=[],
            sim=types.SimpleNamespace(league=types.SimpleNamespace(development_leagues=[block], players_by_id={})),
        )
        block["teams"] = [tm]
        _assign_drafted_prospect(
            session,
            player,
            "MTL",
            {"round": 2, "pick_in_round": 3, "overall_pick": 35, "draft_year": 2026},
            block,
            tm,
            {"key": "p2", "league_code": "NCAA", "age": 19},
        )
        self.assertIn(player, tm["players"])
        self.assertEqual(player.team_id, "NCAA_BU")
        self.assertEqual(player.nhl_rights_team_id, "MTL")
        self.assertTrue(any(getattr(p, "id", None) == "p2" for p in team.prospect_pool))
        self.assertTrue(all("player_ref" not in e for e in team.reserve_list))

    def test_cpu_auto_sign_not_blanket(self):
        from services.draft_rights_engine import should_cpu_auto_sign_elc

        young = types.SimpleNamespace(
            signed_status="unsigned",
            entry_level_contract_eligible=True,
            rights_expiry_year=2030,
            rights_status="exclusive_rights",
            nhl_readiness=55,
            age=18,
            development_path="Junior",
        )
        team = types.SimpleNamespace(team_id="AAA")
        ok, reason = should_cpu_auto_sign_elc(young, team, season_year=2026, league=None)
        self.assertFalse(ok)
        self.assertEqual(reason, "no_clear_reason")

        expiring = types.SimpleNamespace(
            signed_status="unsigned",
            entry_level_contract_eligible=True,
            rights_expiry_year=2027,
            rights_status="rights_expiring",
            nhl_readiness=55,
            age=18,
            development_path="Junior",
        )
        ok2, reason2 = should_cpu_auto_sign_elc(expiring, team, season_year=2026, league=None)
        self.assertTrue(ok2)
        self.assertEqual(reason2, "rights_expiration_risk")

    def test_reserve_list_ids_only(self):
        from services.contract_economy import add_to_reserve_list

        team = types.SimpleNamespace(team_id="CGY", reserve_list=[])
        player = types.SimpleNamespace(
            id="p9",
            name="Reserve Only",
            position="C",
            signed_status="unsigned",
            nhl_rights_team_id="CGY",
            rights_status="exclusive_rights",
            rights_type="chl_exclusive",
            rights_expiry_year=2028,
            current_team_id="OHL_X",
            current_league_id="OHL",
            organizational_status="unsigned_drafted",
        )
        entry = add_to_reserve_list(team, player, draft_year=2026, draft_overall=12, added_season=2025)
        self.assertEqual(entry["player_id"], "p9")
        self.assertNotIn("player_ref", entry)
        self.assertEqual(entry["rights_expiry_year"], 2028)


if __name__ == "__main__":
    unittest.main()
