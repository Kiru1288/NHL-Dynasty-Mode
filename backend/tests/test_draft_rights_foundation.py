"""Tests for draft rights affiliation and reservation model."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


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

    def test_apply_draft_rights_stamps_pick_provenance(self):
        from services.draft_rights_engine import apply_draft_rights, rights_card_payload

        player = types.SimpleNamespace(id="p_prov", name="Prov Prospect", age=18, team_id="OHL_X")
        apply_draft_rights(
            player,
            nhl_team_id="VAN",
            draft_year=2026,
            pick_meta={
                "round": 1,
                "pick_in_round": 10,
                "overall_pick": 10,
                "pick_id": "2026-round1-OTT",
                "original_owner_team_id": "OTT",
                "is_traded": True,
            },
            block={"league_code": "OHL"},
            tm={"team_id": "OHL_X", "name": "X"},
            entry={"league_code": "OHL", "age": 18},
        )
        self.assertEqual(player.draft_pick_id, "2026-round1-OTT")
        self.assertEqual(player.draft_pick_original_team_id, "OTT")
        self.assertEqual(player.drafted_by, "VAN")
        self.assertTrue(player.draft_pick_was_traded)
        card = rights_card_payload(player)
        self.assertEqual(card.get("draft_pick_id"), "2026-round1-OTT")
        self.assertEqual(card.get("draft_pick_original_team_id"), "OTT")

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


class ProspectVisibilityTests(unittest.TestCase):
    """Drafted prospects must land somewhere the roster / trade surfaces read."""

    def _prospect(self, pid="p1", team="TOR"):
        return types.SimpleNamespace(
            id=pid,
            name="Pool Prospect",
            age=20,
            position="C",
            signed_status="unsigned",
            nhl_rights_team_id=team,
            rights_team_id=team,
            rights_status="exclusive_rights",
            development_path="OHL",
            retired=False,
        )

    def test_assign_ahl_blocked_for_unsigned_prospect(self):
        from services.draft_rights_engine import apply_prospect_rights_decision

        player = self._prospect()
        junior = {"team_id": "OHL_TEAM", "players": [player]}
        block = {"league_code": "OHL", "teams": [junior]}
        league = types.SimpleNamespace(development_leagues=[block], players_by_id={})
        team = types.SimpleNamespace(team_id="TOR", ahl_roster=[], reserve_list=[], prospect_pool=[player])
        session = types.SimpleNamespace(sim=types.SimpleNamespace(league=league))

        res = apply_prospect_rights_decision(session, player, team, "assign_ahl", season_year=2026)

        self.assertFalse(res["ok"])
        self.assertNotIn(player, team.ahl_roster)
        self.assertIn(player, junior["players"])

    def test_rerouting_off_ahl_clears_the_roster_spot(self):
        from services.draft_rights_engine import apply_prospect_rights_decision

        player = self._prospect()
        league = types.SimpleNamespace(development_leagues=[], players_by_id={})
        team = types.SimpleNamespace(team_id="TOR", ahl_roster=[player], reserve_list=[], prospect_pool=[player])
        session = types.SimpleNamespace(sim=types.SimpleNamespace(league=league))

        res = apply_prospect_rights_decision(session, player, team, "return_junior", season_year=2026)

        self.assertTrue(res["ok"], res.get("reason"))
        self.assertNotIn(player, team.ahl_roster)

    def test_unsigned_draft_rights_are_tradeable(self):
        from app.sim_engine.trades.trade_asset import (
            player_holds_nhl_spc,
            player_is_tradeable_draft_rights,
        )

        player = self._prospect()
        self.assertFalse(player_holds_nhl_spc(player))
        self.assertTrue(player_is_tradeable_draft_rights(player))

        player.rights_status = "rights_relinquished"
        self.assertFalse(player_is_tradeable_draft_rights(player))

    def test_traded_prospect_lands_in_acquiring_pool_with_rights(self):
        from app.sim_engine.trades.trade_asset import PlayerTradeAsset
        from app.sim_engine.trades.trade_executor import _apply_player_move

        player = self._prospect()
        src = types.SimpleNamespace(
            team_id="TOR",
            roster=[],
            ahl_roster=[],
            echl_roster=[],
            prospect_pool=[player],
            reserve_list=[{"player_id": "p1", "team_id": "TOR"}],
        )
        dst = types.SimpleNamespace(
            team_id="BOS", roster=[], ahl_roster=[], echl_roster=[], prospect_pool=[], reserve_list=[]
        )
        asset = PlayerTradeAsset(player_id="p1", source_team_id="TOR", acquiring_team_id="BOS")

        moved = []
        _apply_player_move(
            asset,
            {"TOR": src, "BOS": dst},
            season_label="2026-27",
            moved_players=moved,
            retained_records=[],
            context={},
        )

        self.assertIn(player, dst.prospect_pool)
        self.assertEqual(src.prospect_pool, [])
        self.assertEqual(player.nhl_rights_team_id, "BOS")
        self.assertEqual([r["player_id"] for r in dst.reserve_list], ["p1"])
        self.assertEqual(src.reserve_list, [])
        self.assertEqual(moved[0]["to_level"], "prospect")


class RosterCleanupPositionTests(unittest.TestCase):
    def test_position_enum_counts_toward_roster_check(self):
        from app.sim_engine.entities.player import Position
        from services.franchise_offseason import _position_code

        self.assertEqual(_position_code(types.SimpleNamespace(position=Position.C)), "C")
        self.assertEqual(_position_code(types.SimpleNamespace(position=Position.G)), "G")
        self.assertEqual(_position_code(types.SimpleNamespace(position="LW")), "LW")
        self.assertEqual(
            _position_code(types.SimpleNamespace(identity=types.SimpleNamespace(position=Position.D))),
            "D",
        )
        self.assertEqual(_position_code({"position": "RW"}), "RW")


class PickSlotValueTests(unittest.TestCase):
    def test_known_slots_are_not_valued_identically(self):
        from app.sim_engine.trades.trade_value import evaluate_pick_asset_value, slot_curve_value

        self.assertGreater(slot_curve_value(1), slot_curve_value(10))
        self.assertGreater(slot_curve_value(23), slot_curve_value(29) + 5.0)
        self.assertGreater(slot_curve_value(32), slot_curve_value(64))

        team = types.SimpleNamespace(team_id="T", id="T", gp=60, pts=55, w=24, l=30, otl=6, roster=[])
        ctx = {
            "season_year": 2026,
            "team_by_id": {"T": team},
            "known_pick_slots": {"2026-round1-A": 23, "2026-round1-B": 29},
        }
        early = evaluate_pick_asset_value(
            {"pick_id": "2026-round1-A", "year": 2026, "round": 1, "original_team_id": "T"},
            team, team, types.SimpleNamespace(), context=ctx,
        )
        late = evaluate_pick_asset_value(
            {"pick_id": "2026-round1-B", "year": 2026, "round": 1, "original_team_id": "T"},
            team, team, types.SimpleNamespace(), context=ctx,
        )
        self.assertGreater(float(early["total"]), float(late["total"]))
        self.assertEqual(early["value_debug"]["known_overall_slot"], 23)
        self.assertEqual(early["projected_slot"], 23)


class TradeWireCopyTests(unittest.TestCase):
    def test_picks_never_render_a_retained_salary_suffix(self):
        from services.franchise_sim import _cpu_trade_asset_lines

        ev = {
            "from_team_id": "WPG",
            "to_team_id": "SEA",
            "execution": {
                "moved_assets": [
                    {
                        "asset_type": "pick",
                        "asset_id": "2026-round1-WPG",
                        "year": 2026,
                        "round": 1,
                        "source_team_id": "WPG",
                        "acquiring_team_id": "SEA",
                    }
                ]
            },
        }
        to_lines, _from_lines = _cpu_trade_asset_lines(ev)
        self.assertEqual(to_lines, ["2026 Round 1"])
        self.assertNotIn("retained", to_lines[0])

    def test_player_retention_still_renders(self):
        from services.franchise_sim import _cpu_trade_asset_lines

        ev = {
            "from_team_id": "WPG",
            "to_team_id": "SEA",
            "execution": {
                "moved_assets": [
                    {
                        "asset_type": "player",
                        "asset_id": "p1",
                        "player_name": "Test Player",
                        "retained_pct": 50,
                        "source_team_id": "WPG",
                        "acquiring_team_id": "SEA",
                    }
                ]
            },
        }
        to_lines, _from_lines = _cpu_trade_asset_lines(ev)
        self.assertEqual(to_lines, ["Test Player (50% retained)"])


if __name__ == "__main__":
    unittest.main()
