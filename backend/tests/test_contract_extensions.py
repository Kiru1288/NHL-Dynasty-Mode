"""Year-end ledger, QO brackets, minor contracts, offer-sheet grid."""

from __future__ import annotations

import unittest
from types import SimpleNamespace


class TestYearEndLedger(unittest.TestCase):
    def test_schedule_a_games_played_bonus(self):
        from services.elc_year_end_ledger import evaluate_contract_performance_bonuses

        player = SimpleNamespace(
            id="p1",
            contract={
                "type": "ELC",
                "bonus_conditions": {
                    "schedule_a": [
                        {"id": "games_played", "label": "GP", "threshold": 40, "amount_m": 0.05},
                        {"id": "points", "label": "PTS", "threshold": 40, "amount_m": 0.05},
                    ],
                    "schedule_b": [],
                },
                "earned_bonuses_m": 0.0,
            },
        )
        session = SimpleNamespace(player_season_stats={"p1": {"gp": 55, "g": 12, "a": 20, "pts": 32}})
        res = evaluate_contract_performance_bonuses(session, player, season_year=2026)
        self.assertTrue(res["ok"])
        self.assertAlmostEqual(res["earned_m"], 0.05, places=3)
        self.assertEqual(len(res["items"]), 1)
        self.assertAlmostEqual(player.contract["earned_bonuses_m"], 0.05, places=3)

    def test_promise_honoured_raises_morale(self):
        from services.elc_year_end_ledger import evaluate_development_promise

        player = SimpleNamespace(
            id="p2",
            development_promise="ahl_featured",
            development_promise_season=2025,
            organizational_status="signed_ahl",
            development_path="AHL",
            psych=SimpleNamespace(morale=0.5),
            org_relationship=0.55,
        )
        session = SimpleNamespace(player_season_stats={"p2": {"gp": 50, "g": 10, "a": 20, "pts": 30}})
        res = evaluate_development_promise(session, player, season_year=2026)
        self.assertTrue(res["honoured"])
        self.assertGreater(player.psych.morale, 0.5)

    def test_midseason_promise_at_risk(self):
        from services.elc_year_end_ledger import evaluate_development_promise_midseason

        player = SimpleNamespace(
            id="p3",
            development_promise="top_six_track",
            development_promise_season=2025,
            organizational_status="signed_ahl",
            development_path="AHL",
            psych=SimpleNamespace(morale=0.55),
            org_relationship=0.55,
            development_promise_honoured=None,
            development_promise_result=None,
        )
        session = SimpleNamespace(player_season_stats={"p3": {"gp": 5, "g": 0, "a": 1, "pts": 1}})
        res = evaluate_development_promise_midseason(session, player, season_year=2026)
        self.assertFalse(res.get("skipped"))
        self.assertFalse(res["on_track"])
        self.assertLess(player.psych.morale, 0.55)

    def test_bonus_overage_from_reserve(self):
        from services.elc_year_end_ledger import apply_earned_bonuses_to_team_cap

        team = SimpleNamespace(performance_bonus_reserve_m=0.1)
        res = apply_earned_bonuses_to_team_cap(team, 0.25, season_year=2026)
        self.assertAlmostEqual(res["from_reserve_m"], 0.1, places=3)
        self.assertAlmostEqual(res["overage_m"], 0.15, places=3)
        self.assertAlmostEqual(team.performance_bonus_reserve_m, 0.0, places=3)
        self.assertTrue(len(team.bonus_overage) >= 1)


class TestQoAndOfferSheet(unittest.TestCase):
    def test_qo_brackets(self):
        from services.contract_economy import qualifying_offer_aav

        self.assertAlmostEqual(qualifying_offer_aav(0.9), round(0.9 * 1.10, 3))
        self.assertAlmostEqual(qualifying_offer_aav(1.2), round(1.2 * 1.05, 3))
        self.assertAlmostEqual(qualifying_offer_aav(2.0), 2.0)

    def test_offer_sheet_grid(self):
        from services.contract_economy import offer_sheet_compensation_tier

        self.assertEqual(offer_sheet_compensation_tier(1.0)["tier"], "none")
        self.assertEqual(offer_sheet_compensation_tier(1.5)["rounds"], [2])
        self.assertEqual(offer_sheet_compensation_tier(2.5)["rounds"], [2, 3])
        self.assertIn(1, offer_sheet_compensation_tier(5.0)["rounds"])
        self.assertEqual(len(offer_sheet_compensation_tier(9.0)["rounds"]), 4)


class TestMinorContracts(unittest.TestCase):
    def test_ahl_no_nhl_cap_or_slot(self):
        from services.contract_economy import (
            build_ahl_contract,
            does_contract_use_contract_slot,
            get_contract_cap_hit,
            sign_minor_or_tryout_contract,
        )

        c = build_ahl_contract(2026, aav_m=0.09)
        self.assertEqual(c["contract_type"], "AHL")
        self.assertAlmostEqual(get_contract_cap_hit(c), 0.0)
        self.assertFalse(does_contract_use_contract_slot(c))

        player = SimpleNamespace(id="m1", contract=None)
        team = SimpleNamespace(team_id="T", roster=[], prospect_pool=[], reserve_list=[])
        league = SimpleNamespace(free_agents=[], teams=[team])
        res = sign_minor_or_tryout_contract(
            player, team, league, 2026, {"contract_category": "ahl", "aav_m": 0.09, "years": 1}
        )
        self.assertTrue(res["ok"])
        self.assertFalse(res["uses_nhl_slot"])
        self.assertIn(player, team.prospect_pool)


if __name__ == "__main__":
    unittest.main()
