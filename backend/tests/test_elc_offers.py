"""Structured ELC offer engine — legal terms, bonuses, acceptance, persistence."""

from __future__ import annotations

import unittest
from types import SimpleNamespace


def _make_player(**kwargs):
    defaults = dict(
        id="p1",
        age=19,
        overall=64,
        nhl_readiness=62,
        signed_status="unsigned",
        entry_level_contract_eligible=True,
        development_path="JUNIOR",
        rights_expiry_year=2028,
        elc_slide_eligible=True,
        elc_slide_years_remaining=1,
        org_relationship=0.55,
        nhl_eta=3,
        expected_role="Org prospect",
        draft_year=2025,
        draft_overall_pick=22,
        contract=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_team():
    return SimpleNamespace(
        team_id="T1",
        id="T1",
        roster=[],
        prospect_pool=[],
        reserve_list=[],
        name="Test",
    )


class TestElcOfferEngine(unittest.TestCase):
    def test_legal_terms_under_21_are_three_years(self):
        from services.elc_offer_engine import legal_elc_terms

        legal = legal_elc_terms(_make_player(age=18), 2026)
        self.assertTrue(legal["elc_eligible"])
        self.assertEqual(legal["legal_terms"], [3])
        self.assertEqual(legal["recommended_term"], 3)
        self.assertTrue(legal["slide_eligible"])

    def test_legal_terms_age_21_allows_two_or_three(self):
        from services.elc_offer_engine import legal_elc_terms

        legal = legal_elc_terms(_make_player(age=21), 2026)
        self.assertIn(2, legal["legal_terms"])
        self.assertIn(3, legal["legal_terms"])

    def test_templates_include_standard_and_max_bonus(self):
        from services.elc_offer_engine import list_offer_templates

        templates = list_offer_templates(_make_player(), 2026)
        ids = {t["template_id"] for t in templates}
        self.assertIn("standard_elc", ids)
        self.assertIn("maximum_bonus_elc", ids)
        self.assertIn("no_bonus_elc", ids)
        std = next(t for t in templates if t["template_id"] == "standard_elc")
        self.assertGreater(float(std["offer"]["signing_bonus_total_m"]), 0)
        self.assertGreater(float(std["offer"]["schedule_a_bonus_m"]), 0)

    def test_max_bonus_has_schedule_b(self):
        from services.elc_offer_engine import build_offer_from_template

        offer = build_offer_from_template(
            _make_player(), season_year=2026, template_id="maximum_bonus_elc"
        )
        self.assertGreater(float(offer["schedule_b_bonus_m"]), 0)
        self.assertTrue(offer["bonus_conditions"]["schedule_b"])

    def test_validate_rejects_illegal_term(self):
        from services.elc_offer_engine import build_offer_from_template, validate_offer

        offer = build_offer_from_template(
            _make_player(age=18), season_year=2026, template_id="standard_elc"
        )
        offer["term_years"] = 1
        result = validate_offer(_make_player(age=18), offer, 2026)
        self.assertFalse(result["allowed"])
        self.assertTrue(result["blocking_reasons"])

    def test_offer_to_contract_preserves_bonuses(self):
        from services.elc_offer_engine import build_offer_from_template, offer_to_contract_dict
        from services.contract_economy import normalize_contract_dict

        offer = build_offer_from_template(
            _make_player(), season_year=2026, template_id="maximum_bonus_elc"
        )
        contract = normalize_contract_dict(offer_to_contract_dict(offer, 2026))
        self.assertEqual(contract["contract_type"], "ELC")
        self.assertAlmostEqual(float(contract["aav_m"]), 0.95, places=2)
        self.assertGreater(float(contract["signing_bonus_m"]), 0)
        self.assertGreater(float(contract["schedule_a_bonus_m"]), 0)
        self.assertGreater(float(contract["schedule_b_bonus_m"]), 0)
        self.assertEqual(len(contract["nhl_salary_by_year_m"]), offer["term_years"])

    def test_acceptance_returns_probability_and_agent_wants(self):
        from services.elc_offer_engine import build_offer_from_template, evaluate_offer_acceptance

        player = _make_player(nhl_readiness=74, willingness_to_sign=True)
        team = _make_team()
        # Patch slots so acceptance is not blocked
        import services.contract_economy as ce

        original = ce.validate_contract_slots
        ce.validate_contract_slots = lambda *a, **k: {
            "ok": True,
            "contract_slots_used": 20,
            "contract_slots_limit": 50,
        }
        try:
            offer = build_offer_from_template(
                player, season_year=2026, template_id="maximum_bonus_elc"
            )
            decision = evaluate_offer_acceptance(player, team, offer, season_year=2026)
            self.assertIn("acceptance_probability", decision)
            self.assertIn("acceptance_pct", decision)
            self.assertTrue(decision["agent_wants"])
            self.assertIn(decision["decision"], ("accepted", "rejected", "countered", "considering"))
        finally:
            ce.validate_contract_slots = original

    def test_slide_extends_expiry_when_under_threshold(self):
        from services.elc_offer_engine import process_elc_slides

        player = _make_player(
            nhl_games_played_this_season=3,
            contract={
                "type": "ELC",
                "contract_type": "ELC",
                "slide_eligible": True,
                "slide_years_used": 0,
                "slide_games_threshold": 10,
                "years_remaining": 3,
                "expiry_year": 2029,
                "aav_m": 0.95,
            },
        )
        team = _make_team()
        team.prospect_pool = [player]
        session = SimpleNamespace(
            sim=SimpleNamespace(league=SimpleNamespace(teams=[team])),
            season_calendar_year=2026,
        )
        result = process_elc_slides(session, 2026)
        self.assertEqual(len(result["slid"]), 1)
        self.assertEqual(player.contract["expiry_year"], 2030)
        self.assertTrue(player.contract["slide_triggered"])


class TestContractAccessors(unittest.TestCase):
    def test_display_summary_reads_year_arrays(self):
        from services.contract_economy import get_contract_display_summary, normalize_contract_dict

        c = normalize_contract_dict({
            "type": "ELC",
            "aav_m": 0.95,
            "years": 3,
            "years_remaining": 3,
            "nhl_salary_by_year_m": [0.95, 0.95, 0.95],
            "minor_salary_by_year_m": [0.085, 0.085, 0.085],
            "signing_bonus_m": 0.2,
            "schedule_a_bonus_m": 0.1,
            "two_way": True,
            "is_entry_level": True,
        })
        summary = get_contract_display_summary(c, 2026)
        self.assertEqual(summary["type"], "ELC")
        self.assertAlmostEqual(summary["nhl_salary_m"], 0.95)
        self.assertAlmostEqual(summary["minor_salary_m"], 0.085)
        self.assertTrue(summary["is_entry_level"])


if __name__ == "__main__":
    unittest.main()
