"""Staggered free-agency decision market."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch


class TestFaMarketDecisions(unittest.TestCase):
    def _player(self, pid, ovr=78, age=28, pos="C", ask=3.0):
        return SimpleNamespace(
            id=pid,
            name=f"Player {pid}",
            overall=ovr,
            age=age,
            position=pos,
            asking_aav_m=ask,
            ask_aav_m=ask,
            contract=None,
        )

    def test_opening_day_does_not_mass_sign(self):
        from services.fa_market_engine import ensure_fa_market_book, tick_free_agency_market

        players = [self._player(f"p{i}", ovr=75 + (i % 10), ask=1.5 + i * 0.1) for i in range(20)]
        team = SimpleNamespace(team_id="CPU1", id="CPU1", roster=[], prospect_pool=[])
        league = SimpleNamespace(free_agents=list(players), teams=[team])
        rng = __import__("random").Random(1)
        session = SimpleNamespace(
            sim=SimpleNamespace(league=league, rng=rng),
            user_team_id="USER",
            season_calendar_year=2026,
            fa_market_book={},
            fa_market_day=0,
            cpu_fa_signings={},
            cpu_fa_wave=0,
        )

        with patch("services.fa_market_engine.evaluate_team_position_needs") as needs, patch(
            "services.fa_market_engine.score_free_agent_fit", return_value=(0.2, [])
        ), patch(
            "services.fa_market_engine.compute_fair_aav", return_value=2.0
        ), patch(
            "services.fa_market_engine.generate_contract_terms", return_value=(None, 2, {})
        ), patch(
            "services.fa_market_engine.cpu_signing_blocked", return_value=False
        ):
            needs.return_value = {
                "cap_space_m": 20.0,
                "slots_remaining": 10,
                "need_score": {"C": 0.8, "W": 0.5, "D": 0.4, "G": 0.2},
                "window": "contender",
            }
            ensure_fa_market_book(session)
            tick = tick_free_agency_market(
                session, days=1, opening_day=True, max_signings_per_day=2, max_offers_per_day=8
            )

        self.assertEqual(tick["day"], 1)
        self.assertLessEqual(len(tick["signings"]), 2)
        book = session.fa_market_book
        states = {e["state"] for e in book["entries"].values()}
        # Most players should still be deciding, not signed en masse
        self.assertTrue(any(s != "signed" for s in states))
        awaitingish = sum(
            1
            for e in book["entries"].values()
            if e["state"] in ("awaiting_offers", "gauging_market", "evaluating_offers", "holding_out")
        )
        # Opening day should leave most of the board undecided (not a mass rush).
        self.assertGreaterEqual(awaitingish, 7)
        self.assertLessEqual(
            sum(1 for e in book["entries"].values() if e["state"] == "signed"),
            2,
        )

    def test_patience_and_urgency_drive_lean(self):
        from services.fa_market_engine import STATE_LEANING, _update_player_decision

        entry = {
            "patience": 0.2,
            "days_on_market": 10,
            "offers": [{"team_id": "T", "aav_m": 2.0, "years": 2, "day": 1}],
            "best_offer_m": 2.0,
            "ask_aav_m": 2.1,
            "fair_aav_m": 2.0,
            "tier": "depth",
            "age": 34,
            "offer_count": 1,
            "position": "C",
            "overall": 74,
            "state": "evaluating_offers",
        }
        book = {"peer_signings": []}
        _update_player_decision(entry, book, day=10)
        self.assertEqual(entry["state"], STATE_LEANING)

    def test_star_holds_without_close_offer(self):
        from services.fa_market_engine import STATE_HOLDING, _update_player_decision

        entry = {
            "patience": 0.9,
            "days_on_market": 3,
            "offers": [{"team_id": "T", "aav_m": 6.0, "years": 5, "day": 1}],
            "best_offer_m": 6.0,
            "ask_aav_m": 9.0,
            "fair_aav_m": 8.5,
            "tier": "star",
            "age": 27,
            "offer_count": 1,
            "position": "C",
            "overall": 90,
            "state": "evaluating_offers",
        }
        book = {"peer_signings": []}
        _update_player_decision(entry, book, day=3)
        self.assertIn(entry["state"], (STATE_HOLDING, "evaluating_offers", "gauging_market"))

    def test_fringe_player_pursuit_requires_need(self):
        """55-OVR players must not receive league-wide bidding wars."""
        from services.fa_market_engine import _collect_cpu_offers, ensure_fa_market_book

        fringe = self._player("fringe1", ovr=55, age=29, pos="LW", ask=0.9)
        star = self._player("star1", ovr=91, age=27, pos="C", ask=10.0)
        teams = []
        for i in range(8):
            teams.append(
                SimpleNamespace(
                    team_id=f"T{i}",
                    id=f"T{i}",
                    name=f"Team {i}",
                    abbreviation=f"T{i}",
                    roster=[],
                    prospect_pool=[],
                    gm_window="contender",
                )
            )
        league = SimpleNamespace(
            free_agents=[fringe, star],
            overseas_free_agents=[],
            teams=teams,
            salary_cap_m=95.0,
        )
        rng = __import__("random").Random(7)
        session = SimpleNamespace(
            sim=SimpleNamespace(league=league, rng=rng),
            user_team_id="USER",
            season_calendar_year=2026,
            fa_market_book={},
            fa_market_day=0,
            cpu_fa_signings={},
            cpu_fa_wave=0,
        )

        with patch("services.fa_market_engine.evaluate_team_position_needs") as needs, patch(
            "services.fa_market_engine.score_free_agent_fit", return_value=(0.55, ["need"])
        ), patch(
            "services.fa_market_engine.compute_fair_aav",
            side_effect=lambda p, *a, **k: 10.0 if _player_ovr_safe(p) >= 88 else 0.85,
        ), patch(
            "services.fa_market_engine.generate_contract_terms", return_value=(None, 2, {})
        ), patch(
            "services.fa_market_engine.cpu_signing_blocked", return_value=None
        ), patch(
            "services.fa_market_engine.sync_all_team_cap_fields", return_value=0
        ), patch(
            "services.fa_market_engine._serious_cpu_offer_aav",
            side_effect=lambda **kw: round(float(kw.get("fair") or 1) * 0.95, 3),
        ):
            # Overloaded wing depth — no need for fringe LW.
            needs.return_value = {
                "cap_space_m": 18.0,
                "spendable_cap_space_m": 15.0,
                "slots_remaining": 8,
                "need_score": {"C": 0.7, "LW": 0.05, "RW": 0.2, "LD": 0.2, "RD": 0.2, "G": 0.1},
                "counts": {"C": 3, "LW": 5, "RW": 4, "LD": 3, "RD": 3, "G": 2},
                "best_ovr": {"C": 84, "LW": 80, "RW": 79, "LD": 81, "RD": 80, "G": 82},
                "overload": {"C": False, "LW": True, "RW": False, "LD": False, "RD": False, "G": False},
                "window": "contender",
            }
            ensure_fa_market_book(session)
            offers = _collect_cpu_offers(session, max_new_offers=40)

        fringe_offers = [o for o in offers if o.get("player_id") == "fringe1"]
        star_offers = [o for o in offers if o.get("player_id") == "star1"]
        self.assertEqual(len(fringe_offers), 0, f"fringe bidding war: {fringe_offers}")
        self.assertGreaterEqual(len(star_offers), 1, "elite player received no serious offers")

    def test_ask_decays_for_unsigned_depth(self):
        from services.fa_market_engine import _ask_for_player
        from services.contract_economy import LEAGUE_MINIMUM_AAV_M

        p = self._player("d1", ovr=62, age=31, pos="C", ask=1.8)
        day0 = _ask_for_player(p, None, days_on_market=0, offer_count=0)
        day20 = _ask_for_player(p, None, days_on_market=20, offer_count=0)
        self.assertLess(day20, day0)
        self.assertLessEqual(day20, LEAGUE_MINIMUM_AAV_M + 0.35)

    def test_september_user_signing_not_stage_gated(self):
        """Eligible UFAs remain signable after free_agency_open clears (preseason)."""
        from services.contract_economy import (
            LEAGUE_MINIMUM_AAV_M,
            normalize_contract_dict,
            sign_player_to_team,
        )

        fa = self._player("ufa1", ovr=74, age=28, pos="C", ask=1.2)
        fa.rights_status = "UFA"
        fa.contract = None
        roster = [
            SimpleNamespace(
                id=f"r{i}",
                overall=78,
                age=26,
                position="C",
                ovr=lambda: 0.78,
                ratings={"dev_potential": 78},
                season_stats={"pts": 20},
                contract=normalize_contract_dict(
                    {"aav_m": 1.0, "cap_hit_m": 1.0, "years_remaining": 2}
                ),
                identity=SimpleNamespace(age=26, position="C", shoots="L", name=f"R{i}"),
                retired=False,
            )
            for i in range(4)
        ]
        for pl in roster:
            for flag in ("is_buried", "buried", "in_minors", "on_ir", "on_ltir"):
                setattr(pl, flag, False)
        team = SimpleNamespace(
            team_id="USER",
            id="USER",
            roster=roster,
            rfa_rights=[],
            buyout_cap_hits=[],
            salary_cap_m=95.0,
            name="User",
        )
        league = SimpleNamespace(
            teams=[team],
            free_agents=[fa],
            overseas_free_agents=[],
            salary_cap_m=95.0,
            cap_floor_m=70.0,
        )
        session = SimpleNamespace(
            sim=SimpleNamespace(league=league, rng=__import__("random").Random(9)),
            user_team_id="USER",
            team_by_id={"USER": team},
            season_calendar_year=2026,
            phase="preseason",
            free_agency_open=False,
            offseason_stage="preseason_start",
        )
        result = sign_player_to_team(
            fa,
            team,
            league,
            2026,
            {
                "aav_m": max(LEAGUE_MINIMUM_AAV_M, 1.5),
                "years": 1,
                "context": "ufa",
                "force": True,
                "_session": session,
            },
        )
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("status"), "accepted")


def _player_ovr_safe(p):
    from services.contract_economy import _player_ovr

    return float(_player_ovr(p))


if __name__ == "__main__":
    unittest.main()
