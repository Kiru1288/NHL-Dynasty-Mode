"""Trade market overhaul regressions — uncapped TV, demands, fairness knobs."""

from __future__ import annotations

import random
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _player(pid, ovr=80, age=27, pos="C", **extra):
    p = types.SimpleNamespace(
        id=pid,
        name=f"P {pid}",
        identity=types.SimpleNamespace(name=f"P {pid}", age=age, position=pos),
        position=pos,
        age=age,
        ovr=lambda: ovr / 99.0 if ovr > 1.5 else ovr,
        overall=ovr,
        contract=types.SimpleNamespace(
            aav_m=4.0,
            cap_hit_m=4.0,
            years_remaining=3,
            type="STANDARD",
            is_nhl_spc=True,
            clause="",
            trade_list_size=0,
        ),
        retired=False,
        signed_status="signed",
        ratings={"dev_potential": ovr + 2},
        season_stats={"pts": 40},
    )
    for k, v in extra.items():
        setattr(p, k, v)
    return p


def _team(tid, roster=None):
    return types.SimpleNamespace(
        team_id=tid,
        id=tid,
        abbr=tid,
        name=tid,
        roster=list(roster or []),
        ahl_roster=[],
        echl_roster=[],
        prospect_pool=[],
        gm_window="bubble",
        salary_cap_m=92.0,
    )


def _league(teams):
    return types.SimpleNamespace(teams=teams, salary_cap_m=92.0, free_agents=[])


class UncappedTradeValueTests(unittest.TestCase):
    def test_star_separates_from_depth(self):
        from app.sim_engine.trades.trade_value import (
            TRADE_VALUE_FORMULA_VERSION,
            evaluate_player_asset_value,
            _talent_base,
        )

        self.assertGreaterEqual(int(TRADE_VALUE_FORMULA_VERSION), 4)
        depth = _talent_base(75)
        mid = _talent_base(82)
        star = _talent_base(88)
        franchise = _talent_base(92)
        self.assertLess(depth, mid)
        self.assertLess(mid, star)
        self.assertLess(star, franchise)
        self.assertGreater(franchise, 100)

        team = _team("AAA", [])
        league = _league([team])
        p75 = _player("d1", ovr=75)
        p90 = _player("s1", ovr=90)
        team.roster = [p75, p90]
        v75 = float(evaluate_player_asset_value(p75, team, team, league, context={}).get("trade_value") or 0)
        v90 = float(evaluate_player_asset_value(p90, team, team, league, context={}).get("trade_value") or 0)
        self.assertGreater(v90, v75 + 40)
        self.assertGreater(v90, 100)


class FairnessKnobTests(unittest.TestCase):
    def test_ambient_gap_allows_lopsided(self):
        from app.sim_engine.trades.cpu_trade_proposer import (
            CPU_AMBIENT_FAIRNESS_GAP_MAX,
            CPU_DESPERATION_GAP_MAX,
            CPU_ONE_FOR_ONE_OVR_GAP_MAX,
        )
        from app.sim_engine.trades import trade_evaluator as te

        self.assertGreaterEqual(CPU_AMBIENT_FAIRNESS_GAP_MAX, 12.0)
        self.assertGreaterEqual(CPU_DESPERATION_GAP_MAX, 20.0)
        self.assertGreaterEqual(CPU_ONE_FOR_ONE_OVR_GAP_MAX, 6.0)
        self.assertGreaterEqual(te.CPU_AMBIENT_FAIRNESS_GAP_MAX, 12.0)


class TradeDemandTests(unittest.TestCase):
    def test_open_demand_before_after_and_popup(self):
        from services.trade_demand_engine import open_trade_demand, ensure_trade_demands

        star = _player("star1", ovr=86, age=28)
        team = _team("OTT", [star])
        team.abbr = "OTT"
        league = _league([team] + [_team(f"T{i}") for i in range(8)])
        for i, t in enumerate(league.teams):
            t.abbr = t.team_id if len(t.team_id) <= 3 else f"T{i}"
        session = types.SimpleNamespace(
            user_team_id="OTT",
            trade_demands={},
            pending_ui_popups=[],
            storyline_events=[],
            notifications=[],
            standings=None,
            sim=types.SimpleNamespace(league=league, rng=random.Random(7)),
        )
        row = open_trade_demand(
            session,
            star,
            team,
            reason="losing",
            calendar_idx=40,
            iso_date="2025-12-01",
            rng=random.Random(7),
        )
        self.assertEqual(row["status"], "open")
        self.assertLess(row["value_after"], row["value_before"])
        self.assertTrue(getattr(star, "_trade_demand_active", False))
        book = ensure_trade_demands(session)
        self.assertIn("star1", book)
        self.assertTrue(session.pending_ui_popups)
        popup = session.pending_ui_popups[-1]
        self.assertEqual(popup.get("kind"), "storyline")
        self.assertIn("trade_demand", popup)
        self.assertEqual(popup["trade_demand"]["value_before"], row["value_before"])

    def test_disruptor_label(self):
        from services.trade_demand_engine import open_trade_demand

        p = _player("bad1", ovr=84)
        team = _team("BUF", [p])
        team.abbr = "BUF"
        league = _league([team, _team("NYR")])
        league.teams[1].abbr = "NYR"
        session = types.SimpleNamespace(
            user_team_id="BUF",
            trade_demands={},
            pending_ui_popups=[],
            storyline_events=[],
            notifications=[],
            standings=None,
            sim=types.SimpleNamespace(league=league, rng=random.Random(3)),
        )
        row = open_trade_demand(
            session,
            p,
            team,
            reason="locker_room_disruptor",
            calendar_idx=50,
            rng=random.Random(3),
        )
        self.assertTrue(row["disruptor"])
        self.assertEqual(row["dossier_label"], "Locker-room disruptor")
        self.assertTrue(getattr(p, "locker_room_disruptor", False))

    def test_mntc_seed(self):
        from services.trade_demand_engine import seed_mntc_destinations

        p = _player("m1", ovr=80)
        p.contract.clause = "M-NTC"
        p.contract.trade_list_size = 5
        teams = [_team(f"T{i:02d}") for i in range(12)]
        for t in teams:
            t.abbr = t.team_id
        league = _league(teams)
        dests = seed_mntc_destinations(p, league, list_size=5, rng=random.Random(1))
        self.assertEqual(len(dests), 5)
        self.assertEqual(len(p.contract.approved_trade_teams), 5)

    def test_caused_demand_not_blocked(self):
        from app.sim_engine.franchise.storyline_engine import should_block_random_storyline_for_user

        session = types.SimpleNamespace(user_team_id="OTT")
        blocked = should_block_random_storyline_for_user(
            {
                "team_id": "OTT",
                "cause_type": "TRADE_DEMAND",
                "cause_event_id": "demand:x",
                "event_type": "trade_request",
                "storyline_text": "Player asks for trade",
            },
            session,
            user_team_id="OTT",
        )
        self.assertFalse(blocked)


if __name__ == "__main__":
    unittest.main()
