"""Tests for Trade Stability Score, agents, and crisis timer."""

from __future__ import annotations

import time
import types
import unittest
import random


def _player(pid: str, *, ovr: float = 84, age: int = 27):
    p = types.SimpleNamespace(
        id=pid,
        player_id=pid,
        retired=False,
        identity=types.SimpleNamespace(name=f"Player {pid}", age=age),
        psych=types.SimpleNamespace(
            morale=0.42,
            confidence=0.45,
            role_satisfaction=0.22,
            coach_trust=0.40,
        ),
        traits=types.SimpleNamespace(
            ego=0.72,
            competitiveness=0.88,
            coachability=0.55,
            mental_toughness=0.48,
            work_ethic=0.50,
            leadership=0.45,
            volatility=0.62,
            patience=0.40,
        ),
        chemistry_profile={
            "competitiveness": 88,
            "loyalty": 42,
            "adaptability": 58,
            "belonging": 48,
        },
        contract=types.SimpleNamespace(clause="", years_remaining=2, term=2),
    )
    p.ovr = lambda: ovr
    return p


def _team(tid: str, roster):
    t = types.SimpleNamespace(team_id=tid, id=tid, abbr=tid[:3].upper(), roster=roster)
    return t


def _league(teams):
    return types.SimpleNamespace(teams=teams)


def _session(league, user_tid: str = "OTT"):
    class Rec:
        wins = 8
        losses = 28
        ot_losses = 4

    standings = types.SimpleNamespace(records={user_tid: Rec()})
    return types.SimpleNamespace(
        user_team_id=user_tid,
        trade_demands={},
        trade_stability_state={},
        pending_ui_popups=[],
        storyline_events=[],
        notifications=[],
        standings=standings,
        agent_relationships={},
        sim=types.SimpleNamespace(league=league, rng=random.Random(11)),
        calendar_cursor=40,
    )


class TradeStabilityTests(unittest.TestCase):
    def test_competitive_winner_mitigates_role_frustration(self):
        from app.sim_engine.franchise.trade_stability_engine import (
            PlayerConcernSnapshot,
            compute_trade_stability,
            gather_player_concerns,
        )

        league = _league([_team("OTT", []), _team("TBL", [])])
        player = _player("p1")
        session = _session(league)

        snap_losing = gather_player_concerns(session, player, league.teams[0])
        score_losing, _ = compute_trade_stability(snap_losing)

        class RecWin:
            wins = 38
            losses = 12
            ot_losses = 6

        session.standings.records["OTT"] = RecWin()
        snap_winning = gather_player_concerns(session, player, league.teams[0])
        score_winning, _ = compute_trade_stability(snap_winning)

        self.assertGreater(score_winning, score_losing)

    def test_low_character_escalates_faster(self):
        from app.sim_engine.franchise.trade_stability_engine import (
            PlayerConcernSnapshot,
            compute_trade_stability,
            stability_to_escalation_level,
            character_escalation_skip,
        )

        snap = PlayerConcernSnapshot(
            role_satisfaction=38,
            gm_trust=44,
            winning_satisfaction=40,
            competitiveness=70,
            character=58,
            mental=60,
            ego=70,
        )
        score, _ = compute_trade_stability(snap)
        level = stability_to_escalation_level(score)
        skip = character_escalation_skip(snap.character, score)
        self.assertGreaterEqual(level + skip, 2)

    def test_agent_assignment_is_deterministic_and_balanced(self):
        from app.sim_engine.franchise.player_agent_engine import AGENT_IDS, assign_agent_id

        counts = {aid: 0 for aid in AGENT_IDS}
        for i in range(500):
            aid = assign_agent_id(_player(f"player_{i}"))
            counts[aid] += 1
        for aid in AGENT_IDS:
            self.assertGreater(counts[aid], 60)
            self.assertLess(counts[aid], 140)

    def test_formal_demand_starts_crisis_timer(self):
        from services.trade_demand_engine import open_trade_demand, ensure_trade_demands

        star = _player("star1", ovr=86)
        team = _team("OTT", [star])
        league = _league([team, _team("NYR", [])])
        session = _session(league, "OTT")

        row = open_trade_demand(
            session,
            star,
            team,
            reason="role",
            calendar_idx=40,
            rng=random.Random(7),
            force_formal=True,
        )
        self.assertEqual(row["status"], "open")
        self.assertIn("crisis", row)
        self.assertGreaterEqual(int(row["crisis"]["initial_seconds"]), 120)
        self.assertLessEqual(int(row["crisis"]["initial_seconds"]), 360)
        book = ensure_trade_demands(session)
        self.assertIn("star1", book)

    def test_crisis_timer_sync_decrements_remaining(self):
        from services.trade_demand_engine import open_trade_demand, sync_trade_demand_crises, ensure_trade_demands

        star = _player("star2", ovr=85)
        team = _team("BUF", [star])
        league = _league([team])
        session = _session(league, "BUF")

        row = open_trade_demand(
            session,
            star,
            team,
            reason="management",
            calendar_idx=12,
            rng=random.Random(3),
            force_formal=True,
        )
        book = ensure_trade_demands(session)
        book["star2"]["crisis"]["last_sync_unix"] = time.time() - 5
        sync_trade_demand_crises(session)
        remaining = float(book["star2"]["crisis"]["remaining_seconds"])
        initial = float(row["crisis"]["initial_seconds"])
        self.assertLess(remaining, initial)

    def test_trade_hub_exposure_does_not_instantly_max_heat(self):
        from app.sim_engine.franchise.trade_stability_engine import apply_trade_hub_exposure

        star = _player("vet1", ovr=88, age=34)
        star.traits.mental_toughness = 0.92
        star.traits.coachability = 0.88
        team = _team("TOR", [star])
        league = _league([team])
        session = _session(league, "TOR")

        result = apply_trade_hub_exposure(session, star, attempt_n=1, rejection_kind="rejected")
        pst = getattr(star, "_franchise_storyline_state", {})
        self.assertGreaterEqual(result["stability_delta"], -5.0)
        self.assertLessEqual(int(pst.get("trade_rumor_heat") or 0), 25)


if __name__ == "__main__":
    unittest.main()
