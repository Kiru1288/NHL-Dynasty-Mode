"""Tests for franchise Entry Draft execution."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from services.franchise_session import FranchiseSession


def _mock_session():
    sim = MagicMock()
    league = MagicMock()
    league.development_leagues = []
    league.teams = []
    league.draft_pick_registry = {}
    sim.league = league
    sim.rng = MagicMock()

    session = FranchiseSession(
        session_id="test-session",
        sim=sim,
        user_team_id="T1",
        head_coach_name="Coach",
        coach_archetype="balanced",
        season_calendar_year=2025,
        team_ids=[f"T{i}" for i in range(1, 33)],
        team_by_id={f"T{i}": MagicMock() for i in range(1, 33)},
    )
    session.draft_lottery_payload = {
        "picks": [{"pick": i, "team_id": f"T{i}", "team_name": f"Team {i}"} for i in range(1, 17)]
    }
    session.draft_lottery_done = True
    return session


class EntryDraftTests(unittest.TestCase):
    def test_build_draft_order_has_224_picks(self):
        from services.franchise_entry_draft import build_full_draft_order, TOTAL_PICKS

        session = _mock_session()
        with patch("services.franchise_sim._build_standings_rows") as mock_rows:
            mock_rows.return_value = [
                {"team_id": f"T{i}", "pts": i, "w": i} for i in range(1, 33)
            ]
            order = build_full_draft_order(session)
        self.assertEqual(len(order), TOTAL_PICKS)
        self.assertEqual(order[0]["overall_pick"], 1)
        self.assertEqual(order[-1]["overall_pick"], TOTAL_PICKS)

    def test_playoff_order_champion_picks_last(self):
        from services.franchise_entry_draft import _build_round1_slot_order

        session = _mock_session()
        session.champion_id = "T32"
        session.playoff_payload = {
            "champion_id": "T32",
            "finalist_ids": ["T31"],
            "series_list": [
                {"round_index": 3, "team_high_id": "T32", "team_low_id": "T31", "wins_high": 4, "wins_low": 2},
                {"round_index": 1, "team_high_id": "T17", "team_low_id": "T18", "wins_high": 4, "wins_low": 1},
            ],
        }
        with patch("services.franchise_sim._build_standings_rows") as mock_rows:
            mock_rows.return_value = [
                {"team_id": f"T{i}", "pts": 100 - i, "w": i} for i in range(1, 33)
            ]
            order, source = _build_round1_slot_order(session)
        self.assertEqual(source, "lottery_playoff_results")
        self.assertEqual(order[-1], "T32")
        self.assertEqual(order[-2], "T31")

    def test_traded_pick_ownership(self):
        from services.franchise_entry_draft import _apply_registry_to_slot

        session = _mock_session()
        league = session.sim.league
        pick_id = "2026-round1-T8"
        league.draft_pick_registry = {
            pick_id: {
                "pick_id": pick_id,
                "year": 2026,
                "round": 1,
                "original_team_id": "T8",
                "current_owner_team_id": "T2",
                "resolved": False,
            }
        }
        slot = {
            "round": 1,
            "pick_in_round": 8,
            "overall_pick": 8,
            "team_id": "T8",
            "original_owner_team_id": "T8",
        }
        with patch("services.franchise_entry_draft._display_team", side_effect=lambda s, tid: tid):
            out = _apply_registry_to_slot(session, slot, 2026)
        self.assertEqual(out["team_id"], "T2")
        self.assertEqual(out["original_owner_team_id"], "T8")
        self.assertTrue(out.get("is_traded"))

    def test_scouting_interview_moves_board(self):
        from services.franchise_entry_draft import _scouting_event_adjustments

        session = _mock_session()
        entry = {"scouting_confidence": 80, "risk": "Low"}
        overlay = {"interview_status": "Completed", "traits": ["Leadership captain"]}
        phil = {"risk_tolerance": 0.5}
        delta, meta, notes = _scouting_event_adjustments(session, "T1", entry, overlay, phil)
        self.assertGreater(delta, 0)
        self.assertTrue(any("Interview" in n for n in notes))

    def test_duplicate_pick_guard(self):
        from services.franchise_entry_draft import initialize_entry_draft, execute_user_draft_pick

        session = _mock_session()
        session.draft_combine_done = True
        entries = [
            {"key": "P1", "name": "A", "rank": 1, "position": "C", "true_ovr": 80, "potential_score": 85},
            {"key": "P2", "name": "B", "rank": 2, "position": "D", "true_ovr": 78, "potential_score": 82},
        ]
        board = {"entries": entries, "class_strength": "Strong class", "total": 2}

        with patch("services.franchise_entry_draft.finalize_draft_class_for_event", return_value=board):
            with patch("services.franchise_entry_draft.build_full_draft_order") as mock_order:
                mock_order.return_value = [
                    {
                        "round": 1,
                        "pick_in_round": 1,
                        "overall_pick": 1,
                        "team_id": "T1",
                        "original_owner_team_id": "T1",
                        "team_name": "Team 1",
                    }
                ]
                initialize_entry_draft(session)

        player = MagicMock()
        player.id = "P1"
        player.drafted = False
        player.nhl_rights_team_id = ""
        player.rights_team_id = ""
        block = {"teams": [{"players": [player]}]}
        tm = block["teams"][0]

        with patch("services.franchise_entry_draft._find_prospect_player", return_value=(player, block, tm)):
            with patch("services.franchise_entry_draft._assign_drafted_prospect") as assign:
                def _mark_drafted(*_a, **_k):
                    player.drafted = True

                assign.side_effect = _mark_drafted
                with patch("services.franchise_entry_draft._mark_pick_resolved"):
                    with patch("services.franchise_sim.get_cached_draft_class_rankings", return_value=board):
                        with patch("services.franchise_sim.invalidate_session_payload_caches"):
                            execute_user_draft_pick(session, "P1")
                            with self.assertRaises(ValueError):
                                execute_user_draft_pick(session, "P1")


    def test_combine_stage_blocks_draft_until_complete(self):
        from services.franchise_entry_draft import initialize_entry_draft

        session = _mock_session()
        session.draft_combine_done = False
        with self.assertRaises(ValueError):
            initialize_entry_draft(session)

    def test_combine_results_persist(self):
        from services.franchise_scouting import run_franchise_draft_combine

        session = _mock_session()
        entries = [
            {
                "key": f"P{i}",
                "name": f"Player {i}",
                "rank": i,
                "position": "C" if i % 5 else "G",
                "true_ovr": 80 - i,
                "potential_score": 85 - i,
                "scouting_confidence": 60,
                "risk": "High" if i == 3 else "Low",
            }
            for i in range(1, 41)
        ]
        board = {"entries": entries, "class_strength": "Strong", "total": 40}
        with patch("services.franchise_scouting.get_cached_draft_class_rankings", return_value=board):
            payload1 = run_franchise_draft_combine(session)
            payload2 = run_franchise_draft_combine(session)
        self.assertTrue(session.draft_combine_done)
        self.assertEqual(payload1.get("invite_count"), payload2.get("invite_count"))
        self.assertEqual(payload1.get("invited_prospect_ids"), payload2.get("invited_prospect_ids"))

    def test_cpu_scouting_profiles_distinct(self):
        from services.franchise_scouting import ensure_team_scouting_profiles

        session = _mock_session()
        profiles = ensure_team_scouting_profiles(session)
        self.assertGreaterEqual(len(profiles), 32)
        qualities = [profiles[t]["scouting_quality"] for t in profiles]
        self.assertGreater(len(set(round(q, 0) for q in qualities)), 5)

    def test_cpu_teams_disagree_on_prospects(self):
        from services.franchise_scouting import run_franchise_draft_combine

        session = _mock_session()
        entries = [
            {"key": "PX", "name": "Test", "rank": 5, "position": "C", "true_ovr": 82, "potential_score": 88, "scouting_confidence": 55, "risk": "High"},
            {"key": "PY", "name": "Other", "rank": 12, "position": "D", "true_ovr": 78, "potential_score": 84, "scouting_confidence": 70, "risk": "Low"},
        ]
        board = {"entries": entries, "total": 2}
        with patch("services.franchise_scouting.get_cached_draft_class_rankings", return_value=board):
            payload = run_franchise_draft_combine(session)
        imps = (session.scouting_state or {}).get("team_impressions") or {}
        deltas = [
            imps[tid]["PX"]["board_delta"]
            for tid in session.team_ids[:6]
            if imps.get(tid, {}).get("PX")
        ]
        self.assertGreater(len(set(round(d, 1) for d in deltas)), 1)

    def test_batch_summary_shape(self):
        from services.franchise_entry_draft import _build_batch_summary

        session = _mock_session()
        picks = [
            {"overall_pick": 10, "final_rank": 25, "prospect_name": "Steal Guy", "team_name": "T1"},
            {"overall_pick": 5, "final_rank": 3, "prospect_name": "Reach Guy", "team_name": "T2"},
        ]
        session.draft_state = {
            "draft_order": [{"team_id": "T1", "round": 1, "pick_in_round": 1, "overall_pick": 1}],
            "overall_pick": 1,
            "drafted_prospect_ids": [],
        }
        with patch("services.franchise_sim.get_cached_draft_class_rankings") as mock_board:
            mock_board.return_value = {"entries": [{"key": "P9", "name": "Best", "rank": 9}]}
            summary = _build_batch_summary(session, picks)
        self.assertEqual(summary.get("picks_made"), 2)
        self.assertIsNotNone(summary.get("biggest_steal"))

    def test_entry_draft_blocked_without_combine_in_offseason(self):
        from services.franchise_offseason import _prepare_draft_payload

        session = _mock_session()
        session.draft_combine_done = False
        with self.assertRaises(ValueError):
            _prepare_draft_payload(session)


if __name__ == "__main__":
    unittest.main()
