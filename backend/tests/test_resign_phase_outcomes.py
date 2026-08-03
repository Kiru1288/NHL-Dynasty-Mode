"""Re-sign phase outcome ledger — Accepted/Rejected rows survive the phase."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _session():
    return types.SimpleNamespace(
        session_id="test",
        user_team_id="OTT",
        resign_phase_outcomes={},
        resign_negotiations={},
        resign_payload={},
        own_fa_window_day=2,
        own_fa_window_signings=[],
        free_agency_open=False,
    )


class ResignPhaseOutcomeTests(unittest.TestCase):
    def test_upsert_and_terminal_stickiness(self):
        from services.franchise_offseason import (
            RESIGN_PHASE_TERMINAL,
            clear_resign_phase_state,
            upsert_resign_phase_outcome,
        )

        session = _session()
        upsert_resign_phase_outcome(
            session,
            player_id="p1",
            phase_status="rejected",
            snapshot_row={"player_id": "p1", "name": "A Player", "expiry_status": "UFA"},
            name="A Player",
            last_offer={"aav_m": 4.0, "years": 3},
            reason="too_low",
        )
        self.assertEqual(session.resign_phase_outcomes["p1"]["phase_status"], "rejected")
        self.assertFalse(session.resign_phase_outcomes["p1"]["terminal"])

        # Rejected can be overwritten by a later accept.
        upsert_resign_phase_outcome(
            session,
            player_id="p1",
            phase_status="accepted",
            terms={"aav_m": 5.5, "years": 4, "expiry_year": 2029},
            name="A Player",
        )
        self.assertEqual(session.resign_phase_outcomes["p1"]["phase_status"], "accepted")
        self.assertTrue(session.resign_phase_outcomes["p1"]["terminal"])
        self.assertIn("accepted", RESIGN_PHASE_TERMINAL)

        # Terminal cannot be downgraded to open/rejected.
        upsert_resign_phase_outcome(session, player_id="p1", phase_status="rejected")
        self.assertEqual(session.resign_phase_outcomes["p1"]["phase_status"], "accepted")

        clear_resign_phase_state(session)
        self.assertEqual(session.resign_phase_outcomes, {})
        self.assertEqual(session.resign_negotiations, {})

    def test_prepare_payload_keeps_accepted_and_released_rows(self):
        from services.franchise_offseason import (
            _prepare_resign_payload,
            upsert_resign_phase_outcome,
        )

        player_signed = types.SimpleNamespace(
            id="signed1",
            name="Signed Guy",
            position=types.SimpleNamespace(value="C"),
            identity=types.SimpleNamespace(name="Signed Guy", position=types.SimpleNamespace(value="C"), age=28),
            contract=types.SimpleNamespace(
                years_remaining=4,
                aav_m=5.0,
                cap_hit_m=5.0,
                expiry_status="UFA",
                expiry_year=2029,
                type="STANDARD",
                is_nhl_spc=True,
            ),
            retired=False,
            age=28,
            overall=84,
            ovr=84,
        )
        team = types.SimpleNamespace(
            team_id="OTT",
            id="OTT",
            roster=[player_signed],
            ahl_roster=[],
            echl_roster=[],
            prospect_pool=[],
            rfa_rights=[],
            buyout_cap_hits=[],
            retained_salary_records=[],
        )
        league = types.SimpleNamespace(teams=[team], free_agents=[], salary_cap_m=88.0, cap_floor_m=65.0)
        session = _session()
        session.team_by_id = {"OTT": team}
        session.sim = types.SimpleNamespace(league=league)
        session.season_calendar_year = 2025
        session.own_fa_window_active = True
        session.own_fa_window_day = 6
        session.free_agency_open = False

        upsert_resign_phase_outcome(
            session,
            player_id="signed1",
            phase_status="accepted",
            snapshot_row={
                "player_id": "signed1",
                "name": "Signed Guy",
                "expiry_status": "UFA",
                "position": "C",
            },
            terms={"aav_m": 5.0, "years": 4},
            name="Signed Guy",
        )
        upsert_resign_phase_outcome(
            session,
            player_id="walked1",
            phase_status="released",
            snapshot_row={
                "player_id": "walked1",
                "name": "Walked Away",
                "expiry_status": "RFA",
                "contract_status": "released",
                "position": "D",
            },
            name="Walked Away",
            reason="walked_away",
        )

        # Patch office builder to avoid full franchise bootstrap.
        import services.franchise_offseason as fo
        import services.contract_economy as ce

        def _fake_office(_session):
            return {
                "contracts": [
                    {
                        "player_id": "signed1",
                        "name": "Signed Guy",
                        "position": "C",
                        "years_remaining": 4,
                        "aav_m": 5.0,
                        "expiry_status": "UFA",
                        "can_negotiate": False,
                        "available_actions": [],
                    }
                ],
                "expiring": [],
                "rfa_rights": [],
                "summary": {"ufaCount": 0, "rfaCount": 0, "pendingDecisions": 0},
                "cap_snapshot": {},
                "contract_slots": {"used": 1, "limit": 50, "open": 49},
                "buyout_candidates": [],
                "team": {},
            }

        original = ce.build_contract_office
        ce.build_contract_office = _fake_office
        try:
            payload = _prepare_resign_payload(session, force=True)
        finally:
            ce.build_contract_office = original

        board = (payload.get("re_sign") or payload.get("contracts") or {})
        rows = board.get("contracts") or []
        by_id = {str(r.get("player_id")): r for r in rows}
        self.assertIn("signed1", by_id)
        self.assertEqual(by_id["signed1"].get("phase_status"), "accepted")
        self.assertIn("walked1", by_id)
        self.assertEqual(by_id["walked1"].get("phase_status"), "released")
        self.assertEqual(board.get("version"), 6)
        self.assertIn("phase_outcomes", board)

    def test_rejected_is_retryable_until_accepted(self):
        from services.franchise_offseason import upsert_resign_phase_outcome

        session = _session()
        upsert_resign_phase_outcome(
            session,
            player_id="p2",
            phase_status="rejected",
            snapshot_row={"player_id": "p2", "name": "Retry Me", "expiry_status": "UFA"},
            reason="too_low",
        )
        upsert_resign_phase_outcome(
            session,
            player_id="p2",
            phase_status="countered",
            last_offer={"aav_m": 4.5, "years": 3},
        )
        self.assertEqual(session.resign_phase_outcomes["p2"]["phase_status"], "countered")
        self.assertFalse(session.resign_phase_outcomes["p2"]["terminal"])

        upsert_resign_phase_outcome(session, player_id="p2", phase_status="pending")
        self.assertEqual(session.resign_phase_outcomes["p2"]["phase_status"], "pending")

        upsert_resign_phase_outcome(
            session,
            player_id="p2",
            phase_status="accepted",
            terms={"aav_m": 5.0, "years": 4},
        )
        self.assertTrue(session.resign_phase_outcomes["p2"]["terminal"])

    def test_leaving_resign_stage_clears_ledger(self):
        from services.franchise_offseason import (
            OFFSEASON_STAGES,
            clear_resign_phase_state,
            upsert_resign_phase_outcome,
        )

        session = _session()
        session.offseason_stage = "re_sign"
        upsert_resign_phase_outcome(
            session,
            player_id="p3",
            phase_status="accepted",
            snapshot_row={"player_id": "p3", "name": "Done"},
        )
        self.assertTrue(session.resign_phase_outcomes)
        # Mimic advance_offseason_stage cleanup when leaving re_sign.
        if session.offseason_stage == "re_sign":
            clear_resign_phase_state(session)
            session.offseason_stage = "free_agency"
        self.assertEqual(session.resign_phase_outcomes, {})
        self.assertIn("free_agency", OFFSEASON_STAGES)


if __name__ == "__main__":
    unittest.main()
