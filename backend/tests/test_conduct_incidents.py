"""Conduct incident state machine — eligibility, GM choices, dress backlash."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIM = os.path.join(ROOT, "SimEngine")
for p in (ROOT, SIM, os.path.join(ROOT, "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.sim_engine.franchise.conduct_incidents import (  # noqa: E402
    apply_dress_backlash,
    apply_gm_conduct_choice,
    create_conduct_incident,
    get_team_org_pressure,
    get_team_revenue_modifier,
    player_eligible_to_dress,
    resolve_incident_availability,
    tick_incident_games,
)
from app.sim_engine.franchise.storyline_conduct import (  # noqa: E402
    apply_conduct_suspension,
    get_base_ovr_display,
)
from app.sim_engine.world.injuries import player_available_for_game  # noqa: E402


def _player(*, pid: str = "p1", name: str = "Test Player", ovr: int = 82) -> SimpleNamespace:
    return SimpleNamespace(
        id=pid,
        player_id=pid,
        name=name,
        ratings={"off_shooting": ovr, "def_stick_checking": ovr, "skg_speed": ovr},
        identity=SimpleNamespace(position="C", name=name),
        psych=SimpleNamespace(morale=0.5, confidence=0.5, role_satisfaction=0.5),
        retired=False,
        ovr=lambda: float(ovr),
    )


class ConductIncidentLifecycleTests(unittest.TestCase):
    def test_create_major_tracks_channels_without_permanent_ovr_wipe(self):
        host = SimpleNamespace()
        pl = _player()
        base_before = get_base_ovr_display(pl)
        inc = create_conduct_incident(
            host,
            player=pl,
            team_id="T1",
            storyline_text="Police investigating alleged assault after bar altercation",
            severity="major",
            storyline_id="story:test:1",
            player_fame=0.8,
            rng=__import__("random").Random(7),
        )
        self.assertEqual(inc["incident_family"], "violence")
        self.assertIn(inc["information_status"], ("reported", "confirmed_investigation"))
        self.assertIn(inc["legal_status"], ("allegation", "investigation"))
        self.assertNotEqual(inc["legal_status"], "conviction")
        self.assertFalse(inc["eligible_to_play"])  # default major → leave or league suspend
        self.assertFalse(player_eligible_to_dress(pl, host))
        self.assertFalse(player_available_for_game(pl))
        # Talent base unchanged (soft readiness may lower effective, not base wipe of 18–28).
        base_after = get_base_ovr_display(pl)
        self.assertEqual(base_before, base_after)
        self.assertLess(abs(base_after - base_before), 1)

    def test_investigation_without_leave_is_eligible_with_backlash(self):
        host = SimpleNamespace()
        pl = _player(pid="p2")
        inc = create_conduct_incident(
            host,
            player=pl,
            team_id="T1",
            storyline_text="Reports of a minor public disorder incident",
            severity="moderate",
            storyline_id="story:test:2",
            player_fame=0.4,
            rng=__import__("random").Random(1),
            auto_league_suspend_major=False,
        )
        # Force investigation-without-leave channel split.
        inc["team_status"] = "monitoring"
        inc["league_status"] = "investigating"
        inc["status"] = "under_investigation"
        inc["games_remaining"] = 0
        from app.sim_engine.franchise.conduct_incidents import recompute_eligibility, _sync_player_flags

        recompute_eligibility(inc)
        host.conduct_incidents[inc["incident_id"]] = inc
        _sync_player_flags(pl, inc)

        self.assertTrue(inc["eligible_to_play"])
        self.assertTrue(player_eligible_to_dress(pl, host))
        self.assertTrue(player_available_for_game(pl))

        org_before = get_team_org_pressure(host, "T1")
        rev_before = get_team_revenue_modifier(host, "T1")
        apply_dress_backlash(host, team_id="T1", player=pl, incident=inc)
        org_after = get_team_org_pressure(host, "T1")
        rev_after = get_team_revenue_modifier(host, "T1")
        self.assertGreater(org_after["media_heat"], org_before["media_heat"])
        self.assertLess(org_after["owner_confidence"], org_before["owner_confidence"])
        self.assertLessEqual(rev_after, rev_before)

    def test_league_suspension_blocks_dress_even_if_team_wants_override(self):
        host = SimpleNamespace()
        pl = _player(pid="p3")
        inc = create_conduct_incident(
            host,
            player=pl,
            team_id="T1",
            storyline_text="League integrity probe into sports betting allegations",
            severity="major",
            rng=__import__("random").Random(3),
        )
        self.assertEqual(inc["league_status"], "suspended")
        self.assertFalse(inc["team_can_override"])
        self.assertFalse(player_eligible_to_dress(pl, host))
        # GM "continue playing" cannot override league suspension.
        apply_gm_conduct_choice(host, incident_id=inc["incident_id"], choice_id="continue_playing", player=pl)
        self.assertFalse(player_eligible_to_dress(pl, host))

    def test_gm_choices_mutate_real_state(self):
        host = SimpleNamespace()
        pl = _player(pid="p4")
        # Start major without auto integrity suspend → admin leave.
        inc = create_conduct_incident(
            host,
            player=pl,
            team_id="T1",
            storyline_text="DUI investigation after traffic stop",
            severity="major",
            rng=__import__("random").Random(11),
        )
        self.assertEqual(inc["incident_family"], "driving")
        self.assertFalse(player_eligible_to_dress(pl, host))

        r1 = apply_gm_conduct_choice(host, incident_id=inc["incident_id"], choice_id="wait_league", player=pl)
        self.assertTrue(r1["ok"])
        self.assertTrue(r1["eligible_to_play"])
        self.assertTrue(player_eligible_to_dress(pl, host))

        r2 = apply_gm_conduct_choice(host, incident_id=inc["incident_id"], choice_id="place_on_leave", player=pl)
        self.assertFalse(r2["eligible_to_play"])
        self.assertFalse(player_eligible_to_dress(pl, host))

        r3 = apply_gm_conduct_choice(host, incident_id=inc["incident_id"], choice_id="support_program", player=pl)
        self.assertTrue(r3["ok"])
        # Support does not clear leave.
        self.assertFalse(player_eligible_to_dress(pl, host))

        r4 = apply_gm_conduct_choice(host, incident_id=inc["incident_id"], choice_id="do_nothing", player=pl)
        self.assertGreater(r4["org"]["media_heat"], 0.2)

    def test_tick_and_resolve_restores_availability(self):
        host = SimpleNamespace()
        pl = _player(pid="p5")
        inc = create_conduct_incident(
            host,
            player=pl,
            team_id="T1",
            storyline_text="Assault allegation under police review",
            severity="major",
            rng=__import__("random").Random(5),
        )
        # Short leave for test.
        inc["games_remaining"] = 2
        host.conduct_incidents[inc["incident_id"]] = inc
        from app.sim_engine.franchise.conduct_incidents import _sync_player_flags

        _sync_player_flags(pl, inc)
        tick_incident_games(host, pl)
        self.assertEqual(host.conduct_incidents[inc["incident_id"]]["games_remaining"], 1)
        cleared = tick_incident_games(host, pl)
        self.assertIsNotNone(cleared)
        self.assertIn(cleared["status"], ("cleared", "disciplined"))
        self.assertTrue(player_eligible_to_dress(pl, host))

    def test_apply_conduct_suspension_wrapper_uses_state_machine(self):
        host = SimpleNamespace()
        pl = _player(pid="p6")
        base = get_base_ovr_display(pl)
        meta = apply_conduct_suspension(
            pl,
            severity="major",
            storyline_id="story:wrap",
            host=host,
            team_id="T9",
            storyline_text="Weapons charge allegation reported",
            player_fame=0.7,
            rng=__import__("random").Random(9),
        )
        self.assertEqual(meta.get("conduct_model"), "state_machine")
        self.assertTrue(meta.get("incident_id"))
        # No permanent −18…−28 talent destruction.
        self.assertEqual(get_base_ovr_display(pl), base)
        self.assertLess(abs(int(meta.get("overall_delta") or 0)), 10)

    def test_legal_cleared_can_still_carry_league_discipline_flag(self):
        host = SimpleNamespace()
        pl = _player(pid="p7")
        inc = create_conduct_incident(
            host,
            player=pl,
            team_id="T1",
            storyline_text="Harassment allegation under review",
            severity="major",
            rng=__import__("random").Random(2),
        )
        resolve_incident_availability(
            host,
            incident=host.conduct_incidents[inc["incident_id"]],
            player=pl,
            legal_outcome="acquittal",
            league_outcome="disciplined",
        )
        row = host.conduct_incidents[inc["incident_id"]]
        self.assertEqual(row["legal_status"], "acquittal")
        self.assertEqual(row["league_status"], "disciplined")
        self.assertEqual(row["status"], "disciplined")
        # Legally cleared path with league discipline still tracks separately.
        self.assertTrue(row["eligible_to_play"] or int(row.get("games_remaining") or 0) == 0)


if __name__ == "__main__":
    unittest.main()
