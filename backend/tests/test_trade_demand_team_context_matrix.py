"""Trade demand matrix: terrible teams vs dynasties × character/mental grid."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# (character, mental, label)
CHAR_MENTAL_GRID = (
    (92, 94, "elite_pro"),
    (78, 72, "solid_vet"),
    (64, 72, "volatile_vet"),
    (58, 55, "low_both"),
    (88, 58, "high_char_low_mental"),
    (55, 90, "low_char_high_mental"),
)

BAD_RECORD = (12, 45, 5)
DYNASTY_RECORD = (44, 20, 4)


def _rec(w: int, l: int, otl: int = 0):
    return types.SimpleNamespace(wins=w, losses=l, ot_losses=otl)


def _player(
    pid: str,
    *,
    character: int,
    mental: int,
    ovr: float = 86.0,
    avg_toi_min: float = 18.0,
    pts: float = 42.0,
    gp: int = 50,
    line_role: str = "L2",
    psych_role_sat: float = 0.70,
):
    toi_sec = int(avg_toi_min * gp * 60)
    pp_toi_sec = int(max(0.0, (avg_toi_min * 0.18)) * gp * 60) if ovr >= 82 else 0
    p = types.SimpleNamespace(
        id=pid,
        player_id=pid,
        position="C",
        character=character,
        mental=mental,
        line_role=line_role,
        retired=False,
        identity=types.SimpleNamespace(name=f"Player {pid}", age=27, position="C"),
        psych=types.SimpleNamespace(
            morale=0.45,
            confidence=0.50,
            role_satisfaction=psych_role_sat,
            coach_trust=0.45,
        ),
        traits=types.SimpleNamespace(
            ego=0.65,
            competitiveness=0.88,
            coachability=character / 100.0,
            mental_toughness=mental / 100.0,
            work_ethic=0.55,
            leadership=0.50,
            volatility=0.40,
            patience=0.45,
        ),
        chemistry_profile={
            "competitiveness": 88,
            "loyalty": 48,
            "adaptability": mental,
            "belonging": 50,
        },
        contract=types.SimpleNamespace(clause="", years_remaining=2, term=2),
        season_stats={
            "gp": gp,
            "pts": pts,
            "toi_sec": toi_sec,
            "pp_toi_sec": pp_toi_sec,
        },
        _franchise_storyline_state={"gm_trust": 0.42},
    )
    p.ovr = lambda o=ovr: o
    return p


def _team(tid: str, roster, *, record):
    w, l, otl = record
    t = types.SimpleNamespace(
        team_id=tid,
        id=tid,
        abbr=tid,
        name=tid,
        roster=list(roster),
        situation="Rebuilding" if w < 20 else "Contending",
    )
    return t, _rec(w, l, otl)


def _session(league, user_tid: str, record):
    w, l, otl = record
    tid = user_tid
    standings = types.SimpleNamespace(records={tid: _rec(w, l, otl)})
    for team in league.teams:
        tw, tl, totl = record
        standings.records[str(team.team_id)] = _rec(tw, tl, totl)
    return types.SimpleNamespace(
        user_team_id=user_tid,
        trade_demands={},
        trade_stability_state={},
        pending_ui_popups=[],
        storyline_events=[],
        notifications=[],
        standings=standings,
        agent_relationships={},
        sim=types.SimpleNamespace(league=league),
        calendar_cursor=55,
    )


def _build_matrix_session(*, record: tuple[int, int, int], deployment: str = "fair"):
    """Six stars on one team with identical OVR but varying character/mental."""
    roster = []
    for idx, (character, mental, label) in enumerate(CHAR_MENTAL_GRID):
        if deployment == "benched_star":
            toi = 9.5 if idx == 0 else 17.5
            pts = 38.0 if idx == 0 else 28.0
            line = "L4" if idx == 0 else "L3"
        elif deployment == "misused":
            toi = 11.0
            pts = 44.0
            line = "L4"
        else:
            toi = 18.5 if idx <= 2 else 16.0
            pts = 40.0 if idx <= 2 else 32.0
            line = "L1" if idx == 0 else "L2" if idx <= 2 else "L3"

        roster.append(
            _player(
                f"bad_{label}" if record == BAD_RECORD else f"dyn_{label}",
                character=character,
                mental=mental,
                avg_toi_min=toi,
                pts=pts,
                line_role=line,
                psych_role_sat=0.75,
            )
        )

    team, _ = _team("TST", roster, record=record)
    league = types.SimpleNamespace(teams=[team])
    session = _session(league, "TST", record)
    return session, team, roster


class DeploymentRoleInferenceTests(unittest.TestCase):
    def test_toi_drives_role_not_psych_stub(self):
        from app.sim_engine.franchise.trade_stability_engine import (
            gather_player_concerns,
            infer_role_satisfaction_from_deployment,
        )

        roster = [
            _player("star_benched", character=78, mental=72, ovr=92, avg_toi_min=9.0, pts=44, line_role="L4"),
            _player("star_used", character=78, mental=72, ovr=92, avg_toi_min=20.5, pts=44, line_role="L1"),
        ]
        team, _ = _team("VAN", roster, record=BAD_RECORD)
        session = _session(types.SimpleNamespace(teams=[team]), "VAN", BAD_RECORD)

        benched = infer_role_satisfaction_from_deployment(roster[0], team, session)
        used = infer_role_satisfaction_from_deployment(roster[1], team, session)
        self.assertLess(benched, 45)
        self.assertGreater(used, 70)

        snap_benched = gather_player_concerns(session, roster[0], team)
        snap_used = gather_player_concerns(session, roster[1], team)
        self.assertLess(snap_benched.role_satisfaction, snap_used.role_satisfaction)

    def test_producing_while_benched_hits_performance_channel(self):
        from app.sim_engine.franchise.trade_stability_engine import (
            compute_trade_stability,
            gather_player_concerns,
        )

        player = _player(
            "producer_bench",
            character=64,
            mental=70,
            ovr=90,
            avg_toi_min=10.5,
            pts=48,
            line_role="L4",
        )
        team, _ = _team("CHI", [player], record=BAD_RECORD)
        session = _session(types.SimpleNamespace(teams=[team]), "CHI", BAD_RECORD)

        snap = gather_player_concerns(session, player, team)
        score, pressures = compute_trade_stability(snap)
        self.assertLess(snap.performance_vs_deployment, 35)
        self.assertGreater(pressures.get("role", 0) + pressures.get("performance", 0), 4.0)
        self.assertLess(score, 72)


class TerribleTeamMatrixTests(unittest.TestCase):
    def test_no_instant_full_crisis_on_bad_team(self):
        from app.sim_engine.franchise.trade_stability_engine import compute_instant_stability

        session, team, roster = _build_matrix_session(record=BAD_RECORD)
        for player in roster:
            row = compute_instant_stability(session, player, team)
            self.assertLess(int(row["escalation_level"]), 4, msg=_player_id(player))

    def test_low_character_low_mental_erodes_fastest(self):
        from app.sim_engine.franchise.trade_stability_engine import apply_daily_stability_update

        session, team, roster = _build_matrix_session(record=BAD_RECORD, deployment="misused")
        scores = {}
        for day in range(42, 82):
            for player in roster:
                apply_daily_stability_update(session, player, team, day)

        for player in roster:
            pid = str(player.id)
            scores[pid] = float(session.trade_stability_state[pid]["trade_stability_score"])

        low_both = scores["bad_low_both"]
        elite = scores["bad_elite_pro"]
        self.assertLess(low_both, elite)
        self.assertLess(low_both, scores["bad_volatile_vet"])

    def test_benched_star_less_stable_than_proper_usage(self):
        from app.sim_engine.franchise.trade_stability_engine import compute_instant_stability

        fair = _player("fair_usage", character=78, mental=72, ovr=90, avg_toi_min=20.0, pts=42, line_role="L1")
        benched = _player("benched_usage", character=78, mental=72, ovr=90, avg_toi_min=9.5, pts=42, line_role="L4")
        team, _ = _team("CHI", [fair, benched], record=BAD_RECORD)
        session = _session(types.SimpleNamespace(teams=[team]), "CHI", BAD_RECORD)

        fair_row = compute_instant_stability(session, fair, team)
        benched_row = compute_instant_stability(session, benched, team)
        self.assertLess(
            float(benched_row["trade_stability_score"]),
            float(fair_row["trade_stability_score"]),
        )
        benched_role = float((benched_row.get("pressures") or {}).get("role") or 0)
        fair_role = float((fair_row.get("pressures") or {}).get("role") or 0)
        self.assertGreater(benched_role, fair_role)


class DynastyTeamMatrixTests(unittest.TestCase):
    def test_dynasty_grid_stays_mostly_stable(self):
        from app.sim_engine.franchise.trade_stability_engine import compute_instant_stability

        session, team, roster = _build_matrix_session(record=DYNASTY_RECORD)
        for player in roster:
            row = compute_instant_stability(session, player, team)
            self.assertGreaterEqual(float(row["trade_stability_score"]), 62, msg=str(player.id))
            self.assertLessEqual(int(row["escalation_level"]), 1, msg=str(player.id))

    def test_dynasty_beats_bad_team_for_same_character_mental(self):
        from app.sim_engine.franchise.trade_stability_engine import apply_daily_stability_update

        bad_session, bad_team, bad_roster = _build_matrix_session(record=BAD_RECORD, deployment="misused")
        dyn_session, dyn_team, dyn_roster = _build_matrix_session(record=DYNASTY_RECORD, deployment="fair")

        for day in range(42, 82):
            for player in bad_roster:
                apply_daily_stability_update(bad_session, player, bad_team, day)
            for player in dyn_roster:
                apply_daily_stability_update(dyn_session, player, dyn_team, day)

        worse_labels = ("low_both", "volatile_vet", "low_char_high_mental")
        for (_character, _mental, label) in CHAR_MENTAL_GRID:
            bad_score = float(bad_session.trade_stability_state[f"bad_{label}"]["trade_stability_score"])
            dyn_score = float(dyn_session.trade_stability_state[f"dyn_{label}"]["trade_stability_score"])
            if label in worse_labels:
                self.assertGreater(dyn_score, bad_score, msg=label)
            else:
                self.assertGreaterEqual(dyn_score, bad_score, msg=label)


def _player_id(player) -> str:
    return str(getattr(player, "id", "") or getattr(player, "player_id", ""))


class FranchiseDeploymentWiringTests(unittest.TestCase):
    def test_session_player_season_stats_drive_toi(self):
        from app.sim_engine.franchise.trade_stability_engine import resolve_player_deployment

        player = _player("franchise_star", character=78, mental=72, ovr=90, avg_toi_min=99.0, pts=99)
        team, _ = _team("TOR", [player], record=BAD_RECORD)
        session = _session(types.SimpleNamespace(teams=[team]), "TOR", BAD_RECORD)
        session.player_season_stats = {
            "franchise_star": {
                "gp": 40,
                "pts": 36,
                "g": 15,
                "a": 21,
                "toi_sec": 40 * 11 * 60,
                "stat_authority": "session.player_season_stats",
            }
        }

        deploy = resolve_player_deployment(session, player, team)
        self.assertEqual(deploy.stat_source, "session.player_season_stats")
        self.assertAlmostEqual(deploy.avg_toi_min or 0, 11.0, places=1)
        self.assertEqual(deploy.pts, 36.0)

    def test_saved_even_strength_lines_set_line_role(self):
        from app.sim_engine.franchise.trade_stability_engine import (
            resolve_player_deployment,
            sync_player_role_from_real_data,
        )

        player = _player("lined_player", character=70, mental=68, ovr=88, avg_toi_min=18.0, pts=30)
        team, _ = _team("TOR", [player], record=BAD_RECORD)
        session = _session(types.SimpleNamespace(teams=[team]), "TOR", BAD_RECORD)
        session.lines = {
            "even_strength": {
                "lines": {
                    "forwards": [
                        {"id": "f1", "name": "Line 1", "slots": {"LW": "", "C": "", "RW": ""}},
                        {"id": "f4", "name": "Line 4", "slots": {"LW": "", "C": "lined_player", "RW": ""}},
                    ],
                    "defense": [],
                    "goalies": [],
                }
            }
        }

        deploy = sync_player_role_from_real_data(session, player, team)
        self.assertEqual(deploy.ev_line_rank, 4)
        self.assertEqual(deploy.line_role, "L4")
        self.assertEqual(getattr(player, "line_role"), "L4")

    def test_scratch_detection_from_lines(self):
        from app.sim_engine.franchise.trade_stability_engine import resolve_player_deployment

        scratch = _player("scratch_me", character=60, mental=58, ovr=84, avg_toi_min=8.0, pts=20)
        team, _ = _team("TOR", [scratch], record=BAD_RECORD)
        session = _session(types.SimpleNamespace(teams=[team]), "TOR", BAD_RECORD)
        session.lines = {
            "even_strength": {
                "lines": {
                    "forwards": [{"id": "f1", "slots": {"LW": "other", "C": "other2", "RW": "other3"}}],
                    "defense": [],
                    "goalies": [],
                }
            }
        }
        deploy = resolve_player_deployment(session, scratch, team)
        self.assertTrue(deploy.scratched)
        self.assertEqual(deploy.line_role, "scratch")

    def test_power_play_unit_from_saved_lines(self):
        from app.sim_engine.franchise.trade_stability_engine import resolve_player_deployment

        player = _player("pp_star", character=80, mental=75, ovr=91, avg_toi_min=19.0, pts=44)
        team, _ = _team("TOR", [player], record=DYNASTY_RECORD)
        session = _session(types.SimpleNamespace(teams=[team]), "TOR", DYNASTY_RECORD)
        session.lines = {
            "even_strength": {
                "lines": {
                    "forwards": [{"id": "f1", "slots": {"LW": "pp_star", "C": "", "RW": ""}}],
                    "defense": [],
                    "goalies": [],
                }
            },
            "power_play": {
                "lines": [
                    {"id": "pp1", "name": "PP1", "slots": {"LW": "pp_star", "C": "", "RW": "", "LD": "", "RD": ""}},
                    {"id": "pp2", "name": "PP2", "slots": {"LW": "", "C": "", "RW": "", "LD": "", "RD": ""}},
                ]
            },
        }
        deploy = resolve_player_deployment(session, player, team)
        self.assertEqual(deploy.ev_line_rank, 1)
        self.assertEqual(deploy.pp_unit, 1)


if __name__ == "__main__":
    unittest.main()
