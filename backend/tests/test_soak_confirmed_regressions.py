"""Regression coverage for soak-confirmed defects."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "SimEngine" / "app"))


def test_continue_offseason_advances_past_retirements_without_from_stage():
    """Empty from_stage must not replay retirements forever (soak STAGE_STALL)."""
    from services.franchise_offseason import continue_offseason
    import services.franchise_offseason as fo

    session = SimpleNamespace(
        phase="offseason",
        season_phase="offseason",
        offseason_stage="retirements",
        retirements_processed=True,
        retirements_payload={"ok": True, "version": 1},
        next_important_event="retirements",
        draft_completed=False,
        next_season_generated=False,
        resign_payload={},
        offseason_completed_stages=[],
        offseason_stage_entered_at={},
        offseason_stage_completed_at={},
        awards_payload={"ok": True},
        salary_cap_payload={},
        development_report_done=False,
        draft_lottery_done=False,
        draft_combine_done=False,
        draft_payload={},
        draft_state={},
        draft_review_payload={},
        prospect_rights_payload={},
        free_agency_market_payload={},
        roster_cleanup_payload={},
        next_season_payload={},
        champion_id="T1",
        stanley_cup_winner="T1",
        playoffs_simulated=True,
        sim=SimpleNamespace(league=SimpleNamespace(teams=[])),
        team_by_id={},
        pending_decisions=[],
    )

    calls = []

    def fake_handler(sess, stage):
        calls.append(stage)
        if stage == "salary_cap":
            sess.salary_cap_payload = {"new_season_cap": 95.0}
        return {stage: True}

    with patch.object(fo, "_stage_handler", fake_handler), patch.object(fo, "_sync_phase_fields", lambda s: None):
        out = continue_offseason(session)

    assert session.offseason_stage == "salary_cap", session.offseason_stage
    assert out.get("offseason_stage") == "salary_cap"
    assert "salary_cap" in calls


def test_franchise_session_deepcopy_roundtrip():
    from services.franchise_sim import start_franchise

    session = start_franchise(
        team_query="Buffalo Sabres",
        head_coach_name="Clone",
        coach_archetype="balanced",
        seed=1042,
    )
    session._draft_rankings_build_lock = __import__("threading").Lock()
    # Strip locks before clone (mirrors production __getstate__).
    for attr in list(vars(session).keys()):
        if attr.endswith("_lock"):
            delattr(session, attr)
    clone = copy.deepcopy(session)
    assert str(clone.user_team_id) == str(session.user_team_id)
    assert len(clone.team_by_id) == 32


def test_startup_schedule_cadence_under_hard_cap():
    from services.franchise_sim import start_franchise, _validate_league_cadence_hard

    session = start_franchise(
        team_query="Toronto Maple Leafs",
        head_coach_name="Sched",
        coach_archetype="balanced",
        seed=2042,
    )
    errors = _validate_league_cadence_hard(session.by_day, session.nhl_calendar)
    # Cadence repair should clear most 5-in-7 / 4-in-4 windows.
    assert len(errors) <= 12, errors[:5]


def test_roster_fill_repairs_composition_at_23_man_ceiling():
    """11F at a full 23-man list must demote surplus D/G then recall a forward (soak NEXT_SEASON_FAIL)."""
    from services.franchise_sim import start_franchise
    from services.contract_economy import run_roster_fill_pass
    from services.roster_compliance import (
        MIN_FORWARDS,
        position_bucket,
        summarize_team_roster_capacity,
    )

    session = start_franchise(
        team_query="Buffalo Sabres",
        head_coach_name="Fill",
        coach_archetype="rebuild",
        seed=1042,
    )
    team = session.team_by_id[session.user_team_id]
    roster = list(team.roster or [])
    forwards = [p for p in roster if position_bucket(p) == "F"]
    assert len(forwards) >= MIN_FORWARDS + 1
    # Park two NHL forwards in AHL so the club sits full with only 11F.
    parked = []
    for victim in forwards[:2]:
        roster = [p for p in roster if p is not victim]
        victim.in_minors = True
        victim.roster_location = "ahl"
        ahl = list(getattr(team, "ahl_roster", None) or [])
        ahl.append(victim)
        team.ahl_roster = ahl
        parked.append(victim)
    # Pad with two surplus defensemen from AHL so nhl_count returns to ceiling.
    defense_ahl = [
        p for p in (team.ahl_roster or [])
        if position_bucket(p) == "D" and p not in parked
    ]
    for pad in defense_ahl[:2]:
        team.ahl_roster = [p for p in (team.ahl_roster or []) if p is not pad]
        pad.in_minors = False
        pad.is_buried = False
        pad.roster_location = "nhl"
        roster.append(pad)
    team.roster = roster
    before = summarize_team_roster_capacity(team)
    assert before["forwards"] < MIN_FORWARDS, before
    assert before["nhl_count"] >= 22, before

    out = run_roster_fill_pass(session, teams=[team])
    after = summarize_team_roster_capacity(team)
    assert after["forwards"] >= MIN_FORWARDS, (before, after, out)
    assert not out.get("unresolved"), out


def test_roster_fill_dedupes_cross_team_nhl_assignment():
    """Stale FA refs must not leave the same player on two NHL rosters."""
    from services.franchise_sim import start_franchise
    from services.contract_economy import run_roster_fill_pass, _player_id

    session = start_franchise(
        team_query="Buffalo Sabres",
        head_coach_name="Dup",
        coach_archetype="rebuild",
        seed=1042,
    )
    teams = list(session.team_by_id.values())
    a, b = teams[0], teams[1]
    victim = (a.roster or [None])[0]
    assert victim is not None
    # Illegally mirror onto another NHL roster.
    b.roster = list(b.roster or []) + [victim]
    run_roster_fill_pass(session)
    seen = {}
    dups = 0
    for tm in teams:
        for p in tm.roster or []:
            pid = _player_id(p)
            if not pid:
                continue
            if pid in seen and seen[pid] != id(tm):
                dups += 1
            else:
                seen[pid] = id(tm)
    assert dups == 0
