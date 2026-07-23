"""
Franchise retirement pass tests.
Run: python -m pytest backend/tests/test_franchise_retirement.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
SIM = ROOT / "SimEngine"
for p in (str(BACKEND), str(SIM)):
    if p not in sys.path:
        sys.path.insert(0, p)

from services.franchise_retirement import (  # noqa: E402
    get_retirements_list,
    run_franchise_retirement_pass,
)
from services.franchise_session import FranchiseSession  # noqa: E402


class _FixedRng:
    """Deterministic RNG that always returns values from a queue."""

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def random(self):
        if self._i >= len(self._values):
            return 0.99
        v = self._values[self._i]
        self._i += 1
        return v

    def gauss(self, *_a, **_k):
        return 0.0

    def shuffle(self, x):
        return None


def _player(pid: str, age: int, ovr: float = 0.58):
    ident = SimpleNamespace(name=f"Player {pid}", age=age, position=SimpleNamespace(value="C"))
    return SimpleNamespace(
        id=pid,
        player_id=pid,
        identity=ident,
        position="C",
        ovr=lambda: ovr,
        overall=ovr * 99,
        retired=False,
        psych=SimpleNamespace(morale=0.42),
        health=SimpleNamespace(wear_and_tear=0.55, injury_history=["knee"] * 3, durability=0.5),
        contract=SimpleNamespace(years_remaining=0),
        traits=SimpleNamespace(volatility=0.5),
        cup_wins=0,
        career_stats={"gp": 620, "g": 180, "a": 220, "pts": 400},
    )


def _team(tid: str, roster):
    return SimpleNamespace(
        team_id=tid,
        id=tid,
        name=f"Team {tid}",
        city="City",
        roster=list(roster),
        needs={},
        retired_alumni=[],
    )


def _session(teams, user_team_id="AAA", rng_values=None):
    league = SimpleNamespace(
        teams=teams,
        retired_players=[],
        free_agents=[],
        overseas_free_agents=[],
        development_leagues=[],
    )
    sim = SimpleNamespace(league=league, rng=_FixedRng(rng_values or [0.01] * 500))
    team_by_id = {str(getattr(t, "team_id", t.id)): t for t in teams}
    session = FranchiseSession(
        session_id="test",
        sim=sim,
        user_team_id=user_team_id,
        head_coach_name="Coach",
        coach_archetype="balanced",
        season_calendar_year=2025,
    )
    session.team_by_id = team_by_id
    session.champion_id = None
    session.player_season_stats = {}
    session.pending_decisions = []
    session.season_history = []
    return session


def test_37_plus_can_retire_at_final_skate():
    old = _player("OLD1", 39, 0.56)
    team = _team("AAA", [old])
    session = _session([team], rng_values=[0.05] * 200)

    payload = run_franchise_retirement_pass(session)

    assert session.retirements_processed is True
    assert payload["summary"]["nhl_count"] >= 1
    assert old.retired is True
    assert len(get_retirements_list(payload)) >= 1
    assert payload["all"][0].get("retirement_type")
    assert payload["all"][0].get("retirement_reason")


def test_idempotent_on_reload():
    old = _player("OLD2", 41, 0.54)
    team = _team("BBB", [old])
    session = _session([team], rng_values=[0.02] * 200)

    first = run_franchise_retirement_pass(session)
    count_first = len(first["all"])
    second = run_franchise_retirement_pass(session)

    assert count_first >= 1
    assert second is first or len(second["all"]) == count_first


def test_retired_removed_from_roster():
    p = _player("OLD3", 42, 0.52)
    team = _team("CCC", [p])
    session = _session([team], rng_values=[0.01] * 200)

    run_franchise_retirement_pass(session)

    assert p not in team.roster
    assert len(session.retired_players_archive) >= 1


def test_elite_under_35_protected():
    star = _player("STAR", 33, 0.93)
    team = _team("DDD", [star])
    session = _session([team], rng_values=[0.01] * 200)

    payload = run_franchise_retirement_pass(session)

    assert star.retired is False
    nhl_confirmed = [r for r in payload["all"] if r.get("confirmed")]
    assert all(r.get("player_id") != "STAR" for r in nhl_confirmed)


def test_user_borderline_generates_decision():
    border = _player("BORD", 35, 0.66)
    border.psych = SimpleNamespace(morale=0.48)
    team = _team("AAA", [border])
    session = _session([team], user_team_id="AAA", rng_values=[0.99] * 200)

    payload = run_franchise_retirement_pass(session)

    assert border.retired is False
    assert len(payload["considering"]) >= 1
    kinds = [d.get("kind") for d in session.pending_decisions]
    assert "retirement_decision" in kinds


def test_development_not_run_in_retirement_pass():
    """Retirement pass ages players but should not mutate OVR via progression."""
    p = _player("DEV1", 38, 0.60)
    before_ovr = p.ovr()
    team = _team("EEE", [p])
    session = _session([team], rng_values=[0.03] * 200)

    run_franchise_retirement_pass(session)

    assert p.ovr() == before_ovr
