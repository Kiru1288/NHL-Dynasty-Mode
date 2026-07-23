"""Pick ownership + tank pressure behavior tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from services.draft_ranking_logic import compute_tank_pressure_for_team  # noqa: E402
from app.sim_engine.trades.trade_pick_registry import (  # noqa: E402
    canonical_pick_id,
    ensure_draft_pick_registry,
    team_owns_own_first,
    transfer_pick,
)


class _Team:
    def __init__(self, team_id: str, status: str = "rebuilding"):
        self.team_id = team_id
        self.team_status = status


class _League:
    def __init__(self, team_ids):
        self.teams = [_Team(tid) for tid in team_ids]
        self.current_season = 2026


def test_team_owns_own_first_default():
    league = _League(["AAA", "BBB"])
    ensure_draft_pick_registry(league, start_year=2026, years_ahead=3)
    own = team_owns_own_first(league, "AAA", draft_year=2027)
    assert own["owns_own_first"] is True
    assert own["pick_ownership_reason"] == "owns_own_first"


def test_traded_first_blocks_hard_tank():
    league = _League(["AAA", "BBB"])
    ensure_draft_pick_registry(league, start_year=2026, years_ahead=3)
    pick_id = canonical_pick_id(2027, 1, "AAA")
    transfer_pick(league, pick_id, "BBB")
    own = team_owns_own_first(league, "AAA", draft_year=2027)
    assert own["owns_own_first"] is False
    assert own["pick_ownership_reason"] == "pick_traded"
    row = compute_tank_pressure_for_team(
        league.teams[0],
        transcendent_present=True,
        owns_own_first=False,
        pick_ownership_reason="pick_traded",
    )
    assert row["tank_mode"] != "hard_tank"


def test_own_first_can_hard_tank():
    tm = _Team("AAA", "tanking")
    row = compute_tank_pressure_for_team(
        tm,
        transcendent_present=True,
        owns_own_first=True,
        pick_ownership_reason="owns_own_first",
    )
    assert row["tank_mode"] == "hard_tank"
    assert row["tank_pressure"] >= 90


def test_other_teams_first_not_permission_to_crater():
    league = _League(["AAA", "BBB"])
    ensure_draft_pick_registry(league, start_year=2026, years_ahead=3)
    own_pick = canonical_pick_id(2027, 1, "AAA")
    other_pick = canonical_pick_id(2027, 1, "BBB")
    transfer_pick(league, own_pick, "BBB")
    transfer_pick(league, other_pick, "AAA")
    own = team_owns_own_first(league, "AAA", draft_year=2027)
    assert own["owns_own_first"] is False
    row = compute_tank_pressure_for_team(
        _Team("AAA", "tanking"),
        transcendent_present=True,
        owns_own_first=False,
        pick_ownership_reason="pick_traded",
    )
    assert row["tank_mode"] != "hard_tank"


def test_protected_pick_caps_hard_tank():
    tm = _Team("AAA", "tanking")
    row = compute_tank_pressure_for_team(
        tm,
        transcendent_present=True,
        owns_own_first=True,
        pick_ownership_reason="protected_pick",
        owns_protected_first=True,
    )
    assert row["tank_mode"] != "hard_tank"
