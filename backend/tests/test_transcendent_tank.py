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
    assert row["tank_mode"] == "seller"


def test_non_contender_at_any_strength_is_at_least_a_seller():
    tm = _Team("AAA", "playoff_contender")
    tm.gm_window = "contender"
    tm._tank_strength = {"is_contender": False, "playoff_gap_pts": 2.0, "strength_rank": 14}
    row = compute_tank_pressure_for_team(
        tm,
        transcendent_present=True,
        owns_own_first=True,
    )
    assert row["is_contender"] is False
    assert row["tank_mode"] in ("seller", "hard_tank")
    assert row["tank_pressure"] >= 70


def test_true_top_group_stays_a_contender_even_if_the_label_says_tank():
    tm = _Team("AAA", "tanking")
    tm.gm_window = "contend"
    tm._tank_strength = {"is_contender": True, "playoff_gap_pts": -8.0, "strength_rank": 2}
    row = compute_tank_pressure_for_team(
        tm,
        transcendent_present=True,
        owns_own_first=True,
    )
    assert row["is_contender"] is True
    assert row["tank_mode"] == "none"
    assert row["tank_pressure"] <= 12


def test_traded_own_pick_still_bids_for_a_lottery_pick():
    tm = _Team("AAA", "tanking")
    row = compute_tank_pressure_for_team(
        tm,
        transcendent_present=True,
        owns_own_first=False,
        pick_ownership_reason="pick_traded",
    )
    assert row["tank_mode"] == "seller"
    assert row["lottery_pick_intent"] == "buy"
    assert row["tank_pressure"] >= 70


def test_scratches_clear_at_season_change():
    from services.transcendent_tank_behavior import clear_tank_scratches

    tm = _Team("AAA", "tanking")
    tm._tank_scratched_ids = {"vet-1", "vet-2"}
    clear_tank_scratches([tm])
    assert tm._tank_scratched_ids == set()


def test_lottery_odds_rise_for_high_pressure_non_playoff_clubs():
    from app.sim_engine.draft.draft_lottery import ODDS_PCT, adjust_odds_for_tank_pressure

    pressures = [95, 90, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    adjusted = adjust_odds_for_tank_pressure(ODDS_PCT, pressures)
    assert len(adjusted) == 16
    assert abs(sum(adjusted) - sum(ODDS_PCT)) < 0.05
    assert adjusted[0] > ODDS_PCT[0]
    assert adjusted[-1] < ODDS_PCT[-1]

