"""League Operations payload contract tests (regular + preseason + slim)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.league_operations import (
    build_cap_forecast_series,
    build_league_operations_payload,
    calculate_escrow_progress,
    calculate_team_revenue,
    slim_league_operations_for_state,
)


def _fake_team(tid: str, abbr: str, *, market: str = "medium"):
    return SimpleNamespace(
        team_id=tid,
        id=tid,
        abbreviation=abbr,
        abbrev=abbr,
        name=abbr,
        city=abbr,
        market_size=market,
        payroll_m=70.0,
        roster=[],
        ahl_roster=[],
        echl_roster=[],
        ownership=SimpleNamespace(patience=0.55),
        arena_quality=0.6,
    )


def _fake_session(*, phase: str = "regular", salary_cap_m: float = 88.0):
    teams = {
        "1": _fake_team("1", "TOR", market="large"),
        "5": _fake_team("5", "OTT", market="small"),
        "9": _fake_team("9", "CBJ", market="small"),
    }
    league = SimpleNamespace(
        salary_cap_m=salary_cap_m,
        salary_cap=salary_cap_m,
        economics=SimpleNamespace(salary_cap=salary_cap_m, cap_floor=65.0),
        season_year=2025,
        teams=list(teams.values()),
    )
    sim = SimpleNamespace(league=league)
    session = SimpleNamespace(
        user_team_id="5",
        season_calendar_year=2025,
        calendar_cursor=0,
        phase=phase,
        season_phase=phase,
        offseason_stage=None,
        sim=sim,
        team_by_id=teams,
        nhl_calendar=[{"segment": "preseason" if phase == "preseason" else "regular"}],
        market_revenue_history={},
        escrow_ledger={},
        _stats_revision=0,
        standings=None,
        fan_profiles={},
        trade_heat_by_team={},
    )
    return session


def test_payload_stamps_95_5_cap_when_league_stuck_at_88():
    session = _fake_session(phase="regular", salary_cap_m=88.0)
    ops = build_league_operations_payload(session)
    assert abs(float(ops["salary_cap"]) - 95.5) < 1e-6
    assert len(ops["cap_forecast"]) >= 3
    assert ops["cap_forecast"][0]["cap"] == pytest.approx(95.5, abs=0.05)
    assert ops["cba"]["display_only"] is True
    assert "teams" in ops and len(ops["teams"]) == 3
    assert all("id" in t and "abbreviation" in t for t in ops["teams"])
    assert all("relocation_risk" in t for t in ops["teams"])
    assert "watchlist" in ops["relocation"]


def test_preseason_team_rows_have_full_schema():
    session = _fake_session(phase="preseason", salary_cap_m=95.5)
    row = calculate_team_revenue(session, session.team_by_id["5"], "5", is_user=True)
    for key in (
        "id",
        "abbreviation",
        "team_id",
        "abbr",
        "relocation_risk",
        "relocation_risk_label",
        "revenue_yoy_delta",
        "revenue_yoy_direction",
        "market_pressure",
    ):
        assert key in row, key
    # Second call should reuse persisted prior (stable YoY).
    row2 = calculate_team_revenue(session, session.team_by_id["5"], "5", is_user=True)
    assert row2["revenue_yoy_delta"] == row["revenue_yoy_delta"]
    assert "5" in session.market_revenue_history


def test_escrow_ledger_persists_on_session():
    session = _fake_session()
    first = calculate_escrow_progress(5000.0, 0.7, session=session)
    second = calculate_escrow_progress(5100.0, 0.72, session=session)
    assert first["escrow_ledger_active"] is True
    assert "2025" in session.escrow_ledger
    assert second["escrow_collected"] > 0


def test_slim_payload_keeps_user_only_teams():
    session = _fake_session(salary_cap_m=95.5)
    full = build_league_operations_payload(session)
    slim = slim_league_operations_for_state(full)
    assert slim.get("_slim") is True
    assert len(slim.get("teams") or []) <= 1
    assert "cap_forecast" in slim or "salary_cap" in slim


def test_cap_forecast_series_sources():
    session = _fake_session(salary_cap_m=95.5)
    cap = {"salary_cap": 95.5, "projected_salary_cap": 102.0, "cap_change_type": "Big Jump"}
    series = build_cap_forecast_series(session, cap)
    assert series[0]["source"] == "current"
    assert series[1]["source"] == "projected_next"
    assert series[2]["source"] == "extrapolated"
