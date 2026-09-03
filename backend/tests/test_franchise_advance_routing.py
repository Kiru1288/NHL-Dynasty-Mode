"""Tests for calendar advance routing: fast single-day, next_game, sparse backfill guard."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "SimEngine"))


def _clone_session(session):
    for attr in list(vars(session).keys()):
        if attr.endswith("_lock"):
            try:
                delattr(session, attr)
            except Exception:
                pass
    return copy.deepcopy(session)


def _fast_forward_to_regular(session, *, max_days: int = 120):
    from services.franchise_sim import advance_franchise_bulk

    if str(getattr(session, "phase", "")) == "regular":
        return session
    session._bulk_auto_resolve_injuries = True
    advance_franchise_bulk(
        session,
        mode="days",
        count=max_days,
        auto_resolve_decisions=True,
    )
    return session


@pytest.fixture(scope="module")
def regular_session():
    from services.franchise_sim import start_franchise

    session = start_franchise(
        team_query="Toronto Maple Leafs",
        head_coach_name="Test Coach",
        coach_archetype="balanced",
        seed=4242,
        player_universe="generated",
    )
    _fast_forward_to_regular(session)
    assert str(getattr(session, "phase", "")) == "regular"
    return session


def test_days_until_user_game_inclusive(regular_session):
    from services.franchise_sim import _days_until_user_game_inclusive

    session = _clone_session(regular_session)
    days = _days_until_user_game_inclusive(session)
    assert 1 <= days <= 82


def test_quick_single_day_uses_bulk_light_path(regular_session):
    from services.franchise_sim import advance_franchise_bulk

    session = _clone_session(regular_session)
    start_cursor = int(session.calendar_cursor)
    result = advance_franchise_bulk(
        session,
        mode="days",
        count=1,
        auto_resolve_decisions=True,
    )

    assert result.get("bulk") is True
    assert int(result.get("steps_completed") or 0) == 1
    assert result.get("status") == "ok"
    assert int(session.calendar_cursor) == start_cursor + 1
    assert bool(getattr(session, "_eligible_sparse_storyline_backfill", False)) is True


def test_next_game_advances_to_user_game_day(regular_session):
    from services.franchise_sim import (
        _days_until_user_game_inclusive,
        advance_franchise_to_next_user_game,
    )

    session = _clone_session(regular_session)
    expected_days = _days_until_user_game_inclusive(session)
    start_cursor = int(session.calendar_cursor)
    uid = str(session.user_team_id)

    result = advance_franchise_to_next_user_game(session, auto_resolve_decisions=True)

    assert result.get("bulk") is True
    assert int(result.get("target_days") or 0) == expected_days
    assert int(result.get("steps_completed") or 0) == expected_days
    assert result.get("status") == "ok"
    assert int(session.calendar_cursor) == start_cursor + expected_days

    # User's game for the day we landed on should already be simmed (cursor moved past it).
    by_day = getattr(session, "by_day", None) or {}
    game_day_idx = start_cursor + expected_days - 1
    had_user_game = any(
        uid in (str(getattr(sl, "home_id", "")), str(getattr(sl, "away_id", "")))
        for sl in (by_day.get(game_day_idx, []) or [])
    )
    if had_user_game:
        assert not list(by_day.get(game_day_idx, []) or []), "user game day should be cleared after sim"


def test_sparse_backfill_skipped_without_bulk_flag(regular_session):
    from services.franchise_sim import _maybe_backfill_sparse_storylines

    session = _clone_session(regular_session)
    session._eligible_sparse_storyline_backfill = False
    before = len(getattr(session, "storyline_events", None) or [])
    _maybe_backfill_sparse_storylines(session)
    after = len(getattr(session, "storyline_events", None) or [])
    assert before == after


def test_sparse_backfill_eligible_after_bulk(regular_session):
    from services.franchise_sim import advance_franchise_bulk

    session = _clone_session(regular_session)
    advance_franchise_bulk(session, mode="days", count=3, auto_resolve_decisions=True)
    assert bool(getattr(session, "_eligible_sparse_storyline_backfill", False)) is True


def test_api_routes_auto_resolve_single_day_through_bulk(regular_session):
    from fastapi.testclient import TestClient

    from main import app
    from services.franchise_store import save_session

    session = _clone_session(regular_session)
    save_session(session)

    client = TestClient(app)
    res = client.post(
        "/api/franchise/advance",
        json={"mode": "day", "count": 1, "auto_resolve": True},
        headers={"x-franchise-session": session.session_id},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    step = body.get("step") or {}
    assert step.get("bulk") is True
    assert int(step.get("steps_completed") or 0) == 1
    assert step.get("status") == "ok"


def test_api_next_game_mode(regular_session):
    from fastapi.testclient import TestClient

    from main import app
    from services.franchise_sim import _days_until_user_game_inclusive
    from services.franchise_store import save_session

    session = _clone_session(regular_session)
    expected = _days_until_user_game_inclusive(session)
    save_session(session)

    client = TestClient(app)
    res = client.post(
        "/api/franchise/advance",
        json={"mode": "next_game", "count": 1, "auto_resolve": True},
        headers={"x-franchise-session": session.session_id},
    )
    assert res.status_code == 200, res.text
    step = res.json().get("step") or {}
    assert step.get("mode") == "next_game"
    assert int(step.get("target_days") or 0) == expected
    assert int(step.get("steps_completed") or 0) == expected
