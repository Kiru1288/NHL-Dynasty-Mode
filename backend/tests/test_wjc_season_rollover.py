"""World Juniors must not resurrect last year's finished desk after rollover."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from services.franchise_sim import (
    _apply_wjc_stock_after,
    _build_wjc_client_payload,
    _wjc_medal_tier_for_country,
)


def _session(*, sy: int, iso: str, bundle: dict | None):
    return SimpleNamespace(
        season_calendar_year=sy,
        calendar_cursor=0,
        current_date=iso,
        nhl_calendar=[{"iso": iso}],
        wjc_tournament_bundle=bundle,
        wjc_stock_evaluated_seasons={sy - 1},
        wjc_nhl_u20_loan={},
        team_by_id={},
        user_team_id="T1",
        sim=SimpleNamespace(rng=__import__("random").Random(1)),
    )


def test_prior_year_completed_wjc_does_not_force_live_desk_in_september():
    prior = {
        "season_sy": 2024,
        "wjc_format_version": 2,
        "tournament_prospects": [{"id": "p1", "name": "Kid"}],
        "medal_labels": {"gold": "CAN", "silver": "USA", "bronze": "SWE"},
        "rr_games": [{"home": "CAN", "away": "USA"}],
        "countries": [{"code": "CAN", "label": "Canada"}],
        "rr_days_total": 9,
        "stock_evaluated": True,
    }
    session = _session(sy=2025, iso="2025-09-15", bundle=prior)

    with mock.patch("services.franchise_sim._calendar_iso_for_day", return_value="2025-09-15"), mock.patch(
        "services.franchise_sim._today_iso", return_value="2025-09-15"
    ):
        payload = _build_wjc_client_payload(session)

    assert payload is not None
    assert payload.get("wjc_phase") == "upcoming"
    assert payload.get("wjc_live") is False
    assert not payload.get("medals_final")
    assert payload.get("medal_labels") == {}
    assert session.wjc_tournament_bundle is None


def test_same_year_completed_wjc_before_window_is_wiped():
    premature = {
        "season_sy": 2025,
        "wjc_format_version": 2,
        "tournament_prospects": [{"id": "p1", "name": "Kid"}],
        "medal_labels": {"gold": "FIN"},
        "rr_games": [],
        "countries": [{"code": "FIN", "label": "Finland"}],
        "rr_days_total": 9,
        "stock_evaluated": True,
    }
    session = _session(sy=2025, iso="2025-10-01", bundle=premature)
    session.wjc_stock_evaluated_seasons = {2025}

    with mock.patch("services.franchise_sim._calendar_iso_for_day", return_value="2025-10-01"), mock.patch(
        "services.franchise_sim._today_iso", return_value="2025-10-01"
    ):
        payload = _build_wjc_client_payload(session)

    assert payload.get("wjc_phase") == "upcoming"
    assert session.wjc_tournament_bundle is None
    assert 2025 not in (session.wjc_stock_evaluated_seasons or set())


def test_wjc_medal_tier_prefers_country_codes():
    assert _wjc_medal_tier_for_country("CAN", "Canada", {"gold": "CAN"}) == "gold"
    assert _wjc_medal_tier_for_country("CAN", "Canada", {}, {"gold": "CAN"}) == "gold"


def test_wjc_stock_applies_gold_medal_bonus():
    prospects = [
        {
            "player_id": "p1",
            "prospect_classification": "draft_eligible",
            "stock_rank_before": 30,
            "wjc_country": "CAN",
            "wjc_country_label": "Canada",
            "position": "F",
        }
    ]
    stats = [
        {
            "player_id": "p1",
            "gp": 5,
            "g": 3,
            "a": 2,
            "pts": 5,
            "plus_minus": 4,
            "position": "F",
        }
    ]
    standings = [{"code": "CAN", "w": 4, "l": 1}]
    without = _apply_wjc_stock_after(prospects, stats, standings, {}, day_multiplier=1.0)[0]
    with_gold = _apply_wjc_stock_after(
        prospects,
        stats,
        standings,
        {"gold": "CAN", "silver": "USA", "bronze": "SWE"},
        day_multiplier=1.0,
    )[0]
    assert with_gold["stock_delta"] > without["stock_delta"]
    assert "Gold medal" in str(with_gold.get("stock_reason") or "")


def test_wjc_stock_rewards_goalie_performance():
    prospects = [
        {
            "player_id": "g1",
            "prospect_classification": "draft_eligible",
            "stock_rank_before": 45,
            "wjc_country": "SWE",
            "wjc_country_label": "Sweden",
            "position": "G",
        }
    ]
    stats = [
        {
            "player_id": "g1",
            "gp": 5,
            "g": 0,
            "a": 0,
            "pts": 0,
            "position": "G",
            "w": 4,
            "l": 1,
            "shutouts": 1,
            "sv_pct": 0.925,
            "sa": 120,
            "sv": 111,
        }
    ]
    standings = [{"code": "SWE", "w": 3, "l": 2}]
    row = _apply_wjc_stock_after(prospects, stats, standings, {}, day_multiplier=1.0)[0]
    assert int(row.get("stock_delta") or 0) >= 20
    assert "SO" in str(row.get("stock_reason") or "")
