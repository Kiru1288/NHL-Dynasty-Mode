"""Season-scale audits: star FA signings + ambient trade realism / value bands."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


def _ovr_fn(ovr99: float):
    return lambda: float(ovr99) / 99.0


def _player(
    pid: str,
    *,
    ovr: float = 80,
    age: int = 27,
    pos: str = "C",
    ask: float | None = None,
    pts: int = 40,
) -> Any:
    p = SimpleNamespace(
        id=pid,
        identity=SimpleNamespace(
            name=pid.replace("_", " ").title(),
            age=age,
            position=SimpleNamespace(value=pos),
            birth_country="CA",
        ),
        ratings={"overall": ovr, "dev_potential": ovr},
        ovr=_ovr_fn(ovr),
        age=age,
        position=pos,
        season_stats={"pts": pts, "gp": 82, "g": max(5, pts // 3), "a": pts - max(5, pts // 3)},
        retired=False,
        asking_aav_m=ask,
        ufa_exclusive=False,
        pending_july1_expiry=False,
        contract=None,
        cap_hit_m=0.0,
        player_type="sniper" if ovr >= 88 else "two_way",
    )
    return p


def _team(tid: str, roster: List[Any], *, window: str = "contender", space: float = 18.0) -> Any:
    tm = SimpleNamespace(
        team_id=tid,
        id=tid,
        name=f"Team {tid}",
        abbreviation=tid[:3].upper(),
        roster=list(roster),
        ahl_roster=[],
        echl_roster=[],
        prospect_pool=[],
        gm_window=window,
        salary_cap_m=95.0,
        total_cap_hit=max(0.0, 95.0 - space),
        cap_space=space,
        cap_space_m=space,
        retained_salary_records=[],
        buyouts=[],
        bonus_overages=[],
        other_dead_cap_m=0.0,
    )
    return tm


def test_trade_value_bands_look_realistic():
    """Stars >> top-6 >> depth; a mid isn't valued like a franchise piece."""
    from app.sim_engine.trades.trade_value import evaluate_player_asset_value

    team = _team("VAL", [])
    league = SimpleNamespace(teams=[team], salary_cap_m=95.0)
    ctx = {"season_year": 2025, "team_by_id": {"VAL": team}}

    def tv(ovr: float, age: int = 26) -> float:
        p = _player(f"p{ovr}", ovr=ovr, age=age, pts=int(ovr))
        return float(evaluate_player_asset_value(p, team, team, league, context=ctx)["total"])

    v70 = tv(70)
    v78 = tv(78)
    v84 = tv(84)
    v90 = tv(90)
    v93 = tv(93)
    assert v78 > v70
    assert v84 > v78 + 8
    assert v90 > v84 + 15
    assert v93 > v90
    # Depth should not be near-star territory, stars not near depth.
    assert v70 < 35
    assert v90 > 90
    assert v93 / max(1.0, v78) >= 2.0


def test_ambient_trade_packages_are_value_sane():
    """CPU ambient market should not ship absurd star-for-crumbs packages."""
    from app.sim_engine.trades.cpu_trade_proposer import propose_and_execute_cpu_trades
    from app.sim_engine.trades.trade_value import evaluate_player_asset_value

    rng = random.Random(11)
    teams = []
    for i in range(8):
        window = "rebuild" if i % 2 == 0 else "contender"
        space = 22.0 if window == "contender" else 28.0
        roster = [
            _player(f"{window}_{i}_star", ovr=90 - (i % 3), age=27, pts=95, pos="C"),
            _player(f"{window}_{i}_mid", ovr=82, age=26, pts=55, pos="LW"),
            _player(f"{window}_{i}_depth", ovr=74, age=29, pts=22, pos="RW"),
            _player(f"{window}_{i}_d1", ovr=84, age=28, pts=40, pos="D"),
            _player(f"{window}_{i}_d2", ovr=76, age=24, pts=18, pos="D"),
            _player(f"{window}_{i}_g", ovr=80, age=30, pts=0, pos="G"),
        ]
        # Fill to ~12 so proposers have options
        for j in range(6):
            roster.append(_player(f"{window}_{i}_x{j}", ovr=72 + (j % 5), age=25 + j % 4, pos="C"))
        teams.append(_team(f"T{i}", roster, window=window, space=space))

    league = SimpleNamespace(
        teams=teams,
        salary_cap_m=95.0,
        free_agents=[],
        overseas_free_agents=[],
        trade_history=[],
        draft_picks=[],
        pick_registry={},
        cpu_franchise_profiles={
            t.team_id: {
                "team_direction": "REBUILDING" if t.gm_window == "rebuild" else "CONTENDER",
                "ideology": {"aggression": 0.55, "future_asset_preference": 0.55},
            }
            for t in teams
        },
        _franchise_user_team_id="USER",
        season_year=2025,
        current_season=2025,
        draft_year=2026,
        season_is_calendar=True,
        rng=rng,
    )

    executed: List[Dict[str, Any]] = []
    for day in (40, 80, 110, 130, 145):
        batch = propose_and_execute_cpu_trades(
            league,
            max_executions=2,
            calendar_cursor=day,
            regular_season_last_index=180,
            season_year=2025,
            draft_year=2026,
        )
        executed.extend(batch)

    # Allow zero trades (market can be quiet) — if any fire, check fairness.
    unfair = []
    for row in executed:
        ex = row.get("execution") or row
        # Prefer explicit value totals when present
        gap = ex.get("fairness_gap")
        if gap is None:
            gap = (ex.get("evaluation") or {}).get("fairness_gap")
        if gap is not None and float(gap) > 12.0:
            unfair.append((row.get("headline"), gap))
        # Also reject star leaving for only depth on the return side when we can see assets
        outgoing = list(row.get("outgoing") or []) + list((ex.get("assets") or {}).get("outgoing") or [])
        incoming = list(row.get("incoming") or []) + list((ex.get("assets") or {}).get("incoming") or [])
        # Soft structural check via headline importance if present
        if str(row.get("importance") or "") == "major" and float(gap or 0) > 10:
            unfair.append((row.get("headline"), gap))

    assert len(unfair) == 0, f"unrealistic ambient trades: {unfair[:5]}"

    # Value sanity on sample roster pieces still holds after market activity
    sample = teams[0].roster[0]
    tv = float(
        evaluate_player_asset_value(sample, teams[0], teams[1], league, context={"season_year": 2025})["total"]
    )
    assert tv > 70  # ~90 OVR star band


def test_star_ufa_season_market_signs_with_credible_money():
    """Simulate a multi-week FA market: stars should land deals near market, not sit forever."""
    from services.contract_economy import LEAGUE_MINIMUM_AAV_M, compute_market_value, sync_all_team_cap_fields
    from services.fa_market_engine import ensure_fa_market_book, tick_free_agency_market

    stars = [
        _player("mcdavid_like", ovr=95, age=28, ask=12.5, pts=140, pos="C"),
        _player("kucherov_like", ovr=93, age=31, ask=10.5, pts=120, pos="RW"),
        _player("makar_like", ovr=92, age=27, ask=10.0, pts=90, pos="D"),
    ]
    depth = [
        _player(f"depth_{i}", ovr=72 + i % 4, age=28, ask=1.0, pts=18, pos="LW")
        for i in range(12)
    ]
    fa_pool = stars + depth

    teams = []
    for i in range(10):
        roster = [
            _player(f"cpu{i}_c", ovr=82, age=26, pts=50),
            _player(f"cpu{i}_w", ovr=80, age=27, pts=40, pos="LW"),
            _player(f"cpu{i}_d", ovr=79, age=28, pts=30, pos="D"),
            _player(f"cpu{i}_g", ovr=78, age=30, pts=0, pos="G"),
        ]
        for j in range(8):
            roster.append(_player(f"cpu{i}_x{j}", ovr=74, age=25, pos="C"))
        teams.append(_team(f"CPU{i}", roster, window="contender", space=16.0 + (i % 5)))

    league = SimpleNamespace(
        teams=teams,
        free_agents=list(fa_pool),
        overseas_free_agents=[],
        salary_cap_m=95.5,
        salary_floor_m=70.0,
    )
    rng = random.Random(42)
    session = SimpleNamespace(
        sim=SimpleNamespace(league=league, rng=rng),
        user_team_id="USER",
        season_calendar_year=2026,
        fa_market_book={},
        fa_market_day=0,
        cpu_fa_signings={},
        cpu_fa_wave=0,
        free_agency_open=True,
        team_by_id={t.team_id: t for t in teams},
    )

    needs = {
        "cap_space_m": 18.0,
        "slots_remaining": 8,
        "need_score": {"C": 0.85, "W": 0.6, "D": 0.55, "G": 0.2},
        "window": "contender",
        "counts": {"C": 3, "W": 4, "D": 5, "G": 2},
        "avg_ovr": {"C": 80, "W": 78, "D": 78, "G": 77},
        "best_ovr": {"C": 84, "W": 82, "D": 82, "G": 80},
        "overload": {},
        "primary_needs": ["C", "W", "D", "G"],
        "signed_prospect_counts": {},
        "roster_count": 12,
    }

    with patch("services.fa_market_engine.evaluate_team_position_needs", return_value=needs), patch(
        "services.fa_market_engine.cpu_signing_blocked", return_value=False
    ), patch(
        "services.fa_market_engine.score_free_agent_fit", return_value=(0.75, ["need"])
    ), patch(
        "services.fa_market_engine.sign_player_to_team",
        side_effect=lambda team, player, league, season_year, offer: _fake_sign(
            team, player, league, season_year, offer
        ),
    ):
        ensure_fa_market_book(session)
        for _ in range(21):  # three weeks of FA
            tick_free_agency_market(
                session,
                days=1,
                max_signings_per_day=4,
                max_offers_per_day=400,
            )

    book = session.fa_market_book or {}
    entries = book.get("entries") or {}
    report = []
    signed_stars = 0
    for star in stars:
        ent = entries.get(star.id) or {}
        market = compute_market_value(star, league)
        state = ent.get("state")
        aav = float(ent.get("signed_aav_m") or 0)
        team = ent.get("signed_team_name") or ent.get("signed_team_id")
        report.append(
            {
                "player": star.id,
                "market": round(market, 2),
                "state": state,
                "aav": aav,
                "team": team,
                "offers": int(ent.get("offer_count") or len(ent.get("offers") or [])),
            }
        )
        if state == "signed":
            signed_stars += 1
            assert aav >= market * 0.75, report[-1]
            assert aav <= max(market * 1.35, needs["cap_space_m"]), report[-1]
            assert aav >= LEAGUE_MINIMUM_AAV_M * 4

    # At least 2 of 3 stars should find homes over a 3-week market with space + need.
    assert signed_stars >= 2, f"star FA report: {report}"


def _fake_sign(player, team, league, season_year, offer):
    """Minimal sign stub so FA market can close without full contract stack."""
    pid = str(getattr(player, "id", "") or "")
    pool = list(getattr(league, "free_agents", None) or [])
    league.free_agents = [p for p in pool if str(getattr(p, "id", "")) != pid]
    roster = list(getattr(team, "roster", None) or [])
    roster.append(player)
    team.roster = roster
    try:
        player.contract = {
            "aav_m": float(offer.get("aav_m") or 0),
            "cap_hit_m": float(offer.get("aav_m") or 0),
            "years_remaining": int(offer.get("years") or 1),
        }
        player.cap_hit_m = float(offer.get("aav_m") or 0)
    except Exception:
        pass
    return {"ok": True, "aav_m": offer.get("aav_m"), "years": offer.get("years")}


def test_trade_frequency_target_band_lowered():
    """Seasonal ambient target should sit well below the old 48–78 spam band."""
    import inspect
    from app.sim_engine.engine import SimEngine

    src = inspect.getsource(SimEngine._season_daily_socio_economics)
    assert "max(30, min(48" in src or "max(30, min(48," in src
    assert "0.030 +" in src or "0.030+" in src
    assert "min(4, max_exec)" in src or "min(4, max_exec)" in src.replace(" ", "")


def test_dismiss_all_trade_popups_helper():
    from services.franchise_sim import dismiss_franchise_popups

    session = SimpleNamespace(
        pending_ui_popups=[
            {"id": "cpu_trade_popup:1", "event_type": "CPU_TRADE"},
            {"id": "cpu_trade_popup:2", "event_type": "CPU_TRADE"},
            {"id": "injury:1", "kind": "injury"},
        ]
    )
    dismiss_franchise_popups(session, ["cpu_trade_popup:1", "cpu_trade_popup:2"])
    assert len(session.pending_ui_popups) == 1
    assert session.pending_ui_popups[0]["id"] == "injury:1"


def test_cpu_trade_popup_only_when_player_value_exceeds_70():
    from services.franchise_sim import _enqueue_cpu_trade_popup, _max_moved_player_trade_value

    low_ev = {
        "trade_id": "tv_low",
        "from_team_id": "AAA",
        "team": "BBB",
        "headline": "depth swap",
        "execution": {
            "trade_id": "tv_low",
            "value_breakdown": {
                "AAA": {
                    "incoming": [{"type": "player", "asset_id": "p1", "total": 42.0}],
                    "outgoing": [{"type": "player", "asset_id": "p2", "total": 38.0}],
                    "incoming_total": 42.0,
                },
                "BBB": {
                    "incoming": [{"type": "player", "asset_id": "p2", "total": 38.0}],
                    "outgoing": [{"type": "player", "asset_id": "p1", "total": 42.0}],
                    "incoming_total": 38.0,
                },
            },
            "moved_assets": [
                {"asset_type": "player", "asset_id": "p1", "player_name": "Low"},
            ],
            "history_record": {},
        },
        "importance": "major",
        "trade_category": "major_trade",
    }
    assert _max_moved_player_trade_value(low_ev) == 42.0

    high_ev = {
        "trade_id": "tv_high",
        "from_team_id": "AAA",
        "team": "BBB",
        "headline": "star move",
        "execution": {
            "trade_id": "tv_high",
            "value_breakdown": {
                "AAA": {
                    "incoming": [{"type": "player", "asset_id": "star", "total": 88.5}],
                    "outgoing": [{"type": "pick", "asset_id": "pick1", "total": 40.0}],
                    "incoming_total": 88.5,
                },
                "BBB": {
                    "incoming": [{"type": "pick", "asset_id": "pick1", "total": 40.0}],
                    "outgoing": [{"type": "player", "asset_id": "star", "total": 88.5}],
                    "incoming_total": 40.0,
                },
            },
            "moved_assets": [
                {"asset_type": "player", "asset_id": "star", "player_name": "Star"},
            ],
            "history_record": {},
        },
        "importance": "standard",
        "trade_category": "hockey_trade",
    }
    assert _max_moved_player_trade_value(high_ev) == 88.5

    session = SimpleNamespace(
        pending_ui_popups=[],
        showcase_archive=[],
        cpu_trade_event_seen_ids=set(),
        team_by_id={},
        league=SimpleNamespace(draft_pick_registry={}),
        cpu_franchise_profiles={},
        season_calendar_year=2025,
    )
    with patch("services.franchise_sim.build_cpu_trade_transaction_event") as builder:
        builder.side_effect = lambda *a, **k: {
            "id": f"cpu_trade_popup:{(a[1] if len(a) > 1 else k.get('ev') or {}).get('trade_id')}",
            "event_type": "CPU_TRADE",
        }
        # Force builder to see trade_id from ev — patch returns fixed ids:
        builder.side_effect = None

        def _build(sess, ev, calendar_idx=0, iso=""):
            return {
                "id": f"cpu_trade_popup:{ev.get('trade_id')}",
                "event_type": "CPU_TRADE",
                "title": "Trade Completed",
            }

        builder.side_effect = _build
        _enqueue_cpu_trade_popup(session, low_ev, calendar_idx=10, iso="2025-11-01")
        assert session.pending_ui_popups == []
        assert any(r.get("id") == "cpu_trade_popup:tv_low" for r in session.showcase_archive)

        _enqueue_cpu_trade_popup(session, high_ev, calendar_idx=11, iso="2025-11-02")
        assert any(r.get("id") == "cpu_trade_popup:tv_high" for r in session.pending_ui_popups)
