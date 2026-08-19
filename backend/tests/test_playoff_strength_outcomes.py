"""Playoff outcomes should follow regular-season strength, not coin-flips.

A 70-win team should almost always beat a 30-win team in a best-of-seven
unless season analytics (xG) say the weaker record was a mirage.
Typical 1-vs-8 series can still produce upsets.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SimEngine"))
sys.path.insert(0, str(ROOT / "backend"))

from app.sim_engine.league.playoffs import (  # noqa: E402
    PlayoffSeries,
    _augment_strength_with_results,
    _simulate_series,
    playoff_game_win_probability,
)
from app.sim_engine.league.standings import StandingsTable, TeamStandingRecord  # noqa: E402
from services.franchise_playoffs import _simulate_one_game  # noqa: E402

N_GAMES = 500
N_SERIES = 400


def _club(tid: str, name: str, wins: int, losses: int, otl: int, gf: int, ga: int, **extra):
    gp = wins + losses + otl
    team = SimpleNamespace(
        team_id=tid,
        id=tid,
        name=name,
        abbr=tid,
        conference="East",
        division="Atlantic",
        roster=[],
        **extra,
    )
    rec = TeamStandingRecord(
        team_id=tid,
        name=name,
        abbr=tid,
        conference="East",
        division="Atlantic",
        gp=gp,
        wins=wins,
        losses=losses,
        otl=otl,
        points=wins * 2 + otl,
        gf=gf,
        ga=ga,
        rw=wins,
        row=wins,
    )
    return team, rec


def _standings(teams_and_recs):
    teams = [t for t, _ in teams_and_recs]
    table = StandingsTable(teams)
    for _, rec in teams_and_recs:
        table.records[rec.team_id] = rec
    return table


def _live_session(strength_map, standings=None, game_results=None, session_id="s0"):
    return SimpleNamespace(
        session_id=session_id,
        strength_map=dict(strength_map),
        season_calendar_year=2026,
        playoff_live={"playoff_day": 1},
        standings=standings,
        game_results=list(game_results or []),
        team_by_id={},
    )


def _series_row(series_id="ser"):
    return {
        "series_id": series_id,
        "team_high_id": "HI",
        "team_low_id": "LO",
        "wins_high": 0,
        "wins_low": 0,
        "game_log": [],
        "status": "active",
        "best_of": 7,
    }


def live_game_high_win_rate(session_factory, n=N_GAMES) -> float:
    wins = 0
    for i in range(n):
        session = session_factory(i)
        row = _series_row(f"g{i}")
        entry = _simulate_one_game(session, row)
        if entry["winner_id"] == "HI":
            wins += 1
    return wins / n


def live_series_high_win_rate(session_factory, n=N_SERIES) -> float:
    wins = 0
    for i in range(n):
        session = session_factory(i)
        row = _series_row(f"ser{i}")
        while row["status"] != "complete":
            _simulate_one_game(session, row)
        if row["winner_id"] == "HI":
            wins += 1
    return wins / n


def atomic_series_high_win_rate(strength_map, n=N_SERIES, seed=7) -> float:
    rng = random.Random(seed)
    wins = 0
    for _ in range(n):
        series = PlayoffSeries(
            round_index=1,
            conference="East",
            seed_high=1,
            seed_low=8,
            team_high_id="HI",
            team_low_id="LO",
        )
        _simulate_series(rng, series, strength_map)
        if series.winner_id() == "HI":
            wins += 1
    return wins / n


def _xg_games(hi_xgf: float, lo_xgf: float, n: int = 82):
    games = []
    for i in range(n):
        games.append(
            {
                "stat_scope": "regular_season",
                "home_id": "HI",
                "away_id": "LO",
                "home_xgf": hi_xgf,
                "away_xgf": lo_xgf,
                "home_xg": hi_xgf,
                "away_xg": lo_xgf,
            }
        )
    return games


def test_playoff_win_probability_scales_with_talent_gap():
    even = playoff_game_win_probability(0.50, 0.50)
    mild = playoff_game_win_probability(0.58, 0.48)
    large = playoff_game_win_probability(0.82, 0.28)
    print(
        f"\nP(game) even={even:.3f} mild(+0.10)={mild:.3f} large(+0.54)={large:.3f}"
    )
    assert 0.48 <= even <= 0.52
    assert mild >= 0.56
    assert large >= 0.78
    assert large > mild > even


def test_live_equal_strength_is_near_coin_flip():
    rate = live_game_high_win_rate(
        lambda i: _live_session({"HI": 0.50, "LO": 0.50}, session_id=f"eq{i}")
    )
    print(f"\nLive game win% equal strength (game 1, home favorite): {rate:.1%}")
    assert 0.48 <= rate <= 0.68


def test_live_70_win_vs_30_win_same_roster_is_not_a_coin_flip():
    """Preseason OVR can be similar; the 70-win season still has to matter."""
    hi, hi_rec = _club("HI", "Titans", 70, 10, 2, 310, 175)
    lo, lo_rec = _club("LO", "Tankers", 30, 46, 6, 175, 310)
    table = _standings([(hi, hi_rec), (lo, lo_rec)])

    def factory(i):
        return _live_session({"HI": 0.52, "LO": 0.50}, standings=table, session_id=f"70v30-{i}")

    game_rate = live_game_high_win_rate(factory)
    series_rate = live_series_high_win_rate(factory)
    print(
        f"\nLive 70-10-2 vs 30-46-6, similar roster OVR: "
        f"game1={game_rate:.1%} series={series_rate:.1%}"
    )
    assert game_rate >= 0.70, f"70-win team game favorite too weak: {game_rate:.1%}"
    assert series_rate >= 0.88, f"30-win team winning too many series: {1 - series_rate:.1%}"


def test_live_analytics_can_justify_a_weaker_record():
    """A 30-win club with elite xG vs a lucky 70-win club should be competitive."""
    hi, hi_rec = _club("HI", "Lucky", 70, 10, 2, 250, 200)
    lo, lo_rec = _club("LO", "Unlucky", 30, 46, 6, 200, 250)
    table = _standings([(hi, hi_rec), (lo, lo_rec)])
    games = _xg_games(hi_xgf=2.05, lo_xgf=3.35)

    def factory(i):
        return _live_session(
            {"HI": 0.51, "LO": 0.51},
            standings=table,
            game_results=games,
            session_id=f"xg{i}",
        )

    series_rate = live_series_high_win_rate(factory)
    print(
        f"\nLive lucky 70-win (xG 2.05) vs unlucky 30-win (xG 3.35): series={series_rate:.1%}"
    )
    assert series_rate <= 0.84, (
        f"analytics did not give the 30-win team a real path: 70-win series {series_rate:.1%}"
    )
    assert series_rate >= 0.45


def test_typical_one_vs_eight_still_allows_upsets():
    hi, hi_rec = _club("HI", "Presidents", 54, 20, 8, 265, 210)
    lo, lo_rec = _club("LO", "Wildcard", 41, 30, 11, 230, 235)
    table = _standings([(hi, hi_rec), (lo, lo_rec)])

    def factory(i):
        return _live_session({"HI": 0.52, "LO": 0.52}, standings=table, session_id=f"1v8-{i}")

    series_rate = live_series_high_win_rate(factory)
    print(f"\nLive typical 1 vs 8 (54 vs 41 wins): series={series_rate:.1%}")
    assert 0.70 <= series_rate <= 0.88, f"1v8 series rate off NHL range: {series_rate:.1%}"


def test_atomic_series_uses_regular_season_results():
    hi, hi_rec = _club("HI", "Titans", 70, 10, 2, 310, 175)
    lo, lo_rec = _club("LO", "Tankers", 30, 46, 6, 175, 310)
    table = _standings([(hi, hi_rec), (lo, lo_rec)])
    smap = _augment_strength_with_results(
        [hi, lo],
        table,
        {"HI": 0.52, "LO": 0.50},
    )
    gap = smap["HI"] - smap["LO"]
    p_game = playoff_game_win_probability(smap["HI"], smap["LO"])
    series_rate = atomic_series_high_win_rate(smap)
    print(
        f"\nAtomic 70 vs 30: strength gap={gap:.3f} P(game)={p_game:.3f} series={series_rate:.1%}"
    )
    assert gap >= 0.22
    assert p_game >= 0.78
    assert series_rate >= 0.88


def test_atomic_analytics_override_when_xg_disagrees_with_record():
    hi, hi_rec = _club("HI", "Lucky", 70, 10, 2, 250, 200, season_xgf=160.0, season_xga=250.0)
    lo, lo_rec = _club("LO", "Unlucky", 30, 46, 6, 200, 250, season_xgf=250.0, season_xga=160.0)
    table = _standings([(hi, hi_rec), (lo, lo_rec)])
    smap = _augment_strength_with_results(
        [hi, lo],
        table,
        {"HI": 0.50, "LO": 0.50},
    )
    series_rate = atomic_series_high_win_rate(smap)
    print(
        f"\nAtomic lucky 70 vs unlucky 30 with flipped xG: "
        f"HI={smap['HI']:.3f} LO={smap['LO']:.3f} series={series_rate:.1%}"
    )
    assert series_rate < 0.88
    assert series_rate > 0.40


def test_playoff_strength_report(capsys):
    """Print a compact Monte Carlo report for the mismatch the user is seeing."""
    rows = []

    even_s = atomic_series_high_win_rate({"HI": 0.50, "LO": 0.50}, n=300, seed=1)
    rows.append(("equal roster / no record blend", even_s))

    hi, hi_rec = _club("HI", "70-win", 70, 10, 2, 310, 175)
    lo, lo_rec = _club("LO", "30-win", 30, 46, 6, 175, 310)
    table = _standings([(hi, hi_rec), (lo, lo_rec)])
    smap = _augment_strength_with_results([hi, lo], table, {"HI": 0.52, "LO": 0.50})
    rows.append(("70-win vs 30-win (atomic, records)", atomic_series_high_win_rate(smap, n=300, seed=2)))

    live = live_series_high_win_rate(
        lambda i: _live_session({"HI": 0.52, "LO": 0.50}, standings=table, session_id=f"rpt{i}"),
        n=300,
    )
    rows.append(("70-win vs 30-win (live franchise games)", live))

    print("\nPlayoff favorite series win rate (best-of-7)")
    print("-" * 56)
    for label, rate in rows:
        print(f"  {label:<42} {rate:6.1%}")
    print("-" * 56)

    captured = capsys.readouterr()
    assert "70-win vs 30-win" in captured.out
