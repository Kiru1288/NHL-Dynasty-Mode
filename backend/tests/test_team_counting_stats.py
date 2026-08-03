"""Team counting stats must use player-ledger SOG for light CPU–CPU seasons."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SimEngine" / "app"))
sys.path.insert(0, str(ROOT / "backend"))

from services.franchise_sim import _build_team_analytics_rows  # noqa: E402


def test_team_diff_uses_standings_gf_ga_fields():
    """Standings records expose gf/ga — analytics must not look for goals_for."""
    session = SimpleNamespace(
        standings=SimpleNamespace(
            records={
                "CAR": SimpleNamespace(
                    gf=220, ga=280, wins=31, losses=46, otl=5, gp=82, points=67
                ),
            }
        ),
        team_by_id={"CAR": SimpleNamespace(name="Carolina Hurricanes", abbrev="CAR")},
        player_season_stats={
            # Poisoned ledger that used to invent +161 DIFF when standings GF was ignored.
            "c1": {
                "player_id": "c1",
                "team_id": "CAR",
                "position": "C",
                "g": 40,
                "a": 50,
                "sog": 200,
                "gp": 82,
                "stat_scope": "regular_season",
            },
            "c2": {
                "player_id": "c2",
                "team_id": "CAR",
                "position": "G",
                "ga": 40,
                "shots_against": 2200,
                "saves": 2160,
                "gp": 70,
                "stat_scope": "regular_season",
            },
        },
        game_results=[],
    )
    rows = {r["team_id"]: r for r in _build_team_analytics_rows(session)}
    car = rows["CAR"]
    assert int(car["gf"]) == 220
    assert int(car["ga"]) == 280
    assert int(car["diff"]) == -60
    assert float(car["sv_pct"]) == round(2160 / 2200, 4)


def test_team_sf_prefers_player_ledger_over_sparse_event_boxes():
    """CPU SF comes from player ledger, not the handful of full-event boxes vs user."""
    session = SimpleNamespace(
        standings=SimpleNamespace(
            records={
                "CPU": SimpleNamespace(
                    gf=250, ga=240, wins=40, losses=30, otl=12, gp=82, points=92
                ),
                "USER": SimpleNamespace(
                    gf=260, ga=250, wins=38, losses=35, otl=9, gp=82, points=85
                ),
            }
        ),
        team_by_id={
            "CPU": SimpleNamespace(name="CPU Club", abbrev="CPU"),
            "USER": SimpleNamespace(name="User Club", abbrev="USR"),
        },
        player_season_stats={
            "c1": {"player_id": "c1", "team_id": "CPU", "position": "C", "g": 30, "a": 40, "pts": 70, "sog": 2200, "gp": 82, "stat_scope": "regular_season"},
            "c2": {"player_id": "c2", "team_id": "CPU", "position": "G", "ga": 240, "shots_against": 2400, "saves": 2160, "gp": 70, "stat_scope": "regular_season"},
            "u1": {"player_id": "u1", "team_id": "USER", "position": "C", "g": 28, "a": 45, "pts": 73, "sog": 2100, "gp": 82, "stat_scope": "regular_season"},
            "u2": {"player_id": "u2", "team_id": "USER", "position": "G", "ga": 250, "shots_against": 2500, "saves": 2250, "gp": 72, "stat_scope": "regular_season"},
        },
        game_results=[
            {
                "stat_scope": "regular_season",
                "home_id": "USER",
                "away_id": "CPU",
                "home_shots": 30,
                "away_shots": 28,
                "home_cf": 55,
                "away_cf": 50,
                "home_xgf": 2.5,
                "away_xgf": 2.2,
                "home_pp_goals": 1,
                "away_pp_goals": 0,
                "home_ppo": 3,
                "away_ppo": 3,
            }
            for _ in range(3)
        ],
    )

    rows = {r["team_id"]: r for r in _build_team_analytics_rows(session)}
    cpu = rows["CPU"]
    assert int(cpu["sf"]) >= 2000, cpu["sf"]
    assert int(cpu["sa"]) >= 2000, cpu["sa"]
    assert float(cpu["sh_pct"]) > 0.05
    assert float(cpu["sh_pct"]) < 0.25
    assert float(cpu["sv_pct"]) > 0.85
    assert float(cpu["sv_pct"]) < 0.95
    assert int(cpu["goal_diff"]) == 10
    assert int(cpu["diff"]) == 10


def test_team_cf_prefers_player_ledger_over_sparse_events():
    session = SimpleNamespace(
        standings=SimpleNamespace(
            records={
                "CPU": SimpleNamespace(
                    gf=250, ga=200, wins=50, losses=25, otl=7, gp=82, points=107
                ),
            }
        ),
        team_by_id={"CPU": SimpleNamespace(name="CPU Club", abbrev="CPU")},
        player_season_stats={
            "c1": {
                "player_id": "c1",
                "team_id": "CPU",
                "position": "C",
                "g": 40,
                "a": 50,
                "pts": 90,
                "sog": 250,
                "gp": 82,
                "cf": 5000,
                "ca": 4200,
                "ff": 3900,
                "fa": 3300,
                "xgf": 220,
                "xga": 180,
                "stat_scope": "regular_season",
            },
            "c2": {
                "player_id": "c2",
                "team_id": "CPU",
                "position": "G",
                "ga": 200,
                "shots_against": 2400,
                "saves": 2200,
                "gp": 70,
                "stat_scope": "regular_season",
            },
        },
        game_results=[
            {
                "stat_scope": "regular_season",
                "home_id": "CPU",
                "away_id": "USR",
                "home_shots": 28,
                "away_shots": 30,
                "home_cf": 35,
                "away_cf": 80,
                "home_xgf": 1.2,
                "away_xgf": 4.5,
            }
            for _ in range(3)
        ],
    )
    rows = {r["team_id"]: r for r in _build_team_analytics_rows(session)}
    cpu = rows["CPU"]
    assert float(cpu["cf_pct"]) > 0.50, cpu["cf_pct"]
    assert float(cpu["xgf_pct"]) > 0.50, cpu["xgf_pct"]
    assert "player_season_stats" in str(cpu.get("team_event_stats_source") or "")
    # On-ice ×5 descaled: 220/5 = 44, not raw 220.
    assert float(cpu["xgf"]) < 100, cpu["xgf"]


def test_light_cpu_cpu_boxes_count_toward_team_cf():
    """CPU–CPU light boxes with CF/xGF must drive season possession, not only vs-user games."""
    from services.franchise_sim import _purge_synthetic_universe_artifacts  # noqa: E402

    light_games = []
    for i in range(70):
        light_games.append(
            {
                "stat_scope": "regular_season",
                "home_id": "CPU",
                "away_id": "CPU2",
                "home_shots": 28,
                "away_shots": 26,
                "home_cf": 55,
                "away_cf": 48,
                "home_ff": 42,
                "away_ff": 38,
                "home_xgf": 3.1,
                "away_xgf": 2.6,
                "light_box": True,
                "stat_source": "light_strength",
                "home_goals": 3,
                "away_goals": 2,
                "status": "final",
            }
        )
    # A few full-event games with inverted possession (the old "sparse overwrite" trap).
    for _ in range(3):
        light_games.append(
            {
                "stat_scope": "regular_season",
                "home_id": "CPU",
                "away_id": "USR",
                "home_shots": 20,
                "away_shots": 35,
                "home_cf": 30,
                "away_cf": 70,
                "home_xgf": 1.0,
                "away_xgf": 4.0,
                "home_goals": 4,
                "away_goals": 1,
                "status": "final",
            }
        )

    session = SimpleNamespace(
        standings=SimpleNamespace(
            records={
                "CPU": SimpleNamespace(
                    gf=250, ga=200, wins=50, losses=25, otl=7, gp=73, points=107
                ),
                "CPU2": SimpleNamespace(
                    gf=200, ga=250, wins=25, losses=40, otl=8, gp=70, points=58
                ),
            }
        ),
        team_by_id={
            "CPU": SimpleNamespace(name="CPU Club", abbrev="CPU"),
            "CPU2": SimpleNamespace(name="CPU2 Club", abbrev="CP2"),
        },
        player_season_stats={
            "c1": {
                "player_id": "c1",
                "team_id": "CPU",
                "position": "C",
                "g": 30,
                "a": 40,
                "sog": 2000,
                "gp": 73,
                "stat_scope": "regular_season",
            },
            "c2": {
                "player_id": "c2",
                "team_id": "CPU",
                "position": "G",
                "ga": 200,
                "shots_against": 2000,
                "saves": 1800,
                "gp": 70,
                "stat_scope": "regular_season",
            },
        },
        game_results=light_games,
        _synthetic_universe_purged_v1=False,
        _synthetic_universe_purged_v2=False,
    )

    # Purge must NOT strip light_strength CF (old v1 bug).
    _purge_synthetic_universe_artifacts(session)
    assert float(session.game_results[0].get("home_cf") or 0) == 55

    rows = {r["team_id"]: r for r in _build_team_analytics_rows(session)}
    cpu = rows["CPU"]
    assert int(cpu.get("team_light_games") or 0) == 70
    assert int(cpu.get("team_event_games") or 0) >= 70
    assert float(cpu["cf_pct"]) > 0.50, cpu["cf_pct"]
    assert str(cpu.get("team_event_stats_source") or "") == "game_results"
