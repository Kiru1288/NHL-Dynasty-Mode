"""
Popup / notification integrity: streak headlines must use a verified streak, and trade
popups must carry real player data (or say plainly that there is none).

Run: cd backend && python -m pytest tests/test_popup_info_integrity.py -q
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "SimEngine", ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.sim_engine.franchise.storyline_engine import _team_current_streak  # noqa: E402
from app.sim_engine.franchise.storyline_procedural import compose_data_story_copy  # noqa: E402


def _game(home, away, hg, ag):
    return {"home_id": home, "away_id": away, "home_goals": hg, "away_goals": ag}


def test_current_streak_counts_consecutive_results_for_one_team():
    games = [
        _game("A", "B", 1, 4),  # A loses (older)
        _game("A", "C", 3, 2),  # A wins
        _game("D", "A", 1, 2),  # A wins (away)
        _game("A", "E", 5, 0),  # A wins
        _game("F", "G", 9, 0),  # unrelated
    ]
    session = SimpleNamespace(game_results=games)
    assert _team_current_streak(session, "A") == ("W", 3)
    assert _team_current_streak(session, "B") == ("W", 1)
    # Overtime/regulation losses both end a win streak and start a losing one.
    session.game_results.append(_game("A", "H", 2, 3))
    assert _team_current_streak(session, "A") == ("L", 1)


def test_no_games_means_no_streak_claim():
    assert _team_current_streak(SimpleNamespace(game_results=[]), "A") == ("", 0)
    assert _team_current_streak(SimpleNamespace(), "A") == ("", 0)


def test_streak_headlines_never_print_a_zero_game_streak():
    rng = random.Random(1)
    for stype in ("win_streak", "losing_skid"):
        for streak in (0, 1, None):
            ctx = {"team": "Ottawa Senators", "record": "10-8-2"}
            if streak is not None:
                ctx["streak"] = streak
            for body in (False, True):
                text = compose_data_story_copy(stype, ctx, rng, body=body)
                assert "0-game" not in text and " 0 straight" not in text and "1-game" not in text
                assert "Ottawa Senators" in text and "10-8-2" in text


def test_verified_streak_is_still_reported():
    rng = random.Random(1)
    ctx = {"team": "Boston Bruins", "record": "20-5-1", "streak": 7}
    assert "7-game win streak" in compose_data_story_copy("win_streak", ctx, rng)
    assert "7 straight" in compose_data_story_copy("win_streak", ctx, rng, body=True)
    assert "7-game skid" in compose_data_story_copy("losing_skid", ctx, rng)


# ---------------------------------------------------------------- trade popup player data
from services import franchise_sim as fs  # noqa: E402


def _player(pos="C", **attrs):
    ident = SimpleNamespace(position=SimpleNamespace(value=pos), name="Test Player", age=24)
    return SimpleNamespace(id="p1", identity=ident, **attrs)


def _session(stats=None, year=2025):
    return SimpleNamespace(player_season_stats=stats or {}, season_calendar_year=year)


def test_trade_stats_current_nhl_skater_is_labelled_and_complete():
    sess = _session({"p1": {"gp": 40, "g": 12, "a": 20, "pts": 32}})
    out = fs._trade_popup_season_stats(sess, "p1", _player("C"))
    assert out["kind"] == "skater" and out["source"] == "nhl_current"
    assert out["label"] == "2025-26 NHL"
    assert (out["gp"], out["g"], out["a"], out["pts"]) == (40, 12, 20, 32)


def test_trade_stats_goalie_gets_goalie_line_not_skater_zeros():
    sess = _session({"p1": {"gp": 30, "w": 18, "l": 9, "otl": 3, "sa": 800, "saves": 736, "ga": 64, "toi_sec": 30 * 3600, "so": 2, "position": "G"}})
    out = fs._trade_popup_season_stats(sess, "p1", _player("G"))
    assert out["kind"] == "goalie"
    assert out["gp"] == 30 and out["w"] == 18 and out["so"] == 2
    assert abs(out["sv_pct"] - 0.92) < 0.001
    assert "g" not in out and "pts" not in out


def test_trade_stats_prospect_uses_minor_league_line():
    prospect = _player(
        "LW",
        _prospect_season_stats={"gp": 22, "goals": 9, "assists": 11, "points": 20},
        _prospect_stint={"league_key": "OHL", "open": True},
    )
    out = fs._trade_popup_season_stats(_session(), "p1", prospect)
    assert out["source"] == "minors" and out["label"] == "2025-26 OHL"
    assert (out["gp"], out["g"], out["a"], out["pts"]) == (22, 9, 11, 20)


def test_trade_stats_falls_back_to_last_nhl_season_then_to_none():
    veteran = _player(
        "D",
        career_stats={"seasons": [
            {"season": "2023-24", "league": "NHL", "gp": 70, "g": 5, "a": 25, "pts": 30},
            {"season": "2024-25", "league": "AHL", "gp": 10, "g": 1, "a": 1, "pts": 2},
        ]},
    )
    out = fs._trade_popup_season_stats(_session(), "p1", veteran)
    assert out["source"] == "nhl_prior" and out["label"] == "2023-24 NHL" and out["pts"] == 30

    out = fs._trade_popup_season_stats(_session(), "p1", _player("D"))
    assert out["source"] == "none" and out["note"] == "No games played this season"


def test_light_bulk_trade_payload_is_flagged_so_popups_can_resolve_it():
    ev = {"team": "1", "from_team_id": "2", "outgoing": ["A Player"], "incoming": ["B Player"], "execution": {}}
    sess = SimpleNamespace(
        team_by_id={}, player_season_stats={}, season_calendar_year=2025,
        cpu_franchise_profiles={}, league=None,
    )
    light = fs._compose_cpu_trade_wire_payload(sess, ev, calendar_idx=3, iso="2025-11-01", resolve_players=False)
    full = fs._compose_cpu_trade_wire_payload(sess, ev, calendar_idx=3, iso="2025-11-01", resolve_players=True)
    assert light["assets_resolved"] is False
    assert full["assets_resolved"] is True
