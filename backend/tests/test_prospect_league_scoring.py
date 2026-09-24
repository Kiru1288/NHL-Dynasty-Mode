"""
Smoke tests for junior/prospect league scoring environments.

Run: python -m pytest backend/tests/test_prospect_league_scoring.py -q
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "SimEngine"
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from app.sim_engine.generation.prospect_league_scoring import (  # noqa: E402
    advance_prospect_stats_to_date,
    ensure_prospect_season_stats,
    expected_games_for_date,
    generate_prospect_scoring_line,
    get_league_scoring_profile,
    initialize_prospect_season,
    normalize_prospect_league_key,
    normalize_league_leader_board,
)


def _prospect(
    *,
    pid: str = "p1",
    age: int = 18,
    ovr: float = 0.48,
    position: str = "LW",
    pipeline_bust: bool = False,
    pipeline_steal: bool = False,
    dev_type: str = "",
    archetype: str = "scoring forward",
    ratings: dict | None = None,
):
    return SimpleNamespace(
        id=pid,
        rng_seed=abs(hash(pid)) % 100000,
        identity=SimpleNamespace(age=age, position=SimpleNamespace(value=position)),
        position=SimpleNamespace(value=position),
        ovr=ovr,
        ratings=ratings or {"shooting_accuracy": 78, "passing_accuracy": 72, "puck_handling": 74},
        archetype=archetype,
        playstyle=archetype,
        pipeline_bust=pipeline_bust,
        pipeline_steal=pipeline_steal,
        dev_type=dev_type,
        psychology=SimpleNamespace(coachability=0.32 if pipeline_bust else 0.62, anxiety=0.55 if pipeline_bust else 0.3),
        traits={"tags": ["boom_bust"]} if dev_type == "volatile" else {},
        chemistry_profile={"personality": "volatile"} if pipeline_bust else {},
        _dev_archetype="HIGH_VARIANCE" if dev_type == "volatile" else "",
        _pipeline_dev_curve="boom_bust" if dev_type == "volatile" else "",
    )


def test_league_aliases_and_profiles():
    assert normalize_prospect_league_key("CHL_OHL") == "OHL"
    assert normalize_prospect_league_key("QMJHL/Q") == "QMJHL"
    assert normalize_prospect_league_key("NCAA Division I cluster") == "NCAA"
    chl = get_league_scoring_profile("CHL_QMJHL")
    ncaa = get_league_scoring_profile("NCAA")
    assert chl["scoring_multiplier"] > ncaa["scoring_multiplier"]
    assert chl["difficulty"] < ncaa["difficulty"]


def test_chl_top_forward_can_hit_elite_ppg():
    rng = random.Random(42)
    lines = []
    for i in range(40):
        p = _prospect(pid=f"elite-{i}", ovr=0.38 + i * 0.004, age=18 + (i % 3))
        line = generate_prospect_scoring_line(p, "CHL_OHL", rng=rng)
        lines.append({**line, "position": "LW"})
    normalize_league_leader_board(lines, "CHL_OHL", rng=rng)
    top_ppg = max(r["ppg"] for r in lines)
    assert top_ppg >= 1.65, f"CHL top PPG too low: {top_ppg}"


def test_risky_non_elite_can_score_big_in_junior():
    rng = random.Random(7)
    highs = []
    for i in range(25):
        p = _prospect(
            pid=f"risk-{i}",
            ovr=0.36,
            age=19,
            pipeline_bust=True,
            dev_type="volatile",
            archetype="sniper",
        )
        initialize_prospect_season(p, "QMJHL", rng=rng, force=True)
        line = advance_prospect_stats_to_date(p, "QMJHL", "2026-04-15", rng=rng)
        highs.append(line["ppg"])
    assert max(highs) >= 1.2
    assert getattr(p, "translation_risk", "") in ("Medium", "High")


def test_september_gp_low_april_near_complete():
    rng = random.Random(21)
    p = _prospect(pid="cal-1", ovr=0.52, age=18)
    initialize_prospect_season(p, "CHL_OHL", rng=rng, force=True)
    sep = advance_prospect_stats_to_date(p, "CHL_OHL", "2025-09-15", rng=rng)
    apr = advance_prospect_stats_to_date(p, "CHL_OHL", "2026-04-15", rng=rng)
    # ~10% of a 58-68 GP season has elapsed by Sep 15 (6-7 GP depending on season length).
    assert sep["gp"] <= 8
    assert apr["gp"] >= 45
    assert sep["ppg"] != apr["ppg"] or sep["points"] != apr["points"]


def test_ncaa_and_shl_not_inflated_like_chl():
    rng = random.Random(99)
    chl_ppgs = []
    ncaa_ppgs = []
    shl_ppgs = []
    for i in range(30):
        p = _prospect(pid=f"cmp-{i}", ovr=0.50)
        chl_ppgs.append(generate_prospect_scoring_line(p, "CHL_WHL", rng=rng)["ppg"])
        ncaa_ppgs.append(generate_prospect_scoring_line(p, "NCAA", rng=rng)["ppg"])
        shl_ppgs.append(generate_prospect_scoring_line(p, "EU_J_SHL", rng=rng)["ppg"])
    assert max(chl_ppgs) > max(ncaa_ppgs) + 0.25
    assert max(ncaa_ppgs) > max(shl_ppgs) + 0.15


def test_goalies_do_not_get_forward_points():
    rng = random.Random(3)
    p = _prospect(pid="g1", position="G", ovr=0.44)
    line = generate_prospect_scoring_line(p, "OHL", rng=rng)
    assert line["points"] == 0
    assert line["goals"] == 0
    assert line.get("save_pct") is not None


def test_offensive_defenseman_chl_range():
    rng = random.Random(11)
    p = _prospect(pid="d1", position="D", ovr=0.52, archetype="offensive defenseman")
    line = generate_prospect_scoring_line(p, "CHL_OHL", rng=rng)
    assert 0.35 <= line["ppg"] <= 1.50


# ---------------------------------------------------------------------------
# Integrity regressions: cadence independence, accumulation, stints, goalies, leaks
# ---------------------------------------------------------------------------
import datetime as _dt  # noqa: E402
import statistics as _st  # noqa: E402

from app.sim_engine.generation import prospect_league_scoring as _pls  # noqa: E402


def _season_dates(step_days: int):
    d = _dt.date(2025, 9, 15)
    end = _dt.date(2026, 4, 15)
    while d <= end:
        yield d.isoformat()
        d += _dt.timedelta(days=step_days)


def _fresh(pid: str, **kw):
    kw.setdefault("ratings", {"shooting_accuracy": 60, "passing_accuracy": 58, "puck_handling": 60})
    return _prospect(pid=pid, **kw)


def _run_skater(pid: str, code: str, step_days: int, seed: int = 5, **kw):
    p = _fresh(pid, **kw)
    rng = random.Random(seed)
    initialize_prospect_season(p, code, rng=rng, season_year=2025, force=True)
    line = {}
    for iso in _season_dates(step_days):
        line = advance_prospect_stats_to_date(p, code, iso, rng=rng, season_year=2025)
    return p, line


def test_ppg_target_is_deterministic_and_never_ratchets():
    p = _fresh("det-1", age=19)
    initialize_prospect_season(p, "CHL_OHL", rng=random.Random(1), season_year=2025, force=True)
    first = _pls.calculate_prospect_ppg_scale(p, "CHL_OHL")
    # Many calls / many RNG states must return the identical target.
    for i in range(50):
        assert _pls.calculate_prospect_ppg_scale(p, "CHL_OHL", random.Random(i)) == first
    for iso in _season_dates(1):
        advance_prospect_stats_to_date(p, "CHL_OHL", iso, rng=random.Random(2), season_year=2025)
    assert p._prospect_expected_ppg == first


def test_season_totals_do_not_depend_on_update_cadence():
    means = {}
    for step in (200, 7, 1):
        ppgs = [_run_skater(f"cad-{i}", "CHL_OHL", step, seed=i, age=18 + i % 3)[1]["ppg"] for i in range(80)]
        means[step] = _st.mean(ppgs)
    base = means[200]
    for step, val in means.items():
        assert abs(val - base) / base < 0.10, f"cadence {step}d shifted mean PPG: {means}"


def test_goals_and_assists_never_decrease():
    for i in range(40):
        p = _fresh(f"mono-{i}", age=18 + i % 3, archetype="sniper" if i % 2 else "playmaker")
        rng = random.Random(i)
        initialize_prospect_season(p, "CHL_OHL", rng=rng, season_year=2025, force=True)
        pg = pa = pp = 0
        for iso in _season_dates(9):
            line = advance_prospect_stats_to_date(p, "CHL_OHL", iso, rng=rng, season_year=2025)
            assert line["goals"] >= pg and line["assists"] >= pa and line["points"] >= pp
            assert line["goals"] + line["assists"] == line["points"]
            pg, pa, pp = line["goals"], line["assists"], line["points"]


def test_injuries_cost_games_and_are_not_replayed():
    p = _fresh("inj-1", age=19)
    p.traits = {"injury_risk_mod": 0.6}  # hazard clamps high -> injuries are near-certain
    rng = random.Random(9)
    initialize_prospect_season(p, "CHL_OHL", rng=rng, season_year=2025, force=True)
    line = {}
    for iso in _season_dates(3):
        line = advance_prospect_stats_to_date(p, "CHL_OHL", iso, rng=rng, season_year=2025)
    expected = expected_games_for_date("CHL_OHL", p._prospect_projected_stats["gp"], "2026-04-15")
    assert line.get("gp_missed", 0) > 0
    assert line["gp"] < expected  # injuries actually reduce games played
    assert line["gp"] + line["gp_missed"] <= expected + 1  # ...and are never simulated later


def test_returning_player_is_not_backfilled_and_stint_is_archived():
    p = _fresh("stint-1", age=22, ovr=0.6)
    rng = random.Random(4)
    initialize_prospect_season(p, "AHL", rng=rng, season_year=2025, force=True)
    advance_prospect_stats_to_date(p, "AHL", "2025-12-15", rng=rng, season_year=2025)
    before = p._prospect_season_stats["gp"]
    assert before > 15
    # Called up to the NHL: stint closes, line is filed away.
    assert _pls.end_prospect_stint(p, reason="left_level")
    assert p.prospect_stat_history and p.prospect_stat_history[-1]["gp"] == before
    # Sent back down in March: no games for the months spent in the NHL.
    line = advance_prospect_stats_to_date(p, "AHL", "2026-03-15", rng=rng, season_year=2025)
    assert line["gp"] == 0
    line = advance_prospect_stats_to_date(p, "AHL", "2026-04-15", rng=rng, season_year=2025)
    total = before + line["gp"]
    proj = p._prospect_projected_stats["gp"]
    assert line["gp"] < before
    assert total <= proj


def test_league_change_opens_new_stint_without_backfill():
    p = _fresh("stint-2", age=19)
    rng = random.Random(6)
    initialize_prospect_season(p, "CHL_OHL", rng=rng, season_year=2025, force=True)
    advance_prospect_stats_to_date(p, "CHL_OHL", "2026-01-15", rng=rng, season_year=2025)
    junior_gp = p._prospect_season_stats["gp"]
    line = advance_prospect_stats_to_date(p, "AHL", "2026-01-20", rng=rng, season_year=2025)
    assert line["gp"] <= 3  # only the days since he arrived
    assert p.prospect_stat_history[-1]["league"] == "OHL"
    assert p.prospect_stat_history[-1]["gp"] == junior_gp


def test_new_season_archives_previous_season():
    p = _fresh("hist-1", age=18)
    rng = random.Random(8)
    initialize_prospect_season(p, "CHL_OHL", rng=rng, season_year=2025, force=True)
    advance_prospect_stats_to_date(p, "CHL_OHL", "2026-04-15", rng=rng, season_year=2025)
    gp = p._prospect_season_stats["gp"]
    advance_prospect_stats_to_date(p, "CHL_OHL", "2026-10-15", rng=rng, season_year=2026)
    assert [h["season_year"] for h in p.prospect_stat_history] == [2025]
    assert p.prospect_stat_history[0]["gp"] == gp
    assert p._prospect_season_stats["gp"] < gp


def _fake_sim(ahl_players):
    tm = SimpleNamespace(ahl_roster=list(ahl_players), echl_roster=[])
    return SimpleNamespace(league=SimpleNamespace(teams=[tm], development_leagues=[]), rng=random.Random(1)), tm


def test_bulk_sync_tracks_call_ups_from_live_rosters():
    stay = _fresh("sync-stay", age=23, ovr=0.6)
    away = _fresh("sync-away", age=23, ovr=0.6)
    sim, tm = _fake_sim([stay, away])
    isos = ["2025-10-15", "2025-11-15", "2025-12-15", "2026-01-15", "2026-02-15", "2026-03-15", "2026-04-15"]
    for iso in isos[:3]:
        _pls.advance_all_development_league_stats(sim, iso, season_year=2025)
    away_gp_at_callup = away._prospect_season_stats["gp"]
    tm.ahl_roster = [stay]  # away is called up
    for iso in isos[3:5]:
        _pls.advance_all_development_league_stats(sim, iso, season_year=2025)
    assert away._prospect_season_stats["gp"] == away_gp_at_callup  # no AHL games while in the NHL
    tm.ahl_roster = [stay, away]  # sent back down
    for iso in isos[5:]:
        _pls.advance_all_development_league_stats(sim, iso, season_year=2025)
    proj = away._prospect_projected_stats["gp"]
    assert away._prospect_season_stats["gp"] < proj * 0.35  # arrived late; not credited the gap
    assert stay._prospect_season_stats["gp"] > away_gp_at_callup


def test_ahl_and_echl_players_of_any_age_get_lines():
    old = _fresh("vet-1", age=31, ovr=0.6)
    sim, tm = _fake_sim([old])
    _pls.advance_all_development_league_stats(sim, "2026-02-15", season_year=2025)
    assert old._prospect_season_stats["gp"] > 0


def test_goalie_line_is_consistent_and_not_a_full_season():
    gps = []
    for i in range(60):
        g = _prospect(pid=f"gk-{i}", position="G", ovr=0.45, ratings={"g_positioning": 60, "g_reflexes": 58})
        rng = random.Random(i)
        initialize_prospect_season(g, "AHL", rng=rng, season_year=2025, force=True)
        line = advance_prospect_stats_to_date(g, "AHL", "2026-04-15", rng=rng, season_year=2025)
        if line["gp"] and line.get("shots_against"):
            derived_sv = 1.0 - line["goals_against"] / line["shots_against"]
            assert abs(line["save_pct"] - round(derived_sv, 3)) < 0.0011
            assert abs(line["gaa"] - round(line["goals_against"] / line["gp"], 2)) < 0.011
            assert line["wins"] + line["losses"] + line["ot_losses"] == line["gp"]
        gps.append(line["gp"])
    assert _st.mean(gps) < 45  # goalies share the crease; not every goalie plays ~62


def test_hidden_potential_cannot_change_public_talent_or_goalie_lines():
    base_ratings = {"shooting_accuracy": 62, "passing_accuracy": 60, "puck_handling": 61, "g_positioning": 60}
    a = _prospect(pid="leak-a", ovr=0.5, ratings=dict(base_ratings))
    b = _prospect(pid="leak-a", ovr=0.5, ratings=dict(base_ratings))  # same id/seed/ratings
    b.draft_value_range = (0.95, 0.99)
    b.potential = 0.99
    b.pipeline_tier = "transcendent"
    b.is_transcendent = True
    b.ratings["dev_potential"] = 99
    assert _pls._offensive_talent_score(a) == _pls._offensive_talent_score(b)
    assert _pls._defensive_talent_score(a) == _pls._defensive_talent_score(b)
    line = {"gp": 40, "goals": 15, "assists": 20, "points": 35, "ppg": 0.875, "pim": 10}
    assert _pls.derive_prospect_analytics(a, "CHL_OHL", line) == _pls.derive_prospect_analytics(b, "CHL_OHL", line)

    ga = _prospect(pid="leak-g", position="G", ovr=0.45, ratings={"g_positioning": 60})
    gb = _prospect(pid="leak-g", position="G", ovr=0.45, ratings={"g_positioning": 60})
    gb.draft_value_range = (0.95, 0.99)
    gb.pipeline_tier = "franchise"
    gb.is_transcendent = True
    la = generate_prospect_scoring_line(ga, "AHL", games_played=60, rng=random.Random(1))
    lb = generate_prospect_scoring_line(gb, "AHL", games_played=60, rng=random.Random(1))
    assert la == lb


def test_talent_drives_scoring_within_a_league():
    rng = random.Random(3)
    lo, hi = [], []
    for i in range(80):
        weak = _prospect(pid=f"w{i}", ovr=0.40, ratings={"shooting_accuracy": 40, "passing_accuracy": 40, "puck_handling": 40})
        strong = _prospect(pid=f"s{i}", ovr=0.60, ratings={"shooting_accuracy": 70, "passing_accuracy": 70, "puck_handling": 70})
        lo.append(_pls.calculate_prospect_ppg_scale(weak, "CHL_OHL", rng))
        hi.append(_pls.calculate_prospect_ppg_scale(strong, "CHL_OHL", rng))
    assert _st.mean(hi) > _st.mean(lo) * 1.5


def test_ncaa_stays_on_the_stat_sync_until_24():
    assert _pls.development_league_stat_max_age("NCAA") == 24
    assert _pls.development_league_stat_max_age("CHL_OHL") == 20
    assert _pls.development_league_stat_max_age("EU_J_SHL") == 20


def test_projection_and_live_target_share_the_same_draw():
    # Bootstrap-style init: the calendar year is not known yet.
    p = _fresh("pin-1", age=19)
    initialize_prospect_season(p, "CHL_OHL", rng=random.Random(1), force=True)
    projected = p._prospect_expected_ppg
    advance_prospect_stats_to_date(p, "CHL_OHL", "2025-12-15", rng=random.Random(1), season_year=2025)
    assert p._prospect_expected_ppg == projected
