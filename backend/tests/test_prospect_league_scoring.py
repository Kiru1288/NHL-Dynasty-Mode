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
    assert sep["gp"] <= 6
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
