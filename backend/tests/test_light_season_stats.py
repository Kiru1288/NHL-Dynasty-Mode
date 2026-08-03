"""Light-path NHL season ledger: TOI, +/-, goalie GAA/GSAx, peripherals."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SimEngine" / "app"))
sys.path.insert(0, str(ROOT / "backend"))

from app.sim_engine.engine import SimEngine  # noqa: E402
from app.sim_engine.generation.player_analytics import (  # noqa: E402
    normalize_goalie_counting_stats,
    normalize_skater_counting_stats,
)


def _skater(pid: str, name: str, pos: str, ovr: int = 80) -> SimpleNamespace:
    return SimpleNamespace(
        id=pid,
        name=name,
        position=pos,
        identity=SimpleNamespace(name=name),
        overall=ovr,
        ratings={"skating": ovr, "shooting": ovr, "hands": ovr, "checking": ovr, "defense": ovr, "IQ": ovr},
    )


def _goalie(pid: str, name: str, ovr: int = 78) -> SimpleNamespace:
    return SimpleNamespace(
        id=pid,
        name=name,
        position="G",
        identity=SimpleNamespace(name=name),
        overall=ovr,
        ratings={"glove": ovr, "blocker": ovr, "rebound": ovr, "IQ": ovr},
    )


def _team(tid: str, skaters, goalies) -> SimpleNamespace:
    return SimpleNamespace(
        id=tid,
        team_id=tid,
        roster=list(skaters) + list(goalies),
        name=f"Team {tid}",
    )


def test_light_strength_writes_plus_minus_toi_and_goalie_xga():
    sim = SimEngine(seed=99, debug=False, populate_initial_rosters=False)
    rng = random.Random(99)

    home_sk = [_skater(f"h{i}", f"Home {i}", "C" if i < 8 else "D", 75 + i) for i in range(18)]
    away_sk = [_skater(f"a{i}", f"Away {i}", "C" if i < 8 else "D", 74 + i) for i in range(18)]
    home_g = [_goalie("hg1", "Home Goalie")]
    away_g = [_goalie("ag1", "Away Goalie")]
    home = _team("H", home_sk, home_g)
    away = _team("A", away_sk, away_g)

    # Monkeypatch lineup helpers to avoid full roster infrastructure.
    sim._gm_build_dressed_lineup = lambda team, _rng: (
        (home_sk if team is home else away_sk),
        (home_g if team is home else away_g),
        [],
        set(),
    )
    sim._gm_skaters = lambda team: list(home_sk if team is home else away_sk)
    sim._gm_goalies = lambda team: list(home_g if team is home else away_g)
    sim._gm_determine_preferred_goalie = lambda gl, team: (gl[0] if gl else None)
    sim._gm_allocate_conserved_toi = lambda _rng, dressed: {
        str(getattr(p, "id")): 900 for p in dressed
    }
    sim._gm_pos_str = lambda p: str(getattr(p, "position", "C"))
    sim._gm_ovr_norm = lambda p: float(getattr(p, "overall", 75)) / 99.0
    sim._gm_ovr_bonus = lambda p: 1.0
    sim._gm_rating_avg = lambda p, _keys: float(getattr(p, "overall", 75))
    sim._gm_role_usage_mult = lambda p: 1.0
    sim._gm_scoring_hub_bonus = lambda p, team: 1.0
    sim._gm_offensive_skill_composite = lambda p: float(getattr(p, "overall", 75))
    sim._gm_physical_weight = lambda p: 1.0
    sim._team_superstar_offense_impact = lambda team: 0.0
    sim._gm_ledger_ensure = SimEngine._gm_ledger_ensure.__get__(sim, SimEngine)
    sim._gm_ledger_add = SimEngine._gm_ledger_add.__get__(sim, SimEngine)
    sim._gm_distribute_integer_shares = SimEngine._gm_distribute_integer_shares.__get__(sim, SimEngine)

    ledger = {}
    box = sim._accumulate_light_strength_game_stats(
        rng, home, away, "H", "A", hg=4, ag=2, ot=False, ledger=ledger
    )
    assert box.get("light_box") is True

    skater_rows = [r for r in ledger.values() if str(r.get("position")) != "G"]
    assert sum(int(r.get("g") or 0) for r in skater_rows) == 6
    assert any(int(r.get("plus_minus") or 0) != 0 for r in skater_rows)
    assert any(float(r.get("gf_on") or 0) > 0 for r in skater_rows)
    assert sum(int(r.get("hit") or 0) for r in skater_rows) >= 8
    assert sum(int(r.get("blk") or 0) for r in skater_rows) >= 6

    hg_row = ledger["hg1"]
    assert int(hg_row.get("toi_sec") or 0) >= 3600
    assert float(hg_row.get("goalie_xga") or 0) > 0
    assert int(hg_row.get("ga") or 0) == 2
    gnorm = normalize_goalie_counting_stats(hg_row)
    assert 1.0 < float(gnorm["gaa"]) < 8.0

    snorm = normalize_skater_counting_stats(next(r for r in skater_rows if int(r.get("plus_minus") or 0) != 0))
    assert int(snorm["plus_minus"]) != 0


def test_goalie_gaa_repairs_underfilled_toi():
    broken = {
        "player_id": "g1",
        "position": "G",
        "gp": 11,
        "ga": 30,
        "shots_against": 340,
        "saves": 310,
        "toi_sec": 3600,  # only one game of TOI accrued
        "goalie_xga": 28.0,
    }
    fixed = normalize_goalie_counting_stats(broken)
    assert float(fixed["gaa"]) < 5.0
    assert float(fixed["toi_sec"]) >= 11 * 3600


def test_light_ixg_is_per_game_not_triangular():
    """iXG must accumulate ~linearly; season-total re-add produced ~3000 iXG / WAR ~200."""
    sim = SimEngine(seed=7, debug=False, populate_initial_rosters=False)
    rng = random.Random(7)

    home_sk = [_skater(f"h{i}", f"Home {i}", "C" if i < 8 else "D", 80 + (i % 5)) for i in range(18)]
    away_sk = [_skater(f"a{i}", f"Away {i}", "C" if i < 8 else "D", 78 + (i % 5)) for i in range(18)]
    home_g = [_goalie("hg1", "Home Goalie")]
    away_g = [_goalie("ag1", "Away Goalie")]
    home = _team("H", home_sk, home_g)
    away = _team("A", away_sk, away_g)

    sim._gm_build_dressed_lineup = lambda team, _rng: (
        (home_sk if team is home else away_sk),
        (home_g if team is home else away_g),
        [],
        set(),
    )
    sim._gm_skaters = lambda team: list(home_sk if team is home else away_sk)
    sim._gm_goalies = lambda team: list(home_g if team is home else away_g)
    sim._gm_determine_preferred_goalie = lambda gl, team: (gl[0] if gl else None)
    sim._gm_allocate_conserved_toi = lambda _rng, dressed: {
        str(getattr(p, "id")): 900 for p in dressed
    }
    sim._gm_pos_str = lambda p: str(getattr(p, "position", "C"))
    sim._gm_ovr_norm = lambda p: float(getattr(p, "overall", 75)) / 99.0
    sim._gm_ovr_bonus = lambda p: 1.0
    sim._gm_rating_avg = lambda p, _keys: float(getattr(p, "overall", 75))
    sim._gm_role_usage_mult = lambda p: 1.0
    sim._gm_scoring_hub_bonus = lambda p, team: 1.0
    sim._gm_offensive_skill_composite = lambda p: float(getattr(p, "overall", 75))
    sim._gm_physical_weight = lambda p: 1.0
    sim._team_superstar_offense_impact = lambda team: 0.0
    sim._gm_ledger_ensure = SimEngine._gm_ledger_ensure.__get__(sim, SimEngine)
    sim._gm_ledger_add = SimEngine._gm_ledger_add.__get__(sim, SimEngine)
    sim._gm_distribute_integer_shares = SimEngine._gm_distribute_integer_shares.__get__(sim, SimEngine)

    ledger = {}
    for _ in range(40):
        box = sim._accumulate_light_strength_game_stats(
            rng, home, away, "H", "A", hg=3, ag=2, ot=False, ledger=ledger
        )
        assert box.get("home_cf", 0) > 0
        assert box.get("home_xgf", 0) > 0

    skater_rows = [r for r in ledger.values() if str(r.get("position")) != "G"]
    max_ixg = max(float(r.get("ixg") or 0) for r in skater_rows)
    max_sog = max(int(r.get("sog") or 0) for r in skater_rows)
    assert max_ixg < 80, max_ixg
    assert max_ixg < max(1.0, max_sog * 0.35), (max_ixg, max_sog)


def test_light_possession_spreads_cf_by_talent():
    """Stronger clubs / stars must clear ~50% CF; not every skater clones team CF%."""
    sim = SimEngine(seed=11, debug=False, populate_initial_rosters=False)
    rng = random.Random(11)

    home_sk = [_skater(f"h{i}", f"Home {i}", "C" if i < 8 else "D", 92 if i < 3 else 72) for i in range(18)]
    away_sk = [_skater(f"a{i}", f"Away {i}", "C" if i < 8 else "D", 70) for i in range(18)]
    home_g = [_goalie("hg1", "Home Goalie", 88)]
    away_g = [_goalie("ag1", "Away Goalie", 72)]
    home = _team("H", home_sk, home_g)
    away = _team("A", away_sk, away_g)

    sim._gm_build_dressed_lineup = lambda team, _rng: (
        (home_sk if team is home else away_sk),
        (home_g if team is home else away_g),
        [],
        set(),
    )
    sim._gm_skaters = lambda team: list(home_sk if team is home else away_sk)
    sim._gm_goalies = lambda team: list(home_g if team is home else away_g)
    sim._gm_determine_preferred_goalie = lambda gl, team: (gl[0] if gl else None)
    sim._gm_allocate_conserved_toi = lambda _rng, dressed: {
        str(getattr(p, "id")): (1200 if int(str(getattr(p, "id"))[1:]) < 3 else 700) for p in dressed
    }
    sim._gm_pos_str = lambda p: str(getattr(p, "position", "C"))
    sim._gm_ovr_norm = lambda p: float(getattr(p, "overall", 75)) / 99.0
    sim._gm_ovr_bonus = lambda p: 1.0
    sim._gm_rating_avg = lambda p, _keys: float(getattr(p, "overall", 75))
    sim._gm_role_usage_mult = lambda p: 1.0
    sim._gm_scoring_hub_bonus = lambda p, team: 1.0
    sim._gm_offensive_skill_composite = lambda p: float(getattr(p, "overall", 75))
    sim._gm_physical_weight = lambda p: 1.0
    sim._team_superstar_offense_impact = lambda team: 0.12 if team is home else 0.0
    sim._team_offense_skill = lambda team: 0.72 if team is home else 0.42
    sim._team_defense_suppression = lambda team: 0.55 if team is home else 0.40
    sim._gm_ledger_ensure = SimEngine._gm_ledger_ensure.__get__(sim, SimEngine)
    sim._gm_ledger_add = SimEngine._gm_ledger_add.__get__(sim, SimEngine)
    sim._gm_distribute_integer_shares = SimEngine._gm_distribute_integer_shares.__get__(sim, SimEngine)
    sim._gm_regulation_attempt_split = SimEngine._gm_regulation_attempt_split.__get__(sim, SimEngine)

    ledger = {}
    for _ in range(20):
        box = sim._accumulate_light_strength_game_stats(
            rng, home, away, "H", "A", hg=4, ag=2, ot=False, ledger=ledger,
            home_strength_scale=1.05, away_strength_scale=0.95,
        )
        assert int(box.get("home_cf") or 0) > int(box.get("away_cf") or 0)

    home_rows = [r for r in ledger.values() if str(r.get("team_id")) == "H" and str(r.get("position")) != "G"]
    cf_pcts = []
    for r in home_rows:
        cf = float(r.get("cf") or 0)
        ca = float(r.get("ca") or 0)
        if cf + ca > 0:
            cf_pcts.append(cf / (cf + ca))
    assert cf_pcts
    assert max(cf_pcts) - min(cf_pcts) > 0.015, (min(cf_pcts), max(cf_pcts))
    assert max(cf_pcts) > 0.54, max(cf_pcts)


def test_repair_inflated_ixg_in_normalize():
    broken = {
        "player_id": "p1",
        "position": "C",
        "gp": 82,
        "g": 95,
        "a": 60,
        "sog": 320,
        "ixg": 2921.9,
        "xa": 2287.4,
        "toi_sec": 82 * 20 * 60,
    }
    fixed = normalize_skater_counting_stats(broken)
    assert float(fixed["ixg"]) < 100
    assert float(fixed["xa"]) < 120
    assert abs(float(fixed["ixg"]) - round(float(fixed["ixg"]))) < 1e-9
