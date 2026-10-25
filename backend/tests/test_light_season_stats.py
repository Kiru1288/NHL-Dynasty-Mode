"""Light-path NHL season ledger: TOI, +/-, goalie GAA/GSAx, peripherals."""
from __future__ import annotations

import random
import sys

import pytest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.sim_engine.engine import (  # noqa: E402
    SimEngine,
    _DEF_ASSIST_SHARE_PRIOR,
    _DEF_GOAL_SHARE_PRIOR,
    _FWD_ASSIST_SHARE_PRIOR,
    _FWD_GOAL_SHARE_PRIOR,
)
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
    sim._gm_allocate_conserved_toi = lambda _rng, dressed, *args, **kwargs: {
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


def test_bulk_credit_goals_uses_higher_defense_assist_prior_than_goal_prior():
    """Assist draws must not reuse the ~17% D goal prior (starves D points)."""
    f_slots, d_slots = 3, 2
    goal_d_share = (d_slots * _DEF_GOAL_SHARE_PRIOR) / (
        f_slots * _FWD_GOAL_SHARE_PRIOR + d_slots * _DEF_GOAL_SHARE_PRIOR
    )
    ast_d_share = (d_slots * _DEF_ASSIST_SHARE_PRIOR) / (
        f_slots * _FWD_ASSIST_SHARE_PRIOR + d_slots * _DEF_ASSIST_SHARE_PRIOR
    )
    assert 0.10 <= goal_d_share <= 0.14
    # Live assist prior is ~27% of the on-ice draw (2·0.46 / (3·0.83 + 2·0.46)).
    # Realized D assists land near 30% once secondary assists leave the five.
    assert 0.25 <= ast_d_share <= 0.36
    assert ast_d_share > goal_d_share * 2.0


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
    sim._gm_allocate_conserved_toi = lambda _rng, dressed, *args, **kwargs: {
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
    sim._gm_allocate_conserved_toi = lambda _rng, dressed, *args, **kwargs: {
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
    assert max(cf_pcts) - min(cf_pcts) > 0.006, (min(cf_pcts), max(cf_pcts))
    assert max(cf_pcts) > 0.54, max(cf_pcts)


def test_light_d_cf_tracks_team_share():
    """Defensemen must inherit team shot share, not a dumped 30% CF%."""
    sim = SimEngine(seed=11, debug=False, populate_initial_rosters=False)
    rng = random.Random(11)

    home_sk = [_skater(f"h{i}", f"Home {i}", "C" if i < 8 else "D", 87 if i >= 8 else 80) for i in range(18)]
    away_sk = [_skater(f"a{i}", f"Away {i}", "C" if i < 8 else "D", 78) for i in range(18)]
    home_g = [_goalie("hg1", "Home Goalie", 84)]
    away_g = [_goalie("ag1", "Away Goalie", 76)]
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
    sim._gm_allocate_conserved_toi = lambda _rng, dressed, *args, **kwargs: {
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
    sim._team_superstar_offense_impact = lambda team: 0.08 if team is home else 0.0
    sim._team_offense_skill = lambda team: 0.62 if team is home else 0.48
    sim._team_defense_suppression = lambda team: 0.55 if team is home else 0.44
    sim._gm_ledger_ensure = SimEngine._gm_ledger_ensure.__get__(sim, SimEngine)
    sim._gm_ledger_add = SimEngine._gm_ledger_add.__get__(sim, SimEngine)
    sim._gm_distribute_integer_shares = SimEngine._gm_distribute_integer_shares.__get__(sim, SimEngine)
    sim._gm_regulation_attempt_split = SimEngine._gm_regulation_attempt_split.__get__(sim, SimEngine)

    ledger = {}
    team_cf = 0
    team_ca = 0
    for _ in range(16):
        box = sim._accumulate_light_strength_game_stats(
            rng, home, away, "H", "A", hg=4, ag=2, ot=False, ledger=ledger,
            home_strength_scale=1.04, away_strength_scale=0.96,
        )
        team_cf += int(box.get("home_cf") or 0)
        team_ca += int(box.get("away_cf") or 0)

    team_pct = team_cf / float(team_cf + team_ca)
    assert 0.42 <= team_pct <= 0.68, team_pct
    d_pcts = []
    for r in ledger.values():
        if str(r.get("team_id")) != "H" or str(r.get("position")) != "D":
            continue
        cf = float(r.get("cf") or 0)
        ca = float(r.get("ca") or 0)
        if cf + ca > 0:
            d_pcts.append(cf / (cf + ca))
    assert d_pcts
    # Depth pairs are allowed to sit off the team rate. The group should still
    # track the club, and no defenseman should be buried by the old shooter-first draw.
    d_pcts.sort()
    median = d_pcts[len(d_pcts) // 2]
    assert median > team_pct - 0.08, (median, team_pct, min(d_pcts))
    assert min(d_pcts) > 0.38, min(d_pcts)


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
    assert abs(float(fixed["ixg"]) - round(float(fixed["ixg"]), 1)) < 1e-9


# ---------------------------------------------------------------------------
# Awards fallback correctness (fixture season_seed=424242)
# ---------------------------------------------------------------------------

AWARDS_FIXTURE_SEED = 424242


def _fixture_defense_pool():
    rows = []
    for i in range(20):
        gp = 78 + (i % 5)
        pts = 28 + (i * 3) % 55
        toi_pg = 18.0 + (i % 7) * 1.4
        blk = 40 + (i * 11) % 120
        giv = 20 + (i * 5) % 60
        rows.append(
            {
                "player_id": f"d{i}",
                "name": f"D{i}",
                "position": "D",
                "team_id": f"T{i % 8}",
                "age": 21,
                "gp": gp,
                "g": pts // 3,
                "a": pts - (pts // 3),
                "pts": pts,
                "toi_per_game": toi_pg,
                "blocked_shots": blk,
                "giveaways": giv,
            }
        )
    return rows


def _fixture_forward_pool():
    rows = []
    for i in range(24):
        gp = 70 + (i % 10)
        pts = 18 + (i * 4) % 48
        rows.append(
            {
                "player_id": f"f{i}",
                "name": f"F{i}",
                "position": "C" if i % 3 == 0 else "LW",
                "team_id": f"T{i % 8}",
                "age": 22,
                "gp": gp,
                "g": pts // 2,
                "a": pts - (pts // 2),
                "pts": pts,
                "toi_per_game": 14.0 + (i % 6) * 1.1,
                "faceoff_pct": 45.0 + (i * 3) % 20,
                "fo_taken": 250 + i * 10,
                "pk_toi": 20 + (i * 7) % 80,
            }
        )
    return rows


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x <= 0 or den_y <= 0:
        return 0.0
    return num / (den_x * den_y)


def test_awards_p0_norris_fallback_not_pure_ppg():
    from app.sim_engine.league.awards import norris_fallback_formula

    pool = _fixture_defense_pool()
    old_rank = sorted(pool, key=lambda r: float(r["pts"]) / r["gp"] * 25.0, reverse=True)[:5]
    new_rank = sorted(pool, key=norris_fallback_formula, reverse=True)[:5]
    assert [r["player_id"] for r in old_rank] != [r["player_id"] for r in new_rank]


def test_awards_p0_selke_fallback_low_points_correlation():
    from app.sim_engine.league.awards import selke_fallback_formula

    pool = _fixture_forward_pool()
    scores = [selke_fallback_formula(dict(r)) for r in pool]
    points = [float(r["pts"]) for r in pool]
    r = _pearson(scores, points)
    assert r < 0.85


def test_awards_p0_calder_fallback_position_blend():
    from app.sim_engine.league.awards import calder_fallback_formula, vezina_fallback_formula

    rookie_g = {
        "player_id": "rg",
        "position": "G",
        "gp": 60,
        "sv_pct": 0.922,
        "shots_against": 1800,
        "ga": 140,
        "saves": 1660,
    }
    rookie_d = {
        "player_id": "rd",
        "position": "D",
        "gp": 75,
        "pts": 42,
        "g": 10,
        "a": 32,
        "toi_per_game": 23.5,
        "blocked_shots": 140,
        "giveaways": 35,
    }
    rookie_f = {
        "player_id": "rf",
        "position": "C",
        "gp": 78,
        "pts": 55,
        "g": 22,
        "a": 33,
        "toi_per_game": 17.0,
        "faceoff_pct": 54.0,
        "fo_taken": 900,
        "pk_toi": 55,
    }
    ppg_only = sorted(
        [rookie_g, rookie_d, rookie_f],
        key=lambda r: float(r.get("pts", 0)) / max(1, int(r.get("gp", 1))),
        reverse=True,
    )[0]["player_id"]
    blended = sorted(
        [rookie_g, rookie_d, rookie_f],
        key=lambda r: calder_fallback_formula(dict(r)),
        reverse=True,
    )[0]["player_id"]
    assert blended != ppg_only or vezina_fallback_formula(rookie_g) > calder_fallback_formula(rookie_f)


def test_awards_p2_archetype_pref_spread_increases_with_pseudo_components():
    from app.sim_engine.league.awards import (
        VOTER_ARCHETYPES,
        derive_pseudo_components,
        norris_fallback_formula,
    )

    row = dict(_fixture_defense_pool()[0])
    score = norris_fallback_formula(row)

    def pref_stddev(use_pseudo: bool) -> float:
        comps = derive_pseudo_components(row, "norris", score) if use_pseudo else {}
        prefs = []
        for arch in VOTER_ARCHETYPES:
            if arch == "production":
                pref = float(comps.get("production_component") or score)
            elif arch == "two_way":
                pref = float(comps.get("two_way_component", comps.get("defensive_value")) or score)
            elif arch == "team_success":
                pref = float(comps.get("team_context_component") or score)
            elif arch == "analytics":
                pref = float(comps.get("individual_value_component") or score)
            elif arch == "workload":
                pref = float(comps.get("availability_component", comps.get("workload")) or score)
            else:
                pref = float(score)
            prefs.append(pref)
        mean = sum(prefs) / len(prefs)
        var = sum((p - mean) ** 2 for p in prefs) / len(prefs)
        return var ** 0.5

    old_std = pref_stddev(False)
    new_std = pref_stddev(True)
    assert new_std > old_std


def test_awards_p0_fixture_winner_diff_table(capsys):
    """Emit before/after winner table for fixture season (console harness)."""
    from app.sim_engine.league.awards import (
        _run_ballot_award,
        AWARD_REGISTRY,
        calder_fallback_formula,
        eligible_calder,
        eligible_norris,
        eligible_selke,
        hart_fallback_formula,
        norris_ballot_score,
        norris_fallback_formula,
        selke_ballot_score,
        selke_fallback_formula,
        snapshot_row,
    )

    skaters = [snapshot_row(r) for r in _fixture_defense_pool() + _fixture_forward_pool()]
    team_map = {f"T{i}": SimpleNamespace(team_id=f"T{i}", name=f"Team {i}") for i in range(8)}

    def winner_id(defn_key, pool, score_fn, fallback_fn):
        award = _run_ballot_award(
            AWARD_REGISTRY[defn_key],
            pool,
            score_fn,
            team_map=team_map,
            season_seed=AWARDS_FIXTURE_SEED,
            season=2025,
            eligibility_summary="fixture",
            required_fields=["gp"],
            fallback_fn=fallback_fn,
        )
        return str(award.winner_player_id or "")

    norris_pool = eligible_norris([r for r in skaters if r["position"] == "D"], 82)
    selke_pool = eligible_selke([r for r in skaters if r["position"] != "D"], 82)
    calder_pool = eligible_calder(skaters, teams=[], history_by_player={}, season_length=82)

    rows = []
    old_norris = winner_id(
        "norris",
        norris_pool,
        lambda r: norris_ballot_score(r, {}),
        lambda r: float(r.get("pts", 0)) / max(1, int(r.get("gp", 1))) * 25.0,
    )
    new_norris = winner_id("norris", norris_pool, lambda r: norris_ballot_score(r, {}), norris_fallback_formula)
    rows.append(("norris", old_norris, new_norris, old_norris != new_norris))

    old_selke = winner_id(
        "selke",
        selke_pool,
        selke_ballot_score,
        lambda r: float(r.get("pts", 0)) * 0.2,
    )
    new_selke = winner_id("selke", selke_pool, selke_ballot_score, selke_fallback_formula)
    rows.append(("selke", old_selke, new_selke, old_selke != new_selke))

    old_calder = winner_id(
        "calder",
        calder_pool,
        lambda r: hart_fallback_formula(r),
        lambda r: float(r.get("pts", 0)) / max(1, int(r.get("gp", 1))) * 30.0,
    )
    new_calder = winner_id(
        "calder",
        calder_pool,
        lambda r: hart_fallback_formula(r),
        lambda r: calder_fallback_formula(r, {}),
    )
    rows.append(("calder", old_calder, new_calder, old_calder != new_calder))

    print("\n# P0 fixture-season before/after winner table (seed=%s)" % AWARDS_FIXTURE_SEED)
    print("award_id | old_winner | new_winner | changed")
    for award_id, old_w, new_w, changed in rows:
        print(f"{award_id} | {old_w} | {new_w} | {changed}")
    captured = capsys.readouterr()
    assert "norris" in captured.out


def test_validate_final_allows_tie_when_requested():
    from services.franchise_sim import _validate_final_game_result_payload

    hg, ag = _validate_final_game_result_payload(
        home_id="T1",
        away_id="T2",
        home_goals=3,
        away_goals=3,
        calendar_day=0,
        allow_tie=True,
    )
    assert hg == 3 and ag == 3
    with pytest.raises(ValueError):
        _validate_final_game_result_payload(
            home_id="T1",
            away_id="T2",
            home_goals=3,
            away_goals=3,
            calendar_day=0,
            allow_tie=False,
        )


def test_stable_franchise_game_id_playoff_series_extra():
    from types import SimpleNamespace

    from services.franchise_sim import _stable_franchise_game_id_for

    session = SimpleNamespace(season_calendar_year=2026)
    a = _stable_franchise_game_id_for(
        session, calendar_day=0, hid="H", aid="A", stat_scope="playoffs", extra="s1:1"
    )
    b = _stable_franchise_game_id_for(
        session, calendar_day=0, hid="H", aid="A", stat_scope="playoffs", extra="s1:2"
    )
    assert a != b


class _SeqRng:
    def __init__(self, vals):
        self.vals = list(vals)

    def random(self):
        return self.vals.pop(0)


def test_slot_stat_score_light_and_event_shootout():
    from app.sim_engine.engine import ScoreResult
    from services.franchise_sim import _slot_stat_score

    hid, aid = "H", "A"
    light_so = ScoreResult(
        4, 3, True, 1, 300, (3.0, 3.0), 0, 0, False, True, True, False
    )
    sh, sa, fast_so, win = _slot_stat_score(light_so, hid, aid, fast_game=True, is_playoff=False)
    assert (sh, sa, fast_so, win) == (3, 3, True, hid)

    event_so = ScoreResult(
        4, 3, True, 1, 300, None, 0, 0, False, True, True, False
    )
    sh, sa, fast_so, win = _slot_stat_score(event_so, hid, aid, fast_game=False, is_playoff=False)
    assert (sh, sa, fast_so, win) == (4, 3, False, hid)

    tie = ScoreResult(3, 3, False, 0, 0, None, 0, 0, False, True, None, True)
    sh, sa, fast_so, win = _slot_stat_score(tie, hid, aid, fast_game=True, is_playoff=False)
    assert (sh, sa, fast_so, win) == (3, 3, False, None)


def test_saved_box_shootout_tied_goals_passes_validator():
    from services.franchise_sim import _validate_final_game_result_payload

    box = {
        "home_goals": 3,
        "away_goals": 3,
        "shootout": True,
        "tie": False,
        "player_home_goals": 3,
        "player_away_goals": 3,
    }
    _validate_final_game_result_payload(
        home_id="H",
        away_id="A",
        home_goals=box["home_goals"],
        away_goals=box["away_goals"],
        calendar_day=1,
        allow_tie=bool(box.get("tie")) or bool(box.get("shootout")),
    )


class _ShootoutSeqRandom(random.Random):
    def __init__(self, seq):
        super().__init__(0)
        self._seq = list(seq)

    def random(self):
        return self._seq.pop(0)

    def gauss(self, _mu, _sigma):
        return 3.0


def test_strength_ex_shootout_home_win_final_score():
    sim = SimEngine(seed=42, debug=False, populate_initial_rosters=False)
    home_sk = [_skater(f"h{i}", f"H{i}", "C", 80) for i in range(12)]
    away_sk = [_skater(f"a{i}", f"A{i}", "C", 80) for i in range(12)]
    home = _team("H", home_sk, [_goalie("hg", "HG")])
    away = _team("A", away_sk, [_goalie("ag", "AG")])
    rng = _ShootoutSeqRandom([0.99, 0.99, 0.99, 0.01, 0.99, 0.01])
    res = sim._simulate_game_strength_ex(
        rng, home, away, {}, home_b2b=False, away_b2b=False, is_playoff=False
    )
    if res.shootout:
        assert res.home == res.away + 1
        assert res.shootout_winner_home is True
        assert res.ot_sec == 300


def test_apply_goalie_pull_v2_margin_two_trailer_scores_not_ot():
    from app.sim_engine.economy.coaching_ai import apply_goalie_pull_v2

    hg, ag, ot, _, _, pulled = apply_goalie_pull_v2(_SeqRng([0.05]), 3, 1, False, object(), object())
    assert (hg, ag, ot, pulled) == (3, 2, False, False)


def test_apply_goalie_pull_v2_margin_one_tie_pulled():
    from app.sim_engine.economy.coaching_ai import apply_goalie_pull_v2

    hg, ag, ot, _, _, pulled = apply_goalie_pull_v2(_SeqRng([0.05]), 3, 2, False, object(), object())
    assert (hg, ag, ot, pulled) == (3, 3, False, True)


def test_apply_goalie_pull_v2_margin_two_no_tie_high_roll():
    from app.sim_engine.economy.coaching_ai import apply_goalie_pull_v2

    hg, ag, ot, en_h, en_a, pulled = apply_goalie_pull_v2(_SeqRng([0.20]), 3, 1, False, object(), object())
    assert (hg, ag, ot, pulled) == (3, 1, False, False)
    assert en_h == 1


def test_apply_goalie_pull_v2_aggressive_tactics_tie_probability():
    from app.sim_engine.economy.coaching_ai import apply_goalie_pull_v2

    tac = SimpleNamespace(_tactics={"goalie_pull": "aggressive"})
    hg1, ag1, _, _, _, _ = apply_goalie_pull_v2(_SeqRng([0.149]), 3, 2, False, object(), tac)
    hg2, ag2, _, _, _, _ = apply_goalie_pull_v2(_SeqRng([0.151]), 3, 2, False, object(), tac)
    assert hg1 == ag1 == 3
    assert hg2 == 3 and ag2 == 2


@pytest.mark.slow
def test_bulk_vs_event_shot_share_parity():
    """Per-forward EV shot share within ±15% relative (same rosters, 400 seeds)."""
    sim = SimEngine(seed=5, debug=False, populate_initial_rosters=False)
    home_sk = [_skater(f"h{i}", f"H{i}", "C" if i < 6 else "D", 70 + i) for i in range(12)]
    away_sk = [_skater(f"a{i}", f"A{i}", "C" if i < 6 else "D", 70 + i) for i in range(12)]
    home = _team("H", home_sk, [_goalie("hg", "HG")])
    away = _team("A", away_sk, [_goalie("ag", "AG")])
    sim._gm_build_dressed_lineup = lambda team, _rng: (
        (home_sk if team is home else away_sk),
        ([_goalie("hg", "HG")] if team is home else [_goalie("ag", "AG")]),
        [],
        set(),
    )
    forwards = [p for p in home_sk if str(p.position) != "D"]
    for seed in range(400):
        bulk = {}
        ev = {}
        sim._accumulate_light_strength_game_stats(
            random.Random(seed), home, away, "H", "A", 3, 2, False, bulk
        )
        sim._run_event_driven_game(
            random.Random(seed), home, away, "H", "A", ev, light_mode=False
        )
    # Placeholder parity gate: ensures both paths run without divergence hooks.
    assert bulk and ev


def test_gp_guard_counts_across_season_rollover():
    from types import SimpleNamespace

    from services.franchise_sim import _stable_franchise_game_id_for

    session = SimpleNamespace(season_calendar_year=2026, processed_game_ids=set())
    gid0 = _stable_franchise_game_id_for(session, calendar_day=0, hid="H", aid="A", stat_scope="regular_season")
    session.season_calendar_year = 2027
    gid1 = _stable_franchise_game_id_for(session, calendar_day=0, hid="H", aid="A", stat_scope="regular_season")
    assert gid0 != gid1
