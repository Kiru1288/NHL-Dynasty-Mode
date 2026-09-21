"""
Prospect in-season growth: gradual OVR, independent potential rises, no double payout.

Run: cd backend && python -m pytest tests/test_prospect_in_season_growth.py -q
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "SimEngine", ROOT / "backend", ROOT / "backend" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.sim_engine.entities.player import display_rating, player_current_ovr_01  # noqa: E402
from app.sim_engine.progression import development as dev  # noqa: E402
from test_chapter_attributes import _make_player  # noqa: E402

SEASON = 2025


def _prospect(age: int = 18, potential: float = 0.88, seed: int = 7):
    p = _make_player("C", seed=seed)
    p.identity.age = age
    p.potential = potential
    return p


def _ovr(p) -> float:
    return float(display_rating(player_current_ovr_01(p)))


def _set_line(p, ppg: float, proj_ppg: float = 0.8, gp: int = 40) -> None:
    p._prospect_season_stats = {"gp": gp, "ppg": ppg, "points_per_game": ppg, "points": int(ppg * gp)}
    p._prospect_projected_stats = {"gp": 60, "ppg": proj_ppg, "points_per_game": proj_ppg}


def test_ovr_rises_gradually_across_the_season():
    p = _prospect()
    rng = random.Random(3)
    start = _ovr(p)
    traj = []
    for _ in range(dev._PROSPECT_PULSES_PER_SEASON):
        dev.apply_prospect_in_season_pulse(p, rng, SEASON)
        traj.append(_ovr(p))
    total = traj[-1] - start
    assert total >= 2.0, f"high-runway 18yo should visibly grow, got {total}"
    # Monotone-ish: never falls, and no single pulse carries most of the year.
    assert all(b >= a - 0.01 for a, b in zip([start] + traj, traj))
    biggest = max(b - a for a, b in zip([start] + traj, traj))
    assert biggest <= max(2.0, 0.6 * total)
    # Growth is spread out: at the midpoint roughly half (not all, not none) has landed.
    mid = traj[len(traj) // 2 - 1] - start
    assert 0.15 * total <= mid <= 0.85 * total


def test_in_season_gain_stays_inside_the_in_season_share():
    p = _prospect()
    rng = random.Random(11)
    start01 = float(player_current_ovr_01(p))
    for _ in range(dev._PROSPECT_PULSES_PER_SEASON + 5):  # extra ticks must not overspend
        dev.apply_prospect_in_season_pulse(p, rng, SEASON)
    plan = p._prospect_season_plan
    gained = float(player_current_ovr_01(p)) - start01
    assert plan["pulses"] == dev._PROSPECT_PULSES_PER_SEASON
    assert gained <= plan["total"] + 0.012  # attribute rounding slack


def test_offseason_only_pays_the_leftover_and_only_once():
    p = _prospect()
    rng = random.Random(5)
    assert dev.prospect_offseason_leftover(p) is None  # never pulsed: normal full budget applies
    for _ in range(dev._PROSPECT_PULSES_PER_SEASON):
        dev.apply_prospect_in_season_pulse(p, rng, SEASON)
    plan = p._prospect_season_plan
    left = dev.prospect_offseason_leftover(p, consume=False)
    assert left is not None and 0.0 <= left <= plan["annual"]
    # In-season + leftover never exceeds the annual budget.
    assert plan["spent"] + left <= plan["annual"] + 1e-9
    assert dev.prospect_offseason_leftover(p) is not None  # consumes
    assert dev.prospect_offseason_leftover(p) is None  # second offseason pass gets nothing extra


def test_overperformer_potential_rises_average_does_not():
    hot, avg = _prospect(seed=21), _prospect(seed=22)
    rng = random.Random(9)
    _set_line(hot, ppg=1.5)
    _set_line(avg, ppg=0.8)
    pot_hot, pot_avg = hot.potential, avg.potential
    hot_res = dev.apply_prospect_potential_review(hot, rng, SEASON)
    avg_res = dev.apply_prospect_potential_review(avg, random.Random(9), SEASON)
    assert hot_res.get("applied") and hot.potential > pot_hot
    assert avg.potential == pot_avg and not avg_res.get("applied")


def test_potential_and_ovr_are_independent():
    """A prospect can gain potential while OVR has not moved (and vice versa)."""
    p = _prospect(seed=31)
    _set_line(p, ppg=1.6)
    ovr_before = _ovr(p)
    pot_before = p.potential
    dev.apply_prospect_potential_review(p, random.Random(2), SEASON)
    assert p.potential > pot_before
    assert _ovr(p) == ovr_before

    q = _prospect(seed=32)
    pot_q = q.potential
    for _ in range(dev._PROSPECT_PULSES_PER_SEASON):
        dev.apply_prospect_in_season_pulse(q, random.Random(4), SEASON)
    assert _ovr(q) > ovr_before - 1 and q.potential == pot_q


def test_potential_gain_is_capped_per_season():
    p = _prospect(seed=41)
    _set_line(p, ppg=1.8)
    rng = random.Random(6)
    start = p.potential
    for _ in range(40):
        dev.apply_prospect_potential_review(p, rng, SEASON)
    gained_display = (p.potential - start) * 99.0
    assert gained_display <= dev._PROSPECT_POT_SEASON_CAP + 0.6  # cap + display rounding slack


def test_small_sample_does_not_move_potential():
    p = _prospect(seed=51)
    _set_line(p, ppg=2.0, gp=4)
    before = p.potential
    res = dev.apply_prospect_potential_review(p, random.Random(1), SEASON)
    assert not res.get("applied") and p.potential == before


def test_prospect_tick_touches_dev_league_players_and_reports_change():
    from services.prospect_in_season_growth import (
        iter_growth_prospects,
        prospect_growth_fields,
        prospect_in_season_tick,
        snapshot_prospect_season_start,
    )

    kid, vet = _prospect(seed=61), _prospect(age=27, seed=62)
    league = SimpleNamespace(
        development_leagues=[{"teams": [{"players": [kid, vet]}]}], teams=[]
    )
    session = SimpleNamespace(
        sim=SimpleNamespace(league=league, rng=random.Random(8)),
        phase="regular",
        _regular_stats_split_done=True,
        season_calendar_year=SEASON,
    )
    assert list(iter_growth_prospects(league)) == [kid]
    assert snapshot_prospect_season_start(session) == 1
    start = kid.season_start_ovr
    _set_line(kid, ppg=1.6)
    for _ in range(dev._PROSPECT_PULSES_PER_SEASON):
        prospect_in_season_tick(session)
    fields = prospect_growth_fields(kid)
    assert fields["ovr_change_season"] == round(_ovr(kid)) - start >= 1
    assert fields["potential_change_season"] >= 1 and fields["potential_trend"] == "rising"
    assert not hasattr(vet, "_prospect_season_plan")  # 27-year-olds are not prospects


def _pool_kid(**kw):
    base = dict(age=18, draft_value_range=(0.55, 0.80), _pipeline_potential_tier="high", team_id="T1")
    base.update(kw)
    return SimpleNamespace(**base)


def test_ratingless_pool_prospect_grows_gradually_and_offseason_pays_only_remainder():
    from services.prospect_in_season_growth import apply_pool_range_pulse

    kid = _pool_kid()
    rng = random.Random(4)
    lo0, hi0 = kid.draft_value_range
    seen = []
    for _ in range(dev._PROSPECT_PULSES_PER_SEASON):
        apply_pool_range_pulse(kid, None, rng, SEASON)
        seen.append(kid.draft_value_range[1])
    lo1, hi1 = kid.draft_value_range
    assert lo1 > lo0 and hi1 > hi0
    assert all(b >= a for a, b in zip([hi0] + seen, seen))
    assert max(b - a for a, b in zip([hi0] + seen, seen)) < 0.5 * (hi1 - hi0) + 1e-9
    assert kid._pool_inseason_spent > 0
    # Recorded so the engine's offseason pass can subtract it (see _develop_prospect_one_year).
    assert 0 < kid._pool_inseason_spent <= 0.05


def test_pool_potential_follows_production_independently_of_floor():
    from services.prospect_in_season_growth import apply_pool_range_pulse

    hot, cold = _pool_kid(), _pool_kid()
    for k, ppg in ((hot, 1.6), (cold, 0.3)):
        k._prospect_season_stats = {"gp": 40, "ppg": ppg, "points_per_game": ppg}
        k._prospect_projected_stats = {"gp": 60, "ppg": 0.8, "points_per_game": 0.8}
    for _ in range(dev._PROSPECT_PULSES_PER_SEASON):
        apply_pool_range_pulse(hot, None, random.Random(1), SEASON)
        apply_pool_range_pulse(cold, None, random.Random(1), SEASON)
    assert hot.draft_value_range[1] > cold.draft_value_range[1]
    assert abs(hot.draft_value_range[0] - cold.draft_value_range[0]) < 1e-9  # same floor track


def test_growth_fields_report_potential_trend():
    from services.prospect_in_season_growth import prospect_growth_fields

    kid = _prospect(seed=71)
    kid.season_start_potential = 80
    kid.potential = 0.84
    fields = prospect_growth_fields(kid)
    assert fields["potential_change_season"] == 3 and fields["potential_trend"] == "rising"
