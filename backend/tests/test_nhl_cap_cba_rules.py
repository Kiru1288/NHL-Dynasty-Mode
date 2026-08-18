"""NHL CBA cap rule tests: 95.5 upper, bury threshold, AHL residual."""

from __future__ import annotations

from types import SimpleNamespace

from app.sim_engine.economy.cap_engine import (
    apply_nhl_salary_cap_for_season,
    buried_cap_hit_millions,
    calculate_team_cap_snapshot,
    nhl_bury_threshold_millions,
    nhl_lower_limit_millions,
    nhl_upper_limit_millions,
    team_buried_cap_hit_millions,
)


def _p(*, aav: float, minors: bool = False, nmc: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        retired=False,
        in_minors=minors,
        is_buried=minors,
        buried=minors,
        on_ir=False,
        on_ltir=False,
        contract={
            "aav_m": aav,
            "cap_hit_m": aav,
            "no_move_clause": nmc,
            "nmc": nmc,
            "source": "real_nhl_spotrac",
        },
        cap_hit_m=aav,
        aav_m=aav,
        ovr=lambda: 0.75,
    )


def test_2025_cap_bounds_match_nhl():
    assert nhl_upper_limit_millions(2025) == 95.5
    assert nhl_lower_limit_millions(2025) == 79.5
    assert nhl_bury_threshold_millions(2025) == 1.15


def test_bury_residual_and_ahl_counted():
    # $5M in AHL → residual 5 - 1.15 = 3.85
    ahl_p = _p(aav=5.0, minors=True)
    assert abs(buried_cap_hit_millions(ahl_p, season_start_year=2025) - 3.85) < 1e-6
    # Sub-threshold fully relieved
    cheap = _p(aav=0.9, minors=True)
    assert buried_cap_hit_millions(cheap, season_start_year=2025) == 0.0

    team = SimpleNamespace(
        roster=[_p(aav=4.0) for _ in range(20)],
        ahl_roster=[ahl_p],
        echl_roster=[],
        total_cap_hit=0,
    )
    buried = team_buried_cap_hit_millions(team, season_start_year=2025)
    assert abs(buried - 3.85) < 1e-6


def test_apply_season_cap_and_snapshot_upper():
    league = SimpleNamespace(economics=SimpleNamespace(salary_cap=88.0, cap_floor=65.0))
    apply_nhl_salary_cap_for_season(league, 2025)
    assert float(league.salary_cap_m) == 95.5
    assert float(league.economics.salary_cap) == 95.5

    roster = [_p(aav=4.0) for _ in range(23)]
    team = SimpleNamespace(roster=roster, ahl_roster=[], echl_roster=[], total_cap_hit=0)
    snap = calculate_team_cap_snapshot(team, league, season_label="2025-26")
    assert snap["upperLimit"] == 95.5
    assert abs(snap["activeRosterCapHit"] - 92.0) < 1e-6
    assert snap["usableCapSpace"] > 0


def test_season_label_overrides_stale_88_even_if_league_year_lags():
    """Session season must win: league attrs stuck on 2024/$88 must not leak into space."""
    league = SimpleNamespace(
        season_year=2024,
        salary_cap_m=88.0,
        salary_cap=88.0,
        economics=SimpleNamespace(salary_cap=88.0, cap_floor=72.0),
    )
    roster = [_p(aav=4.0) for _ in range(20)]
    team = SimpleNamespace(roster=roster, ahl_roster=[], echl_roster=[], total_cap_hit=0)
    snap = calculate_team_cap_snapshot(team, league, season_label="2025-26")
    assert snap["upperLimit"] == 95.5
    assert abs(snap["usableCapSpace"] - (95.5 - 80.0)) < 1e-6
    assert float(league.salary_cap_m) == 95.5


def test_sens_style_overage_was_wrong_cap_not_payroll():
    """Reproduce 8.4 over on $88 cap with ~$96.4 payroll → fine on $95.5."""
    # 23 * ~4.19 ≈ 96.4
    roster = [_p(aav=4.191) for _ in range(23)]
    team = SimpleNamespace(roster=roster, ahl_roster=[], echl_roster=[], total_cap_hit=0)
    bad = SimpleNamespace(salary_cap_m=88.0, economics=SimpleNamespace(salary_cap=88.0, cap_floor=65.0))
    snap_bad = calculate_team_cap_snapshot(team, bad, season_label="2025-26")
    # With season_label, bounds are corrected to 95.5 — space is near-zero, not ~-8.4 under $88.
    assert snap_bad["upperLimit"] == 95.5
    assert snap_bad["usableCapSpace"] > -2.0

    good = SimpleNamespace()
    apply_nhl_salary_cap_for_season(good, 2025)
    snap_good = calculate_team_cap_snapshot(team, good, season_label="2025-26")
    # ~0.9 over or under depending on rounding — not 8.4
    assert snap_good["usableCapSpace"] > -2.0
