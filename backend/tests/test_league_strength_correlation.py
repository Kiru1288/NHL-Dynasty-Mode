"""Stronger rosters should win more often (regular season + playoffs) — with printed rates."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.sim_engine.engine import SimEngine  # noqa: E402
from app.sim_engine.league.playoffs import PlayoffSeries, _simulate_series  # noqa: E402


def _skater(pid: str, ovr: int, pos: str = "C") -> SimpleNamespace:
    ratings = {
        "skating": ovr,
        "shooting": ovr,
        "hands": ovr,
        "passing": ovr,
        "checking": ovr,
        "defense": ovr,
        "IQ": ovr,
    }
    return SimpleNamespace(
        id=pid,
        player_id=pid,
        name=pid,
        position=pos,
        identity=SimpleNamespace(name=pid, position=pos, age=25),
        ratings=ratings,
        # storyline_conduct._ovr_display treats ovr() as 0–1; franchise ratings use 0–100 scale.
        ovr=lambda o=ovr: float(o) / 99.0,
        overall=ovr,
    )


def _team_graded(tid: str, star_ovr: int, depth_ovr: int) -> SimpleNamespace:
    """Star/depth split so strength_map does not flatline at the 0.96 depth cap."""
    roster = [_skater(f"{tid}_s{i}", star_ovr, "C") for i in range(4)]
    roster += [_skater(f"{tid}_f{i}", depth_ovr, "C") for i in range(8)]
    roster += [_skater(f"{tid}_d{i}", star_ovr if i < 2 else depth_ovr, "D") for i in range(6)]
    g_ovr = int(round(star_ovr * 0.92 + depth_ovr * 0.08))
    roster.append(
        SimpleNamespace(
            id=f"{tid}_g",
            position="G",
            identity=SimpleNamespace(name="G", position="G"),
            ratings={"glove": g_ovr, "blocker": g_ovr, "rebound": g_ovr, "IQ": g_ovr},
            ovr=lambda o=g_ovr: float(o) / 99.0,
            overall=g_ovr,
        )
    )
    return SimpleNamespace(team_id=tid, id=tid, name=tid, roster=roster)


class FranchiseStrengthMapRefreshTests(unittest.TestCase):
    def test_refresh_strength_map_reflects_roster_tier(self):
        from services.franchise_sim import _franchise_refresh_strength_map  # noqa: WPS433

        sim = SimEngine(seed=3, debug=False, populate_initial_rosters=False)
        league = SimpleNamespace(teams=[_team_graded("A", 91, 77), _team_graded("B", 70, 63)])
        sim.league = league
        session = SimpleNamespace(sim=sim, strength_map={})
        _franchise_refresh_strength_map(session)
        sm = session.strength_map
        print(f"\n[refresh] strength_map A={sm['A']:.3f} B={sm['B']:.3f}")
        self.assertGreater(sm["A"], sm["B"] + 0.04)


class LeagueStrengthCorrelationTests(unittest.TestCase):
    def test_elite_roster_beats_weak_roster_head_to_head(self):
        sim = SimEngine(seed=42, debug=False, populate_initial_rosters=False)
        elite = _team_graded("ELT", star_ovr=94, depth_ovr=78)
        weak = _team_graded("WEK", star_ovr=72, depth_ovr=64)
        sm = sim._build_strength_map([elite, weak])
        self.assertGreater(sm["ELT"], sm["WEK"] + 0.06, msg=f"strength gap too small: {sm}")

        rng = random.Random(9001)
        elite_pts = 0.0
        n = 400
        for _ in range(n):
            hg, ag, _ = sim._simulate_game_strength(rng, elite, weak, sm, noise_scale=1.0)
            if hg > ag:
                elite_pts += 2.0
            elif hg < ag:
                pass
            else:
                elite_pts += 1.0
        rate = elite_pts / (n * 2.0)
        print(
            f"\n[regular] ELT strength={sm['ELT']:.3f} vs WEK={sm['WEK']:.3f} "
            f"-> elite point share {rate:.1%} over {n} games"
        )
        self.assertGreaterEqual(rate, 0.58)

    def test_strength_rank_mostly_matches_mini_league_standings(self):
        sim = SimEngine(seed=7, debug=False, populate_initial_rosters=False)
        tiers = [(92, 76), (86, 74), (80, 72), (74, 68), (68, 62)]
        teams = [_team_graded(f"T{i}", star, depth) for i, (star, depth) in enumerate(tiers)]
        sm = sim._build_strength_map(teams)
        strength_rank = sorted(sm.items(), key=lambda x: x[1], reverse=True)

        points: dict[str, int] = {t.team_id: 0 for t in teams}
        rng = random.Random(2026)
        for _ in range(24):
            for home in teams:
                for away in teams:
                    if home is away:
                        continue
                    hg, ag, _ = sim._simulate_game_strength(
                        rng, home, away, sm, noise_scale=1.0
                    )
                    if hg > ag:
                        points[home.team_id] += 2
                    elif hg < ag:
                        points[away.team_id] += 2
                    else:
                        points[home.team_id] += 1
                        points[away.team_id] += 1

        points_rank = sorted(points.items(), key=lambda x: x[1], reverse=True)
        print(f"\n[mini-league] strength order: {strength_rank}")
        print(f"[mini-league] points order:   {points_rank}")

        top_strength = {strength_rank[0][0], strength_rank[1][0]}
        top_points = {points_rank[0][0], points_rank[1][0]}
        self.assertTrue(top_strength & top_points, msg=(strength_rank, points_rank))
        bottom_strength = {strength_rank[-1][0], strength_rank[-2][0]}
        bottom_points = {points_rank[-1][0], points_rank[-2][0]}
        self.assertTrue(bottom_strength & bottom_points, msg=(strength_rank, points_rank))

    def test_playoff_favorite_wins_most_series(self):
        sm = {"HI": 0.84, "LO": 0.58}
        rng = random.Random(314)
        hi_wins = 0
        n = 120
        for _ in range(n):
            series = PlayoffSeries(
                round_index=1,
                conference="East",
                seed_high=1,
                seed_low=8,
                team_high_id="HI",
                team_low_id="LO",
            )
            _simulate_series(rng, series, sm)
            if series.winner_id() == "HI":
                hi_wins += 1
        rate = hi_wins / n
        print(f"\n[playoffs] HI ({sm['HI']:.2f}) vs LO ({sm['LO']:.2f}) series win%: {rate:.1%}")
        self.assertGreaterEqual(rate, 0.72)


if __name__ == "__main__":
    unittest.main(verbosity=2)
