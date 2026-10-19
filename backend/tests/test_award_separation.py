"""Hart, Selke, and Art Ross must reward different seasons."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.sim_engine.league.awards import (
    eligible_selke,
    hart_ballot_score,
    selke_ballot_score,
    selke_fallback_formula,
)


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


def _ross():
    return {
        "player_id": "ross",
        "name": "Scoring Champ",
        "position": "RW",
        "team_id": "DAL",
        "gp": 82,
        "g": 52,
        "a": 68,
        "pts": 120,
        "plus_minus": 8,
        "toi_sec": 82 * 21 * 60,
        "pp_toi_sec": 82 * 3.2 * 60,
        "pk_toi_sec": 82 * 0.25 * 60,
        "blk": 18,
        "hit": 40,
        "_team_point_share": 0.26,
    }


def _mvp():
    return {
        "player_id": "mvp",
        "name": "Club Engine",
        "position": "C",
        "team_id": "WPG",
        "gp": 80,
        "g": 38,
        "a": 58,
        "pts": 96,
        "plus_minus": 28,
        "toi_sec": 80 * 20.4 * 60,
        "pp_toi_sec": 80 * 2.6 * 60,
        "pk_toi_sec": 80 * 1.4 * 60,
        "blk": 52,
        "hit": 70,
        "_team_point_share": 0.24,
    }


def _selke():
    return {
        "player_id": "selke",
        "name": "Shutdown Centre",
        "position": "C",
        "team_id": "BOS",
        "gp": 82,
        "g": 22,
        "a": 36,
        "pts": 58,
        "plus_minus": 16,
        "toi_sec": 82 * 19.2 * 60,
        "pp_toi_sec": 82 * 1.3 * 60,
        "pk_toi_sec": 82 * 2.4 * 60,
        "blk": 92,
        "hit": 140,
        "fo_taken": 1400,
        "faceoff_pct": 56.0,
        "shg": 3,
        "_team_point_share": 0.14,
    }


TEAM_CTX = {
    "DAL": {"points_pct_norm": 0.28, "playoff_qualified": False},
    "WPG": {"points_pct_norm": 0.96, "playoff_qualified": True},
    "BOS": {"points_pct_norm": 0.72, "playoff_qualified": True},
}


class AwardSeparationTests(unittest.TestCase):
    def test_hart_is_not_the_art_ross(self):
        ross = _ross()
        mvp = _mvp()
        self.assertGreater(float(ross["pts"]), float(mvp["pts"]))
        self.assertGreater(hart_ballot_score(mvp, TEAM_CTX), hart_ballot_score(ross, TEAM_CTX))

    def test_selke_is_not_the_scorer(self):
        ross = _ross()
        shutdown = _selke()
        self.assertGreater(float(ross["pts"]), float(shutdown["pts"]))
        self.assertGreater(selke_ballot_score(shutdown), selke_ballot_score(ross))
        pool = eligible_selke([ross, shutdown, _mvp()], 82)
        ids = {r["player_id"] for r in pool}
        self.assertIn("selke", ids)
        self.assertNotIn("ross", ids)

    def test_three_trophies_three_players(self):
        ross, mvp, shutdown = _ross(), _mvp(), _selke()
        art_winner = max((ross, mvp, shutdown), key=lambda r: r["pts"])["player_id"]
        hart_winner = max((ross, mvp, shutdown), key=lambda r: hart_ballot_score(r, TEAM_CTX))["player_id"]
        selke_winner = max((ross, mvp, shutdown), key=selke_ballot_score)["player_id"]
        self.assertEqual(art_winner, "ross")
        self.assertEqual(hart_winner, "mvp")
        self.assertEqual(selke_winner, "selke")
        self.assertEqual(len({art_winner, hart_winner, selke_winner}), 3)

    def test_selke_fallback_still_uncorrelated_with_points(self):
        pool = []
        for i in range(24):
            pts = 18 + (i * 4) % 48
            pool.append(
                {
                    "player_id": f"f{i}",
                    "position": "C" if i % 3 == 0 else "LW",
                    "gp": 70 + (i % 10),
                    "pts": pts,
                    "g": pts // 2,
                    "a": pts - pts // 2,
                    "toi_per_game": 14.0 + (i % 6) * 1.1,
                    "faceoff_pct": 45.0 + (i * 3) % 20,
                    "fo_taken": 250 + i * 10,
                    "pk_toi": 20 + (i * 7) % 80,
                    "blk": 20 + (i * 9) % 90,
                }
            )
        scores = [selke_fallback_formula(dict(r)) for r in pool]
        points = [float(r["pts"]) for r in pool]
        self.assertLess(_pearson(scores, points), 0.85)


if __name__ == "__main__":
    unittest.main()
