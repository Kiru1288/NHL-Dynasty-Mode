"""Adaptive trade market scaling — O(1) knobs and engine day scale."""

from __future__ import annotations

import random
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.sim_engine.trades.trade_market_adaptive import (
    build_pool_sampling_weights,
    compute_engine_trade_day_scale,
    compute_proposer_adaptive_knobs,
)


class AdaptiveKnobTests(unittest.TestCase):
    def test_pick_pressure_increases_when_pick_rate_low(self):
        league = types.SimpleNamespace(
            cpu_market_runtime={
                "telemetry": {
                    "trades": 20,
                    "with_pick": 2,
                    "rebuild_sales": 10,
                    "rebuild_sales_with_futures": 2,
                }
            }
        )
        low = compute_proposer_adaptive_knobs(
            league,
            calendar_cursor=100,
            regular_season_last_index=192,
            deadline_phase=0.2,
            max_executions=2,
        )
        league.cpu_market_runtime["telemetry"]["with_pick"] = 12
        high = compute_proposer_adaptive_knobs(
            league,
            calendar_cursor=100,
            regular_season_last_index=192,
            deadline_phase=0.2,
            max_executions=2,
        )
        self.assertGreater(low["pick_pressure"], high["pick_pressure"])
        self.assertGreater(low["prefer_pick_only_boost"], high["prefer_pick_only_boost"])
        self.assertLess(low["value_delta_pick_threshold"], high["value_delta_pick_threshold"])

    def test_knob_compute_is_bounded_constant_keys(self):
        league = types.SimpleNamespace(cpu_market_runtime={})
        knobs = compute_proposer_adaptive_knobs(
            league,
            calendar_cursor=0,
            regular_season_last_index=192,
            deadline_phase=0.0,
            max_executions=1,
        )
        expected_keys = {
            "pick_rate",
            "pick_pressure",
            "fairness_gap_max",
            "attempt_budget",
            "peer_modulo",
        }
        self.assertTrue(expected_keys.issubset(knobs.keys()))
        self.assertGreaterEqual(knobs["attempt_budget"], 12)

    def test_engine_scale_trade_deficit_raises_prob(self):
        rng = random.Random(42)
        league_low = types.SimpleNamespace(
            cpu_market_runtime={"season_cpu_trades": 40, "seasonal_target": 44, "post_deadline": False}
        )
        league_high_deficit = types.SimpleNamespace(
            cpu_market_runtime={"season_cpu_trades": 2, "seasonal_target": 44, "post_deadline": False}
        )
        low = compute_engine_trade_day_scale(
            league_low, day=120, max_day=192, deadline_phase=0.3, rng=rng
        )
        high = compute_engine_trade_day_scale(
            league_high_deficit, day=120, max_day=192, deadline_phase=0.3, rng=rng
        )
        self.assertGreater(high["trade_prob"], low["trade_prob"])
        self.assertGreaterEqual(high["max_executions"], low["max_executions"])

    def test_pool_weights_linear_in_pool_size(self):
        pool = [types.SimpleNamespace(id=f"t{i}") for i in range(5)]
        counts = {f"t{i}": i for i in range(5)}

        def tid(tm):
            return tm.id

        def ideo(_tm, buyer_side=False):
            return 0.5

        w = build_pool_sampling_weights(pool, counts, team_id_fn=tid, ideology_fn=ideo, buyer_side=False)
        self.assertEqual(len(w), len(pool))
        self.assertTrue(all(x >= 0.05 for x in w))


if __name__ == "__main__":
    unittest.main()
