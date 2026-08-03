"""Micro-bench light vs full game stat accumulation (no full franchise boot)."""
from __future__ import annotations

import os
import sys
import time
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIM = os.path.join(ROOT, "SimEngine")
for p in (SIM, os.path.join(ROOT, "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> None:
    from app.sim_engine.engine import SimEngine

    t0 = time.perf_counter()
    sim = SimEngine(seed=42, debug=False, populate_initial_rosters=True)
    boot_ms = (time.perf_counter() - t0) * 1000.0
    teams = list(sim.league.teams)
    home, away = teams[0], teams[1]
    print(f"boot_ms={boot_ms:.0f} teams={len(teams)}")

    def bench(label, fn, n):
        led: dict = {}
        # warm
        fn(led, 0)
        led.clear()
        t1 = time.perf_counter()
        for i in range(n):
            fn(led, i)
        ms = (time.perf_counter() - t1) * 1000.0 / n
        sample = next(iter(led.values()), {})
        print(
            f"{label}_ms_per_game={ms:.2f} n={n} "
            f"cf={sample.get('cf')} xgf={sample.get('xgf')} g={sample.get('g')} pts={sample.get('pts')}"
        )
        return ms

    light_ms = bench(
        "light",
        lambda led, i: sim._accumulate_light_strength_game_stats(
            random.Random(1000 + i), home, away, "H", "A", 3 + (i % 3), 2, False, led
        ),
        50,
    )
    full_ms = bench(
        "full",
        lambda led, i: sim.accumulate_unified_game_stats(
            random.Random(2000 + i),
            home,
            away,
            "H",
            "A",
            3 + (i % 3),
            2,
            False,
            led,
            light_mode=False,
            build_game_payload=False,
        ),
        15,
    )
    speedup = (full_ms / light_ms) if light_ms > 0 else 0
    print(f"speedup_light_vs_full={speedup:.1f}x")
    # Extrapolate ~16 games/day * 7 days
    print(
        f"est_7d_stat_only_light_s={light_ms * 16 * 7 / 1000:.1f} "
        f"est_7d_stat_only_full_s={full_ms * 16 * 7 / 1000:.1f}"
    )


if __name__ == "__main__":
    main()
