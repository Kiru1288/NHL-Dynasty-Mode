"""
Benchmark franchise calendar advance: 7 / 15 / 30 days with component timings.

Usage (from repo root):
  python backend/scripts/bench_franchise_advance.py
  python backend/scripts/bench_franchise_advance.py --seed 42 --team TOR --days 7,15,30
  python backend/scripts/bench_franchise_advance.py --skip-regular  # stay at franchise start
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIM = os.path.join(ROOT, "SimEngine")
BACKEND = os.path.join(ROOT, "backend")
for p in (SIM, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("NHL_PERF", "1")


class TimerBoard:
    def __init__(self) -> None:
        self.totals_ms: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)

    def wrap(self, name: str, fn: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.totals_ms[name] += (time.perf_counter() - t0) * 1000.0
                self.counts[name] += 1

        wrapped.__name__ = getattr(fn, "__name__", name)
        return wrapped

    def rows(self) -> List[Tuple[str, float, int, float]]:
        out = []
        for name, total in self.totals_ms.items():
            n = self.counts[name]
            out.append((name, total, n, total / n if n else 0.0))
        out.sort(key=lambda r: -r[1])
        return out

    def reset(self) -> None:
        self.totals_ms.clear()
        self.counts.clear()


def _fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.0f}ms"


def _clone_session(session: Any) -> Any:
    for attr in list(vars(session).keys()):
        if attr.endswith("_lock"):
            try:
                delattr(session, attr)
            except Exception:
                pass
    try:
        return copy.deepcopy(session)
    except Exception:
        import pickle

        return pickle.loads(pickle.dumps(session, protocol=pickle.HighestProtocol))


def _fast_forward_to_regular(session: Any, *, max_days: int = 120) -> Dict[str, Any]:
    from services.franchise_sim import advance_franchise_bulk

    info: Dict[str, Any] = {"days_advanced": 0, "phase": str(getattr(session, "phase", ""))}
    if info["phase"] == "regular":
        return info
    session._bulk_auto_resolve_injuries = True
    result = advance_franchise_bulk(
        session,
        mode="days",
        count=max_days,
        auto_resolve_decisions=True,
    )
    info["days_advanced"] = int(result.get("steps_completed") or 0)
    info["phase"] = str(getattr(session, "phase", ""))
    info["cursor"] = int(getattr(session, "calendar_cursor", 0) or 0)
    info["iso"] = str(result.get("iso") or "")
    return info


def _install_wrappers(fs_module: Any, board: TimerBoard) -> Dict[str, Any]:
    originals: Dict[str, Any] = {}
    targets = [
        ("daily_league_tick", "_franchise_daily_league_tick"),
        ("simulate_slots_day", "_simulate_slots_for_day"),
        ("finalize_day", "_finalize_regular_calendar_day"),
        ("prospect_sync", "_sync_prospect_stats_to_calendar"),
        ("payload_invalidate", "invalidate_session_payload_caches"),
    ]
    for label, attr in targets:
        if hasattr(fs_module, attr):
            originals[attr] = getattr(fs_module, attr)
            setattr(fs_module, attr, board.wrap(label, originals[attr]))

    # Storyline passes (imported inside finalize — patch at source)
    try:
        from app.sim_engine.franchise import storyline_engine as se

        for label, attr in (
            ("storylines_data", "franchise_record_data_storylines"),
            ("storylines_cause", "franchise_cause_storyline_daily_pass"),
        ):
            if hasattr(se, attr):
                originals[f"se.{attr}"] = getattr(se, attr)
                setattr(se, attr, board.wrap(label, originals[f"se.{attr}"]))
    except Exception:
        pass

    return originals


def _restore_wrappers(fs_module: Any, originals: Dict[str, Any]) -> None:
    for key, fn in originals.items():
        if key.startswith("se."):
            from app.sim_engine.franchise import storyline_engine as se

            setattr(se, key[3:], fn)
        elif hasattr(fs_module, key):
            setattr(fs_module, key, fn)


def _run_advance(session: Any, days: int, *, auto_resolve: bool = True) -> Dict[str, Any]:
    from services.franchise_sim import advance_franchise_bulk

    t0 = time.perf_counter()
    result = advance_franchise_bulk(
        session,
        mode="days",
        count=int(days),
        auto_resolve_decisions=bool(auto_resolve),
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    steps_ok = int(result.get("steps_completed") or 0)
    return {
        "wall_ms": wall_ms,
        "steps_ok": steps_ok,
        "stopped": result.get("stopped_reason"),
        "status": result.get("status"),
        "light_mode": True,
        "cursor": int(getattr(session, "calendar_cursor", 0) or 0),
        "iso": str(result.get("iso") or ""),
    }


def _print_component_table(board: TimerBoard, wall_ms: float, days: int) -> None:
    rows = board.rows()
    if not rows:
        print("  (no component wrappers fired)")
        return
    accounted = sum(r[1] for r in rows)
    print(f"  {'component':<22} {'total':>10} {'calls':>7} {'avg':>10} {'%wall':>7}")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 7} {'-' * 10} {'-' * 7}")
    for name, total, count, avg in rows:
        pct = (100.0 * total / wall_ms) if wall_ms > 0 else 0
        print(f"  {name:<22} {_fmt_ms(total):>10} {count:7d} {_fmt_ms(avg):>10} {pct:6.1f}%")
    print(f"  {'(wrapped sum)':<22} {_fmt_ms(accounted):>10}          {_fmt_ms(accounted / max(days, 1)):>10} per day")
    unwrapped = max(0.0, wall_ms - accounted)
    if unwrapped > 50:
        print(f"  {'(unwrapped/other)':<22} {_fmt_ms(unwrapped):>10}          {100 * unwrapped / wall_ms:6.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark franchise advance timings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--team", type=str, default="TOR")
    parser.add_argument("--days", type=str, default="7,15,30")
    parser.add_argument("--skip-regular", action="store_true", help="Do not fast-forward to regular season")
    args = parser.parse_args()

    day_counts = [int(x.strip()) for x in args.days.split(",") if x.strip()]
    if not day_counts:
        raise SystemExit("No day counts specified")

    import services.franchise_sim as fs
    from services.franchise_sim import start_franchise
    from services.perf_profiler import reset as perf_reset, snapshot as perf_snapshot

    print("=" * 72)
    print("Franchise advance benchmark")
    print(f"  seed={args.seed}  team={args.team}  day_counts={day_counts}")
    print("=" * 72)

    t_boot = time.perf_counter()
    base = start_franchise(
        team_query=args.team,
        head_coach_name="Bench Coach",
        coach_archetype="balanced",
        seed=args.seed,
        player_universe="generated",
    )
    boot_ms = (time.perf_counter() - t_boot) * 1000.0
    print(f"\nBoot franchise: {_fmt_ms(boot_ms)}  phase={getattr(base, 'phase', '?')}  cursor={getattr(base, 'calendar_cursor', '?')}")

    if not args.skip_regular:
        ff = _fast_forward_to_regular(base)
        print(
            f"Fast-forward to regular: +{ff.get('days_advanced', 0)} days -> "
            f"phase={ff.get('phase')} cursor={ff.get('cursor')} iso={ff.get('iso')}"
        )

    baseline = _clone_session(base)
    print(f"Baseline snapshot: phase={getattr(baseline, 'phase', '?')} cursor={getattr(baseline, 'calendar_cursor', '?')}")

    results: List[Dict[str, Any]] = []

    for days in day_counts:
        print("\n" + "-" * 72)
        print(f"RUN: {days} calendar days (bulk advance, auto_resolve=True)")
        print("-" * 72)

        session = _clone_session(baseline)
        board = TimerBoard()
        perf_reset()
        originals = _install_wrappers(fs, board)

        try:
            meta = _run_advance(session, days)
        finally:
            _restore_wrappers(fs, originals)

        wall = float(meta["wall_ms"])
        per_day = wall / max(int(meta["steps_ok"]) or days, 1)
        light = "yes" if meta["light_mode"] else "no"
        print(f"  Wall time:     {_fmt_ms(wall)}  ({_fmt_ms(per_day)} / day)")
        print(f"  Steps OK:      {meta['steps_ok']}  stopped={meta.get('stopped')}")
        print(f"  Light stats:   {light}  (bulk light strength path for multi-day advance)")
        print(f"  End cursor:    {meta['cursor']}  iso={meta['iso']}")
        print("\n  Component breakdown:")
        _print_component_table(board, wall, days)

        perf = perf_snapshot(top_n=15)
        top = perf.get("top_by_total_ms") or []
        if top:
            print("\n  perf_profiler top buckets:")
            for row in top[:10]:
                print(
                    f"    {row.get('name', '?'):<36} "
                    f"total={_fmt_ms(float(row.get('total_ms') or 0))} "
                    f"avg={_fmt_ms(float(row.get('avg_ms') or 0))} "
                    f"n={row.get('count')}"
                )

        results.append({"days": days, "wall_ms": wall, "per_day_ms": per_day, "light_mode": light == "yes", **meta})

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  {'days':>5}  {'wall':>10}  {'per day':>10}  {'light':>6}  {'steps':>6}")
    for r in results:
        print(
            f"  {r['days']:5d}  {_fmt_ms(r['wall_ms']):>10}  "
            f"{_fmt_ms(r['per_day_ms']):>10}  {'yes' if r['light_mode'] else 'no':>6}  "
            f"{r['steps_ok']:6d}"
        )
    print()


if __name__ == "__main__":
    main()
