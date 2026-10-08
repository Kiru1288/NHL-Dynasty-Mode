"""
Adaptive CPU trade market scaling.

Design goals:
- O(1) knob computation per sim day (telemetry is fixed-size counters).
- O(T) per trade batch for pool weights (T = team count, ~32).
- O(R log R) candidate ranking per (seller, buyer, motive) cache key, not per attempt.

Runtime state lives on league.cpu_market_runtime (dict).
"""

from __future__ import annotations

from typing import Any, Dict, List

# Season targets (tunable without touching proposer loops).
TARGET_PICK_RATE = 0.36
TARGET_REBUILD_FUTURES_RATE = 0.55
MIN_PICK_RATE_BEFORE_PRESSURE = 0.22
MAX_FAIRNESS_GAP = 18.0
BASE_FAIRNESS_GAP = 14.0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def season_day_ratio(calendar_cursor: int, regular_season_last_index: int) -> float:
    total = max(1, int(regular_season_last_index))
    return max(0.0, min(1.0, float(calendar_cursor) / float(total)))


def _telemetry_rates(telemetry: Dict[str, Any]) -> Dict[str, float]:
    trades = max(0, _safe_int(telemetry.get("trades"), 0))
    if trades <= 0:
        return {"pick_rate": 0.0, "rebuild_futures_rate": 0.0, "trades": 0.0}
    with_pick = max(0, _safe_int(telemetry.get("with_pick"), 0))
    rebuild_sales = max(0, _safe_int(telemetry.get("rebuild_sales"), 0))
    rebuild_futures = max(0, _safe_int(telemetry.get("rebuild_sales_with_futures"), 0))
    return {
        "pick_rate": with_pick / trades,
        "rebuild_futures_rate": (rebuild_futures / rebuild_sales) if rebuild_sales else 0.0,
        "trades": float(trades),
    }


def compute_proposer_adaptive_knobs(
    league: Any,
    *,
    calendar_cursor: int,
    regular_season_last_index: int,
    deadline_phase: float,
    max_executions: int,
    base_fairness_gap: float = BASE_FAIRNESS_GAP,
) -> Dict[str, Any]:
    """
    O(1) — reads league.cpu_market_runtime telemetry only.
    """
    runtime = getattr(league, "cpu_market_runtime", None)
    if not isinstance(runtime, dict):
        runtime = {}
    telemetry = runtime.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {}

    rates = _telemetry_rates(telemetry)
    pick_rate = rates["pick_rate"]
    rebuild_fut = rates["rebuild_futures_rate"]

    pick_deficit = max(0.0, TARGET_PICK_RATE - pick_rate)
    rebuild_deficit = max(0.0, TARGET_REBUILD_FUTURES_RATE - rebuild_fut)

    # Smooth pressure: only ramp after we have signal (>= 6 trades logged).
    trades_n = rates["trades"]
    signal = min(1.0, trades_n / 12.0) if trades_n > 0 else 0.0

    pick_pressure = min(1.0, signal * (pick_deficit / max(TARGET_PICK_RATE, 0.01)))
    futures_pressure = min(1.0, signal * (rebuild_deficit / max(TARGET_REBUILD_FUTURES_RATE, 0.01)))

    deadline = max(0.0, min(1.0, _safe_float(deadline_phase)))
    day_ratio = season_day_ratio(calendar_cursor, regular_season_last_index)

    fairness_gap = base_fairness_gap
    if pick_rate < MIN_PICK_RATE_BEFORE_PRESSURE and trades_n >= 8:
        fairness_gap = min(MAX_FAIRNESS_GAP, fairness_gap + 2.0 + pick_pressure * 3.0)
    if deadline > 0.55:
        fairness_gap = min(MAX_FAIRNESS_GAP, fairness_gap + 1.0)

    prefer_pick_only_boost = 0.12 * pick_pressure + 0.08 * futures_pressure
    if deadline > 0.45:
        prefer_pick_only_boost += 0.06

    peer_modulo = 9
    if pick_pressure > 0.45:
        peer_modulo = 11
    if futures_pressure > 0.5:
        peer_modulo = max(peer_modulo, 12)

    attempt_mult = 16 if deadline <= 0.45 else 22
    attempt_mult += int(round(4 * pick_pressure))
    attempt_budget = max(12, int(max_executions) * attempt_mult)

    max_exec_boost = 0
    if pick_rate < TARGET_PICK_RATE * 0.65 and trades_n >= 10:
        max_exec_boost = 1
    if deadline > 0.75 and pick_rate < TARGET_PICK_RATE:
        max_exec_boost = max(max_exec_boost, 1)

    value_delta_pick_threshold = 2.5 - (1.2 * pick_pressure)
    rebuild_depth_pick_chance = 0.38 + 0.22 * futures_pressure

    runtime["adaptive_knobs"] = {
        "pick_rate": round(pick_rate, 4),
        "rebuild_futures_rate": round(rebuild_fut, 4),
        "pick_pressure": round(pick_pressure, 4),
        "futures_pressure": round(futures_pressure, 4),
        "fairness_gap_max": round(fairness_gap, 3),
        "peer_modulo": peer_modulo,
        "attempt_budget": attempt_budget,
        "prefer_pick_only_boost": round(prefer_pick_only_boost, 4),
        "max_exec_boost": max_exec_boost,
        "value_delta_pick_threshold": round(value_delta_pick_threshold, 3),
        "rebuild_depth_pick_chance": round(rebuild_depth_pick_chance, 4),
        "day_ratio": round(day_ratio, 4),
        "deadline_phase": round(deadline, 4),
    }
    setattr(league, "cpu_market_runtime", runtime)
    return runtime["adaptive_knobs"]


def compute_engine_trade_day_scale(
    league: Any,
    *,
    day: int,
    max_day: int,
    deadline_phase: float,
    rng: Any,
) -> Dict[str, Any]:
    """
    O(H) trade history scan is handled in engine (incremental index).
    This function is O(1) given precomputed season_cpu_trades / deficit.
    """
    runtime = getattr(league, "cpu_market_runtime", None)
    if not isinstance(runtime, dict):
        runtime = {}
        setattr(league, "cpu_market_runtime", runtime)

    season_cpu_trades = _safe_int(runtime.get("season_cpu_trades"), 0)
    if "seasonal_target" not in runtime:
        try:
            runtime["seasonal_target"] = int(44 + round((rng.random() - 0.5) * 12))
        except Exception:
            runtime["seasonal_target"] = 44
    seasonal_target = max(38, min(58, _safe_int(runtime.get("seasonal_target"), 44)))

    day_ratio = season_day_ratio(day, max_day)
    if day_ratio < 0.25:
        expected_curve = 0.14
    elif day_ratio < 0.5:
        expected_curve = 0.34
    elif day_ratio < 0.75:
        expected_curve = 0.62
    elif day_ratio < 0.9:
        expected_curve = 0.86
    else:
        expected_curve = 0.98

    expected_by_now = max(0, int(round(float(seasonal_target) * expected_curve)))
    trade_deficit = max(0, expected_by_now - season_cpu_trades)

    deadline = max(0.0, min(1.0, _safe_float(deadline_phase)))
    trade_prob = 0.030 + 0.26 * deadline
    trade_prob += min(0.10, 0.015 * max(0, trade_deficit - 2))
    if deadline > 0.7:
        trade_prob += 0.05

    telemetry = runtime.get("telemetry") or {}
    rates = _telemetry_rates(telemetry if isinstance(telemetry, dict) else {})
    if rates["pick_rate"] < TARGET_PICK_RATE * 0.7 and rates["trades"] >= 8:
        trade_prob += 0.025

    trade_prob = max(0.022, min(0.72, trade_prob))

    max_exec = 1
    if deadline > 0.50 or trade_deficit >= 4:
        max_exec += 1
    if deadline > 0.75 or trade_deficit >= 6:
        max_exec += 1
    if deadline > 0.90 or trade_deficit >= 8:
        max_exec += 1
    max_exec = max(1, min(4, max_exec))

    post_deadline = bool(runtime.get("post_deadline"))
    forced_market_check = (not post_deadline) and trade_deficit >= 4 and (int(day) % 3 == 0)

    return {
        "trade_prob": trade_prob,
        "max_executions": max_exec,
        "forced_market_check": forced_market_check,
        "trade_deficit": trade_deficit,
        "seasonal_target": seasonal_target,
        "season_cpu_trades": season_cpu_trades,
        "day_ratio": day_ratio,
    }


def build_pool_sampling_weights(
    pool: List[Any],
    team_trade_counts: Dict[str, int],
    *,
    team_id_fn: Any,
    ideology_fn: Any,
    buyer_side: bool,
) -> List[float]:
    """O(|pool|) weights for random.choices — computed once per pool refresh."""
    weights: List[float] = []
    for tm in pool:
        tid = str(team_id_fn(tm))
        w = 1.0 + ideology_fn(tm, buyer_side=buyer_side)
        w /= 1.0 + 0.85 * float(team_trade_counts.get(tid, 0))
        weights.append(max(0.05, w))
    return weights
