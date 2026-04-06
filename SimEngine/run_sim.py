# backend/app/run_sim.py
"""
NHL Franchise Mode — Universe Generator + Career Runner (NEW)
============================================================

This runner is a top-level orchestration / testing harness.

It:
- Creates/loads a league universe (teams, rosters, coaches, economy, era, parity)
- Simulates league-wide seasons (Universe thread) + optional player career (Career thread)
- Produces human-readable narrative timeline logs + machine-readable JSON artifacts
- Supports deterministic reproduction via one master seed with seed-splitting
- Supports scenario injection (force trades, cap shocks, era shifts, rebuilds, etc.)
- Owns all I/O; engine must not write files
- Never crashes silently: it will dump context, last log lines, snapshots, seed/config on failure

Standard library only in this runner.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field, asdict, is_dataclass
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
import uuid
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, Callable
from collections import deque

# =============================================================================
# Optional engine imports (defensive)
# =============================================================================

SIMENGINE_IMPORT_ERROR: Optional[str] = None
DRAFTLOTTERY_IMPORT_ERROR: Optional[str] = None
WAIVERS_IMPORT_ERROR: Optional[str] = None

SimEngine = None
LotteryTeam = None
run_draft_lottery = None
update_priority_after_claim = None  # optional if exists in your waiver module
waiver_ai_process = None  # optional entrypoint if you have one

try:
    # Your existing engine location may differ; keep this defensive.
    # Example in your repo: backend/app/sim_engine/engine.py => app.sim_engine.engine
    from app.sim_engine.engine import SimEngine as _SimEngine  # type: ignore
    SimEngine = _SimEngine
except Exception as e:
    SIMENGINE_IMPORT_ERROR = f"{type(e).__name__}: {e}"

_ENGINE_ECON: Any = None
try:
    from app.sim_engine import engine as _ENGINE_ECON  # type: ignore
except Exception:
    _ENGINE_ECON = None

try:
    # Required by prompt
    from app.sim_engine.draft.draft_lottery import LotteryTeam as _LotteryTeam, run_draft_lottery as _run_draft_lottery  # type: ignore
    LotteryTeam = _LotteryTeam
    run_draft_lottery = _run_draft_lottery
except Exception as e:
    DRAFTLOTTERY_IMPORT_ERROR = f"{type(e).__name__}: {e}"

try:
    # Your waiver AI entrypoints may differ; keep it defensive.
    # If you have a module that exposes update_priority_after_claim, import it.
    from app.sim_engine.waivers.waivers import update_priority_after_claim as _update_priority_after_claim  # type: ignore
    update_priority_after_claim = _update_priority_after_claim
except Exception as e:
    WAIVERS_IMPORT_ERROR = f"{type(e).__name__}: {e}"

try:
    from app.sim_engine.generation.name_generator import generate_human_identity as _generate_human_identity  # type: ignore
    generate_human_identity = _generate_human_identity
except Exception:
    generate_human_identity = None  # fallback to placeholder names if unavailable

try:
    from app.sim_engine.generation.draft_class_generator import generate_draft_class as _generate_draft_class_ref  # type: ignore  # noqa: F401
except Exception:
    _generate_draft_class_ref = None

try:
    from app.sim_engine.progression import run_player_progression as _run_player_progression  # type: ignore
except Exception:
    _run_player_progression = None

try:
    from app.sim_engine.progression.development import assign_career_phase_from_age as _assign_career_phase_from_age  # type: ignore
except Exception:
    _assign_career_phase_from_age = None

try:
    from app.sim_engine.tuning.era_modifiers import apply_era_to_league as _tuning_apply_era_to_league  # type: ignore
    from app.sim_engine.tuning import normalization as _tuning_normalization  # type: ignore
    from app.sim_engine.tuning import validators as _tuning_validators  # type: ignore
    from app.sim_engine.tuning import probability_tables as _tuning_probability_tables  # type: ignore
    from app.sim_engine.tuning.era_modifiers import resolve_era_profile as _tuning_resolve_era_profile  # type: ignore
except Exception:
    _tuning_apply_era_to_league = None
    _tuning_normalization = None
    _tuning_validators = None
    _tuning_probability_tables = None
    _tuning_resolve_era_profile = None

try:
    from app.sim_engine.entities.team import update_team_gm_strategic_profile as _update_team_gm_strategic_profile  # type: ignore
except Exception:
    _update_team_gm_strategic_profile = None

# =============================================================================
# Constants / Small helpers
# =============================================================================

LOG_LEVELS = ("minimal", "normal", "debug", "insane")
RUN_MODES = ("combined", "career_only", "universe_only", "regression")

DEFAULT_ERAS = (
    "speed_and_skill",
    "dead_puck",
    "goalie_dominance",
    "power_play_era",
    "run_and_gun",
    "two_way_chess",
)

DEFAULT_TEAM_ARCHETYPES = (
    "draft_and_develop",
    "win_now",
    "contender",
    "balanced",
    "rebuild",
    "chaos_agent",
)

EVENT_TYPES = (
    "TRADE",
    "SIGNING",
    "WAIVER",
    "COACH_FIRE",
    "COACH_HIRE",
    "LOTTERY",
    "DRAFT",
    "ERA_SHIFT",
    "CAP_SHOCK",
    "REBUILD_COMMIT",
    "EXPANSION",
    "RELOCATION",
    "SUSPENSION",
    "NOTE",
)

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def now_utc_compact() -> str:
    # Timestamp for run folder
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def fmt_money(millions: float) -> str:
    # millions -> dollars string
    dollars = int(round(millions * 1_000_000))
    return f"{dollars:,}"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def stable_json_dumps(obj: Any, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

# =============================================================================
# RNG Seed splitting utilities (deterministic)
# =============================================================================

def split_seed(master_seed: int, label: str) -> int:
    """
    Deterministically derives a sub-seed from (master_seed, label).
    """
    h = hashlib.sha256(f"{master_seed}::{label}".encode("utf-8")).digest()
    # Use 64-bit chunk
    return int.from_bytes(h[:8], "big", signed=False)

def rng_from_seed(seed: int) -> random.Random:
    r = random.Random()
    r.seed(seed)
    return r


def draft_lottery_seed(master_seed: int, year: int) -> int:
    """Same seed the runner uses for entry draft + lottery (single source of truth)."""
    return split_seed(master_seed, f"draft::{year}")


def build_full_draft_order_32(standings: List[TeamStanding], lottery_pick_order_16: List[str]) -> List[str]:
    """Lottery order (non-playoff 16) + playoff tail worst→best among top 16 by standings."""
    if len(standings) < 16:
        return list(lottery_pick_order_16)
    tail = [s.team_id for s in reversed(standings[:16])]
    return list(lottery_pick_order_16[:16]) + tail


def total_prospect_pipeline(teams: List[Any]) -> int:
    return sum(len(getattr(t, "prospect_pool", None) or []) for t in teams)


def _runner_roster_player_age(p: Any) -> int:
    ident = getattr(p, "identity", None)
    if ident is not None:
        try:
            if isinstance(ident, dict):
                return int(ident.get("age", 0) or 0)
            return int(getattr(ident, "age", 0) or 0)
        except (TypeError, ValueError):
            pass
    try:
        return int(_safe_getattr(p, "age", 0))
    except (TypeError, ValueError):
        return 0


def league_age_distribution_stats(teams: List[Any]) -> Dict[str, Any]:
    """Roster demographics for soft age balancing (counts + percentages 0–100)."""
    u24 = prime = v30p = 0
    for t in teams:
        for p in getattr(t, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            age = _runner_roster_player_age(p)
            if age < 24:
                u24 += 1
            elif age <= 29:
                prime += 1
            else:
                v30p += 1
    total = u24 + prime + v30p
    if total <= 0:
        return {
            "total": 0,
            "u24": 0,
            "prime": 0,
            "v30_plus": 0,
            "pct_u24": 0.0,
            "pct_prime": 0.0,
            "pct_v30": 0.0,
        }
    return {
        "total": total,
        "u24": u24,
        "prime": prime,
        "v30_plus": v30p,
        "pct_u24": 100.0 * u24 / total,
        "pct_prime": 100.0 * prime / total,
        "pct_v30": 100.0 * v30p / total,
    }


def _normalize_age_stats_from_engine(d: Dict[str, Any]) -> Dict[str, Any]:
    """Map SimEngine.compute_league_age_distribution() to league_age_distribution_stats shape."""
    t = int(d.get("total_players", 0) or 0)
    if t <= 0:
        return {
            "total": 0,
            "u24": 0,
            "prime": 0,
            "v30_plus": 0,
            "pct_u24": 0.0,
            "pct_prime": 0.0,
            "pct_v30": 0.0,
        }
    return {
        "total": t,
        "u24": int(d.get("count_under_24", 0)),
        "prime": int(d.get("count_prime", 0)),
        "v30_plus": int(d.get("count_30_plus", 0)),
        "pct_u24": float(d.get("pct_under_24", 0.0)),
        "pct_prime": float(d.get("pct_prime", 0.0)),
        "pct_v30": float(d.get("pct_30_plus", 0.0)),
    }


def _emit_age_distribution_check(logger: RunnerLogger, stats: Dict[str, Any]) -> None:
    if int(stats.get("total", 0) or 0) <= 0:
        return
    logger.emit("AGE DISTRIBUTION CHECK", "normal")
    logger.emit(f"  U24: {stats['u24']} ({stats['pct_u24']:.1f}%)", "normal")
    logger.emit(f"  24–29: {stats['prime']} ({stats['pct_prime']:.1f}%)", "normal")
    logger.emit(f"  30+: {stats['v30_plus']} ({stats['pct_v30']:.1f}%)", "normal")


def _emit_age_balance_action(logger: RunnerLogger, typ: str, reason: str) -> None:
    logger.emit("AGE BALANCE ACTION:", "normal")
    logger.emit(f"  type={typ}", "normal")
    logger.emit(f"  reason={reason}", "normal")


def age_balance_segment_counts(teams: List[Any]) -> Tuple[int, int, int]:
    u24 = prime = vet30 = 0
    for t in teams:
        for p in getattr(t, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            age = _runner_roster_player_age(p)
            if age < 24:
                u24 += 1
            elif age <= 29:
                prime += 1
            else:
                vet30 += 1
    return u24, prime, vet30


def _team_archetype(state: UniverseState, team_id: str) -> str:
    return str(state.team_archetypes.get(team_id, "balanced") or "balanced")


def _team_mean_roster_age(team: Any) -> float:
    roster = list(getattr(team, "roster", None) or [])
    ages: List[int] = []
    for p in roster:
        if getattr(p, "retired", False):
            continue
        ages.append(int(_runner_roster_player_age(p) or 26))
    if not ages:
        return 27.0
    return float(sum(ages)) / float(len(ages))


def _identity_exp_and_noise_mods(
    state: UniverseState,
    team_id: str,
    team_obj: Any,
    r: random.Random,
) -> Tuple[float, float, float]:
    """
    Returns (delta on win expectation before clamp, noise multiplier, logged strength_mod).
    Subtle identity shaping + carry from prior seasons.
    """
    arch = _team_archetype(state, team_id).lower()
    carry = float((state.identity_carry_momentum or {}).get(team_id, 0.0))
    carry = clamp(carry, -0.09, 0.09)
    mean_age = _team_mean_roster_age(team_obj) if team_obj is not None else 27.0

    exp_delta = 0.45 * carry
    noise_mult = 1.0

    if arch == "win_now":
        exp_delta += 0.011
        noise_mult *= 0.88
        if mean_age > 29.4:
            exp_delta -= 0.0065
        elif mean_age < 26.2:
            exp_delta += 0.0035
    elif arch == "rebuild":
        exp_delta -= 0.017
        noise_mult *= 1.17
        if mean_age < 24.6:
            exp_delta -= 0.004
    elif arch == "draft_and_develop":
        exp_delta -= 0.0055 + 0.0020 * max(0.0, 26.0 - mean_age) / 6.0
        noise_mult *= 1.06
        exp_delta += 0.35 * carry
    elif arch == "contender":
        exp_delta += 0.0065
        noise_mult *= 0.94
        if mean_age > 30.0:
            exp_delta -= 0.005
        elif mean_age < 25.5:
            exp_delta += 0.0025
    elif arch == "chaos_agent":
        exp_delta += r.uniform(-0.019, 0.021)
        noise_mult *= 1.38
    else:
        noise_mult *= 1.0

    strength_mod = round(exp_delta, 4)
    return exp_delta, noise_mult, strength_mod


def _identity_guardrail_adjust_standings(
    standings: List[TeamStanding],
    state: UniverseState,
    r: random.Random,
) -> List[TeamStanding]:
    if len(standings) < 8:
        return standings
    out = list(standings)
    out.sort(key=lambda s: (s.points, s.goal_diff), reverse=True)
    n = len(out)
    playoff_cut = max(16, n // 2)

    for i, s in enumerate(out):
        arch = _team_archetype(state, s.team_id).lower()
        pts = s.points
        pct = s.point_pct
        gd = s.goal_diff

        if i < 3 and arch == "rebuild" and r.random() < 0.58:
            drop = r.randint(2, 7)
            pts = max(52, pts - drop)
            pct = clamp(float(pts) / 164.0, 0.30, 0.70)
            gd = int(clamp(gd - r.randint(4, 14), -120, 120))
            out[i] = dataclasses.replace(
                s,
                points=pts,
                point_pct=pct,
                goal_diff=gd,
                bucket=_bucket_from_point_pct(pct),
            )

    out.sort(key=lambda s: (s.points, s.goal_diff), reverse=True)

    for i, s in enumerate(out):
        arch = _team_archetype(state, s.team_id).lower()
        if i >= playoff_cut and arch in ("win_now", "contender") and r.random() < (0.36 if arch == "win_now" else 0.28):
            bump = r.randint(2, 6)
            pts = min(118, s.points + bump)
            pct = clamp(float(pts) / 164.0, 0.30, 0.70)
            gd = int(clamp(s.goal_diff + r.randint(2, 10), -120, 120))
            out[i] = dataclasses.replace(
                s,
                points=pts,
                point_pct=pct,
                goal_diff=gd,
                bucket=_bucket_from_point_pct(pct),
            )

    out.sort(key=lambda s: (s.points, s.goal_diff), reverse=True)
    return out


def _identity_trajectory_label(prev_pts: Optional[int], cur_pts: int) -> str:
    if prev_pts is None:
        return "stable"
    d = cur_pts - prev_pts
    if d >= 4:
        return "rising"
    if d <= -4:
        return "falling"
    return "stable"


def _identity_update_carry_momentum(
    state: UniverseState,
    standings: List[TeamStanding],
    r: random.Random,
) -> None:
    for s in standings:
        tid = s.team_id
        arch = _team_archetype(state, tid).lower()
        prev = (state.identity_prev_points or {}).get(tid)
        cur = int(s.points)
        delta = 0 if prev is None else cur - prev
        m = float((state.identity_carry_momentum or {}).get(tid, 0.0))

        if arch == "rebuild":
            if cur < 84:
                m += 0.0065
            if cur > 96:
                m -= 0.0055
            if delta > 11:
                m -= 0.004
            elif delta < -9:
                m += 0.0035
        elif arch == "win_now":
            if s.bucket in ("contender", "playoff"):
                m = min(0.065, m + 0.0028)
            elif cur < 87:
                m = max(-0.075, m - 0.005)
            if delta < -8:
                m = max(-0.075, m - 0.003)
        elif arch == "draft_and_develop":
            if 74 <= cur <= 93:
                m = min(0.055, m + 0.0022)
            if delta > 7:
                m = min(0.06, m + 0.0012)
        elif arch == "contender":
            if s.bucket == "contender":
                m = min(0.062, m + 0.0024)
            elif s.bucket in ("playoff",):
                m = min(0.05, m + 0.0014)
            elif cur < 86:
                m = max(-0.07, m - 0.0042)
            if delta < -7:
                m = max(-0.072, m - 0.0028)
        elif arch == "chaos_agent":
            m += r.uniform(-0.0035, 0.0035)

        state.identity_carry_momentum[tid] = clamp(m, -0.09, 0.09)
        state.identity_prev_points[tid] = cur


def _emit_identity_impact_logs(
    logger: RunnerLogger,
    state: UniverseState,
    standings: List[TeamStanding],
) -> None:
    if not standings:
        return
    mods = getattr(state, "identity_last_year_mod", None) or {}
    logger.emit("IDENTITY IMPACT:", "normal")
    for s in sorted(standings, key=lambda x: x.team_id):
        tid = s.team_id
        t = _team_archetype(state, tid)
        sm = mods.get(tid, 0.0)
        logger.emit(f"  team={tid} type={t} strength_mod={sm:+.4f}", "normal")


def _emit_identity_trajectory_logs(
    logger: RunnerLogger,
    state: UniverseState,
    standings: List[TeamStanding],
) -> None:
    if not standings:
        return
    prev_map = getattr(state, "identity_prev_points", None) or {}
    logger.emit("IDENTITY TRAJECTORY:", "normal")
    for s in sorted(standings, key=lambda x: x.team_id):
        tid = s.team_id
        t = _team_archetype(state, tid)
        prev = prev_map.get(tid)
        tr = _identity_trajectory_label(prev, int(s.points))
        logger.emit(f"  team={tid} trend={tr}", "normal")


def _identity_aggression_label(mode: str) -> str:
    m = (mode or "").lower()
    if m in ("win_now", "chaos_agent"):
        return "high"
    if m in ("rebuild", "draft_and_develop"):
        return "low"
    return "moderate"


def _identity_trade_bias(state: UniverseState, team_id: str) -> float:
    m = _team_archetype(state, team_id)
    if m == "win_now":
        return 1.22
    if m == "chaos_agent":
        return 1.34
    if m == "rebuild":
        return 1.14
    if m == "draft_and_develop":
        return 0.80
    return 1.0


def _emit_identity_behavior_logs(logger: RunnerLogger, state: UniverseState, standings: List[TeamStanding], trade_events: List[UniverseEvent]) -> None:
    if not standings:
        return
    top = sorted(standings, key=lambda s: (-s.points, -getattr(s, "goal_diff", 0)))[:10]
    for s in top:
        mode = _team_archetype(state, s.team_id)
        agg = _identity_aggression_label(mode)
        n = sum(1 for e in trade_events if s.team_id in (getattr(e, "teams", None) or []))
        logger.emit(f"IDENTITY BEHAVIOR: team={s.team_id} mode={mode} aggression={agg} trades_involving={n}", "normal")


def _runner_player_cap_hit_m(player: Any) -> float:
    for k in ("cap_hit_m", "contract_aav_m", "aav_m"):
        v = getattr(player, k, None)
        if v is not None:
            try:
                x = float(v)
                if x > 0:
                    return x
            except (TypeError, ValueError):
                pass
    c = getattr(player, "contract", None)
    if c is not None:
        for k in ("cap_hit_m", "cap_hit", "aav_m", "aav"):
            v = getattr(c, k, None)
            if v is not None:
                try:
                    x = float(v)
                    if x > 0:
                        return x
                except (TypeError, ValueError):
                    pass
    try:
        ovr = float(player.ovr()) if callable(getattr(player, "ovr", None)) else float(getattr(player, "ovr", 0.5))
    except Exception:
        ovr = 0.5
    return max(0.75, 1.0 + 9.0 * max(0.0, ovr - 0.50))


def _team_payroll_with_bad_contracts(team: Any) -> Tuple[float, float]:
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    payroll = 0.0
    bad = 0.0
    for p in roster:
        ch = _runner_player_cap_hit_m(p)
        payroll += ch
        age = _runner_roster_player_age(p)
        try:
            ovr = float(p.ovr()) if callable(getattr(p, "ovr", None)) else float(getattr(p, "ovr", 0.5))
        except Exception:
            ovr = 0.5
        if age >= 34 and ch >= 5.5 and ovr < 0.58:
            bad += ch * 0.18
        elif age >= 32 and ch >= 6.5:
            bad += ch * 0.10
        elif age >= 30 and ch >= 8.0 and ovr < 0.65:
            bad += ch * 0.07
    return payroll, bad


def _assign_cap_pressure_tiers(rows: List[Dict[str, Any]], rng: random.Random) -> None:
    """League-wide tier polish: keep non-uniform distribution (some high/cap hell, some low)."""
    n = len(rows)
    if n <= 0:
        return
    for r in rows:
        u = float(r.get("effective_ratio", r["ratio"]))
        if _ENGINE_ECON is not None and hasattr(_ENGINE_ECON, "cap_tier_from_usage_ratio"):
            r["pressure"] = str(_ENGINE_ECON.cap_tier_from_usage_ratio(u))
        elif u > 1.0:
            r["pressure"] = "cap_hell"
        elif u >= 0.95:
            r["pressure"] = "critical"
        elif u >= 0.85:
            r["pressure"] = "high"
        elif u >= 0.70:
            r["pressure"] = "moderate"
        else:
            r["pressure"] = "low"
    if n < 6:
        return
    nh = sum(1 for r in rows if r["pressure"] in ("high", "critical", "cap_hell"))
    target_h = max(1, int(round(0.255 * n)))
    if nh < target_h:
        pool = [r for r in rows if r["pressure"] == "moderate"]
        pool.sort(key=lambda x: -float(x.get("effective_ratio", x["ratio"])))
        for r in pool:
            if nh >= target_h:
                break
            r["pressure"] = "high"
            nh += 1
    n_hell = sum(1 for r in rows if r["pressure"] == "cap_hell")
    target_hell = max(0, min(n // 12, int(round(0.075 * n))))
    if target_hell > n_hell and n >= 10:
        pool = [r for r in rows if r["pressure"] == "critical" and float(r.get("ratio", 0)) >= 0.97]
        rng.shuffle(pool)
        for r in pool:
            if n_hell >= target_hell:
                break
            if rng.random() < 0.55:
                r["pressure"] = "cap_hell"
                n_hell += 1
    nlow = sum(1 for r in rows if r["pressure"] == "low")
    target_l = int(round(0.34 * n))
    if nlow < target_l:
        pool = [r for r in rows if r["pressure"] == "moderate"]
        pool.sort(key=lambda x: float(x.get("effective_ratio", x["ratio"])))
        for r in pool:
            if nlow >= target_l:
                break
            r["pressure"] = "low"
            nlow += 1


def _execute_league_cap_consequence_pass(
    teams: List[Any],
    league: Any,
    state: UniverseState,
    events: List[UniverseEvent],
    logger: RunnerLogger,
    rng: random.Random,
    year: int,
    run_cfg: RunConfig,
    tuning_year: Dict[str, Any],
) -> None:
    cap_m = float(state.salary_cap_m)
    if cap_m <= 0.0 or not teams:
        return
    if _ENGINE_ECON is not None:
        try:
            setattr(state, "runner_contract_inflation", float(_ENGINE_ECON.calculate_contract_inflation(state)))
        except Exception:
            try:
                setattr(state, "runner_contract_inflation", 1.0)
            except Exception:
                pass
    rows: List[Dict[str, Any]] = []
    numeric_by_tid: Dict[str, float] = {}
    strategy_by_tid: Dict[str, str] = {}
    team_by_id_pre = {_team_id(t): t for t in teams}
    for team in teams:
        tid = _team_id(team)
        arche = _team_archetype(state, str(tid))
        try:
            setattr(team, "_runner_team_archetype", arche)
        except Exception:
            pass
        target_usage: Optional[float] = None
        if _ENGINE_ECON is not None:
            try:
                _ENGINE_ECON.mark_team_roster_bad_contracts(team)
                ps = str((getattr(state, "power_states", None) or {}).get(str(tid), "") or "")
                target_usage = float(_ENGINE_ECON.resolve_cap_target_ratio_for_identity(arche, ps, rng))
                setattr(team, "cap_target_usage", target_usage)
                _ENGINE_ECON.nudge_team_payroll_toward_cap_target(
                    team, cap_m, target_usage, rng, archetype=str(arche)
                )
            except Exception:
                target_usage = getattr(team, "cap_target_usage", None)
        payroll, bad = _team_payroll_with_bad_contracts(team)
        ratio = float(payroll) / cap_m if cap_m else 0.0
        effective_ratio = float(payroll + bad) / cap_m if cap_m else 0.0
        try:
            setattr(team, "total_salary", float(payroll))
            setattr(team, "salary_cap_m", float(cap_m))
        except Exception:
            pass
        numeric_p = 0.0
        tier = "moderate"
        if _ENGINE_ECON is not None:
            try:
                numeric_p = float(
                    _ENGINE_ECON.calculate_cap_pressure(team, salary_cap_m=cap_m, total_salary_m=payroll)
                )
                tier = str(_ENGINE_ECON.cap_tier_from_usage_ratio(effective_ratio))
            except Exception:
                numeric_p = 0.2 + ratio * 0.3 if ratio < 0.75 else 0.7
                tier = "moderate"
        else:
            if ratio < 0.75:
                numeric_p = 0.2 + ratio * 0.3
            elif ratio < 0.90:
                numeric_p = 0.4 + (ratio - 0.75) * 1.2
            elif ratio < 1.00:
                numeric_p = 0.7 + (ratio - 0.90) * 2.5
            else:
                numeric_p = 1.0 + (ratio - 1.00) * 3.0
            if ratio > 1.0:
                tier = "cap_hell"
            elif ratio >= 0.95:
                tier = "critical"
            elif ratio >= 0.85:
                tier = "high"
            elif ratio >= 0.70:
                tier = "moderate"
            else:
                tier = "low"
        numeric_by_tid[str(tid)] = round(numeric_p, 4)
        rows.append(
            {
                "team_id": tid,
                "ratio": ratio,
                "effective_ratio": effective_ratio,
                "payroll_m": payroll,
                "bad_contract_m": bad,
                "pressure": tier,
                "numeric_pressure": numeric_p,
                "strategy": "balanced",
                "target_usage": target_usage,
            }
        )

    _assign_cap_pressure_tiers(rows, rng)

    for r in rows:
        tid = str(r["team_id"])
        tm = team_by_id_pre.get(tid)
        strategy = str(r.get("strategy", "balanced"))
        if tm is not None and _ENGINE_ECON is not None:
            try:
                strategy = str(
                    _ENGINE_ECON.update_team_strategy(
                        tm,
                        pressure=float(r["numeric_pressure"]),
                        salary_cap_m=cap_m,
                        total_salary_m=float(r["payroll_m"]),
                        forced_pressure_tier=str(r["pressure"]),
                    )
                )
                _ENGINE_ECON.apply_cap_pressure_effects(tm, salary_cap_m=cap_m)
            except Exception:
                pass
        r["strategy"] = strategy
        strategy_by_tid[tid] = strategy

    n_cap_hell = sum(1 for r in rows if str(r.get("pressure")) == "cap_hell")
    if n_cap_hell > 0:
        try:
            state.chaos_index = clamp(
                float(state.chaos_index) + 0.011 * float(n_cap_hell) + rng.random() * 0.008,
                0.06,
                0.94,
            )
        except Exception:
            pass

    pressure_map = {str(r["team_id"]): str(r["pressure"]) for r in rows}
    usage_pct_map = {str(r["team_id"]): round(100.0 * float(r["ratio"]), 1) for r in rows}
    state.runner_cap_pressure_by_team = dict(pressure_map)
    state.runner_cap_usage_pct_by_team = dict(usage_pct_map)
    state.runner_cap_numeric_pressure_by_team = dict(numeric_by_tid)
    state.runner_team_strategy_by_team = dict(strategy_by_tid)
    if league is not None:
        setattr(league, "_runner_cap_team_pressure", dict(pressure_map))
        setattr(league, "_runner_cap_team_usage_pct", dict(usage_pct_map))
        setattr(league, "_runner_cap_numeric_pressure", dict(numeric_by_tid))
        setattr(league, "_runner_team_strategy", dict(strategy_by_tid))

    logger.emit("CAP STATUS:", "normal")
    for r in sorted(rows, key=lambda x: str(x["team_id"])):
        logger.emit(
            f"  team={r['team_id']} usage={round(100.0 * float(r['ratio']), 1)}% pressure={r['pressure']}",
            "normal",
        )
        logger.emit(
            f"Cap Pressure: {round(float(r.get('numeric_pressure', 0)), 3)} Strategy: {r.get('strategy', 'balanced')}",
            "normal",
        )
    tuning_year["cap_league"] = {"teams": rows}

    for r in sorted(rows, key=lambda x: str(x["team_id"])):
        tu = r.get("target_usage")
        tu_s = f" target={float(tu):.2f}" if tu is not None else ""
        logger.emit(
            f"CAP SUMMARY: team={r['team_id']} usage={round(100.0 * float(r['ratio']), 1)}% "
            f"pressure={r['pressure']}{tu_s}",
            "normal",
        )

    team_by_id = team_by_id_pre
    for team in teams:
        tid = str(_team_id(team))
        tm = team_by_id.get(tid)
        if tm is None or _ENGINE_ECON is None:
            continue
        bad_logged = 0
        for p in [x for x in (getattr(tm, "roster", None) or []) if not getattr(x, "retired", False)]:
            try:
                if not _ENGINE_ECON.is_bad_contract(p):
                    continue
            except Exception:
                continue
            if bad_logged >= 3:
                break
            bad_logged += 1
            val = _runner_player_cap_hit_m(p)
            pln = _player_display_name(p)
            logger.emit(f"BAD CONTRACT FLAG: {pln} value={val}", "normal")
            logger.emit(f"Bad contract identified: {pln}", "normal")
            print(f"BAD CONTRACT FLAG: {pln} value={val}")
        try:
            cc = _ENGINE_ECON.cap_casualty_check(tm, salary_cap_m=cap_m)
        except Exception:
            cc = None
        if cc:
            pl = cc.get("player")
            pname = getattr(pl, "name", None) if pl is not None else None
            if not pname and pl is not None:
                pname = _player_display_name(pl)
            pname = str(pname or "?")
            logger.emit(f"CAP CASUALTY: {pname} moved due to cap pressure", "normal")
            print(f"CAP CASUALTY: {pname} moved due to cap pressure")
            events.append(
                UniverseEvent(
                    event_id=str(uuid.uuid4()),
                    year=year,
                    day=None,
                    type="NOTE",
                    teams=[tid],
                    headline=f"CAP CASUALTY: {tid} must move {pname} (salary pressure)",
                    details={
                        "reason": "cap_casualty",
                        "player_label": pname,
                        "cap_dump": True,
                        "storyline": "salary_cap_casualty",
                    },
                    impact_score=0.82,
                    tags=["salary_cap", "trade_rumor", "cap_casualty", "storyline"],
                )
            )

    for r in rows:
        pr = str(r["pressure"])
        if pr not in ("critical", "cap_hell"):
            continue
        tid = str(r["team_id"])
        if pr == "cap_hell":
            logger.emit("CAP EVENT: Team enters cap hell — emergency moves triggered", "normal")
        tm = team_by_id.get(tid)
        roster = [p for p in (getattr(tm, "roster", None) or []) if not getattr(p, "retired", False)] if tm else []
        for action, sub in (
            ("trade", "contract_dump"),
            ("waive", "roster_trim"),
            ("release", "buyout_termination"),
        ):
            label = "contract_slot"
            if roster:
                label = _player_display_name(rng.choice(roster))
            reason = "cap_overflow" if pr == "critical" else "cap_hell_emergency"
            logger.emit(f"CAP EVENT: team={tid} action={action} reason={reason}", "normal")
            if action == "trade":
                logger.emit("CAP ACTION: Salary dump trade executed", "normal")
            if action == "waive":
                logger.emit("CAP ACTION: Player waived due to cap constraints", "normal")
            events.append(
                UniverseEvent(
                    event_id=str(uuid.uuid4()),
                    year=year,
                    day=None,
                    type="NOTE",
                    teams=[tid],
                    headline=f"CAP COMPLIANCE ({action}): {tid} clears space — {label} ({sub})",
                    details={"cap_action": action, "player_label": label, "reason": reason, "usage_ratio": r["ratio"]},
                    impact_score=0.82 if pr == "cap_hell" else 0.72,
                    tags=["salary_cap", "compliance", action],
                )
            )

    for r in rows:
        if str(r["pressure"]) == "high" and rng.random() < 0.48:
            logger.emit("CAP EVENT: Cap crunch forces trade consideration", "normal")
            logger.emit(f"  team={r['team_id']} usage={round(100.0 * float(r['ratio']), 1)}%", "normal")

    bad_dump_logged = 0
    for r in rows:
        if str(r["pressure"]) not in ("high", "critical", "cap_hell"):
            continue
        if bad_dump_logged >= 10:
            break
        if rng.random() >= 0.40:
            continue
        if float(r.get("bad_contract_m", 0) or 0) <= 0.35:
            continue
        tid = str(r["team_id"])
        bad_dump_logged += 1
        logger.emit(f"CAP EVENT: team={tid} action=trade reason=bad_contract_pressure", "normal")
        events.append(
            UniverseEvent(
                event_id=str(uuid.uuid4()),
                year=year,
                day=None,
                type="NOTE",
                teams=[tid],
                headline=f"CAP PRESSURE: {tid} working the phones on a salary dump (bad-contract leverage)",
                details={"reason": "bad_contract_pressure", "bad_contract_m": r["bad_contract_m"]},
                impact_score=0.48,
                tags=["salary_cap", "trade_rumor"],
            )
        )


def _emit_cap_event_logs(logger: RunnerLogger, standings: List[TeamStanding], state: UniverseState, r: random.Random, year: int) -> None:
    """Optional extra cap narrative for moderate-pressure teams (always logged, never silent)."""
    if not standings:
        return
    pmap = getattr(state, "runner_cap_pressure_by_team", None) or {}
    for s in standings[:18]:
        p = str(pmap.get(s.team_id, "moderate")).lower()
        if p != "moderate":
            continue
        if r.random() < 0.14:
            move = "trade" if r.random() < 0.50 else "waive"
            logger.emit(f"CAP EVENT: team={s.team_id} action={move} reason=cap_flexibility_review", "normal")


def _sanitize_narrative_lines(lines: List[str], ctx: Dict[str, Any]) -> List[str]:
    """De-duplicate phrasing; gate rare superlatives; drop rookie-campaign lines for veterans."""
    out: List[str] = []
    seen: set = set()
    gen_left = int(ctx.get("generational_left", 2))
    hist_left = int(ctx.get("historic_left", 4))
    face_left = int(ctx.get("face_left", 2))
    _no_dedupe = {
        "player journey:",
        "storyline:",
        "news:",
        "------------------------------",
        "player journeys (season digest)",
        "key storylines",
        "narrative events",
        "==============================",
    }
    for raw in lines:
        line = (raw or "").strip()
        if not line:
            continue
        low = line.lower()
        if low in _no_dedupe or low.startswith("====") or line.startswith("  • "):
            try:
                line.encode("cp1252")
            except UnicodeEncodeError:
                line = line.encode("cp1252", errors="replace").decode("cp1252")
            out.append(line)
            continue
        if "rookie campaign" in low or "rookie season" in low:
            if any(x in low for x in ("veteran", "age 3", "31", "32", "33", "34", "35", "36", "37")):
                continue
        key = low[:220]
        if key in seen:
            continue
        seen.add(key)
        if "generational" in low:
            if gen_left <= 0:
                line = line.replace("generational", "high-end").replace("Generational", "High-end")
            else:
                gen_left -= 1
        if "historic" in low and hist_left <= 0:
            line = line.replace("historic", "strong").replace("Historic", "Strong")
        elif "historic" in low:
            hist_left -= 1
        if "face of the league" in low:
            if face_left <= 0:
                continue
            face_left -= 1
        try:
            line.encode("cp1252")
        except UnicodeEncodeError:
            line = line.encode("cp1252", errors="replace").decode("cp1252")
        out.append(line)
    ctx["generational_left"] = gen_left
    ctx["historic_left"] = hist_left
    ctx["face_left"] = face_left
    return out

# =============================================================================
# Safe serialization helpers
# =============================================================================

def safe_to_primitive(obj: Any, _depth: int = 0, _max_depth: int = 6) -> Any:
    """
    Convert arbitrary objects into JSON-safe primitives.
    Never raises.
    """
    try:
        if _depth > _max_depth:
            return str(obj)

        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # dataclass
        if is_dataclass(obj):
            return {k: safe_to_primitive(v, _depth + 1, _max_depth) for k, v in asdict(obj).items()}

        # enums
        # (Avoid importing Enum; treat by attribute)
        if hasattr(obj, "name") and hasattr(obj, "value"):
            # Could be Enum-like
            try:
                return str(obj.name)
            except Exception:
                pass

        # dict
        if isinstance(obj, dict):
            return {str(k): safe_to_primitive(v, _depth + 1, _max_depth) for k, v in obj.items()}

        # list/tuple/set
        if isinstance(obj, (list, tuple, set)):
            return [safe_to_primitive(v, _depth + 1, _max_depth) for v in obj]

        # objects with __dict__
        if hasattr(obj, "__dict__"):
            d = {}
            for k, v in vars(obj).items():
                if k.startswith("_"):
                    continue
                d[str(k)] = safe_to_primitive(v, _depth + 1, _max_depth)
            if d:
                return d

        return str(obj)
    except Exception:
        return str(obj)

def safe_to_dict(obj: Any) -> Dict[str, Any]:
    prim = safe_to_primitive(obj)
    return prim if isinstance(prim, dict) else {"value": prim}

# =============================================================================
# Config dataclasses (MUST exist)
# =============================================================================

@dataclass
class ScenarioEvent:
    type: str
    year: int
    target_team_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScenarioConfig:
    events: List[ScenarioEvent] = field(default_factory=list)

@dataclass
class UniverseConfig:
    # Enable toggles
    enable_trades: bool = True
    enable_free_agency: bool = True
    enable_draft: bool = True
    enable_waivers: bool = True
    enable_coach_changes: bool = True
    enable_injuries: bool = False
    enable_expansion: bool = False
    enable_relocation: bool = False

    # Core indices / targets
    chaos: float = 0.50
    parity_target: float = 0.50
    league_health_target: float = 0.50

    # Economics
    salary_cap_start: float = 92.0  # millions
    cap_growth_rate_mean: float = 0.048
    cap_growth_volatility: float = 0.020
    escrow_pressure: float = 0.10
    revenue_shock_frequency: float = 0.12  # per year chance

    # Era / rules
    era_shift_frequency: float = 0.08
    rules_change_frequency: float = 0.03  # narrative-only for now

    # Competitive environment
    chaos_index: float = 0.50
    upset_multiplier: float = 1.0
    trade_chaos_multiplier: float = 1.0
    contender_bias: float = 0.55
    rebuild_patience: float = 0.55

    # Transactions
    trade_rate: float = 0.55
    blockbuster_rate: float = 0.07
    rental_rate: float = 0.18
    cap_dump_rate: float = 0.12
    retention_rate: float = 0.22
    free_agency_rate: float = 0.60
    hometown_discount_rate: float = 0.03  # narrative

    waiver_rate: float = 0.25

    # Teams
    archetype_drift_rate: float = 0.18
    coach_firing_rate: float = 0.22
    development_quality_spread: float = 0.18
    market_size_spread: float = 0.20

    # Player career knobs (mostly narrative in runner)
    breakout_rate: float = 0.08
    bust_rate: float = 0.05
    injury_rate: float = 0.10
    regression_resistance_mean: float = 0.50
    contract_demand_inflation: float = 0.03

    # Testing / Debug
    dump_rng_traces: bool = False
    print_trade_eval: bool = False
    print_team_decision_reasons: bool = False

@dataclass
class RunConfig:
    seed: int
    years: int = 40
    start_year: int = 2025
    debug: bool = False
    mode: str = "combined"  # combined|career_only|universe_only|regression
    log_level: str = "normal"  # minimal|normal|debug|insane
    output_dir: str = "runs"
    write_json: bool = True
    write_debug_state: bool = False
    pretty_print: bool = False
    flush_each_year: bool = True

    # Regression harness
    regression_baseline_dir: str = "regression_baselines"

    # Optional hook after each simulated year (perspective mode / tooling). Engine responsibility
    # unchanged; callback is runner-level observation only. Signature:
    #   callback(year, ustate, uni_result, career_result, league_season_result, sim, league) -> None
    per_year_callback: Optional[Callable[..., None]] = None

# =============================================================================
# Event model (structured narrative)
# =============================================================================

@dataclass
class UniverseEvent:
    event_id: str
    year: int
    day: Optional[int]
    type: str
    teams: List[str]
    headline: str
    details: Dict[str, Any] = field(default_factory=dict)
    impact_score: float = 0.0
    tags: List[str] = field(default_factory=list)

# =============================================================================
# Output manager
# =============================================================================

class RunOutput:
    def __init__(self, run_dir: Path, pretty: bool, timeline_path: Optional[Path] = None):
        self.run_dir = run_dir
        self.pretty = pretty
        self.timeline_path = timeline_path if timeline_path is not None else run_dir / "timeline.log"
        self.errors_path = run_dir / "errors.log"
        self._timeline_fp = None
        self._timeline_overwrite = timeline_path is not None

    def open(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if self._timeline_overwrite else "a"
        self._timeline_fp = open(self.timeline_path, mode, encoding="utf-8")

    def close(self) -> None:
        try:
            if self._timeline_fp:
                self._timeline_fp.flush()
                self._timeline_fp.close()
        finally:
            self._timeline_fp = None

    def append_timeline(self, line: str, flush: bool = False) -> None:
        if self._timeline_fp is None:
            self.open()
        assert self._timeline_fp is not None
        self._timeline_fp.write(line + "\n")
        if flush:
            self._timeline_fp.flush()

    def write_json(self, filename: str, data: Any) -> None:
        path = self.run_dir / filename
        text = stable_json_dumps(safe_to_primitive(data), pretty=self.pretty)
        path.write_text(text, encoding="utf-8")

    def write_text(self, filename: str, text: str) -> None:
        (self.run_dir / filename).write_text(text, encoding="utf-8")

    def append_error(self, text: str) -> None:
        with open(self.errors_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

# =============================================================================
# Logging with ring buffer (last N lines)
# =============================================================================

def _ascii_safe_log_line(s: str) -> str:
    """Windows consoles / cp1252 timeline files: replace unencodable chars."""
    try:
        str(s).encode("cp1252")
        return str(s)
    except UnicodeEncodeError:
        return str(s).encode("cp1252", errors="replace").decode("cp1252")


class RunnerLogger:
    def __init__(self, out: RunOutput, log_level: str, flush_each_year: bool):
        self.out = out
        self.log_level = log_level if log_level in LOG_LEVELS else "normal"
        self.flush_each_year = flush_each_year
        self.ring = deque(maxlen=200)

    def _level_rank(self, lvl: str) -> int:
        return {"minimal": 0, "normal": 1, "debug": 2, "insane": 3}.get(lvl, 1)

    def emit(self, line: str, level: str = "normal", also_print: bool = True) -> None:
        if self._level_rank(level) <= self._level_rank(self.log_level):
            safe = _ascii_safe_log_line(line)
            self.ring.append(safe)
            self.out.append_timeline(safe, flush=False)
            if also_print:
                print(safe)

    def flush(self) -> None:
        # flush file
        try:
            if self.out._timeline_fp:
                self.out._timeline_fp.flush()
        except Exception:
            pass

    def last_lines(self) -> List[str]:
        return list(self.ring)

# =============================================================================
# Safe league/team accessors
# =============================================================================

def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default

def _get_league_teams(league: Any) -> List[Any]:
    """
    Do not assume storage shape. Support:
    - league.teams (list or dict)
    - league.get_teams()
    - league.team_map (dict)
    - league.franchises / league.clubs (rare)
    """
    if league is None:
        return []
    # 1) teams attribute
    t = _safe_getattr(league, "teams", None)
    if isinstance(t, list):
        return t
    if isinstance(t, dict):
        return list(t.values())

    # 2) get_teams method
    gt = _safe_getattr(league, "get_teams", None)
    if callable(gt):
        try:
            val = gt()
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return list(val.values())
        except Exception:
            pass

    # 3) team_map
    tm = _safe_getattr(league, "team_map", None)
    if isinstance(tm, dict):
        return list(tm.values())

    # 4) franchises/clubs
    fr = _safe_getattr(league, "franchises", None)
    if isinstance(fr, list):
        return fr
    cl = _safe_getattr(league, "clubs", None)
    if isinstance(cl, list):
        return cl

    return []

def _team_id(team: Any) -> str:
    for attr in ("team_id", "id", "abbr", "code", "name"):
        v = _safe_getattr(team, attr, None)
        if isinstance(v, str) and v:
            return v
    # fallback stable-ish
    return f"TEAM_{abs(id(team)) % 10000:04d}"

def _team_name(team: Any) -> str:
    for attr in ("name", "display_name", "full_name", "team_name"):
        v = _safe_getattr(team, attr, None)
        if isinstance(v, str) and v:
            return v
    return _team_id(team)

def _team_expected_win_pct(team: Any) -> float:
    """
    Try to call team._expected_win_pct() or team.expected_win_pct or similar.
    Fall back to 0.50.
    """
    m = _safe_getattr(team, "_expected_win_pct", None)
    if callable(m):
        try:
            v = float(m())
            return clamp(v, 0.25, 0.75)
        except Exception:
            pass
    m2 = _safe_getattr(team, "expected_win_pct", None)
    if callable(m2):
        try:
            v = float(m2())
            return clamp(v, 0.25, 0.75)
        except Exception:
            pass
    v = _safe_getattr(team, "expected_win_pct", None)
    if isinstance(v, (int, float)):
        return clamp(float(v), 0.25, 0.75)
    return 0.50

# =============================================================================
# Scenario injection
# =============================================================================

def load_scenario_file(path: Optional[str]) -> ScenarioConfig:
    if not path:
        return ScenarioConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    events_raw = data.get("events", data if isinstance(data, list) else [])
    events: List[ScenarioEvent] = []
    if isinstance(events_raw, list):
        for e in events_raw:
            if not isinstance(e, dict):
                continue
            events.append(
                ScenarioEvent(
                    type=str(e.get("type", "NOTE")).upper(),
                    year=int(e.get("year", 0)),
                    target_team_id=e.get("target_team_id"),
                    payload=dict(e.get("payload", {})) if isinstance(e.get("payload", {}), dict) else {},
                )
            )
    return ScenarioConfig(events=events)

def scenario_events_for_year(scn: ScenarioConfig, year: int) -> List[ScenarioEvent]:
    return [e for e in scn.events if e.year == year]

# =============================================================================
# Universe simulation state (economics, era, indices, waiver priority)
# =============================================================================

@dataclass
class UniverseState:
    salary_cap_m: float
    cap_growth_rate: float
    inflation_factor: float
    league_health: float
    parity_index: float
    chaos_index: float
    active_era: str

    waiver_priority: List[str] = field(default_factory=list)  # team_ids, best priority first

    # team identity drift (runner-side cache)
    team_archetypes: Dict[str, str] = field(default_factory=dict)
    coach_ids: Dict[str, str] = field(default_factory=dict)
    coach_security: Dict[str, float] = field(default_factory=dict)

    # era transition: years in current era, last shift year (for 6–15 year duration)
    years_in_era: int = field(default_factory=lambda: 0)
    last_era_shift_year: Optional[int] = None

    # dynasty / power structure (team_id -> state string)
    power_states: Dict[str, str] = field(default_factory=dict)
    cup_wins_by_team: Dict[str, int] = field(default_factory=dict)
    finals_appearances_by_team: Dict[str, int] = field(default_factory=dict)

    # regression/debug
    rng_traces: List[Dict[str, Any]] = field(default_factory=list)

    # last universe-year tuning diagnostics (for timeline log)
    tuning_report: Dict[str, Any] = field(default_factory=dict)

    # team identity → simulation carry (long arcs; not cosmetic)
    identity_carry_momentum: Dict[str, float] = field(default_factory=dict)
    identity_prev_points: Dict[str, int] = field(default_factory=dict)
    identity_last_year_mod: Dict[str, float] = field(default_factory=dict)
    # seasons spent in current team_archetypes[tid] (for inertia / anti-flip)
    identity_archetype_seasons: Dict[str, int] = field(default_factory=dict)
    # consecutive years: made playoffs but lost early as aggressive identity
    identity_early_exit_streak: Dict[str, int] = field(default_factory=dict)

    # salary cap snapshot (runner-computed; refreshed each universe year after economics)
    runner_cap_pressure_by_team: Dict[str, str] = field(default_factory=dict)
    runner_cap_usage_pct_by_team: Dict[str, float] = field(default_factory=dict)
    # contract demand multiplier from engine econ (used by attractiveness / FA logic)
    runner_contract_inflation: float = 1.0

# =============================================================================
# Universe event generation helpers
# =============================================================================

def _trace_roll(state: UniverseState, cfg: UniverseConfig, label: str, r: random.Random, value: float) -> None:
    if cfg.dump_rng_traces:
        state.rng_traces.append({"label": label, "value": value})

def _choose_era(r: random.Random) -> str:
    return r.choice(list(DEFAULT_ERAS))


# =============================================================================
# League retirement (universe layer: age, decline, wear, role -> retire)
# =============================================================================

def _player_ovr_retirement(player: Any) -> float:
    try:
        if callable(getattr(player, "ovr", None)):
            return float(player.ovr())
        return float(_safe_getattr(player, "ovr", 0.5))
    except Exception:
        return 0.5


def _retirement_probability(
    player: Any,
    rng: random.Random,
    dist_ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """
    Base retirement chance by age band + OVR (stars last longer).
    Bounded wear/morale/vol so random stacks cannot explode counts (global limiter does the rest).
    Optional dist_ctx: mild demographic nudge only (capped).
    """
    dist_ctx = dist_ctx or {}
    pct_v = float(dist_ctx.get("pct_v30", 25.0))
    age = int(_runner_roster_player_age(player) or _safe_getattr(player, "age", 27))
    ovr = _player_ovr_retirement(player)
    wear = float(_safe_getattr(getattr(player, "health", None), "wear_and_tear", 0.3) or 0.3)
    morale = float(_safe_getattr(getattr(player, "psych", None), "morale", 0.5) or 0.5)
    vol = float(_safe_getattr(getattr(player, "traits", None), "volatility", 0.5) or 0.5)

    if age < 30:
        if ovr < 0.46 and age >= 28:
            return (min(0.012, 0.004 + 0.002 * (age - 28)), "fringe_non_caliber")
        return (0.0, "too_young")

    reason = "age"
    if age >= 40:
        base = 0.48 + min(0.22, (age - 40) * 0.06)
    elif age >= 38:
        base = 0.38
    elif age >= 36:
        base = 0.28
    elif age >= 35:
        base = 0.18
    elif age >= 33:
        base = 0.10
    elif age >= 32:
        base = 0.035
    else:
        base = 0.012

    if ovr < 0.56:
        base += min(0.12, 0.06 + (0.58 - ovr) * 0.22)
        reason = "depth_decline"
    elif ovr < 0.68 and age >= 34:
        base += 0.05
        reason = "role_player_decline"

    if ovr >= 0.91:
        base *= 0.35
        reason = "elite_extend"
    elif ovr >= 0.86 and age < 37:
        base *= 0.50
        reason = "star_extend"
    elif ovr >= 0.80:
        base *= 0.78

    base += min(0.07, wear * 0.10)
    base += min(0.04, (1.0 - morale) * 0.05)
    base += vol * 0.02

    if pct_v > 30.0 and age >= 33:
        base *= 1.0 + min(0.06, (pct_v - 30.0) * 0.006)
        if ovr < 0.60:
            base *= 1.03
    elif pct_v < 20.0 and age >= 33:
        base *= 0.94

    base = clamp(base, 0.0, 0.68)
    return (base, reason)


def _run_league_retirements(
    teams: List[Any],
    league: Any,
    rng: random.Random,
    year: int,
    events: List[UniverseEvent],
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Controlled retirement: target ~25–45, soft cap 60, hard cap 70.
    Global probability scaling + priority trim + star protection; min floor without blowing caps.
    """
    RET_SOFT_CAP = 60
    RET_HARD_CAP = 70
    RET_MIN_FLOOR = 25

    guard = getattr(league, "_retirement_population_guard", None) or {}
    pct_u_guard = float(guard.get("pct_u24", 100.0))
    fill_guard = float(guard.get("roster_fill_ratio", 1.0))
    pop_stress = pct_u_guard < 13.5 or fill_guard < 0.88
    if pop_stress:
        RET_MIN_FLOOR = 16
        RET_SOFT_CAP = min(RET_SOFT_CAP, 48)
        RET_HARD_CAP = min(RET_HARD_CAP, 56)

    retired_list: List[Dict[str, Any]] = []
    if not hasattr(league, "retired_players") or league.retired_players is None:
        league.retired_players = []

    new_retirements_year = 0

    def _emit_retirement(player: Any, tid: str, reason: str) -> None:
        nonlocal new_retirements_year
        pname = getattr(player, "name", getattr(getattr(player, "identity", None), "name", "Unknown"))
        age = int(_runner_roster_player_age(player) or _safe_getattr(player, "age", 0))
        ovr = _player_ovr_retirement(player)
        retired_list.append({"name": pname, "age": age, "team_id": tid, "ovr": ovr, "reason": reason})
        events.append(
            UniverseEvent(
                event_id=str(uuid.uuid4()),
                year=year,
                day=None,
                type="RETIREMENT",
                teams=[tid],
                headline=f"RETIREMENT: {pname} retires at age {age}",
                details={"player": pname, "age": age, "team_id": tid, "reason": reason, "ovr": ovr},
                impact_score=0.40 if ovr >= 0.82 else 0.25,
                tags=["retirement"],
            )
        )
        new_retirements_year += 1

    def _remove_from_league(player: Any, roster: List[Any], tid: str) -> None:
        if player in roster:
            roster.remove(player)
        if hasattr(league, "players") and league.players is not None and player in league.players:
            try:
                league.players.remove(player)
            except ValueError:
                pass

    dist_ctx = getattr(league, "_age_balance_retirement_ctx", None)

    carried = 0
    for team in teams:
        for p in getattr(team, "roster", None) or []:
            if getattr(p, "retired", False):
                carried += 1
    if carried > RET_HARD_CAP:
        flagged_pool: List[Tuple[Any, Any, str, int, float]] = []
        for team in teams:
            tid = _team_id(team)
            roster = list(getattr(team, "roster", None) or [])
            for p in roster:
                if getattr(p, "retired", False):
                    age = int(_runner_roster_player_age(p) or _safe_getattr(p, "age", 0))
                    ovr = _player_ovr_retirement(p)
                    flagged_pool.append((p, team, tid, age, ovr))
        flagged_pool.sort(key=lambda x: (-x[3], x[4]))
        for p, _, _, _, _ in flagged_pool[RET_HARD_CAP:]:
            try:
                p.retired = False
                p.retirement_reason = None
            except Exception:
                pass
        carried = sum(
            1
            for tm in teams
            for pl in getattr(tm, "roster", None) or []
            if getattr(pl, "retired", False)
        )

    universe_soft = max(0, RET_SOFT_CAP - carried)
    universe_hard = max(0, RET_HARD_CAP - carried)

    # --- carry-over: already flagged retired ---
    for team in teams:
        roster = list(getattr(team, "roster", None) or [])
        tid = _team_id(team)
        to_remove: List[Any] = []
        for player in roster:
            if getattr(player, "retired", False):
                league.retired_players.append(player)
                pname = getattr(player, "name", getattr(getattr(player, "identity", None), "name", "Unknown"))
                age = int(_runner_roster_player_age(player) or _safe_getattr(player, "age", 0))
                ovr = _player_ovr_retirement(player)
                reason = getattr(player, "retirement_reason", "progression")
                retired_list.append({"name": pname, "age": age, "team_id": tid, "ovr": ovr, "reason": reason})
                events.append(
                    UniverseEvent(
                        event_id=str(uuid.uuid4()),
                        year=year,
                        day=None,
                        type="RETIREMENT",
                        teams=[tid],
                        headline=f"RETIREMENT: {pname} retires at age {age}",
                        details={"player": pname, "age": age, "team_id": tid, "reason": reason, "ovr": ovr},
                        impact_score=0.40 if ovr >= 0.82 else 0.25,
                        tags=["retirement"],
                    )
                )
                to_remove.append(player)
        for p in to_remove:
            _remove_from_league(p, roster, tid)
        team.roster = roster

    # --- elite cutoff (~top 10% OVR league-wide) for star protection ---
    ov_sample: List[float] = []
    for team in teams:
        for player in getattr(team, "roster", None) or []:
            if getattr(player, "retired", False):
                continue
            ov_sample.append(_player_ovr_retirement(player))
    ov_sample.sort()
    elite_cut = 0.88
    if len(ov_sample) >= 8:
        elite_cut = ov_sample[max(0, int(len(ov_sample) * 0.90) - 1)]

    cand: List[Dict[str, Any]] = []
    for team in teams:
        tid = _team_id(team)
        for player in getattr(team, "roster", None) or []:
            if getattr(player, "retired", False):
                continue
            age = int(_runner_roster_player_age(player) or _safe_getattr(player, "age", 0))
            ovr = _player_ovr_retirement(player)
            prob, reason = _retirement_probability(player, rng, dist_ctx)
            wear = float(_safe_getattr(getattr(player, "health", None), "wear_and_tear", 0.3) or 0.3)
            if ovr >= elite_cut and age < 35:
                if wear > 0.82 and rng.random() < 0.022:
                    pass
                else:
                    prob = 0.0
            prob = float(clamp(prob, 0.0, 0.68))
            cand.append(
                {
                    "player": player,
                    "tid": tid,
                    "age": age,
                    "ovr": ovr,
                    "prob": prob,
                    "reason": reason,
                }
            )

    if pop_stress and cand:
        for c in cand:
            c["prob"] = float(clamp(float(c["prob"]) * 0.52, 0.0, 0.62))
        try:
            if logger is not None:
                logger.emit(
                    "RETIREMENT GUARD: youth/roster stress — scaled retirement probability down "
                    f"(pct_u24~{pct_u_guard:.1f} fill~{fill_guard:.2f})",
                    "normal",
                )
        except Exception:
            pass

    projected_raw = sum(float(c["prob"]) for c in cand)
    prob_adjusted = False
    if universe_soft <= 0:
        for c in cand:
            c["prob"] = 0.0
        prob_adjusted = True
    elif projected_raw > float(universe_soft):
        f = float(universe_soft) / projected_raw
        for c in cand:
            c["prob"] = float(clamp(float(c["prob"]) * f, 0.0, 0.65))
        prob_adjusted = True

    s2 = sum(float(c["prob"]) for c in cand)
    if carried + s2 < float(RET_MIN_FLOOR) and universe_soft > 0:
        low = [c for c in cand if c["age"] >= 33 and c["ovr"] < 0.63 and float(c["prob"]) < 0.62]
        if low:
            need = float(RET_MIN_FLOOR) - float(carried) - s2
            bump = min(0.028, max(0.0, need) / max(len(low), 1))
            for c in low:
                c["prob"] = float(clamp(float(c["prob"]) + bump, 0.0, 0.65))
            prob_adjusted = True

    winners: List[Dict[str, Any]] = []
    for c in cand:
        if float(c["prob"]) <= 0.0:
            continue
        if rng.random() < float(c["prob"]):
            winners.append(c)

    raw_win = len(winners)
    reduced_from = raw_win
    if universe_hard > 0 and len(winners) > universe_hard:
        winners.sort(key=lambda x: (-int(x["age"]), float(x["ovr"])))
        winners = winners[:universe_hard]
    if universe_soft > 0 and len(winners) > universe_soft:
        winners.sort(key=lambda x: (-int(x["age"]), float(x["ovr"])))
        winners = winners[:universe_soft]
    elif universe_soft <= 0 and raw_win > 0:
        winners = []
    cap_applied = raw_win > len(winners)

    for c in winners:
        player = c["player"]
        tid = c["tid"]
        reason = str(c["reason"])
        player.retired = True
        if not getattr(player, "retirement_reason", None):
            try:
                player.retirement_reason = reason
            except Exception:
                pass
        league.retired_players.append(player)
        _emit_retirement(player, tid, reason)

    # --- floor: priority pool if league total still below target ---
    total_so_far = carried + new_retirements_year
    if total_so_far < RET_MIN_FLOOR:
        need = RET_MIN_FLOOR - total_so_far
        room = max(0, RET_SOFT_CAP - total_so_far)
        need = min(need, room)
        pool: List[Tuple[Any, str, int, float]] = []
        for team in teams:
            tid = _team_id(team)
            roster = list(getattr(team, "roster", None) or [])
            for player in roster:
                if getattr(player, "retired", False):
                    continue
                age = int(_runner_roster_player_age(player) or _safe_getattr(player, "age", 0))
                if age < 33:
                    continue
                ovr = _player_ovr_retirement(player)
                if ovr >= elite_cut and age < 35:
                    continue
                pool.append((player, tid, age, ovr))
        pool.sort(key=lambda x: (-x[2], x[3]))
        for player, tid, _, _ in pool[:need]:
            if carried + new_retirements_year >= RET_SOFT_CAP:
                break
            if getattr(player, "retired", False):
                continue
            if rng.random() < 0.52:
                player.retired = True
                if not getattr(player, "retirement_reason", None):
                    try:
                        player.retirement_reason = "demographic_floor"
                    except Exception:
                        pass
                league.retired_players.append(player)
                _emit_retirement(player, tid, "demographic_floor")

    final_ct = carried + new_retirements_year
    if logger is not None:
        try:
            logger.emit("RETIREMENT CONTROL:", "normal")
            logger.emit(
                f"  projected={projected_raw:.1f} final={final_ct} adjusted={'yes' if prob_adjusted else 'no'}",
                "normal",
            )
            if cap_applied:
                logger.emit("RETIREMENT CAP APPLIED:", "normal")
                logger.emit(f"  reduced_from={reduced_from} to={len(winners)}", "normal")
        except Exception:
            pass

    # --- remove retired from rosters ---
    for team in teams:
        roster = list(getattr(team, "roster", None) or [])
        tid = _team_id(team)
        to_remove = [p for p in roster if getattr(p, "retired", False)]
        for p in to_remove:
            _remove_from_league(p, roster, tid)
        team.roster = roster

    return retired_list


def _run_player_progression_pass(teams: List[Any], rng: random.Random, logger: Any = None) -> None:
    """
    Run progression pipeline for every roster player (development, potential, regression, role, retirement).
    Sets player.retired and player.retirement_reason when progression retirement check returns True.
    """
    if _run_player_progression is None:
        return
    dev_log: List[str] = []
    for team in teams:
        roster = getattr(team, "roster", None) or []
        for player in roster:
            if getattr(player, "retired", False):
                continue
            try:
                _, retired = _run_player_progression(player, rng)
                if retired:
                    player.retired = True
                    if not getattr(player, "retirement_reason", None):
                        try:
                            player.retirement_reason = "progression"
                        except Exception:
                            pass
                for attr in ("_dev_report_pending_line", "_bust_steal_pending_line"):
                    ln = getattr(player, attr, None)
                    if ln:
                        dev_log.append(str(ln))
                        try:
                            delattr(player, attr)
                        except Exception:
                            try:
                                setattr(player, attr, None)
                            except Exception:
                                pass
            except Exception:
                pass
    if dev_log and logger is not None:
        try:
            logger.emit("ENVIRONMENT / DEVELOPMENT (roster pass excerpts)", "normal")
            for ln in dev_log[:60]:
                logger.emit(f"  {ln}", "normal")
        except Exception:
            pass


def _prime_lifecycle_event_caps(league: Any, state: UniverseState) -> None:
    """Scale special-event volume with chaos (governor); reset counters each offseason pass."""
    if league is None:
        return
    c = float(state.chaos_index)
    try:
        setattr(league, "_lifecycle_cap_breakouts", int(8 + 26 * c))
        setattr(league, "_lifecycle_cap_declines", int(6 + 22 * c))
        setattr(league, "_lifecycle_used_breakouts", 0)
        setattr(league, "_lifecycle_used_declines", 0)
    except Exception:
        pass


def _run_career_lifecycle_pass(
    teams: List[Any],
    rng: random.Random,
    logger: Optional[RunnerLogger],
    league: Any = None,
    state: Optional[UniverseState] = None,
    season_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Once per player per season: run_career_lifecycle_for_player with skip_base_progress=True
    (run_player_progression already ran this year) then resolve_authoritative_major_progression_event
    for at most one major event: breakout, late bloom, bust trend, or aging decline — global caps primed above.
    """
    stats: Dict[str, Any] = {"players_touched": 0, "special_events": 0, "skipped": False}
    if _ENGINE_ECON is None:
        stats["skipped"] = True
        return stats
    verbose_main = bool(logger and getattr(logger, "log_level", "normal") in ("debug", "insane"))

    def _emit(line: str) -> None:
        if logger is not None:
            try:
                logger.emit(line, "normal")
            except Exception:
                pass

    if league is not None and _ENGINE_ECON is not None:
        try:
            _aging_n = 0
            for _tm in teams:
                for _pl in getattr(_tm, "roster", None) or []:
                    if not getattr(_pl, "retired", False):
                        _aging_n += 1
            _ENGINE_ECON.prime_league_season_aging_v3(league, int(_aging_n))
            _ENGINE_ECON.prime_league_season_breakout_v3(
                league, int(_aging_n), season_year=int(season_year) if season_year is not None else None
            )
            _ENGINE_ECON.reset_career_breakout_season_flags(teams)
        except Exception:
            pass

    for team in teams:
        roster = getattr(team, "roster", None) or []
        for player in roster:
            if getattr(player, "retired", False):
                continue
            try:
                lines = _ENGINE_ECON.run_career_lifecycle_for_player(
                    player,
                    rng,
                    do_print=not bool(logger),
                    log_emit=_emit if logger is not None else None,
                    verbose_main_line=verbose_main,
                    league=league,
                    skip_base_progress=True,
                    season_year=season_year,
                )
                stats["players_touched"] += 1
                for ln in lines:
                    if any(k in ln for k in ("BREAKOUT:", "BUST TREND:", "LATE BLOOM:", "AGING DECLINE:")):
                        stats["special_events"] += 1
            except Exception:
                pass
    return stats


def _tuning_ctx_from_state(state: UniverseState) -> Dict[str, Any]:
    return {
        "active_era": state.active_era,
        "chaos_index": state.chaos_index,
        "parity_index": state.parity_index,
        "league_health": state.league_health,
        "salary_cap_m": state.salary_cap_m,
        "team_archetypes": dict(state.team_archetypes or {}),
    }


def _sync_ctx_to_state(state: UniverseState, ctx: Dict[str, Any]) -> None:
    state.chaos_index = float(ctx.get("chaos_index", state.chaos_index))
    state.parity_index = float(ctx.get("parity_index", state.parity_index))
    state.league_health = float(ctx.get("league_health", state.league_health))


def _tuning_start_of_season(league: Any, teams: List[Any], state: UniverseState) -> Dict[str, Any]:
    if _tuning_apply_era_to_league is None:
        return {}
    ctx = _tuning_ctx_from_state(state)
    setattr(league, "_tuning_context", ctx)
    try:
        return _tuning_apply_era_to_league(ctx, league, teams)
    except Exception:
        return {}


def _tuning_after_progression(
    league: Any,
    teams: List[Any],
    state: UniverseState,
    year: int = 0,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if _tuning_probability_tables is None or _tuning_normalization is None or _tuning_validators is None:
        return out
    ctx = _tuning_ctx_from_state(state)
    setattr(league, "_tuning_context", ctx)
    players: List[Any] = []
    role_moves = 0
    for team in teams:
        roster = getattr(team, "roster", None) or []
        for p in roster:
            if getattr(p, "retired", False):
                continue
            players.append(p)
            try:
                old = getattr(p, "role", None)
                _tuning_probability_tables.determine_player_role(p, team, ctx)
                if getattr(p, "role", None) != old:
                    role_moves += 1
            except Exception:
                pass
    try:
        out["normalize_league"] = _tuning_normalization.normalize_league_stats(ctx)
        _sync_ctx_to_state(state, ctx)
    except Exception:
        pass
    try:
        out["normalize_players"] = _tuning_normalization.normalize_player_ratings(players, ctx)
    except Exception:
        pass
    try:
        out["normalize_teams"] = _tuning_normalization.normalize_team_strengths(teams, ctx)
    except Exception:
        pass
    try:
        rfv = rng if rng is not None else random.Random(int(year) + 4049)
        out["validation"] = _tuning_validators.run_full_validation(
            ctx, league, teams, players, rng=rfv, year=int(year)
        )
        _sync_ctx_to_state(state, ctx)
    except Exception:
        pass
    out["role_updates"] = role_moves
    out["chaos_influence"] = float(_tuning_probability_tables.chaos_multiplier(ctx))
    try:
        out["macro_progression_scales"] = _tuning_normalization.macro_progression_scales(ctx)
    except Exception:
        pass
    return out


def _tuning_end_of_year(
    league: Any,
    teams: List[Any],
    state: UniverseState,
    year: int = 0,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    if _tuning_normalization is None or _tuning_validators is None:
        return {}
    ctx = _tuning_ctx_from_state(state)
    setattr(league, "_tuning_context", ctx)
    out: Dict[str, Any] = {}
    r = rng if rng is not None else random.Random((int(year) * 1009 + 733) % (2**31))
    try:
        out["macro_feedback"] = _tuning_normalization.apply_season_feedback_to_league_state(ctx, teams)
        _sync_ctx_to_state(state, ctx)
        out["end_normalize_league"] = _tuning_normalization.normalize_league_stats(ctx)
        _sync_ctx_to_state(state, ctx)
        out["end_validation"] = _tuning_validators.validate_league_state(ctx, league, teams, rng=r, year=int(year))
        _sync_ctx_to_state(state, ctx)
    except Exception:
        pass
    return out


def _advance_roster_ages_and_development(teams: List[Any], league: Any, state: UniverseState, rng: random.Random) -> None:
    """
    Start of offseason: one +1 age tick per player per year + development/decline.
    Player.advance_year() already increments identity.age — do not increment twice.
    """
    team_instability = max(0.05, 1.0 - state.parity_index)
    for team in teams:
        roster = getattr(team, "roster", None) or []
        dev_quality = float(getattr(team, "development_quality", 0.5))
        dev_mod = dev_quality - 0.5
        for player in roster:
            if getattr(player, "retired", False):
                continue
            advance_fn = getattr(player, "advance_year", None)
            try:
                if callable(advance_fn):
                    age = int(_safe_getattr(player, "age", 25))
                    age_damp = clamp(1.0 - max(0.0, (age - 26)) / 10.0, 0.35, 1.0)
                    morale = float(_safe_getattr(getattr(player, "psych", None), "morale", 0.5) or 0.5)
                    injury_risk = float(_safe_getattr(getattr(player, "health", None), "injury_risk_baseline", 0.1) or 0.1)
                    sys_dev = (
                        float(_ENGINE_ECON.team_system_development_modifier(team))
                        if _ENGINE_ECON is not None
                        else 0.0
                    )
                    advance_fn(
                        season_morale=morale,
                        season_injury_risk=injury_risk,
                        team_instability=team_instability,
                        development_modifier=dev_mod * age_damp + sys_dev,
                    )
                else:
                    if hasattr(player, "identity") and hasattr(player.identity, "age"):
                        player.identity.age += 1
            except Exception:
                pass
            if _assign_career_phase_from_age is not None:
                try:
                    _assign_career_phase_from_age(player)
                except Exception:
                    pass


def _sync_playoff_streaks_and_contender_flags(teams: List[Any], standings: List[TeamStanding]) -> None:
    """Update missed playoff counters + contender tag for team identity evolution next season."""
    if not teams or not standings:
        return
    for i, s in enumerate(standings):
        tid = s.team_id
        tm = next((t for t in teams if _team_id(t) == tid), None)
        if tm is None:
            continue
        if i < 16:
            setattr(tm, "missed_playoffs_years", 0)
        else:
            prev = int(getattr(tm, "missed_playoffs_years", 0) or 0)
            setattr(tm, "missed_playoffs_years", prev + 1)
        setattr(tm, "is_contender", str(s.bucket) == "contender")


def _dynamic_team_identity_evolution_pass(
    state: UniverseState,
    teams: List[Any],
    standings: List[TeamStanding],
    league: Any,
    rng: random.Random,
    year: int,
    logger: RunnerLogger,
    champion_id: str,
    tuning_year: Dict[str, Any],
) -> List[str]:
    """
    Post-results franchise direction: react to standings, aging, cap, pipeline, playoff pain.
    Updates state.team_archetypes, team carry attributes, and trade pressure when panicking.
    """
    out: List[str] = []
    if not teams or not standings:
        return out
    sorted_s = sorted(standings, key=lambda x: (x.points, x.goal_diff), reverse=True)
    rank_by_tid = {s.team_id: idx for idx, s in enumerate(sorted_s)}
    stand_by_tid = {s.team_id: s for s in sorted_s}
    n_teams = max(1, len(sorted_s))
    playoff_cut = min(16, max(8, n_teams // 2))

    for team in teams:
        tid = _team_id(team)
        s = stand_by_tid.get(tid)
        if s is None:
            continue
        rank = int(rank_by_tid.get(tid, 99))
        cur = str(_team_archetype(state, tid) or "balanced").lower()
        made_po = rank < playoff_cut
        miss = int(getattr(team, "missed_playoffs_years", 0) or 0)
        cap_p = str((getattr(state, "runner_cap_pressure_by_team", None) or {}).get(tid, "moderate")).lower()
        pipe = 0.48
        avg_age = 27.0
        frac30 = 0.0
        if _ENGINE_ECON is not None:
            try:
                sig = _ENGINE_ECON.runner_team_roster_identity_signals(team)
                pipe = float(sig.get("pipeline_score", 0.48) or 0.48)
                avg_age = float(sig.get("avg_age", 27.0) or 27.0)
                frac30 = float(sig.get("frac_30p", 0.0) or 0.0)
            except Exception:
                pass
        try:
            expw = float(_team_expected_win_pct(team))
        except Exception:
            expw = 0.50
        underperf = float(s.point_pct) < (expw - 0.028)

        ee_prev = int(state.identity_early_exit_streak.get(tid, 0) or 0)
        if made_po and rank >= 8 and cur in ("win_now", "contender") and str(tid) != str(champion_id):
            state.identity_early_exit_streak[tid] = ee_prev + 1
        else:
            state.identity_early_exit_streak[tid] = 0
        ee = int(state.identity_early_exit_streak.get(tid, 0) or 0)

        target = cur
        panic = False
        window_note = ""
        reason = ""

        if cur == "chaos_agent" and rng.random() < 0.36:
            target = str(rng.choice(list(DEFAULT_TEAM_ARCHETYPES)))
            reason = "chaos pivot"
        elif miss >= 3 and cur in ("win_now", "contender", "balanced"):
            target = "rebuild"
            panic = True
            reason = f"missed playoffs {miss} years"
        elif cap_p == "cap_hell" and cur in ("win_now", "contender"):
            target = "rebuild" if rng.random() < 0.55 else "draft_and_develop"
            panic = True
            reason = "cap hell pressure"
        elif miss >= 2 and cur in ("win_now", "contender") and s.bucket in ("rebuild", "bubble"):
            target = "draft_and_develop" if pipe > 0.54 else "rebuild"
            panic = rng.random() < 0.62
            reason = "playoff drought + weak record"
        elif ee >= 2 and cur in ("win_now", "contender"):
            target = "balanced"
            reason = "repeated early playoff exits"
        elif underperf and miss >= 1 and cur == "win_now" and s.bucket not in ("contender",):
            target = "contender"
            reason = "underperformed vs roster strength — easing timeline"
        elif frac30 >= 0.40 and avg_age >= 29.6 and cur == "win_now":
            target = "contender"
            window_note = "window closing (aging core detected)"
            reason = "aging core, win-now cooling"
        elif frac30 >= 0.43 and avg_age >= 30.1 and cur == "contender" and s.bucket != "contender":
            target = "draft_and_develop" if pipe > 0.50 else "balanced"
            window_note = "window decline (slipping performance)"
            reason = "core aging out of true contention"
        elif cap_p in ("critical",) and cur == "win_now" and rng.random() < 0.45:
            target = "contender"
            reason = "cap squeeze — throttle to contender mode"
        elif pipe >= 0.62 and s.bucket in ("rebuild", "bubble") and cur in ("rebuild", "balanced"):
            target = "draft_and_develop"
            reason = "strong prospect pipeline"
        elif pipe <= 0.30 and cur in ("rebuild", "draft_and_develop") and s.bucket in ("bubble", "playoff"):
            target = "balanced" if rng.random() < 0.55 else "contender"
            reason = "thin pipeline — push for NHL talent"
        else:
            elite_cut = max(5, min(8, n_teams // 4))
            if rank < elite_cut and s.bucket == "contender":
                if str(tid) == str(champion_id):
                    target = "win_now"
                    reason = "Cup + elite standing"
                elif cur not in ("rebuild", "chaos_agent"):
                    target = "contender" if cur != "win_now" or rng.random() < 0.35 else "win_now"
                    reason = "elite regular season"
            elif made_po and s.bucket == "playoff" and cur in ("balanced", "draft_and_develop", "rebuild"):
                target = "contender"
                reason = "playoff berth — enter contention lane"
            elif not made_po and rank >= n_teams - 7:
                if cur not in ("chaos_agent",):
                    target = "rebuild" if miss >= 1 or s.bucket == "rebuild" else cur
                    if target == "rebuild":
                        reason = "bottom feed — commit to lottery path"
            elif s.bucket == "bubble":
                if rng.random() < 0.48:
                    target = "balanced"
                    reason = "mid-table — stay flexible"
                else:
                    target = "draft_and_develop" if pipe > 0.52 else "contender"
                    reason = "mid-table — pick a lane"
            elif s.bucket == "rebuild" and not made_po:
                target = "rebuild" if miss >= 1 else "draft_and_develop"
                reason = "rebuild posture reinforced"

        if cur == "chaos_agent" and reason and "chaos" not in reason:
            if rng.random() < 0.28:
                target = cur
                reason = ""

        seasons_here = int(state.identity_archetype_seasons.get(tid, 0) or 0)
        if target != cur and cur != "chaos_agent" and not panic:
            resist = clamp(0.30 + 0.09 * min(seasons_here, 7), 0.22, 0.78)
            if rng.random() < resist:
                target = cur
                reason = ""

        new_arch = str(target).lower()
        if new_arch not in DEFAULT_TEAM_ARCHETYPES:
            new_arch = "balanced"

        try:
            setattr(team, "_runner_team_archetype", new_arch)
            setattr(team, "_runner_window_status", window_note or ("window open" if new_arch in ("win_now", "contender") else "window closed" if new_arch == "rebuild" else "neutral"))
            traj = _identity_trajectory_label(state.identity_prev_points.get(tid), int(s.points))
            if new_arch == "rebuild":
                traj_l = "full rebuild arc" if miss >= 2 else "leaning rebuild"
            elif new_arch == "win_now":
                traj_l = "all-in contender" if traj == "rising" else "win-now pressure"
            elif new_arch == "contender":
                traj_l = f"{traj} contender"
            elif new_arch == "draft_and_develop":
                traj_l = "development track"
            else:
                traj_l = f"{traj} / flexible"
            setattr(team, "_runner_franchise_trajectory", traj_l)
        except Exception:
            pass

        if new_arch != cur:
            state.team_archetypes[tid] = new_arch
            state.identity_archetype_seasons[tid] = 1
            msg = f"IDENTITY SHIFT: {_team_name(team)}: {cur} → {new_arch}"
            if reason:
                msg += f" ({reason})"
            out.append(msg)
            try:
                logger.emit(msg, "normal")
            except Exception:
                pass
            if new_arch == "rebuild" and cur == "win_now":
                slam = f"WINDOW STATUS: {_team_name(team)}: window slammed shut (forced teardown)"
                out.append(slam)
                try:
                    logger.emit(slam, "normal")
                except Exception:
                    pass
        else:
            state.identity_archetype_seasons[tid] = seasons_here + 1

        if window_note:
            wmsg = f"WINDOW STATUS: {_team_name(team)}: {window_note}"
            out.append(wmsg)
            try:
                logger.emit(wmsg, "normal")
            except Exception:
                pass

        if panic:
            pmsg = f"PANIC EVENT: {_team_name(team)}: GM under pressure — aggressive retool initiated"
            out.append(pmsg)
            try:
                logger.emit(pmsg, "normal")
            except Exception:
                pass
            try:
                tm0 = float(getattr(team, "_runner_trade_pressure_mult", 1.0) or 1.0)
                setattr(team, "_runner_trade_pressure_mult", min(2.75, tm0 * 1.38))
                setattr(team, "_runner_panic_sell_bias", 1.0)
            except Exception:
                pass

        if new_arch != cur or window_note or panic:
            tlab = str(getattr(team, "_runner_franchise_trajectory", "") or "")
            if tlab:
                tmsg = f"TRAJECTORY: {_team_name(team)}: Team trajectory: {tlab}"
                out.append(tmsg)
                try:
                    logger.emit(tmsg, "normal")
                except Exception:
                    pass

    if league is not None:
        try:
            setattr(league, "_runner_team_archetypes", dict(state.team_archetypes or {}))
        except Exception:
            pass
    tuning_year["identity_evolution"] = list(out)
    return out


def _runner_update_rivalry_narrative(
    league: Any,
    standings: List[TeamStanding],
    rng: random.Random,
    logger: RunnerLogger,
    year: int,
) -> None:
    if league is None or not standings:
        return
    top = standings[: min(8, len(standings))]
    heat: Dict[str, float] = dict(getattr(league, "_runner_rivalry_heat", None) or {})
    for i, a in enumerate(top):
        for b in top[i + 1 :]:
            if rng.random() >= 0.20:
                continue
            k = "|".join(sorted([str(a.team_id), str(b.team_id)]))
            heat[k] = float(heat.get(k, 0.35)) + rng.uniform(0.03, 0.08)
            try:
                logger.emit(
                    f"RIVALRY HEAT: {a.team_name} vs {b.team_name} (y{year} heat={heat[k]:.2f})",
                    "normal",
                )
            except Exception:
                pass
    try:
        setattr(league, "_runner_rivalry_heat", heat)
    except Exception:
        pass


def _emit_awards_media_hof_org_pack(
    teams: List[Any],
    standings: List[TeamStanding],
    state: UniverseState,
    logger: RunnerLogger,
    year: int,
) -> None:
    if not standings:
        return
    era = str(state.active_era or "modern").replace("_", " ")
    try:
        logger.emit(
            f"MEDIA ERA: {era} climate — system-fit and coaching noise now move macro results.",
            "normal",
        )
    except Exception:
        pass
    lead = standings[0]
    try:
        logger.emit(
            f"AWARDS WATCH: Hart pace headline tied to {lead.team_name} (synthetic macro leader).",
            "normal",
        )
    except Exception:
        pass
    try:
        logger.emit(
            "HALL OF FAME: Induction class forming from veteran tiers + peak résumés (watch list).",
            "normal",
        )
    except Exception:
        pass
    for t in (teams or [])[:10]:
        try:
            mkt = getattr(getattr(t, "market", None), "market_size", "medium")
            mp = float(getattr(getattr(t, "market", None), "media_pressure", 0.5) or 0.5)
            own = getattr(t, "ownership", None)
            amb = float(getattr(own, "ambition", 0.5) or 0.5) if own else 0.5
            pat = float(getattr(own, "patience", 0.5) or 0.5) if own else 0.5
            logger.emit(
                f"ORG & FAN PULSE: {_team_name(t)} market={mkt} media={mp:.2f} "
                f"owner_ambition={amb:.2f} owner_patience={pat:.2f}",
                "normal",
            )
        except Exception:
            pass


def _init_team_identities(teams: List[Any], r: random.Random) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, float]]:
    archetypes: Dict[str, str] = {}
    coach_ids: Dict[str, str] = {}
    coach_sec: Dict[str, float] = {}
    for t in teams:
        tid = _team_id(t)
        archetypes[tid] = r.choice(list(DEFAULT_TEAM_ARCHETYPES))
        coach_ids[tid] = _safe_getattr(t, "coach_id", None) or f"Coach_{abs(split_seed(12345, tid)) % 1000:03d}"
        coach_sec[tid] = float(clamp(r.random() * 0.35 + 0.45, 0.05, 0.95))
    return archetypes, coach_ids, coach_sec

def _economics_advance(state: UniverseState, ucfg: UniverseConfig, r: random.Random, year: int,
                      injected: List[ScenarioEvent]) -> Tuple[UniverseState, List[UniverseEvent]]:
    events: List[UniverseEvent] = []

    # base growth with volatility
    growth = ucfg.cap_growth_rate_mean + r.uniform(-ucfg.cap_growth_volatility, ucfg.cap_growth_volatility)
    growth = clamp(growth, -0.08, 0.12)

    # scenario: forced cap shock
    forced_shock = [e for e in injected if e.type.upper() in ("FORCE_SALARY_CAP_CRASH", "CAP_SHOCK")]
    if forced_shock:
        mag = float(forced_shock[0].payload.get("magnitude", -0.10))
        growth = mag
        events.append(
            UniverseEvent(
                event_id=str(uuid.uuid4()),
                year=year,
                day=None,
                type="CAP_SHOCK",
                teams=[],
                headline=f"CAP SHOCK: League salary cap {'drops' if mag < 0 else 'spikes'} ({mag:+.1%})",
                details={"magnitude": mag, "reason": forced_shock[0].payload.get("reason", "scenario_injection")},
                impact_score=0.95,
                tags=["economy", "scenario"],
            )
        )

    # random revenue shock
    roll = r.random()
    _trace_roll(state, ucfg, "econ_shock_roll", r, roll)
    if not forced_shock and roll < ucfg.revenue_shock_frequency:
        # choose shock type
        shock_type = r.choice(["cap_spike", "cap_dip", "revenue_boom", "revenue_bust"])
        if shock_type in ("cap_spike", "revenue_boom"):
            mag = r.uniform(0.03, 0.10)
        else:
            mag = r.uniform(-0.12, -0.03)

        growth = mag
        headline = "CAP SPIKE" if mag > 0 else "CAP DIP"
        events.append(
            UniverseEvent(
                event_id=str(uuid.uuid4()),
                year=year,
                day=None,
                type="CAP_SHOCK",
                teams=[],
                headline=f"{headline}: Salary cap shock hits the league ({mag:+.1%})",
                details={"magnitude": mag, "shock_type": shock_type},
                impact_score=0.85,
                tags=["economy", shock_type],
            )
        )

    # apply growth
    new_cap = max(40.0, state.salary_cap_m * (1.0 + growth))

    # inflation (simple)
    inflation = 1.0 + clamp(ucfg.contract_demand_inflation + r.uniform(-0.01, 0.02), -0.05, 0.10)

    # league health drift toward target with noise
    health = clamp(state.league_health + (ucfg.league_health_target - state.league_health) * 0.25 + r.uniform(-0.03, 0.03), 0.10, 0.95)

    # parity / chaos drift with more variation (not stuck at 0.47-0.55)
    parity_drift = (ucfg.parity_target - state.parity_index) * 0.15 + r.uniform(-0.06, 0.06)
    parity = clamp(state.parity_index + parity_drift, 0.12, 0.88)
    chaos_drift = (ucfg.chaos - state.chaos_index) * 0.15 + r.uniform(-0.05, 0.05)
    chaos = clamp(state.chaos_index + chaos_drift, 0.10, 0.95)

    infl_proxy = state
    try:
        infl_proxy = dataclasses.replace(state, cap_growth_rate=growth, chaos_index=chaos)
    except Exception:
        pass
    runner_infl = 1.0
    if _ENGINE_ECON is not None:
        try:
            runner_infl = float(_ENGINE_ECON.calculate_contract_inflation(infl_proxy))
        except Exception:
            runner_infl = 1.0

    new_state = dataclasses.replace(
        state,
        salary_cap_m=new_cap,
        cap_growth_rate=growth,
        inflation_factor=inflation,
        league_health=health,
        parity_index=parity,
        chaos_index=chaos,
        runner_contract_inflation=runner_infl,
    )
    return new_state, events

def _maybe_era_shift(state: UniverseState, ucfg: UniverseConfig, r: random.Random, year: int,
                    injected: List[ScenarioEvent]) -> Tuple[UniverseState, List[UniverseEvent]]:
    events: List[UniverseEvent] = []
    years_in_era = getattr(state, "years_in_era", 0) + 1
    last_shift = getattr(state, "last_era_shift_year", None)
    state = dataclasses.replace(state, years_in_era=years_in_era)

    forced = [e for e in injected if e.type.upper() in ("FORCE_ERA_SHIFT", "ERA_SHIFT")]
    if forced:
        era = str(forced[0].payload.get("era", state.active_era))
        if era != state.active_era:
            events.append(
                UniverseEvent(
                    event_id=str(uuid.uuid4()),
                    year=year,
                    day=None,
                    type="ERA_SHIFT",
                    teams=[],
                    headline=f"ERA SHIFT (forced): League meta becomes '{era}'",
                    details={"from": state.active_era, "to": era, "reason": forced[0].payload.get("reason", "scenario_injection")},
                    impact_score=0.75,
                    tags=["era", "scenario"],
                )
            )
            state = dataclasses.replace(state, active_era=era, years_in_era=0, last_era_shift_year=year)
        return state, events

    # Organic shift: allow from year 3; cap era at ~5 years (force turnover / avoid stagnation)
    if years_in_era < 3:
        return state, events
    force = years_in_era >= 5
    transition_chance = 0.07 + (years_in_era - 3) * 0.065
    transition_chance = clamp(transition_chance, 0.07, 0.42)
    roll = r.random()
    _trace_roll(state, ucfg, "era_shift_roll", r, roll)
    if force or roll < transition_chance:
        era = _choose_era(r)
        if era != state.active_era:
            reason = "max_era_duration" if force else "trend_change"
            events.append(
                UniverseEvent(
                    event_id=str(uuid.uuid4()),
                    year=year,
                    day=None,
                    type="ERA_SHIFT",
                    teams=[],
                    headline=f"ERA SHIFT: from={state.active_era} to={era} reason={reason}",
                    details={"from": state.active_era, "to": era, "years_in_previous_era": years_in_era, "reason": reason},
                    impact_score=0.70,
                    tags=["era"],
                )
            )
            state = dataclasses.replace(state, active_era=era, years_in_era=0, last_era_shift_year=year)
    return state, events

def _team_identity_drift(state: UniverseState, ucfg: UniverseConfig, r: random.Random, year: int,
                         teams: List[Any]) -> List[UniverseEvent]:
    events: List[UniverseEvent] = []
    for t in teams:
        tid = _team_id(t)
        cur = state.team_archetypes.get(tid, "balanced")

        roll = r.random()
        _trace_roll(state, ucfg, f"archetype_drift_roll::{tid}", r, roll)
        if roll < ucfg.archetype_drift_rate:
            # Weighted drift (rare; performance-driven pass is primary)
            # Order: draft_and_develop, win_now, contender, balanced, rebuild, chaos_agent
            candidates = list(DEFAULT_TEAM_ARCHETYPES)
            if cur == "rebuild":
                weights = [0.32, 0.08, 0.06, 0.22, 0.24, 0.08]
            elif cur == "win_now":
                weights = [0.10, 0.22, 0.20, 0.26, 0.10, 0.12]
            elif cur == "contender":
                weights = [0.12, 0.16, 0.24, 0.26, 0.10, 0.12]
            else:
                weights = [0.20, 0.12, 0.16, 0.26, 0.14, 0.12]
            new = r.choices(candidates, weights=weights, k=1)[0]
            if new != cur:
                state.team_archetypes[tid] = new
                events.append(
                    UniverseEvent(
                        event_id=str(uuid.uuid4()),
                        year=year,
                        day=None,
                        type="NOTE",
                        teams=[tid],
                        headline=f"Identity drift: {_team_name(t)} pivots from '{cur}' to '{new}'",
                        details={"team_id": tid, "from": cur, "to": new},
                        impact_score=0.20,
                        tags=["team_identity"],
                    )
                )
    return events

# =============================================================================
# Standings approximation (macro layer)
# =============================================================================

@dataclass
class TeamStanding:
    team_id: str
    team_name: str
    expected_win_pct: float
    point_pct: float
    points: int
    goal_diff: int
    bucket: str  # contender/playoff/bubble/rebuild

def _bucket_from_point_pct(p: float) -> str:
    if p > 0.58:
        return "contender"
    if p >= 0.52:
        return "playoff"
    if p >= 0.47:
        return "bubble"
    return "rebuild"

def _simulate_standings(
    teams: List[Any],
    state: UniverseState,
    ucfg: UniverseConfig,
    r: random.Random,
    league: Optional[Any] = None,
) -> List[TeamStanding]:
    standings: List[TeamStanding] = []
    # Convert chaos/parity to noise
    # More parity => compress spread; more chaos => more randomness
    base_noise = 0.06 * (0.5 + state.chaos_index)  # ~0.06-0.09
    if state.parity_index > 0.52:
        base_noise *= max(0.72, 1.0 - 0.20 * (state.parity_index - 0.52) / 0.48)
    parity_compress = 0.35 + 0.65 * state.parity_index  # 0.35-1.0
    if state.league_health < 0.44:
        parity_compress = min(1.0, parity_compress + 0.06 * (0.44 - state.league_health))
    era_m = float(getattr(league, "_era_scoring_multiplier", 1.0) or 1.0) if league is not None else 1.0

    state.identity_last_year_mod.clear()

    for t in teams:
        tid = _team_id(t)
        name = _team_name(t)
        exp = _team_expected_win_pct(t)

        # compress expectation toward 0.5 with parity
        exp_adj = 0.50 + (exp - 0.50) * (1.0 - 0.55 * parity_compress)

        id_exp_d, noise_id_mult, sm_log = _identity_exp_and_noise_mods(state, tid, t, r)
        exp_adj = clamp(exp_adj + id_exp_d, 0.28, 0.72)
        if _ENGINE_ECON is not None:
            try:
                exp_adj = clamp(
                    exp_adj + float(_ENGINE_ECON.team_identity_win_pct_nudge(t, state.active_era)),
                    0.28,
                    0.72,
                )
            except Exception:
                pass
        state.identity_last_year_mod[tid] = sm_log

        # noise (identity volatility: win_now steadier, rebuild/chaos swingier)
        noise = r.gauss(0.0, base_noise * noise_id_mult)
        pct = clamp(exp_adj + noise, 0.30, 0.70)
        parity_pull = 0.09 * float(state.parity_index)
        health_mix = 0.035 * max(0.0, float(state.league_health) - 0.50)
        toward_half = min(0.20, parity_pull + health_mix)
        pct = 0.50 + (pct - 0.50) * (1.0 - toward_half)
        pct = clamp(pct, 0.30, 0.70)

        # 82-game-ish points approximation: points ~ 164 * point_pct (era nudges offense environment)
        pct_era = clamp(0.50 + (pct - 0.50) * (0.92 + 0.08 * era_m), 0.30, 0.70)
        points = int(round(164 * pct_era))

        # goal diff rough: scale by deviation from 0.5; era scoring multiplier widens/tightens goal diff
        gd = int(round((pct_era - 0.50) * 240 * era_m + r.gauss(0, 15 + 5 * era_m)))

        standings.append(
            TeamStanding(
                team_id=tid,
                team_name=name,
                expected_win_pct=exp,
                point_pct=pct_era,
                points=points,
                goal_diff=gd,
                bucket=_bucket_from_point_pct(pct_era),
            )
        )

    standings.sort(key=lambda s: (s.points, s.goal_diff), reverse=True)
    return standings


def _sync_team_gm_strategic_profiles(
    teams: List[Any],
    standings: List[TeamStanding],
    state: UniverseState,
    rng: random.Random,
) -> None:
    if _update_team_gm_strategic_profile is None or not teams or not standings:
        return
    stand_map = {s.team_id: s for s in standings}
    pmap = getattr(state, "runner_cap_pressure_by_team", None) or {}
    for t in teams:
        tid = _team_id(t)
        st = stand_map.get(tid)
        if st is None:
            continue
        arch = str(_team_archetype(state, tid) or "")
        pres = str(pmap.get(tid, "moderate")).lower()
        pipe = float(getattr(t, "prospect_pipeline_score", 0.5) or 0.5)
        _update_team_gm_strategic_profile(
            t,
            runner_archetype=arch,
            point_pct=float(st.point_pct),
            standings_bucket=st.bucket,
            pipeline_score=pipe,
            cap_pressure=pres,
            rng=rng,
        )


def _resolve_playoff_champion(
    standings: List[TeamStanding],
    r: random.Random,
    state: Optional[UniverseState] = None,
) -> Tuple[str, str]:
    """Lightweight playoff: top 4 by points; weighted random for champion/runner-up (bias toward #1)."""
    if len(standings) < 2:
        return (standings[0].team_id, standings[0].team_id) if standings else ("", "")
    top4 = standings[:4]
    weights = [0.45, 0.28, 0.17, 0.10][:len(top4)]
    if state is not None:
        adj: List[float] = []
        for i, s in enumerate(top4):
            w = weights[i]
            a0 = _team_archetype(state, s.team_id).lower()
            if a0 == "win_now":
                w *= 1.12
            elif a0 == "contender":
                w *= 1.06
            if a0 == "rebuild":
                w *= 0.88
            adj.append(w)
        ssum = sum(adj) or 1.0
        weights = [x / ssum for x in adj]
    champion = r.choices(top4, weights=weights, k=1)[0]
    rest = [s for s in top4 if s.team_id != champion.team_id]
    if not rest:
        return champion.team_id, champion.team_id
    runner_weights = [0.40, 0.35, 0.25][:len(rest)]
    runner_up = r.choices(rest, weights=runner_weights, k=1)[0]
    return champion.team_id, runner_up.team_id

def _update_power_states(standings: List[TeamStanding], state: UniverseState,
                        champion_id: str, runner_up_id: str) -> Dict[str, str]:
    """Derive power state labels from standings and cup/finals history."""
    cup_wins = getattr(state, "cup_wins_by_team", None) or {}
    finals = getattr(state, "finals_appearances_by_team", None) or {}
    out: Dict[str, str] = {}
    for s in standings:
        tid = s.team_id
        cups = cup_wins.get(tid, 0)
        fin = finals.get(tid, 0)
        bucket = s.bucket
        if cups >= 2 and fin >= 3:
            out[tid] = "dynasty"
        elif cups >= 1 and bucket == "contender":
            out[tid] = "powerhouse"
        elif bucket == "contender" and (fin >= 1 or cups >= 1):
            out[tid] = "repeat_contender" if fin > 1 else "rising_contender"
        elif bucket == "contender":
            out[tid] = "contender"
        elif bucket == "playoff":
            out[tid] = "fragile_contender" if s.goal_diff < 10 else "playoff_team"
        elif bucket == "bubble":
            out[tid] = "paper_tiger" if s.points >= 90 else "mid_tier"
        elif bucket == "rebuild" and s.points < 75:
            out[tid] = "tank_spiral"
        else:
            out[tid] = "rebuild" if bucket == "rebuild" else "mid_tier"
    return out

# =============================================================================
# Waivers (runner-side priority lifecycle + candidate generation)
# =============================================================================

@dataclass
class WaiverClaim:
    player_label: str
    from_team: str
    to_team: str
    reason: str

def _reset_waiver_priority_from_standings(standings: List[TeamStanding]) -> List[str]:
    # lowest points first get top priority
    rev = list(reversed(standings))
    return [s.team_id for s in rev]

def _player_display_name(player: Any) -> str:
    """Full generated name for reporting (identity.name or name)."""
    return str(
        getattr(getattr(player, "identity", None), "name", None)
        or getattr(player, "name", "Unknown")
    )

def _identity_pick_player_to_waive(
    roster: List[Any],
    team_id: str,
    state: UniverseState,
    r: random.Random,
) -> Optional[Any]:
    active = [p for p in roster if not getattr(p, "retired", False)]
    if not active:
        return None
    arch = _team_archetype(state, team_id).lower()
    weights: List[float] = []
    for p in active:
        try:
            ov = float(p.ovr()) if callable(getattr(p, "ovr", None)) else float(getattr(p, "ovr", 0.5))
        except Exception:
            ov = 0.5
        age = float(_runner_roster_player_age(p) or 26)
        w = 1.0
        if arch == "win_now":
            w = max(0.06, (1.04 - ov) * (1.0 + 0.034 * max(0.0, age - 28.0)))
        elif arch == "contender":
            w = max(0.06, (1.02 - ov) * (1.0 + 0.028 * max(0.0, age - 29.0)))
        elif arch == "rebuild":
            w = max(0.06, (0.52 + ov * 0.48) * (1.0 + 0.028 * max(0.0, 30.0 - age)))
        elif arch == "draft_and_develop":
            w = max(0.06, (1.02 - ov) * (1.0 + 0.022 * max(0.0, age - 29.0)))
        elif arch == "chaos_agent":
            w = r.uniform(0.32, 1.68)
        weights.append(w)
    return r.choices(active, weights=weights, k=1)[0]


def _generate_waiver_candidates(
    teams: List[Any],
    standings: List[TeamStanding],
    state: UniverseState,
    ucfg: UniverseConfig,
    r: random.Random,
    *,
    cap_tight: bool = False,
    logger: Optional[Any] = None,
) -> List[Tuple[str, str]]:
    """
    Returns list of (team_id, player_label) "waived".
    Uses real roster player names when teams/rosters exist; otherwise narrative placeholder.
    """
    waived: List[Tuple[str, str]] = []
    team_by_id = {_team_id(t): t for t in (teams or [])}

    for s in standings:
        base = ucfg.waiver_rate * 0.35
        if s.bucket == "rebuild":
            base += 0.05
        if s.bucket == "bubble":
            base += 0.03
        if s.goal_diff < -20:
            base += 0.03
        if state.salary_cap_m < 85:
            base += 0.02
        if cap_tight:
            base += 0.06
        pres = str((getattr(state, "runner_cap_pressure_by_team", None) or {}).get(s.team_id, "moderate")).lower()
        if pres == "cap_hell":
            base += 0.15
        elif pres == "critical":
            base += 0.095
        elif pres == "high":
            base += 0.048

        if r.random() < base:
            n = 1 + (1 if r.random() < 0.35 else 0) + (1 if r.random() < 0.10 else 0)
            team = team_by_id.get(s.team_id)
            roster = list(getattr(team, "roster", None) or []) if team else []
            # Prefer real player names from roster (depth-ish: sample from roster)
            for _ in range(n):
                if roster:
                    p = _identity_pick_player_to_waive(roster, s.team_id, state, r) or r.choice(roster)
                    if not getattr(p, "retired", False):
                        label = _player_display_name(p)
                    else:
                        role = r.choice(["fringe winger", "7th D", "backup goalie"])
                        label = f"{role.title()}_{abs(split_seed(state.salary_cap_m.__hash__(), s.team_id + role)) % 10000:04d}"
                else:
                    role = r.choice(["fringe winger", "7th D", "backup goalie", "underperforming veteran", "young bubble prospect"])
                    label = f"{role.title()}_{abs(split_seed(state.salary_cap_m.__hash__(), s.team_id + role)) % 10000:04d}"
                if logger is not None and pres in ("cap_hell", "critical") and r.random() < 0.72:
                    logger.emit("CAP ACTION: Player waived due to cap constraints", "normal")
                waived.append((s.team_id, label))
    return waived

def _process_waivers(waiver_priority: List[str], waived: List[Tuple[str, str]],
                    standings: Dict[str, TeamStanding], ucfg: UniverseConfig, r: random.Random) -> Tuple[List[WaiverClaim], List[str]]:
    """
    Simplified waiver claims:
    - Teams near top of priority more likely to claim.
    - Claim if "need" heuristic matches (goal diff, bucket).
    """
    claims: List[WaiverClaim] = []
    priority = list(waiver_priority)

    def needs_team(team_id: str, player_label: str) -> bool:
        s = standings.get(team_id)
        if not s:
            return False
        if "goalie" in player_label.lower() and s.goal_diff < -10:
            return True
        if "7th d" in player_label.lower() and s.goal_diff < -20:
            return True
        if "fringe winger" in player_label.lower() and s.bucket in ("rebuild", "bubble"):
            return True
        if "young" in player_label.lower() and s.bucket == "rebuild":
            return True
        return r.random() < 0.18  # opportunistic

    for from_team, player_label in waived:
        # iterate waiver order
        claimed_by: Optional[str] = None
        for idx, team_id in enumerate(priority[:min(len(priority), 16)]):
            if team_id == from_team:
                continue
            claim_bias = 0.30 - (idx * 0.012)  # top priority more likely
            claim_bias = clamp(claim_bias, 0.05, 0.30)
            if r.random() < claim_bias and needs_team(team_id, player_label):
                claimed_by = team_id
                break

        if claimed_by:
            claims.append(
                WaiverClaim(
                    player_label=player_label,
                    from_team=from_team,
                    to_team=claimed_by,
                    reason="depth add / cap play / upside swing",
                )
            )
            # update priority after claim (if function exists, use it; else move claimant to bottom)
            if update_priority_after_claim is not None:
                try:
                    priority = update_priority_after_claim(priority, claimed_by)  # type: ignore
                except Exception:
                    # fallback
                    if claimed_by in priority:
                        priority.remove(claimed_by)
                        priority.append(claimed_by)
            else:
                if claimed_by in priority:
                    priority.remove(claimed_by)
                    priority.append(claimed_by)

    return claims, priority

# =============================================================================
# Trades (narrative-first with evaluation)
# =============================================================================

@dataclass
class TradeAsset:
    label: str
    value: float
    cap_m: float

def _generate_asset(r: random.Random, kind: str) -> TradeAsset:
    # Simplified asset values
    if kind == "top4d_rental":
        return TradeAsset("Top-4 D rental", r.uniform(0.62, 0.82), r.uniform(3.0, 7.5))
    if kind == "top6f":
        return TradeAsset("Top-6 forward", r.uniform(0.58, 0.80), r.uniform(4.0, 9.5))
    if kind == "middle6w":
        return TradeAsset("Middle-6 winger", r.uniform(0.42, 0.62), r.uniform(2.0, 5.5))
    if kind == "backupg":
        return TradeAsset("Backup goalie", r.uniform(0.28, 0.48), r.uniform(1.0, 3.5))
    if kind == "prospect_high":
        return TradeAsset("Prospect (high ceiling)", r.uniform(0.55, 0.75), 0.95)
    if kind == "prospect_mid":
        return TradeAsset("Prospect (mid)", r.uniform(0.35, 0.55), 0.95)
    if kind == "1st":
        return TradeAsset("1st round pick", r.uniform(0.55, 0.72), 0.0)
    if kind == "2nd":
        return TradeAsset("2nd round pick", r.uniform(0.25, 0.38), 0.0)
    if kind == "cap_dump":
        return TradeAsset("Cap dump contract", r.uniform(-0.20, -0.05), r.uniform(4.0, 9.0))
    return TradeAsset("Depth piece", r.uniform(0.12, 0.28), r.uniform(0.85, 2.2))

def _trade_eval(
    buyer_gain: float,
    seller_gain: float,
    chaos: float,
    r: random.Random,
    *,
    relaxed: bool = False,
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Accept if both gain enough OR chaos allows a lopsided outcome rarely.
    Returns (accepted, fairness, breakdown)
    """
    fairness = 1.0 - abs(buyer_gain - seller_gain)
    fairness = clamp(fairness, 0.0, 1.0)
    b_thr = -0.015 if relaxed else 0.03
    s_thr = -0.015 if relaxed else 0.03
    both_ok = buyer_gain >= b_thr and seller_gain >= s_thr

    if both_ok:
        return True, fairness, {"both_ok": True, "chaos_override": False, "relaxed": relaxed}

    # chaos override
    base_ov = 0.11 if relaxed else 0.04
    override_chance = clamp(base_ov * (0.5 + chaos), 0.03, 0.22)
    roll = r.random()
    if roll < override_chance:
        return True, fairness, {"both_ok": False, "chaos_override": True, "override_roll": roll, "override_chance": override_chance, "relaxed": relaxed}
    return False, fairness, {"both_ok": False, "chaos_override": False, "override_roll": roll, "override_chance": override_chance, "relaxed": relaxed}


MIN_SEASON_TRADES = 10
MAX_SEASON_TRADES = 25


def _league_trade_identity_scale(state: UniverseState, standings: List[TeamStanding]) -> float:
    if not standings:
        return 1.0
    acc = 0.0
    for s in standings:
        a = _team_archetype(state, s.team_id).lower()
        if a == "win_now":
            acc += 1.12
        elif a == "rebuild":
            acc += 1.08
        elif a == "chaos_agent":
            acc += 1.18
        elif a == "draft_and_develop":
            acc += 0.74
        else:
            acc += 1.0
    return acc / len(standings)


def _pick_weighted_trade_team(
    candidates: List[TeamStanding],
    state: UniverseState,
    r: random.Random,
    role: str,
    teams_by_id: Optional[Dict[str, Any]] = None,
) -> Optional[TeamStanding]:
    if not candidates:
        return None
    ws: List[float] = []
    for s in candidates:
        a = _team_archetype(state, s.team_id).lower()
        win = ""
        if teams_by_id:
            to = teams_by_id.get(s.team_id)
            if to is not None:
                win = str(getattr(to, "window", getattr(to, "gm_window", "")) or "").lower()
        if role == "buyer":
            w = 1.0
            if win == "contender":
                w = 2.18
            elif win == "emerging":
                w = 1.28
            elif win == "declining":
                w = 0.92
            elif win == "rebuild":
                w = 0.14
            if a == "win_now":
                w *= 2.05
            elif a == "chaos_agent":
                w *= 1.58
            elif a == "draft_and_develop":
                w *= 0.50
        else:
            w = 1.0
            if win == "rebuild":
                w = 2.05
            elif win == "declining":
                w = 1.72
            elif win == "contender":
                w = 0.20
            elif win == "emerging":
                w = 0.88
            if a == "rebuild":
                w *= 1.88
            elif a == "chaos_agent":
                w *= 1.48
            elif a == "draft_and_develop":
                w *= 0.60
        ws.append(max(0.08, w))
    return r.choices(candidates, weights=ws, k=1)[0]


def _generate_trades(
    standings: List[TeamStanding],
    state: UniverseState,
    ucfg: UniverseConfig,
    r: random.Random,
    year: int,
    injected: List[ScenarioEvent],
    teams: Optional[List[Any]] = None,
) -> Tuple[List[UniverseEvent], Dict[str, Any]]:
    events: List[UniverseEvent] = []
    stats: Dict[str, Any] = {
        "total_trades": 0,
        "forced_trades": 0,
        "deadline_trades": 0,
        "deadline_buyers": 0,
        "deadline_sellers": 0,
    }
    deadline_buyer_ids: Set[str] = set()
    deadline_seller_ids: Set[str] = set()

    forced = [e for e in injected if e.type.upper() == "FORCE_TRADE"]
    if forced:
        fe = forced[0]
        a = fe.payload.get("from_team_id") or fe.payload.get("team_a") or fe.target_team_id
        b = fe.payload.get("to_team_id") or fe.payload.get("team_b")
        headline = str(fe.payload.get("headline", "FORCED TRADE executes."))
        if a and b:
            events.append(
                UniverseEvent(
                    event_id=str(uuid.uuid4()),
                    year=year,
                    day=None,
                    type="TRADE",
                    teams=[str(a), str(b)],
                    headline=f"TRADE (forced): {headline}",
                    details={"payload": fe.payload, "reason": "scenario_injection"},
                    impact_score=0.90,
                    tags=["trade", "scenario"],
                )
            )
        stats["total_trades"] = len(events)
        return events, stats

    if not standings:
        return events, stats

    num_rebuild = sum(1 for s in standings if s.bucket == "rebuild")
    num_contender = sum(1 for s in standings if s.bucket in ("contender", "playoff"))
    cap_tight = float(state.salary_cap_m) < 90.5
    cap_pressure = 1.0 if cap_tight else (0.7 if state.salary_cap_m < 95 else 0.5)
    pmap = getattr(state, "runner_cap_pressure_by_team", None) or {}
    crisis_n = sum(
        1 for _tid, pr in pmap.items() if str(pr).lower() in ("high", "critical", "cap_hell")
    )
    cap_pressure *= 1.0 + 0.12 * min(1.0, crisis_n / max(1, len(standings)))
    trade_pressure = ucfg.trade_rate * (
        0.82 + 0.32 * min(1.0, num_rebuild / 8) + 0.22 * min(1.0, num_contender / 10) + 0.28 * cap_pressure + 0.16 * state.chaos_index
    )
    trade_pressure = clamp(trade_pressure, 0.38, 0.97)
    trade_pressure *= 1.0 + 0.085 * min(1.0, crisis_n / 12.0)
    trade_pressure = clamp(trade_pressure, 0.38, 0.985)

    if _tuning_probability_tables is not None and standings:
        try:
            ctx = _tuning_ctx_from_state(state)

            class _TeamProbProxy:
                __slots__ = ("team_id",)

                def __init__(self, tid: str) -> None:
                    self.team_id = tid

            probs = [
                _tuning_probability_tables.get_trade_probability(_TeamProbProxy(s.team_id), ctx)
                for s in standings
            ]
            if probs:
                mult = (sum(probs) / len(probs)) / 0.5
                trade_pressure = clamp(trade_pressure * mult, 0.35, 0.98)
        except Exception:
            pass

    roll = r.random()
    _trace_roll(state, ucfg, "trade_year_roll", r, roll)

    id_scale = _league_trade_identity_scale(state, standings)
    desired = int(round(clamp(16.0 * id_scale + (3 if cap_tight else 0), float(MIN_SEASON_TRADES), float(MAX_SEASON_TRADES))))
    if roll <= trade_pressure:
        desired = min(MAX_SEASON_TRADES, max(desired, MIN_SEASON_TRADES + r.randint(0, 8)))

    buyers = [s for s in standings if s.bucket in ("contender", "playoff")]
    sellers = [s for s in standings if s.bucket in ("rebuild",)]
    bubbles = [s for s in standings if s.bucket == "bubble"]

    if r.random() < 0.52:
        sellers.extend(r.sample(bubbles, k=min(len(bubbles), 4)))
    if r.random() < 0.38:
        buyers.extend(r.sample(bubbles, k=min(len(bubbles), 3)))

    if not buyers:
        buyers = standings[:14] if standings else []
    if not sellers:
        sellers = list(reversed(standings))[:14] if standings else []
    if not buyers or not sellers:
        stats["total_trades"] = len(events)
        return events, stats

    teams_by_id: Dict[str, Any] = {_team_id(t): t for t in (teams or [])}

    max_attempts = max(220, desired * 14)
    chaos_f = state.chaos_index * ucfg.trade_chaos_multiplier

    casualty_by_tid: Dict[str, Dict[str, Any]] = {}
    hell_ids = {str(k) for k, v in pmap.items() if str(v).lower() == "cap_hell"}
    if teams and _ENGINE_ECON is not None:
        for t in teams:
            try:
                cc = _ENGINE_ECON.cap_casualty_check(t, salary_cap_m=state.salary_cap_m)
            except Exception:
                cc = None
            if cc:
                casualty_by_tid[str(_team_id(t))] = cc
        if casualty_by_tid:
            desired = min(MAX_SEASON_TRADES, desired + min(6, len(casualty_by_tid)))
        if hell_ids:
            desired = min(MAX_SEASON_TRADES, desired + min(8, len(hell_ids) * 2))

    def one_trade_attempt(
        *,
        relaxed: bool,
        deadline_spike: bool,
        cap_story: bool,
        shape: int,
        force_append: bool,
    ) -> bool:
        b_pick = _pick_weighted_trade_team(buyers, state, r, "buyer", teams_by_id) or r.choice(buyers)
        pool_s = [s for s in sellers if s.team_id != b_pick.team_id] or sellers
        s_pick = _pick_weighted_trade_team(pool_s, state, r, "seller", teams_by_id) or r.choice(pool_s)
        buyer, seller = b_pick, s_pick

        b_ent = teams_by_id.get(buyer.team_id) if teams_by_id else None
        s_ent = teams_by_id.get(seller.team_id) if teams_by_id else None
        b_win = str(getattr(b_ent, "window", getattr(b_ent, "gm_window", "")) or "").lower() if b_ent else ""
        s_win = str(getattr(s_ent, "window", getattr(s_ent, "gm_window", "")) or "").lower() if s_ent else ""

        need_def = buyer.goal_diff < -18
        need_off = buyer.points < 94
        force_cap_dump = seller.team_id in casualty_by_tid and r.random() < 0.78
        cap_crunch = (
            cap_story
            or (cap_tight and r.random() < 0.55)
            or force_cap_dump
            or (buyer.team_id in hell_ids or seller.team_id in hell_ids)
        )
        seller_fire_sale = (
            s_win not in ("contender",)
            and (s_win in ("rebuild", "declining") or seller.bucket == "rebuild")
            and r.random() < 0.48
        )
        deadline_style = deadline_spike or (buyer.bucket in ("contender", "playoff") and r.random() < 0.58)
        if b_win == "contender" and (deadline_spike or deadline_style):
            deadline_style = True
        elif b_win == "rebuild" and not cap_crunch:
            deadline_style = False

        if force_cap_dump:
            target_kind = "cap_dump"
            trade_category = "cap dump"
        elif cap_crunch and r.random() < min(0.55, ucfg.cap_dump_rate + 0.22):
            target_kind = "cap_dump"
            trade_category = "cap dump"
        elif seller_fire_sale and r.random() < 0.48:
            target_kind = r.choice(["top6f", "top4d_rental", "middle6w"])
            trade_category = "rebuild fire sale"
        elif deadline_style and need_def:
            target_kind = "top4d_rental"
            trade_category = "deadline rental"
        elif deadline_style and need_off:
            target_kind = "top6f"
            trade_category = "deadline add"
        elif need_def:
            target_kind = "top4d_rental"
            trade_category = "hockey trade"
        elif need_off:
            target_kind = "top6f"
            trade_category = "hockey trade"
        elif r.random() < 0.22:
            target_kind = r.choice(["prospect_high", "prospect_mid"])
            trade_category = "young player swap"
        else:
            target_kind = r.choice(["middle6w", "backupg", "top4d_rental"])
            trade_category = r.choice(["hockey trade", "retool move", "change-of-scenery swap"])

        b_arche = _team_archetype(state, buyer.team_id).lower()
        s_arche = _team_archetype(state, seller.team_id).lower()
        if b_arche == "win_now" and r.random() < 0.46:
            target_kind = r.choice(["top6f", "top4d_rental", "top4d_rental", "top6f", target_kind])
        elif b_arche == "draft_and_develop" and r.random() < 0.28:
            target_kind = r.choice(["prospect_high", "prospect_mid", "prospect_high", target_kind])
        if s_arche == "rebuild" and r.random() < 0.38:
            target_kind = r.choice(["prospect_mid", "prospect_high", "1st", target_kind])

        asset_in = _generate_asset(r, target_kind)
        package: List[TradeAsset] = []

        if shape == 0:
            asset_in = _generate_asset(r, r.choice(["top6f", "middle6w", "top4d_rental"]))
            package = [
                _generate_asset(r, r.choice(["middle6w", "backupg"])),
                _generate_asset(r, r.choice(["middle6w", "prospect_mid"])),
            ]
            trade_category = "player-for-player swap"
        elif shape == 1:
            asset_in = _generate_asset(r, r.choice(["top6f", "top4d_rental"]))
            package = [_generate_asset(r, "1st"), _generate_asset(r, "2nd"), _generate_asset(r, "prospect_mid")]
            trade_category = "picks-for-player package"
        elif shape == 2:
            package = [_generate_asset(r, "1st"), _generate_asset(r, "prospect_high")]
            if r.random() < 0.55:
                package.append(_generate_asset(r, "middle6w"))
            if not trade_category.startswith("deadline"):
                trade_category = "multi-asset hockey trade"
        else:
            if asset_in.value > 0.70 or r.random() < ucfg.blockbuster_rate + 0.06:
                package.append(_generate_asset(r, "1st"))
                package.append(_generate_asset(r, "prospect_high"))
            elif asset_in.value > 0.55:
                package.append(_generate_asset(r, "1st") if r.random() < 0.55 else _generate_asset(r, "2nd"))
                package.append(_generate_asset(r, "prospect_mid"))
            else:
                package.append(_generate_asset(r, "2nd"))
                if r.random() < 0.38:
                    package.append(_generate_asset(r, "prospect_mid"))
            if r.random() < 0.26:
                package.append(_generate_asset(r, r.choice(["middle6w", "2nd", "prospect_mid", "backupg"])))
            if r.random() < 0.18 and trade_category in ("deadline rental", "deadline add", "hockey trade") and r.random() < 0.45:
                package.append(_generate_asset(r, r.choice(["middle6w", "prospect_mid", "middle6w"])))

        def _asset_pickish(a: TradeAsset) -> bool:
            lb = (a.label or "").lower()
            return "pick" in lb or "prospect" in lb

        pickish_n = sum(1 for a in package if _asset_pickish(a))
        ain_l = (asset_in.label or "").lower()
        ain_pickish = "pick" in ain_l or "prospect" in ain_l

        if s_win == "contender" and asset_in.value >= 0.54 and pickish_n >= 2 and not cap_crunch:
            if r.random() < 0.72:
                return False
        if (
            b_win == "rebuild"
            and target_kind in ("top6f", "top4d_rental", "middle6w")
            and trade_category != "cap dump"
            and not cap_crunch
        ):
            if r.random() < 0.84:
                return False
        if (
            b_win == "rebuild"
            and not ain_pickish
            and target_kind not in ("prospect_high", "prospect_mid", "cap_dump")
            and trade_category != "cap dump"
        ):
            if r.random() < 0.81:
                return False

        buyer_out_value = sum(a.value for a in package)
        buyer_gain = asset_in.value - buyer_out_value
        seller_gain = buyer_out_value - asset_in.value * 0.85
        if b_win == "rebuild" and not ain_pickish and target_kind not in ("prospect_high", "prospect_mid", "cap_dump") and trade_category != "cap dump":
            buyer_gain *= 0.66
        if b_win == "contender" and s_win in ("rebuild", "declining"):
            buyer_gain *= 1.05
            seller_gain *= 1.04
        if b_ent:
            rb_b = str(getattr(b_ent, "gm_risk_band", "medium")).lower()
            if rb_b == "low":
                buyer_gain *= 0.94
            elif rb_b == "high":
                buyer_gain *= 1.05
        if s_ent:
            rb_s = str(getattr(s_ent, "gm_risk_band", "medium")).lower()
            if rb_s == "low":
                seller_gain *= 0.94
            elif rb_s == "high":
                seller_gain *= 1.04
        buyer_gain *= _identity_trade_bias(state, buyer.team_id)
        seller_gain *= _identity_trade_bias(state, seller.team_id)
        if buyer.bucket in ("contender", "playoff") and seller.bucket == "rebuild":
            buyer_gain *= 1.09
            seller_gain *= 1.07
        if deadline_style or deadline_spike:
            buyer_gain *= 1.13
            seller_gain *= 1.09
        if cap_crunch:
            buyer_gain *= 1.06
            seller_gain *= 1.05

        accepted, fairness, breakdown = _trade_eval(buyer_gain, seller_gain, chaos_f, r, relaxed=relaxed)
        if not accepted and force_append:
            accepted = True
            breakdown = {**breakdown, "forced_minimum": True}

        if not accepted:
            return False

        retained = 0.0
        if asset_in.cap_m > 3.0 and r.random() < ucfg.retention_rate:
            retained = clamp(r.uniform(0.10, 0.50), 0.0, 0.50)

        headline = (
            f"TRADE ({trade_category}): {buyer.team_name} acquires {asset_in.label} from {seller.team_name} for "
            + " + ".join(a.label for a in package)
        )

        details: Dict[str, Any] = {
            "buyer": buyer.team_id,
            "seller": seller.team_id,
            "asset_in": safe_to_primitive(asset_in),
            "assets_out": safe_to_primitive(package),
            "cap_in_m": asset_in.cap_m,
            "cap_retained_pct": retained,
            "buyer_gain": buyer_gain,
            "seller_gain": seller_gain,
            "fairness": fairness,
            "breakdown": breakdown,
            "reason": "deadline push" if deadline_style else ("cap pressure" if cap_crunch else "needs-based"),
            "trade_category": trade_category,
            "deadline_spike": bool(deadline_style or deadline_spike),
            "cap_pressure_trade": bool(cap_crunch),
        }

        tags = ["trade", trade_category.replace(" ", "_").lower()]
        if asset_in.label.lower().startswith("top-4"):
            tags.append("defense")
        if "pick" in headline.lower():
            tags.append("futures")
        if retained > 0:
            tags.append("retained")
        if details["deadline_spike"]:
            tags.append("deadline")
        if breakdown.get("forced_minimum"):
            tags.append("forced_minimum")

        impact = clamp(0.35 + asset_in.value * 0.75, 0.15, 0.95)

        events.append(
            UniverseEvent(
                event_id=str(uuid.uuid4()),
                year=year,
                day=None,
                type="TRADE",
                teams=[buyer.team_id, seller.team_id],
                headline=headline,
                details=details,
                impact_score=impact,
                tags=tags,
            )
        )
        if details["deadline_spike"]:
            stats["deadline_trades"] += 1
            deadline_buyer_ids.add(buyer.team_id)
            deadline_seller_ids.add(seller.team_id)
        return True

    attempt = 0
    while len(events) < desired and attempt < max_attempts:
        attempt += 1
        shape = (year + attempt * 17 + len(events) * 3) % 4
        d_spike = r.random() < 0.48 or (cap_tight and r.random() < 0.40)
        one_trade_attempt(
            relaxed=False,
            deadline_spike=d_spike,
            cap_story=cap_tight and r.random() < 0.34,
            shape=shape,
            force_append=False,
        )

    forced_fill = 0
    while len(events) < MIN_SEASON_TRADES and forced_fill < 40:
        forced_fill += 1
        shape = (year + forced_fill + len(events)) % 4
        if not one_trade_attempt(
            relaxed=True,
            deadline_spike=r.random() < 0.35,
            cap_story=cap_tight,
            shape=shape,
            force_append=True,
        ):
            one_trade_attempt(
                relaxed=True,
                deadline_spike=True,
                cap_story=True,
                shape=(shape + 1) % 4,
                force_append=True,
            )

    stats["total_trades"] = len(events)
    stats["deadline_buyers"] = len(deadline_buyer_ids)
    stats["deadline_sellers"] = len(deadline_seller_ids)
    stats["forced_trades"] = sum(
        1 for e in events if (getattr(e, "details", None) or {}).get("breakdown", {}).get("forced_minimum")
    )
    return events, stats

# =============================================================================
# Free agency (narrative)
# =============================================================================

@dataclass
class FreeAgent:
    name: str
    role: str
    rating: float
    ask_m: float
    prefs: Dict[str, float]  # weights for money/contender/location/role

def _fa_player_name(r: random.Random) -> str:
    """Use global name generator for FA names when available; else placeholder."""
    if generate_human_identity is not None:
        try:
            identity = generate_human_identity(r)
            return identity.full_name
        except Exception:
            pass
    return f"UFA_{r.randint(1000, 9999)}"

def _gen_fa_pool(r: random.Random, inflation: float) -> List[FreeAgent]:
    pool: List[FreeAgent] = []
    # stars (rare)
    if r.random() < 0.25:
        pool.append(
            FreeAgent(
                name=_fa_player_name(r),
                role=r.choice(["elite winger", "1D", "franchise center", "star goalie"]),
                rating=r.uniform(0.82, 0.93),
                ask_m=r.uniform(9.0, 13.5) * inflation,
                prefs={"money": 0.40, "contender": 0.30, "location": 0.15, "role": 0.15},
            )
        )

    # regular tiers
    tiers = [
        ("top-6 forward", (0.65, 0.80), (6.0, 9.0)),
        ("top-4 defense", (0.62, 0.78), (5.5, 8.8)),
        ("starter goalie", (0.62, 0.80), (5.0, 9.5)),
        ("middle-6 forward", (0.48, 0.63), (2.8, 5.5)),
        ("2nd-pair D", (0.48, 0.62), (2.9, 5.8)),
        ("backup goalie", (0.38, 0.52), (1.2, 2.9)),
        ("depth", (0.20, 0.40), (0.85, 1.4)),
    ]
    count = r.randint(18, 34)
    for _ in range(count):
        role, (rl, rh), (al, ah) = r.choice(tiers)
        pool.append(
            FreeAgent(
                name=_fa_player_name(r),
                role=role,
                rating=r.uniform(rl, rh),
                ask_m=r.uniform(al, ah) * inflation,
                prefs={
                    "money": r.uniform(0.30, 0.55),
                    "contender": r.uniform(0.15, 0.40),
                    "location": r.uniform(0.05, 0.25),
                    "role": r.uniform(0.05, 0.25),
                },
            )
        )
    return pool

def _team_attractiveness(team: TeamStanding, state: UniverseState, r: random.Random) -> float:
    # contender & big market narrative effect
    market = 0.50 + r.uniform(-0.15, 0.15)
    contender = 0.70 if team.bucket == "contender" else 0.58 if team.bucket == "playoff" else 0.50
    return clamp(0.40 * market + 0.60 * contender, 0.10, 0.95)

def _fa_cap_signing_cap(pressure: str) -> int:
    p = (pressure or "moderate").lower()
    if p == "cap_hell":
        return 1
    if p == "critical":
        return 2
    if p == "high":
        return 4
    if p == "moderate":
        return 9
    return 14


def _fa_signings(
    standings: List[TeamStanding],
    state: UniverseState,
    ucfg: UniverseConfig,
    r: random.Random,
    year: int,
    injected: List[ScenarioEvent],
    teams: Optional[List[Any]] = None,
    logger: Optional[Any] = None,
) -> Tuple[List[UniverseEvent], Dict[str, Any]]:
    events: List[UniverseEvent] = []
    fa_stats: Dict[str, Any] = {"major_signings": 0}
    team_by_id = {_team_id(t): t for t in (teams or [])}
    infl_f = float(getattr(state, "runner_contract_inflation", 1.0) or 1.0)

    roll = r.random()
    _trace_roll(state, ucfg, "fa_year_roll", r, roll)
    fa_gate = max(0.58, float(ucfg.free_agency_rate) * 0.88)
    if roll > fa_gate:
        return events, fa_stats

    pool = _gen_fa_pool(r, state.inflation_factor)
    for _ in range(1 + (1 if r.random() < 0.85 else 0)):
        pool.append(
            FreeAgent(
                name=_fa_player_name(r),
                role=r.choice(["elite winger", "1D", "franchise center", "star goalie"]),
                rating=r.uniform(0.82, 0.93),
                ask_m=r.uniform(9.0, 13.5) * state.inflation_factor,
                prefs={"money": 0.40, "contender": 0.30, "location": 0.15, "role": 0.15},
            )
        )

    # each team attempts some signings depending on window
    teams_sorted = sorted(standings, key=lambda s: s.points, reverse=True)
    max_signings = int(clamp(28 + r.randint(0, 28), 22, 58))
    signings_done = 0

    # build attractiveness snapshot (stable per year)
    attract = {t.team_id: _team_attractiveness(t, state, r) for t in teams_sorted}

    def pick_team_for_player(fa: FreeAgent, k_sample: int, score_floor: float) -> Optional[TeamStanding]:
        candidates = r.sample(teams_sorted, k=min(len(teams_sorted), k_sample))
        scored: List[Tuple[float, TeamStanding]] = []
        for tm in candidates:
            arche = _team_archetype(state, tm.team_id)
            # money (simulate offers): contenders slightly lower; rebuilds overpay
            overpay = 1.0
            if tm.bucket == "rebuild":
                overpay = r.uniform(1.05, 1.25)
            elif tm.bucket == "contender":
                overpay = r.uniform(0.95, 1.10)
            if arche == "win_now":
                overpay *= r.uniform(1.04, 1.18)
            if arche == "rebuild":
                overpay *= r.uniform(0.92, 1.08)
            if arche == "draft_and_develop":
                overpay *= r.uniform(0.88, 0.98)
            offer = fa.ask_m * overpay * infl_f

            # role opportunity: rebuild/bubble offer bigger role
            role_score = 0.70 if tm.bucket in ("rebuild", "bubble") else 0.55 if tm.bucket == "playoff" else 0.50

            score = (
                fa.prefs["money"] * (offer / (fa.ask_m * 1.20)) +
                fa.prefs["contender"] * (0.85 if tm.bucket in ("contender", "playoff") else 0.45) +
                fa.prefs["location"] * attract[tm.team_id] +
                fa.prefs["role"] * role_score
            )
            if arche == "win_now" and fa.rating >= 0.70:
                score += 0.07
            if arche == "win_now" and fa.rating < 0.58:
                score -= 0.045
            if arche == "rebuild" and fa.rating >= 0.78:
                score -= 0.06
            if arche == "rebuild" and fa.rating < 0.52:
                score += 0.035
            if arche == "draft_and_develop" and fa.rating < 0.56:
                score += 0.055
            if arche == "chaos_agent":
                score += r.uniform(-0.04, 0.10)
            pres = str((getattr(state, "runner_cap_pressure_by_team", None) or {}).get(tm.team_id, "moderate")).lower()
            if pres == "high":
                score *= 0.78
            elif pres == "critical":
                score *= 0.58
            elif pres == "cap_hell":
                score *= 0.42
            elif pres == "low":
                score *= 1.05
            tent = team_by_id.get(tm.team_id)
            gwin = (
                str(getattr(tent, "window", getattr(tent, "gm_window", "")) or "").lower()
                if tent is not None
                else ""
            )
            if gwin == "rebuild" and fa.rating >= 0.72:
                score -= 0.12
            elif gwin == "contender" and fa.rating >= 0.68:
                score += 0.085
            elif gwin == "declining" and fa.rating >= 0.74:
                score -= 0.055
            if tent is not None:
                gmstr = str(getattr(tent, "gm_strategy", "") or "").lower()
                if gmstr == "draft_focus" and fa.rating >= 0.76 and float(fa.ask_m) > 6.5:
                    score -= 0.09
                if gmstr in ("win_now", "aggressive") and fa.rating >= 0.70:
                    score += 0.05
            if tent is not None and _ENGINE_ECON is not None:
                try:
                    if _ENGINE_ECON.prefers_free_agent_match(tent, float(fa.rating), 24):
                        score *= 1.065
                    else:
                        score *= 0.91
                    ef = float(
                        _ENGINE_ECON.era_system_fit_multiplier(
                            state.active_era, str(getattr(tent, "system", "balanced") or "balanced")
                        )
                    )
                    score *= clamp(0.97 + (ef - 1.0) * 0.45, 0.93, 1.08)
                except Exception:
                    pass
            # sprinkle chaos
            score += r.uniform(-0.05, 0.05) * (0.5 + state.chaos_index)
            scored.append((score, tm))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] > score_floor:
            return scored[0][1]
        return None

    major_threshold = 0.75
    pool_sorted = sorted(pool, key=lambda x: x.rating, reverse=True)
    top_n = min(5, len(pool_sorted))
    fa_signed_by_team: Dict[str, int] = {}

    for idx, fa in enumerate(pool_sorted):
        if signings_done >= max_signings:
            break

        k_cand = 16 if idx < top_n or fa.rating >= 0.72 else 10
        floor = 0.36 if idx < top_n or fa.rating >= 0.72 else 0.44
        tm = pick_team_for_player(fa, k_cand, floor)
        if tm is None and fa.rating >= 0.70:
            tm = pick_team_for_player(fa, min(20, len(teams_sorted)), 0.30)
        if tm is None:
            continue

        arche_sel = _team_archetype(state, tm.team_id)
        overpay_sel = 1.0
        if tm.bucket == "rebuild":
            overpay_sel = r.uniform(1.05, 1.25)
        elif tm.bucket == "contender":
            overpay_sel = r.uniform(0.95, 1.10)
        if arche_sel == "win_now":
            overpay_sel *= r.uniform(1.04, 1.18)
        if arche_sel == "rebuild":
            overpay_sel *= r.uniform(0.92, 1.08)
        if arche_sel == "draft_and_develop":
            overpay_sel *= r.uniform(0.88, 0.98)
        if arche_sel == "chaos_agent":
            overpay_sel *= 1.0 + r.uniform(0.0, 0.40)
        tm_ent = team_by_id.get(tm.team_id)
        if tm_ent is not None:
            gw_o = str(getattr(tm_ent, "window", getattr(tm_ent, "gm_window", "")) or "").lower()
            if gw_o == "contender":
                overpay_sel *= r.uniform(1.03, 1.16)
            elif gw_o == "rebuild":
                overpay_sel *= r.uniform(0.86, 0.97)
        contract_m = float(fa.ask_m) * overpay_sel * infl_f
        if tm_ent is not None and _ENGINE_ECON is not None:
            try:
                if not _ENGINE_ECON.can_afford(tm_ent, contract_m, salary_cap_m=state.salary_cap_m):
                    continue
            except Exception:
                pass

        pres_tm = str((getattr(state, "runner_cap_pressure_by_team", None) or {}).get(tm.team_id, "moderate")).lower()
        if fa_signed_by_team.get(tm.team_id, 0) >= _fa_cap_signing_cap(pres_tm):
            continue

        if pres_tm == "cap_hell" and float(fa.ask_m) > 5.25:
            continue
        if pres_tm == "critical" and float(fa.ask_m) > 9.5 and r.random() < 0.62:
            continue

        if tm.bucket == "rebuild" and fa.rating > 0.78 and r.random() < 0.38:
            continue
        gwin_tm = (
            str(getattr(tm_ent, "window", getattr(tm_ent, "gm_window", "")) or "").lower()
            if tm_ent is not None
            else ""
        )
        if gwin_tm == "rebuild" and fa.rating > 0.72 and r.random() < 0.52:
            continue
        if gwin_tm == "rebuild" and str(getattr(tm_ent, "gm_strategy", "") or "").lower() == "draft_focus" and fa.rating > 0.68 and float(fa.ask_m) > 5.8 and r.random() < 0.45:
            continue

        signings_done += 1
        fa_signed_by_team[tm.team_id] = fa_signed_by_team.get(tm.team_id, 0) + 1

        is_major = fa.rating >= major_threshold or fa.ask_m >= 8.5
        if is_major:
            fa_stats["major_signings"] = int(fa_stats["major_signings"]) + 1

        if (
            logger is not None
            and str(arche_sel).lower() == "win_now"
            and is_major
            and r.random() < 0.38
        ):
            logger.emit("CAP ACTION: Overpay signing (win_now aggression)", "normal")

        headline = f"SIGNING: {tm.team_name} signs {fa.name} ({fa.role}) at ~${fmt_money(contract_m)} AAV"

        details = {
            "team_id": tm.team_id,
            "player": safe_to_primitive(fa),
            "contender_status": tm.bucket,
            "attractiveness": attract[tm.team_id],
            "fa_bidders_sample": k_cand,
            "contract_inflation": infl_f,
            "signed_aav_m": contract_m,
            "storyline": "free_agency_cap_filtered",
        }

        events.append(
            UniverseEvent(
                event_id=str(uuid.uuid4()),
                year=year,
                day=None,
                type="SIGNING",
                teams=[tm.team_id],
                headline=headline,
                details=details,
                impact_score=0.80 if is_major else 0.35,
                tags=["free_agency", "major" if is_major else "wire"],
            )
        )

        if is_major and r.random() < 0.18 * (0.5 + state.chaos_index):
            events[-1].tags.append("unexpected")

    return events, fa_stats

# =============================================================================
# Coaches (carousel)
# =============================================================================

def _coach_changes(standings: List[TeamStanding], state: UniverseState, ucfg: UniverseConfig, r: random.Random, year: int,
                   injected: List[ScenarioEvent]) -> List[UniverseEvent]:
    events: List[UniverseEvent] = []

    # scenario: force_rebuild / force_cup_run affects coaching in narrative
    forced_rebuild = [e for e in injected if e.type.upper() == "FORCE_REBUILD"]
    if forced_rebuild:
        tid = forced_rebuild[0].target_team_id
        if tid:
            events.append(
                UniverseEvent(
                    event_id=str(uuid.uuid4()),
                    year=year,
                    day=None,
                    type="REBUILD_COMMIT",
                    teams=[tid],
                    headline=f"{tid} commits to a rebuild (forced scenario).",
                    details={"payload": forced_rebuild[0].payload},
                    impact_score=0.65,
                    tags=["rebuild", "scenario"],
                )
            )
            state.team_archetypes[tid] = "rebuild"

    if not ucfg.enable_coach_changes:
        return events

    # evaluate bottom teams
    bottom = list(reversed(standings))[:max(4, len(standings) // 6)]
    for s in bottom:
        tid = s.team_id
        sec = float(state.coach_security.get(tid, 0.55))
        arche = state.team_archetypes.get(tid, "balanced")

        underperf = (s.point_pct - 0.50) < -0.03
        mismatch = (arche == "win_now" and s.bucket == "rebuild")

        fire_base = ucfg.coach_firing_rate
        if underperf:
            fire_base += 0.10
        if mismatch:
            fire_base += 0.10
        fire_base += clamp(0.55 - sec, -0.20, 0.25)

        roll = r.random()
        _trace_roll(state, ucfg, f"coach_fire_roll::{tid}", r, roll)
        if roll < clamp(fire_base, 0.02, 0.75):
            old = state.coach_ids.get(tid, f"Coach_{abs(split_seed(year, tid))%1000:03d}")
            events.append(
                UniverseEvent(
                    event_id=str(uuid.uuid4()),
                    year=year,
                    day=None,
                    type="COACH_FIRE",
                    teams=[tid],
                    headline=f"COACHING: {s.team_name} fires {old} after disappointing season",
                    details={"team_id": tid, "coach_id": old, "underperf": underperf, "mismatch": mismatch, "job_security": sec},
                    impact_score=0.55,
                    tags=["coaching"],
                )
            )

            # hire
            cand_type = r.choice(["rookie", "established", "veteran"])
            specialty = r.choice(["systems", "tactics", "development", "motivation", "defense", "offense"])
            new = f"Coach_{cand_type[:1].upper()}{r.randint(100,999)}_{specialty[:3].upper()}"
            fit = 0.55
            if arche == "rebuild" and specialty in ("development", "motivation"):
                fit = 0.78
            if arche == "win_now" and specialty in ("systems", "tactics", "defense"):
                fit = 0.75
            fit += r.uniform(-0.08, 0.08)

            state.coach_ids[tid] = new
            state.coach_security[tid] = clamp(0.55 + 0.30 * fit, 0.15, 0.95)

            events.append(
                UniverseEvent(
                    event_id=str(uuid.uuid4()),
                    year=year,
                    day=None,
                    type="COACH_HIRE",
                    teams=[tid],
                    headline=f"COACHING: {s.team_name} hires {new} ({specialty}) to reset the room",
                    details={"team_id": tid, "coach_id": new, "specialty": specialty, "fit_score": fit},
                    impact_score=0.50,
                    tags=["coaching"],
                )
            )

    return events

# =============================================================================
# Draft lottery integration
# =============================================================================

def _draft_lottery(
    standings: List[TeamStanding],
    year: int,
    lottery_seed: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns (lottery_results, pick_order_team_ids for bottom-16 block).
    Uses the same LotteryTeam(points) + run_draft_lottery(seed=...) API as SimEngine.run_offseason_draft.
    """
    bottom16 = list(reversed(standings))[:16]
    pick_order = [s.team_id for s in bottom16]

    if LotteryTeam is None or run_draft_lottery is None:
        results = [{"pick": i + 1, "team_id": tid, "note": "fallback_no_lottery_module"} for i, tid in enumerate(pick_order)]
        return results, pick_order

    lot_teams = [LotteryTeam(team_id=s.team_id, points=int(s.points)) for s in bottom16]  # type: ignore[misc]
    try:
        lottery = run_draft_lottery(teams=lot_teams, seed=lottery_seed)  # type: ignore[call-arg]
        pick_ids = list(getattr(lottery, "pick_order", []) or [])
        winners = list(getattr(lottery, "lottery_winners", []) or [])
        wset = {str(x) for x in winners}
        results = [
            {
                "pick": i + 1,
                "team_id": str(tid),
                "lottery_winner": str(tid) in wset,
            }
            for i, tid in enumerate(pick_ids)
        ]
        return results, pick_ids
    except Exception:
        results = [{"pick": i + 1, "team_id": tid, "note": "lottery_failed_fallback"} for i, tid in enumerate(pick_order)]
        return results, pick_order

# =============================================================================
# Universe yearly pipeline (ORDER MATTERS)
# =============================================================================

@dataclass
class UniverseYearResult:
    year: int
    state: UniverseState
    standings: List[TeamStanding]
    events: List[UniverseEvent]
    lottery_results: List[Dict[str, Any]] = field(default_factory=list)
    draft_pick_order: List[str] = field(default_factory=list)
    full_draft_order_32: List[str] = field(default_factory=list)
    waiver_claims: List[WaiverClaim] = field(default_factory=list)
    waived_count: int = 0
    playoff_champion_id: Optional[str] = None

def simulate_universe_year(
    league: Any,
    state: UniverseState,
    run_cfg: RunConfig,
    uni_cfg: UniverseConfig,
    scn_cfg: ScenarioConfig,
    year: int,
    logger: RunnerLogger,
    rng: random.Random,
    sim: Optional[Any] = None,
    narrative_context: Optional[Dict[str, Any]] = None,
) -> UniverseYearResult:
    """
    Universe pipeline:
    - Advance league context (league.advance_season if exists)
    - Economics + era
    - Standings approximation
    - Waiver priority reset
    - Coach changes
    - Trades
    - Waivers
    - Draft lottery (+ optional draft)
    - Free agency
    - Injuries (optional / narrative)
    - Finalize summary
    """
    injected = scenario_events_for_year(scn_cfg, year)
    events: List[UniverseEvent] = []

    # Phase 0: advance season context
    adv = _safe_getattr(league, "advance_season", None)
    if callable(adv):
        try:
            adv()  # engine should do internal state; runner owns I/O
        except Exception as e:
            events.append(
                UniverseEvent(
                    event_id=str(uuid.uuid4()),
                    year=year,
                    day=None,
                    type="NOTE",
                    teams=[],
                    headline=f"WARNING: league.advance_season() failed ({type(e).__name__}). Universe will continue with approximations.",
                    details={"error": str(e)},
                    impact_score=0.10,
                    tags=["warning"],
                )
            )

    teams = _get_league_teams(league)
    if not teams:
        # Universe can still produce minimal macro state
        events.append(
            UniverseEvent(
                event_id=str(uuid.uuid4()),
                year=year,
                day=None,
                type="NOTE",
                teams=[],
                headline="WARNING: No teams found in league object. Universe year will be minimal.",
                details={},
                impact_score=0.05,
                tags=["warning"],
            )
        )

    tuning_year: Dict[str, Any] = {}

    # Phase 0b: narrative modifier layer → era → ages → run_player_progression (once) → lifecycle (major progression only, no duplicate base progress) → soft OVR guard → identity → emergence pass → retirements
    if teams:
        if narrative_context is not None:
            try:
                from app.sim_engine.narrative.player_journeys import apply_narrative_mechanics_to_rosters

                apply_narrative_mechanics_to_rosters(
                    league, narrative_context, int(year), rng, max_trace_lines=36
                )
            except Exception:
                pass
        try:
            from app.sim_engine.progression import development as _dev_prime

            _dev_prime.prime_development_environment_for_rosters(teams, rng)
        except Exception:
            pass
        tuning_year["era_start"] = _tuning_start_of_season(league, teams, state)
        if _ENGINE_ECON is not None:
            try:
                _ENGINE_ECON.runner_identity_bootstrap(teams, rng)
                dch = float(_ENGINE_ECON.league_chaos_delta_from_team_systems(teams))
                state.chaos_index = clamp(state.chaos_index + dch, 0.08, 0.98)
            except Exception:
                pass
        _advance_roster_ages_and_development(teams, league, state, rng)
        if sim is not None:
            rlc = getattr(sim, "restore_line_chemistry_ratings", None)
            if callable(rlc):
                try:
                    rlc()
                except Exception:
                    pass
        _run_player_progression_pass(teams, rng, logger)
        try:
            setattr(league, "_tuning_context", _tuning_ctx_from_state(state))
        except Exception:
            pass
        _prime_lifecycle_event_caps(league, state)
        tuning_year["career_lifecycle"] = _run_career_lifecycle_pass(
            teams, rng, logger, league=league, state=state, season_year=int(year)
        )
        if _ENGINE_ECON is not None:
            try:
                _ENGINE_ECON.apply_league_ovr_soft_regression_if_needed(teams, rng)
            except Exception:
                pass
        _cl = tuning_year.get("career_lifecycle") or {}
        if int(_cl.get("special_events", 0) or 0) > 0:
            try:
                logger.emit(
                    f"CAREER LIFECYCLE: year={year} special_events={_cl.get('special_events', 0)} "
                    f"players_touched={_cl.get('players_touched', 0)}",
                    "normal",
                )
            except Exception:
                pass
        if _ENGINE_ECON is not None:
            try:
                def _tid_emit(line: str) -> None:
                    logger.emit(line, "normal")

                tuning_year["team_identity_fx"] = _ENGINE_ECON.runner_identity_annual_application(
                    teams,
                    rng,
                    state.active_era,
                    year=int(year),
                    log_emit=_tid_emit,
                    do_print=False,
                )
            except Exception:
                tuning_year["team_identity_fx"] = []
        if sim is not None:
            reb = getattr(sim, "apply_progression_rebalance", None)
            if callable(reb):
                try:
                    reb(rng)
                except Exception:
                    pass
        if sim is not None and teams:
            dpp = getattr(sim, "run_player_distribution_pass", None)
            if callable(dpp):
                try:
                    tuning_year["player_distribution_pipeline"] = dpp(rng)
                except Exception:
                    tuning_year["player_distribution_pipeline"] = {}
        tuning_year["post_progression"] = _tuning_after_progression(league, teams, state, year=year, rng=rng)
        if sim is not None and teams:
            pnr = getattr(sim, "post_normalize_distribution_rescue", None)
            if callable(pnr):
                try:
                    tuning_year["player_distribution_post_normalize"] = pnr(rng)
                except Exception:
                    tuning_year["player_distribution_post_normalize"] = {}
            apr = getattr(sim, "apply_percentile_player_roles", None)
            if callable(apr):
                try:
                    tuning_year["player_distribution_role_moves"] = apr()
                except Exception:
                    tuning_year["player_distribution_role_moves"] = 0
            summ = getattr(sim, "summarize_roster_distribution", None)
            if callable(summ):
                try:
                    tuning_year["player_distribution_summary"] = summ()
                except Exception:
                    tuning_year["player_distribution_summary"] = {}
        if sim is not None:
            spp = getattr(sim, "run_player_storyline_pass", None)
            if callable(spp):
                try:
                    raw_sl = spp(rng, year)
                    if isinstance(raw_sl, dict):
                        tuning_year["player_storylines"] = raw_sl.get("player_storylines", [])
                        tuning_year["narrative_consequences"] = raw_sl.get("narrative_consequences", [])
                        ld = raw_sl.get("league_delta") or {}
                        tuning_year["narrative_consequence_league_delta"] = ld
                        state.chaos_index = clamp(
                            state.chaos_index + float(ld.get("chaos_index", 0) or 0),
                            0.08,
                            0.98,
                        )
                        state.parity_index = clamp(
                            state.parity_index + float(ld.get("parity_index", 0) or 0),
                            0.10,
                            0.90,
                        )
                        nbal = raw_sl.get("narrative_balance")
                        if isinstance(nbal, dict):
                            tuning_year["narrative_balance"] = dict(nbal)
                    else:
                        tuning_year["player_storylines"] = list(raw_sl or [])
                        tuning_year["narrative_consequences"] = []
                except Exception:
                    tuning_year["player_storylines"] = []
                    tuning_year["narrative_consequences"] = []
        if sim is not None:
            try:
                if getattr(sim, "apply_emergence_and_bust_pass", None) is not None:
                    sim.apply_emergence_and_bust_pass(league, rng)
                if getattr(sim, "apply_aging_calibration", None) is not None:
                    sim.apply_aging_calibration(league, rng)
                if getattr(sim, "league_balance_check", None) is not None:
                    sim.league_balance_check(league, rng, year)
            except Exception:
                pass
        if sim is not None:
            if league is not None:
                try:
                    setattr(league, "_runner_team_archetypes", dict(getattr(state, "team_archetypes", None) or {}))
                except Exception:
                    pass
            alc = getattr(sim, "apply_forward_line_chemistry_pass", None)
            if callable(alc):
                try:
                    tuning_year["line_chemistry"] = alc()
                    if league is not None:
                        tuning_year["player_archetype_logs"] = list(
                            getattr(league, "_player_archetype_assignment_logs", None) or []
                        )
                except Exception:
                    tuning_year["line_chemistry"] = []
                    tuning_year["player_archetype_logs"] = []

        if sim is not None:
            cdf = getattr(sim, "compute_league_age_distribution", None)
            if callable(cdf):
                try:
                    stats_age_pre = _normalize_age_stats_from_engine(cdf())
                except Exception:
                    stats_age_pre = league_age_distribution_stats(teams)
            else:
                stats_age_pre = league_age_distribution_stats(teams)
        else:
            stats_age_pre = league_age_distribution_stats(teams)
        pct_v = float(stats_age_pre.get("pct_v30", 0.0))
        pct_u = float(stats_age_pre.get("pct_u24", 0.0))
        setattr(
            league,
            "_age_balance_retirement_ctx",
            {"pct_u24": pct_u, "pct_v30": pct_v},
        )
        if pct_v > 30.0:
            _emit_age_balance_action(logger, "retirement_boost", "veteran_overload")
        dev_strength = 0.0
        u24_enter = 17.0
        u24_clear = 20.75
        latched = bool(getattr(league, "_u24_dev_boost_latched", False))
        if pct_u < u24_enter:
            latched = True
        elif pct_u > u24_clear:
            latched = False
        setattr(league, "_u24_dev_boost_latched", latched)
        if pct_u < u24_enter:
            dev_strength = min(1.0, (u24_enter - pct_u) / 9.0)
            setattr(league, "_age_balance_dev_strength", dev_strength)
            _emit_age_balance_action(logger, "dev_boost", "U24_low")
        elif latched and pct_u <= u24_clear:
            dev_strength = min(0.38, (u24_clear - pct_u) / 16.0)
            setattr(league, "_age_balance_dev_strength", dev_strength)
        else:
            setattr(league, "_age_balance_dev_strength", 0.0)

        n_active_pre_ret = sum(
            1
            for tm in teams
            for pl in getattr(tm, "roster", None) or []
            if not getattr(pl, "retired", False)
        )
        roster_slots_target = max(1, 18 * len(teams))
        fill_ratio = float(n_active_pre_ret) / float(roster_slots_target)
        try:
            setattr(
                league,
                "_retirement_population_guard",
                {"pct_u24": float(pct_u), "roster_fill_ratio": fill_ratio, "n_active": int(n_active_pre_ret)},
            )
        except Exception:
            pass

        retired_list = _run_league_retirements(teams, league, rng, year, events, logger=logger)
        try:
            setattr(league, "_runner_retirements_this_year", len(retired_list))
        except Exception:
            setattr(league, "_runner_retirements_this_year", 0)

        if sim is not None and dev_strength > 0.0:
            ydev = getattr(sim, "apply_age_balance_youth_development", None)
            if callable(ydev):
                try:
                    ydev(rng)
                except Exception:
                    pass
        setattr(league, "_age_balance_dev_strength", 0.0)

        if sim is not None:
            cdf2 = getattr(sim, "compute_league_age_distribution", None)
            if callable(cdf2):
                try:
                    stats_age_post = _normalize_age_stats_from_engine(cdf2())
                except Exception:
                    stats_age_post = league_age_distribution_stats(teams)
            else:
                stats_age_post = league_age_distribution_stats(teams)
        else:
            stats_age_post = league_age_distribution_stats(teams)
        _emit_age_distribution_check(logger, stats_age_post)
        if sim is not None and float(stats_age_post.get("pct_u24", 25.0) or 25.0) < 10.5:
            try:
                eco_fn = getattr(sim, "ecosystem_operational_repairs", None)
                if callable(eco_fn):
                    for _eln in eco_fn(teams, rng, year)[:12]:
                        logger.emit(f"ECOSYSTEM REPAIR (U24 stress): {_eln}", "normal")
            except Exception:
                pass

    # Phase 1: economics
    state, econ_events = _economics_advance(state, uni_cfg, rng, year, injected)
    events.extend(econ_events)

    if teams:
        _execute_league_cap_consequence_pass(teams, league, state, events, logger, rng, year, run_cfg, tuning_year)

    # Phase 2: era/meta
    state, era_events = _maybe_era_shift(state, uni_cfg, rng, year, injected)
    events.extend(era_events)
    for ev in era_events:
        if (getattr(ev, "type", "") or "").upper() == "ERA_SHIFT":
            logger.emit(getattr(ev, "headline", "ERA SHIFT"), "normal")

    # Phase 3: rare ambient identity drift (performance evolution runs after results)
    if teams and rng.random() < float(getattr(uni_cfg, "archetype_drift_rate", 0.10) or 0.10) * 0.065:
        events.extend(_team_identity_drift(state, uni_cfg, rng, year, teams))

    # Phase 4: standings baseline (era scoring multiplier from tuning when available)
    standings = _simulate_standings(teams, state, uni_cfg, rng, league) if teams else []
    if teams and standings:
        _sync_playoff_streaks_and_contender_flags(teams, standings)
        standings = _identity_guardrail_adjust_standings(standings, state, rng)
        _runner_update_rivalry_narrative(league, standings, rng, logger, year)
        _emit_awards_media_hof_org_pack(teams, standings, state, logger, year)
        _emit_identity_impact_logs(logger, state, standings)
        _emit_identity_trajectory_logs(logger, state, standings)
    playoff_champion_id: Optional[str] = None

    # Phase 4b: playoff resolution + dynasty state (cup/finals, power_states)
    if len(standings) >= 2:
        champion_id, runner_up_id = _resolve_playoff_champion(standings, rng, state)
        playoff_champion_id = champion_id
        cw = dict(getattr(state, "cup_wins_by_team", None) or {})
        fa = dict(getattr(state, "finals_appearances_by_team", None) or {})
        cw[champion_id] = cw.get(champion_id, 0) + 1
        fa[champion_id] = fa.get(champion_id, 0) + 1
        fa[runner_up_id] = fa.get(runner_up_id, 0) + 1
        power_states = _update_power_states(standings, state, champion_id, runner_up_id)
        state = dataclasses.replace(state, cup_wins_by_team=cw, finals_appearances_by_team=fa, power_states=power_states)

    # Phase 4c: dynamic franchise identity (react to results, age, cap, pipeline) → trades/FA use updated archetypes
    if teams and standings:
        _dynamic_team_identity_evolution_pass(
            state,
            teams,
            standings,
            league,
            rng,
            year,
            logger,
            str(playoff_champion_id or ""),
            tuning_year,
        )
        _identity_update_carry_momentum(state, standings, rng)
        _sync_team_gm_strategic_profiles(teams, standings, state, rng)

    # Phase 5: waiver priority reset (reverse standings)
    if standings and uni_cfg.enable_waivers:
        state.waiver_priority = _reset_waiver_priority_from_standings(standings)

    # Phase 6: coach carousel
    events.extend(_coach_changes(standings, state, uni_cfg, rng, year, injected))

    cap_tight_year = float(state.salary_cap_m) < 90.5

    # Phase 7: trades
    trade_events: List[UniverseEvent] = []
    trade_engine_stats: Dict[str, Any] = {}
    if uni_cfg.enable_trades and standings:
        trade_events, trade_engine_stats = _generate_trades(
            standings, state, uni_cfg, rng, year, injected, teams=teams
        )
        events.extend(trade_events)
        tuning_year["trade_engine"] = trade_engine_stats
        _emit_identity_behavior_logs(logger, state, standings, trade_events)
        logger.emit("TRADE ENGINE:", "normal")
        logger.emit(
            f"  total_trades={trade_engine_stats.get('total_trades', 0)} forced_trades={trade_engine_stats.get('forced_trades', 0)}",
            "normal",
        )
        logger.emit("DEADLINE ACTIVITY:", "normal")
        logger.emit(
            f"  trades={trade_engine_stats.get('deadline_trades', 0)} buyers={trade_engine_stats.get('deadline_buyers', 0)} sellers={trade_engine_stats.get('deadline_sellers', 0)}",
            "normal",
        )
    if standings:
        _emit_identity_behavior_logs(logger, state, standings, trade_events)

    # Phase 8: waivers
    waiver_claims: List[WaiverClaim] = []
    waived_count = 0
    if uni_cfg.enable_waivers and standings:
        waived = _generate_waiver_candidates(
            teams, standings, state, uni_cfg, rng, cap_tight=cap_tight_year, logger=logger
        )
        waived_count = len(waived)
        if waived:
            stand_map = {s.team_id: s for s in standings}
            claims, new_priority = _process_waivers(state.waiver_priority, waived, stand_map, uni_cfg, rng)
            waiver_claims = claims
            state.waiver_priority = new_priority

            # log waiver events
            for c in claims:
                events.append(
                    UniverseEvent(
                        event_id=str(uuid.uuid4()),
                        year=year,
                        day=None,
                        type="WAIVER",
                        teams=[c.from_team, c.to_team],
                        headline=f"WAIVERS: {c.to_team} claims {c.player_label} from {c.from_team}",
                        details=safe_to_primitive(dataclasses.asdict(c)),
                        impact_score=0.25,
                        tags=["waivers"],
                    )
                )

    # Phase 9: draft lottery + draft (optional)
    lottery_results: List[Dict[str, Any]] = []
    pick_order: List[str] = []
    full_draft_order_32: List[str] = []
    if uni_cfg.enable_draft and standings:
        lot_seed = draft_lottery_seed(run_cfg.seed, year)
        lottery_results, pick_order = _draft_lottery(standings, year, lot_seed)
        full_draft_order_32 = build_full_draft_order_32(standings, pick_order)
        if league is not None:
            setattr(league, "_runner_full_draft_order_32", full_draft_order_32)
        top5 = lottery_results[:5]
        events.append(
            UniverseEvent(
                event_id=str(uuid.uuid4()),
                year=year,
                day=None,
                type="LOTTERY",
                teams=[x["team_id"] for x in top5 if "team_id" in x],
                headline="DRAFT LOTTERY: " + ", ".join([f"#{x['pick']} {x.get('team_id','?')}" for x in top5]),
                details={"results": lottery_results},
                impact_score=0.70,
                tags=["draft"],
            )
        )
        draft_details: Dict[str, Any] = {"full_order_32": full_draft_order_32, "lottery_order_16": pick_order[:16]}
        if generate_human_identity is not None:
            try:
                examples = []
                for _ in range(3):
                    ident = generate_human_identity(rng)
                    examples.append(f"{ident.full_name} ({ident.nationality})")
                draft_details["example_prospects"] = examples
            except Exception:
                pass
        events.append(
            UniverseEvent(
                event_id=str(uuid.uuid4()),
                year=year,
                day=None,
                type="DRAFT",
                teams=pick_order[:10],
                headline=f"DRAFT: Entry draft order set ({len(full_draft_order_32)} slots); matches engine lottery seed.",
                details=draft_details,
                impact_score=0.25,
                tags=["draft"],
            )
        )

    # Phase 10: free agency
    fa_summary: Dict[str, Any] = {"major_signings": 0}
    if uni_cfg.enable_free_agency and standings:
        fa_events, fa_summary = _fa_signings(
            standings, state, uni_cfg, rng, year, injected, teams=teams, logger=logger
        )
        events.extend(fa_events)
        tuning_year["free_agency"] = fa_summary
        _emit_cap_event_logs(logger, standings, state, rng, year)
    if uni_cfg.enable_free_agency:
        logger.emit("FREE AGENCY SUMMARY:", "normal")
        logger.emit(f"  major_signings={fa_summary.get('major_signings', 0)}", "normal")

    # Phase 11: injuries (optional narrative)
    if uni_cfg.enable_injuries and standings:
        roll = rng.random()
        _trace_roll(state, uni_cfg, "injury_year_roll", rng, roll)
        if roll < uni_cfg.injury_rate:
            victim = rng.choice(standings[:max(8, len(standings)//4)])
            events.append(
                UniverseEvent(
                    event_id=str(uuid.uuid4()),
                    year=year,
                    day=None,
                    type="NOTE",
                    teams=[victim.team_id],
                    headline=f"INJURY: {victim.team_name} suffers a major injury scare (narrative placeholder).",
                    details={"severity": rng.choice(["week-to-week", "month-to-month", "career-altering (rare)"])},
                    impact_score=0.40,
                    tags=["injury", "narrative"],
                )
            )

    # Phase 12: end-of-year league macro tuning (post transactions / full season sim)
    if teams:
        tuning_year["end_of_year"] = _tuning_end_of_year(league, teams, state, year=year, rng=rng)
    state.tuning_report = tuning_year

    # Sort events: major first for printing (but keep list stable)
    # We'll handle printing tiers separately.
    return UniverseYearResult(
        year=year,
        state=state,
        standings=standings,
        events=events,
        lottery_results=lottery_results,
        draft_pick_order=pick_order,
        full_draft_order_32=full_draft_order_32,
        waiver_claims=waiver_claims,
        waived_count=waived_count,
        playoff_champion_id=playoff_champion_id,
    )

# =============================================================================
# Career layer (SimEngine integration)
# =============================================================================

@dataclass
class CareerYearResult:
    year: int
    player_id: Optional[str] = None
    team_id: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)

def _create_random_player(r: random.Random) -> Dict[str, Any]:
    # Runner-only placeholder identity (engine likely has its own player factory)
    pos = r.choice(["C", "LW", "RW", "D", "G"])
    shoots = r.choice(["L", "R"])
    return {
        "name": f"SimPlayer_{r.randint(1000,9999)}",
        "age": r.randint(17, 20),
        "position": pos,
        "shoots": shoots,
        "ovr": round(r.uniform(0.42, 0.58), 3),
    }

def _pick_player_team(teams: List[Any], r: random.Random) -> Optional[Any]:
    if not teams:
        return None
    return r.choice(teams)

def run_career_year(sim: Any, year: int, run_cfg: RunConfig, logger: RunnerLogger) -> CareerYearResult:
    """
    Calls sim.sim_year(debug_dump=True) if available.
    Never raises; returns best-effort summary.
    """
    res = CareerYearResult(year=year)
    if sim is None:
        res.summary["note"] = "SimEngine not available (import failed). Career layer skipped."
        return res

    sim_year_fn = _safe_getattr(sim, "sim_year", None)
    if not callable(sim_year_fn):
        res.summary["note"] = "SimEngine has no sim_year() method. Career layer skipped."
        return res

    try:
        # Prefer debug_dump if supported
        try:
            out = sim_year_fn(debug_dump=bool(run_cfg.debug))  # type: ignore
        except TypeError:
            out = sim_year_fn()  # type: ignore

        res.summary["engine_return"] = safe_to_primitive(out)

        # attempt to capture player/team references from engine
        player = _safe_getattr(sim, "player", None) or _safe_getattr(sim, "career_player", None)
        team = _safe_getattr(sim, "team", None) or _safe_getattr(sim, "career_team", None)

        if player is not None:
            res.player_id = _safe_getattr(player, "name", None) or _safe_getattr(player, "player_id", None) or str(player)
            res.summary["player"] = safe_to_primitive(player)

        if team is not None:
            res.team_id = _team_id(team)
            res.summary["team"] = {"team_id": res.team_id, "name": _team_name(team)}

        return res
    except Exception as e:
        res.summary["error"] = f"{type(e).__name__}: {e}"
        res.summary["traceback"] = traceback.format_exc().splitlines()[-20:]
        return res


def _emit_world_simulation_report(logger: RunnerLogger, league: Any, year: int) -> None:
    """Log world layer (momentum, fatigue, injuries, chemistry) after structural league season."""
    snap = getattr(league, "_world_season_snapshot", None)
    if not snap or int(snap.get("year", -1)) != int(year):
        return

    def L(line: str, lvl: str = "normal") -> None:
        logger.emit(line, lvl)

    L("", "normal")
    L("TEAM DYNAMICS", "normal")
    L(f"  (league season {year}, chaos_index={snap.get('chaos_index', 0):.3f})", "normal")
    for row in (snap.get("teams") or [])[:10]:
        L(
            f"  {row.get('team_id','?')}: momentum {float(row.get('momentum',0)):+.3f}  "
            f"chemistry {float(row.get('chemistry',0)):.3f}  avg_morale {float(row.get('avg_morale',50)):.1f}  "
            f"avg_fatigue {float(row.get('avg_fatigue',0)):.1f}",
            "normal",
        )
    L("", "normal")
    L("PLAYER CONDITIONS (sample)", "normal")
    teams_list = _get_league_teams(league)
    shown = 0
    for t in teams_list:
        roster = list(getattr(t, "roster", None) or [])
        roster.sort(key=lambda p: float(getattr(p, "_world_fatigue", 0) or 0), reverse=True)
        for p in roster[:4]:
            if shown >= 14:
                break
            if getattr(p, "retired", False):
                continue
            nm = getattr(p, "name", getattr(getattr(p, "identity", None), "name", "?"))
            fat = float(getattr(p, "_world_fatigue", 0) or 0)
            mor = float(getattr(p, "_world_morale_100", 50) or 50)
            psych = getattr(p, "psych", None)
            if psych is not None and hasattr(psych, "morale"):
                mor = float(psych.morale) * 100.0
            inj = int(getattr(p, "_world_injury_games_remaining", 0) or 0)
            tier = getattr(p, "_world_injury_tier", None) or "healthy"
            st = "healthy"
            if inj > 0:
                st = f"{tier} ({inj}g)"
            L(f"  {nm}: fatigue={fat:.0f} morale={mor:.0f} injury={st}", "normal")
            shown += 1
        if shown >= 14:
            break
    L("", "normal")
    L("SCHEDULE STRESS", "normal")
    b2b = snap.get("back_to_backs_ranked") or []
    if b2b:
        L("  Most back-to-back segments (team, count):", "normal")
        for tid, c in b2b[:8]:
            L(f"    {tid}: {int(c)}", "normal")
    fat_r = snap.get("fatigue_ranked") or []
    if fat_r:
        L("  Highest avg roster fatigue:", "normal")
        for tid, av in fat_r[:8]:
            L(f"    {tid}: {float(av):.1f}", "normal")
    L("", "normal")
    L("INJURY REPORT", "normal")
    majors = snap.get("major_injuries") or []
    if majors:
        for rec in majors[-10:]:
            L(
                f"  MAJOR: {rec.get('player','?')} ({rec.get('team_id','?')}) ~{rec.get('games','?')} games",
                "normal",
            )
    else:
        L("  (no major injuries logged this season)", "normal")
    prone = snap.get("injury_prone") or []
    if prone:
        L("  Injury events (players):", "normal")
        for rec in prone[:10]:
            L(f"    {rec.get('name','?')} ({rec.get('team_id','?')}): x{int(rec.get('count',0))}", "normal")
    L("", "normal")


# =============================================================================
# NHL UNIVERSE block printing
# =============================================================================


def _runner_lookup_player_ovr01(league: Any, player_name: str) -> float:
    if league is None:
        return 0.5
    for t in _get_league_teams(league):
        for p in getattr(t, "roster", None) or []:
            ident = getattr(p, "identity", None)
            nm = str(
                getattr(p, "name", None)
                or (getattr(ident, "name", None) if ident is not None else None)
                or ""
            )
            if nm == str(player_name):
                try:
                    fn = getattr(p, "ovr", None)
                    return float(fn()) if callable(fn) else float(fn or 0.5)
                except Exception:
                    return 0.5
    return 0.5


def _rebalance_journey_events_for_season_log(
    league: Any,
    journey_events: List[Dict[str, Any]],
    validation_out: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Post-process player_journeys output for logs only: cap rookie / breakout / superstar noise.
    Does not modify player_journeys.py; keeps the league profile tags as-is.
    """
    if not journey_events:
        return journey_events
    validation_out.setdefault("rookie_spam_trimmed", 0)
    validation_out.setdefault("journey_tag_trimmed", 0)
    drop_keys: Set[Tuple[str, str]] = set()
    for tg, cap in (
        ("breakout", 14),
        ("breakout_repeat", 10),
        ("superstar", 12),
        ("generational_talent", 4),
        ("late_bloomer", 10),
        ("team_change", 22),
        ("repeated_decline", 12),
        ("fall_from_grace", 6),
        ("volatile_star", 14),
        ("underdog_rise", 14),
    ):
        rows = [e for e in journey_events if e.get("tag") == tg]
        if len(rows) <= cap:
            continue
        scored = sorted(
            rows,
            key=lambda e: _runner_lookup_player_ovr01(league, str(e.get("player_name") or "")),
            reverse=True,
        )
        for e in scored[cap:]:
            drop_keys.add((str(tg), str(e.get("player_name") or "")))
        validation_out["journey_tag_trimmed"] += len(scored) - cap

    rookie_rows = [e for e in journey_events if e.get("tag") == "rookie_sensation"]
    rookie_drop_names: Set[str] = set()
    msgs_by_name: Dict[str, str] = {}
    if rookie_rows:
        scored_r = sorted(
            rookie_rows,
            key=lambda e: _runner_lookup_player_ovr01(league, str(e.get("player_name") or "")),
            reverse=True,
        )
        for rank, e in enumerate(scored_r):
            nm = str(e.get("player_name") or "")
            if rank < 3:
                msgs_by_name[nm] = (
                    f"{nm} delivered a historic rookie campaign — one of the season's defining newcomers."
                )
            elif rank < 10:
                msgs_by_name[nm] = (
                    f"{nm} authored a strong rookie season that earned real attention around the league."
                )
            elif rank < 18:
                msgs_by_name[nm] = f"{nm} put together a solid, workmanlike rookie year."
            else:
                rookie_drop_names.add(nm)
                validation_out["rookie_spam_trimmed"] += 1

    out: List[Dict[str, Any]] = []
    for ev in journey_events:
        tg = str(ev.get("tag") or "")
        nm = str(ev.get("player_name") or "")
        if (tg, nm) in drop_keys:
            continue
        if tg == "rookie_sensation" and nm in rookie_drop_names:
            continue
        ev2 = dict(ev)
        if tg == "rookie_sensation" and nm in msgs_by_name:
            ev2["message"] = msgs_by_name[nm]
        out.append(ev2)
    return out


def print_year_header(
    logger: RunnerLogger,
    year: int,
    state: UniverseState,
    standings: List[TeamStanding],
    events: List[UniverseEvent],
    log_level: str,
    league: Any = None,
    uni_result: Any = None,
    playoff_champion_id: Optional[str] = None,
    narrative_context: Optional[Dict[str, Any]] = None,
    narrative_journey_rng: Optional[random.Random] = None,
) -> None:
    """Write a full historical timeline log for the year (10–20x longer, chronicle-style). All output goes to runs/ via logger."""
    def L(line: str, lvl: str = "normal") -> None:
        logger.emit(line, lvl)

    ctx = narrative_context if narrative_context is not None else {}
    ctx.setdefault("generational_left", 2)
    ctx.setdefault("historic_left", 5)
    ctx.setdefault("face_left", 3)

    stand_map = {s.team_id: s for s in (standings or [])}
    teams_list = _get_league_teams(league) if league is not None else []
    waiver_claims = list(getattr(uni_result, "waiver_claims", None) or []) if uni_result else []
    era_raw = (state.active_era or "").replace("_", " ").title() or "Modern"
    econ_climate = "Stable Expansion" if state.league_health >= 0.55 else "Cap Pressure" if state.league_health >= 0.40 else "Contraction Risk"

    # ---------- HEADER ----------
    L("")
    L("=================================================", "normal")
    L(f"NHL UNIVERSE — {year}", "normal")
    L("=================================================", "normal")
    L("")

    # ---------- LEAGUE STATE ----------
    L("LEAGUE STATE", "normal")
    L(f"Salary Cap: {fmt_money(state.salary_cap_m)}", "normal")
    L(f"Cap Growth: {state.cap_growth_rate:+.3f}", "normal")
    L(f"Parity Index: {state.parity_index:.3f}", "normal")
    L(f"Chaos Index: {state.chaos_index:.3f}", "normal")
    L(f"League Health: {state.league_health:.3f}", "normal")
    L(f"Era: {era_raw}", "normal")
    L(f"Economic Climate: {econ_climate}", "normal")
    L("")

    # ---------- LEAGUE TUNING REPORT ----------
    tr = getattr(state, "tuning_report", None) or {}
    if tr:
        L("LEAGUE TUNING REPORT", "normal")
        es = tr.get("era_start") or {}
        if es and isinstance(es, dict):
            prof = es.get("profile") or {}
            L(
                f"  Era modifiers: scoring x{float(prof.get('scoring_multiplier', 1.0)):.2f}  "
                f"defense eff x{float(prof.get('defense_effectiveness', 1.0)):.2f}  "
                f"goalie value x{float(prof.get('goalie_value', 1.0)):.2f}  "
                f"trade agg x{float(prof.get('trade_aggression', 1.0)):.2f}  "
                f"dev boost x{float(prof.get('prospect_growth_boost', 1.0)):.2f}",
                "normal",
            )
            L(
                f"  Era apply: players={es.get('players', 0)}  rating_keys_nudged={es.get('rating_keys_touched', 0)}",
                "normal",
            )
        pp = tr.get("post_progression") or {}
        if pp:
            nl = pp.get("normalize_league") or {}
            np = pp.get("normalize_players") or {}
            nt = pp.get("normalize_teams") or {}
            val = pp.get("validation") or {}
            L(
                f"  Chaos influence (prob spread): x{float(pp.get('chaos_influence', 1.0)):.2f}  "
                f"Role updates (tuning): {pp.get('role_updates', 0)}",
                "normal",
            )
            L(
                f"  Normalization: league d_chaos={nl.get('chaos_pull', 0):+.4f} d_parity={nl.get('parity_pull', 0):+.4f}  "
                f"players mean {np.get('mean_before', 0):.3f}->{np.get('mean_after', 0):.3f}  "
                f"teams boosted={nt.get('teams_boosted', 0)} trimmed={nt.get('teams_trimmed', 0)}",
                "normal",
            )
            lv = val.get("league") or {}
            if lv.get("issues"):
                L(f"  Validation (league): issues={lv.get('issues')} fixes={lv.get('fixes')}", "normal")
            td = val.get("teams") or []
            cor = sum(1 for x in td if x.get("fixes"))
            if cor:
                L(f"  Validation (teams): auto-corrected {cor} team(s) (roster/cap checks)", "normal")
            pd = val.get("players") or {}
            if pd.get("adjusted", 0) or pd.get("issues"):
                L(
                    f"  Validation (distribution): adjusted={pd.get('adjusted', 0)} issues={pd.get('issues', [])}",
                    "normal",
                )
        ey = tr.get("end_of_year") or {}
        if ey:
            L(
                f"  End-year tuning: league_norm={bool(ey.get('end_normalize_league'))}  "
                f"validation_issues={(ey.get('end_validation') or {}).get('issues', [])}",
                "normal",
            )
        L("")

    # ---------- PLAYER DISTRIBUTION REPORT ----------
    ds = tr.get("player_distribution_summary") if isinstance(tr, dict) else None
    if ds and int(ds.get("n_players", 0) or 0) > 0:
        L("PLAYER DISTRIBUTION REPORT", "normal")
        L(f"  Elite (85+): {ds.get('count_elite_85p', 0)}", "normal")
        L(f"  Top Line (78–84): {ds.get('count_top_line_78_85', 0)}", "normal")
        L(f"  Middle (70–77): {ds.get('count_middle_70_78', 0)}", "normal")
        L(f"  Depth (<70): {ds.get('count_bottom_under_70', 0)}", "normal")
        L(
            f"  Rating spread (NHL-scale proxy): min={ds.get('min_nhl_ovr', 0)} max={ds.get('max_nhl_ovr', 0)}  "
            f"stdev~{ds.get('std_ovr01', 0)} (OVR 0–1)",
            "normal",
        )
        pipe = tr.get("player_distribution_pipeline") if isinstance(tr, dict) else None
        if pipe and isinstance(pipe, dict):
            L(
                f"  Pipeline: variance_hits={pipe.get('variance_touches', 0)} elite_fix={pipe.get('elite_adjusted', 0)} "
                f"depth_fix={pipe.get('bottom_adjusted', 0)} tier_nudge={pipe.get('tier_adjusted', 0)}",
                "normal",
            )
        postn = tr.get("player_distribution_post_normalize") if isinstance(tr, dict) else None
        if postn and isinstance(postn, dict) and postn.get("widened"):
            L(
                f"  Post-normalize spread rescue: std {postn.get('std_before')} → {postn.get('std_after')}",
                "normal",
            )
        if isinstance(tr, dict) and tr.get("player_distribution_role_moves") is not None:
            L(f"  Percentile role assignments (moves): {tr.get('player_distribution_role_moves', 0)}", "normal")
        L("")

    # ---------- TEAM TUNING EFFECTS ----------
    if tr and teams_list:
        L("TEAM TUNING EFFECTS", "normal")
        for t in teams_list[:32]:
            tid = _team_id(t)
            arche = state.team_archetypes.get(tid, "balanced") if hasattr(state, "team_archetypes") else "balanced"
            tagg = float(getattr(t, "_tuning_trade_aggression", 1.0) or 1.0)
            L(f"  {_team_name(t)}: identity={arche}  trade_pressure_mult~x{tagg:.2f}", "normal")
        L("")

    id_evo = tr.get("identity_evolution") if isinstance(tr, dict) else None
    if id_evo:
        L("FRANCHISE IDENTITY EVOLUTION", "normal")
        for ln in id_evo[:56]:
            if isinstance(ln, str) and ln.strip():
                L(f"  {ln.strip()}", "normal")
        L("")

    # ---------- PLAYER TUNING EFFECTS ----------
    if tr and teams_list:
        L("PLAYER TUNING EFFECTS", "normal")
        shown = 0
        for t in teams_list:
            for p in list(getattr(t, "roster", None) or [])[:40]:
                if shown >= 12:
                    break
                pname = getattr(p, "name", getattr(getattr(p, "identity", None), "name", "Unknown"))
                role = getattr(p, "role", "") or "—"
                rn = getattr(p, "role_narrative", None) or role
                L(f"  {pname}: role={role} narrative={rn}", "normal")
                shown += 1
            if shown >= 12:
                break
        if shown == 0:
            L("  (no role data this year)", "normal")
        L("")

    # ---------- PLAYER STORYLINES ----------
    ps_lines = tr.get("player_storylines") if isinstance(tr, dict) else None
    if ps_lines:
        L("PLAYER STORYLINES", "normal")
        for rec in ps_lines[:72]:
            if not isinstance(rec, dict):
                continue
            nm = rec.get("player", "?")
            tn = rec.get("team", "?")
            ch = rec.get("character", "")
            pers = rec.get("personality", "") or "—"
            L(f"- {nm} ({tn})  character: {ch}  [{pers}]", "normal")
            slist = rec.get("storylines") or []
            pols = rec.get("storyline_polarities") or []
            atiers = rec.get("arc_tiers") or []
            for i, sl in enumerate(slist):
                ptyp = pols[i] if i < len(pols) else "neutral"
                at = atiers[i] if i < len(atiers) else ""
                tier_note = f"  arc_tier: {at}" if at else ""
                L(f"  storyline: \"{sl}\"", "normal")
                L(f"  storyline_type: {ptyp}{tier_note}", "normal")
            L(f"  effect: {rec.get('effect', '')}", "normal")
            L(f"  duration: {rec.get('duration', '')}", "normal")
        L("")

    # ---------- NARRATIVE CONSEQUENCES (systemic ripples) ----------
    ncons = tr.get("narrative_consequences") if isinstance(tr, dict) else None
    if ncons:
        L("NARRATIVE CONSEQUENCE", "normal")
        for nc in ncons[:72]:
            if not isinstance(nc, dict):
                continue
            pnm = nc.get("player_name", "?")
            et = nc.get("event_type", "generic")
            stx = nc.get("storyline_text", "") or et
            atier = nc.get("arc_tier", "") or ""
            suf = f"  arc_tier={atier}" if atier else ""
            L(f"- {pnm} — \"{stx}\"  [{et}]{suf}", "normal")
            pl = nc.get("player_line") or "(no extra player nudge)"
            tl = nc.get("team_line") or "—"
            ll = nc.get("league_line") or "—"
            rl = nc.get("role_line") or ""
            trd = nc.get("trade_line") or ""
            rip = nc.get("teammates_rippled", 0)
            L(f"  Player: {pl}", "normal")
            if rl:
                L(f"  Role: {rl}", "normal")
            if trd:
                L(f"  Trade: {trd}", "normal")
            L(f"  Team: {tl}", "normal")
            if rip:
                L(f"  Locker room ripple: {rip} teammates (morale nudge)", "normal")
            L(f"  League: {ll}", "normal")
        L("")

    # ---------- PLAYER ARCHETYPE (assignment sample) ----------
    pal = tr.get("player_archetype_logs") if isinstance(tr, dict) else None
    if pal:
        for ln in pal[:56]:
            if isinstance(ln, str) and ln.strip():
                L(ln.strip(), "normal")
        L("")

    # ---------- LINE CHEMISTRY REPORT ----------
    lc_rows = tr.get("line_chemistry") if isinstance(tr, dict) else None
    if lc_rows:
        L("LINE CHEMISTRY REPORT", "normal")
        for row in lc_rows[:96]:
            if not isinstance(row, dict):
                continue
            tnm = row.get("team", "?")
            unit = row.get("unit", "forwards")
            lbl = row.get("line", "?")
            chem = row.get("chemistry", 0.0)
            eff = row.get("effect", "")
            st = row.get("styles", "")
            note = row.get("note", "") or ""
            L(f"LINE CHEMISTRY: {lbl} ({tnm}) [{unit}]", "normal")
            L(f"    {st}", "normal")
            L(f"    Chemistry: {float(chem):.2f}", "normal")
            L(f"    Effect: {eff}", "normal")
            if note:
                L(f"    {note}", "normal")
        L("")

    # ---------- TEAM ECOSYSTEM SNAPSHOT ----------
    L("TEAM ECOSYSTEM SNAPSHOT", "normal")
    for s in (standings or [])[:32]:
        tid = getattr(s, "team_id", "")
        name = getattr(s, "team_name", tid)
        pts = getattr(s, "points", 0)
        bucket = getattr(s, "bucket", "bubble")
        pct = getattr(s, "point_pct", 0.50)
        arche = state.team_archetypes.get(tid, "balanced") if hasattr(state, "team_archetypes") else "balanced"
        strength = int(round(50 + (pct - 0.5) * 120)) if pct else 50
        strength = max(1, min(120, strength))
        cap_label = (getattr(state, "runner_cap_pressure_by_team", None) or {}).get(tid, "moderate")
        strat = (getattr(state, "runner_team_strategy_by_team", None) or {}).get(tid, "")
        cap_num = (getattr(state, "runner_cap_numeric_pressure_by_team", None) or {}).get(tid, None)
        window = "rebuild" if bucket == "rebuild" else "balanced" if bucket in ("bubble", "playoff") else "contender"
        traj = "lottery" if bucket == "rebuild" else "rising contender" if bucket == "contender" else "middle of pack"
        fwin = ""
        ftraj = ""
        for tm in teams_list:
            if _team_id(tm) == tid:
                ws = getattr(tm, "_runner_window_status", None)
                ft = getattr(tm, "_runner_franchise_trajectory", None)
                if ws:
                    fwin = f"  Franchise window: {ws}"
                if ft:
                    ftraj = f"  Franchise trajectory: {ft}"
                break
        L(f"  {name}", "normal")
        extra = f"  Strategy: {strat}" if strat else ""
        if cap_num is not None:
            extra = f"{extra}  CapIdx: {float(cap_num):.3f}" if extra else f"  CapIdx: {float(cap_num):.3f}"
        L(
            f"    Identity: {arche}  Record: {pts} pts  Strength: {strength}  Cap Pressure: {cap_label}  Window: {window}  Trajectory: {traj}{extra}",
            "normal",
        )
        if fwin:
            L(f"   {fwin.strip()}", "normal")
        if ftraj:
            L(f"   {ftraj.strip()}", "normal")
    L("")

    # ---------- PLAYER ECOSYSTEM SNAPSHOT (by lifecycle) ----------
    L("PLAYER ECOSYSTEM SNAPSHOT", "normal")
    all_ovrs: List[Tuple[Any, Any, float, int]] = []  # (player, team, ovr, age)
    for t in teams_list:
        roster = list(getattr(t, "roster", None) or [])
        for p in roster:
            ov = p.ovr() if callable(getattr(p, "ovr", None)) else float(getattr(p, "ovr", 0.5))
            age = int(_safe_getattr(getattr(p, "identity", None), "age", 26))
            all_ovrs.append((p, t, ov, age))
    all_ovrs.sort(key=lambda x: x[2], reverse=True)
    if all_ovrs:
        top5 = all_ovrs[:5]
        for p, t, ov, age in top5:
            pname = getattr(p, "name", getattr(getattr(p, "identity", None), "name", "Unknown"))
            tname = _team_name(t)
            L(f"  Elite: {pname} ({tname}) — {ov:.2f} OVR age {age}", "normal")
        under24 = [(p, t, ov, age) for p, t, ov, age in all_ovrs if age < 24]
        age25_30 = [(p, t, ov, age) for p, t, ov, age in all_ovrs if 25 <= age <= 30]
        vet31 = [(p, t, ov, age) for p, t, ov, age in all_ovrs if age >= 31]
        under24.sort(key=lambda x: x[2], reverse=True)
        age25_30.sort(key=lambda x: x[2], reverse=True)
        vet31.sort(key=lambda x: x[2], reverse=True)
        L("  Top under 24:", "normal")
        for p, t, ov, age in under24[:5]:
            pname = getattr(p, "name", getattr(getattr(p, "identity", None), "name", "Unknown"))
            L(f"    {pname} — {ov:.2f} OVR", "normal")
        L("  Top 25–30:", "normal")
        for p, t, ov, age in age25_30[:5]:
            pname = getattr(p, "name", getattr(getattr(p, "identity", None), "name", "Unknown"))
            L(f"    {pname} — {ov:.2f} OVR", "normal")
        L("  Top veterans 31+:", "normal")
        for p, t, ov, age in vet31[:5]:
            pname = getattr(p, "name", getattr(getattr(p, "identity", None), "name", "Unknown"))
            L(f"    {pname} — {ov:.2f} OVR", "normal")
        if len(all_ovrs) >= 15:
            mid = all_ovrs[len(all_ovrs)//2]
            p, t, ov, age = mid[0], mid[1], mid[2], mid[3]
            pname = getattr(p, "name", getattr(getattr(p, "identity", None), "name", "Unknown"))
            L(f"  Median roster player: {pname} — {ov:.2f} OVR", "normal")
    L("")

    # ---------- TRANSACTION WIRE ----------
    L("TRANSACTION WIRE", "normal")
    trades = [e for e in events if (getattr(e, "type", "") or "").upper() == "TRADE"]
    signings = [e for e in events if (getattr(e, "type", "") or "").upper() in ("SIGNING", "ECONOMY") and "sign" in (getattr(e, "headline", "") or "").lower()]
    waivers = [e for e in events if "waiver" in (getattr(e, "type", "") or "").lower() or "waiver" in (getattr(e, "headline", "") or "").lower()]
    for e in trades[:25]:
        L(f"  {getattr(e, 'headline', e)}", "normal")
    for e in signings[:20]:
        h = getattr(e, "headline", str(e))
        if h.upper().startswith("SIGNING:"):
            L(f"  {h}", "normal")
        else:
            L(f"  SIGNING: {h}", "normal")
    for e in waivers[:15]:
        L(f"  {getattr(e, 'headline', e)}", "normal")
    L("")

    # ---------- TEAM IDENTITY SHIFTS ----------
    L("TEAM IDENTITY SHIFTS", "normal")
    identity_ev = [e for e in events if "identity" in (getattr(e, "headline", "") or "").lower() or "drift" in (getattr(e, "headline", "") or "").lower() or "pivot" in (getattr(e, "headline", "") or "").lower()]
    for e in identity_ev[:15]:
        L(f"  {getattr(e, 'headline', e)}", "normal")
    if not identity_ev:
        L("  No major identity shifts this year.", "normal")
    L("")

    # ---------- COACHING EVENTS ----------
    L("COACHING EVENTS", "normal")
    coach_ev = [e for e in events if "coach" in (getattr(e, "headline", "") or "").lower() or getattr(e, "type", "") == "COACH_FIRE" or getattr(e, "type", "") == "COACH_HIRE"]
    for e in coach_ev[:15]:
        L(f"  {getattr(e, 'headline', e)}", "normal")
    if not coach_ev:
        L("  No coaching changes this year.", "normal")
    L("")

    # ---------- PROSPECT PIPELINES ----------
    L("PROSPECT PIPELINES", "normal")
    draft_ev = [e for e in events if (getattr(e, "type", "") or "").upper() == "DRAFT"]
    for e in draft_ev[:5]:
        L(f"  {getattr(e, 'headline', e)}", "normal")
    lottery_ev = [e for e in events if (getattr(e, "type", "") or "").upper() == "LOTTERY"]
    for e in lottery_ev[:3]:
        L(f"  {getattr(e, 'headline', e)}", "normal")
    L("")

    # ---------- FREE AGENCY MARKET ----------
    L("FREE AGENCY MARKET", "normal")
    for e in signings[:12]:
        L(f"  {getattr(e, 'headline', e)}", "normal")
    if not signings:
        L("  Free agency: moderate activity (see transaction wire).", "normal")
    L("")

    # ---------- TRADE MARKET SUMMARY ----------
    L("TRADE MARKET SUMMARY", "normal")
    if trades:
        categories: Dict[str, int] = {}
        for e in trades:
            det = getattr(e, "details", None) or {}
            cat = det.get("trade_category", "trade")
            categories[cat] = categories.get(cat, 0) + 1
        L(f"  Total trades: {len(trades)}", "normal")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            L(f"  {cat}: {count}", "normal")
        for e in trades[:15]:
            L(f"  • {getattr(e, 'headline', e)}", "normal")
    else:
        L("  Trade market: see TRADE ENGINE summary in year log.", "normal")
    L("")

    # ---------- WAIVER CLAIMS ----------
    L("WAIVER CLAIMS", "normal")
    for wc in waiver_claims[:15]:
        lbl = getattr(wc, "player_label", getattr(wc, "player_id", "Player"))
        to_t = getattr(wc, "to_team", "")
        fr_t = getattr(wc, "from_team", "")
        L(f"  {to_t} claims {lbl} from {fr_t}", "normal")
    if not waiver_claims:
        L("  No waiver claims this year.", "normal")
    L("")

    # ---------- PLAYER DEVELOPMENT EVENTS ----------
    L("PLAYER DEVELOPMENT EVENTS", "normal")
    break_ev = [e for e in events if "breakout" in (getattr(e, "headline", "") or "").lower() or "development" in (getattr(e, "headline", "") or "").lower()]
    for e in break_ev[:10]:
        L(f"  {getattr(e, 'headline', e)}", "normal")
    L("")

    # ---------- RETIREMENT SUMMARY ----------
    L("RETIREMENT SUMMARY", "normal")
    ret_ev = [e for e in events if (getattr(e, "type", "") or "").upper() == "RETIREMENT"]
    if ret_ev:
        ages: List[float] = []
        notables: List[str] = []
        stars = 0
        for e in ret_ev:
            det = getattr(e, "details", None) or {}
            age = det.get("age")
            if age is not None:
                ages.append(float(age))
            ovr = det.get("ovr") or 0
            if ovr >= 0.85:
                stars += 1
            pname = det.get("player", getattr(e, "headline", ""))
            if age is not None and (ovr >= 0.85 or age >= 36):
                notables.append(f"{pname} (age {int(age)})")
        L(f"  Total retirements: {len(ret_ev)}", "normal")
        if ages:
            L(f"  Average retirement age: {sum(ages)/len(ages):.1f}", "normal")
        if stars:
            L(f"  Star-level retirements: {stars}", "normal")
        if notables:
            L("  Notable: " + ", ".join(notables[:8]), "normal")
        for e in ret_ev[:20]:
            L(f"  • {getattr(e, 'headline', e)}", "normal")
    else:
        L("  No retirements reported this year.", "normal")
    L("")

    # ---------- EMERGENCE SUMMARY ----------
    L("EMERGENCE SUMMARY", "normal")
    break_ev = [e for e in events if "breakout" in (getattr(e, "headline", "") or "").lower() or "emergence" in (getattr(e, "headline", "") or "").lower()]
    young_elite: List[Tuple[Any, Any, float, int]] = []
    if all_ovrs:
        young_elite = [(p, t, ov, age) for p, t, ov, age in all_ovrs if age < 24 and ov >= 0.72]
        young_elite.sort(key=lambda x: x[2], reverse=True)
        for p, t, ov, age in young_elite[:5]:
            pname = getattr(p, "name", getattr(getattr(p, "identity", None), "name", "Unknown"))
            L(f"  Young impact: {pname} (age {age}) — {ov:.2f} OVR", "normal")
    for e in break_ev[:8]:
        L(f"  {getattr(e, 'headline', e)}", "normal")
    if not break_ev and (not all_ovrs or not young_elite):
        L("  No breakout/emergence events logged this year.", "normal")
    L("")

    # ---------- ERA AND LEAGUE IDENTITY ----------
    L("ERA AND LEAGUE IDENTITY", "normal")
    years_in_era = getattr(state, "years_in_era", 0)
    L(f"  Active era: {era_raw} (year {years_in_era})", "normal")
    if years_in_era >= 8:
        L("  Transition pressure rising (era duration); shift possible in coming years.", "normal")
    L(f"  Era rewards: {era_raw} style; roster and coach preferences follow league meta.", "normal")
    L("")

    # ---------- DYNASTY / POWER STRUCTURE ----------
    L("POWER STRUCTURE", "normal")
    power_states = getattr(state, "power_states", None) or {}
    cup_wins = getattr(state, "cup_wins_by_team", None) or {}
    dynasties = [tid for tid, ps in power_states.items() if ps == "dynasty"]
    powerhouses = [tid for tid, ps in power_states.items() if ps == "powerhouse"]
    repeat = [tid for tid, ps in power_states.items() if ps == "repeat_contender"]
    fragile = [tid for tid, ps in power_states.items() if ps == "fragile_contender"]
    tank = [tid for tid, ps in power_states.items() if ps == "tank_spiral"]
    stand_map = {s.team_id: s for s in (standings or [])}
    def _name(tid: str) -> str:
        s = stand_map.get(tid)
        return getattr(s, "team_name", tid) if s else tid
    if dynasties:
        L("  Dynasties / mini-dynasties: " + ", ".join(_name(t) for t in dynasties), "normal")
    if powerhouses:
        L("  Powerhouses: " + ", ".join(_name(t) for t in powerhouses), "normal")
    if repeat:
        L("  Repeat contenders: " + ", ".join(_name(t) for t in repeat[:5]), "normal")
    if fragile:
        L("  Fragile contenders: " + ", ".join(_name(t) for t in fragile[:5]), "normal")
    if tank:
        L("  Tank / rebuild spiral: " + ", ".join(_name(t) for t in tank[:5]), "normal")
    if not (dynasties or powerhouses or repeat or fragile or tank):
        L("  No dominant power clusters; league structure normal.", "normal")
    L("")

    # ---------- LIFECYCLE HEALTH ----------
    L("LIFECYCLE HEALTH", "normal")
    if all_ovrs:
        ages_only = [age for _, _, _, age in all_ovrs]
        avg_age = sum(ages_only) / len(ages_only)
        L(f"  Average league age: {avg_age:.1f}", "normal")
        under24_count = sum(1 for a in ages_only if a < 24)
        over30_count = sum(1 for a in ages_only if a >= 31)
        L(f"  Roster players under 24: {under24_count}; 31+: {over30_count}", "normal")
        top50_ages = [age for _, _, _, age in sorted(all_ovrs, key=lambda x: -x[2])[:50]]
        if top50_ages:
            L(f"  Elite pool (top 50) avg age: {sum(top50_ages)/len(top50_ages):.1f}", "normal")
        L("  League talent pool: healthy turnover from retirements and prospect pipeline." if ret_ev else "  League talent pool: stable; monitor for stagnation.", "normal")
    else:
        L("  No roster data for lifecycle metrics.", "normal")
    L("")

    # ---------- NARRATIVE EVENTS (player journeys, storylines, news) ----------
    if narrative_context is not None and league is not None and standings is not None:
        try:
            from app.sim_engine.narrative.player_journeys import update_player_journeys
            from app.sim_engine.narrative.storylines import update_storylines
            from app.sim_engine.narrative.news_feed import format_narrative_section
            journey_events = update_player_journeys(
                league, year, narrative_context, rng=narrative_journey_rng
            )
            _journey_val: Dict[str, int] = {}
            journey_events = _rebalance_journey_events_for_season_log(league, journey_events, _journey_val)
            _journey_val["rookie_headlines_logged"] = sum(
                1 for e in journey_events if e.get("tag") == "rookie_sensation"
            )
            narrative_context["last_journey_validation"] = dict(_journey_val)
            storyline_events = update_storylines(
                standings, playoff_champion_id, state, year, narrative_context
            )
            champ_id = playoff_champion_id or ""
            champ_name = (
                getattr(stand_map.get(champ_id), "team_name", None) or champ_id
                if champ_id and stand_map else ""
            )
            major = [e for e in events if getattr(e, "impact_score", 0) >= 0.60]
            narrative_lines = format_narrative_section(
                journey_events,
                storyline_events,
                major_headlines=major[:8],
                playoff_champion_name=champ_name or None,
                year=year,
                narrative_context=narrative_context,
                league=league,
            )
            for line in _sanitize_narrative_lines(narrative_lines, ctx):
                L(line, "normal")
        except Exception:
            pass

    # ---------- MAJOR HEADLINES (legacy) ----------
    major = [e for e in events if getattr(e, "impact_score", 0) >= 0.60]
    if major:
        L("MAJOR HEADLINES", "normal")
        for e in major[:10]:
            L(f"  • {getattr(e, 'headline', e)}", "normal")
        L("")

    # ---------- NARRATIVE BALANCE / VALIDATION (engine + journey log caps) ----------
    jv_nb: Dict[str, Any] = {}
    if isinstance(narrative_context, dict):
        jv_nb = dict(narrative_context.get("last_journey_validation") or {})
    nb = dict(tr.get("narrative_balance") or {}) if isinstance(tr, dict) else {}
    L("NARRATIVE BALANCE:", "normal")
    L(
        f"  major_arcs={nb.get('major_arcs', 0)}  mid_arcs={nb.get('mid_arcs', 0)}  "
        f"minor_events={nb.get('minor_events', 0)}  rookie_headlines={int(jv_nb.get('rookie_headlines_logged', 0))}  "
        f"suppressed_events={nb.get('suppressed_events', 0)}",
        "normal",
    )
    L("NARRATIVE VALIDATION:", "normal")
    L(
        f"  repeated_templates_trimmed={nb.get('repeated_templates_trimmed', 0)}  "
        f"major_arc_cooldowns_applied={nb.get('major_arc_cooldowns_applied', 0)}  "
        f"rookie_spam_trimmed={int(jv_nb.get('rookie_spam_trimmed', 0))}  "
        f"journey_tag_trimmed={int(jv_nb.get('journey_tag_trimmed', 0))}",
        "normal",
    )
    L("")

    # ---------- LEAGUE NARRATIVE ----------
    L("LEAGUE NARRATIVE", "normal")
    top = standings[:2] if standings else []
    bottom = list(reversed(standings))[:2] if standings else []
    champ_id = playoff_champion_id or ""
    champ_name = (getattr(stand_map.get(champ_id), "team_name", champ_id) if stand_map.get(champ_id) else champ_id) if champ_id else ""
    top_names = [getattr(s, "team_name", getattr(s, "team_id", "")) for s in top]
    bottom_names = [getattr(s, "team_name", getattr(s, "team_id", "")) for s in bottom]
    narrative = f"The {year} season saw {era_raw} as the dominant style. "
    if top_names:
        narrative += f"{' and '.join(top_names)} led the league; "
    if bottom_names:
        narrative += f"{', '.join(bottom_names)} struggled. "
    if champ_name:
        narrative += f"{champ_name} captured the Stanley Cup. "
    narrative += f"Parity stood at {state.parity_index:.2f}; league health at {state.league_health:.2f}."
    L(narrative, "normal")
    L("")

    # ---------- CLOSING SEPARATOR ----------
    L("=================================================", "normal")
    L("", "normal")

# =============================================================================
# Regression harness (compare fingerprints)
# =============================================================================

def compute_run_fingerprint(run_dir: Path) -> Dict[str, Any]:
    """
    Produce a deterministic fingerprint of run artifacts in this folder.
    Intended for regression mode comparisons.
    """
    fp: Dict[str, Any] = {"files": {}}
    for path in sorted(run_dir.glob("*")):
        if path.is_file():
            b = path.read_bytes()
            fp["files"][path.name] = {"sha256": sha256_bytes(b), "bytes": len(b)}
    fp["overall_sha256"] = sha256_bytes(stable_json_dumps(fp, pretty=False).encode("utf-8"))
    return fp

def regression_compare_or_write(run_cfg: RunConfig, out: RunOutput, logger: RunnerLogger) -> None:
    base_dir = Path(run_cfg.regression_baseline_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = base_dir / f"baseline_seed_{run_cfg.seed}.json"

    fp = compute_run_fingerprint(out.run_dir)

    if baseline_path.exists():
        old = json.loads(baseline_path.read_text(encoding="utf-8"))
        if old.get("overall_sha256") == fp.get("overall_sha256"):
            logger.emit(f"[REGRESSION] PASS — fingerprint matches baseline for seed {run_cfg.seed}", "normal")
        else:
            logger.emit(f"[REGRESSION] FAIL — fingerprint mismatch for seed {run_cfg.seed}", "normal")
            logger.emit(f"Old: {old.get('overall_sha256')}", "normal")
            logger.emit(f"New: {fp.get('overall_sha256')}", "normal")
            # write diff hint
            out.write_json("regression_fingerprint_new.json", fp)
            out.write_json("regression_fingerprint_old.json", old)
    else:
        baseline_path.write_text(stable_json_dumps(fp, pretty=True), encoding="utf-8")
        logger.emit(f"[REGRESSION] Baseline created for seed {run_cfg.seed}: {baseline_path}", "normal")

# =============================================================================
# Main runner orchestration
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NHL Franchise Mode — Universe Generator + Career Runner (NEW)")
    p.add_argument("--seed", type=int, default=None, help="Master seed (default: time_ns).")
    p.add_argument("--years", type=int, default=40, help="Years to simulate.")
    p.add_argument("--start_year", type=int, default=2025, help="Start year.")
    p.add_argument("--mode", type=str, default="combined", choices=RUN_MODES, help="Run mode.")
    p.add_argument("--output_dir", type=str, default="runs", help="Output directory root.")
    p.add_argument("--log_level", type=str, default="normal", choices=LOG_LEVELS, help="Logging verbosity.")
    p.add_argument("--no_json", action="store_true", help="Disable JSON outputs.")
    p.add_argument("--dump_state", action="store_true", help="Write heavy debug_state_<year>.json dumps.")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    p.add_argument("--debug", action="store_true", help="Enable engine debug (if supported).")
    p.add_argument("--scenario_file", type=str, default=None, help="Path to scenario JSON.")
    # universe knobs (some)
    p.add_argument("--chaos", type=float, default=None, help="Override chaos target (0-1).")
    p.add_argument("--parity", type=float, default=None, help="Override parity target (0-1).")
    p.add_argument("--cap_start", type=float, default=None, help="Override salary cap start (millions).")
    p.add_argument("--cap_growth", type=float, default=None, help="Override cap growth mean.")
    p.add_argument("--disable_trades", action="store_true", help="Disable universe trades.")
    p.add_argument("--disable_fa", action="store_true", help="Disable free agency.")
    p.add_argument("--disable_draft", action="store_true", help="Disable draft/lottery.")
    p.add_argument("--disable_waivers", action="store_true", help="Disable waivers.")
    p.add_argument("--disable_coaches", action="store_true", help="Disable coach carousel.")
    p.add_argument("--dump_rng_traces", action="store_true", help="Dump RNG traces into debug_state.")
    p.add_argument("--print_trade_eval", action="store_true", help="Print trade evaluation breakdown (debug/insane recommended).")
    p.add_argument("--print_team_decisions", action="store_true", help="Print team decision reasons (debug/insane).")
    return p

def run_simulation_core(
    run_cfg: RunConfig,
    uni_cfg: UniverseConfig,
    scn_cfg: ScenarioConfig,
    league: Any,
    sim: Any,
    ustate: UniverseState,
    narrative_context: Dict[str, Any],
    logger: RunnerLogger,
    out: RunOutput,
    last_context: Dict[str, Any],
) -> UniverseState:
    """
    Multi-year simulation loop shared with main(). Mutates league/sim/narrative_context/last_context.
    Returns final UniverseState. Optional run_cfg.per_year_callback receives keyword args each year.
    """
    for i in range(run_cfg.years):
        year = run_cfg.start_year + i
        last_context["year"] = year

        # Universe + career pipeline in proper order
        uni_result: Optional[UniverseYearResult] = None
        career_result: Optional[CareerYearResult] = None
        league_season_result: Any = None

        # UNIVERSE
        if run_cfg.mode in ("combined", "universe_only", "regression"):
            last_context["phase"] = "universe_year"
            year_rng = rng_from_seed(split_seed(run_cfg.seed, f"universe::{year}"))
            uni_result = simulate_universe_year(
                league=league,
                state=ustate,
                run_cfg=run_cfg,
                uni_cfg=uni_cfg,
                scn_cfg=scn_cfg,
                year=year,
                logger=logger,
                rng=year_rng,
                sim=sim,
                narrative_context=narrative_context,
            )
            ustate = uni_result.state

            # Print NHL universe block (full historical timeline log to runs/)
            print_year_header(
                logger, year, ustate, uni_result.standings, uni_result.events, run_cfg.log_level,
                league=league, uni_result=uni_result,
                playoff_champion_id=getattr(uni_result, "playoff_champion_id", None),
                narrative_context=narrative_context,
                narrative_journey_rng=rng_from_seed(split_seed(run_cfg.seed, f"journeys::{year}")),
            )

            # Prospect development: age prospects, skill growth, cap pool (before draft)
            if sim is not None and uni_cfg.enable_draft:
                progress_fn = getattr(sim, "progress_prospects", None)
                if callable(progress_fn):
                    _lg_pr = getattr(sim, "league", None) or league
                    try:
                        setattr(_lg_pr, "_promotion_team_archetypes", dict(ustate.team_archetypes or {}))
                    except Exception:
                        setattr(_lg_pr, "_promotion_team_archetypes", {})
                    progress_fn()
                    _pbuf = list(getattr(sim, "_pipeline_log_buffer", None) or [])
                    for _ln in _pbuf[:180]:
                        logger.emit(_ln, "normal")
                    try:
                        sim._pipeline_log_buffer = []
                    except Exception:
                        pass

            # Prospect pipeline: structured draft tiers via generation.draft_class_generator; SimEngine.generate_prospect_class
            # consumes pipeline_boost_elite when league._boost_next_draft_class (set by validators on thin pools).
            if sim is not None and uni_result.standings and uni_cfg.enable_draft:
                draft_seed_i = draft_lottery_seed(run_cfg.seed, year)
                draft_rng = rng_from_seed(draft_seed_i)
                n_prospects = 0
                n_drafted = 0
                promoted_ages: List[int] = []
                promoted_potentials: List[float] = []
                try:
                    _league_d = getattr(sim, "league", None) or league
                    _teams_d = getattr(_league_d, "teams", None) or []
                    pool_before = total_prospect_pipeline(_teams_d)
                    if sim is not None:
                        cdm = getattr(sim, "compute_league_age_distribution", None)
                        if callable(cdm):
                            try:
                                dm = _normalize_age_stats_from_engine(cdm())
                            except Exception:
                                dm = league_age_distribution_stats(_teams_d)
                        else:
                            dm = league_age_distribution_stats(_teams_d)
                    else:
                        dm = league_age_distribution_stats(_teams_d)
                    setattr(_league_d, "_age_balance_promotion", dm)
                    setattr(_league_d, "_age_balance_prune", dm)
                    try:
                        setattr(_league_d, "_promotion_team_archetypes", dict(ustate.team_archetypes or {}))
                    except Exception:
                        setattr(_league_d, "_promotion_team_archetypes", {})
                    gen_prospects = getattr(sim, "generate_prospect_class", None)
                    if callable(gen_prospects):
                        n_prospects = gen_prospects(year, draft_rng)
                        top_ovr = getattr(sim, "_last_draft_class_top_ovr", 0.0)
                        logger.emit("", "normal")
                        logger.emit("DRAFT CLASS GENERATED", "normal")
                        logger.emit(f"  Prospects: {n_prospects}", "normal")
                        logger.emit(f"  Top Prospect OVR: {top_ovr:.2f}", "normal")
                        logger.emit("", "normal")
                        _gbuf = list(getattr(sim, "_pipeline_log_buffer", None) or [])
                        for _gl in _gbuf[:220]:
                            logger.emit(_gl, "normal")
                        try:
                            sim._pipeline_log_buffer = []
                        except Exception:
                            pass
                    non_playoff = [(s.team_id, s.points) for s in sorted(uni_result.standings, key=lambda s: (s.points, getattr(s, "goal_diff", 0)))[:16]]
                    run_draft = getattr(sim, "run_universe_draft", None)
                    if callable(run_draft):
                        full_order = list(
                            getattr(uni_result, "full_draft_order_32", None)
                            or getattr(_league_d, "_runner_full_draft_order_32", None)
                            or []
                        )
                        draft_result = run_draft(
                            non_playoff,
                            year,
                            draft_rng,
                            standings=uni_result.standings,
                            full_team_order=full_order if full_order else None,
                            draft_seed=draft_seed_i,
                        )
                        if isinstance(draft_result, tuple) and len(draft_result) >= 3:
                            n_drafted, promoted_ages, promoted_potentials = draft_result[0], draft_result[1], draft_result[2]
                        else:
                            n_drafted = draft_result if not isinstance(draft_result, tuple) else (draft_result[0] if draft_result else 0)
                        raw = getattr(sim, "last_draft_results", None)
                        results = ((raw.get("results") if isinstance(raw, dict) else raw) or []) if n_prospects > 0 else []
                        # === NHL ENTRY DRAFT === logging
                        logger.emit("", "normal")
                        logger.emit("=== NHL ENTRY DRAFT ===", "normal")
                        lot = getattr(sim, "last_draft_lottery", None)
                        if lot and getattr(lot, "lottery_winners", None):
                            w = lot.lottery_winners
                            if len(w) >= 1:
                                logger.emit(f"Lottery Winner (1st): {w[0]}", "normal")
                            if len(w) >= 2:
                                logger.emit(f"2nd Pick: {w[1]}", "normal")
                        for rec in results[:10]:
                            pick = rec.get("pick", 0)
                            tid = rec.get("team_id", "?")
                            name = rec.get("prospect_name", "Unknown")
                            payload = rec.get("player_payload") or {}
                            proj = (payload.get("projection") or {})
                            dvr = payload.get("draft_value_range") or (0.5, 0.6)
                            pot = (float(dvr[0]) + float(dvr[1])) / 2.0 if len(dvr) >= 2 else 0.55
                            pos = "?"
                            try:
                                ident = (payload.get("identity") or {})
                                pos = str(ident.get("position", "?"))
                                if hasattr(pos, "value"):
                                    pos = getattr(pos, "value", pos)
                            except Exception:
                                pass
                            logger.emit(f"  {pick}. {tid} -> {pos} {name} (POT {pot:.2f})", "normal")
                        logger.emit("", "normal")
                        logger.emit("DRAFT COMPLETE", "normal")
                        logger.emit(f"  Players Drafted (to pipeline): {len(results)}", "normal")
                        logger.emit("", "normal")
                        logger.emit("ROOKIE CLASS ENTERING NHL", "normal")
                        logger.emit(f"  Promoted prospects: {n_drafted}", "normal")
                        if promoted_ages:
                            logger.emit(f"  Average rookie age: {sum(promoted_ages) / len(promoted_ages):.1f}", "normal")
                        if promoted_potentials:
                            logger.emit(f"  Top rookie potential: {max(promoted_potentials):.2f}", "normal")
                        pool_sizes = []
                        pool_pot = []
                        try:
                            _league = getattr(sim, "league", None) or league
                            for t in getattr(_league, "teams", None) or []:
                                pool = getattr(t, "prospect_pool", None) or []
                                pool_sizes.append(len(pool))
                                for p in pool:
                                    dr = getattr(p, "draft_value_range", (0.5, 0.5))
                                    if dr and len(dr) >= 2:
                                        pool_pot.append((float(dr[0]) + float(dr[1])) / 2.0)
                        except Exception:
                            pass
                        if pool_sizes:
                            logger.emit(f"  Team prospect pool size: min={min(pool_sizes)}, max={max(pool_sizes)}, total={sum(pool_sizes)}", "normal")
                        if pool_pot:
                            logger.emit(f"  Average prospect potential: {sum(pool_pot)/len(pool_pot):.2f}", "normal")
                        logger.emit("", "normal")
                        pc = getattr(sim, "_last_promotion_control", None) or getattr(_league_d, "_last_promotion_control", None)
                        if isinstance(pc, dict) and pc:
                            logger.emit("PROMOTION CONTROL:", "normal")
                            logger.emit(
                                f"  target={pc.get('target', '?')} actual={pc.get('actual', '?')} reason={pc.get('reason', 'retirements/roster_need')}",
                                "normal",
                            )
                            if pc.get("cap_applied"):
                                logger.emit("PROMOTION CAP APPLIED:", "normal")
                                logger.emit(f"  limited_to={pc.get('actual', '?')}", "normal")
                        logger.emit("", "normal")
                        drafted_ct = len(results)
                        pool_after = total_prospect_pipeline(getattr(getattr(sim, "league", None) or league, "teams", None) or [])
                        logger.emit(
                            f"PIPELINE CHECK: drafted={drafted_ct} promoted={n_drafted} pool_before={pool_before} pool_after={pool_after}",
                            "normal",
                        )
                        try:
                            _eco_lines = getattr(sim, "ecosystem_operational_repairs", None)
                            if callable(_eco_lines):
                                for _el in _eco_lines(
                                    getattr(_league_d, "teams", None) or [], draft_rng, year
                                )[-18:]:
                                    logger.emit(f"  ECOSYSTEM / PIPELINE HEALTH: {_el}", "normal")
                        except Exception:
                            pass
                        if n_drafted < 2 and n_prospects >= 120:
                            logger.emit(
                                "  [ECOSYSTEM NOTE] NHL rookie promotions below typical floor — "
                                "U24% and promotion caps may be binding; watch PROMOTION CONTROL.",
                                "normal",
                            )
                        try:
                            _ltm = getattr(_league, "teams", None) or []
                            _sn = 0
                            _tn = 0
                            for _tm in _ltm:
                                _psz = len(getattr(_tm, "prospect_pool", None) or [])
                                _tidz = getattr(_tm, "team_id", "?")
                                if _psz >= 10 and _sn < 10:
                                    logger.emit(
                                        f"PIPELINE STATUS: team={_tidz} pipeline depth: {_psz} prospects (healthy)",
                                        "normal",
                                    )
                                    _sn += 1
                                elif _psz < 6 and _tn < 12:
                                    logger.emit(
                                        f"PIPELINE STATUS: team={_tidz} pipeline depth: {_psz} prospects (thin)",
                                        "normal",
                                    )
                                    _tn += 1
                        except Exception:
                            pass
                        _post_draft_buf = list(getattr(sim, "_pipeline_log_buffer", None) or [])
                        for _pdl in _post_draft_buf[:240]:
                            logger.emit(_pdl, "normal")
                        try:
                            sim._pipeline_log_buffer = []
                        except Exception:
                            pass
                        try:
                            _gpp_n = getattr(_league, "global_player_pool", None) or getattr(
                                _league, "global_prospect_pool", None
                            )
                            if isinstance(_gpp_n, list) and _gpp_n:
                                logger.emit(
                                    f"GLOBAL POOL POST-DRAFT: remaining_unassigned={len(_gpp_n)}",
                                    "normal",
                                )
                                _sm10 = getattr(_league, "_last_draft_class_strength_mean10", None)
                                if _sm10 is not None:
                                    logger.emit(
                                        f"  Draft strength (mean top-10 ceiling): {_sm10:.3f}",
                                        "normal",
                                    )
                        except Exception:
                            pass
                        if n_drafted > pool_before + drafted_ct + 2:
                            logger.emit(
                                f"PIPELINE CHECK WARN: promoted={n_drafted} exceeds pipeline+draft headroom (pool_before={pool_before}, drafted={drafted_ct})",
                                "normal",
                            )
                        if n_drafted == 0 and len(results) > 0:
                            logger.emit("[INFO] Prospects in pipeline; promotion when development_years_remaining<=0, age>=22, or (potential>=0.80 and age>=19).", "normal")
                        elif n_drafted == 0 and not results:
                            logger.emit("[WARN] Draft produced 0 players. Check team_id alignment and league.teams.", "normal")
                        # Validation: each team should have up to 7 picks; all drafted prospects have team_id
                        try:
                            per_team: Dict[str, int] = {}
                            for rec in results:
                                tid = rec.get("team_id")
                                if tid is not None:
                                    per_team[str(tid)] = per_team.get(str(tid), 0) + 1
                            n_teams = len(per_team)
                            if results and n_teams > 0:
                                picks_per = list(per_team.values())
                                if max(picks_per) > 7 or (len(results) < 32 and min(picks_per) < 7):
                                    logger.emit(f"  [CHECK] Draft picks per team: min={min(picks_per)}, max={max(picks_per)} (expected up to 7 each)", "normal")
                        except Exception:
                            pass
                except Exception as e:
                    logger.emit(f"[WARN] Prospect/draft pipeline failed: {type(e).__name__}: {e}", "normal")
                # Track consecutive years with no rookies for critical warning
                last_zero_rookies = getattr(sim, "_last_zero_rookies_year", None)
                if n_drafted == 0 and last_zero_rookies is not None and year == last_zero_rookies + 1:
                    logger.emit("[CRITICAL] Multiple seasons with 0 rookies entering league. Prospect pipeline may be broken.", "normal")
                if n_drafted == 0:
                    try:
                        sim._last_zero_rookies_year = year
                    except Exception:
                        pass
                elif n_drafted > 0:
                    try:
                        setattr(sim, "_last_zero_rookies_year", None)
                    except Exception:
                        pass

            # League talent metrics (diagnostics each season)
            if sim is not None and uni_result is not None:
                try:
                    metrics_fn = getattr(sim, "get_league_talent_metrics", None)
                    if callable(metrics_fn) and league is not None:
                        ovr_stats = metrics_fn(league)
                        trades = [e for e in uni_result.events if (getattr(e, "type", "") or "").upper() == "TRADE"]
                        signings = [e for e in uni_result.events if (getattr(e, "type", "") or "").upper() == "SIGNING"]
                        draft_strength = getattr(sim, "_last_draft_class_top_ovr", 0.0)
                        logger.emit("", "normal")
                        logger.emit("LEAGUE TALENT METRICS", "normal")
                        logger.emit(f"  Top Player OVR:    {ovr_stats.get('top_ovr', 0):.3f}", "normal")
                        logger.emit(f"  Top 10 Avg:        {ovr_stats.get('top_10_avg', 0):.3f}", "normal")
                        logger.emit(f"  Top 50 Avg:        {ovr_stats.get('top_50_avg', 0):.3f}", "normal")
                        logger.emit(f"  League Mean:       {ovr_stats.get('mean_ovr', 0):.3f}", "normal")
                        logger.emit(f"  League Median:     {ovr_stats.get('median_ovr', 0):.3f}", "normal")
                        logger.emit(f"  Draft Class Strength: {draft_strength:.3f}", "normal")
                        logger.emit(f"  Trades:            {len(trades)}", "normal")
                        logger.emit(f"  Free Agent Signings: {len(signings)}", "normal")
                        logger.emit("", "normal")
                except Exception:
                    pass

            # Write universe JSON
            if run_cfg.write_json:
                out.write_json(
                    f"universe_year_{year}.json",
                    {
                        "year": year,
                        "economics": {
                            "salary_cap_m": ustate.salary_cap_m,
                            "cap_growth_rate": ustate.cap_growth_rate,
                            "inflation_factor": ustate.inflation_factor,
                            "league_health_score": ustate.league_health,
                        },
                        "meta": {
                            "era": ustate.active_era,
                            "parity_index": ustate.parity_index,
                            "chaos_index": ustate.chaos_index,
                        },
                        "standings": [safe_to_primitive(s) for s in uni_result.standings],
                        "transactions": {
                            "events": [safe_to_primitive(e) for e in uni_result.events],
                            "waived_count": uni_result.waived_count,
                            "waiver_claims": [safe_to_primitive(c) for c in uni_result.waiver_claims],
                        },
                        "draft": {
                            "lottery_results": uni_result.lottery_results,
                            "pick_order": uni_result.draft_pick_order,
                        },
                        "waiver_priority": ustate.waiver_priority,
                    },
                )

        # CAREER
        if run_cfg.mode in ("combined", "career_only", "regression"):
            last_context["phase"] = "career_year"
            career_result = run_career_year(sim, year, run_cfg, logger)
            if run_cfg.write_json:
                out.write_json(
                    f"career_year_{year}.json",
                    {
                        "year": year,
                        "player_id": career_result.player_id,
                        "team_id": career_result.team_id,
                        "summary": safe_to_primitive(career_result.summary),
                    },
                )

        # LEAGUE (structural season using engine.league *)
        if sim is not None and hasattr(sim, "simulate_league_season") and run_cfg.mode in ("combined", "career_only", "regression"):
            last_context["phase"] = "league_season"
            league_rng = rng_from_seed(split_seed(run_cfg.seed, f"league::{year}"))
            try:
                league_season_result = sim.simulate_league_season(year, league_rng)  # type: ignore
            except Exception:
                league_season_result = None
            if league_season_result is not None:
                _emit_world_simulation_report(logger, league, year)
            if league_season_result is not None and run_cfg.write_json:
                # Minimal JSON-friendly dump; detailed structures live inside engine.
                out.write_json(
                    f"league_season_{year}.json",
                    {
                        "year": year,
                        "standings": league_season_result.standings.as_table_rows(),
                        "playoffs": {
                            "champion_id": getattr(getattr(league_season_result, "playoff_result", None), "champion_id", None),
                        },
                        "awards": {
                            name: {
                                "winner_team_id": award.winner_team_id,
                                "winner_name": award.winner_name,
                                "rationale": award.rationale,
                            }
                            for name, award in league_season_result.awards.items()
                        },
                    },
                )

        # Heavy debug dump (optional)
        if run_cfg.write_debug_state:
            last_context["phase"] = "debug_dump"
            dump: Dict[str, Any] = {
                "year": year,
                "run_config": safe_to_primitive(run_cfg),
                "universe_config": safe_to_primitive(uni_cfg),
                "universe_state": safe_to_primitive(ustate),
                "last_context": safe_to_primitive(last_context),
            }
            if uni_cfg.dump_rng_traces:
                dump["rng_traces"] = safe_to_primitive(ustate.rng_traces[-200:])
            # include league/sim snapshots best-effort
            dump["league"] = safe_to_primitive(league)
            dump["sim"] = safe_to_primitive(sim)
            out.write_json(f"debug_state_{year}.json", dump)

        try:
            _pycb = getattr(run_cfg, "per_year_callback", None)
            if _pycb is not None:
                _pycb(
                    year=year,
                    ustate=ustate,
                    uni_result=uni_result,
                    career_result=career_result,
                    league_season_result=league_season_result,
                    sim=sim,
                    league=league,
                )
        except Exception:
            pass

        if run_cfg.flush_each_year:
            logger.flush()

    return ustate


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else int(_safe_getattr(time_module := __import__("time"), "time_ns")())
    run_cfg = RunConfig(
        seed=seed,
        years=int(args.years),
        start_year=int(args.start_year),
        debug=bool(args.debug),
        mode=str(args.mode),
        log_level=str(args.log_level),
        output_dir=str(args.output_dir),
        write_json=not bool(args.no_json),
        write_debug_state=bool(args.dump_state),
        pretty_print=bool(args.pretty),
        flush_each_year=True,
    )

    uni_cfg = UniverseConfig()
    # apply CLI overrides
    if args.chaos is not None:
        uni_cfg.chaos = clamp(float(args.chaos), 0.0, 1.0)
        uni_cfg.chaos_index = uni_cfg.chaos
    if args.parity is not None:
        uni_cfg.parity_target = clamp(float(args.parity), 0.0, 1.0)
    if args.cap_start is not None:
        uni_cfg.salary_cap_start = float(args.cap_start)
    if args.cap_growth is not None:
        uni_cfg.cap_growth_rate_mean = float(args.cap_growth)

    if args.disable_trades:
        uni_cfg.enable_trades = False
    if args.disable_fa:
        uni_cfg.enable_free_agency = False
    if args.disable_draft:
        uni_cfg.enable_draft = False
    if args.disable_waivers:
        uni_cfg.enable_waivers = False
    if args.disable_coaches:
        uni_cfg.enable_coach_changes = False

    uni_cfg.dump_rng_traces = bool(args.dump_rng_traces)
    uni_cfg.print_trade_eval = bool(args.print_trade_eval)
    uni_cfg.print_team_decision_reasons = bool(args.print_team_decisions)

    # scenarios
    print(">>> run_sim.py STARTED")
    try:
        scn_cfg = load_scenario_file(args.scenario_file)
    except Exception as e:
        print(f"[FATAL] Failed to load scenario file: {type(e).__name__}: {e}")
        return 2

    # output folder: always backend/runs/<run_folder>/ (never under app/)
    _this_file = Path(__file__).resolve()
    _base_dir = _this_file.parent
    if _base_dir.name == "app":
        _base_dir = _base_dir.parent
    runs_dir = _base_dir / run_cfg.output_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_folder_name = f"{now_utc_compact()}_{run_cfg.seed}"
    run_dir = runs_dir / run_folder_name
    timeline_path = run_dir / "timeline_log.txt"
    out = RunOutput(run_dir=run_dir, pretty=run_cfg.pretty_print, timeline_path=timeline_path)
    out.open()
    logger = RunnerLogger(out=out, log_level=run_cfg.log_level, flush_each_year=run_cfg.flush_each_year)
    _rel = f"{run_cfg.output_dir}/{run_folder_name}"
    print(f">>> Run folder: {_rel}")
    print(f">>> Writing output to {_rel}/timeline_log.txt")

    # Save config upfront
    out.write_json("run_config.json", {"run": safe_to_primitive(run_cfg), "universe": safe_to_primitive(uni_cfg), "scenario": safe_to_primitive(scn_cfg)})
    if SIMENGINE_IMPORT_ERROR:
        out.write_json("engine_import_warning.json", {"SimEngine_import_error": SIMENGINE_IMPORT_ERROR})
    if DRAFTLOTTERY_IMPORT_ERROR:
        out.write_json("draft_lottery_import_warning.json", {"DraftLottery_import_error": DRAFTLOTTERY_IMPORT_ERROR})
    if WAIVERS_IMPORT_ERROR:
        out.write_json("waivers_import_warning.json", {"Waivers_import_error": WAIVERS_IMPORT_ERROR})

    # master rng + split rngs
    master_rng = rng_from_seed(run_cfg.seed)
    uni_rng = rng_from_seed(split_seed(run_cfg.seed, "universe"))
    career_rng = rng_from_seed(split_seed(run_cfg.seed, "career"))
    meta_rng = rng_from_seed(split_seed(run_cfg.seed, "meta"))

    # Instantiate league + engine (defensive)
    sim = None
    league = None

    # Try to build SimEngine if available
    if SimEngine is not None:
        try:
            # Prefer SimEngine(seed=..., debug=...)
            try:
                sim = SimEngine(seed=run_cfg.seed, debug=run_cfg.debug)  # type: ignore
            except TypeError:
                sim = SimEngine(seed=run_cfg.seed)  # type: ignore
        except Exception as e:
            sim = None
            logger.emit(f"[WARN] SimEngine init failed: {type(e).__name__}: {e}", "normal")
            out.append_error(traceback.format_exc())

    # === AUTO-BOOTSTRAP LEAGUE IF EMPTY (SAFE ADDITION) ===
    try:
        league_obj = getattr(sim, "league", None)

        if league_obj is None:
            print("[BOOTSTRAP] SimEngine has no league attribute.")
        else:
            teams = None

            # Try multiple safe access patterns
            if hasattr(league_obj, "teams"):
                teams = league_obj.teams
            elif hasattr(league_obj, "get_teams"):
                try:
                    teams = league_obj.get_teams()
                except Exception:
                    teams = None
            elif hasattr(league_obj, "team_map"):
                teams = list(league_obj.team_map.values())

            # If still empty, attempt generation
            if not teams:
                print("[BOOTSTRAP] League detected but no teams found. Attempting generation bootstrap...")

                # Attempt known generation hooks safely
                if hasattr(league_obj, "generate_default_teams"):
                    league_obj.generate_default_teams()
                    print("[BOOTSTRAP] Called league.generate_default_teams()")

                elif hasattr(sim, "generate_league"):
                    sim.generate_league()
                    print("[BOOTSTRAP] Called sim.generate_league()")

                elif hasattr(sim, "setup_league"):
                    sim.setup_league()
                    print("[BOOTSTRAP] Called sim.setup_league()")

                else:
                    print("[BOOTSTRAP] No known generation hook found. League remains empty.")

    except Exception as bootstrap_error:
        print(f"[BOOTSTRAP ERROR] {bootstrap_error}")

    # Try to locate league from sim
    if sim is not None:
        league = _safe_getattr(sim, "league", None) or _safe_getattr(sim, "universe", None)
        if league is not None:
            try:
                setattr(league, "_runner_sim_engine", sim)
            except Exception:
                pass

    # If no league available, runner still works in limited mode (career-only may still run)
    teams = _get_league_teams(league)
    if not teams and run_cfg.mode in ("combined", "universe_only", "regression"):
        logger.emit("[WARN] League teams not found. Universe layer will be minimal unless your engine exposes league teams.", "normal")

    # Setup player career thread (runner-only placeholders; engine may override)
    if sim is not None:
        # if engine has player already, keep it; else assign a minimal placeholder if supported
        if _safe_getattr(sim, "player", None) is None and hasattr(sim, "__dict__"):
            sim.__dict__["player"] = _create_random_player(career_rng)
        if _safe_getattr(sim, "team", None) is None:
            pick = _pick_player_team(teams, career_rng)
            if pick is not None and hasattr(sim, "__dict__"):
                sim.__dict__["team"] = pick

    # Initialize universe state
    if not teams:
        # still init with a synthetic identity map
        arche, coach_ids, coach_sec = {}, {}, {}
    else:
        arche, coach_ids, coach_sec = _init_team_identities(teams, meta_rng)

    ustate = UniverseState(
        salary_cap_m=float(uni_cfg.salary_cap_start),
        cap_growth_rate=float(uni_cfg.cap_growth_rate_mean),
        inflation_factor=1.0,
        league_health=clamp(uni_cfg.league_health_target + meta_rng.uniform(-0.05, 0.05), 0.10, 0.95),
        parity_index=clamp(uni_cfg.parity_target + meta_rng.uniform(-0.05, 0.05), 0.10, 0.90),
        chaos_index=clamp(uni_cfg.chaos + meta_rng.uniform(-0.05, 0.05), 0.10, 0.95),
        active_era=_choose_era(meta_rng),
        waiver_priority=[_team_id(t) for t in teams] if teams else [],
        team_archetypes=arche,
        coach_ids=coach_ids,
        coach_security=coach_sec,
        rng_traces=[],
        tuning_report={},
    )

    logger.emit("=================================================", "normal")
    logger.emit(f"CAREER + UNIVERSE SIM — { _dt.datetime.utcnow().isoformat(timespec='seconds') }Z", "normal")
    logger.emit("=================================================", "normal")
    logger.emit(f"Seed   : {run_cfg.seed}", "normal")
    logger.emit(f"Mode   : {run_cfg.mode}", "normal")
    logger.emit(f"Years  : {run_cfg.years}", "normal")
    logger.emit(f"Start  : {run_cfg.start_year}", "normal")
    logger.emit("-------------------------------------------------", "normal")
    if SIMENGINE_IMPORT_ERROR:
        logger.emit(f"[WARN] SimEngine import failed: {SIMENGINE_IMPORT_ERROR}", "normal")
    if DRAFTLOTTERY_IMPORT_ERROR:
        logger.emit(f"[WARN] Draft lottery import failed: {DRAFTLOTTERY_IMPORT_ERROR}", "debug")
    if WAIVERS_IMPORT_ERROR:
        logger.emit(f"[WARN] Waivers import warning: {WAIVERS_IMPORT_ERROR}", "debug")

    print(">>> Simulation wiring complete")

    # Core loop with crash safety
    last_context: Dict[str, Any] = {"phase": "init", "year": None}

    narrative_context: Dict[str, Any] = {}
    try:
        ustate = run_simulation_core(
            run_cfg=run_cfg,
            uni_cfg=uni_cfg,
            scn_cfg=scn_cfg,
            league=league,
            sim=sim,
            ustate=ustate,
            narrative_context=narrative_context,
            logger=logger,
            out=out,
            last_context=last_context,
        )


        # Regression mode check
        if run_cfg.mode == "regression":
            regression_compare_or_write(run_cfg, out, logger)

        logger.emit("Simulation finished successfully.", "normal")
        print(">>> Simulation finished successfully")
        return 0

    except Exception as e:
        # Never crash silently
        tb = traceback.format_exc()
        logger.emit("=================================================", "normal")
        logger.emit("[FATAL] Simulation crashed.", "normal")
        logger.emit(f"Seed: {run_cfg.seed}", "normal")
        logger.emit(f"Mode: {run_cfg.mode}", "normal")
        logger.emit(f"Last context: {safe_to_primitive(last_context)}", "normal")
        logger.emit(f"Error: {type(e).__name__}: {e}", "normal")
        logger.emit("-------------------------------------------------", "normal")
        logger.emit("Last 200 log lines:", "normal")
        for line in logger.last_lines():
            logger.emit(line, "normal", also_print=False)
        logger.emit("=================================================", "normal")

        out.append_error(tb)

        # dump crash snapshot
        try:
            crash = {
                "seed": run_cfg.seed,
                "mode": run_cfg.mode,
                "last_context": safe_to_primitive(last_context),
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb.splitlines(),
                "universe_state": safe_to_primitive(ustate),
            }
            out.write_json("crash_snapshot.json", crash)
            out.write_text("last_200_lines.log", "\n".join(logger.last_lines()))
        except Exception:
            # last resort: don't re-crash
            pass

        print(f"[FATAL] Crash details written to: {out.run_dir}")
        return 1

    finally:
        out.close()

if __name__ == "__main__":
    raise SystemExit(main())
