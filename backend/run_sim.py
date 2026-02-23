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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable
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
    def __init__(self, run_dir: Path, pretty: bool):
        self.run_dir = run_dir
        self.pretty = pretty
        self.timeline_path = run_dir / "timeline.log"
        self.errors_path = run_dir / "errors.log"
        self._timeline_fp = None

    def open(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._timeline_fp = open(self.timeline_path, "a", encoding="utf-8")

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
            self.ring.append(line)
            self.out.append_timeline(line, flush=False)
            if also_print:
                print(line)

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

    # regression/debug
    rng_traces: List[Dict[str, Any]] = field(default_factory=list)

# =============================================================================
# Universe event generation helpers
# =============================================================================

def _trace_roll(state: UniverseState, cfg: UniverseConfig, label: str, r: random.Random, value: float) -> None:
    if cfg.dump_rng_traces:
        state.rng_traces.append({"label": label, "value": value})

def _choose_era(r: random.Random) -> str:
    return r.choice(list(DEFAULT_ERAS))

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

    # parity / chaos drift (can be affected by shocks)
    parity = clamp(state.parity_index + (ucfg.parity_target - state.parity_index) * 0.20 + r.uniform(-0.03, 0.03), 0.10, 0.90)
    chaos = clamp(state.chaos_index + (ucfg.chaos - state.chaos_index) * 0.20 + r.uniform(-0.03, 0.03), 0.10, 0.95)

    new_state = dataclasses.replace(
        state,
        salary_cap_m=new_cap,
        cap_growth_rate=growth,
        inflation_factor=inflation,
        league_health=health,
        parity_index=parity,
        chaos_index=chaos,
    )
    return new_state, events

def _maybe_era_shift(state: UniverseState, ucfg: UniverseConfig, r: random.Random, year: int,
                    injected: List[ScenarioEvent]) -> Tuple[UniverseState, List[UniverseEvent]]:
    events: List[UniverseEvent] = []

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
            state = dataclasses.replace(state, active_era=era)
        return state, events

    roll = r.random()
    _trace_roll(state, ucfg, "era_shift_roll", r, roll)
    if roll < ucfg.era_shift_frequency:
        era = _choose_era(r)
        if era != state.active_era:
            events.append(
                UniverseEvent(
                    event_id=str(uuid.uuid4()),
                    year=year,
                    day=None,
                    type="ERA_SHIFT",
                    teams=[],
                    headline=f"ERA SHIFT: The league swings into '{era}'",
                    details={"from": state.active_era, "to": era},
                    impact_score=0.70,
                    tags=["era"],
                )
            )
            state = dataclasses.replace(state, active_era=era)
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
            # Weighted drift: rebuild teams drift less into win_now unless chaos
            candidates = list(DEFAULT_TEAM_ARCHETYPES)
            if cur == "rebuild":
                weights = [0.34, 0.10, 0.30, 0.20, 0.06]
            elif cur == "win_now":
                weights = [0.12, 0.40, 0.30, 0.10, 0.08]
            else:
                weights = [0.25, 0.22, 0.30, 0.18, 0.05]
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

def _simulate_standings(teams: List[Any], state: UniverseState, ucfg: UniverseConfig, r: random.Random) -> List[TeamStanding]:
    standings: List[TeamStanding] = []
    # Convert chaos/parity to noise
    # More parity => compress spread; more chaos => more randomness
    base_noise = 0.06 * (0.5 + state.chaos_index)  # ~0.06-0.09
    parity_compress = 0.35 + 0.65 * state.parity_index  # 0.35-1.0

    for t in teams:
        tid = _team_id(t)
        name = _team_name(t)
        exp = _team_expected_win_pct(t)

        # compress expectation toward 0.5 with parity
        exp_adj = 0.50 + (exp - 0.50) * (1.0 - 0.55 * parity_compress)

        # noise
        noise = r.gauss(0.0, base_noise)
        pct = clamp(exp_adj + noise, 0.30, 0.70)

        # 82-game-ish points approximation: points ~ 164 * point_pct
        points = int(round(164 * pct))

        # goal diff rough: scale by deviation from 0.5
        gd = int(round((pct - 0.50) * 240 + r.gauss(0, 18)))

        standings.append(
            TeamStanding(
                team_id=tid,
                team_name=name,
                expected_win_pct=exp,
                point_pct=pct,
                points=points,
                goal_diff=gd,
                bucket=_bucket_from_point_pct(pct),
            )
        )

    standings.sort(key=lambda s: (s.points, s.goal_diff), reverse=True)
    return standings

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

def _generate_waiver_candidates(teams: List[Any], standings: List[TeamStanding], state: UniverseState,
                               ucfg: UniverseConfig, r: random.Random) -> List[Tuple[str, str]]:
    """
    Returns list of (team_id, player_label) "waived".
    Narrative candidates: cap crunch / surplus / underperforming vets / fringe youth.
    """
    waived: List[Tuple[str, str]] = []
    # pick some teams likely to waive
    for s in standings:
        # higher chance if bubble/rebuild or if "cap crunch" (narrative)
        base = ucfg.waiver_rate * 0.35
        if s.bucket == "rebuild":
            base += 0.05
        if s.bucket == "bubble":
            base += 0.03
        if s.goal_diff < -20:
            base += 0.03
        if state.salary_cap_m < 85:
            base += 0.02

        if r.random() < base:
            # number waived by team 1-3
            n = 1 + (1 if r.random() < 0.35 else 0) + (1 if r.random() < 0.10 else 0)
            for _ in range(n):
                role = r.choice(["fringe winger", "7th D", "backup goalie", "underperforming veteran", "young bubble prospect"])
                label = f"{role.title()}_{abs(split_seed(state.salary_cap_m.__hash__(), s.team_id + role)) % 10000:04d}"
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

def _trade_eval(buyer_gain: float, seller_gain: float, chaos: float, r: random.Random) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Accept if both gain enough OR chaos allows a lopsided outcome rarely.
    Returns (accepted, fairness, breakdown)
    """
    fairness = 1.0 - abs(buyer_gain - seller_gain)
    fairness = clamp(fairness, 0.0, 1.0)
    both_ok = (buyer_gain >= 0.10 and seller_gain >= 0.10)

    if both_ok:
        return True, fairness, {"both_ok": True, "chaos_override": False}

    # chaos override
    override_chance = clamp(0.04 * (0.5 + chaos), 0.01, 0.08)
    roll = r.random()
    if roll < override_chance:
        return True, fairness, {"both_ok": False, "chaos_override": True, "override_roll": roll, "override_chance": override_chance}
    return False, fairness, {"both_ok": False, "chaos_override": False, "override_roll": roll, "override_chance": override_chance}

def _generate_trades(standings: List[TeamStanding], state: UniverseState, ucfg: UniverseConfig, r: random.Random, year: int,
                     injected: List[ScenarioEvent]) -> List[UniverseEvent]:
    events: List[UniverseEvent] = []

    # scenario: force_trade
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
        return events

    # probability any trades this year
    roll = r.random()
    _trace_roll(state, ucfg, "trade_year_roll", r, roll)
    if roll > ucfg.trade_rate:
        return events

    # determine number of trades
    n = 1
    if r.random() < 0.45:
        n += 1
    if r.random() < 0.20:
        n += 1

    # build buyer/seller pools
    buyers = [s for s in standings if s.bucket in ("contender", "playoff")]
    sellers = [s for s in standings if s.bucket in ("rebuild",)]
    bubbles = [s for s in standings if s.bucket == "bubble"]

    # allow bubble to sell sometimes
    if r.random() < 0.35:
        sellers += r.sample(bubbles, k=min(len(bubbles), 2))

    if not buyers or not sellers:
        return events

    for _ in range(n):
        buyer = r.choice(buyers)
        seller = r.choice([s for s in sellers if s.team_id != buyer.team_id] or sellers)

        # identify needs
        need_def = buyer.goal_diff < -20
        need_off = buyer.points < 95
        cap_crunch = state.salary_cap_m < 88 and r.random() < 0.25

        # choose target asset kind
        if cap_crunch and r.random() < ucfg.cap_dump_rate:
            target_kind = "cap_dump"
        elif need_def:
            target_kind = "top4d_rental"
        elif need_off:
            target_kind = "top6f"
        else:
            target_kind = r.choice(["middle6w", "backupg", "top4d_rental"])

        asset_in = _generate_asset(r, target_kind)

        # buyer pays
        package: List[TradeAsset] = []
        if asset_in.value > 0.70 or r.random() < ucfg.blockbuster_rate:
            package.append(_generate_asset(r, "1st"))
            package.append(_generate_asset(r, "prospect_high"))
        elif asset_in.value > 0.55:
            package.append(_generate_asset(r, "1st") if r.random() < 0.55 else _generate_asset(r, "2nd"))
            package.append(_generate_asset(r, "prospect_mid"))
        else:
            package.append(_generate_asset(r, "2nd"))
            if r.random() < 0.35:
                package.append(_generate_asset(r, "prospect_mid"))

        buyer_out_value = sum(a.value for a in package)
        buyer_gain = asset_in.value - buyer_out_value
        seller_gain = buyer_out_value - asset_in.value * 0.85  # seller values futures slightly less/more varied

        accepted, fairness, breakdown = _trade_eval(buyer_gain, seller_gain, state.chaos_index * ucfg.trade_chaos_multiplier, r)

        if not accepted:
            continue

        # retention narrative
        retained = 0.0
        if asset_in.cap_m > 3.0 and r.random() < ucfg.retention_rate:
            retained = clamp(r.uniform(0.10, 0.50), 0.0, 0.50)

        headline = f"{buyer.team_name} acquires {asset_in.label} from {seller.team_name} for " + " + ".join(a.label for a in package)

        details = {
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
            "reason": "deadline push" if buyer.bucket == "contender" else "needs-based",
        }

        tags = ["trade"]
        if asset_in.label.lower().startswith("top-4"):
            tags.append("defense")
        if "pick" in headline.lower():
            tags.append("futures")
        if retained > 0:
            tags.append("retained")

        impact = clamp(0.35 + asset_in.value * 0.75, 0.15, 0.95)

        events.append(
            UniverseEvent(
                event_id=str(uuid.uuid4()),
                year=year,
                day=None,
                type="TRADE",
                teams=[buyer.team_id, seller.team_id],
                headline="TRADE: " + headline,
                details=details,
                impact_score=impact,
                tags=tags,
            )
        )

    return events

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

def _gen_fa_pool(r: random.Random, inflation: float) -> List[FreeAgent]:
    pool: List[FreeAgent] = []
    # stars (rare)
    if r.random() < 0.25:
        pool.append(
            FreeAgent(
                name=f"UFA_Star_{r.randint(100,999)}",
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
                name=f"UFA_{role.replace(' ','_')}_{r.randint(1000,9999)}",
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

def _fa_signings(standings: List[TeamStanding], state: UniverseState, ucfg: UniverseConfig, r: random.Random,
                 year: int, injected: List[ScenarioEvent]) -> List[UniverseEvent]:
    events: List[UniverseEvent] = []

    # chance FA period is "active" this year
    roll = r.random()
    _trace_roll(state, ucfg, "fa_year_roll", r, roll)
    if roll > ucfg.free_agency_rate:
        return events

    pool = _gen_fa_pool(r, state.inflation_factor)

    # each team attempts some signings depending on window
    # (very simplified — narrative-first)
    teams_sorted = sorted(standings, key=lambda s: s.points, reverse=True)
    max_signings = int(clamp(8 + r.randint(0, 8), 6, 20))
    signings_done = 0

    # build attractiveness snapshot (stable per year)
    attract = {t.team_id: _team_attractiveness(t, state, r) for t in teams_sorted}

    def pick_team_for_player(fa: FreeAgent) -> Optional[TeamStanding]:
        # sample a few candidate teams
        candidates = r.sample(teams_sorted, k=min(len(teams_sorted), 10))
        scored: List[Tuple[float, TeamStanding]] = []
        for tm in candidates:
            # money (simulate offers): contenders slightly lower; rebuilds overpay
            overpay = 1.0
            if tm.bucket == "rebuild":
                overpay = r.uniform(1.05, 1.25)
            elif tm.bucket == "contender":
                overpay = r.uniform(0.95, 1.10)
            offer = fa.ask_m * overpay

            # role opportunity: rebuild/bubble offer bigger role
            role_score = 0.70 if tm.bucket in ("rebuild", "bubble") else 0.55 if tm.bucket == "playoff" else 0.50

            score = (
                fa.prefs["money"] * (offer / (fa.ask_m * 1.20)) +
                fa.prefs["contender"] * (0.85 if tm.bucket in ("contender", "playoff") else 0.45) +
                fa.prefs["location"] * attract[tm.team_id] +
                fa.prefs["role"] * role_score
            )
            # sprinkle chaos
            score += r.uniform(-0.05, 0.05) * (0.5 + state.chaos_index)
            scored.append((score, tm))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] > 0.55:
            return scored[0][1]
        return None

    major_threshold = 0.75

    for fa in sorted(pool, key=lambda x: x.rating, reverse=True):
        if signings_done >= max_signings:
            break

        tm = pick_team_for_player(fa)
        if tm is None:
            continue

        # "unexpected signing" control chaos
        if tm.bucket == "rebuild" and fa.rating > 0.78 and r.random() < 0.55:
            # sometimes stars avoid rebuilds
            continue

        signings_done += 1

        is_major = fa.rating >= major_threshold or fa.ask_m >= 8.5

        headline = f"SIGNING: {tm.team_name} signs {fa.name} ({fa.role}) at ~${fmt_money(fa.ask_m)} AAV"

        details = {
            "team_id": tm.team_id,
            "player": safe_to_primitive(fa),
            "contender_status": tm.bucket,
            "attractiveness": attract[tm.team_id],
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

        # occasional "unexpected"
        if is_major and r.random() < 0.18 * (0.5 + state.chaos_index):
            events[-1].tags.append("unexpected")

    return events

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

def _draft_lottery(standings: List[TeamStanding], year: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns (lottery_results, pick_order_team_ids)
    """
    # bottom 16
    bottom16 = list(reversed(standings))[:16]
    pick_order = [s.team_id for s in bottom16]  # initial order; lottery changes top

    if LotteryTeam is None or run_draft_lottery is None:
        # fallback: no lottery module available
        results = [{"pick": i + 1, "team_id": tid, "note": "fallback_no_lottery_module"} for i, tid in enumerate(pick_order)]
        return results, pick_order

    # Build LotteryTeam list with simplistic odds (descending)
    # If your LotteryTeam expects different fields, safe_to_primitive will still work,
    # but we try to match common pattern (team_id, odds).
    lot_teams = []
    # Use crude odds: worst gets highest
    total = sum(range(1, len(bottom16) + 1))
    for idx, s in enumerate(bottom16):
        weight = (len(bottom16) - idx) / total  # highest for worst
        try:
            lot_teams.append(LotteryTeam(team_id=s.team_id, odds=weight))  # type: ignore
        except Exception:
            # fallback signature
            try:
                lot_teams.append(LotteryTeam(s.team_id, weight))  # type: ignore
            except Exception:
                lot_teams.append({"team_id": s.team_id, "odds": weight})

    try:
        lottery = run_draft_lottery(lot_teams)  # type: ignore
        # Expect list of LotteryTeam or dict-like in pick order
        results: List[Dict[str, Any]] = []
        pick_ids: List[str] = []
        for i, lt in enumerate(lottery):
            tid = None
            if isinstance(lt, dict):
                tid = lt.get("team_id") or lt.get("id")
            else:
                tid = getattr(lt, "team_id", None) or getattr(lt, "id", None)
            if tid is None:
                tid = str(lt)
            pick_ids.append(str(tid))
            results.append({"pick": i + 1, "team_id": str(tid)})
        return results, pick_ids
    except Exception:
        # fallback
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
    waiver_claims: List[WaiverClaim] = field(default_factory=list)
    waived_count: int = 0

def simulate_universe_year(
    league: Any,
    state: UniverseState,
    run_cfg: RunConfig,
    uni_cfg: UniverseConfig,
    scn_cfg: ScenarioConfig,
    year: int,
    logger: RunnerLogger,
    rng: random.Random,
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

    # Phase 1: economics
    state, econ_events = _economics_advance(state, uni_cfg, rng, year, injected)
    events.extend(econ_events)

    # Phase 2: era/meta
    state, era_events = _maybe_era_shift(state, uni_cfg, rng, year, injected)
    events.extend(era_events)

    # Phase 3: team identity drift (philosophy)
    if teams:
        events.extend(_team_identity_drift(state, uni_cfg, rng, year, teams))

    # Phase 4: standings baseline
    standings = _simulate_standings(teams, state, uni_cfg, rng) if teams else []

    # Phase 5: waiver priority reset (reverse standings)
    if standings and uni_cfg.enable_waivers:
        state.waiver_priority = _reset_waiver_priority_from_standings(standings)

    # Phase 6: coach carousel
    events.extend(_coach_changes(standings, state, uni_cfg, rng, year, injected))

    # Phase 7: trades
    if uni_cfg.enable_trades:
        trade_events = _generate_trades(standings, state, uni_cfg, rng, year, injected)
        events.extend(trade_events)

    # Phase 8: waivers
    waiver_claims: List[WaiverClaim] = []
    waived_count = 0
    if uni_cfg.enable_waivers and standings:
        waived = _generate_waiver_candidates(teams, standings, state, uni_cfg, rng)
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
    if uni_cfg.enable_draft and standings:
        lottery_results, pick_order = _draft_lottery(standings, year)
        # log top 5
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
        # optional draft selection placeholder
        events.append(
            UniverseEvent(
                event_id=str(uuid.uuid4()),
                year=year,
                day=None,
                type="DRAFT",
                teams=pick_order[:10],
                headline="DRAFT: Top picks set (mock) — draft selection simulation not enabled in runner.",
                details={"pick_order": pick_order[:32]},
                impact_score=0.25,
                tags=["draft", "mock"],
            )
        )

    # Phase 10: free agency
    if uni_cfg.enable_free_agency and standings:
        events.extend(_fa_signings(standings, state, uni_cfg, rng, year, injected))

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

    # Sort events: major first for printing (but keep list stable)
    # We'll handle printing tiers separately.
    return UniverseYearResult(
        year=year,
        state=state,
        standings=standings,
        events=events,
        lottery_results=lottery_results,
        draft_pick_order=pick_order,
        waiver_claims=waiver_claims,
        waived_count=waived_count,
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

# =============================================================================
# NHL UNIVERSE block printing
# =============================================================================

def print_year_header(logger: RunnerLogger, year: int, state: UniverseState, standings: List[TeamStanding], events: List[UniverseEvent], log_level: str) -> None:
    top = standings[:2] if standings else []
    bottom = list(reversed(standings))[:2] if standings else []

    logger.emit("=================================================", "normal")
    logger.emit(f"NHL UNIVERSE — {year}", "normal")
    logger.emit(f"Salary Cap            : {fmt_money(state.salary_cap_m)}", "normal")
    logger.emit(f"Cap Growth Rate       : {state.cap_growth_rate:+.3f}", "normal")
    logger.emit(f"Inflation Factor      : {state.inflation_factor:.3f}", "debug")
    logger.emit(f"Parity Index          : {state.parity_index:.3f}", "normal")
    logger.emit(f"Chaos Index           : {state.chaos_index:.3f}", "normal")
    logger.emit(f"League Health         : {state.league_health:.3f}", "normal")
    logger.emit(f"Active Era            : {state.active_era}", "normal")
    logger.emit("-------------------------------------------------", "normal")

    if top:
        logger.emit("Top Teams:", "normal")
        for s in top:
            logger.emit(f"  {s.team_id} — {s.points}", "normal")
    if bottom:
        logger.emit("Bottom Teams:", "normal")
        for s in bottom:
            logger.emit(f"  {s.team_id} — {s.points}", "normal")

    # Major headlines vs wire
    major = [e for e in events if e.impact_score >= 0.60]
    wire = [e for e in events if e.impact_score < 0.60]

    if major:
        logger.emit("", "normal")
        logger.emit("MAJOR HEADLINES:", "normal")
        for e in major[:10]:
            logger.emit(f"• {e.headline}", "normal")

    if wire and logger._level_rank(log_level) >= logger._level_rank("normal"):
        logger.emit("WIRE:", "normal")
        for e in wire[:12]:
            logger.emit(f"  - {e.headline}", "normal")

    # Waiver priority snapshot (top 5)
    if state.waiver_priority:
        logger.emit("Waiver Priority (Top 5): " + ", ".join(state.waiver_priority[:5]), "debug")

    logger.emit("=================================================", "normal")

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
    try:
        scn_cfg = load_scenario_file(args.scenario_file)
    except Exception as e:
        print(f"[FATAL] Failed to load scenario file: {type(e).__name__}: {e}")
        return 2

    # output folder
    run_dir = Path(run_cfg.output_dir) / f"{now_utc_compact()}_{run_cfg.seed}"
    out = RunOutput(run_dir=run_dir, pretty=run_cfg.pretty_print)
    out.open()
    logger = RunnerLogger(out=out, log_level=run_cfg.log_level, flush_each_year=run_cfg.flush_each_year)

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

    # Try to locate league from sim
    if sim is not None:
        league = _safe_getattr(sim, "league", None) or _safe_getattr(sim, "universe", None)

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

    # Core loop with crash safety
    last_context: Dict[str, Any] = {"phase": "init", "year": None}

    try:
        for i in range(run_cfg.years):
            year = run_cfg.start_year + i
            last_context["year"] = year

            # Universe + career pipeline in proper order
            uni_result: Optional[UniverseYearResult] = None
            career_result: Optional[CareerYearResult] = None

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
                )
                ustate = uni_result.state

                # Print NHL universe block
                print_year_header(logger, year, ustate, uni_result.standings, uni_result.events, run_cfg.log_level)

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

            if run_cfg.flush_each_year:
                logger.flush()

        # Regression mode check
        if run_cfg.mode == "regression":
            regression_compare_or_write(run_cfg, out, logger)

        logger.emit("Simulation finished successfully.", "normal")
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
