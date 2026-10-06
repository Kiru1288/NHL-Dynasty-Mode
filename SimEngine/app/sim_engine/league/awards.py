"""
Canonical league awards registry and computation.

Official winners, Award Watch official races, eligibility, normalization,
deterministic ballot simulation, and payload assembly all live here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

from .playoffs import PlayoffResult
from .standings import StandingsTable, TeamStandingRecord

BALLOT_POINTS = [10.0, 7.0, 5.0, 3.0, 1.0]
VOTER_COUNT = 190

# Real NHL ballot formats. PHWA trophies use 10-7-5-3-1 five-deep ballots; the
# Vezina is voted by the 32 GMs (5-3-1); Conn Smythe by an 18-member PHWA panel;
# Jack Adams by the NHL Broadcasters' Association; GM of the Year by the 32 GMs
# plus an executive/media panel; Ted Lindsay by NHLPA membership.
BALLOT_CONFIG: Dict[str, Dict[str, Any]] = {
    "hart": {"voters": 190, "points": [10.0, 7.0, 5.0, 3.0, 1.0], "body": "PHWA"},
    "norris": {"voters": 190, "points": [10.0, 7.0, 5.0, 3.0, 1.0], "body": "PHWA"},
    "calder": {"voters": 190, "points": [10.0, 7.0, 5.0, 3.0, 1.0], "body": "PHWA"},
    "selke": {"voters": 190, "points": [10.0, 7.0, 5.0, 3.0, 1.0], "body": "PHWA"},
    "lady_byng": {"voters": 190, "points": [10.0, 7.0, 5.0, 3.0, 1.0], "body": "PHWA"},
    "vezina": {"voters": 32, "points": [5.0, 3.0, 1.0], "body": "NHL general managers"},
    "ted_lindsay": {"voters": 640, "points": [5.0, 3.0, 1.0], "body": "NHLPA members"},
    "conn_smythe": {"voters": 18, "points": [5.0, 3.0, 1.0], "body": "PHWA panel"},
    "jack_adams": {"voters": 96, "points": [5.0, 3.0, 1.0], "body": "NHL Broadcasters' Association"},
    "gm_of_the_year": {"voters": 42, "points": [5.0, 3.0, 1.0], "body": "GMs + executive/media panel"},
}


def ballot_config_for(award_id: str) -> Dict[str, Any]:
    cfg = BALLOT_CONFIG.get(str(award_id or ""))
    if cfg:
        return dict(cfg)
    return {"voters": VOTER_COUNT, "points": list(BALLOT_POINTS), "body": "PHWA"}
SUBJECTIVE_TROPHY_PUBLIC_CASE = {
    "hart": "Most valuable all-around season.",
    "norris": "Premier defenseman season.",
    "selke": "Elite defensive forward season.",
    "vezina": "Best goaltender season.",
    "calder": "Top first-year NHL player.",
    "lady_byng": "Production with exceptional discipline.",
    "ted_lindsay": "Players' view of most outstanding player.",
    "conn_smythe": "Most valuable playoff performer.",
    "jack_adams": "Coach who most outperformed his roster's expectations.",
    "gm_of_the_year": "Front office that built the season's biggest overachiever.",
}

VOTER_ARCHETYPES = (
    "production",
    "two_way",
    "team_success",
    "analytics",
    "workload",
    "traditional",
)


def normalize_percentage(value: Any) -> float:
    """Return a fraction in [0, 1]. Values > 1 are treated as percent points."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    if v > 1.0:
        v = v / 100.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _seed_int(season_seed: Any, *parts: Any) -> int:
    raw = "|".join(str(p) for p in (season_seed, *parts) if p is not None)
    if not raw:
        raw = "awards-default"
    return int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:12], 16)


def _rng(season_seed: Any, *parts: Any) -> random.Random:
    return random.Random(_seed_int(season_seed, *parts))


def validate_required_award_fields(row: Mapping[str, Any], fields: Sequence[str]) -> List[str]:
    missing: List[str] = []
    for key in fields:
        if key not in row or row.get(key) is None:
            missing.append(str(key))
            continue
        val = row.get(key)
        if isinstance(val, str) and not val.strip():
            missing.append(str(key))
    return missing


def percentile_rank(values: Sequence[float], value: float) -> float:
    if not values:
        return 0.5
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    # Midrank percentile
    below = sum(1 for v in ordered if v < value)
    equal = sum(1 for v in ordered if v == value)
    return (below + 0.5 * equal) / float(n)


def robust_z(values: Sequence[float], value: float, *, clamp: float = 3.0) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = ordered[len(ordered) // 2]
    abs_dev = sorted(abs(v - mid) for v in ordered)
    mad = abs_dev[len(abs_dev) // 2] or 1e-6
    z = 0.6745 * (float(value) - mid) / mad
    return max(-clamp, min(clamp, z))


def _norm01_from_z(z: float) -> float:
    return max(0.0, min(1.0, 0.5 + z / 6.0))


def normalize_pool_metric(pool: Sequence[Mapping[str, Any]], getter: Callable[[Mapping[str, Any]], float]) -> Dict[str, float]:
    vals = [getter(r) for r in pool]
    out: Dict[str, float] = {}
    for row in pool:
        pid = _pid(row)
        out[pid] = percentile_rank(vals, getter(row))
    return out


def _pid(row: Mapping[str, Any]) -> str:
    return str(row.get("player_id") or row.get("id") or row.get("entity_id") or "")


def _tid(row: Mapping[str, Any]) -> str:
    return str(row.get("team_id") or "")


def _isolated_scoring_pool(pool: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Strip award-local scoring artifacts so shared stat rows cannot leak across ballots."""
    out: List[Dict[str, Any]] = []
    for row in pool:
        if not isinstance(row, Mapping):
            continue
        clean = {
            k: v
            for k, v in row.items()
            if not str(k).startswith("_") and k not in ("component_scores",)
        }
        out.append(clean)
    return out


STAT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "gp": ("gp", "games_played"),
    "g": ("g", "goals"),
    "a": ("a", "assists"),
    "pts": ("pts", "points"),
    "pim": ("pim", "penalty_minutes"),
    "plus_minus": ("plus_minus", "pm", "plusminus"),
    "shots": ("shots", "sog"),
    "toi": ("toi", "toi_minutes"),
    "toi_pg": ("toi_per_game", "toi_per_gp", "avg_toi"),
    "ev_toi": ("ev_toi", "es_toi"),
    "pk_toi": ("pk_toi", "sh_toi"),
    "es_pts": ("ev_points", "es_points"),
    "takeaways": ("takeaways", "tk"),
    "giveaways": ("giveaways", "gv", "giv"),
    "blocks": ("blocked_shots", "blk", "blocks"),
    "fow": ("fow", "faceoffs_won"),
    "fol": ("fol", "faceoffs_lost"),
    "fo_taken": ("fo_taken", "faceoffs_taken"),
    "fo_pct": ("faceoff_pct", "fo_pct"),
    "minors": ("minor_penalties", "minors"),
    "majors": ("major_penalties", "majors", "fights", "misconducts"),
    "xgf_pct": ("xgf_pct",),
    "xga_60": ("xga_per_60", "xga60"),
    "cf_pct": ("cf_pct", "corsi_pct"),
    "war": ("war",),
    "impact": ("impact_score",),
    "sa": ("shots_against", "sa"),
    "ga": ("ga", "goals_against"),
    "en_ga": ("empty_net_goals", "en_goals"),
    "saves": ("saves",),
    "sv_pct": ("sv_pct", "save_pct"),
    "gs": ("games_started", "gs", "starts"),
    "so": ("shutouts", "so"),
    "w": ("w", "wins"),
    "gaa": ("gaa", "goals_against_average"),
    "sog": ("sog", "shots"),
    "gsax": ("gsax",),
    "hdsv": ("high_danger_save_pct",),
    "qs_pct": ("quality_start_pct",),
    "final_round_pts": ("final_round_points",),
    "elim_pts": ("elimination_game_points",),
}


def stat(row: Mapping[str, Any], key: str) -> Optional[float]:
    aliases = STAT_ALIASES.get(key, (key,))
    for alias in aliases:
        if alias not in row:
            continue
        val = row.get(alias)
        if val is None:
            continue
        try:
            x = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            return x
    return None


def normalize_percentage_optional(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if v > 1.0:
        v = v / 100.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@dataclass(frozen=True)
class Comp:
    key: str
    label: str
    get: Callable[[Mapping[str, Any]], Optional[float]]
    weight: float
    higher_better: bool = True
    shrink: Optional[Tuple[str, float]] = None
    chain: Tuple[Callable[[Mapping[str, Any]], Optional[float]], ...] = ()
    scope: Optional[Callable[[Mapping[str, Any]], bool]] = None
    archetypes: FrozenSet[str] = frozenset()
    fmt: str = "int"


@dataclass
class ScoredPool:
    scores: Dict[str, float]
    pct_by_comp: Dict[str, Dict[str, float]]
    raw_by_comp: Dict[str, Dict[str, float]]
    ranks: Dict[str, int]
    active: List[Comp]
    dropped: List[Comp]
    coverage: Dict[str, float]
    imputed: Dict[str, List[str]]
    path_counts: Dict[str, int]


def _midrank_percentile(sorted_vals: Sequence[float], value: float) -> float:
    if not sorted_vals:
        return 0.5
    left = bisect_left(sorted_vals, value)
    right = bisect_right(sorted_vals, value)
    below = left
    equal = max(0, right - left)
    n = len(sorted_vals)
    return (below + 0.5 * equal) / float(n)


def _coverage(pool: Sequence[Mapping[str, Any]], getter: Callable[[Mapping[str, Any]], Optional[float]], scope: Optional[Callable[[Mapping[str, Any]], bool]]) -> float:
    if not pool:
        return 0.0
    ok = 0
    for row in pool:
        if scope is not None and not scope(row):
            ok += 1
            continue
        if getter(row) is not None:
            ok += 1
    return ok / float(len(pool))


def score_pool(
    award_id: str,
    pool: Sequence[Mapping[str, Any]],
    comps: Sequence[Comp],
    *,
    ref_pool: Optional[Sequence[Mapping[str, Any]]] = None,
    min_coverage: float = 0.60,
) -> ScoredPool:
    ref = list(ref_pool or pool)
    active: List[Comp] = []
    dropped: List[Comp] = []
    pct_by_comp: Dict[str, Dict[str, float]] = {}
    raw_by_comp: Dict[str, Dict[str, float]] = {}
    imputed: Dict[str, List[str]] = {_pid(r): [] for r in pool}

    for comp in comps:
        getter = comp.get
        for alt in comp.chain:
            if _coverage(ref, alt, comp.scope) >= min_coverage:
                getter = alt
                break
        cov = _coverage(ref, getter, comp.scope)
        if cov < min_coverage:
            dropped.append(comp)
            continue
        active.append(comp)
        shrink_key, shrink_k = comp.shrink or ("", 0.0)
        present: List[Tuple[str, float]] = []
        med = 0.5
        if shrink_k > 0 and shrink_key:
            vals = [getter(r) for r in ref if (comp.scope is None or comp.scope(r)) and getter(r) is not None]
            if vals:
                vals.sort()
                med = vals[len(vals) // 2]
        for row in ref:
            pid = _pid(row)
            in_scope = comp.scope is None or comp.scope(row)
            raw = getter(row) if in_scope else None
            if raw is None:
                pct_by_comp.setdefault(comp.key, {})[pid] = 0.5
                raw_by_comp.setdefault(comp.key, {})[pid] = raw if raw is not None else float("nan")
                if in_scope:
                    imputed.setdefault(pid, []).append(comp.key)
                continue
            n = stat(row, shrink_key) if shrink_k > 0 and shrink_key else None
            adj = raw
            if shrink_k > 0 and n is not None and n > 0:
                adj = (n * raw + shrink_k * med) / (n + shrink_k)
            present.append((pid, adj))
        sorted_vals = sorted(v for _, v in present)
        for pid, adj in present:
            pct = _midrank_percentile(sorted_vals, adj)
            if not comp.higher_better:
                pct = 1.0 - pct
            pct_by_comp.setdefault(comp.key, {})[pid] = pct
            raw_by_comp.setdefault(comp.key, {})[pid] = adj

    total_w = sum(c.weight for c in active) or 1.0
    scores: Dict[str, float] = {}
    ranks: Dict[str, int] = {}
    for row in pool:
        pid = _pid(row)
        s = 0.0
        for comp in active:
            w = comp.weight / total_w
            s += w * pct_by_comp.get(comp.key, {}).get(pid, 0.5)
        scores[pid] = s

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for i, (pid, _) in enumerate(ordered, start=1):
        ranks[pid] = i

    return ScoredPool(
        scores=scores,
        pct_by_comp=pct_by_comp,
        raw_by_comp=raw_by_comp,
        ranks=ranks,
        active=list(active),
        dropped=dropped,
        coverage={c.key: _coverage(ref, c.get, c.scope) for c in active},
        imputed=imputed,
        path_counts={"full": len(pool), "fallback": 0},
    )


def _onice_get(row: Mapping[str, Any]) -> Optional[float]:
    x = normalize_percentage_optional(row.get("xgf_pct"))
    if x is not None:
        return x
    c = normalize_percentage_optional(row.get("cf_pct"))
    if c is not None:
        return c
    pm = stat(row, "plus_minus")
    if pm is None:
        return None
    return max(0.0, min(1.0, 0.5 + pm / 80.0))


def _value_get(row: Mapping[str, Any]) -> Optional[float]:
    w = stat(row, "war")
    if w is not None:
        return w
    return stat(row, "impact")


def _pts_pg(row: Mapping[str, Any]) -> Optional[float]:
    gp = stat(row, "gp")
    pts = stat(row, "pts")
    if gp is None or pts is None or gp <= 0:
        return None
    return pts / gp


def _team_pts_pct(row: Mapping[str, Any], team_ctx: Mapping[str, Mapping[str, Any]]) -> Optional[float]:
    ctx = team_ctx.get(_tid(row), {})
    v = ctx.get("points_pct_norm")
    if v is None:
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _team_share_pts(row: Mapping[str, Any], team_ctx: Mapping[str, Mapping[str, Any]]) -> Optional[float]:
    ctx = team_ctx.get(_tid(row), {})
    gf = ctx.get("team_gf")
    pts = stat(row, "pts")
    if gf is None or pts is None:
        return None
    try:
        gfv = float(gf)
        if gfv <= 0:
            return None
        return pts / gfv
    except (TypeError, ValueError):
        return None


def _toi_pg_minutes(row: Mapping[str, Any]) -> float:
    for key in ("toi_per_game", "toi_per_gp", "avg_toi"):
        if row.get(key) is not None:
            return _safe_float(row.get(key))
    gp = max(1, _gp(row))
    total = _safe_float(row.get("toi"), _safe_float(row.get("toi_minutes"), 0.0))
    if total <= 0 and row.get("toi_sec") is not None:
        total = _safe_float(row.get("toi_sec")) / 60.0  # franchise ledger stores seconds
    if total > 0:
        return total / float(gp)
    return 0.0


def _pk_toi_pg_minutes(row: Mapping[str, Any]) -> float:
    for key in ("pk_toi_per_game", "pk_toi_pg"):
        if row.get(key) is not None:
            return _safe_float(row.get(key))
    per_gp = _optional_stat_per_gp(row, ("pk_toi", "sh_toi"))
    if per_gp is not None:
        return per_gp
    share = _safe_float(row.get("pk_toi_share"), 0.0)
    if share > 0:
        return share
    return 0.0


def _is_centre(row: Mapping[str, Any]) -> bool:
    return _pos(row) in {"C", "CTR", "CENTER"}


def _selke_fo_pct(row: Mapping[str, Any]) -> Optional[float]:
    fo_taken = stat(row, "fo_taken")
    if fo_taken is None:
        fo_taken = _safe_int(row.get("faceoffs_taken"), 0)
    if fo_taken < 300:
        return None
    fo = stat(row, "fo_pct")
    if fo is not None:
        return fo
    return normalize_percentage_optional(row.get("faceoff_pct"))


def _selke_tk_60(row: Mapping[str, Any]) -> Optional[float]:
    raw = row.get("takeaways_per_60")
    if raw is not None:
        return _safe_float(raw)
    tk = stat(row, "takeaways")
    if tk is None:
        return None
    toi = stat(row, "toi")
    if toi is not None and toi > 0:
        return float(tk) / float(toi) * 60.0
    gp = stat(row, "gp")
    if gp is not None and gp > 0:
        return float(tk) / float(gp)
    return None


def _selke_tk_gv_pg(row: Mapping[str, Any]) -> Optional[float]:
    tk = stat(row, "takeaways")
    gv = stat(row, "giveaways")
    gp = stat(row, "gp")
    if tk is None or gv is None or gp is None or gp <= 0:
        return None
    return (float(tk) - float(gv)) / float(gp)


def _selke_blocks_60(row: Mapping[str, Any]) -> Optional[float]:
    blk60 = row.get("blocks_per_60") or row.get("blocked_shots_per_60")
    if blk60 is not None:
        return _safe_float(blk60)
    blocks = stat(row, "blocks")
    toi = stat(row, "toi")
    if blocks is not None and toi is not None and toi > 0:
        return float(blocks) / float(toi) * 60.0
    gp = stat(row, "gp")
    if blocks is not None and gp is not None and gp > 0:
        return float(blocks) / float(gp)
    return None


def _selke_comps() -> Tuple[Comp, ...]:
    return (
        Comp("xga_60", "Expected goals against/60", lambda r: stat(r, "xga_60"), 0.18, higher_better=False, archetypes=frozenset({"defense", "analytics"})),
        Comp("onice", "On-ice impact", _onice_get, 0.18, chain=(_value_get,), archetypes=frozenset({"two_way", "analytics"})),
        Comp("pk_toi_pg", "PK time per game", _pk_toi_pg_minutes, 0.14, archetypes=frozenset({"special_teams"})),
        Comp("tk_60", "Takeaways per 60", _selke_tk_60, 0.12, archetypes=frozenset({"defense"})),
        Comp("fo_pct", "Faceoff % (centres)", _selke_fo_pct, 0.10, scope=_is_centre, archetypes=frozenset({"centre"})),
        Comp("tk_gv_pg", "Takeaways minus giveaways/GP", _selke_tk_gv_pg, 0.06, archetypes=frozenset({"defense"})),
        Comp("blocks_60", "Blocks per 60", _selke_blocks_60, 0.04, archetypes=frozenset({"defense"})),
        Comp("pts_pg", "Points per game", _pts_pg, 0.10, shrink=("gp", 20.0), archetypes=frozenset({"production"})),
        Comp("toi_pg", "Ice time per game", _toi_pg_minutes, 0.08, archetypes=frozenset({"workload"})),
    )


def _hart_comps(team_ctx: Mapping[str, Mapping[str, Any]]) -> Tuple[Comp, ...]:
    return (
        Comp("pts_pg", "Points per game", _pts_pg, 0.26, shrink=("gp", 20.0), archetypes=frozenset({"production", "traditional"})),
        Comp("pts", "Points", lambda r: stat(r, "pts"), 0.12, archetypes=frozenset({"production"})),
        Comp("g", "Goals", lambda r: stat(r, "g"), 0.06, archetypes=frozenset({"production"})),
        Comp("onice", "On-ice impact", _onice_get, 0.14, archetypes=frozenset({"two_way", "analytics"})),
        Comp("value", "Individual value", _value_get, 0.14, archetypes=frozenset({"analytics"})),
        Comp("team_sh", "Share of team scoring", lambda r: _team_share_pts(r, team_ctx), 0.10, archetypes=frozenset({"team_success"})),
        Comp("team", "Team standing", lambda r: _team_pts_pct(r, team_ctx), 0.12, archetypes=frozenset({"team_success"})),
        Comp("avail", "Availability", lambda r: (stat(r, "gp") / 82.0) if stat(r, "gp") is not None else None, 0.06, archetypes=frozenset({"workload"})),
    )


def score_award(
    award_id: str,
    pool: Sequence[Mapping[str, Any]],
    *,
    team_ctx: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ref_pool: Optional[Sequence[Mapping[str, Any]]] = None,
    season_length: int = 82,
) -> Optional[ScoredPool]:
    # Hart, Lindsay, and Selke are scored by the ballot functions below.
    # A second percentile model was handing the trophy to a different player
    # than the race the league had been watching.
    return None


def _pts(row: Mapping[str, Any]) -> int:
    return _safe_int(row.get("pts"), _safe_int(row.get("g")) + _safe_int(row.get("a")))


def _goals(row: Mapping[str, Any]) -> int:
    return _safe_int(row.get("g"), _safe_int(row.get("goals")))


def _pos(row: Mapping[str, Any]) -> str:
    return str(row.get("position") or row.get("pos") or "").upper()


def _gp(row: Mapping[str, Any]) -> int:
    return _safe_int(row.get("gp"), _safe_int(row.get("games_played")))


def _is_goalie(row: Mapping[str, Any]) -> bool:
    return _pos(row) == "G"


def _is_defense(row: Mapping[str, Any]) -> bool:
    p = _pos(row).replace(" ", "")
    return p in {"D", "LD", "RD", "LHD", "RHD", "DEF", "DEFENSE", "DEFENCE"}


def _is_forward(row: Mapping[str, Any]) -> bool:
    return not _is_goalie(row) and not _is_defense(row)


def _stat_scope(row: Mapping[str, Any]) -> str:
    return str(row.get("stat_scope") or row.get("scope") or "regular_season").strip().lower()


def filter_regular_season_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    allowed = {"", "regular", "regular_season", "rs", "none"}
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        scope = _stat_scope(row)
        if scope in allowed or scope == "none":
            out.append(dict(row))
        elif scope in {"playoff", "playoffs", "preseason", "exhibition", "international", "allstar", "all_star"}:
            continue
        else:
            # Unknown scopes treated conservatively as non-regular except missing (already covered).
            continue
    return out


def filter_playoff_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        scope = _stat_scope(row)
        if scope in {"playoff", "playoffs", "postseason"}:
            out.append(dict(row))
    return out


def _season_games_threshold(season_length: int, share: float, minimum: int = 1) -> int:
    return max(minimum, int(math.ceil(float(season_length) * float(share))))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _defn(
    award_id: str,
    name: str,
    *,
    recipient_type: str,
    category: str,
    display_metric: str,
    official: bool = True,
    watch_enabled: bool = True,
    ceremony_enabled: bool = True,
    eligibility: str = "",
    score: str = "",
    tiebreakers: Optional[List[str]] = None,
    required_fields: Optional[List[str]] = None,
    finalist_count: int = 3,
    supports_shared_winners: bool = False,
    public_status: str = "official",
    watch_type: str = "official_live_race",
) -> Dict[str, Any]:
    return {
        "award_id": award_id,
        "name": name,
        "recipient_type": recipient_type,
        "category": category,
        "official": official,
        "watch_enabled": watch_enabled,
        "ceremony_enabled": ceremony_enabled,
        "eligibility": eligibility or award_id,
        "score": score or award_id,
        "tiebreakers": list(tiebreakers or []),
        "required_fields": list(required_fields or []),
        "finalist_count": int(finalist_count),
        "supports_shared_winners": bool(supports_shared_winners),
        "display_metric": display_metric,
        "public_status": public_status,
        "watch_type": watch_type,
    }


AWARD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "presidents": _defn(
        "presidents",
        "Presidents' Trophy",
        recipient_type="team",
        category="team_result",
        display_metric="PTS",
        watch_type="official_live_race",
        eligibility="presidents",
        score="presidents",
        tiebreakers=["points", "regulation_wins", "wins", "goal_diff"],
    ),
    "stanley": _defn(
        "stanley",
        "Stanley Cup",
        recipient_type="team",
        category="playoff",
        display_metric="Champion",
        watch_enabled=False,
        watch_type="official_live_race",
    ),
    "conference_champions": _defn(
        "conference_champions",
        "Conference Champions",
        recipient_type="team",
        category="playoff",
        display_metric="Champion",
        watch_enabled=False,
        ceremony_enabled=False,
        supports_shared_winners=True,
    ),
    "art_ross": _defn(
        "art_ross",
        "Art Ross Trophy",
        recipient_type="player",
        category="stat_race",
        display_metric="PTS",
        supports_shared_winners=True,
        eligibility="art_ross",
        score="art_ross",
        tiebreakers=["points", "goals", "ppg", "shared"],
    ),
    "rocket": _defn(
        "rocket",
        "Rocket Richard Trophy",
        recipient_type="player",
        category="stat_race",
        display_metric="G",
        supports_shared_winners=True,
        eligibility="rocket",
        score="rocket",
        tiebreakers=["goals", "fewer_gp", "points", "shared"],
    ),
    "hart": _defn(
        "hart",
        "Hart Memorial Trophy",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        required_fields=["gp"],
        eligibility="hart",
        score="hart",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "norris": _defn(
        "norris",
        "James Norris Memorial Trophy",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        required_fields=["gp"],
        eligibility="norris",
        score="norris",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "selke": _defn(
        "selke",
        "Frank J. Selke Trophy",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        required_fields=["gp"],
        eligibility="selke",
        score="selke",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "calder": _defn(
        "calder",
        "Calder Memorial Trophy",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        required_fields=["gp"],
        eligibility="calder",
        score="calder",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "vezina": _defn(
        "vezina",
        "Vezina Trophy",
        recipient_type="goalie",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        required_fields=["gp"],
        eligibility="vezina",
        score="vezina",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "conn_smythe": _defn(
        "conn_smythe",
        "Conn Smythe Trophy",
        recipient_type="player",
        category="playoff",
        display_metric="Playoff ballot points",
        watch_type="official_projected_ballot",
        eligibility="conn_smythe",
        score="conn_smythe",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "jennings": _defn(
        "jennings",
        "William M. Jennings Trophy",
        recipient_type="multiple",
        category="team_result",
        display_metric="Team GA",
        supports_shared_winners=True,
        eligibility="jennings",
        score="jennings",
        tiebreakers=["team_ga", "shared"],
    ),
    "lady_byng": _defn(
        "lady_byng",
        "Lady Byng Memorial Trophy",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        ceremony_enabled=True,
        eligibility="lady_byng",
        score="lady_byng",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "ted_lindsay": _defn(
        "ted_lindsay",
        "Ted Lindsay Award",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        ceremony_enabled=True,
        eligibility="ted_lindsay",
        score="ted_lindsay",
        required_fields=["gp"],
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "masterton": _defn(
        "masterton",
        "Bill Masterton Memorial Trophy",
        recipient_type="player",
        category="selection",
        display_metric="Selection",
        watch_enabled=False,
        watch_type="watch_only",
        ceremony_enabled=True,
        eligibility="masterton",
        score="masterton",
        required_fields=["gp"],
    ),
    "messier": _defn(
        "messier",
        "Mark Messier Leadership Award",
        recipient_type="player",
        category="selection",
        display_metric="Selection",
        watch_enabled=False,
        watch_type="watch_only",
        ceremony_enabled=True,
        eligibility="messier",
        score="messier",
        required_fields=["gp"],
    ),
    "jack_adams": _defn(
        "jack_adams",
        "Jack Adams Award",
        recipient_type="coach",
        category="ballot",
        display_metric="Ballot points",
        watch_enabled=True,
        watch_type="official_live_race",
        ceremony_enabled=True,
        eligibility="jack_adams",
        score="jack_adams",
        required_fields=["gp"],
    ),
    "gm_of_the_year": _defn(
        "gm_of_the_year",
        "GM of the Year",
        recipient_type="gm",
        category="ballot",
        display_metric="Wins above expected",
        watch_enabled=True,
        watch_type="official_live_race",
        ceremony_enabled=True,
        eligibility="gm_of_the_year",
        score="gm_of_the_year",
        required_fields=["gp"],
    ),
    "all_star_1": _defn(
        "all_star_1",
        "First NHL All-Star Team",
        recipient_type="multiple",
        category="selection",
        display_metric="Selection",
        supports_shared_winners=True,
        watch_enabled=False,
        ceremony_enabled=True,
    ),
    "all_star_2": _defn(
        "all_star_2",
        "Second NHL All-Star Team",
        recipient_type="multiple",
        category="selection",
        display_metric="Selection",
        supports_shared_winners=True,
        watch_enabled=False,
        ceremony_enabled=True,
    ),
}

NAME_TO_ID = {v["name"]: k for k, v in AWARD_REGISTRY.items()}
# Legacy short names used in ceremony catalogs.
NAME_TO_ID.update(
    {
        "Norris Trophy": "norris",
        "Selke Trophy": "selke",
        "Maurice Richard Trophy": "rocket",
        "James Norris Memorial Trophy": "norris",
        "Frank J. Selke Trophy": "selke",
    }
)


@dataclass
class Award:
    name: str
    winner_team_id: Optional[str] = None
    winner_name: Optional[str] = None
    winner_player_id: Optional[str] = None
    winner_team_name: Optional[str] = None
    finalists: List[Any] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    winner_stats: Optional[Dict[str, Any]] = None
    rationale: str = ""
    # Extended canonical fields
    award_id: str = ""
    official: bool = True
    status: str = "complete"
    category: str = ""
    recipient_type: str = "player"
    winners: List[Dict[str, Any]] = field(default_factory=list)
    shared: bool = False
    full_results: List[Dict[str, Any]] = field(default_factory=list)
    display_metric: str = ""
    calculation_quality: str = "full"
    fallback_reason: Optional[str] = None
    public_rationale: str = ""
    eligibility_summary: str = ""
    stat_scope: str = "regular_season"
    season: Any = None
    unavailable_reason: Optional[str] = None
    voting: Optional[Dict[str, Any]] = None
    result: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Identity / Calder
# ---------------------------------------------------------------------------

def _player_from_rosters(teams: Optional[Sequence[Any]], pid: str) -> Any:
    if not teams or not pid:
        return None
    for t in teams:
        for p in getattr(t, "roster", None) or []:
            if str(getattr(p, "id", "") or "") == str(pid):
                return p
    return None


def _player_age(row: Mapping[str, Any], teams: Optional[Sequence[Any]] = None) -> Optional[int]:
    for key in ("age", "player_age", "season_age"):
        if row.get(key) is not None:
            return _safe_int(row.get(key), 0) or None
    p = _player_from_rosters(teams, _pid(row))
    if p is None:
        return None
    try:
        return int(getattr(p, "age", getattr(getattr(p, "identity", None), "age", None)))
    except Exception:
        return None


def calder_eligibility(
    row: Mapping[str, Any],
    *,
    teams: Optional[Sequence[Any]] = None,
    history: Optional[Mapping[str, Any]] = None,
    season_length: int = 82,
) -> Dict[str, Any]:
    """
    Canonical Calder eligibility.

    Uses season-history style fields when present:
      previous_nhl_gp, prior_nhl_seasons, nhl_gp_before, seasons_played, is_rookie/rookie
    Age is only one part of the rule (must be <= 25 for first NHL season under NHL-style caps).
    """
    pid = _pid(row)
    hist = dict(history or {})
    if pid and isinstance(hist.get(pid), Mapping):
        hist = {**hist, **dict(hist.get(pid) or {})}

    gp = _gp(row)
    min_gp = _season_games_threshold(season_length, 0.30, minimum=20)
    age = _player_age(row, teams)
    # NHL age rule is "under 26 on Sept. 15 of the season" — prefer the season-start
    # age computed from the birth date over the (possibly already aged-up) roster age.
    if hist.get("age_sept15") is not None:
        age = _safe_int(hist.get("age_sept15"), 0) or age
    per_season_facts = hist.get("max_prior_season_gp") is not None

    prior_gp = None
    for key in ("previous_nhl_gp", "prior_nhl_gp", "nhl_gp_before", "career_nhl_gp_before_season"):
        if row.get(key) is not None or hist.get(key) is not None:
            prior_gp = _safe_int(row.get(key, hist.get(key)), 0)
            break

    prior_seasons = None
    for key in ("prior_nhl_seasons", "nhl_seasons_before", "seasons_played", "pro_seasons"):
        if row.get(key) is not None or hist.get(key) is not None:
            prior_seasons = _safe_int(row.get(key, hist.get(key)), 0)
            break

    flagged_rookie = None
    if "is_rookie" in row or "rookie" in row or "is_rookie" in hist or "rookie" in hist:
        flagged_rookie = bool(row.get("is_rookie", row.get("rookie", hist.get("is_rookie", hist.get("rookie")))))

    draft_year = _safe_int(row.get("draft_year", hist.get("draft_year")), 0)
    rookie_class = _safe_int(row.get("rookie_class_year", hist.get("rookie_class_year")), 0)
    # A stored false flag used to make every NHL player ineligible, because the
    # stats row writes is_rookie=False whenever the field was missing.
    # Rookies are the current draft class, or players with no prior season rows.
    live_min = 1 if row.get("live_race") or hist.get("live_race") else min_gp
    if draft_year and rookie_class and draft_year == rookie_class:
        return {
            "eligible": gp >= live_min,
            "confidence": "full",
            "eligibility_confidence": "full",
            "details": {
                "gp": gp,
                "min_gp": live_min,
                "age": age,
                "draft_year": draft_year,
                "prior_nhl_seasons": prior_seasons,
                "is_rookie_flag": True,
                "reasons": [f"Drafted in {draft_year}."],
            },
        }
    if prior_seasons is not None:
        if prior_seasons <= 0:
            # Missing age is not a rookie. The roster stamp sets is_rookie only for
            # the draft class and players 25-and-under with no prior season rows.
            if flagged_rookie is False:
                young_enough = False
            elif flagged_rookie is True:
                young_enough = True
            else:
                young_enough = age is not None and age <= 25
            return {
                "eligible": gp >= live_min and young_enough,
                "confidence": "full",
                "eligibility_confidence": "full",
                "details": {
                    "gp": gp,
                    "min_gp": live_min,
                    "age": age,
                    "draft_year": draft_year or None,
                    "prior_nhl_seasons": 0,
                    "is_rookie_flag": True,
                    "reasons": ["No prior NHL season on the roster."],
                },
            }
        return {
            "eligible": False,
            "confidence": "full",
            "eligibility_confidence": "full",
            "details": {
                "gp": gp,
                "min_gp": live_min,
                "age": age,
                "draft_year": draft_year or None,
                "prior_nhl_seasons": prior_seasons,
                "is_rookie_flag": False,
                "reasons": ["Already has NHL season rows on the roster."],
            },
        }

    confidence = "full"
    reasons: List[str] = []

    if prior_gp is None and prior_seasons is None and flagged_rookie is None:
        confidence = "fallback"
        seasons_in_league = None
        for key in ("seasons_in_league", "nhl_seasons", "league_seasons_played", "seasons_played"):
            if row.get(key) is not None or hist.get(key) is not None:
                seasons_in_league = _safe_int(row.get(key, hist.get(key)), 0)
                break
        if age is None:
            eligible = False
            reasons.append("Missing rookie history and age; conservative deny.")
        elif age <= 23 and gp >= min_gp:
            if seasons_in_league is not None and seasons_in_league > 1:
                eligible = False
                reasons.append("Fallback: already more than one league season.")
            else:
                if seasons_in_league is None:
                    log_fallback_degradation("calder", ["seasons_in_league"])
                eligible = True
                reasons.append("Fallback: age<=23 with meaningful GP and no prior history fields.")
        else:
            eligible = False
            reasons.append("Fallback: insufficient evidence of first NHL season.")
        return {
            "eligible": eligible,
            "confidence": confidence,
            "eligibility_confidence": confidence,
            "details": {
                "gp": gp,
                "min_gp": min_gp,
                "age": age,
                "prior_nhl_gp": prior_gp,
                "prior_nhl_seasons": prior_seasons,
                "seasons_in_league": seasons_in_league,
                "is_rookie_flag": flagged_rookie,
                "reasons": reasons,
            },
        }

    # NHL-style thresholds (documented approximation using available fields).
    prior_gp_ok = True if prior_gp is None else prior_gp < 25
    seasons_ok = True if prior_seasons is None else prior_seasons <= 0
    age_ok = True if age is None else age <= 25
    gp_ok = gp >= min_gp

    if flagged_rookie is True and per_season_facts:
        # Flag was derived from per-season NHL GP (>25 GP in any prior season, or 6+ GP
        # in each of two prior seasons, disqualifies) — career totals do not apply.
        eligible = gp_ok and age_ok
        reasons.append(str(hist.get("calder_basis") or "Per-season NHL rookie rule with participation and age checks."))
    elif flagged_rookie is True:
        eligible = gp_ok and age_ok and prior_gp_ok
        reasons.append("Trusted is_rookie/rookie flag with participation and age checks.")
    elif flagged_rookie is False and not (per_season_facts and prior_gp_ok and (True if prior_seasons is None else prior_seasons < 2)):
        eligible = False
        reasons.append(str(hist.get("calder_basis") or "is_rookie/rookie flag is false."))
    else:
        eligible = gp_ok and age_ok and prior_gp_ok and seasons_ok
        reasons.append("History-derived first-year checks.")

    if prior_gp is None or prior_seasons is None:
        confidence = "fallback" if confidence == "full" and flagged_rookie is None else confidence

    return {
        "eligible": bool(eligible),
        "confidence": confidence,
        "eligibility_confidence": confidence,
        "details": {
            "gp": gp,
            "min_gp": min_gp,
            "age": age,
            "prior_nhl_gp": prior_gp,
            "prior_nhl_seasons": prior_seasons,
            "max_prior_season_gp": hist.get("max_prior_season_gp"),
            "is_rookie_flag": flagged_rookie,
            "age_ok": age_ok,
            "gp_ok": gp_ok,
            "reasons": reasons,
        },
    }


# ---------------------------------------------------------------------------
# Team context / snapshots
# ---------------------------------------------------------------------------

def _team_name_from_id(teams: Dict[str, Any], tid: str, default: Optional[str] = None) -> str:
    t = teams.get(str(tid))
    if t is None:
        return default or str(tid)
    name = getattr(t, "name", None)
    city = getattr(t, "city", None)
    if city and name:
        return f"{city} {name}"
    if name:
        return str(name)
    return default or str(tid)


def _standing_stats(rec: TeamStandingRecord) -> Dict[str, Any]:
    gp = max(1, _safe_int(getattr(rec, "games_played", None), _safe_int(getattr(rec, "gp", None), 0)) or (_safe_int(rec.wins) + _safe_int(rec.losses) + _safe_int(rec.otl)))
    pts = int(getattr(rec, "points", 0) or 0)
    return {
        "points": pts,
        "pts": pts,
        "wins": int(getattr(rec, "wins", 0) or 0),
        "w": int(getattr(rec, "wins", 0) or 0),
        "losses": int(getattr(rec, "losses", 0) or 0),
        "l": int(getattr(rec, "losses", 0) or 0),
        "otl": int(getattr(rec, "otl", 0) or 0),
        "goals_for": int(getattr(rec, "gf", 0) or 0),
        "goals_against": int(getattr(rec, "ga", 0) or 0),
        "goal_diff": int(rec.goal_diff()),
        "record": f"{int(rec.wins)}-{int(rec.losses)}-{int(rec.otl)}",
        "points_pct": float(pts) / float(2 * gp) if gp else 0.0,
        "gp": gp,
    }


def build_team_context(standings: StandingsTable) -> Dict[str, Dict[str, Any]]:
    tbl = list(standings.league_table() or [])
    if not tbl:
        return {}
    pts_pcts = []
    gds = []
    for rec in tbl:
        st = _standing_stats(rec)
        pts_pcts.append(st["points_pct"])
        gds.append(float(st["goal_diff"]))
    playoff_cut = 16 if len(tbl) >= 16 else max(1, len(tbl) // 2)
    out: Dict[str, Dict[str, Any]] = {}
    for i, rec in enumerate(tbl):
        st = _standing_stats(rec)
        tid = str(rec.team_id)
        out[tid] = {
            **st,
            "league_rank": i + 1,
            "points_pct_norm": percentile_rank(pts_pcts, st["points_pct"]),
            "goal_diff_norm": percentile_rank(gds, float(st["goal_diff"])),
            "playoff_qualified": i < playoff_cut,
            "distance_from_cutoff": float(playoff_cut - (i + 1)),
            "standing_percentile": 1.0 - (float(i) / float(max(1, len(tbl) - 1))),
        }
    return out


def snapshot_row(row: Mapping[str, Any], *, teams: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    snap = dict(row)
    snap["entity_id"] = _pid(row)
    snap["player_id"] = _pid(row)
    snap["team_id"] = _tid(row)
    snap["position"] = _pos(row)
    snap["age"] = _player_age(row, teams)
    snap["gp"] = _gp(row)
    if "is_rookie" in row or "rookie" in row:
        snap["is_rookie"] = bool(row.get("is_rookie", row.get("rookie")))
    snap["previous_nhl_gp"] = row.get("previous_nhl_gp", row.get("prior_nhl_gp"))
    snap["is_captain"] = bool(row.get("is_captain", row.get("captain", False)))
    return snap


# ---------------------------------------------------------------------------
# Goalie workload / EN handling
# ---------------------------------------------------------------------------

def goalie_sv_pct(row: Mapping[str, Any]) -> float:
    if row.get("sv_pct") is not None:
        return normalize_percentage(row.get("sv_pct"))
    sa = _safe_int(row.get("shots_against"), 0)
    en = _safe_int(row.get("empty_net_goals"), _safe_int(row.get("en_goals"), 0))
    ga = max(0, _safe_int(row.get("ga"), _safe_int(row.get("goals_against"))) - en)
    saves = _safe_int(row.get("saves"), 0)
    if saves <= 0 and sa > 0:
        saves = max(0, sa - ga)
    denom = saves + ga
    return float(saves) / float(denom) if denom > 0 else 0.0


def goalie_workload_ok(row: Mapping[str, Any], *, season_length: int = 82) -> Tuple[bool, Dict[str, Any]]:
    gp = _gp(row)
    gs = _safe_int(row.get("games_started"), _safe_int(row.get("gs"), gp))
    minutes = _safe_float(row.get("minutes"), _safe_float(row.get("toi_minutes"), _safe_float(row.get("toi"), 0.0)))
    if minutes <= 0 and gp > 0:
        # Derive approximate minutes only from starts when available.
        minutes = float(gs) * 60.0
    shots = _safe_int(row.get("shots_against"), 0)
    team_minutes = _safe_float(row.get("team_goalie_minutes"), float(season_length) * 60.0)
    share = minutes / team_minutes if team_minutes > 0 else 0.0
    min_gs = _season_games_threshold(season_length, 0.30, minimum=20)
    min_minutes = float(min_gs) * 50.0
    ok = (gs >= min_gs) or (minutes >= min_minutes and shots >= min_gs * 22)
    # Pure relief: high appearances, low starts/minutes → not eligible
    if gp >= min_gs and gs < max(8, min_gs // 3) and minutes < min_minutes:
        ok = False
    return ok, {
        "gp": gp,
        "games_started": gs,
        "minutes": minutes,
        "shots_against": shots,
        "team_minutes_share": share,
        "min_gs": min_gs,
        "eligible": ok,
    }


def log_fallback_degradation(award_id: str, missing_fields: Sequence[str]) -> None:
    fields = [str(f) for f in missing_fields if f]
    if not fields:
        return
    logger.info(
        "Award %s fallback degraded; missing fields: %s",
        str(award_id),
        ", ".join(fields),
    )


def _optional_stat_per_gp(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    gp = max(1, _gp(row))
    for key in keys:
        if key in row and row.get(key) is not None:
            return float(_safe_float(row.get(key))) / float(gp)
    return None


def _toi_per_gp_normalized(row: Mapping[str, Any]) -> float:
    gp = max(1, _gp(row))
    toi_pg = None
    for key in ("toi_per_game", "toi_per_gp", "avg_toi"):
        if row.get(key) is not None:
            toi_pg = _safe_float(row.get(key))
            break
    if toi_pg is None:
        total_toi = _safe_float(row.get("toi"), _safe_float(row.get("toi_minutes"), 0.0))
        if total_toi > 0:
            toi_pg = total_toi / float(gp)
    if toi_pg is None or toi_pg <= 0:
        return 0.0
    return min(1.5, toi_pg / 22.0)


def hart_fallback_formula(row: Mapping[str, Any]) -> float:
    ppg = float(_pts(row)) / max(1, _gp(row))
    avail = min(1.0, float(_gp(row)) / 70.0)
    offense = ppg * 40.0
    availability = avail * 10.0
    row["_fallback_terms"] = {
        "production_component": offense,
        "availability_component": availability,
        "team_context_component": 0.0,
        "two_way_component": 0.0,
        "individual_value_component": offense * 0.25,
    }
    return offense + availability


def norris_fallback_formula(row: Mapping[str, Any], *, award_id: str = "norris") -> float:
    ppg = float(_pts(row)) / max(1, _gp(row))
    offense_term = ppg * 12.0
    toi_norm = _toi_per_gp_normalized(row)
    toi_term = toi_norm * 8.0
    blocked_pg = _optional_stat_per_gp(row, ("blocked_shots", "blk", "blocks"))
    giveaways_pg = _optional_stat_per_gp(row, ("giveaways", "giv", "gv"))
    missing: List[str] = []
    defense_term = 0.0
    if blocked_pg is not None:
        defense_term += blocked_pg * 3.0
    else:
        missing.append("blocked_shots")
    if giveaways_pg is not None:
        defense_term -= giveaways_pg * 2.0
    else:
        missing.append("giveaways")
    if blocked_pg is None and giveaways_pg is None:
        log_fallback_degradation(award_id, missing)
    elif missing:
        log_fallback_degradation(award_id, missing)
    row["_fallback_terms"] = {
        "production_component": offense_term,
        "two_way_component": defense_term + toi_term,
        "availability_component": toi_term,
        "individual_value_component": offense_term * 0.35,
        "team_context_component": toi_norm * 4.0,
    }
    return offense_term + defense_term + toi_term


def selke_fallback_formula(row: Mapping[str, Any], *, award_id: str = "selke") -> float:
    toi_norm = _toi_per_gp_normalized(row)
    toi_term = toi_norm * 10.0
    fo_taken = _safe_int(row.get("fo_taken"), _safe_int(row.get("faceoffs_taken"), 0))
    fo_pct = normalize_percentage(row.get("faceoff_pct")) if fo_taken > 0 else 0.5
    fo_term = fo_pct * 0.15
    pk_toi_pg = _optional_stat_per_gp(row, ("pk_toi", "sh_toi", "pk_toi_per_game"))
    if pk_toi_pg is None:
        pk_share = _safe_float(row.get("pk_toi_share"), 0.0)
        if pk_share > 0:
            pk_toi_pg = pk_share
        else:
            log_fallback_degradation(award_id, ["pk_toi"])
            pk_toi_pg = 0.0
    pk_term = pk_toi_pg * 6.0
    pts_penalty = float(_pts(row)) * 0.05
    row["_fallback_terms"] = {
        "production_component": fo_term,
        "two_way_component": toi_term + pk_term - pts_penalty,
        "availability_component": toi_term,
        "individual_value_component": -pts_penalty,
        "team_context_component": pk_term,
    }
    return toi_term + fo_term + pk_term - pts_penalty


def vezina_fallback_formula(row: Mapping[str, Any]) -> float:
    # Save quality must dominate: .033 of SV% is the gap between elite and replacement,
    # so it is scaled to ~40 points while workload tops out at ~20 (the old sv*50 + GP
    # mix let a 66-GP .887 starter beat every .915+ goalie).
    sv_raw = goalie_sv_pct(row)
    gp = float(_gp(row))
    sv = max(0.0, (sv_raw - 0.870) * 1200.0)
    workload = min(gp, 65.0) / 65.0 * 20.0 + _safe_float(row.get("so", row.get("shutouts")), 0.0) * 0.75
    row["_fallback_terms"] = {
        "production_component": sv,
        "goals_saved_above_expected": sv * 0.4,
        "workload": workload,
        "availability_component": workload,
        "individual_value_component": sv * 0.55,
    }
    return sv + workload + _safe_float(row.get("w", row.get("wins")), 0.0) * 0.15


def calder_fallback_formula(
    row: Mapping[str, Any],
    team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> float:
    if _is_goalie(row):
        return vezina_fallback_formula(row)
    if _is_defense(row):
        return norris_fallback_formula(row, award_id="calder")
    hart = hart_fallback_formula(row)
    hart_terms = dict(row.get("_fallback_terms") or {})
    selke = selke_fallback_formula(row, award_id="calder")
    selke_terms = dict(row.get("_fallback_terms") or {})
    row["_fallback_terms"] = {
        "production_component": hart_terms.get("production_component", hart * 0.6) * 0.6
        + selke_terms.get("production_component", selke * 0.1) * 0.4,
        "two_way_component": hart_terms.get("two_way_component", 0.0) * 0.6
        + selke_terms.get("two_way_component", selke * 0.7) * 0.4,
        "individual_value_component": hart * 0.55 + selke * 0.25,
        "availability_component": hart_terms.get("availability_component", 0.0) * 0.6
        + selke_terms.get("availability_component", 0.0) * 0.4,
        "team_context_component": selke_terms.get("team_context_component", 0.0) * 0.4,
    }
    return 0.6 * hart + 0.4 * selke


def derive_pseudo_components(row: Mapping[str, Any], award_id: str, canonical_score: float) -> Dict[str, float]:
    terms = dict(row.get("_fallback_terms") or {})
    if terms:
        return terms
    aid = str(award_id or "").lower()
    if aid == "norris":
        norris_fallback_formula(row, award_id=aid)
        return dict(row.get("_fallback_terms") or {})
    if aid == "selke":
        selke_fallback_formula(row, award_id=aid)
        return dict(row.get("_fallback_terms") or {})
    if aid == "hart":
        hart_fallback_formula(row)
        return dict(row.get("_fallback_terms") or {})
    if aid == "vezina":
        vezina_fallback_formula(row)
        return dict(row.get("_fallback_terms") or {})
    if aid == "calder":
        calder_fallback_formula(row)
        return dict(row.get("_fallback_terms") or {})
    if aid == "lady_byng":
        score = lady_byng_score(row)
        return {
            "production_component": float(_pts(row)) / max(1, _gp(row)) * 40.0,
            "two_way_component": _safe_float(row.get("discipline_score"), 50.0) * 0.4,
            "individual_value_component": score * 0.5,
        }
    if aid == "ted_lindsay":
        ppg = float(_pts(row)) / max(1, _gp(row))
        return {
            "production_component": ppg * 38.0,
            "individual_value_component": _safe_float(row.get("impact_score"), float(_pts(row)) * 0.4) * 0.35,
            "two_way_component": normalize_percentage(row.get("xgf_pct")) * 10.0,
        }
    return {
        "production_component": float(canonical_score) * 0.55,
        "two_way_component": float(canonical_score) * 0.25,
        "individual_value_component": float(canonical_score) * 0.20,
    }


def _merge_history_from_rosters(
    teams: Optional[Sequence[Any]],
    history_by_player: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(history_by_player or {})
    for team in teams or []:
        for player in getattr(team, "roster", None) or []:
            pid = str(getattr(player, "id", "") or "")
            if not pid:
                continue
            blob = getattr(player, "player_award_history", None)
            if isinstance(blob, dict):
                base = dict(merged.get(pid) or {})
                base.update(blob)
                merged[pid] = base
    return merged


# BLOCKED: no injury ledger accessible from awards.py scope — Masterton cannot populate until
# upstream ledger exposed. Searched FranchiseSession.injury_log_all / injury_log_major in
# backend/services/franchise_sim.py; compute_awards() receives player_season_stats only.
def populate_masterton_inputs(player_id: str, season: Any, injury_ledger: Optional[Sequence[Mapping[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    if not injury_ledger:
        return None
    missed = 0
    returned = False
    for inj in injury_ledger:
        if str(inj.get("player_id") or "") != str(player_id):
            continue
        missed += _safe_int(inj.get("games"), _safe_int(inj.get("games_initial"), 0))
        if str(inj.get("status") or "").upper() in {"ACTIVE", "RETURNED", "CLEARED"}:
            returned = True
    if missed <= 0 and not returned:
        return None
    return {"injury_games_missed": missed, "games_returned": returned}


# ---------------------------------------------------------------------------
# Component scoring
# ---------------------------------------------------------------------------

def _component_bundle(
    pool: Sequence[Mapping[str, Any]],
    specs: Mapping[str, Callable[[Mapping[str, Any]], float]],
    weights: Mapping[str, float],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], str, Optional[str]]:
    """Normalize each component across the pool, then weighted sum → canonical score."""
    norms: Dict[str, Dict[str, float]] = {k: normalize_pool_metric(pool, fn) for k, fn in specs.items()}
    quality = "full"
    reason = None
    # If all raw values are zero for a heavy component, mark documented_fallback
    scores: Dict[str, float] = {}
    for row in pool:
        pid = _pid(row)
        parts = {k: norms[k].get(pid, 0.0) for k in specs}
        total_w = sum(float(weights.get(k, 0.0)) for k in specs) or 1.0
        score = sum(parts[k] * float(weights.get(k, 0.0)) for k in specs) / total_w
        scores[pid] = score
        # stash components on mutable row copy handled by caller
        row.setdefault("_components", {})
        row["_components"] = parts  # type: ignore[index]
    return { _pid(r): dict(r.get("_components") or {}) for r in pool }, scores, quality, reason


def hart_components_for_row(row: Mapping[str, Any], team_ctx: Mapping[str, Any]) -> Dict[str, float]:
    pts = float(_pts(row))
    gp = max(1, _gp(row))
    production = pts / gp
    two_way = _safe_float(row.get("defense_score"), 0.0) * 0.5 + normalize_percentage(row.get("xgf_pct")) * 50.0
    individual = _safe_float(row.get("war"), _safe_float(row.get("impact_score"), pts * 0.02))
    team = (
        _safe_float(team_ctx.get("points_pct_norm"), 0.5) * 0.55
        + _safe_float(team_ctx.get("goal_diff_norm"), 0.5) * 0.25
        + (0.2 if team_ctx.get("playoff_qualified") else 0.0)
    )
    availability = min(1.0, float(gp) / 70.0)
    return {
        "production_component": production,
        "two_way_component": two_way,
        "individual_value_component": individual,
        "team_context_component": team,
        "availability_component": availability,
    }


def _as_share(value: Any) -> Optional[float]:
    """0-1 rate. Values above 1.5 are treated as 0-100 percentages."""
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    if abs(x) > 1.5:
        x = x / 100.0
    return x


def _share_from_counts(for_value: Any, against_value: Any) -> Optional[float]:
    try:
        f = float(for_value)
        a = float(against_value)
    except (TypeError, ValueError):
        return None
    if f + a <= 0:
        return None
    return f / (f + a)


def _team_success(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]]) -> float:
    ctx = dict((team_context_by_tid or {}).get(_tid(row), {}))
    return max(0.0, min(1.0, _safe_float(ctx.get("points_pct_norm"), 0.5)))


def prepare_award_context(rows: Sequence[Any]) -> None:
    """Stamp relative on-ice rates and team point share onto mutable stat rows."""
    sums: Dict[str, Dict[str, List[float]]] = {}
    points: Dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        tid = _tid(row) or "_"
        cf = _as_share(row.get("cf_pct"))
        if cf is None:
            cf = _share_from_counts(row.get("cf") or row.get("corsi_for"), row.get("ca") or row.get("corsi_against"))
        xgf = _as_share(row.get("xgf_pct"))
        if xgf is None:
            xgf = _share_from_counts(row.get("xgf"), row.get("xga"))
        gf = _as_share(row.get("gf_pct") if row.get("gf_pct") is not None else row.get("on_ice_gf_pct"))
        if gf is None:
            gf = _share_from_counts(
                row.get("gf_on") if row.get("gf_on") is not None else row.get("on_ice_gf"),
                row.get("ga_on") if row.get("ga_on") is not None else row.get("on_ice_ga"),
            )
        row["_cf_pct"] = cf
        row["_xgf_pct"] = xgf
        row["_gf_pct"] = gf
        xga60 = row.get("xga_per_60")
        try:
            xga60_f = float(xga60) if xga60 is not None else 0.0
        except (TypeError, ValueError):
            xga60_f = 0.0
        if xga60_f <= 0:
            try:
                xga_f = float(row.get("xga") or 0)
                toi_f = float(row.get("toi_sec") or 0)
            except (TypeError, ValueError):
                xga_f, toi_f = 0.0, 0.0
            row["_xga60"] = (xga_f / (toi_f / 3600.0)) if toi_f > 0 and xga_f > 0 else None
        else:
            row["_xga60"] = xga60_f
        bucket = sums.setdefault(tid, {"cf": [0.0, 0.0], "xgf": [0.0, 0.0], "gf": [0.0, 0.0]})
        for key, val in (("cf", cf), ("xgf", xgf), ("gf", gf)):
            if val is None:
                continue
            bucket[key][0] += float(val)
            bucket[key][1] += 1.0
        if not _is_goalie(row):
            points[tid] = points.get(tid, 0.0) + float(_pts(row))
    for row in rows:
        if not isinstance(row, dict):
            continue
        tid = _tid(row) or "_"
        bucket = sums.get(tid) or {}

        def _rel(stored_key: str, own_key: str, mean_key: str) -> float:
            stored = _as_share(row.get(stored_key))
            if stored is not None:
                return stored
            own = row.get(own_key)
            mean = bucket.get(mean_key) or [0.0, 0.0]
            if own is None or mean[1] <= 0:
                return 0.0
            return float(own) - (mean[0] / mean[1])

        row["_rel_cf"] = _rel("rel_cf_pct", "_cf_pct", "cf")
        row["_rel_xgf"] = _rel("rel_xgf_pct", "_xgf_pct", "xgf")
        row["_rel_gf"] = _rel("rel_gf_pct", "_gf_pct", "gf")
        team_pts = points.get(tid, 0.0)
        if row.get("team_point_share") is not None:
            row["_team_point_share"] = _safe_float(row.get("team_point_share"), 0.0)
        elif team_pts > 0 and not _is_goalie(row):
            row["_team_point_share"] = float(_pts(row)) / team_pts
        else:
            row["_team_point_share"] = 0.0


def _hart_core(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]]) -> float:
    """Scoring, how much of the team runs through him, and whether that team is winning."""
    return (
        float(_pts(row)) * 1.15
        + _safe_float(row.get("_team_point_share"), _safe_float(row.get("team_point_share"), 0.0)) * 55.0
        + _team_success(row, team_context_by_tid) * 22.0
        + _safe_float(row.get("war"), 0.0) * 9.0
    )


def hart_ballot_score(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]] = None, rank_by_tid: Optional[Mapping[str, int]] = None) -> float:
    """Points, team importance, team success, relative corsi, and WAR."""
    ctx = team_context_by_tid
    if not ctx and rank_by_tid is not None:
        rk = int(rank_by_tid.get(_tid(row), 16) or 16)
        teams_n = max(2, int(rank_by_tid.get("__team_count__", 32) or 32))
        ctx = {_tid(row): {"points_pct_norm": max(0.0, 1.0 - (rk - 1) / float(teams_n - 1))}}
    rel_cf = _safe_float(row.get("_rel_cf"), 0.0)
    if row.get("_rel_cf") is None:
        rel_cf = _as_share(row.get("rel_cf_pct")) or 0.0
    return _hart_core(row, ctx) + rel_cf * 160.0


def norris_ballot_score(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]] = None) -> float:
    """Hart ingredients for a defenceman, with relative on-ice scoring in place of raw corsi."""
    rel_on = row.get("_rel_xgf")
    if rel_on is None:
        rel_on = _as_share(row.get("rel_xgf_pct"))
    if rel_on is None:
        rel_on = row.get("_rel_cf") if row.get("_rel_cf") is not None else (_as_share(row.get("rel_cf_pct")) or 0.0)
    xga = row.get("_xga60")
    xga_term = (2.55 - float(xga)) * 8.0 if xga is not None else 0.0
    return _hart_core(row, team_context_by_tid) + float(rel_on) * 170.0 + xga_term


def selke_ballot_score(row: Mapping[str, Any]) -> float:
    """Defensive on-ice results. Penalty-kill time is not required."""
    xga = row.get("_xga60")
    xga_term = (2.55 - float(xga)) * 28.0 if xga is not None else 0.0
    rel = row.get("_rel_xgf")
    if rel is None:
        rel = _as_share(row.get("rel_xgf_pct"))
    if rel is None:
        rel = row.get("_rel_cf") if row.get("_rel_cf") is not None else (_as_share(row.get("rel_cf_pct")) or 0.0)
    gp = max(1, _gp(row))
    blocks = _safe_float(row.get("blocked_shots"), _safe_float(row.get("blocks"), _safe_float(row.get("blk"), 0.0)))
    toi_min = _toi_pg_minutes(row) * gp
    blocks_60 = (blocks / toi_min * 60.0) if toi_min > 0 else (blocks / gp)
    pm = _safe_float(row.get("plus_minus"), _safe_float(row.get("plusMinus"), 0.0))
    return (
        xga_term
        + float(rel) * 220.0
        + blocks_60 * 5.0
        + pm * 0.25
        + float(_pts(row)) * 0.10
    )


def vezina_ballot_score(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]] = None) -> float:
    ctx = dict((team_context_by_tid or {}).get(_tid(row), {}))
    sv = goalie_sv_pct(row)
    hd = normalize_percentage(row.get("high_danger_save_pct")) if row.get("high_danger_save_pct") is not None else sv
    gsax = _safe_float(row.get("gsax"), 0.0)
    workload = _safe_float(row.get("games_started"), _safe_float(row.get("gs"), float(_gp(row))))
    steal = _safe_float(row.get("steal_rate"), _safe_float(row.get("quality_start_pct"), 0.0))
    consistency = _safe_float(row.get("quality_start_pct"), sv)
    # Soft team defence adjustment: harder workload (worse team GD) gets slight lift.
    team_def = 1.0 - _safe_float(ctx.get("goal_diff_norm"), 0.5)
    return (
        gsax * 2.2
        + sv * 40.0
        + hd * 18.0
        + (workload / 70.0) * 12.0
        + normalize_percentage(steal) * 10.0
        + normalize_percentage(consistency) * 8.0
        + team_def * 6.0
        + min(1.0, float(_gp(row)) / 60.0) * 5.0
    )


def lady_byng_score(row: Mapping[str, Any]) -> float:
    """Most points, fewest penalty minutes."""
    return float(_pts(row)) - _safe_float(row.get("pim"), 0.0) * 0.85


def ted_lindsay_score(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]] = None) -> float:
    """Hart formula, with on-ice xGF% and GF% where Hart uses relative corsi."""
    xgf = row.get("_xgf_pct")
    if xgf is None:
        xgf = _as_share(row.get("xgf_pct"))
    gf = row.get("_gf_pct")
    if gf is None:
        gf = _as_share(row.get("gf_pct") if row.get("gf_pct") is not None else row.get("on_ice_gf_pct"))
    return (
        _hart_core(row, team_context_by_tid)
        + ((float(xgf) if xgf is not None else 0.5) - 0.5) * 90.0
        + ((float(gf) if gf is not None else 0.5) - 0.5) * 80.0
    )


def calder_position_score(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]] = None) -> float:
    # One scale for every position (the old mix compared Vezina-scale goalie scores
    # with Hart-scale skater scores, so rookie goalies swept the Calder).
    gp = max(1, _gp(row))
    if _is_goalie(row):
        sv = goalie_sv_pct(row)
        wins = _safe_float(row.get("w", row.get("wins")), 0.0)
        val = (sv - 0.900) * 1500.0 + wins * 0.9 + _safe_float(row.get("gsax"), 0.0) * 1.5
        return val * min(1.0, gp / 40.0)
    pts = float(_pts(row))
    toi_pg = _toi_pg_minutes(row)
    val = pts * (1.35 if _is_defense(row) else 1.0) + float(_goals(row)) * 0.25 + max(0.0, toi_pg - 12.0) * 1.5
    return val + _safe_float(row.get("plus_minus"), 0.0) * 0.15


def conn_smythe_score(row: Mapping[str, Any], *, champion_id: Optional[str] = None) -> float:
    gp = max(1, _gp(row))
    base = (float(_pts(row)) / gp) * 30.0 + float(_goals(row)) * 1.2
    if _is_goalie(row):
        base = goalie_sv_pct(row) * 55.0 + _safe_float(row.get("gsax"), 0.0) * 2.0 + float(gp) * 1.5
    elif _is_defense(row):
        base += _safe_float(row.get("defense_score"), 40.0) * 0.2
    else:
        base += _safe_float(row.get("defense_score"), 30.0) * 0.12
    base += _safe_float(row.get("elimination_game_points"), 0.0) * 2.5
    base += _safe_float(row.get("clutch_score"), 0.0) * 0.2
    base += _safe_float(row.get("final_round_points"), 0.0) * 2.0
    if champion_id and str(_tid(row)) == str(champion_id):
        base *= 1.18
    return base


# ---------------------------------------------------------------------------
# Ballot simulation
# ---------------------------------------------------------------------------

def simulate_award_ballots(
    scored_rows: Sequence[Tuple[float, Dict[str, Any]]],
    *,
    award_id: str,
    season_seed: Any,
    voter_count: int = VOTER_COUNT,
    points: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Deterministic individual ballots across voter archetypes.
    scored_rows: (canonical_score, row) already sorted optional.
    """
    if not scored_rows:
        return {
            "candidates": [],
            "voter_count": 0,
            "margin": 0.0,
            "seed": _seed_int(season_seed, award_id),
        }

    curve = [float(x) for x in (points or BALLOT_POINTS)] or list(BALLOT_POINTS)
    ordered = sorted(scored_rows, key=lambda p: p[0], reverse=True)
    rng = _rng(season_seed, "ballot", award_id)
    tallies: Dict[str, Dict[str, Any]] = {}
    for score, row in ordered:
        pid = _pid(row)
        comps = dict(row.get("_components") or row.get("component_scores") or {})
        if not comps or comps == {"canonical": float(score)}:
            comps = derive_pseudo_components(row, award_id, float(score))
        tallies[pid] = {
            "row": row,
            "canonical_score": float(score),
            "ballot_points": 0.0,
            "first_place_votes": 0,
            "placements": [0] * len(curve),
            "component_scores": comps,
        }

    archetype_bias = {
        "production": 0.12,
        "two_way": 0.08,
        "team_success": 0.10,
        "analytics": 0.09,
        "workload": 0.07,
        "traditional": 0.06,
    }

    for v in range(int(voter_count)):
        arch = VOTER_ARCHETYPES[v % len(VOTER_ARCHETYPES)]
        ranked = []
        for score, row in ordered:
            pid = _pid(row)
            noise = rng.uniform(-0.045, 0.045)
            pref = 0.0
            comps = tallies[pid]["component_scores"] or derive_pseudo_components(row, award_id, float(score))
            if arch == "production":
                pref = _safe_float(
                    comps.get("pts_pg", comps.get("production_component")),
                    float(score),
                )
            elif arch == "two_way":
                pref = _safe_float(
                    comps.get("onice", comps.get("two_way_component", comps.get("defensive_value"))),
                    float(score),
                )
            elif arch == "team_success":
                pref = _safe_float(
                    comps.get("team", comps.get("team_context_component")),
                    float(score),
                )
            elif arch == "analytics":
                pref = _safe_float(
                    comps.get("value", comps.get("individual_value_component", comps.get("goals_saved_above_expected"))),
                    float(score),
                )
            elif arch == "workload":
                pref = _safe_float(
                    comps.get("avail", comps.get("availability_component", comps.get("workload"))),
                    float(score),
                )
            else:
                pref = float(score)
            adj = float(score) + float(pref) * float(archetype_bias[arch]) + noise * max(0.15, abs(float(score)))
            ranked.append((adj, pid))
        ranked.sort(key=lambda x: x[0], reverse=True)
        for place, (_adj, pid) in enumerate(ranked[: len(curve)]):
            pts = curve[place]
            tallies[pid]["ballot_points"] += pts
            tallies[pid]["placements"][place] += 1
            if place == 0:
                tallies[pid]["first_place_votes"] += 1

    finished = sorted(
        tallies.values(),
        key=lambda c: (c["ballot_points"], c["first_place_votes"], c["canonical_score"]),
        reverse=True,
    )
    for i, c in enumerate(finished):
        c["finish"] = i + 1
    margin = 0.0
    if len(finished) >= 2:
        margin = float(finished[0]["ballot_points"] - finished[1]["ballot_points"])
    return {
        "candidates": finished,
        "voter_count": int(voter_count),
        "margin": margin,
        "seed": _seed_int(season_seed, award_id),
        "points": list(curve),
    }


def _fmt_stat_value(value: Any, fmt: str) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if fmt == "pct1":
        return f"{x * 100:.1f}%" if x <= 1 else f"{x:.1f}%"
    if fmt == "sv3":
        return f"{x:.3f}" if x <= 1 else f"{x:.3f}"
    if fmt == "dec2":
        return f"{x:.2f}"
    if fmt == "signed":
        return f"{x:+.0f}" if abs(x - int(x)) < 1e-6 else f"{x:+.2f}"
    if fmt == "toi":
        m = int(x)
        sec = int(round((x - m) * 60))
        if sec == 60:
            m, sec = m + 1, 0
        return f"{m}:{sec:02d}"
    return str(int(round(x)))


def _stat_line_entry(row: Mapping[str, Any], key: str, label: str, fmt: str = "int") -> Optional[Dict[str, Any]]:
    val = stat(row, key)
    if val is None:
        return None
    return {"key": key, "label": label, "value": val, "fmt": fmt, "display": _fmt_stat_value(val, fmt)}


def _build_award_evidence(
    defn: Mapping[str, Any],
    full_results: List[Dict[str, Any]],
    finalists: List[Dict[str, Any]],
    *,
    voting: Optional[Dict[str, Any]] = None,
    quality: str = "full",
) -> Dict[str, Any]:
    aid = str(defn.get("award_id") or "")
    if not full_results:
        return {}
    winner_row = full_results[0]
    runner = full_results[1] if len(full_results) > 1 else None
    comps = dict(winner_row.get("component_scores") or {})
    criteria = [{"key": k, "label": k.replace("_", " ").title(), "weight": round(v, 3)} for k, v in comps.items() if v is not None]
    if criteria:
        tw = sum(c["weight"] for c in criteria) or 1.0
        for c in criteria:
            c["weight"] = round(c["weight"] / tw, 3)

    stat_keys: List[Tuple[str, str, str]] = []
    if aid in {"hart", "art_ross", "ted_lindsay", "lady_byng", "calder"}:
        stat_keys = [("gp", "GP", "int"), ("g", "G", "int"), ("pts", "PTS", "int"), ("plus_minus", "+/-", "signed")]
    elif aid == "rocket":
        stat_keys = [("g", "G", "int"), ("gp", "GP", "int"), ("pts", "PTS", "int")]
    elif aid == "norris":
        stat_keys = [("pts", "PTS", "int"), ("toi_pg", "TOI/GP", "dec2"), ("blocks", "BLK", "int"), ("gp", "GP", "int")]
    elif aid == "selke":
        stat_keys = [("pts", "PTS", "int"), ("takeaways", "TK", "int"), ("fo_pct", "FO%", "pct1"), ("plus_minus", "+/-", "signed")]
    elif aid == "vezina":
        stat_keys = [("gs", "GS", "int"), ("sv_pct", "SV%", "sv3"), ("ga", "GA", "int"), ("so", "SO", "int")]

    winner_stats: List[Dict[str, Any]] = []
    for key, label, fmt in stat_keys:
        ent = _stat_line_entry(winner_row, key, label, fmt)
        if ent:
            winner_stats.append(ent)

    why: List[str] = []
    if voting and winner_row.get("ballot_points") is not None:
        margin = float(voting.get("margin") or 0.0)
        fpv = int(winner_row.get("first_place_votes") or 0)
        vc = int(voting.get("voter_count") or VOTER_COUNT)
        ru_name = str(runner.get("name") or "the field") if runner else "the field"
        why.append(
            f"{fpv} of {vc} first-place votes · {float(winner_row.get('ballot_points') or 0):.0f} points · "
            f"+{margin:.0f} over {ru_name}"
        )
    if not why and winner_stats:
        lead = winner_stats[0]
        why.append(f"{lead['label']}: {lead.get('display') or _fmt_stat_value(lead['value'], lead['fmt'])}")
    if not why:
        why.append(str(defn.get("name") or "Award") + " on season results.")

    fin_evidence = []
    for cand in finalists[:3]:
        line = []
        for key, label, fmt in stat_keys[:3]:
            ent = _stat_line_entry(cand, key, label, fmt)
            if ent:
                line.append(ent)
        fin_evidence.append(
            {
                "entity_id": cand.get("entity_id") or cand.get("player_id"),
                "name": cand.get("name"),
                "team_id": cand.get("team_id"),
                "team_name": cand.get("team_name"),
                "position": cand.get("position"),
                "score": cand.get("canonical_score"),
                "stat_line": line,
                "ballot": {
                    "points": cand.get("ballot_points"),
                    "first_place_votes": cand.get("first_place_votes"),
                }
                if cand.get("ballot_points") is not None
                else None,
            }
        )

    method = "ballot" if defn.get("category") == "ballot" else ("stat_race" if defn.get("category") == "stat_race" else "selection")
    return {
        "method": method,
        "pool": {"size": len(full_results), "noun": "eligible players"},
        "criteria": criteria,
        "excluded_criteria": [],
        "winner": {
            "entity_id": winner_row.get("entity_id") or winner_row.get("player_id"),
            "stat_line": winner_stats,
            "components": [
                {"key": k, "label": k.replace("_", " ").title(), "value": v, "fmt": "dec2", "pct": v, "weight": v}
                for k, v in comps.items()
            ],
            "ballot": {
                "points": winner_row.get("ballot_points"),
                "first_place_votes": winner_row.get("first_place_votes"),
                "voter_count": (voting or {}).get("voter_count"),
                "margin": (voting or {}).get("margin"),
                "runner_up_name": runner.get("name") if runner else None,
            }
            if winner_row.get("ballot_points") is not None
            else None,
        },
        "finalists": fin_evidence,
        "why": why[:3],
        "close_vote": bool(voting and float(voting.get("margin") or 0) < 12.0),
        "closest_of_night": False,
        "data_quality": {"dropped": [], "imputed_share": {}, "gate_relaxed": False} if quality == "full" else {"dropped": [], "imputed_share": {}, "gate_relaxed": quality != "full"},
    }


def _award_stat_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact season line carried on every award candidate so the ceremony can
    explain *why* (stat lines, ranks, "why he won") without the raw stat rows."""
    gp = _gp(row)
    out: Dict[str, Any] = {"gp": gp}
    if row.get("age") is not None:
        out["age"] = _safe_int(row.get("age"), 0) or None
    if _is_goalie(row):
        sa = _safe_int(row.get("shots_against"), _safe_int(row.get("goalie_shots_against"), 0))
        ga = _safe_int(row.get("ga"), _safe_int(row.get("goalie_ga"), _safe_int(row.get("goals_against"), 0)))
        toi_min = _safe_float(row.get("toi_sec"), 0.0) / 60.0 or _safe_float(row.get("toi_minutes"), 0.0)
        out.update(
            {
                "w": _safe_int(row.get("w", row.get("wins")), 0),
                "l": _safe_int(row.get("l", row.get("losses")), 0),
                "otl": _safe_int(row.get("otl"), 0),
                "so": _safe_int(row.get("so", row.get("shutouts")), 0),
                "shots_against": sa,
                "goals_against": ga,
                "saves": _safe_int(row.get("saves"), max(0, sa - ga)),
            }
        )
        if sa > 0 or row.get("sv_pct") is not None:
            out["sv_pct"] = round(goalie_sv_pct(row), 4)
        if toi_min > 0:
            out["gaa"] = round(ga * 60.0 / toi_min, 2)
        return out
    out.update(
        {
            "g": _goals(row),
            "a": _safe_int(row.get("a", row.get("assists")), 0),
            "pts": _pts(row),
            "plus_minus": _safe_int(row.get("plus_minus", row.get("pm")), 0),
            "pim": _safe_int(row.get("pim"), 0),
        }
    )
    for key, alias in (("sog", ("sog", "shots")), ("blk", ("blk", "blocked_shots", "blocks")), ("hit", ("hit", "hits")),
                       ("takeaways", ("takeaways", "tk")), ("ppg", ("ppg",)), ("ppa", ("ppa",)), ("shg", ("shg",))):
        for k in alias:
            if row.get(k) is not None:
                out[key] = _safe_int(row.get(k), 0)
                break
    toi = _toi_pg_minutes(row)
    if toi > 0:
        out["toi_pg"] = round(toi, 2)
    pk = _safe_float(row.get("pk_toi_sec"), 0.0)
    if pk > 0 and gp > 0:
        out["pk_toi_pg"] = round(pk / 60.0 / gp, 2)
    return out


def _candidate_from_tally(
    tally: Mapping[str, Any],
    team_map: Mapping[str, Any],
    *,
    display_metric: str,
    display_value: Any = None,
) -> Dict[str, Any]:
    row = dict(tally.get("row") or {})
    tid = _tid(row)
    finish = _safe_int(tally.get("finish"), 0)
    return {
        "entity_id": _pid(row),
        "player_id": _pid(row),
        "name": str(row.get("name") or ""),
        "team_id": tid,
        "team_name": _team_name_from_id(dict(team_map), tid, str(row.get("team_name") or "")),
        "position": _pos(row),
        "finish": finish,
        "rank": finish,
        "canonical_score": float(tally.get("canonical_score") or 0.0),
        "ballot_points": float(tally.get("ballot_points") or 0.0) if tally.get("ballot_points") is not None else None,
        "first_place_votes": int(tally.get("first_place_votes") or 0) if tally.get("first_place_votes") is not None else None,
        "votes": int(round(float(tally.get("ballot_points") or tally.get("display_value") or 0.0))),
        "display_value": display_value if display_value is not None else tally.get("display_value"),
        "display_metric": display_metric,
        "component_scores": dict(tally.get("component_scores") or {}),
        "eligibility": dict(row.get("_eligibility") or {}),
        "is_winner": finish == 1,
        "points": _pts(row),
        "goals": _goals(row),
        "assists": _safe_int(row.get("a")),
        "gp": _gp(row),
        "placements": list(tally.get("placements") or []) or None,
        **{k: v for k, v in _award_stat_fields(row).items() if k not in ("gp",)},
    }


def _rationale_from_components(name: str, comps: Mapping[str, Any], quality: str, fallback_reason: Optional[str]) -> str:
    if quality == "unavailable":
        return fallback_reason or "Required season data was unavailable."
    if not comps:
        return f"{name} earned the award on the final scoreboard."
    ranked = sorted(((k, _safe_float(v)) for k, v in comps.items()), key=lambda kv: kv[1], reverse=True)
    top = [k.replace("_", " ").replace(" component", "") for k, _ in ranked[:2]]
    if len(top) == 1:
        return f"Driven by {top[0]}."
    return f"Led by {top[0]} and {top[1]}."


# ---------------------------------------------------------------------------
# Award builders
# ---------------------------------------------------------------------------

def _unavailable_award(defn: Mapping[str, Any], *, reason: str, season: Any = None) -> Award:
    aid = str(defn["award_id"])
    return Award(
        name=str(defn["name"]),
        award_id=aid,
        official=bool(defn.get("official", True)),
        status="unavailable",
        category=str(defn.get("category") or ""),
        recipient_type=str(defn.get("recipient_type") or "player"),
        display_metric=str(defn.get("display_metric") or ""),
        calculation_quality="unavailable",
        fallback_reason=reason,
        unavailable_reason=reason,
        public_rationale=reason,
        rationale=reason,
        eligibility_summary=reason,
        season=season,
        finalists=[],
        candidates=[],
        winners=[],
        full_results=[],
        result={
            "award_id": aid,
            "name": defn["name"],
            "official": True,
            "status": "unavailable",
            "category": defn.get("category"),
            "recipient_type": defn.get("recipient_type"),
            "winner": None,
            "winners": [],
            "shared": False,
            "finalists": [],
            "full_results": [],
            "display_metric": defn.get("display_metric"),
            "calculation_quality": "unavailable",
            "fallback_reason": reason,
            "public_rationale": reason,
            "unavailable_reason": reason,
            "eligibility_summary": reason,
            "stat_scope": "regular_season",
            "season": season,
        },
    )


def _finalize_player_award(
    defn: Mapping[str, Any],
    full_results: List[Dict[str, Any]],
    *,
    season: Any,
    quality: str,
    fallback_reason: Optional[str],
    eligibility_summary: str,
    stat_scope: str,
    voting: Optional[Dict[str, Any]] = None,
    shared_override: Optional[bool] = None,
) -> Award:
    if not full_results:
        return _unavailable_award(defn, reason="No eligible candidates.", season=season)

    winners = [full_results[0]]
    shared = False
    if shared_override is True:
        # All-Star teams pass a full positional roster. Keep every selection.
        winners = list(full_results)
        shared = len(winners) > 1
    elif defn.get("supports_shared_winners") and len(full_results) > 1:
        a0 = full_results[0]
        a1 = full_results[1]
        if shared_override is True or (
            a0.get("display_value") is not None
            and a0.get("display_value") == a1.get("display_value")
            and a0.get("ballot_points") in (None, a1.get("ballot_points"))
        ):
            # collect equals
            key_fields = ("display_value", "ballot_points", "canonical_score")
            winners = [full_results[0]]
            for cand in full_results[1:]:
                if all(cand.get(k) == full_results[0].get(k) for k in key_fields if full_results[0].get(k) is not None):
                    winners.append(cand)
                else:
                    break
            shared = len(winners) > 1

    for i, cand in enumerate(full_results):
        cand["finish"] = i + 1
        cand["rank"] = i + 1
        cand["is_winner"] = any(_pid(cand) == _pid(w) or cand.get("entity_id") == w.get("entity_id") for w in winners)

    top = winners[0]
    comps = dict(top.get("component_scores") or {})
    rationale = _rationale_from_components(str(defn["name"]), comps, quality, fallback_reason)
    finalists = full_results[: max(1, int(defn.get("finalist_count") or 3))]
    evidence = _build_award_evidence(defn, full_results, finalists, voting=voting, quality=quality)
    if evidence.get("why"):
        rationale = str(evidence["why"][0])
    result = {
        "award_id": defn["award_id"],
        "name": defn["name"],
        "official": True,
        "status": "complete",
        "category": defn.get("category"),
        "recipient_type": defn.get("recipient_type"),
        "winner": top,
        "winners": winners,
        "shared": shared,
        "finalists": finalists,
        "full_results": full_results,
        "display_metric": defn.get("display_metric"),
        "calculation_quality": quality,
        "fallback_reason": fallback_reason,
        "public_rationale": rationale,
        "eligibility_summary": eligibility_summary,
        "stat_scope": stat_scope,
        "season": season,
        "voting": voting,
        "evidence": evidence,
    }
    return Award(
        name=str(defn["name"]),
        award_id=str(defn["award_id"]),
        official=True,
        status="complete",
        category=str(defn.get("category") or ""),
        recipient_type=str(defn.get("recipient_type") or "player"),
        winner_team_id=str(top.get("team_id") or ""),
        winner_name=str(top.get("name") or ""),
        winner_player_id=str(top.get("player_id") or top.get("entity_id") or ""),
        winner_team_name=str(top.get("team_name") or ""),
        finalists=finalists,
        candidates=full_results[:5],
        winners=winners,
        shared=shared,
        full_results=full_results,
        display_metric=str(defn.get("display_metric") or ""),
        calculation_quality=quality,
        fallback_reason=fallback_reason,
        public_rationale=rationale,
        rationale=rationale,
        eligibility_summary=eligibility_summary,
        stat_scope=stat_scope,
        season=season,
        voting=voting,
        winner_stats={
            "points": top.get("points"),
            "goals": top.get("goals"),
            "gp": top.get("gp"),
            "ballot_points": top.get("ballot_points"),
            "first_place_votes": top.get("first_place_votes"),
        },
        result=result,
    )


def _score_pool_with_quality(
    pool: List[Dict[str, Any]],
    score_fn: Callable[[Dict[str, Any]], float],
    required_fields: Sequence[str],
    fallback_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
) -> Tuple[List[Tuple[float, Dict[str, Any]]], str, Optional[str]]:
    if not pool:
        return [], "unavailable", "No eligible candidates."
    missing_counts = 0
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in pool:
        row = dict(row)
        missing = validate_required_award_fields(row, required_fields)
        optional_analytics = [f for f in ("war", "impact_score", "xgf_pct", "gsax", "defense_score") if f in required_fields or True]
        # Only fail hard if NONE of analytics-ish fields exist when required includes them
        hard_missing = [f for f in missing if f in {"gp"}]
        if hard_missing:
            continue
        analytics_present = any(row.get(f) is not None for f in ("war", "impact_score", "gsax", "defense_score", "analytics_rating"))
        quality_row = "full" if analytics_present else "documented_fallback"
        if quality_row == "documented_fallback":
            missing_counts += 1
            s = float(fallback_fn(row)) if fallback_fn else float(score_fn(row))
        else:
            s = float(score_fn(row))
        row["_calc_quality"] = quality_row
        scored.append((s, row))
    if not scored:
        return [], "unavailable", "Candidates missing required participation fields."
    quality = "documented_fallback" if missing_counts == len(scored) else ("documented_fallback" if missing_counts else "full")
    reason = "Missing advanced analytics on one or more candidates; used documented participation fallback." if quality != "full" else None
    scored.sort(key=lambda p: p[0], reverse=True)
    return scored, quality, reason


def _run_ballot_award(
    defn: Mapping[str, Any],
    pool: List[Dict[str, Any]],
    score_fn: Callable[[Dict[str, Any]], float],
    *,
    team_map: Dict[str, Any],
    season_seed: Any,
    season: Any,
    eligibility_summary: str,
    required_fields: Sequence[str],
    fallback_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    stat_scope: str = "regular_season",
    team_ctx: Optional[Mapping[str, Mapping[str, Any]]] = None,
    season_length: int = 82,
) -> Award:
    pool = _isolated_scoring_pool(pool)
    aid = str(defn.get("award_id") or "")
    scored, quality, reason = _score_pool_with_quality(pool, score_fn, required_fields, None)
    if quality == "unavailable":
        return _unavailable_award(defn, reason=reason or "Unavailable", season=season)
    # The formula leader keeps the trophy. Voter noise does not pass him.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    full: List[Dict[str, Any]] = []
    for score, row in scored:
        full.append(
            {
                "entity_id": _pid(row),
                "player_id": _pid(row),
                "name": str(row.get("name") or ""),
                "team_id": _tid(row),
                "team_name": _team_name_from_id(team_map, _tid(row), str(row.get("team_name") or "")),
                "position": _pos(row),
                "canonical_score": float(score),
                "ballot_points": float(score),
                "first_place_votes": None,
                "votes": int(round(float(score))),
                "display_value": round(float(score), 2),
                "display_metric": str(defn.get("display_metric") or ""),
                "component_scores": {},
                "eligibility": dict(row.get("_eligibility") or {}),
                "points": _pts(row),
                "goals": _goals(row),
                "assists": _safe_int(row.get("a")),
                "gp": _gp(row),
            }
        )
    return _finalize_player_award(
        defn,
        full,
        season=season,
        quality=quality,
        fallback_reason=reason,
        eligibility_summary=eligibility_summary,
        stat_scope=stat_scope,
        voting={
            "voter_count": 0,
            "margin": float(scored[0][0] - scored[1][0]) if len(scored) > 1 else 0.0,
            "seed": _seed_int(season_seed, aid),
            "ballot_points_curve": [],
            "voting_body": "Season formula",
            "leader_locked": True,
        },
    )


def _run_stat_race(
    defn: Mapping[str, Any],
    pool: List[Dict[str, Any]],
    primary: Callable[[Dict[str, Any]], float],
    tiebreak_keys: Sequence[Callable[[Dict[str, Any]], float]],
    *,
    team_map: Dict[str, Any],
    season: Any,
    eligibility_summary: str,
) -> Award:
    pool = _isolated_scoring_pool(pool)
    if not pool:
        return _unavailable_award(defn, reason="No eligible candidates.", season=season)

    def sort_key(r: Dict[str, Any]) -> Tuple:
        return tuple(fn(r) for fn in (primary, *tiebreak_keys))

    ordered = sorted(pool, key=sort_key, reverse=True)
    full: List[Dict[str, Any]] = []
    for i, row in enumerate(ordered):
        val = primary(row)
        full.append(
            {
                "entity_id": _pid(row),
                "player_id": _pid(row),
                "name": str(row.get("name") or ""),
                "team_id": _tid(row),
                "team_name": _team_name_from_id(team_map, _tid(row), str(row.get("team_name") or "")),
                "position": _pos(row),
                "finish": i + 1,
                "rank": i + 1,
                "canonical_score": float(val),
                "ballot_points": None,
                "first_place_votes": None,
                "votes": int(round(float(val))),
                "display_value": val,
                "display_metric": defn["display_metric"],
                "component_scores": {"race_value": float(val)},
                "eligibility": dict(row.get("_eligibility") or {}),
                "is_winner": i == 0,
                "points": _pts(row),
                "goals": _goals(row),
                "assists": _safe_int(row.get("a")),
                "gp": _gp(row),
                **{k: v for k, v in _award_stat_fields(row).items() if k not in ("gp",)},
            }
        )
    # Shared winners if exact primary matches after tiebreak equivalence
    shared = False
    if defn.get("supports_shared_winners") and len(full) > 1:
        if all(sort_key(ordered[0])[j] == sort_key(ordered[1])[j] for j in range(len(tiebreak_keys) + 1)):
            shared = True
    return _finalize_player_award(
        defn,
        full,
        season=season,
        quality="full",
        fallback_reason=None,
        eligibility_summary=eligibility_summary,
        stat_scope="regular_season",
        shared_override=shared,
    )


# ---------------------------------------------------------------------------
# Eligibility pools
# ---------------------------------------------------------------------------

def eligible_art_ross(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    return [r for r in rows if not _is_goalie(r) and _gp(r) >= 1]


def eligible_rocket(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    return [r for r in rows if not _is_goalie(r) and _gp(r) >= 1]


def eligible_hart(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    need = _season_games_threshold(season_length, 0.45, minimum=30)
    return [r for r in rows if not _is_goalie(r) and _gp(r) >= need]


def eligible_norris(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    need = _season_games_threshold(season_length, 0.45, minimum=30)
    out = []
    for r in rows:
        if not _is_defense(r) or _gp(r) < need:
            continue
        toi = _toi_pg_minutes(r)
        if toi > 0 and toi < 12.0:
            continue
        out.append(r)
    return out


def eligible_selke(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    need = _season_games_threshold(season_length, 0.45, minimum=30)
    # Penalty-kill time is not a requirement.
    return [r for r in rows if _is_forward(r) and _gp(r) >= need]


def eligible_lady_byng(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    need = _season_games_threshold(season_length, 0.40, minimum=28)
    return [r for r in rows if not _is_goalie(r) and _gp(r) >= need and _pts(r) > 0]


def eligible_vezina(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if not _is_goalie(r):
            continue
        ok, details = goalie_workload_ok(r, season_length=season_length)
        r = dict(r)
        r["_eligibility"] = details
        if ok:
            out.append(r)
    return out


def eligible_calder(
    rows: Sequence[Dict[str, Any]],
    *,
    teams: Optional[Sequence[Any]],
    history_by_player: Optional[Mapping[str, Any]],
    season_length: int,
) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        elig = calder_eligibility(
            r,
            teams=teams,
            history=(history_by_player or {}).get(_pid(r)) if history_by_player else history_by_player,
            season_length=season_length,
        )
        rr = dict(r)
        rr["_eligibility"] = elig
        rr["eligibility_confidence"] = elig.get("confidence")
        det_age = (elig.get("details") or {}).get("age")
        if det_age is not None:
            rr["age"] = det_age  # season-start (Sept. 15) age used by the rule
        if elig.get("eligible"):
            out.append(rr)
    return out


# ---------------------------------------------------------------------------
# Conference champions / Jennings / team awards
# ---------------------------------------------------------------------------

def extract_conference_champions(
    playoff_result: Optional[PlayoffResult],
    team_map: Dict[str, Any],
    *,
    season: Any,
) -> List[Dict[str, Any]]:
    if playoff_result is None:
        return []
    series = list(getattr(playoff_result, "series_list", None) or [])
    if not series:
        return []
    # Highest round_index with a non-null conference before the final (conference None).
    conf_series = [s for s in series if getattr(s, "conference", None)]
    if not conf_series:
        # Fallback: treat Cup finalists as East/West unknown mirrors only if conferences missing.
        return []
    max_round = max(int(getattr(s, "round_index", 0) or 0) for s in conf_series)
    finals = [s for s in conf_series if int(getattr(s, "round_index", 0) or 0) == max_round]
    champs: List[Dict[str, Any]] = []
    for s in finals:
        wid = str(s.winner_id())
        lid = str(s.loser_id())
        conf = str(getattr(s, "conference", "") or "")
        champs.append(
            {
                "conference": conf,
                "team_id": wid,
                "team_name": _team_name_from_id(team_map, wid),
                "final_opponent_id": lid,
                "final_opponent_name": _team_name_from_id(team_map, lid),
                "series_result": s.series_score() if hasattr(s, "series_score") else "",
                "season": season,
                "name": _team_name_from_id(team_map, wid),
                "entity_id": wid,
                "is_winner": True,
            }
        )
    return champs


def compute_jennings(
    standings: StandingsTable,
    goalie_rows: Sequence[Dict[str, Any]],
    team_map: Dict[str, Any],
    *,
    season: Any,
    season_length: int,
) -> Award:
    defn = AWARD_REGISTRY["jennings"]
    tbl = list(standings.league_table() or [])
    if not tbl:
        return _unavailable_award(defn, reason="Standings unavailable for Jennings.", season=season)
    best = sorted(tbl, key=lambda r: (int(getattr(r, "ga", 0) or 0), -int(getattr(r, "points", 0) or 0)))
    team_rec = best[0]
    team_ga = int(getattr(team_rec, "ga", 0) or 0)
    tid = str(team_rec.team_id)
    # Qualifying goalies: >= 25% of team games started/played
    team_gp = max(1, _safe_int(getattr(team_rec, "gp", None), season_length))
    min_apps = max(1, int(math.ceil(team_gp * 0.25)))
    recipients = []
    for g in goalie_rows:
        if str(_tid(g)) != tid:
            continue
        ok, details = goalie_workload_ok(g, season_length=season_length)
        apps = max(_safe_int(g.get("games_started"), 0), _gp(g))
        if apps >= min_apps or ok:
            recipients.append(
                {
                    "entity_id": _pid(g),
                    "player_id": _pid(g),
                    "name": str(g.get("name") or ""),
                    "team_id": tid,
                    "team_name": _team_name_from_id(team_map, tid),
                    "position": "G",
                    "finish": 1,
                    "rank": 1,
                    "canonical_score": float(team_ga),
                    "display_value": team_ga,
                    "display_metric": "Team GA",
                    "component_scores": {"team_goals_against": float(team_ga)},
                    "eligibility": details,
                    "is_winner": True,
                    "qualification_details": {"min_apps": min_apps, "apps": apps},
                    "gp": _gp(g),
                    "points": 0,
                    "goals": 0,
                    "assists": 0,
                    "votes": team_ga,
                }
            )
    award_status = "complete"
    if not recipients:
        # Explicit fallback: team credited with fewest GA; no individual goalie share.
        award_status = "team_only"
        recipients = [
            {
                "entity_id": tid,
                "name": _team_name_from_id(team_map, tid, team_rec.name),
                "team_id": tid,
                "team_name": _team_name_from_id(team_map, tid, team_rec.name),
                "position": None,
                "finish": 1,
                "canonical_score": float(team_ga),
                "display_value": team_ga,
                "display_metric": "Team GA",
                "is_winner": True,
                "votes": team_ga,
                "award_status": "team_only",
            }
        ]
    award = _finalize_player_award(
        defn,
        recipients,
        season=season,
        quality="full",
        fallback_reason=None,
        eligibility_summary=f"Fewest team goals against ({team_ga}); goalie threshold {min_apps} apps.",
        stat_scope="regular_season",
        shared_override=len(recipients) > 1,
    )
    award.winner_team_id = tid
    award.winner_team_name = _team_name_from_id(team_map, tid, team_rec.name)
    award.winner_name = ", ".join(r["name"] for r in recipients)
    award.winner_stats = {"goals_against": team_ga, "team_goals_against": team_ga}
    if award.result is not None:
        award.result["winner_team"] = {
            "team_id": tid,
            "team_name": award.winner_team_name,
            "team_goals_against": team_ga,
        }
        award.result["recipients"] = recipients
        award.result["team_goals_against"] = team_ga
        award.result["qualification_details"] = {"min_apps": min_apps}
        award.result["award_status"] = award_status
    return award


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Voting explanations ("why he won" + vote table) for the ceremony
# ---------------------------------------------------------------------------

_EXPLAIN_POOL_NOUN = {
    "hart": "skaters",
    "ted_lindsay": "skaters",
    "art_ross": "the league",
    "rocket": "the league",
    "norris": "defencemen",
    "selke": "forwards",
    "calder": "rookies",
    "vezina": "starting goalies",
    "lady_byng": "skaters",
    "conn_smythe": "playoff skaters",
}

_SKATER_LINE = [("gp", "GP", "int"), ("g", "G", "int"), ("a", "A", "int"), ("pts", "PTS", "int"), ("plus_minus", "+/-", "signed"), ("toi_pg", "TOI/GP", "toi")]
_GOALIE_LINE = [("gp", "GP", "int"), ("w", "W", "int"), ("sv_pct", "SV%", "sv3"), ("gaa", "GAA", "dec2"), ("so", "SO", "int")]
_EXPLAIN_LINES: Dict[str, List[Tuple[str, str, str]]] = {
    "hart": _SKATER_LINE,
    "ted_lindsay": _SKATER_LINE,
    "art_ross": [("pts", "PTS", "int"), ("g", "G", "int"), ("a", "A", "int"), ("gp", "GP", "int"), ("plus_minus", "+/-", "signed")],
    "rocket": [("g", "G", "int"), ("sog", "SOG", "int"), ("pts", "PTS", "int"), ("gp", "GP", "int")],
    "norris": [("pts", "PTS", "int"), ("g", "G", "int"), ("toi_pg", "TOI/GP", "toi"), ("plus_minus", "+/-", "signed"), ("blk", "BLK", "int"), ("gp", "GP", "int")],
    "selke": [("plus_minus", "+/-", "signed"), ("pts", "PTS", "int"), ("toi_pg", "TOI/GP", "toi"), ("pk_toi_pg", "PK TOI/GP", "toi"), ("takeaways", "TK", "int"), ("blk", "BLK", "int")],
    "calder": _SKATER_LINE,
    "vezina": _GOALIE_LINE,
    "lady_byng": [("pts", "PTS", "int"), ("pim", "PIM", "int"), ("gp", "GP", "int"), ("g", "G", "int"), ("a", "A", "int")],
    "conn_smythe": [("gp", "GP", "int"), ("g", "G", "int"), ("a", "A", "int"), ("pts", "PTS", "int"), ("plus_minus", "+/-", "signed")],
}
_LOWER_IS_BETTER = frozenset({"gaa", "pim"})
_RANKED_KEYS = frozenset({"pts", "g", "a", "plus_minus", "toi_pg", "sv_pct", "gaa", "w", "so", "pim", "sog", "blk", "takeaways", "pk_toi_pg"})


def _ordinal(n: int) -> str:
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _xv(row: Mapping[str, Any], key: str) -> Optional[float]:
    val = row.get(key)
    if val is None:
        return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _pool_rank(pool: Sequence[Mapping[str, Any]], row: Mapping[str, Any], key: str) -> Optional[Tuple[int, int]]:
    mine = _xv(row, key)
    if mine is None:
        return None
    vals = [v for v in (_xv(r, key) for r in pool) if v is not None]
    if len(vals) < 2:
        return None
    if key in _LOWER_IS_BETTER:
        better = sum(1 for v in vals if v < mine - 1e-9)
    else:
        better = sum(1 for v in vals if v > mine + 1e-9)
    return better + 1, len(vals)


def _explain_line_specs(aid: str, row: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    if _is_goalie(row):
        return _GOALIE_LINE
    return _EXPLAIN_LINES.get(aid, _SKATER_LINE)


def _explain_stat_line(aid: str, row: Mapping[str, Any], pool: Sequence[Mapping[str, Any]], *, limit: int = 6, ranks: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    same_kind = [r for r in pool if _is_goalie(r) == _is_goalie(row)]
    for key, label, fmt in _explain_line_specs(aid, row):
        val = _xv(row, key)
        if val is None:
            continue
        ent: Dict[str, Any] = {"key": key, "label": label, "value": val, "fmt": fmt, "display": _fmt_stat_value(val, fmt)}
        if ranks and key in _RANKED_KEYS:
            rk = _pool_rank(same_kind, row, key)
            if rk:
                ent["rank"], ent["of"] = rk
        out.append(ent)
        if len(out) >= limit:
            break
    return out


def _short_line(aid: str, row: Mapping[str, Any]) -> str:
    if _is_goalie(row):
        bits = []
        if row.get("sv_pct") is not None:
            bits.append(f"{_fmt_stat_value(row.get('sv_pct'), 'sv3').lstrip('0')} SV%")
        if row.get("gaa") is not None:
            bits.append(f"{float(row['gaa']):.2f} GAA")
        bits.append(f"{_safe_int(row.get('w'))} W")
        return ", ".join(bits)
    if aid == "rocket":
        return f"{_safe_int(row.get('g'))} G in {_gp(row)} GP"
    if aid == "lady_byng":
        return f"{_pts(row)} PTS, {_safe_int(row.get('pim'))} PIM"
    return f"{_safe_int(row.get('g'))}-{_safe_int(row.get('a'))}-{_pts(row)} in {_gp(row)} GP"


def _why_lines(
    aid: str,
    winner: Mapping[str, Any],
    full: Sequence[Mapping[str, Any]],
    *,
    team_ctx: Mapping[str, Mapping[str, Any]],
    voting: Optional[Mapping[str, Any]],
) -> List[str]:
    noun = _EXPLAIN_POOL_NOUN.get(aid, "the field")
    runner = full[1] if len(full) > 1 else None
    rname = str(runner.get("name") or "") if runner else ""
    gp = _gp(winner)
    lines: List[str] = []
    playoff = aid == "conn_smythe"
    games = "playoff games" if playoff else "games"

    if _is_goalie(winner):
        sv = winner.get("sv_pct")
        gaa = winner.get("gaa")
        goalies = [r for r in full if _is_goalie(r)]
        if sv is not None:
            rk = _pool_rank(goalies, winner, "sv_pct")
            head = f"Posted a {_fmt_stat_value(sv, 'sv3').lstrip('0')} save percentage"
            if gaa is not None:
                head += f" and a {float(gaa):.2f} GAA"
            head += f" over {gp} {games}"
            if rk and len(goalies) > 1:
                head += f" ({_ordinal(rk[0])} in SV% among {noun if not playoff else 'playoff goalies'})"
            lines.append(head + ".")
        rec = f"{_safe_int(winner.get('w'))}-{_safe_int(winner.get('l'))}-{_safe_int(winner.get('otl'))}"
        so = _safe_int(winner.get("so"))
        lines.append(f"Went {rec} with {so} shutout{'s' if so != 1 else ''}, facing {_safe_int(winner.get('shots_against'))} shots.")
    else:
        pts, g, a = _pts(winner), _goals(winner), _safe_int(winner.get("a", winner.get("assists")))
        skaters = [r for r in full if not _is_goalie(r)]
        if aid == "rocket":
            lead = g - _goals(runner) if runner else 0
            line = f"Scored {g} goals in {gp} {games}"
            if runner and lead > 0:
                line += f", {lead} more than {rname}"
            elif runner and lead == 0:
                line += f", tied with {rname} and won the tiebreak"
            lines.append(line + ".")
        elif aid == "art_ross":
            lead = pts - _pts(runner) if runner else 0
            line = f"Won the scoring race with {pts} points ({g} G, {a} A) in {gp} {games}"
            if runner and lead > 0:
                line += f", {lead} clear of {rname}"
            elif runner and lead == 0:
                line += f" — tied with {rname}, won on goals"
            lines.append(line + ".")
        elif aid == "lady_byng":
            pim = _safe_int(winner.get("pim"))
            lines.append(f"Produced {pts} points while taking just {pim} penalty minutes in {gp} {games} ({pim / max(1, gp):.2f} PIM per game).")
        else:
            rk = _pool_rank(skaters, winner, "pts")
            line = f"{pts} points ({g} G, {a} A) in {gp} {games}"
            if rk:
                line += f" — {_ordinal(rk[0])} among {noun}"
                if rk[0] == 1 and runner and not _is_goalie(runner):
                    gap = pts - _pts(runner)
                    if gap > 0:
                        line += f", {gap} ahead of {rname}"
            lines.append(line + ".")
        toi = winner.get("toi_pg")
        pm = winner.get("plus_minus")
        bits: List[str] = []
        if toi:
            rk = _pool_rank(skaters, winner, "toi_pg")
            t = f"averaged {_fmt_stat_value(toi, 'toi')} a night"
            if rk and rk[0] <= 3 and aid in ("norris", "selke", "calder", "hart"):
                t += f" ({_ordinal(rk[0])} among {noun})"
            bits.append(t)
        if aid == "selke" and winner.get("pk_toi_pg"):
            bits.append(f"{_fmt_stat_value(winner['pk_toi_pg'], 'toi')} on the PK")
        if pm is not None:
            bits.append(f"finished {int(pm):+d}")
        if aid in ("selke", "norris"):
            for k, lab in (("takeaways", "takeaways"), ("blk", "blocked shots"), ("hit", "hits")):
                if winner.get(k):
                    bits.append(f"{_safe_int(winner[k])} {lab}")
                    break
        if bits and aid not in ("art_ross", "rocket", "lady_byng"):
            s = ", ".join(bits)
            lines.append(s[0].upper() + s[1:] + ".")

    ctx = dict((team_ctx or {}).get(str(winner.get("team_id") or ""), {}))
    if ctx and not playoff and aid in ("hart", "ted_lindsay", "vezina", "norris", "selke", "calder"):
        team = str(winner.get("team_name") or "his team")
        tl = f"{team} went {ctx.get('record')} ({ctx.get('points')} pts, {_ordinal(int(ctx.get('league_rank') or 0))} overall)"
        gf = _safe_int(ctx.get("goals_for"))
        if aid == "hart" and gf > 0 and not _is_goalie(winner):
            tl += f"; he factored on {_pts(winner) / gf:.0%} of the team's goals"
        lines.append(tl + ".")
    if aid == "calder":
        age = winner.get("age")
        lines.append(
            f"Rookie-eligible{f' at age {age}' if age else ''}: never more than 25 NHL games in a prior season, "
            "never 6+ games in two prior seasons, and under 26 on Sept. 15."
        )

    if voting and winner.get("ballot_points") is not None:
        fpv = _safe_int(winner.get("first_place_votes"))
        vc = _safe_int(voting.get("voter_count"), VOTER_COUNT)
        margin = float(voting.get("margin") or 0.0)
        v = f"Voting: {fpv} of {vc} first-place votes and {float(winner.get('ballot_points') or 0):.0f} points"
        if rname:
            v += f", {margin:.0f} ahead of {rname}"
        lines.append(v + ".")
    return lines


def _attach_award_explanations(awards: Mapping[str, Award], team_ctx: Mapping[str, Mapping[str, Any]]) -> None:
    """Rebuild each player award's evidence with stat-based reasons and a ballot table."""
    for award in awards.values():
        try:
            aid = str(award.award_id or "")
            if aid not in _EXPLAIN_LINES or award.status != "complete" or not award.full_results:
                continue
            full = list(award.full_results)
            winner = full[0]
            voting = award.voting or None
            res = award.result if isinstance(award.result, dict) else {}
            ev = dict(res.get("evidence") or {})
            why = _why_lines(aid, winner, full, team_ctx=team_ctx, voting=voting)
            if not why:
                continue
            ev["why"] = why[:5]
            ev.setdefault("winner", {})
            ev["winner"] = dict(ev.get("winner") or {})
            ev["winner"]["stat_line"] = _explain_stat_line(aid, winner, full)
            fin_ev = []
            by_id = {str(f.get("entity_id")): f for f in list(ev.get("finalists") or []) if isinstance(f, dict)}
            for cand in award.finalists or full[:3]:
                base = dict(by_id.get(str(cand.get("entity_id") or cand.get("player_id")), {}))
                base.update(
                    {
                        "entity_id": cand.get("entity_id") or cand.get("player_id"),
                        "name": cand.get("name"),
                        "team_id": cand.get("team_id"),
                        "team_name": cand.get("team_name"),
                        "position": cand.get("position"),
                        "stat_line": _explain_stat_line(aid, cand, full, limit=3, ranks=False),
                    }
                )
                fin_ev.append(base)
            ev["finalists"] = fin_ev
            table = []
            for cand in full[:5]:
                table.append(
                    {
                        "rank": cand.get("finish") or cand.get("rank"),
                        "entity_id": cand.get("entity_id") or cand.get("player_id"),
                        "name": cand.get("name"),
                        "team_id": cand.get("team_id"),
                        "team_name": cand.get("team_name"),
                        "position": cand.get("position"),
                        "points": cand.get("ballot_points"),
                        "first_place_votes": cand.get("first_place_votes"),
                        "placements": cand.get("placements"),
                        "value": cand.get("display_value") if cand.get("ballot_points") is None else None,
                        "summary": _short_line(aid, cand),
                        "is_winner": bool(cand.get("is_winner")),
                    }
                )
            ev["voting_table"] = table
            if voting and winner.get("ballot_points") is not None:
                cfg = ballot_config_for(aid)
                ev["ballot_format"] = {
                    "voters": _safe_int(voting.get("voter_count"), VOTER_COUNT),
                    "points": list(voting.get("ballot_points_curve") or cfg.get("points") or BALLOT_POINTS),
                    "body": cfg.get("body") or "PHWA",
                }
            res["evidence"] = ev
            award.result = res
            award.rationale = award.public_rationale = why[0]
            res["public_rationale"] = why[0]
            if award.winner_stats is not None:
                award.winner_stats = {**dict(award.winner_stats), **{k: winner.get(k) for k in ("assists", "plus_minus", "toi_pg", "sv_pct", "gaa", "w", "so") if winner.get(k) is not None}}
        except Exception:
            logger.debug("award explanation failed for %s", getattr(award, "award_id", "?"), exc_info=True)


def _race_metric(label: str, value: Any, kind: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    try:
        if kind == "int":
            number = int(round(float(value)))
        elif kind == "signed":
            number = round(float(value), 1)
        elif kind == "pct":
            number = float(value)
        else:
            number = round(float(value), 2)
    except (TypeError, ValueError):
        return None
    return {"label": label, "value": number, "kind": kind}


def _race_card(row: Mapping[str, Any], metrics: Sequence[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    shown = [m for m in metrics if m]
    labels = {str(m.get("label")) for m in shown}
    for label, key, kind in (
        ("xGF%", "xgf_pct", "pct"),
        ("Flurry", "oi_flurry_xgf_pct", "pct"),
        ("Score", "oi_score_xgf_pct", "pct"),
        ("HD/60", "ind_hd_per_60", "signed"),
        ("SH Ax", "sh_above_expected", "pct"),
        ("iXG/60", "ixg_per_60", "signed"),
        ("G/60", "goals_per_60", "signed"),
    ):
        if label in labels or row.get(key) is None:
            continue
        metric = _race_metric(label, row.get(key), kind)
        if metric:
            shown.append(metric)
            labels.add(label)
    card: Dict[str, Any] = {
        "player_id": _pid(row),
        "name": str(row.get("name") or ""),
        "team_id": _tid(row),
        "team_abbr": str(row.get("team_abbr") or row.get("team") or row.get("abbrev") or ""),
        "position": _pos(row),
        "gp": _gp(row),
        "metrics": shown,
    }
    for key in (
        "nhl_headshot_url",
        "headshot_url",
        "nhl_player_id",
        "nhl_id",
        "headshot_id",
        "real_nhl_import",
        "jersey_number",
    ):
        if row.get(key) not in (None, ""):
            card[key] = row.get(key)
    return card


def _top_race(rows: Sequence[Mapping[str, Any]], score_fn: Any, limit: int = 5) -> List[Mapping[str, Any]]:
    ranked = sorted(rows, key=score_fn, reverse=True)
    return ranked[:limit]


def build_live_awards_race(
    player_rows: Sequence[Mapping[str, Any]],
    *,
    teams: Optional[Sequence[Any]] = None,
    standings: Any = None,
    season_length: int = 82,
    limit: int = 5,
    sealed: bool = False,
) -> Dict[str, Any]:
    """
    In-season race using the same ballot and counting formulas as compute_awards.

    Display fields are counted stats already on the ledger (points, goals, WAR,
    GSAx, save percentage, standings). Missing counters stay off the card.
    """
    rows = [dict(r) for r in (player_rows or []) if isinstance(r, Mapping) and _gp(r) >= 1]
    if sealed:
        return {
            "boards": _sealed_award_boards(),
            "source": "awards_backend",
            "sealed": True,
            "sealed_message": (
                "The awards race is sealed for the final 15 days of the season. "
                "Whoever is leading when the season ends keeps the trophy."
            ),
        }
    prepare_award_context(rows)
    skaters = [r for r in rows if not _is_goalie(r)]
    goalies = [r for r in rows if _is_goalie(r)]
    forwards = [r for r in skaters if _is_forward(r)]
    defense = [r for r in skaters if _is_defense(r)]
    team_ctx: Dict[str, Dict[str, Any]] = {}
    if standings is not None:
        try:
            team_ctx = build_team_context(standings)
        except Exception:
            team_ctx = {}

    def _war(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if row.get("war") is None:
            return None
        return _race_metric("WAR", row.get("war"), "signed")

    def _gsax(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if row.get("gsax") is None or row.get("gsax_valid") is False:
            return None
        return _race_metric("GSAx", row.get("gsax"), "signed")

    hart = [
        _race_card(
            r,
            [
                _race_metric("PTS", _pts(r), "int"),
                _race_metric("CF rel", (r.get("_rel_cf") or 0) * 100.0, "signed"),
                _war(r),
                _race_metric("Share", r.get("_team_point_share"), "pct") if r.get("_team_point_share") else None,
            ],
        )
        for r in _top_race(skaters, lambda row: hart_ballot_score(row, team_ctx), limit)
    ]
    vezina = [
        _race_card(
            r,
            [
                _gsax(r),
                _race_metric("SV%", goalie_sv_pct(r), "pct") if _safe_int(r.get("shots_against"), _safe_int(r.get("sa"), 0)) > 0 else None,
                _race_metric("W", _safe_int(r.get("w"), _safe_int(r.get("wins"), 0)), "int"),
                _race_metric("L", _safe_int(r.get("l"), _safe_int(r.get("losses"), 0)), "int"),
            ],
        )
        for r in _top_race(
            [g for g in goalies if _safe_int(g.get("shots_against"), _safe_int(g.get("sa"), 0)) > 0 or _gp(g) > 0],
            lambda row: vezina_ballot_score(row, team_ctx),
            limit,
        )
    ]
    norris = [
        _race_card(
            r,
            [
                _race_metric("PTS", _pts(r), "int"),
                _race_metric("xGF rel", (r.get("_rel_xgf") or 0) * 100.0, "signed"),
                _war(r),
            ],
        )
        for r in _top_race(defense, lambda row: norris_ballot_score(row, team_ctx), limit)
    ]
    try:
        for _race_row in rows:
            if isinstance(_race_row, dict):
                _race_row["live_race"] = True
        calder_pool = eligible_calder(rows, teams=teams, history_by_player=None, season_length=int(season_length or 82))
    except Exception:
        calder_pool = []
    calder = [
        _race_card(
            r,
            [
                _gsax(r) if _is_goalie(r) else _war(r),
                _race_metric("SV%", goalie_sv_pct(r), "pct") if _is_goalie(r) else _race_metric("PTS", _pts(r), "int"),
                _race_metric("G", _goals(r), "int") if not _is_goalie(r) else None,
            ],
        )
        for r in _top_race(calder_pool, lambda row: calder_position_score(row, team_ctx), limit)
    ]
    selke = [
        _race_card(
            r,
            [
                _race_metric("xGA/60", r.get("_xga60"), "signed") if r.get("_xga60") is not None else None,
                _race_metric("xGF rel", (r.get("_rel_xgf") or 0) * 100.0, "signed"),
                _race_metric("PTS", _pts(r), "int"),
            ],
        )
        for r in _top_race(forwards, selke_ballot_score, limit)
    ]
    art_ross = [
        _race_card(
            r,
            [
                _race_metric("PTS", _pts(r), "int"),
                _race_metric("G", _goals(r), "int"),
                _race_metric("A", _safe_int(r.get("a"), 0), "int"),
            ],
        )
        for r in _top_race(
            skaters,
            lambda row: (float(_pts(row)), float(_goals(row)), -float(_gp(row))),
            limit,
        )
    ]
    rocket = [
        _race_card(
            r,
            [
                _race_metric("G", _goals(r), "int"),
                _race_metric("PTS", _pts(r), "int"),
                _race_metric("SOG", _safe_int(r.get("sog"), _safe_int(r.get("shots"), 0)), "int"),
            ],
        )
        for r in _top_race(
            skaters,
            lambda row: (float(_goals(row)), float(_pts(row)), -float(_gp(row))),
            limit,
        )
    ]

    lindsay = [
        _race_card(
            r,
            [
                _race_metric("PTS", _pts(r), "int"),
                _race_metric("xGF%", r.get("_xgf_pct"), "pct") if r.get("_xgf_pct") is not None else None,
                _race_metric("GF%", r.get("_gf_pct"), "pct") if r.get("_gf_pct") is not None else None,
                _war(r),
            ],
        )
        for r in _top_race(skaters, lambda row: ted_lindsay_score(row, team_ctx), limit)
    ]
    lady = [
        _race_card(
            r,
            [
                _race_metric("PTS", _pts(r), "int"),
                _race_metric("PIM", _safe_int(r.get("pim"), 0), "int"),
            ],
        )
        for r in _top_race(skaters, lady_byng_score, limit)
    ]
    boards = [
        {"id": "hart", "name": "Hart", "blurb": "Scoring, importance, team success, relative CF, WAR", "leaders": hart},
        {"id": "ted_lindsay", "name": "Ted Lindsay", "blurb": "Hart formula with on-ice xGF% and GF%", "leaders": lindsay},
        {"id": "vezina", "name": "Vezina", "blurb": "Best goaltender", "leaders": vezina},
        {"id": "norris", "name": "Norris", "blurb": "Defenceman value and relative on-ice scoring", "leaders": norris},
        {"id": "calder", "name": "Calder", "blurb": "Top first-year player", "leaders": calder},
        {"id": "selke", "name": "Selke", "blurb": "Defensive on-ice results", "leaders": selke},
        {"id": "lady_byng", "name": "Lady Byng", "blurb": "Most points, fewest penalty minutes", "leaders": lady},
        {"id": "art_ross", "name": "Art Ross", "blurb": "Scoring leader", "leaders": art_ross},
        {"id": "rocket", "name": "Rocket Richard", "blurb": "Goals leader", "leaders": rocket},
    ]
    if standings is not None and teams:
        assign_team_season_expectations(teams)
        masterton = _oldest_club_race(skaters, teams, team_ctx, best=False, limit=limit)
        messier = _oldest_club_race(skaters, teams, team_ctx, best=True, limit=limit)
        if masterton:
            boards.append(masterton)
        if messier:
            boards.append(messier)

    jack, gm_board = _expectation_races(teams, standings, limit=limit)
    if jack:
        boards.append(jack)
    if gm_board:
        boards.append(gm_board)
    return {"boards": boards, "source": "awards_backend"}


def _sealed_award_boards() -> List[Dict[str, Any]]:
    names = (
        ("hart", "Hart"),
        ("ted_lindsay", "Ted Lindsay"),
        ("vezina", "Vezina"),
        ("norris", "Norris"),
        ("calder", "Calder"),
        ("selke", "Selke"),
        ("lady_byng", "Lady Byng"),
        ("art_ross", "Art Ross"),
        ("rocket", "Rocket Richard"),
        ("masterton", "Masterton"),
        ("messier", "Messier"),
        ("jack_adams", "Jack Adams"),
        ("gm_of_the_year", "GM of the Year"),
    )
    return [
        {"id": aid, "name": name, "blurb": "Sealed", "leaders": [], "sealed": True}
        for aid, name in names
    ]


def _oldest_club_race(
    skaters: Sequence[Mapping[str, Any]],
    teams: Sequence[Any],
    team_ctx: Mapping[str, Mapping[str, Any]],
    *,
    best: bool,
    limit: int,
) -> Optional[Dict[str, Any]]:
    tid = _club_id_by_standing(team_ctx, best=best)
    if not tid:
        return None
    pool = [r for r in skaters if str(_tid(r)) == str(tid)]
    ranked = sorted(pool, key=lambda r: (_player_age(r, teams) or 0, _pts(r)), reverse=True)[:limit]
    if not ranked:
        return None
    leaders = []
    for row in ranked:
        age = _player_age(row, teams)
        leaders.append(_race_card(row, [
            _race_metric("AGE", age, "int") if age else None,
            _race_metric("PTS", _pts(row), "int"),
        ]))
    if best:
        return {"id": "messier", "name": "Messier", "blurb": "Oldest player on the first-place team", "leaders": leaders}
    return {"id": "masterton", "name": "Masterton", "blurb": "Oldest player on the last-place team", "leaders": leaders}


def _expectation_races(
    teams: Optional[Sequence[Any]],
    standings: Any,
    *,
    limit: int = 5,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if standings is None or not teams:
        return None, None
    try:
        team_ctx = build_team_context(standings)
    except Exception:
        return None, None
    assign_team_season_expectations(teams)

    def _board(kind: str) -> Optional[Dict[str, Any]]:
        rows = _expectation_rows(teams, team_ctx, kind=kind)
        rows.sort(key=lambda r: (_safe_float(r.get("wins_above"), 0.0), _safe_float(r.get("points"), 0.0)), reverse=True)
        leaders = []
        for row in rows[:limit]:
            leaders.append({
                "name": row.get("name"),
                "team_id": row.get("team_id"),
                "team_abbr": "",
                "position": row.get("position"),
                "gp": row.get("gp"),
                "metrics": [
                    {"label": "W+/-", "value": round(_safe_float(row.get("wins_above"), 0.0), 1), "kind": "signed"},
                    {"label": "W", "value": int(row.get("actual_wins") or 0), "kind": "int"},
                    {"label": "xW", "value": round(_safe_float(row.get("expected_wins"), 0.0), 1), "kind": "signed"},
                ],
            })
        if not leaders:
            return None
        if kind == "gm":
            return {"id": "gm_of_the_year", "name": "GM of the Year", "blurb": "Wins versus the wins the roster was expected to get", "leaders": leaders}
        return {"id": "jack_adams", "name": "Jack Adams", "blurb": "Wins versus the wins the roster was expected to get", "leaders": leaders}

    return _board("coach"), _board("gm")


def _jack_adams_race(teams: Optional[Sequence[Any]], standings: Any, *, limit: int = 5) -> Optional[Dict[str, Any]]:
    """Coaches ranked by recorded standings. Expected-points only when the club actually has one."""
    if standings is None:
        return None
    try:
        table = list(standings.league_table() or [])
    except Exception:
        return None
    team_map: Dict[str, Any] = {}
    for team in teams or []:
        tid = getattr(team, "team_id", None)
        if tid is None:
            tid = getattr(team, "id", None)
        if tid is not None:
            team_map[str(tid)] = team
    leaders: List[Dict[str, Any]] = []
    for rec in table:
        team = team_map.get(str(getattr(rec, "team_id", "") or ""))
        if team is None:
            continue
        coach = getattr(team, "coach", None) or getattr(team, "head_coach", None)
        name = str(getattr(coach, "name", "") or getattr(team, "coach_name", "") or "").strip()
        if not name:
            continue
        points = int(getattr(rec, "points", 0) or 0)
        try:
            goal_diff = int(rec.goal_diff())
        except Exception:
            goal_diff = int(getattr(rec, "gf", 0) or 0) - int(getattr(rec, "ga", 0) or 0)
        expected = getattr(team, "expected_points", None)
        metrics: List[Dict[str, Any]] = [
            {"label": "PTS", "value": points, "kind": "int"},
            {"label": "GD", "value": goal_diff, "kind": "signed"},
        ]
        if expected is not None:
            try:
                metrics.insert(0, {"label": "vs exp", "value": round(float(points) - float(expected), 1), "kind": "signed"})
            except (TypeError, ValueError):
                pass
        abbr = str(getattr(team, "abbreviation", None) or getattr(team, "abbr", None) or "")
        leaders.append(
            {
                "name": name,
                "team_id": str(getattr(rec, "team_id", "") or ""),
                "team_abbr": abbr,
                "position": "HC",
                "gp": int(getattr(rec, "gp", 0) or 0),
                "metrics": metrics,
                "_points": points,
                "_gd": goal_diff,
            }
        )
    if not leaders:
        return None
    if any("vs exp" == m.get("label") for row in leaders for m in row.get("metrics") or []):
        leaders.sort(key=lambda row: (row.get("metrics") or [{}])[0].get("value") or 0, reverse=True)
    else:
        leaders.sort(key=lambda row: (int(row.get("_points") or 0), int(row.get("_gd") or 0)), reverse=True)
    trimmed = []
    for row in leaders[:limit]:
        row.pop("_points", None)
        row.pop("_gd", None)
        trimmed.append(row)
    return {
        "id": "jack_adams",
        "name": "Jack Adams",
        "blurb": "Coach of the year",
        "leaders": trimmed,
    }


def compute_awards(
    standings: StandingsTable,
    playoff_result: Optional[PlayoffResult],
    teams: List[Any],
    player_season_stats: Optional[List[Dict[str, Any]]] = None,
    *,
    playoff_player_stats: Optional[List[Dict[str, Any]]] = None,
    season_seed: Any = None,
    season_year: Any = None,
    season_length: int = 82,
    history_by_player: Optional[Dict[str, Any]] = None,
) -> Dict[str, Award]:
    if season_seed is None:
        season_seed = season_year  # otherwise every season reuses the same ballot RNG

    awards: Dict[str, Award] = {}
    team_map: Dict[str, Any] = {}
    for t in teams or []:
        tid = getattr(t, "team_id", None)
        if tid is None:
            tid = getattr(t, "id", None)
        if tid is not None:
            team_map[str(tid)] = t

    season = season_year
    history_by_player = _merge_history_from_rosters(teams, history_by_player)
    team_ctx = build_team_context(standings)
    tbl = list(standings.league_table() or [])

    # Presidents'
    prez = standings.presidents_trophy_winner()
    if prez is not None:
        defn = AWARD_REGISTRY["presidents"]
        name = _team_name_from_id(team_map, str(prez.team_id), prez.name)
        full = []
        for i, rec in enumerate(tbl):
            st = _standing_stats(rec)
            full.append(
                {
                    "entity_id": str(rec.team_id),
                    "name": _team_name_from_id(team_map, str(rec.team_id), rec.name),
                    "team_id": str(rec.team_id),
                    "team_name": _team_name_from_id(team_map, str(rec.team_id), rec.name),
                    "finish": i + 1,
                    "rank": i + 1,
                    "canonical_score": float(st["points"]),
                    "display_value": st["points"],
                    "display_metric": "PTS",
                    "votes": st["points"],
                    "is_winner": i == 0,
                    **st,
                }
            )
        awards[defn["name"]] = _finalize_player_award(
            defn,
            full,
            season=season,
            quality="full",
            fallback_reason=None,
            eligibility_summary="League standings points race.",
            stat_scope="regular_season",
        )
        awards[defn["name"]].winner_team_id = str(prez.team_id)
        awards[defn["name"]].winner_team_name = name
        awards[defn["name"]].winner_name = name
        awards[defn["name"]].winner_stats = _standing_stats(prez)
        awards[defn["name"]].rationale = (
            f"Best regular-season record ({prez.points} pts, {prez.wins}-{prez.losses}-{prez.otl}, "
            f"GD {prez.goal_diff():+d})."
        )
        awards[defn["name"]].public_rationale = awards[defn["name"]].rationale

    # Stanley Cup
    if playoff_result is not None:
        defn = AWARD_REGISTRY["stanley"]
        champ_id = str(playoff_result.champion_id)
        champ_name = _team_name_from_id(team_map, champ_id)
        candidates = []
        for tid in list(playoff_result.finalist_ids or []):
            rec = next((r for r in tbl if str(r.team_id) == str(tid)), None)
            st = _standing_stats(rec) if rec is not None else {}
            candidates.append(
                {
                    "entity_id": str(tid),
                    "name": _team_name_from_id(team_map, str(tid)),
                    "team_id": str(tid),
                    "team_name": _team_name_from_id(team_map, str(tid)),
                    "finish": 1 if str(tid) == champ_id else 2,
                    "canonical_score": 1.0 if str(tid) == champ_id else 0.0,
                    "display_value": "Champion" if str(tid) == champ_id else "Finalist",
                    "display_metric": "Champion",
                    "is_winner": str(tid) == champ_id,
                    "votes": 1 if str(tid) == champ_id else 0,
                    **st,
                }
            )
        candidates.sort(key=lambda c: (not c["is_winner"], c.get("finish", 99)))
        award = _finalize_player_award(
            defn,
            candidates or [
                {
                    "entity_id": champ_id,
                    "name": champ_name,
                    "team_id": champ_id,
                    "team_name": champ_name,
                    "finish": 1,
                    "is_winner": True,
                    "display_value": "Champion",
                    "display_metric": "Champion",
                    "votes": 1,
                    "canonical_score": 1.0,
                }
            ],
            season=season,
            quality="full",
            fallback_reason=None,
            eligibility_summary="Stanley Cup playoff champion.",
            stat_scope="playoffs",
        )
        award.winner_team_id = champ_id
        award.winner_name = champ_name
        award.winner_team_name = champ_name
        award.rationale = "Won the Stanley Cup after navigating the playoff bracket."
        award.public_rationale = award.rationale
        awards[defn["name"]] = award

        # Conference champions
        confs = extract_conference_champions(playoff_result, team_map, season=season)
        cdef = AWARD_REGISTRY["conference_champions"]
        if confs:
            for i, c in enumerate(confs):
                c["finish"] = i + 1
                c["display_value"] = c.get("conference") or "Conference"
                c["display_metric"] = "Champion"
                c["canonical_score"] = 1.0
                c["votes"] = 1
            awards[cdef["name"]] = _finalize_player_award(
                cdef,
                confs,
                season=season,
                quality="full",
                fallback_reason=None,
                eligibility_summary="Conference playoff champions.",
                stat_scope="playoffs",
                shared_override=True,
            )
        else:
            awards[cdef["name"]] = _unavailable_award(
                cdef,
                reason="Conference metadata unavailable on playoff bracket.",
                season=season,
            )

    raw_rows = [dict(r) for r in (player_season_stats or []) if isinstance(r, Mapping)]
    # Assert mixed scopes do not contaminate regular-season awards.
    rows = filter_regular_season_rows(raw_rows)
    # If callers passed only bare rows without scope, filter keeps them (missing → regular).
    if not rows and raw_rows:
        # Rows may have been wrongly tagged; only accept if none were playoff-tagged.
        if not any(_stat_scope(r) in {"playoff", "playoffs"} for r in raw_rows):
            rows = [dict(r) for r in raw_rows]

    snap_rows = [snapshot_row(r, teams=teams) for r in rows]
    prepare_award_context(snap_rows)
    skaters = [r for r in snap_rows if not _is_goalie(r)]
    goalies = [r for r in snap_rows if _is_goalie(r)]

    # --- TEMP Selke audit: remove once checked ---
    _analytics = ("war", "impact_score", "gsax", "defense_score", "analytics_rating")
    for _r in sorted(eligible_selke(skaters, season_length), key=selke_ballot_score, reverse=True)[:5]:
        logger.warning(
            "SELKE %-22s full=%6.2f fb=%6.2f path=%s def=%s xgf=%s fo_taken=%s fo%%=%s pk_toi=%s pk_ga60=%s tk60=%s pts=%s",
            _r.get("name"),
            selke_ballot_score(_r),
            selke_fallback_formula(dict(_r)),
            "full" if any(_r.get(f) is not None for f in _analytics) else "fallback",
            _r.get("defense_score"),
            _r.get("xgf_pct"),
            _r.get("fo_taken", _r.get("faceoffs_taken")),
            _r.get("faceoff_pct"),
            _r.get("pk_toi"),
            _r.get("pk_xga_per_60", _r.get("pk_ga_per_60")),
            _r.get("takeaways_per_60"),
            _r.get("pts"),
        )

    # Art Ross
    awards[AWARD_REGISTRY["art_ross"]["name"]] = _run_stat_race(
        AWARD_REGISTRY["art_ross"],
        eligible_art_ross(skaters, season_length),
        lambda r: float(_pts(r)),
        [lambda r: float(_goals(r)), lambda r: float(_pts(r)) / max(1, _gp(r)), lambda r: -float(_gp(r))],
        team_map=team_map,
        season=season,
        eligibility_summary="Regular-season skater points race.",
    )

    # Rocket
    awards[AWARD_REGISTRY["rocket"]["name"]] = _run_stat_race(
        AWARD_REGISTRY["rocket"],
        eligible_rocket(skaters, season_length),
        lambda r: float(_goals(r)),
        [lambda r: -float(_gp(r)), lambda r: float(_pts(r))],
        team_map=team_map,
        season=season,
        eligibility_summary="Regular-season goals race.",
    )

    # Hart
    awards[AWARD_REGISTRY["hart"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["hart"],
        eligible_hart(skaters, season_length),
        lambda r: hart_ballot_score(r, team_ctx),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary=f"Meaningful skater participation (>= {_season_games_threshold(season_length, 0.45, 30)} GP).",
        required_fields=["gp"],
        fallback_fn=hart_fallback_formula,
        team_ctx=team_ctx,
        season_length=season_length,
    )

    # Norris
    awards[AWARD_REGISTRY["norris"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["norris"],
        eligible_norris(skaters, season_length),
        lambda r: norris_ballot_score(r, team_ctx),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Defencemen with meaningful GP/TOI.",
        required_fields=["gp"],
        fallback_fn=norris_fallback_formula,
    )

    # Selke
    awards[AWARD_REGISTRY["selke"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["selke"],
        eligible_selke(skaters, season_length),
        selke_ballot_score,
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Forwards with meaningful GP and usage (14:00 TOI/GP or 1:00 PK/GP).",
        required_fields=["gp"],
        team_ctx=team_ctx,
        season_length=season_length,
    )

    # Calder
    calder_pool = eligible_calder(
        snap_rows,
        teams=teams,
        history_by_player=history_by_player,
        season_length=season_length,
    )
    awards[AWARD_REGISTRY["calder"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["calder"],
        calder_pool,
        lambda r: calder_position_score(r, team_ctx),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="NHL rookie rule: no more than 25 GP in any prior season, not 6+ GP in each of two prior seasons, under 26 on Sept. 15.",
        required_fields=["gp"],
        team_ctx=team_ctx,
        # Same position-neutral scale whether or not analytics fields exist.
        fallback_fn=lambda r: (calder_fallback_formula(r, team_ctx), calder_position_score(r, team_ctx))[1],
    )

    # Vezina
    awards[AWARD_REGISTRY["vezina"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["vezina"],
        eligible_vezina(goalies, season_length),
        lambda r: vezina_ballot_score(r, team_ctx),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Starter-level goalie workload (starts/minutes/shots).",
        required_fields=["gp"],
        fallback_fn=vezina_fallback_formula,
    )

    # Lady Byng
    awards[AWARD_REGISTRY["lady_byng"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["lady_byng"],
        eligible_lady_byng(skaters, season_length),
        lady_byng_score,
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Meaningful games with offensive contribution and discipline.",
        required_fields=["gp"],
        fallback_fn=lady_byng_score,
    )

    # Ted Lindsay
    awards[AWARD_REGISTRY["ted_lindsay"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["ted_lindsay"],
        eligible_hart(skaters, season_length),
        lambda r: ted_lindsay_score(r, team_ctx),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Player-focused outstanding season (distinct from Hart team weighting).",
        required_fields=["gp"],
        fallback_fn=lambda r: float(_pts(r)) / max(1, _gp(r)) * 35.0,
        team_ctx=team_ctx,
        season_length=season_length,
    )

    # Jennings
    awards[AWARD_REGISTRY["jennings"]["name"]] = compute_jennings(
        standings, goalies, team_map, season=season, season_length=season_length
    )

    # Conn Smythe
    po_rows = filter_playoff_rows(list(playoff_player_stats or []))
    if not po_rows and playoff_player_stats:
        # If explicitly provided without scope, treat as playoff.
        po_rows = [dict(r) for r in playoff_player_stats if isinstance(r, Mapping)]
    if playoff_result is None:
        awards[AWARD_REGISTRY["conn_smythe"]["name"]] = Award(
            name=AWARD_REGISTRY["conn_smythe"]["name"],
            award_id="conn_smythe",
            status="pending",
            official=True,
            category="playoff",
            recipient_type="player",
            display_metric="Playoff ballot points",
            calculation_quality="unavailable",
            unavailable_reason="Conn Smythe requires a completed Cup Final.",
            rationale="Conn Smythe requires a completed Cup Final.",
            public_rationale="Conn Smythe requires a completed Cup Final.",
            season=season,
            result={
                "award_id": "conn_smythe",
                "name": AWARD_REGISTRY["conn_smythe"]["name"],
                "status": "pending",
                "official": True,
                "winner": None,
                "winners": [],
                "shared": False,
                "finalists": [],
                "full_results": [],
                "display_metric": "Playoff ballot points",
                "calculation_quality": "unavailable",
                "public_rationale": "Conn Smythe requires a completed Cup Final.",
                "stat_scope": "playoffs",
                "season": season,
            },
        )
    elif not po_rows:
        awards[AWARD_REGISTRY["conn_smythe"]["name"]] = _unavailable_award(
            AWARD_REGISTRY["conn_smythe"],
            reason="Playoff player statistics unavailable.",
            season=season,
        )
    else:
        champ = str(getattr(playoff_result, "champion_id", "") or "")
        need = 1
        pool = _isolated_scoring_pool([snapshot_row(r, teams=teams) for r in po_rows if _gp(r) >= need])
        awards[AWARD_REGISTRY["conn_smythe"]["name"]] = _run_ballot_award(
            AWARD_REGISTRY["conn_smythe"],
            pool,
            lambda r: conn_smythe_score(r, champion_id=champ),
            team_map=team_map,
            season_seed=season_seed,
            season=season,
            eligibility_summary="Playoff-only participation after Cup Final.",
            required_fields=["gp"],
            fallback_fn=lambda r: float(_pts(r)) + float(_goals(r)),
            stat_scope="playoffs",
        )

    awards[AWARD_REGISTRY["masterton"]["name"]] = _try_oldest_on_club(
        AWARD_REGISTRY["masterton"], skaters, teams, team_ctx, team_map, season, season_seed, best_club=False,
    )
    awards[AWARD_REGISTRY["messier"]["name"]] = _try_oldest_on_club(
        AWARD_REGISTRY["messier"], skaters, teams, team_ctx, team_map, season, season_seed, best_club=True,
    )
    awards[AWARD_REGISTRY["jack_adams"]["name"]] = _try_expectation_award(
        AWARD_REGISTRY["jack_adams"], teams, team_ctx, team_map, season, season_seed, kind="coach",
    )
    awards[AWARD_REGISTRY["gm_of_the_year"]["name"]] = _try_expectation_award(
        AWARD_REGISTRY["gm_of_the_year"], teams, team_ctx, team_map, season, season_seed, kind="gm",
    )
    a1, a2 = _try_all_star_teams(skaters, goalies, team_map, season, season_seed, team_ctx)
    awards[AWARD_REGISTRY["all_star_1"]["name"]] = a1
    awards[AWARD_REGISTRY["all_star_2"]["name"]] = a2

    _attach_award_explanations(awards, team_ctx)

    if os.environ.get("NHL_FRANCHISE_AUDIT") == "1":
        _log_awards_audit_bundle(awards, skaters=skaters, defense=[r for r in skaters if _is_defense(r)], goalies=goalies, playoff_rows=po_rows)

    return awards


def _log_awards_audit_bundle(
    awards: Mapping[str, Award],
    *,
    skaters: Sequence[Mapping[str, Any]],
    defense: Sequence[Mapping[str, Any]],
    goalies: Sequence[Mapping[str, Any]],
    playoff_rows: Sequence[Mapping[str, Any]],
) -> None:
    for name, aw in awards.items():
        if getattr(aw, "status", None) != "complete":
            continue
        ser = serialize_award(aw)
        top5 = list(ser.get("full_results") or [])[:5]
        logger.info(
            "AWARDS_AUDIT %s",
            json.dumps(
                {
                    "award_id": ser.get("award_id"),
                    "pool_size": len(ser.get("full_results") or []),
                    "path_counts": {"full": len(ser.get("full_results") or []), "fallback": 0},
                    "top5": [
                        {
                            "name": c.get("name"),
                            "pos": c.get("position"),
                            "team": c.get("team_name"),
                            "score": c.get("canonical_score"),
                            "ballot_points": c.get("ballot_points"),
                        }
                        for c in top5
                    ],
                    "winner": ser.get("winner_name"),
                    "public_rationale": ser.get("public_rationale"),
                },
                default=str,
            ),
        )
    def _keys(row: Optional[Mapping[str, Any]]) -> List[str]:
            return sorted(str(k) for k in (row or {}).keys())

    logger.info(
        "AWARDS_AUDIT_ROW_KEYS %s",
        json.dumps(
            {
                "forward": _keys(next((r for r in skaters if _is_forward(r)), None)),
                "defense": _keys(next((r for r in defense), None)),
                "goalie": _keys(next((r for r in goalies), None)),
                "playoff": _keys(next((r for r in playoff_rows), None)),
            }
        ),
    )


def assign_team_season_expectations(
    teams: Sequence[Any],
    strength_map: Optional[Mapping[str, Any]] = None,
    *,
    games: int = 82,
) -> None:
    """Opening-night wins and points, from team strength. Set once per season."""
    games_n = max(1, int(games or 82))
    for idx, team in enumerate(teams or []):
        if getattr(team, "expected_wins", None) is not None and getattr(team, "expected_points", None) is not None:
            continue
        tid = getattr(team, "team_id", None)
        if tid is None:
            tid = getattr(team, "id", None)
        if tid is None:
            tid = idx
        strength = None
        if strength_map is not None:
            try:
                strength = float(strength_map.get(str(tid)))  # type: ignore[union-attr]
            except (TypeError, ValueError):
                strength = None
        if strength is None:
            vals: List[float] = []
            for player in getattr(team, "roster", None) or []:
                raw = getattr(player, "ovr", None)
                try:
                    val = float(raw() if callable(raw) else raw)
                except (TypeError, ValueError):
                    continue
                if val <= 1.5:
                    val *= 99.0
                if val > 0:
                    vals.append(val)
            avg = (sum(vals) / len(vals)) if vals else 78.0
            strength = max(0.2, min(0.95, (avg - 68.0) / 28.0))
        strength = max(0.15, min(1.0, float(strength)))
        win_rate = 0.33 + 0.44 * strength
        expected_wins = round(games_n * win_rate, 1)
        setattr(team, "expected_wins", expected_wins)
        setattr(team, "expected_points", round(expected_wins * 2.08, 1))


def _club_id_by_standing(team_ctx: Mapping[str, Mapping[str, Any]], *, best: bool) -> Optional[str]:
    ranked: List[Tuple[float, float, float, str]] = []
    for tid, ctx in (team_ctx or {}).items():
        ranked.append((
            _safe_float(ctx.get("points"), 0.0),
            _safe_float(ctx.get("wins"), 0.0),
            _safe_float(ctx.get("goal_diff"), 0.0),
            str(tid),
        ))
    if not ranked:
        return None
    ranked.sort(reverse=best)
    return ranked[0][3]


def _try_oldest_on_club(defn, rows, teams, team_ctx, team_map, season, season_seed, *, best_club: bool) -> Award:
    tid = _club_id_by_standing(team_ctx, best=best_club)
    pool = [r for r in rows if tid and str(_tid(r)) == str(tid) and _gp(r) >= 1 and not _is_goalie(r)]
    if not pool:
        label = "best" if best_club else "worst"
        return _unavailable_award(defn, reason=f"No skaters on the {label} club.", season=season)
    summary = (
        "Oldest skater on the first-place club."
        if best_club
        else "Oldest skater on the last-place club."
    )
    return _run_ballot_award(
        defn,
        pool,
        lambda r: float(_player_age(r, teams) or 0),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary=summary,
        required_fields=["gp"],
    )


def _expectation_rows(teams, team_ctx, *, kind: str) -> List[Dict[str, Any]]:
    assign_team_season_expectations(teams)
    rows: List[Dict[str, Any]] = []
    for team in teams or []:
        tid = getattr(team, "team_id", None)
        if tid is None:
            tid = getattr(team, "id", None)
        ctx = (team_ctx or {}).get(str(tid), {})
        actual_wins = ctx.get("wins")
        if actual_wins is None:
            continue
        expected_wins = _safe_float(getattr(team, "expected_wins", None), 0.0)
        if kind == "gm":
            name = str(getattr(team, "gm_name", "") or "").strip() or f"GM {getattr(team, 'abbreviation', '') or tid}"
            entity = f"gm:{tid}"
            position = "GM"
        else:
            coach = getattr(team, "coach", None) or getattr(team, "head_coach", None)
            name = str(getattr(coach, "name", "") or getattr(team, "coach_name", "") or "").strip()
            entity = str(getattr(coach, "id", "") or getattr(team, "coach_id", "") or f"coach:{tid}")
            position = "HC"
        if not name:
            continue
        rows.append({
            "player_id": entity,
            "name": name,
            "team_id": str(tid or ""),
            "gp": int(ctx.get("gp") or 1),
            "position": position,
            "wins_above": float(actual_wins) - expected_wins,
            "expected_wins": expected_wins,
            "actual_wins": float(actual_wins),
            "points": int(ctx.get("points") or 0),
        })
    return rows


def _try_expectation_award(defn, teams, team_ctx, team_map, season, season_seed, *, kind: str) -> Award:
    rows = _expectation_rows(teams, team_ctx, kind=kind)
    if not rows:
        return _unavailable_award(defn, reason="Standings are not available yet.", season=season)
    who = "General manager" if kind == "gm" else "Coach"
    return _run_ballot_award(
        defn,
        rows,
        lambda r: _safe_float(r.get("wins_above"), 0.0),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary=f"{who} wins compared with the wins expected from the roster.",
        required_fields=["gp"],
    )


def _try_all_star_teams(skaters, goalies, team_map, season, season_seed, team_ctx) -> Tuple[Award, Award]:
    def pick(pos_pool, n, score_fn):
        ranked = sorted(pos_pool, key=score_fn, reverse=True)
        return ranked[:n]

    centers = [r for r in skaters if _pos(r) in {"C", "CENTER"}] or [r for r in skaters if _is_forward(r)]
    wings = [r for r in skaters if _pos(r) in {"L", "R", "LW", "RW", "W"}] or [r for r in skaters if _is_forward(r)]
    defs = [r for r in skaters if _is_defense(r)]
    if not (centers and wings and defs and goalies):
        reason = "Insufficient positional season data for All-Star Teams."
        return (
            _unavailable_award(AWARD_REGISTRY["all_star_1"], reason=reason, season=season),
            _unavailable_award(AWARD_REGISTRY["all_star_2"], reason=reason, season=season),
        )

    def skor(r):
        return hart_ballot_score(r, team_ctx) if not _is_defense(r) else norris_ballot_score(r, team_ctx)

    first = pick(centers, 1, skor) + pick(wings, 2, skor) + pick(defs, 2, skor) + pick(goalies, 1, lambda r: vezina_ballot_score(r, team_ctx))
    used = {_pid(r) for r in first}
    rest_c = [r for r in centers if _pid(r) not in used]
    rest_w = [r for r in wings if _pid(r) not in used]
    rest_d = [r for r in defs if _pid(r) not in used]
    rest_g = [r for r in goalies if _pid(r) not in used]
    second = pick(rest_c, 1, skor) + pick(rest_w, 2, skor) + pick(rest_d, 2, skor) + pick(rest_g, 1, lambda r: vezina_ballot_score(r, team_ctx))

    def to_award(defn, rows):
        full = []
        for i, r in enumerate(rows):
            full.append(
                {
                    "entity_id": _pid(r),
                    "player_id": _pid(r),
                    "name": str(r.get("name") or ""),
                    "team_id": _tid(r),
                    "team_name": _team_name_from_id(team_map, _tid(r)),
                    "position": _pos(r),
                    "finish": i + 1,
                    "canonical_score": float(skor(r) if not _is_goalie(r) else vezina_ballot_score(r, team_ctx)),
                    "display_value": "Selection",
                    "display_metric": "Selection",
                    "is_winner": True,
                    "votes": 1,
                    "points": _pts(r),
                    "goals": _goals(r),
                    "gp": _gp(r),
                    "component_scores": {},
                    "eligibility": {},
                }
            )
        return _finalize_player_award(
            defn,
            full,
            season=season,
            quality="full",
            fallback_reason=None,
            eligibility_summary="Position-aware season-end All-Star selections.",
            stat_scope="regular_season",
            shared_override=True,
        )

    return to_award(AWARD_REGISTRY["all_star_1"], first), to_award(AWARD_REGISTRY["all_star_2"], second)


def serialize_award(award: Award) -> Dict[str, Any]:
    base = {
        "name": award.name,
        "award_id": award.award_id or NAME_TO_ID.get(award.name, ""),
        "winner_name": award.winner_name,
        "winner_team_id": award.winner_team_id,
        "winner_player_id": award.winner_player_id,
        "winner_team_name": award.winner_team_name,
        "finalists": list(award.finalists or []),
        "candidates": list(award.candidates or []),
        "winner_stats": dict(award.winner_stats or {}),
        "rationale": award.rationale or award.public_rationale,
        "public_rationale": award.public_rationale or award.rationale,
        "winners": list(award.winners or []),
        "shared": bool(award.shared),
        "full_results": list(award.full_results or []),
        "status": award.status,
        "official": award.official,
        "category": award.category,
        "recipient_type": award.recipient_type,
        "display_metric": award.display_metric,
        "calculation_quality": award.calculation_quality,
        "fallback_reason": award.fallback_reason,
        "unavailable_reason": award.unavailable_reason,
        "eligibility_summary": award.eligibility_summary,
        "stat_scope": award.stat_scope,
        "season": award.season,
        "voting": award.voting,
        "evidence": (award.result or {}).get("evidence") if award.result else None,
    }
    if award.result:
        base.update({k: v for k, v in award.result.items() if k not in base or base.get(k) in (None, "", [], {})})
        base["result"] = award.result
    return base


def build_awards_payload(
    awards: Mapping[str, Award],
    *,
    season: Any = None,
    season_seed: Any = None,
    season_length: int = 82,
) -> Dict[str, Any]:
    awards_dict = {k: serialize_award(v) for k, v in awards.items()}
    official_results = []
    full_ballots = {}
    team_achievements = []
    all_star_teams: Dict[str, Any] = {}
    reveal = []
    for name, aw in awards.items():
        ser = awards_dict[name]
        aid = ser.get("award_id") or NAME_TO_ID.get(name, "")
        official_results.append(ser)
        if ser.get("full_results"):
            full_ballots[aid or name] = ser["full_results"]
        if aid in {"presidents", "stanley", "conference_champions", "jennings"}:
            team_achievements.append(ser)
        if aid == "all_star_1":
            all_star_teams["first"] = ser
        if aid == "all_star_2":
            all_star_teams["second"] = ser
        if ser.get("status") == "complete" and ser.get("ceremony_enabled", AWARD_REGISTRY.get(aid, {}).get("ceremony_enabled", True)):
            if aid and AWARD_REGISTRY.get(aid, {}).get("ceremony_enabled", True):
                reveal.append(aid)

    order = [
        "presidents",
        "jennings",
        "rocket",
        "art_ross",
        "lady_byng",
        "selke",
        "calder",
        "norris",
        "vezina",
        "ted_lindsay",
        "all_star_2",
        "all_star_1",
        "hart",
        "conn_smythe",
        "stanley",
    ]
    reveal_order = [a for a in order if a in reveal] + [a for a in reveal if a not in order]

    return {
        "season": season,
        "status": "complete",
        "official_results": official_results,
        "full_ballots": full_ballots,
        "team_achievements": team_achievements,
        "all_star_teams": all_star_teams,
        "ceremony": {
            "reveal_order": reveal_order,
            "catalog": {k: {"award_id": v["award_id"], "name": v["name"], "display_metric": v["display_metric"], "ceremony_enabled": v["ceremony_enabled"], "official": v["official"]} for k, v in AWARD_REGISTRY.items()},
        },
        "metadata": {
            "computed_at_stage": "post_cup_pre_offseason_mutation",
            "seed": season_seed,
            "season_length": season_length,
            "registry_version": 2,
        },
        # Legacy
        "awards": awards_dict,
        "items": list(awards_dict.values()),
    }


def apply_career_award_history(
    teams: Sequence[Any],
    awards: Mapping[str, Award],
    season: Any,
    *,
    result_id: str,
    history_by_player: Optional[MutableMapping[str, Any]] = None,
) -> int:
    """Idempotently append award history onto player objects. Returns writes count."""
    writes = 0
    for _name, award in awards.items():
        if getattr(award, "status", "complete") != "complete":
            continue
        recipients = list(award.winners or [])
        if not recipients and award.winner_player_id:
            recipients = [
                {
                    "player_id": award.winner_player_id,
                    "entity_id": award.winner_player_id,
                    "name": award.winner_name,
                    "team_id": award.winner_team_id,
                    "team_name": award.winner_team_name,
                    "finish": 1,
                    "ballot_points": None,
                    "first_place_votes": None,
                }
            ]
        for rec in recipients:
            pid = str(rec.get("player_id") or rec.get("entity_id") or "")
            if not pid:
                continue
            player = _player_from_rosters(teams, pid)
            if player is None:
                continue
            entry = {
                "award_result_id": f"{result_id}:{award.award_id or NAME_TO_ID.get(award.name, award.name)}:{pid}",
                "award_id": award.award_id or NAME_TO_ID.get(award.name, ""),
                "award_name": award.name,
                "season": season,
                "team_id": rec.get("team_id") or award.winner_team_id,
                "team_name": rec.get("team_name") or award.winner_team_name,
                "position": rec.get("position"),
                "ballot_points": rec.get("ballot_points"),
                "first_place_votes": rec.get("first_place_votes"),
                "finish": rec.get("finish") or 1,
                "winning_stats": dict(award.winner_stats or {}),
                "calculation_quality": award.calculation_quality,
            }
            history = list(getattr(player, "career_awards", None) or [])
            if any(isinstance(h, dict) and h.get("award_result_id") == entry["award_result_id"] for h in history):
                continue
            # Also support string awards_won list for compat without duplicating same season+award
            won = list(getattr(player, "awards_won", None) or [])
            token = f"{entry['award_id']}:{season}"
            if token not in won:
                won.append(token)
                try:
                    player.awards_won = won
                except Exception:
                    pass
            history.append(entry)
            try:
                player.career_awards = history
            except Exception:
                pass
            hist_blob = dict(getattr(player, "player_award_history", None) or {})
            if history_by_player is not None:
                hist_blob = {**dict(history_by_player.get(pid) or {}), **hist_blob}
            awards_hist = list(hist_blob.get("awards") or [])
            if not any(
                isinstance(h, dict) and h.get("award_result_id") == entry["award_result_id"] for h in awards_hist
            ):
                awards_hist.append(entry)
            hist_blob["awards"] = awards_hist
            hist_blob["last_award_season"] = season
            try:
                player.player_award_history = hist_blob
            except Exception:
                pass
            if history_by_player is not None:
                history_by_player[pid] = hist_blob
            writes += 1
    return writes


def compute_official_watch_lists(
    rows: Iterable[Mapping[str, Any]],
    *,
    standings: Any = None,
    rank_by_tid: Optional[Mapping[str, int]] = None,
    limit: int = 10,
    season_length: int = 82,
    history_by_player: Optional[Mapping[str, Any]] = None,
    teams: Optional[Sequence[Any]] = None,
    season_seed: Any = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Official Award Watch lists using the same scoring as compute_awards."""
    team_ctx: Dict[str, Dict[str, Any]] = {}
    if standings is not None and hasattr(standings, "league_table"):
        try:
            team_ctx = build_team_context(standings)
        except Exception:
            team_ctx = {}
    if not team_ctx and rank_by_tid:
        for tid, rk in rank_by_tid.items():
            team_ctx[str(tid)] = {
                "points_pct_norm": max(0.0, 1.0 - (int(rk) - 1) / 31.0),
                "goal_diff_norm": 0.5,
                "playoff_qualified": int(rk) <= 16,
            }

    snap = [snapshot_row(r, teams=teams) for r in filter_regular_season_rows(list(rows))]
    if not snap:
        snap = [snapshot_row(r, teams=teams) for r in rows if isinstance(r, Mapping)]

    skaters = [r for r in snap if not _is_goalie(r)]
    goalies = [r for r in snap if _is_goalie(r)]
    out: Dict[str, List[Dict[str, Any]]] = {}

    def pack(award_id: str, pool: List[Dict[str, Any]], score_fn, metric_key: str) -> List[Dict[str, Any]]:
        defn = AWARD_REGISTRY[award_id]
        ranked = sorted(pool, key=lambda r: float(score_fn(r)), reverse=True)[: max(1, int(limit))]
        rows_out = []
        for i, r in enumerate(ranked):
            score = float(score_fn(r))
            rows_out.append(
                {
                    "player_id": _pid(r),
                    "name": r.get("name"),
                    "team_id": _tid(r),
                    "position": _pos(r),
                    "gp": _gp(r),
                    "pts": _pts(r),
                    "g": _goals(r),
                    "award_score": score,
                    "award_name": defn["name"],
                    "award_trophy_key": award_id,
                    "official": True,
                    "watch_type": defn.get("watch_type"),
                    "ceremony_enabled": defn.get("ceremony_enabled"),
                    "display_metric": defn.get("display_metric"),
                    "display_value": score if metric_key == "score" else r.get(metric_key),
                    "calculation_quality": "documented_fallback"
                    if not any(r.get(f) is not None for f in ("war", "impact_score", "gsax", "defense_score"))
                    and award_id in {"hart", "norris", "selke", "vezina", "calder"}
                    else "full",
                    "eligibility_confidence": (r.get("_eligibility") or {}).get("confidence"),
                    "rank": i + 1,
                }
            )
        return rows_out

    out["art_ross"] = pack("art_ross", eligible_art_ross(skaters, season_length), _pts, "pts")
    out["rocket"] = pack("rocket", eligible_rocket(skaters, season_length), _goals, "g")
    out["hart"] = pack("hart", eligible_hart(skaters, season_length), lambda r: hart_ballot_score(r, team_ctx), "score")
    out["norris"] = pack("norris", eligible_norris(skaters, season_length), lambda r: norris_ballot_score(r, team_ctx), "score")
    out["selke"] = pack("selke", eligible_selke(skaters, season_length), selke_ballot_score, "score")
    out["calder"] = pack(
        "calder",
        eligible_calder(snap, teams=teams, history_by_player=history_by_player, season_length=season_length),
        lambda r: calder_position_score(r, team_ctx),
        "score",
    )
    out["vezina"] = pack("vezina", eligible_vezina(goalies, season_length), lambda r: vezina_ballot_score(r, team_ctx), "score")
    out["lady_byng"] = pack("lady_byng", eligible_lady_byng(skaters, season_length), lady_byng_score, "score")
    out["ted_lindsay"] = pack(
        "ted_lindsay",
        eligible_hart(skaters, season_length),
        lambda r: ted_lindsay_score(r, team_ctx),
        "score",
    )
    out["conn_smythe"] = []  # requires playoff scope; filled when playoff rows provided by caller
    out["jennings"] = []
    if standings is not None and hasattr(standings, "league_table"):
        try:
            tbl = list(standings.league_table() or [])
            for i, rec in enumerate(sorted(tbl, key=lambda r: int(getattr(r, "ga", 0) or 0))[:limit]):
                out["jennings"].append(
                    {
                        "team_id": str(rec.team_id),
                        "name": getattr(rec, "name", str(rec.team_id)),
                        "ga": int(getattr(rec, "ga", 0) or 0),
                        "award_score": -float(getattr(rec, "ga", 0) or 0),
                        "award_name": AWARD_REGISTRY["jennings"]["name"],
                        "award_trophy_key": "jennings",
                        "official": True,
                        "watch_type": "official_live_race",
                        "display_metric": "Team GA",
                        "display_value": int(getattr(rec, "ga", 0) or 0),
                        "rank": i + 1,
                        "calculation_quality": "full",
                    }
                )
        except Exception:
            pass
    return out
