"""
In-season prospect growth.

Prospects (junior / college / Europe / minors / drafted-but-unsigned) grow *gradually*
through the season: one small OVR pulse per tick, plus a monthly potential review driven by
their real production. OVR and potential move on separate tracks, so they need not rise
together. The offseason pass only pays out whatever growth is still owed
(see ``prospect_offseason_leftover``), so the season's budget is never paid twice.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional

# Players up to this age are treated as growing prospects.
PROSPECT_MAX_AGE_DEV_LEAGUE = 23
PROSPECT_MAX_AGE_ORG = 22


def _age(p: Any) -> int:
    ident = getattr(p, "identity", None)
    raw = getattr(ident, "age", None) if ident is not None else None
    if raw is None:
        raw = getattr(p, "age", 99)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 99


def _has_ratings(p: Any) -> bool:
    r = getattr(p, "ratings", None)
    return isinstance(r, dict) and bool(r)


def iter_growth_prospects(league: Any) -> Iterable[Any]:
    """Every growing prospect once (dev leagues, team prospect pools, AHL/ECHL kids)."""
    seen: set = set()

    def _ok(p: Any, max_age: int) -> bool:
        if p is None or getattr(p, "retired", False) or id(p) in seen:
            return False
        if _age(p) > max_age:
            return False
        seen.add(id(p))
        return True

    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if _ok(p, PROSPECT_MAX_AGE_DEV_LEAGUE):
                    yield p
    for tm in getattr(league, "teams", None) or []:
        for attr in ("prospect_pool", "ahl_roster", "echl_roster"):
            for p in getattr(tm, attr, None) or []:
                if _ok(p, PROSPECT_MAX_AGE_ORG):
                    yield p


def is_growth_prospect(p: Any, on_nhl_roster: bool = False) -> bool:
    if on_nhl_roster or p is None or getattr(p, "retired", False):
        return False
    return _age(p) <= PROSPECT_MAX_AGE_DEV_LEAGUE


def snapshot_prospect_season_start(session: Any) -> int:
    """Freeze display OVR/potential at season open so the UI can show growth since October."""
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return 0
    try:
        from app.sim_engine.entities.player import display_rating, player_current_ovr_01
    except Exception:
        return 0
    n = 0
    for p in iter_growth_prospects(league):
        try:
            if not _has_value_range(p):
                ovr = int(round(display_rating(player_current_ovr_01(p))))
                setattr(p, "season_start_ovr", ovr)
                setattr(p, "_season_start_ovr", ovr)
                setattr(p, "_in_season_ovr_delta_accum", 0.0)
            pot = _pot_display(p)
            if pot is not None:
                setattr(p, "season_start_potential", pot)
            n += 1
        except Exception:
            continue
    return n


def _has_value_range(p: Any) -> bool:
    dr = getattr(p, "draft_value_range", None)
    return not _has_ratings(p) and isinstance(dr, (tuple, list)) and len(dr) >= 2


def _pot_display(p: Any) -> Optional[int]:
    """Displayed potential (0-99): player.potential, ratings.dev_potential, or the pool band's top."""
    from app.sim_engine.entities.player import display_rating, normalize_rating

    raw = getattr(p, "potential", None)
    if raw is None:
        r = getattr(p, "ratings", None)
        raw = r.get("dev_potential") if isinstance(r, dict) else None
    if raw is None and _has_value_range(p):
        try:
            raw = float(p.draft_value_range[1])
        except (TypeError, ValueError):
            raw = None
    if raw is None:
        return None
    try:
        return int(round(display_rating(normalize_rating(raw))))
    except Exception:
        return None


def _team_lookup(league: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tm in getattr(league, "teams", None) or []:
        for attr in ("team_id", "id", "abbr"):
            v = getattr(tm, attr, None)
            if v is not None and str(v).strip():
                out[str(v).strip()] = tm
    return out


def _apply_environment(p: Any, team: Optional[Any]) -> None:
    """Same contextual modifiers as the unsigned-prospect path (league, coaching, org plan)."""
    from services.unsigned_prospect_development import _league_quality

    league_id = str(getattr(p, "current_league_id", "") or getattr(p, "development_path", "") or "")
    lq = _league_quality(league_id)
    coach = float(getattr(p, "coaching_quality", None) or 0.55)
    plan = 0.55
    if team is not None:
        plan = float(
            getattr(team, "prospect_pipeline_score", None)
            or getattr(team, "development_plan_score", None)
            or 0.55
        )
    org_mod = 0.86 + 0.22 * min(1.0, plan) - 0.04 * max(0.0, plan - 0.85)
    coach_mod = 0.90 + 0.16 * min(1.0, coach)
    try:
        setattr(p, "_league_quality_mod", 0.88 + 0.2 * lq)
        setattr(p, "_dev_env_growth_mult", org_mod * coach_mod)
    except Exception:
        pass


_POOL_TIER_GROWTH = {
    "elite": (0.018, 0.048),
    "high": (0.014, 0.034),
    "mid": (0.004, 0.019),
    "depth": (0.002, 0.013),
    "longshot": (0.0, 0.010),
}
_POOL_CURVE_MULT = {"fast": 1.35, "normal": 1.0, "slow": 0.62, "boom_bust": 1.05}
_POOL_INSEASON_SHARE = 0.6  # of expected annual growth; the offseason pays the rest


def _pool_tier(p: Any, mid: float) -> str:
    tier = str(getattr(p, "_pipeline_potential_tier", "") or "").lower()
    if tier in _POOL_TIER_GROWTH:
        return tier
    for name, floor in (("elite", 0.82), ("high", 0.72), ("mid", 0.58), ("depth", 0.48)):
        if mid >= floor:
            return name
    return "longshot"


def apply_pool_range_pulse(p: Any, team: Optional[Any], rng: random.Random, season_id: Any) -> bool:
    """
    Gradual in-season growth for pipeline prospects that only carry a value band
    (``draft_value_range`` = (floor, ceiling)). The floor tracks current ability and rises
    steadily; the ceiling (potential) rises faster or slower with real production, so the two
    need not move together. The offseason pass subtracts ``_pool_inseason_spent``.
    """
    from app.sim_engine.progression.development import (
        _PROSPECT_POT_SEASON_CAP,
        _PROSPECT_POT_SEASON_CAP_TRANSCENDENT,
        _PROSPECT_PULSES_PER_SEASON,
        prospect_performance_evidence,
    )

    lo, hi = float(p.draft_value_range[0]), float(p.draft_value_range[1])
    plan = getattr(p, "_pool_season_plan", None)
    if not isinstance(plan, dict) or plan.get("season") != season_id:
        tier = _pool_tier(p, (lo + hi) / 2.0)
        base_lo, base_hi = _POOL_TIER_GROWTH[tier]
        curve = str(getattr(p, "_pipeline_dev_curve", "normal") or "normal").lower()
        pscore = 0.5
        if team is not None:
            pscore = max(0.0, min(1.0, float(getattr(team, "prospect_pipeline_score", 0.5) or 0.5)))
        env = 0.82 + 0.36 * pscore
        total = rng.uniform(base_lo, base_hi) * _POOL_CURVE_MULT.get(curve, 1.0) * env * _POOL_INSEASON_SHARE
        plan = {"season": season_id, "total": total, "spent": 0.0, "pulses": 0, "hi_gain": 0.0,
                "start_hi": hi}
        setattr(p, "_pool_season_plan", plan)
        setattr(p, "_pool_inseason_spent", 0.0)
    if plan["pulses"] >= _PROSPECT_PULSES_PER_SEASON:
        return False
    plan["pulses"] += 1
    remaining = plan["total"] - plan["spent"]
    if remaining <= 1e-5:
        return False
    step = min(remaining, plan["total"] / _PROSPECT_PULSES_PER_SEASON * rng.uniform(0.7, 1.3))

    evidence, gp = prospect_performance_evidence(p)
    factor = max(0.4, min(1.6, 1.0 + 0.8 * evidence * min(1.0, gp / 30.0)))
    cap = (
        _PROSPECT_POT_SEASON_CAP_TRANSCENDENT
        if getattr(p, "_pipeline_franchise_flag", False)
        else _PROSPECT_POT_SEASON_CAP
    ) / 99.0
    hi_gain = min(step * factor, max(0.0, cap - plan["hi_gain"]))
    ceil = float(getattr(p, "_pipeline_ceiling", plan["start_hi"]))
    new_hi = max(lo, min(hi + hi_gain, ceil + 0.04, 0.99))
    new_lo = min(lo + step * 0.55, new_hi - 0.02)
    p.draft_value_range = (new_lo, new_hi)
    plan["spent"] += step
    plan["hi_gain"] += max(0.0, new_hi - hi)
    setattr(p, "_pool_inseason_spent", plan["spent"])
    return True


def prospect_in_season_tick(session: Any) -> Dict[str, int]:
    """One growth pulse for every prospect; potential review every few pulses."""
    from app.sim_engine.progression.development import (
        _PROSPECT_POT_REVIEW_EVERY,
        apply_prospect_in_season_pulse,
        apply_prospect_potential_review,
    )

    phase = str(getattr(session, "phase", "") or "")
    if phase in ("offseason", "preseason", "post_cup"):
        return {"moved": 0, "potential": 0}
    if not bool(getattr(session, "_regular_stats_split_done", False)):
        return {"moved": 0, "potential": 0}
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return {"moved": 0, "potential": 0}
    rng = getattr(getattr(session, "sim", None), "rng", None)
    if not isinstance(rng, random.Random):
        rng = random.Random()
    season_id = int(getattr(session, "season_calendar_year", 2025) or 2025)
    teams = _team_lookup(league)

    moved = 0
    pot_moved = 0
    for p in iter_growth_prospects(league):
        if _has_value_range(p):
            try:
                rights = getattr(p, "nhl_rights_team_id", None) or getattr(p, "team_id", None)
                if apply_pool_range_pulse(p, teams.get(str(rights)) if rights is not None else None, rng, season_id):
                    moved += 1
            except Exception:
                pass
            continue
        if not _has_ratings(p):
            continue
        try:
            rights = getattr(p, "nhl_rights_team_id", None)
            _apply_environment(p, teams.get(str(rights)) if rights is not None else None)
            if apply_prospect_in_season_pulse(p, rng, season_id) != 0.0:
                moved += 1
            plan = getattr(p, "_prospect_season_plan", None) or {}
            pulses = int(plan.get("pulses", 0))
            if pulses and pulses % _PROSPECT_POT_REVIEW_EVERY == 0:
                res = apply_prospect_potential_review(p, rng, season_id)
                if res.get("applied") and abs(float(res.get("display_delta", 0.0) or 0.0)) > 0:
                    pot_moved += 1
        except Exception:
            continue
    if moved or pot_moved:
        try:
            session._cached_roster_browser_payload = None
        except Exception:
            pass
    return {"moved": moved, "potential": pot_moved}


def prospect_growth_fields(p: Any) -> Dict[str, Any]:
    """UI payload: OVR change and potential change since season start."""
    out: Dict[str, Any] = {}
    try:
        from app.sim_engine.entities.player import display_rating, player_current_ovr_01

        start = getattr(p, "season_start_ovr", None)
        if start is not None:
            cur = int(round(display_rating(player_current_ovr_01(p))))
            out["ovr_change_season"] = cur - int(start)
        pstart = getattr(p, "season_start_potential", None)
        pcur = _pot_display(p)
        if pstart is not None and pcur is not None:
            out["potential_change_season"] = pcur - int(pstart)
    except Exception:
        return out
    change = out.get("potential_change_season")
    if change is not None:
        out["potential_trend"] = "rising" if change >= 1 else ("falling" if change <= -1 else "steady")
    return out
