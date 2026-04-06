# app/sim_engine/tuning/probability_tables.py
"""
Context-aware probabilities: trades, breakouts, regression, narrative roles.
Chaos index widens effective randomness (higher chaos → more extreme rolls).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from app.sim_engine.tuning.era_modifiers import resolve_era_profile


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def chaos_multiplier(league_state: Mapping[str, Any]) -> float:
    """Higher chaos → scale toward more volatile outcomes (1.0 at chaos 0.5)."""
    c = float(league_state.get("chaos_index", 0.5) or 0.5)
    return _clamp(0.72 + 0.56 * c, 0.65, 1.35)


def get_trade_probability(team: Any, league_state: Mapping[str, Any]) -> float:
    """
    Non-linear trade likelihood from team identity, cap pressure proxy, contention, chaos.
    Returns value in ~[0.12, 0.92] suitable for year-level pressure scaling.
    """
    tid = str(getattr(team, "team_id", getattr(team, "id", "")) or "")
    arche = str(league_state.get("team_archetypes", {}).get(tid, "balanced"))
    cap_m = float(league_state.get("salary_cap_m", 92.0) or 92.0)
    cap_pressure = _clamp((95.0 - cap_m) / 18.0, 0.0, 1.0)
    parity = float(league_state.get("parity_index", 0.5) or 0.5)
    chaos_m = chaos_multiplier(league_state)

    base = 0.38
    if "rebuild" in arche or arche == "chaos_agent":
        base += 0.14
    elif arche == "win_now":
        base += 0.10
    elif arche == "draft_and_develop":
        base += 0.04

    base += 0.12 * cap_pressure
    base += 0.08 * (1.0 - parity)
    # Non-linear squash
    x = base * chaos_m
    return _clamp(0.5 * (1.0 - math.exp(-2.2 * x)), 0.12, 0.92)


def get_breakout_probability(player: Any, league_state: Mapping[str, Any]) -> float:
    """Young players with morale/role upside; era dev boost; chaos widens tail."""
    ident = getattr(player, "identity", None)
    age = int(ident.age) if ident is not None and hasattr(ident, "age") else int(getattr(player, "age", 24))
    ovr_fn = getattr(player, "ovr", None)
    try:
        ovr = float(ovr_fn()) if callable(ovr_fn) else float(getattr(player, "ovr", 0.5))
    except Exception:
        ovr = 0.5
    if ovr > 1.2:
        ovr = ovr / 99.0
    morale = float(getattr(getattr(player, "psych", None), "morale", 0.5) or 0.5)
    era = resolve_era_profile(str(league_state.get("active_era", "") or ""))
    dev = float(getattr(player, "_tuning_era_dev", era.get("prospect_growth_boost", 1.0)))
    chaos_m = chaos_multiplier(league_state)
    health = float(league_state.get("league_health", 0.55) or 0.55)
    parity = float(league_state.get("parity_index", 0.5) or 0.5)

    if age >= 26 or ovr >= 0.90:
        return 0.01
    p = 0.05
    if age <= 21:
        p += 0.06
    elif age <= 24:
        p += 0.03
    p += 0.08 * max(0.0, morale - 0.45)
    p += 0.06 * max(0.0, 0.78 - ovr)
    p *= dev
    p *= chaos_m
    if health < 0.43:
        p *= 1.0 + 0.12 * (0.43 - health)
    if parity > 0.63:
        p *= 1.0 - 0.08 * min(1.0, (parity - 0.63) / 0.25)
    return _clamp(p, 0.02, 0.22)


def get_regression_probability(player: Any, age: int, era_key: str) -> float:
    """Post-peak regression likelihood; dead puck / goalie eras slow star fade slightly for goalies."""
    era = resolve_era_profile(era_key or "")
    aging = float(era.get("aging_penalty", 1.0))
    ovr_fn = getattr(player, "ovr", None)
    try:
        ovr = float(ovr_fn()) if callable(ovr_fn) else float(getattr(player, "ovr", 0.5))
    except Exception:
        ovr = 0.5
    if ovr > 1.2:
        ovr = ovr / 99.0
    peak = 27
    career = getattr(player, "career", None)
    if career is not None and hasattr(career, "expected_peak_age"):
        peak = int(career.expected_peak_age)
    if age <= peak:
        return 0.02
    years_past = age - peak
    p = 0.08 + 0.035 * years_past
    p *= aging
    if ovr >= 0.85:
        p *= 0.82
    wear = float(getattr(getattr(player, "health", None), "wear_and_tear", 0.0) or 0.0)
    p += 0.12 * wear
    return _clamp(p, 0.04, 0.55)


def determine_player_role(player: Any, team: Any, league_state: Mapping[str, Any]) -> str:
    """
    Narrative-driven role from performance trend, age, usage proxy, era.
    Sets player.role and player.role_narrative when possible.
    """
    ovr_fn = getattr(player, "ovr", None)
    try:
        ovr = float(ovr_fn()) if callable(ovr_fn) else float(getattr(player, "ovr", 0.5))
    except Exception:
        ovr = 0.5
    if ovr > 1.2:
        ovr = ovr / 99.0
    ident = getattr(player, "identity", None)
    age = int(ident.age) if ident is not None and hasattr(ident, "age") else int(getattr(player, "age", 26))
    trend = float(getattr(getattr(player, "context", None), "recent_performance_trend", 0.5) or 0.5)
    gp = int(getattr(player, "games_played", 0) or 0)
    usage = min(1.0, gp / 82.0) if gp else 0.45

    pos = getattr(player, "position", None)
    is_g = getattr(pos, "value", str(pos)) == "G"
    is_d = getattr(pos, "value", str(pos)) == "D"

    era = resolve_era_profile(str(league_state.get("active_era", "") or ""))
    off_era = era.get("offense_archetype_boost", 1.0)
    # Skew effective ovr for forwards in offensive eras
    adj = ovr
    if not is_g and not is_d:
        adj = _clamp(ovr + 0.02 * (off_era - 1.0), 0.0, 0.99)

    declining = age >= 32 and trend < 0.42
    rising = trend > 0.58 and age < 28

    if is_g:
        if adj >= 0.88 and not declining:
            base = "elite"
        elif adj >= 0.78:
            base = "starter"
        elif adj >= 0.62:
            base = "backup"
        elif age <= 22:
            base = "prospect"
        else:
            base = "depth"
    elif is_d:
        if adj >= 0.88 and not declining:
            base = "elite"
        elif adj >= 0.77:
            base = "top_4"
        elif adj >= 0.64:
            base = "middle_6"
        elif adj >= 0.52:
            base = "bottom_6"
        elif age <= 22:
            base = "prospect"
        else:
            base = "depth"
    else:
        if adj >= 0.88 and not declining:
            base = "elite"
        elif adj >= 0.77:
            base = "top_line"
        elif adj >= 0.64:
            base = "middle_6"
        elif adj >= 0.52:
            base = "bottom_6"
        elif age <= 22:
            base = "prospect"
        else:
            base = "depth"

    narrative = base
    if declining and base in ("elite", "top_line", "top_4", "starter"):
        narrative = "declining_star"
    elif rising and base in ("middle_6", "bottom_6", "depth", "prospect"):
        if usage > 0.55:
            narrative = "breakout_candidate"
        else:
            narrative = "pressed_prospect"

    # Usage suppresses role if barely playing
    if usage < 0.25 and age >= 20 and base not in ("prospect",):
        narrative = "depth"
        base = "depth"

    try:
        setattr(player, "role", base)
        setattr(player, "role_narrative", narrative)
        setattr(player, "_tuning_usage_factor", usage)
    except Exception:
        pass
    return str(base)
