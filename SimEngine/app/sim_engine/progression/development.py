# app/sim_engine/progression/development.py
"""
Young player growth based on potential, morale, games played, development rate, team role.
Under-24 get most growth; low ice time = slower development.

Career arc: phase from age (prospect → declining), yearly trend (hot / stable / declining).
"""

from typing import Any, List, Optional

import random

# --- Career arc phases (age bands; assigned each year on player.career_phase) ---
PHASE_PROSPECT = "prospect"
PHASE_EMERGING = "emerging"
PHASE_PRIME = "prime"
PHASE_VETERAN = "veteran"
PHASE_DECLINING = "declining"


def career_phase_for_age(age: int) -> str:
    if age <= 22:
        return PHASE_PROSPECT
    if age <= 25:
        return PHASE_EMERGING
    if age <= 30:
        return PHASE_PRIME
    if age <= 35:
        return PHASE_VETERAN
    return PHASE_DECLINING


def assign_career_phase_from_age(player: Any) -> str:
    ph = career_phase_for_age(_age(player))
    setattr(player, "career_phase", ph)
    if getattr(player, "trend", None) is None:
        setattr(player, "trend", "stable")
    if getattr(player, "_trend_remaining", None) is None:
        setattr(player, "_trend_remaining", 0)
    return ph


def tick_career_trend(player: Any) -> None:
    tr = int(getattr(player, "_trend_remaining", 0) or 0)
    if tr > 0:
        tr -= 1
        setattr(player, "_trend_remaining", tr)
    if tr <= 0:
        setattr(player, "trend", "stable")


def set_player_trend(player: Any, trend: str, seasons: int, rng: Any) -> None:
    setattr(player, "trend", str(trend).lower())
    if not isinstance(rng, random.Random):
        rng = random.Random()
    s = int(seasons) if seasons else int(rng.randint(1, 2))
    s = max(1, min(3, s))
    setattr(player, "_trend_remaining", s)


def career_arc_development_multiplier(phase: str) -> float:
    p = str(phase or "").lower()
    if p == PHASE_PROSPECT:
        return 1.22
    if p == PHASE_EMERGING:
        return 1.08
    if p == PHASE_PRIME:
        return 0.52
    if p == PHASE_VETERAN:
        return 0.32
    if p == PHASE_DECLINING:
        return 0.18
    return 1.0


def career_arc_decline_probability_multiplier(phase: str) -> float:
    p = str(phase or "").lower()
    if p == PHASE_EMERGING:
        return 0.2
    if p == PHASE_PRIME:
        return 0.45
    if p == PHASE_VETERAN:
        return 1.05
    if p == PHASE_DECLINING:
        return 1.42
    return 1.0

# Use same clamp as player entity if available; otherwise local
def _clamp_rating(x: float, lo: int = 20, hi: int = 99) -> int:
    return int(max(lo, min(hi, round(x))))


def _ovr(player: Any) -> float:
    ovr_fn = getattr(player, "ovr", None)
    if callable(ovr_fn):
        try:
            return float(ovr_fn())
        except Exception:
            pass
    return getattr(player, "ovr", 0.5)


def _age(player: Any) -> int:
    identity = getattr(player, "identity", None)
    if identity is not None and hasattr(identity, "age"):
        return int(identity.age)
    return int(getattr(player, "age", 26))


def _morale(player: Any) -> float:
    psych = getattr(player, "psych", None)
    if psych is not None and hasattr(psych, "morale"):
        return float(psych.morale)
    return float(getattr(player, "morale", 0.5))


def _potential(player: Any) -> float:
    p = getattr(player, "potential", None)
    if p is not None:
        return float(p)
    return _ovr(player) * 1.05


def _development_rate(player: Any) -> float:
    rate = getattr(player, "development_rate", None)
    if rate is not None:
        return float(rate)
    career = getattr(player, "career", None)
    if career is not None and hasattr(career, "breakout_probability"):
        return 0.4 + 0.3 * float(career.breakout_probability)
    return 0.5


def _games_played(player: Any) -> int:
    return int(getattr(player, "games_played", 0))


def _infer_team_dev_window(team: Any) -> str:
    blob = " ".join(
        str(getattr(team, a, "") or "") for a in ("window", "gm_window", "strategy", "status", "archetype")
    ).lower()
    if any(x in blob for x in ("rebuild", "tank", "lottery")):
        return "rebuild"
    if any(x in blob for x in ("contend", "win_now", "powerhouse", "championship")):
        return "contender"
    return "neutral"


def prime_development_environment_for_rosters(teams: Optional[List[Any]], rng: Any) -> None:
    """
    Per-season: tag each roster player with team development environment multipliers.
    Called from run_sim before progression (same year as narrative apply).
    """
    if not teams or not isinstance(teams, list):
        return
    for team in teams:
        tid = str(getattr(team, "team_id", getattr(team, "id", "")) or "")
        if not tid:
            continue
        window = _infer_team_dev_window(team)
        pscore = float(getattr(team, "prospect_pipeline_score", 0.5) or 0.5)
        pscore = max(0.0, min(1.0, pscore))
        g_mult = 1.0
        v_mult = 1.0
        if window == "rebuild":
            g_mult *= 1.075
            v_mult *= 1.055
        elif window == "contender":
            g_mult *= 0.925
            v_mult *= 1.095
        g_mult *= 0.76 + 0.48 * pscore
        g_mult = max(0.72, min(1.36, g_mult))
        v_mult = max(0.78, min(1.34, v_mult))
        roster = getattr(team, "roster", None) or []
        for p in roster:
            if getattr(p, "retired", False):
                continue
            ctx = getattr(p, "context", None)
            cid = str(getattr(ctx, "current_team_id", "") or "") if ctx is not None else ""
            if cid != tid:
                continue
            setattr(p, "_dev_env_team_window", window)
            setattr(p, "_dev_env_growth_mult", g_mult)
            setattr(p, "_dev_env_variance_mult", v_mult)


def _diminishing_stack(*factors: float) -> float:
    out = 1.0
    for x in factors:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            xf = 1.0
        out *= max(0.52, min(1.52, xf))
    if out > 1.31:
        excess = out - 1.31
        out = 1.31 + (excess**0.78)
    return max(0.58, min(1.45, out))


def _dev_archetype_phase_roll(archetype: str, age: int, curve_hint: str, rng: random.Random) -> str:
    arch = (str(archetype or "").upper() or "SAFE_LOW_CEILING").strip()
    if not arch:
        arch = "SAFE_LOW_CEILING"
    ch = str(curve_hint or "").lower()
    stall = spike = reg = 0.09
    if arch == "FAST_RISER":
        stall, spike, reg = 0.085, 0.095, 0.028
        if age >= 21:
            stall, spike = 0.145, 0.055
    elif arch == "LATE_BLOOMER":
        stall, spike, reg = 0.155, 0.05, 0.038
        if 20 <= age <= 24:
            stall, spike = 0.105, 0.135
    elif arch == "HIGH_VARIANCE":
        stall, spike, reg = 0.115, 0.155, 0.075
    elif arch == "SAFE_LOW_CEILING":
        stall, spike, reg = 0.098, 0.048, 0.024
    elif arch == "ELITE_CEILING_VOLATILE":
        stall, spike, reg = 0.105, 0.125, 0.085
    elif arch == "STALLED_DEVELOPER":
        stall, spike, reg = 0.215, 0.035, 0.045
    else:
        stall, spike, reg = 0.11, 0.055, 0.03
    if ch == "slow":
        stall += 0.045
        spike = max(0.02, spike - 0.02)
    if ch == "boom_bust":
        spike += 0.045
        reg += 0.035
    stall = max(0.05, min(0.36, stall))
    spike = max(0.025, min(0.30, spike))
    reg = max(0.015, min(0.20, reg))
    u = rng.random()
    b1, b2, b3 = stall, stall + spike, stall + spike + reg
    if u < b1:
        return "STALL"
    if u < b2:
        return "SPIKE"
    if u < b3:
        return "REGRESSION"
    return "NORMAL"


def _lazy_assign_dev_archetype(player: Any, potential: float, rng: random.Random) -> str:
    """One-time spread when roster players never got engine pipeline archetypes (avoids all SAFE_LOW_CEILING)."""
    p = float(potential)
    if p >= 0.84:
        opts = [
            ("ELITE_CEILING_VOLATILE", 0.24),
            ("HIGH_VARIANCE", 0.18),
            ("FAST_RISER", 0.20),
            ("LATE_BLOOMER", 0.14),
            ("STALLED_DEVELOPER", 0.16),
            ("SAFE_LOW_CEILING", 0.08),
        ]
    elif p >= 0.72:
        opts = [
            ("HIGH_VARIANCE", 0.18),
            ("FAST_RISER", 0.17),
            ("LATE_BLOOMER", 0.17),
            ("STALLED_DEVELOPER", 0.15),
            ("ELITE_CEILING_VOLATILE", 0.14),
            ("SAFE_LOW_CEILING", 0.10),
        ]
    elif p >= 0.58:
        opts = [
            ("LATE_BLOOMER", 0.20),
            ("HIGH_VARIANCE", 0.18),
            ("FAST_RISER", 0.15),
            ("STALLED_DEVELOPER", 0.16),
            ("SAFE_LOW_CEILING", 0.12),
            ("ELITE_CEILING_VOLATILE", 0.10),
        ]
    else:
        opts = [
            ("LATE_BLOOMER", 0.22),
            ("HIGH_VARIANCE", 0.18),
            ("STALLED_DEVELOPER", 0.18),
            ("SAFE_LOW_CEILING", 0.14),
            ("FAST_RISER", 0.12),
            ("ELITE_CEILING_VOLATILE", 0.09),
        ]
    names = [n for n, _ in opts]
    weights = [w for _, w in opts]
    s = sum(weights) or 1.0
    wn = [x / s for x in weights]
    return str(rng.choices(names, weights=wn, k=1)[0])


def _ice_time_modifier(player: Any) -> float:
    """Higher role / more games = more ice time effect. 0.5 = baseline."""
    gp = _games_played(player)
    role = getattr(player, "role", None) or ""
    role_low = str(role).lower()
    # Top roles / high GP = better development
    if "elite" in role_low or "top_line" in role_low or "top_4" in role_low:
        base = 1.2
    elif "middle" in role_low:
        base = 1.0
    else:
        base = 0.85
    if gp >= 70:
        return base * 1.0
    if gp >= 50:
        return base * 0.9
    if gp >= 30:
        return base * 0.75
    return base * 0.6  # low ice time


def apply_player_development(player: Any, rng: Any) -> None:
    """
    Apply young-player growth based on potential, morale, games played,
    development rate, team role, dev archetype, environment, narrative, and NHL transition.
    Mutates player ratings in-place when present. All randomness via rng.
    """
    if not isinstance(rng, random.Random):
        rng = random.Random()

    age = _age(player)
    # Only meaningful growth for young players; late bloomers 24-26 get small chance
    if age > 26:
        if 24 <= age <= 26 and rng.random() > 0.15:
            return  # 15% late-bloomer development chance
        elif age > 26:
            return

    phase = str(getattr(player, "career_phase", "") or career_phase_for_age(age))

    ovr = _ovr(player)
    potential = _potential(player)
    morale = _morale(player)
    dev_rate = _development_rate(player)
    ice_mod = _ice_time_modifier(player)

    # growth = potential * 0.15 * morale_mod * ice_mod * development_rate
    growth_base = min(0.15, max(0.0, (potential - ovr) * 0.5)) + 0.02
    growth_base *= (0.7 + 0.6 * morale)  # morale modifier
    growth_base *= ice_mod
    growth_base *= (0.6 + 0.8 * dev_rate)

    if age < 21:
        growth_base *= 1.4
    elif age < 24:
        growth_base *= 1.22
    else:
        growth_base *= 0.68

    growth_base *= career_arc_development_multiplier(phase)
    # Dampen vs engine lifecycle + major events (avoid secret second breakout path).
    growth_base *= 0.52

    archetype = str(getattr(player, "_dev_archetype", "") or "").strip()
    if not archetype:
        archetype = _lazy_assign_dev_archetype(player, potential, rng)
        setattr(player, "_dev_archetype", archetype)
    curve_hint = str(
        getattr(player, "_pipeline_dev_curve", "") or getattr(player, "_dev_curve_hint", "") or "normal"
    ).lower()
    dev_phase = _dev_archetype_phase_roll(archetype, age, curve_hint, rng)
    adj = int(getattr(player, "_nhl_adjustment_years_remaining", 0) or 0)
    if adj > 0:
        u = rng.random()
        if u < 0.16:
            dev_phase = "STALL"
        elif u < 0.29:
            dev_phase = "SPIKE" if rng.random() < 0.52 else "REGRESSION"
        elif u < 0.40:
            dev_phase = "REGRESSION"

    env_g = float(getattr(player, "_dev_env_growth_mult", 1.0) or 1.0)
    env_v = float(getattr(player, "_dev_env_variance_mult", 1.0) or 1.0)
    nar = float(getattr(player, "_narrative_prog_growth_mult", 1.0) or 1.0)
    narr_cons = float(getattr(player, "_narrative_consistency_shift", 0.0) or 0.0)
    growth_base *= _diminishing_stack(env_g, nar)

    role = str(getattr(player, "role", "") or "").lower()
    role_env = 1.0
    if any(k in role for k in ("elite", "top_line", "top_4")):
        role_env = 1.09
    elif any(k in role for k in ("fourth", "depth", "scratch", "press")):
        role_env = 0.88
    growth_base *= role_env
    if narr_cons < -0.04:
        growth_base *= max(0.82, 1.0 + 0.38 * narr_cons)

    if dev_phase == "STALL":
        growth_base *= rng.uniform(0.03, 0.19)
    elif dev_phase == "SPIKE":
        growth_base *= rng.uniform(1.92, 3.22)
        if age <= 22 and archetype in ("HIGH_VARIANCE", "ELITE_CEILING_VOLATILE"):
            growth_base *= rng.uniform(1.03, 1.11)
    elif dev_phase == "REGRESSION":
        growth_base *= rng.uniform(-1.22, -0.32)
    else:
        growth_base *= 0.91 + 0.17 * min(1.15, env_v)

    if adj > 0:
        growth_base *= 0.79 + 0.07 * min(2, adj)

    bp = float(getattr(player, "_bust_pressure", 0.08) or 0.08)
    sm = float(getattr(player, "_steal_momentum", 0.06) or 0.06)
    if dev_phase == "REGRESSION":
        bp += rng.uniform(0.045, 0.11)
    elif dev_phase == "SPIKE":
        sm += rng.uniform(0.055, 0.13)
    elif dev_phase == "STALL" and age < 24:
        bp += rng.uniform(0.025, 0.07)
    setattr(player, "_bust_pressure", max(0.0, min(0.96, bp)))
    setattr(player, "_steal_momentum", max(0.0, min(0.96, sm)))

    pname = getattr(player, "name", None) or getattr(getattr(player, "identity", None), "name", None) or "Player"
    tw = str(getattr(player, "_dev_env_team_window", "") or "")
    if dev_phase != "NORMAL" or rng.random() < 0.048:
        setattr(
            player,
            "_dev_report_pending_line",
            (
                f"PROSPECT DEVELOPMENT REPORT: {pname} archetype={archetype} growth_phase={dev_phase} "
                f"age={age} env_window={tw or 'n/a'} env_growth_x={env_g:.2f} ice_role_x={role_env:.2f}"
            ),
        )
    if float(getattr(player, "_bust_pressure", 0) or 0) >= 0.5 and rng.random() < 0.22:
        setattr(
            player,
            "_bust_steal_pending_line",
            (
                f"BUST/STEAL TRACKING: {pname} trending_bust_pressure="
                f"{float(getattr(player, '_bust_pressure', 0) or 0):.2f} steal_momentum="
                f"{float(getattr(player, '_steal_momentum', 0) or 0):.2f}"
            ),
        )
    elif float(getattr(player, "_steal_momentum", 0) or 0) >= 0.55 and rng.random() < 0.2:
        setattr(
            player,
            "_bust_steal_pending_line",
            (
                f"BUST/STEAL TRACKING: {pname} emerging_steal_signal momentum="
                f"{float(getattr(player, '_steal_momentum', 0) or 0):.2f}"
            ),
        )

    setattr(player, "_dev_last_phase", dev_phase)

    if adj > 0:
        setattr(player, "_nhl_adjustment_years_remaining", max(0, adj - 1))

    # Cap so we don't push OVR above ceiling
    ceiling = min(0.92, potential + 0.02)
    if ovr >= ceiling and dev_phase != "REGRESSION":
        return

    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict) or not ratings:
        return

    # Convert growth to per-rating delta (in 0-99 scale)
    per_k = growth_base * 99.0 / len(ratings)
    if dev_phase == "REGRESSION":
        per_k = max(-0.36, min(-0.045, per_k))
    else:
        per_k = min(float(per_k), 0.40)
    for k in list(ratings.keys()):
        ratings[k] = _clamp_rating(ratings[k] + per_k)
