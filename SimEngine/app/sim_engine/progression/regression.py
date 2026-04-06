# app/sim_engine/progression/regression.py
"""
Decline from injury history, morale, and wear. Age-based OVR loss is gated through
career_aging_decline_try (cooldown, brackets, archetypes) from the career lifecycle.
"""

from typing import Any, Dict, List, Optional, Tuple

import random

from app.sim_engine.progression.development import career_phase_for_age


def _clamp_rating(x: float, lo: int = 20, hi: int = 99) -> int:
    return int(max(lo, min(hi, round(x))))


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _age(player: Any) -> int:
    identity = getattr(player, "identity", None)
    if identity is not None and hasattr(identity, "age"):
        return int(identity.age)
    return int(getattr(player, "age", 27))


def _peak_age(player: Any) -> int:
    career = getattr(player, "career", None)
    if career is not None and hasattr(career, "expected_peak_age"):
        return int(career.expected_peak_age)
    return int(getattr(player, "peak_age", 27))


def _morale(player: Any) -> float:
    psych = getattr(player, "psych", None)
    if psych is not None and hasattr(psych, "morale"):
        return float(psych.morale)
    return float(getattr(player, "morale", 0.5))


def _injury_history(player: Any) -> List[Dict]:
    health = getattr(player, "health", None)
    if health is not None and hasattr(health, "injury_history"):
        h = getattr(health, "injury_history", [])
        if isinstance(h, list):
            return h
    return getattr(player, "injury_history", []) or []


def _wear_and_tear(player: Any) -> float:
    health = getattr(player, "health", None)
    if health is not None and hasattr(health, "wear_and_tear"):
        return float(health.wear_and_tear)
    return 0.0


def _ovr(player: Any) -> float:
    ovr_fn = getattr(player, "ovr", None)
    if callable(ovr_fn):
        try:
            return float(ovr_fn())
        except Exception:
            pass
    return float(getattr(player, "ovr", 0.5))


def _ovr_0_100(player: Any) -> float:
    v = _ovr(player)
    if v <= 1.0:
        v = v * 100.0
    return max(1.0, min(99.0, v))


def _is_goalie(player: Any) -> bool:
    pos = getattr(player, "position", None)
    if pos is None:
        return False
    return getattr(pos, "value", str(pos)) == "G"


def _usage_stress_from_role(player: Any) -> float:
    r = str(getattr(player, "role", "") or "").lower()
    if r in ("elite", "top_line", "top_4", "starter"):
        return 1.0
    if r in ("middle_6",):
        return 0.62
    if r in ("bottom_6",):
        return 0.48
    if r in ("backup", "depth", "prospect"):
        return 0.32
    return 0.55


def _update_hidden_decline_wear(player: Any, rng: Any) -> float:
    age = _age(player)
    usage = _usage_stress_from_role(player)
    inj_hist = _injury_history(player)
    w_tear = _wear_and_tear(player)
    health = getattr(player, "health", None)
    days_inj = 0
    if health is not None:
        try:
            days_inj = int(getattr(health, "days_injured_career", 0) or 0)
        except (TypeError, ValueError):
            days_inj = 0

    prev = float(getattr(player, "_aging_decline_wear", 0.0) or 0.0)
    prev = _clamp01(prev)
    age_u = max(0.0, float(age - 24) / 14.0)
    inj_w = min(1.0, 0.035 * float(len(inj_hist)) + 0.28 * _clamp01(w_tear))
    inj_days_u = min(1.0, float(days_inj) / 750.0)
    delta = (
        0.018 * age_u
        + 0.028 * usage
        + 0.42 * inj_w
        + 0.18 * inj_days_u
        + float(rng.uniform(0.0, 0.012))
    )
    nw = prev * 0.92 + delta
    nw = _clamp01(nw)
    setattr(player, "_aging_decline_wear", nw)
    return nw


def _archetype_modifiers(player: Any, age: int) -> Tuple[float, float]:
    prob_m = 1.0
    sev_m = 1.0
    sty = str(getattr(player, "playstyle", "") or "").lower()
    g = _is_goalie(player)
    if g:
        if age < 34:
            prob_m *= 0.78
            sev_m *= 0.85
        else:
            prob_m *= 0.92
            sev_m *= 0.95
        return prob_m, sev_m
    if sty in ("sniper", "offensive_d") or "offensive" in sty:
        if age >= 31:
            prob_m *= 1.14
            sev_m *= 1.08
    elif sty == "playmaker":
        prob_m *= 0.88
        sev_m *= 0.90
    elif sty in ("grinder", "enforcer", "enforcer_d") or "power" in sty:
        if age >= 28:
            prob_m *= 1.10
            sev_m *= 1.05
    elif sty in ("defensive_d", "defensive", "two_way_d", "two_way"):
        prob_m *= 0.86
        sev_m *= 0.88
    return prob_m, sev_m


def _is_elite_veteran(player: Any) -> bool:
    age = _age(player)
    ovr = _ovr(player)
    role = str(getattr(player, "role", "") or "").lower()
    return age >= 30 and ovr >= 0.82 and ("elite" in role or "top_line" in role or "top_4" in role)


def apply_regression(player: Any, rng: Any) -> None:
    """
    Small injury/morale-driven regression after peak age. Age-based OVR cliff is handled
    by career_aging_decline_try in the career lifecycle (logged as AGING DECLINE).
    """
    age = _age(player)
    peak = _peak_age(player)
    if age <= peak:
        return

    injury_hist = _injury_history(player)
    injury_penalty = min(0.022, 0.004 * len(injury_hist)) + 0.008 * _wear_and_tear(player)
    decline = float(injury_penalty)

    morale = _morale(player)
    if morale < 0.4:
        decline += 0.012
    elif morale < 0.55:
        decline += 0.006

    if _is_elite_veteran(player):
        decline *= 0.65

    decline *= float(getattr(player, "_narrative_regression_rate_mult", 1.0) or 1.0)

    if decline <= 0.0:
        return

    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict) or not ratings:
        return

    per_k = decline * 99.0 / len(ratings)
    for k in list(ratings.keys()):
        ratings[k] = _clamp_rating(ratings[k] - per_k)


def career_aging_decline_try(player: Any, rng: Any, league: Any = None) -> Optional[float]:
    """
    Probabilistic aging decline for career lifecycle. Returns OVR points lost (positive
    magnitude) for the caller to apply; does not mutate ratings (engine applies delta).

    Smooth age-tier curve (logged by engine as AGING DECLINE), optional rare late spike,
    elite protection, yearly hard caps, and anti-chain damping after a prior decline.
    """
    if getattr(player, "retired", False):
        return None

    if not isinstance(rng, random.Random):
        rng = random.Random()

    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict) or not ratings:
        return None

    wear = _update_hidden_decline_wear(player, rng)

    cd = int(getattr(player, "decline_cooldown", 0) or 0)
    if cd > 0:
        setattr(player, "decline_cooldown", max(0, cd - 1))
        setattr(
            player,
            "_aging_decline_last_reason",
            {"triggered": False, "blocked": "cooldown", "wear": round(wear, 3)},
        )
        return None

    age = _age(player)
    ovr100 = _ovr_0_100(player)
    phase = str(getattr(player, "career_phase", "") or career_phase_for_age(age))

    if age <= 24:
        setattr(
            player,
            "_aging_decline_last_reason",
            {"triggered": False, "age": age, "career_phase": phase, "blocked": "young"},
        )
        return None

    if age <= 29:
        p = 0.05
        amt_lo, amt_hi = 0.5, 1.0
    elif age <= 33:
        p = 0.20
        amt_lo, amt_hi = 0.5, 1.5
    elif age <= 36:
        p = 0.35
        amt_lo, amt_hi = 1.0, 2.5
    else:
        p = 0.55
        amt_lo, amt_hi = 1.5, 3.5

    chance_mult = 1.0
    amt_mult = 1.0
    if ovr100 >= 90.0:
        chance_mult *= 0.6
        amt_mult *= 0.7
    elif ovr100 >= 88.0:
        chance_mult *= 0.8
        amt_mult *= 0.7

    p = float(p) * chance_mult

    career = getattr(player, "career", None)
    rr = 0.5
    if career is not None and hasattr(career, "regression_resistance"):
        rr = float(getattr(career, "regression_resistance", 0.5) or 0.5)
    rr = _clamp01(rr)
    p *= 0.65 + 0.35 * (1.0 - rr)

    p *= 0.65
    if age <= 29:
        p *= 0.5

    if league is not None:
        try:
            n_decl = int(getattr(league, "_aging_season_decline_count", 0) or 0)
            n_tot = int(getattr(league, "_aging_season_player_total", 0) or 0)
            if n_tot > 0 and n_decl > n_tot * 0.22:
                p *= 0.5
        except (TypeError, ValueError):
            pass

    p *= float(getattr(player, "_narrative_decline_p_mult", 1.0) or 1.0)
    p = max(0.0, min(0.92, p))

    rare_spike = False
    if age >= 35 and rng.random() < float(rng.uniform(0.005, 0.01)):
        rare_spike = True
        mag = float(rng.uniform(3.5, 5.0))
    elif rng.random() < p:
        mag = float(rng.uniform(amt_lo, amt_hi))
    else:
        setattr(
            player,
            "_aging_decline_last_reason",
            {
                "triggered": False,
                "roll_p": round(p, 4),
                "wear": round(wear, 3),
                "age": age,
                "career_phase": phase,
            },
        )
        setattr(player, "_aging_decline_chain_active", False)
        return None
    _, asev = _archetype_modifiers(player, age)
    mag *= amt_mult * asev

    if bool(getattr(player, "_aging_decline_chain_active", False)):
        mag *= 0.6

    cap = 1.8 if age < 34 else 3.2
    mag = min(float(mag), cap)
    mag = max(0.25, float(mag))

    setattr(player, "decline_cooldown", int(rng.randint(1, 2)))
    setattr(player, "_aging_decline_chain_active", True)
    reason: Dict[str, Any] = {
        "triggered": True,
        "age_curve": age,
        "career_phase": phase,
        "rare_spike": rare_spike,
        "wear_factor": round(wear, 3),
        "archetype_effect": round(asev, 3),
        "roll_p": round(p, 4),
        "magnitude": round(mag, 3),
    }
    setattr(player, "_aging_decline_last_reason", reason)
    return mag
