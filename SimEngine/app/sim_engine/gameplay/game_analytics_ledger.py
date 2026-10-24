"""
Possession / expected-goals ledger helpers for event-driven game simulation.

Events create CF/CA/FF/FA, raw xG, iXG, GF-on/GA-on, and goalie xGA.
Season xGF% uses cumulative xGF / (xGF + xGA), not averaged per-game percentages.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

_FLOAT_KEYS = frozenset(
    {
        "cf",
        "ca",
        "ff",
        "fa",
        "xgf",
        "xga",
        "ixg",
        "xa",
        "gf_on",
        "ga_on",
        "xgf_pct_sum",
        "on_ice_shots_for",
        "on_ice_shots_against",
        "goalie_xga",
    }
)

# Single league-scoring calibration point. Structure, finishing and goalie quality
# supply the spread; this only sets the overall goal environment. (0.81 before block
# and miss rates depended on chance type; that sends richer shots to the net, so it
# was re-tuned to keep league goals per game where they were.)
GOAL_PROBABILITY_CALIBRATION: float = 0.743

# Abstract chance types: midpoint raw xG vs average shooter / average goalie.
CHANCE_TYPE_RAW_XG: Dict[str, float] = {
    "LOW_DANGER_PERIMETER": 0.035,
    "POINT_SHOT": 0.045,
    "RUSH_MEDIUM": 0.095,
    "SLOT": 0.125,
    "HIGH_DANGER_SLOT": 0.185,
    "NET_FRONT": 0.210,
    "REBOUND": 0.195,
    "ONE_TIMER": 0.155,
    "PP_SLOT": 0.165,
    "PP_ONE_TIMER": 0.195,
    "SH_RUSH": 0.110,
}

_CHANCE_EV_WEIGHTS: List[Tuple[str, float]] = [
    ("LOW_DANGER_PERIMETER", 0.28),
    ("POINT_SHOT", 0.18),
    ("RUSH_MEDIUM", 0.14),
    ("SLOT", 0.16),
    ("HIGH_DANGER_SLOT", 0.08),
    ("NET_FRONT", 0.06),
    ("REBOUND", 0.04),
    ("ONE_TIMER", 0.06),
]

_CHANCE_PP_WEIGHTS: List[Tuple[str, float]] = [
    ("PP_SLOT", 0.32),
    ("PP_ONE_TIMER", 0.22),
    ("POINT_SHOT", 0.16),
    ("SLOT", 0.14),
    ("NET_FRONT", 0.10),
    ("LOW_DANGER_PERIMETER", 0.06),
]

_CHANCE_SH_WEIGHTS: List[Tuple[str, float]] = [
    ("SH_RUSH", 0.38),
    ("RUSH_MEDIUM", 0.28),
    ("LOW_DANGER_PERIMETER", 0.18),
    ("SLOT", 0.16),
]

# Where a chance comes from decides how often it reaches the net. Point shots run
# into shin pads through three bodies; a rebound at the paint almost never gets
# blocked. One-timers and perimeter looks miss wide; a tap-in rarely does.
_CHANCE_BLOCK_SHAPE: Dict[str, float] = {
    "LOW_DANGER_PERIMETER": 1.10,
    "POINT_SHOT": 1.70,
    "RUSH_MEDIUM": 0.80,
    "SLOT": 0.90,
    "HIGH_DANGER_SLOT": 0.70,
    "NET_FRONT": 0.40,
    "REBOUND": 0.35,
    "ONE_TIMER": 1.00,
    "PP_SLOT": 0.85,
    "PP_ONE_TIMER": 1.05,
    "SH_RUSH": 0.65,
}

_CHANCE_ON_GOAL_SHAPE: Dict[str, float] = {
    "LOW_DANGER_PERIMETER": 0.92,
    "POINT_SHOT": 0.95,
    "RUSH_MEDIUM": 1.02,
    "SLOT": 1.03,
    "HIGH_DANGER_SLOT": 1.05,
    "NET_FRONT": 1.10,
    "REBOUND": 1.08,
    "ONE_TIMER": 0.93,
    "PP_SLOT": 1.04,
    "PP_ONE_TIMER": 0.95,
    "SH_RUSH": 1.00,
}

# League-average on-goal rate of an unblocked attempt by strength, for an average
# shooter (matches the base on-goal formula in the event loop at shot quality ~0.70).
UNBLOCKED_ON_GOAL_BASE: Dict[str, float] = {"EV": 0.676, "PP": 0.776, "SH": 0.616}

# Recorded xG = chance of a goal from an UNBLOCKED attempt for a league-average
# shooter vs a league-average goalie, so league goals ~= league xG at each strength.
# Tuned against simulated games; it only scales the stat, never the goal roll.
XG_RECORD_CALIBRATION: Dict[str, float] = {"EV": 0.84, "PP": 0.89, "SH": 0.88}


# The mix of chance types the event loop actually produces (league pool after shot
# quality bias and shooter diets), measured over simulated games. Per-type rates are
# centred on THIS mix, not the raw pool, which is far heavier on perimeter looks.
_REALISED_EV_MIX: List[Tuple[str, float]] = [
    ("LOW_DANGER_PERIMETER", 0.152), ("POINT_SHOT", 0.174), ("RUSH_MEDIUM", 0.156),
    ("SLOT", 0.171), ("HIGH_DANGER_SLOT", 0.117), ("NET_FRONT", 0.095),
    ("REBOUND", 0.046), ("ONE_TIMER", 0.090),
]
_REALISED_PP_MIX: List[Tuple[str, float]] = [
    ("PP_ONE_TIMER", 0.364), ("PP_SLOT", 0.256), ("NET_FRONT", 0.139),
    ("POINT_SHOT", 0.117), ("SLOT", 0.108), ("LOW_DANGER_PERIMETER", 0.016),
]
_REALISED_SH_MIX: List[Tuple[str, float]] = [
    ("SH_RUSH", 0.416), ("RUSH_MEDIUM", 0.321), ("SLOT", 0.180), ("LOW_DANGER_PERIMETER", 0.083),
]


def _pool_normalised(shape: Mapping[str, float], pool: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    # Mean-centre each strength's mix on 1.0 so per-type rates spread without moving
    # how many attempts get blocked or reach the net league-wide.
    tot = sum(w for _, w in pool) or 1.0
    mean = sum(w * float(shape.get(c, 1.0)) for c, w in pool) / tot or 1.0
    return {c: float(v) / mean for c, v in shape.items()}


_BLOCK_MULT_BY_STRENGTH: Dict[str, Dict[str, float]] = {
    "EV": _pool_normalised(_CHANCE_BLOCK_SHAPE, _REALISED_EV_MIX),
    "PP": _pool_normalised(_CHANCE_BLOCK_SHAPE, _REALISED_PP_MIX),
    "SH": _pool_normalised(_CHANCE_BLOCK_SHAPE, _REALISED_SH_MIX),
}
_ON_GOAL_MULT_BY_STRENGTH: Dict[str, Dict[str, float]] = {
    "EV": _pool_normalised(_CHANCE_ON_GOAL_SHAPE, _REALISED_EV_MIX),
    "PP": _pool_normalised(_CHANCE_ON_GOAL_SHAPE, _REALISED_PP_MIX),
    "SH": _pool_normalised(_CHANCE_ON_GOAL_SHAPE, _REALISED_SH_MIX),
}


def _strength_key(strength: str) -> str:
    st = str(strength or "EV").upper()
    return st if st in ("PP", "SH") else "EV"


def chance_block_mult(chance_type: str, strength: str) -> float:
    """How much more (or less) often this kind of attempt gets blocked."""
    return float(_BLOCK_MULT_BY_STRENGTH[_strength_key(strength)].get(str(chance_type), 1.0))


def chance_on_goal_mult(chance_type: str, strength: str) -> float:
    """How much more (or less) often an unblocked attempt of this kind hits the net."""
    return float(_ON_GOAL_MULT_BY_STRENGTH[_strength_key(strength)].get(str(chance_type), 1.0))


def unblocked_attempt_xg(raw_xg: float, chance_type: str, strength: str, structural_adj: float = 1.0) -> float:
    """
    Recorded xG for one unblocked attempt (Fenwick xG, as public models use).

    Folds in the chance type's league on-goal rate and the shot's structure
    (screens, traffic, lanes, rush edge, line chemistry), but not who shot it or
    who was in net, so finishing and goaltending show up as goals above xG.
    """
    sk = _strength_key(strength)
    on_goal = UNBLOCKED_ON_GOAL_BASE[sk] * chance_on_goal_mult(chance_type, strength)
    sa = max(0.45, min(2.20, float(structural_adj)))
    return max(0.002, float(raw_xg) * min(0.92, on_goal) * sa * XG_RECORD_CALIBRATION[sk])


def ledger_add_value(row: Dict[str, Any], key: str, value: float | int) -> None:
    if not value:
        return
    if key in _FLOAT_KEYS:
        row[key] = round(float(row.get(key, 0) or 0) + float(value), 4)
    else:
        row[key] = int(row.get(key, 0) or 0) + int(round(float(value)))


def season_xgf_pct_from_row(row: Mapping[str, Any]) -> float:
    """Canonical season on-ice xGF% = total xGF / (total xGF + total xGA)."""
    xgf = float(row.get("xgf") or 0.0)
    xga = float(row.get("xga") or 0.0)
    den = xgf + xga
    if den > 0:
        return xgf / den
    gp = int(row.get("xgf_pct_gp") or 0)
    if gp > 0:
        return float(row.get("xgf_pct_sum") or 0.0) / float(gp)
    return 0.5


def estimate_pp_opportunities(
    rng: random.Random,
    pp_goals: int,
    opponent_pim: int,
    *,
    explicit_ppo: Optional[int] = None,
) -> int:
    """Power-play opportunities reconciled with penalty minutes when available."""
    if explicit_ppo is not None:
        return int(max(int(pp_goals), explicit_ppo))
    base = rng.randint(2, 5)
    pim_bonus = int(max(0, opponent_pim) / 12)
    ppo = max(int(pp_goals), base + pim_bonus)
    return int(min(9, max(int(pp_goals), ppo)))


def pick_chance_type(
    rng: random.Random,
    strength: str,
    *,
    quality_bias: float = 0.5,
) -> str:
    """Pick abstract chance type from strength state and unit quality bias."""
    st = str(strength or "EV").upper()
    if st == "PP":
        pool = list(_CHANCE_PP_WEIGHTS)
    elif st == "SH":
        pool = list(_CHANCE_SH_WEIGHTS)
    else:
        pool = list(_CHANCE_EV_WEIGHTS)
    labels = [c for c, _ in pool]
    weights = [w * (1.0 + 0.35 * quality_bias if "HIGH" in c or "NET" in c or "ONE" in c else 1.0) for c, w in pool]
    return rng.choices(labels, weights=weights, k=1)[0]


def pick_chance_type_for_shooter(
    rng: random.Random,
    strength: str,
    affinity: Optional[Mapping[str, float]] = None,
    *,
    quality_bias: float = 0.5,
    team_mix: Optional[Mapping[str, float]] = None,
) -> str:
    """
    Pick a chance type conditioned on THIS shooter's shot diet.

    `affinity` maps chance type -> multiplier centred on 1.0, so a one-timer
    specialist genuinely takes one-timers and a net-front forward genuinely
    lives in the blue paint, instead of every shooter drawing the league pool.
    `team_mix` does the same for the systems on the ice: what the attacking club
    plays for and what the defending club concedes.
    """
    st = str(strength or "EV").upper()
    if st == "PP":
        pool = list(_CHANCE_PP_WEIGHTS)
    elif st == "SH":
        pool = list(_CHANCE_SH_WEIGHTS)
    else:
        pool = list(_CHANCE_EV_WEIGHTS)
    labels: List[str] = []
    weights: List[float] = []
    for c, w in pool:
        wt = float(w)
        if "HIGH" in c or "NET" in c or "ONE" in c:
            wt *= 1.0 + 0.35 * float(quality_bias)
        if affinity:
            wt *= max(0.05, float(affinity.get(c, 1.0)))
        if team_mix:
            wt *= max(0.20, float(team_mix.get(c, 1.0)))
        labels.append(c)
        weights.append(max(1e-6, wt))
    return rng.choices(labels, weights=weights, k=1)[0]


def raw_xg_for_chance(chance_type: str, rng: random.Random) -> float:
    """Pre-finisher, pre-goalie expected goal probability for an abstract chance."""
    base = float(CHANCE_TYPE_RAW_XG.get(str(chance_type), 0.065))
    jitter = rng.uniform(0.88, 1.12)
    return max(0.018, min(0.42, base * jitter))


def zero_assist_probability(chance_type: str, strength: str) -> float:
    """Probability of an unassisted goal for this chance type."""
    ct = str(chance_type or "")
    st = str(strength or "EV").upper()
    if ct in ("SH_RUSH", "RUSH_MEDIUM"):
        return 0.035 if st == "SH" else 0.030
    if ct in ("REBOUND",):
        return 0.018
    if ct in ("NET_FRONT",):
        return 0.020
    if st == "PP":
        return 0.012 if ct in ("PP_ONE_TIMER", "PP_SLOT") else 0.020
    if ct in ("POINT_SHOT",):
        return 0.020
    if ct in ("LOW_DANGER_PERIMETER",):
        return 0.028
    if ct in ("SLOT", "HIGH_DANGER_SLOT", "ONE_TIMER"):
        return 0.016
    return 0.022


def assist_count_probability(chance_type: str, strength: str) -> Tuple[float, float, float]:
    """Return (p0, p1, p2) assist counts excluding scorer."""
    p0 = zero_assist_probability(chance_type, strength)
    ct = str(chance_type or "")
    st = str(strength or "EV").upper()
    # Two-assist share. Raised so more goals carry a secondary assist
    # (about 86% on the light path). A set power play still chains more passes than a rush.
    if st == "PP":
        p2 = 0.82 if ct in ("PP_ONE_TIMER", "PP_SLOT") else 0.76
    elif ct in ("RUSH_MEDIUM", "SH_RUSH"):
        p2 = 0.54 if st == "SH" else 0.60
    elif ct in ("REBOUND", "NET_FRONT"):
        p2 = 0.64
    elif ct in ("POINT_SHOT", "SLOT", "HIGH_DANGER_SLOT", "ONE_TIMER"):
        p2 = 0.70
    else:
        p2 = 0.66
    p2 = max(0.0, min(0.90, p2))
    p1 = max(0.0, 1.0 - p0 - p2)
    s = p0 + p1 + p2
    if s <= 0:
        return 0.12, 0.44, 0.44
    return p0 / s, p1 / s, p2 / s


def league_assist_health_metrics(goal_events: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """
    Permanent tuning health: assists-per-goal and 0/1/2-assist goal shares.
    Each goal event should expose assist_count (0, 1, or 2).
    """
    total_goals = 0
    total_assists = 0
    n0 = n1 = n2 = 0
    for ev in goal_events or []:
        if not isinstance(ev, Mapping):
            continue
        ac = ev.get("assist_count")
        if ac is None:
            alist = ev.get("assists") or []
            ac = len(alist) if isinstance(alist, list) else 0
        try:
            ac_i = max(0, min(2, int(ac)))
        except (TypeError, ValueError):
            ac_i = 0
        total_goals += 1
        total_assists += ac_i
        if ac_i <= 0:
            n0 += 1
        elif ac_i == 1:
            n1 += 1
        else:
            n2 += 1
    if total_goals <= 0:
        return {
            "player_assists_per_goal": 0.0,
            "unassisted_goal_pct": 0.0,
            "one_assist_goal_pct": 0.0,
            "two_assist_goal_pct": 0.0,
            "total_goals": 0,
        }
    g = float(total_goals)
    return {
        "player_assists_per_goal": round(total_assists / g, 4),
        "unassisted_goal_pct": round(n0 / g, 4),
        "one_assist_goal_pct": round(n1 / g, 4),
        "two_assist_goal_pct": round(n2 / g, 4),
        "total_goals": int(total_goals),
    }


def resolve_goal_probability(
    raw_xg: float,
    finishing_adj: float,
    goalie_adj: float,
    *,
    situational_adj: float = 1.0,
) -> float:
    """Final goal probability after shooter finishing, goalie quality and on-ice structure.

    `situational_adj` carries the structural read of the moment — screens, net-front
    traffic, lane-blocking defenders, goalie sightline and rush vs. set play. It is
    built mean-centred on 1.0, so it supplies spread without shifting league scoring.
    """
    prob = float(raw_xg) * GOAL_PROBABILITY_CALIBRATION * max(0.55, min(1.45, float(finishing_adj)))
    prob *= max(0.55, min(1.45, float(goalie_adj)))
    # Wide enough for a genuine screened point shot or a fully-boxed-out net front.
    prob *= max(0.45, min(2.20, float(situational_adj)))
    return max(0.010, min(0.72, prob))


def credit_shot_attempt_event(
    ledger: Dict[str, Dict[str, Any]],
    *,
    attacking_skaters: Sequence[Any],
    defending_skaters: Sequence[Any],
    shooter: Any,
    defending_goalie: Any,
    team_id: str,
    opp_team_id: str,
    raw_xg: float,
    outcome: str,
    blocker: Optional[Any],
    ledger_add: Callable[..., None],
    player_id: Callable[[Any], str],
    strength: str = "EV",
) -> None:
    """
    Record one abstract shot attempt and its statistical consequences.

    outcome: BLOCKED | MISSED | SAVED | GOAL
    raw_xg: the attempt's recorded (unblocked-attempt) xG. Misses carry it too,
    as in public Fenwick-based models; goalie xGA stays on shots on goal.
    """
    rxg = round(float(raw_xg), 4)

    # On-ice credits are batched into one ledger_add per skater (each key is summed
    # independently, so this matches crediting them one call at a time).
    if outcome == "BLOCKED":
        att_kw: Dict[str, Any] = {"cf": 1.0}
        def_kw: Dict[str, Any] = {"ca": 1.0}
    elif outcome == "MISSED":
        att_kw = {"cf": 1.0, "ff": 1.0, "xgf": rxg}
        def_kw = {"ca": 1.0, "fa": 1.0, "xga": rxg}
    else:
        att_kw = {"cf": 1.0, "ff": 1.0, "on_ice_shots_for": 1.0, "xgf": rxg}
        def_kw = {"ca": 1.0, "fa": 1.0, "on_ice_shots_against": 1.0, "xga": rxg}
    for p in attacking_skaters:
        ledger_add(ledger, p, team_id, **att_kw)
    for p in defending_skaters:
        ledger_add(ledger, p, opp_team_id, **def_kw)

    if outcome == "BLOCKED":
        ledger_add(ledger, shooter, team_id, blocked_attempts_for=1)
        if blocker is not None:
            ledger_add(ledger, blocker, opp_team_id, blk=1)
        return

    if outcome == "MISSED":
        ledger_add(ledger, shooter, team_id, missed_shots=1, ixg=rxg)
        return

    ledger_add(ledger, shooter, team_id, sog=1, ixg=rxg)
    if str(strength or "").upper() == "PP":
        ledger_add(ledger, shooter, team_id, pp_sog=1)
    if defending_goalie is not None:
        goalie_kw: Dict[str, Any] = {"goalie_xga": rxg, "goalie_shots_against": 1}
        if outcome == "GOAL":
            goalie_kw["goalie_ga"] = 1
        ledger_add(ledger, defending_goalie, opp_team_id, **goalie_kw)

    if outcome == "GOAL":
        goal_kw: Dict[str, Any] = {"g": 1, "gf_on": 1.0}
        if str(strength).upper() == "PP":
            goal_kw["ppg"] = 1
        elif str(strength).upper() == "SH":
            goal_kw["shg"] = 1
        ledger_add(ledger, shooter, team_id, **goal_kw)
        for p in attacking_skaters:
            if p is not shooter:
                ledger_add(ledger, p, team_id, gf_on=1.0)
        for p in defending_skaters:
            ledger_add(ledger, p, opp_team_id, ga_on=1.0)


def credit_assist_xa(
    ledger: Dict[str, Dict[str, Any]],
    player: Any,
    team_id: str,
    *,
    ledger_add: Callable[..., None],
    primary_assist_weight: Callable[[Any], float],
    xa_value: Optional[float] = None,
) -> None:
    w = max(0.04, float(primary_assist_weight(player)))
    xa = float(xa_value) if xa_value is not None else max(0.06, min(0.28, 0.10 + 0.06 * w))
    ledger_add(ledger, player, team_id, xa=xa)


def validate_game_integrity(
    home_agg: Mapping[str, Any],
    away_agg: Mapping[str, Any],
) -> List[str]:
    """Return list of integrity violation messages (empty if valid)."""
    issues: List[str] = []
    for label, agg in (("home", home_agg), ("away", away_agg)):
        g = int(agg.get("goals", 0) or 0)
        sog = int(agg.get("sog", 0) or 0)
        ff = int(agg.get("ff", 0) or 0)
        cf = int(agg.get("cf", 0) or 0)
        if g > sog:
            issues.append(f"{label}: goals ({g}) > SOG ({sog})")
        if sog > ff:
            issues.append(f"{label}: SOG ({sog}) > FF ({ff})")
        if ff > cf:
            issues.append(f"{label}: FF ({ff}) > CF ({cf})")
    int(home_agg.get("goals", 0) or 0)
    int(away_agg.get("goals", 0) or 0)
    h_sog = int(home_agg.get("sog", 0) or 0)
    a_sog = int(away_agg.get("sog", 0) or 0)
    h_sa = int(home_agg.get("goalie_sa", 0) or 0)
    a_sa = int(away_agg.get("goalie_sa", 0) or 0)
    if h_sa != a_sog:
        issues.append(f"home SA ({h_sa}) != away SOG ({a_sog})")
    if a_sa != h_sog:
        issues.append(f"away SA ({h_sa}) != home SOG ({h_sog})")
    return issues
