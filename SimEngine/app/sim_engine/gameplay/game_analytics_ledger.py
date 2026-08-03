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
        return 0.070 if st == "SH" else 0.060
    if ct in ("REBOUND",):
        return 0.035
    if ct in ("NET_FRONT",):
        return 0.038
    if st == "PP":
        return 0.025 if ct in ("PP_ONE_TIMER", "PP_SLOT") else 0.038
    if ct in ("POINT_SHOT",):
        return 0.040
    if ct in ("LOW_DANGER_PERIMETER",):
        return 0.055
    if ct in ("SLOT", "HIGH_DANGER_SLOT", "ONE_TIMER"):
        return 0.032
    return 0.042


def assist_count_probability(chance_type: str, strength: str) -> Tuple[float, float, float]:
    """Return (p0, p1, p2) assist counts excluding scorer."""
    p0 = zero_assist_probability(chance_type, strength)
    ct = str(chance_type or "")
    st = str(strength or "EV").upper()
    if st == "PP":
        p2 = 0.72 if ct in ("PP_ONE_TIMER", "PP_SLOT") else 0.66
    elif ct in ("RUSH_MEDIUM", "SH_RUSH"):
        p2 = 0.58 if st == "SH" else 0.62
    elif ct in ("REBOUND", "NET_FRONT"):
        p2 = 0.66
    elif ct in ("POINT_SHOT", "SLOT", "HIGH_DANGER_SLOT", "ONE_TIMER"):
        p2 = 0.68
    else:
        p2 = 0.64
    p2 = max(0.0, min(0.78, p2))
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
    """Final goal probability after shooter finishing and goalie quality.

    Scaled so full-event league SV% lands near .900–.907 (NHL-like) instead of
    the historical ~.87 band that made user-team goalies look broken vs light sim.
    """
    # Tuned so full-event GPG stays near modern NHL (~3.0/team) without the old
    # overscore band, while not systematically trailing the light counting path.
    prob = float(raw_xg) * 0.86 * max(0.55, min(1.45, float(finishing_adj)))
    prob *= max(0.55, min(1.45, float(goalie_adj)))
    prob *= max(0.85, min(1.15, float(situational_adj)))
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
    """
    att_ids = [player_id(p) for p in attacking_skaters if player_id(p)]
    def_ids = [player_id(p) for p in defending_skaters if player_id(p)]
    rxg = round(float(raw_xg), 4)

    for p in attacking_skaters:
        ledger_add(ledger, p, team_id, cf=1.0)
    for p in defending_skaters:
        ledger_add(ledger, p, opp_team_id, ca=1.0)

    if outcome == "BLOCKED":
        ledger_add(ledger, shooter, team_id, blocked_attempts_for=1)
        if blocker is not None:
            ledger_add(ledger, blocker, opp_team_id, blk=1)
        return

    for p in attacking_skaters:
        ledger_add(ledger, p, team_id, ff=1.0)
    for p in defending_skaters:
        ledger_add(ledger, p, opp_team_id, fa=1.0)

    if outcome == "MISSED":
        ledger_add(ledger, shooter, team_id, missed_shots=1)
        return

    for p in attacking_skaters:
        ledger_add(ledger, p, team_id, on_ice_shots_for=1.0)
    for p in defending_skaters:
        ledger_add(ledger, p, opp_team_id, on_ice_shots_against=1.0)

    ledger_add(ledger, shooter, team_id, sog=1, ixg=rxg)
    if str(strength or "").upper() == "PP":
        ledger_add(ledger, shooter, team_id, pp_sog=1)
    for p in attacking_skaters:
        ledger_add(ledger, p, team_id, xgf=rxg)
    for p in defending_skaters:
        ledger_add(ledger, p, opp_team_id, xga=rxg)
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
    hg = int(home_agg.get("goals", 0) or 0)
    ag = int(away_agg.get("goals", 0) or 0)
    h_sog = int(home_agg.get("sog", 0) or 0)
    a_sog = int(away_agg.get("sog", 0) or 0)
    h_sa = int(home_agg.get("goalie_sa", 0) or 0)
    a_sa = int(away_agg.get("goalie_sa", 0) or 0)
    if h_sa != a_sog:
        issues.append(f"home SA ({h_sa}) != away SOG ({a_sog})")
    if a_sa != h_sog:
        issues.append(f"away SA ({h_sa}) != home SOG ({h_sog})")
    return issues
