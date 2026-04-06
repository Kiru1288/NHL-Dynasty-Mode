# app/sim_engine/tuning/normalization.py
"""
Long-term stability: soft regression toward realistic means for league macro, player OVR distribution, team strength.
Era-aware via league_state['active_era'].
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from app.sim_engine.tuning.era_modifiers import resolve_era_profile


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def macro_progression_scales(league_state: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    """
    Active control: chaos / parity / league health / era shape volatility of breakouts, declines, busts.
    Returns multipliers (~0.5–1.45) applied inside engine career lifecycle checks.
    """
    ls = league_state or {}
    c = float(ls.get("chaos_index", 0.5) or 0.5)
    p = float(ls.get("parity_index", 0.5) or 0.5)
    h = float(ls.get("league_health", 0.55) or 0.55)
    era = str(ls.get("active_era", "") or "")
    prof = resolve_era_profile(era)
    sc = float(prof.get("scoring_multiplier", 1.0) or 1.0)
    off_boost = max(0.0, sc - 1.0)

    breakout_p_mult = 0.56 + 0.88 * c + 0.10 * off_boost
    if h < 0.44:
        breakout_p_mult *= 1.0 + 0.16 * (0.44 - h)
    if p > 0.62:
        breakout_p_mult *= 1.0 - 0.11 * min(1.0, (p - 0.62) / 0.26)
    breakout_p_mult = max(0.48, min(1.52, breakout_p_mult))

    decline_mag_mult = 0.70 + 0.58 * c
    if h > 0.58:
        decline_mag_mult *= 1.0 - 0.14 * min(1.0, (h - 0.58) / 0.35)
    if h < 0.38:
        decline_mag_mult *= 0.90
    decline_mag_mult = max(0.62, min(1.42, decline_mag_mult))

    bust_p_mult = 0.62 + 0.72 * c
    bust_p_mult = max(0.50, min(1.45, bust_p_mult))

    late_bloom_p_mult = 0.68 + 0.62 * c + 0.06 * off_boost
    if p > 0.60:
        late_bloom_p_mult *= 0.94
    late_bloom_p_mult = max(0.55, min(1.40, late_bloom_p_mult))

    return {
        "breakout_p_mult": breakout_p_mult,
        "decline_mag_mult": decline_mag_mult,
        "bust_p_mult": bust_p_mult,
        "late_bloom_p_mult": late_bloom_p_mult,
    }


def apply_season_feedback_to_league_state(
    league_state: MutableMapping[str, Any],
    teams: Sequence[Any],
) -> Dict[str, Any]:
    """
    Close the loop: elite counts + team strength spread nudge chaos/parity/health toward long-run balance.
    Mutates league_state in place (same keys as UniverseState tuning ctx).
    """
    report: Dict[str, Any] = {
        "elite_n": 0,
        "chaos_delta": 0.0,
        "parity_delta": 0.0,
        "health_delta": 0.0,
    }
    if not teams:
        return report

    ovrs: List[float] = []
    team_means: List[float] = []
    for t in teams:
        roster = [p for p in getattr(t, "roster", None) or [] if not getattr(p, "retired", False)]
        if not roster:
            continue
        m = sum(_ovr(p) for p in roster) / len(roster)
        team_means.append(m)
        for p in roster:
            try:
                ovrs.append(_ovr(p))
            except Exception:
                pass

    if not ovrs:
        return report

    elite_n = sum(1 for x in ovrs if x >= 0.88)
    report["elite_n"] = int(elite_n)
    n = len(ovrs)
    target_elite = max(5, min(24, n // 52))

    chaos = float(league_state.get("chaos_index", 0.5) or 0.5)
    parity = float(league_state.get("parity_index", 0.5) or 0.5)
    health = float(league_state.get("league_health", 0.55) or 0.55)

    if elite_n < target_elite:
        d_ch = 0.010 * (1.0 - 0.35 * chaos)
        league_state["chaos_index"] = _clamp01(chaos + d_ch)
        league_state["league_health"] = _clamp01(health + 0.007)
        report["chaos_delta"] += d_ch
        report["health_delta"] += 0.007
    elif elite_n > target_elite + 8:
        league_state["parity_index"] = _clamp01(parity + 0.014)
        league_state["league_health"] = _clamp01(health - 0.005)
        report["parity_delta"] += 0.014
        report["health_delta"] -= 0.005

    if len(team_means) >= 10:
        var = float(statistics.pvariance(team_means))
        parity2 = float(league_state.get("parity_index", 0.5) or 0.5)
        if var > 0.0028 and parity2 > 0.50:
            dp = 0.009 * min(1.0, (var - 0.0028) / 0.004)
            league_state["parity_index"] = _clamp01(float(league_state["parity_index"]) + dp)
            report["parity_delta"] += dp
        elif var < 0.00075 and parity2 < 0.48:
            dp = -0.007
            league_state["parity_index"] = _clamp01(float(league_state["parity_index"]) + dp)
            report["parity_delta"] += dp

    return report


def _ovr(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    if callable(fn):
        try:
            return float(fn())
        except Exception:
            pass
    v = getattr(player, "ovr", 0.5)
    return float(v) / 99.0 if float(v) > 1.5 else float(v)


def _age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    if ident is not None and hasattr(ident, "age"):
        return int(ident.age)
    return int(getattr(player, "age", 26))


def normalize_league_stats(league_state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """
    Keep macro indices in realistic bands; damp runaway chaos/parity. Mutates league_state in place.
    """
    report: Dict[str, Any] = {"chaos_pull": 0.0, "parity_pull": 0.0, "health_pull": 0.0}
    chaos = float(league_state.get("chaos_index", 0.5) or 0.5)
    parity = float(league_state.get("parity_index", 0.5) or 0.5)
    health = float(league_state.get("league_health", 0.55) or 0.55)
    era = str(league_state.get("active_era", "") or "")
    prof = resolve_era_profile(era)
    # Run-and-gun / power play: slightly higher acceptable chaos ceiling
    chaos_target = 0.52 + 0.06 * max(0.0, prof.get("scoring_multiplier", 1.0) - 1.0)
    parity_target = 0.48 + 0.04 * (prof.get("defense_effectiveness", 1.0) - 1.0)

    new_chaos = chaos + 0.18 * (chaos_target - chaos)
    new_parity = parity + 0.14 * (parity_target - parity)
    # League health tracks midpoint of stability
    stability = 1.0 - abs(new_chaos - 0.5) - abs(new_parity - 0.5) * 0.5
    new_health = health + 0.12 * (0.55 + 0.25 * stability - health)

    report["chaos_pull"] = new_chaos - chaos
    report["parity_pull"] = new_parity - parity
    report["health_pull"] = new_health - health

    league_state["chaos_index"] = _clamp01(new_chaos)
    league_state["parity_index"] = _clamp01(new_parity)
    league_state["league_health"] = _clamp01(new_health)
    return report


def normalize_player_ratings(
    players: Sequence[Any],
    league_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Soft caps on runaway growth/elite saturation; gentle pull toward tier means (not hard clamps).
    """
    report = {"nudged_high": 0, "nudged_low": 0, "mean_before": 0.0, "mean_after": 0.0}
    plist = [p for p in players if p is not None and not getattr(p, "retired", False)]
    if not plist:
        return report

    ovrs = [_ovr(p) for p in plist]
    mean_o = sum(ovrs) / len(ovrs)
    report["mean_before"] = mean_o
    era = str((league_state or {}).get("active_era", "") or "")
    prof = resolve_era_profile(era)
    # Higher scoring era: allow slightly higher mean ceiling
    mean_target = 0.695 + 0.02 * (prof.get("scoring_multiplier", 1.0) - 1.0)
    mean_target = _clamp01(mean_target)

    ratings: List[Any] = []
    for p in plist:
        r = getattr(p, "ratings", None)
        if r and isinstance(r, dict):
            ratings.append((p, r))

    # Pull global mean toward target (very soft)
    if ratings and abs(mean_o - mean_target) > 0.018:
        alpha = 0.06 * math.copysign(1.0, mean_target - mean_o)
        scale = 1.0 + alpha
        for p, rdict in ratings:
            cur = _ovr(p)
            if (mean_o > mean_target and cur > mean_target) or (mean_o < mean_target and cur < mean_target):
                for k in list(rdict.keys()):
                    try:
                        rdict[k] = int(max(20, min(99, round(float(rdict[k]) * scale))))
                    except (TypeError, ValueError):
                        pass

    # Recompute
    ovrs2 = [_ovr(p) for p in plist]
    mean2 = sum(ovrs2) / len(ovrs2)
    report["mean_after"] = mean2

    # Elite saturation: too many >0.88 → micro-decay (chaos allows deeper elite bench; low health eases decay)
    chaos = float((league_state or {}).get("chaos_index", 0.5) or 0.5)
    health = float((league_state or {}).get("league_health", 0.55) or 0.55)
    elite = [p for p in plist if _ovr(p) >= 0.88]
    cap_base = max(8, len(plist) // 45)
    cap_mult = 1.0 + 0.32 * max(0.0, chaos - 0.5) - 0.12 * max(0.0, health - 0.58)
    elite_cap = max(6, int(round(cap_base * max(0.82, min(1.35, cap_mult)))))
    if len(elite) > elite_cap:
        decay = 0.997 - 0.0008 * max(0.0, chaos - 0.5) + 0.0006 * max(0.0, 0.5 - health)
        decay = max(0.9945, min(0.9985, decay))
        for p in elite:
            rdict = getattr(p, "ratings", None)
            if not rdict or not isinstance(rdict, dict):
                continue
            for k in list(rdict.keys()):
                try:
                    rdict[k] = int(max(20, min(99, round(float(rdict[k]) * decay))))
                except (TypeError, ValueError):
                    pass
            report["nudged_high"] += 1

    # Talent collapse guard: mean too low → lift bottom quartile
    if mean2 < 0.635:
        sorted_p = sorted(plist, key=_ovr)
        n = max(1, len(sorted_p) // 4)
        for p in sorted_p[:n]:
            rdict = getattr(p, "ratings", None)
            if not rdict or not isinstance(rdict, dict):
                continue
            for k in list(rdict.keys()):
                try:
                    rdict[k] = int(max(20, min(99, round(float(rdict[k]) * 1.004))))
                except (TypeError, ValueError):
                    pass
            report["nudged_low"] += 1

    return report


def normalize_team_strengths(
    teams: Sequence[Any],
    league_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Prevent permanent basement / dynasty lock-in: nudge team state and roster averages slightly toward league center.
    """
    report = {"teams_boosted": 0, "teams_trimmed": 0}
    if not teams:
        return report

    team_means: List[tuple] = []
    for t in teams:
        roster = getattr(t, "roster", None) or []
        active = [p for p in roster if not getattr(p, "retired", False)]
        if not active:
            continue
        m = sum(_ovr(p) for p in active) / len(active)
        team_means.append((t, m, active))

    if not team_means:
        return report

    league_mean = sum(m for _, m, _ in team_means) / len(team_means)
    center = league_mean
    parity = float((league_state or {}).get("parity_index", 0.5) or 0.5)
    health = float((league_state or {}).get("league_health", 0.55) or 0.55)
    pull = 0.012 + 0.024 * parity
    sorted_means = sorted((m for _, m, _ in team_means), reverse=True)
    top_gap = (sorted_means[0] - sorted_means[1]) if len(sorted_means) >= 2 else 0.0
    dynasty_trim = 1.0
    if parity > 0.56 and top_gap > 0.045:
        dynasty_trim = max(0.988, 1.0 - 0.06 * (parity - 0.56) * min(1.0, top_gap / 0.08))

    for t, tm, active in team_means:
        gap = center - tm
        if abs(gap) < 0.022:
            continue
        # Soft nudge entire roster (stronger toward mean when parity is high)
        factor = 1.0 + pull * gap / (abs(gap) + 0.08)
        lo_b, hi_b = 0.982, 1.018
        if parity > 0.55:
            lo_b, hi_b = 0.978, 1.022
        if tm >= sorted_means[0] - 1e-6 and dynasty_trim < 1.0:
            factor *= dynasty_trim
        factor = max(lo_b, min(hi_b, factor))
        if health < 0.42 and gap > 0:
            factor = min(hi_b, factor * 1.004)
        state = getattr(t, "state", None)
        if state is not None:
            try:
                cs = float(getattr(state, "competitive_score", 0.5))
                setattr(state, "competitive_score", _clamp01(cs + 0.04 * gap))
            except Exception:
                pass
        for p in active:
            rdict = getattr(p, "ratings", None)
            if not rdict or not isinstance(rdict, dict):
                continue
            for k in list(rdict.keys()):
                try:
                    rdict[k] = int(max(20, min(99, round(float(rdict[k]) * factor))))
                except (TypeError, ValueError):
                    pass
        if gap > 0:
            report["teams_boosted"] += 1
        else:
            report["teams_trimmed"] += 1

    return report
