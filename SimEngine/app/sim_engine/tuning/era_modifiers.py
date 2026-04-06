# app/sim_engine/tuning/era_modifiers.py
"""
Dynamic era influence: scoring environment, archetype weights, development/aging, trade aggression, goalie value.
Applies modifiers directly to team/player objects and league-level simulation hooks.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

# Keys align with universe DEFAULT_ERAS and common aliases
ERA_PROFILES: Dict[str, Dict[str, float]] = {
    "run_and_gun": {
        "scoring_multiplier": 1.15,
        "defense_effectiveness": 0.88,
        "offense_archetype_boost": 1.06,
        "prospect_growth_boost": 1.10,
        "aging_penalty": 1.05,
        "trade_aggression": 1.22,
        "goalie_value": 0.94,
    },
    "dead_puck": {
        "scoring_multiplier": 0.86,
        "defense_effectiveness": 1.18,
        "offense_archetype_boost": 0.94,
        "prospect_growth_boost": 0.90,
        "aging_penalty": 0.96,
        "trade_aggression": 0.88,
        "goalie_value": 1.28,
    },
    "goalie_dominance": {
        "scoring_multiplier": 0.90,
        "defense_effectiveness": 1.08,
        "offense_archetype_boost": 0.97,
        "prospect_growth_boost": 0.95,
        "aging_penalty": 0.98,
        "trade_aggression": 1.05,
        "goalie_value": 1.35,
    },
    "speed_and_skill": {
        "scoring_multiplier": 1.08,
        "defense_effectiveness": 0.94,
        "offense_archetype_boost": 1.04,
        "prospect_growth_boost": 1.06,
        "aging_penalty": 1.02,
        "trade_aggression": 1.12,
        "goalie_value": 0.98,
    },
    "power_play_era": {
        "scoring_multiplier": 1.12,
        "defense_effectiveness": 0.92,
        "offense_archetype_boost": 1.05,
        "prospect_growth_boost": 1.04,
        "aging_penalty": 1.03,
        "trade_aggression": 1.15,
        "goalie_value": 0.96,
    },
    "two_way_chess": {
        "scoring_multiplier": 0.96,
        "defense_effectiveness": 1.06,
        "offense_archetype_boost": 0.99,
        "prospect_growth_boost": 1.02,
        "aging_penalty": 1.01,
        "trade_aggression": 1.08,
        "goalie_value": 1.08,
    },
}


def _normalize_era_key(name: str) -> str:
    s = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return s


def resolve_era_profile(active_era: str) -> Dict[str, float]:
    """Return merged profile for active era string; default to balanced two_way_chess-like."""
    k = _normalize_era_key(active_era)
    if k in ERA_PROFILES:
        return dict(ERA_PROFILES[k])
    # fuzzy contains
    for key, prof in ERA_PROFILES.items():
        if key in k or k in key:
            return dict(prof)
    return {
        "scoring_multiplier": 1.0,
        "defense_effectiveness": 1.0,
        "offense_archetype_boost": 1.0,
        "prospect_growth_boost": 1.0,
        "aging_penalty": 1.0,
        "trade_aggression": 1.0,
        "goalie_value": 1.0,
    }


def _is_goalie(player: Any) -> bool:
    pos = getattr(player, "position", None)
    if pos is None:
        return False
    return getattr(pos, "value", str(pos)) == "G"


def _rating_nudge_keys(ratings: Dict[str, Any], prefix: str, factor: float) -> int:
    changed = 0
    for key in list(ratings.keys()):
        if str(key).startswith(prefix):
            try:
                ratings[key] = int(max(20, min(99, round(float(ratings[key]) * factor))))
                changed += 1
            except (TypeError, ValueError):
                pass
    return changed


def apply_era_modifiers(league_state: Mapping[str, Any], team: Any, player: Any) -> Dict[str, Any]:
    """
    Apply era profile to a single player in team context. Mutates player ratings (soft),
    career/psych when present, and caches multipliers on objects for other systems.

    league_state: mapping with at least 'active_era', optionally 'league_health', 'chaos_index'.

    Returns a small summary dict for aggregation/logging.
    """
    era = str(league_state.get("active_era", "") or "")
    prof = resolve_era_profile(era)
    health = float(league_state.get("league_health", 0.55) or 0.55)
    chaos = float(league_state.get("chaos_index", 0.5) or 0.5)
    # Era intensity: healthier league = slightly stronger era expression
    intensity = 0.65 + 0.35 * health - 0.08 * (chaos - 0.5)

    off_boost = 1.0 + (prof["offense_archetype_boost"] - 1.0) * intensity
    def_eff = 1.0 + (prof["defense_effectiveness"] - 1.0) * intensity
    g_val = 1.0 + (prof["goalie_value"] - 1.0) * intensity

    summary: Dict[str, Any] = {"era": era, "rating_keys_touched": 0}

    # Cache for progression / emergence / aging in engine
    setattr(player, "_tuning_era_dev", float(prof["prospect_growth_boost"]))
    setattr(player, "_tuning_era_aging", float(prof["aging_penalty"]))
    setattr(player, "_tuning_goalie_value", float(g_val))
    setattr(team, "_tuning_trade_aggression", float(prof["trade_aggression"]))

    ratings = getattr(player, "ratings", None)
    if ratings and isinstance(ratings, dict):
        if _is_goalie(player):
            n = _rating_nudge_keys(ratings, "g_", g_val ** 0.25)
            summary["rating_keys_touched"] += n
        else:
            n1 = _rating_nudge_keys(ratings, "of_", off_boost ** 0.35)
            n2 = _rating_nudge_keys(ratings, "ps_", off_boost ** 0.20)
            n3 = _rating_nudge_keys(ratings, "df_", def_eff ** 0.30)
            summary["rating_keys_touched"] += n1 + n2 + n3

    career = getattr(player, "career", None)
    if career is not None:
        try:
            peak = int(getattr(career, "expected_peak_age", 27))
            if prof["aging_penalty"] > 1.02:
                setattr(career, "expected_peak_age", min(32, peak + 1))
            elif prof["aging_penalty"] < 0.98:
                setattr(career, "expected_peak_age", max(24, peak - 1))
        except Exception:
            pass

    psych = getattr(player, "psych", None)
    if psych is not None and hasattr(psych, "morale"):
        try:
            delta = 0.015 * (prof["scoring_multiplier"] - 1.0)
            m = float(psych.morale) + delta
            psych.morale = max(0.0, min(1.0, m))
        except Exception:
            pass

    return summary


def apply_era_to_league(
    league_state: Mapping[str, Any],
    league: Any,
    teams: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Apply era modifiers to all roster players; set league-level scoring multipliers for sim layers.
    """
    prof = resolve_era_profile(str(league_state.get("active_era", "") or ""))
    agg = {"players": 0, "rating_keys_touched": 0, "profile": prof}
    tlist = teams if teams is not None else (getattr(league, "teams", None) or [])
    setattr(league, "_era_scoring_multiplier", float(prof["scoring_multiplier"]))
    setattr(league, "_era_defense_effectiveness", float(prof["defense_effectiveness"]))
    setattr(league, "_era_goalie_value", float(prof["goalie_value"]))
    for team in tlist:
        roster = getattr(team, "roster", None) or []
        for player in roster:
            if getattr(player, "retired", False):
                continue
            s = apply_era_modifiers(league_state, team, player)
            agg["players"] += 1
            agg["rating_keys_touched"] += int(s.get("rating_keys_touched", 0))
    return agg
