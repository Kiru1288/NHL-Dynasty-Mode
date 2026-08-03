# app/sim_engine/progression/development.py
"""
Young player growth based on potential, morale, games played, development rate, team role.
Under-24 get most growth; low ice time = slower development.

Career arc: phase from age (prospect → declining), yearly trend (hot / stable / declining).

Extended systems (franchise-friendly, all self-contained in this module):
  - career stage detection
  - NHL readiness scoring
  - development fit scoring
  - targeted attribute growth by position/playstyle
  - prime refinement (small polish for 25–31)
  - veteran decline profile metadata
  - structured development reports for UI
"""

from typing import Any, Dict, List, Optional, Tuple

import random

from app.sim_engine.entities.player import (
    clamp01,
    display_rating,
    normalize_rating,
    normalize_rating_gap,
    player_current_ovr_01,
    persist_recomputed_ovr,
)

# --- Career arc phases (age bands; assigned each year on player.career_phase) ---
PHASE_PROSPECT = "prospect"
PHASE_EMERGING = "emerging"
PHASE_PRIME = "prime"
PHASE_VETERAN = "veteran"
PHASE_DECLINING = "declining"

# --- Franchise career-stage labels (richer than career_phase) ---
STAGE_PROSPECT = "prospect"
STAGE_YOUNG_NHL = "young_nhl_player"
STAGE_EMERGING_CORE = "emerging_core"
STAGE_PRIME = "prime"
STAGE_VETERAN = "veteran"
STAGE_DECLINING = "declining"
STAGE_LATE_CAREER = "late_career"


# ---------------------------------------------------------------------------
# Safe helpers — never crash on partial player objects
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _get_player_position(player: Any) -> str:
    pos = getattr(player, "position", None)
    if pos is not None:
        v = getattr(pos, "value", None)
        if v is not None:
            return str(v).upper()
        return str(pos).upper()
    ident = getattr(player, "identity", None)
    if ident is not None:
        ip = getattr(ident, "position", None)
        if ip is not None:
            return str(getattr(ip, "value", ip)).upper()
    return str(getattr(player, "pos", "") or "F").upper()


def _is_goalie(player: Any) -> bool:
    return _get_player_position(player) in ("G", "GOALIE", "GOALTENDER")


def _is_defense(player: Any) -> bool:
    return _get_player_position(player) in ("D", "LD", "RD", "DEF", "DEFENSE")


def _get_player_age(player: Any) -> int:
    identity = getattr(player, "identity", None)
    if identity is not None and hasattr(identity, "age"):
        return _safe_int(identity.age, 26)
    return _safe_int(getattr(player, "age", 26), 26)


def _get_player_ovr(player: Any) -> float:
    try:
        from app.sim_engine.entities.player import player_current_ovr_01

        return float(player_current_ovr_01(player))
    except Exception:
        ovr_fn = getattr(player, "ovr", None)
        if callable(ovr_fn):
            try:
                from app.sim_engine.entities.player import normalize_rating

                return float(normalize_rating(ovr_fn()))
            except Exception:
                pass
        from app.sim_engine.entities.player import normalize_rating

        return float(normalize_rating(getattr(player, "ovr", 0.5)))


def _get_player_ovr_0_100(player: Any) -> float:
    from app.sim_engine.entities.player import display_rating

    return float(display_rating(_get_player_ovr(player)))


def _get_player_potential(player: Any) -> float:
    from app.sim_engine.entities.player import normalize_rating

    profile = getattr(player, "development_profile", None)
    if isinstance(profile, dict) and profile.get("expected_ceiling") is not None:
        return float(normalize_rating(profile.get("expected_ceiling")))
    p = getattr(player, "potential", None)
    if p is not None:
        return float(normalize_rating(p))
    ratings = getattr(player, "ratings", None)
    if isinstance(ratings, dict):
        for key in ("dev_potential", "potential", "dev_ceiling"):
            if ratings.get(key) is not None:
                return float(normalize_rating(ratings.get(key)))
    return _clamp(_get_player_ovr(player) * 1.05, 0.2, 0.99)


def _get_player_archetype(player: Any) -> str:
    return str(
        getattr(player, "_dev_archetype", "")
        or getattr(player, "archetype", "")
        or ""
    ).strip()


def _get_player_dev_type(player: Any) -> str:
    return str(getattr(player, "dev_type", "standard") or "standard").lower()


def _get_player_role(player: Any) -> str:
    return str(getattr(player, "role", "") or "").lower()


def _get_player_morale(player: Any) -> float:
    psych = getattr(player, "psych", None)
    if psych is not None and hasattr(psych, "morale"):
        return _clamp(_safe_float(psych.morale, 0.5), 0.0, 1.0)
    return _clamp(_safe_float(getattr(player, "morale", 0.5), 0.5), 0.0, 1.0)


def _get_games_played(player: Any) -> int:
    return _safe_int(getattr(player, "games_played", 0), 0)


def _safe_setattr(player: Any, key: str, value: Any) -> None:
    try:
        setattr(player, key, value)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Career stage — franchise-readable lifecycle bucket
# ---------------------------------------------------------------------------

def determine_development_career_stage(player: Any) -> str:
    """
    Return one of: prospect, young_nhl_player, emerging_core, prime,
    veteran, declining, late_career.
    Goalies mature slightly later; uses NHL GP when available.
    """
    age = _get_player_age(player)
    gp = _get_games_played(player)
    ovr = _get_player_ovr_0_100(player)
    pot = _get_player_potential(player) * 100.0
    goalie = _is_goalie(player)
    defense = _is_defense(player)

    late_threshold = 24 if goalie else (23 if defense else 22)
    young_cutoff = 22 if goalie else 21

    if age >= 38:
        return STAGE_LATE_CAREER
    if age >= 35:
        return STAGE_DECLINING
    if age >= 32:
        return STAGE_VETERAN
    if 25 <= age <= 31:
        if ovr >= 78 and pot >= 80:
            return STAGE_EMERGING_CORE if age <= 27 else STAGE_PRIME
        return STAGE_PRIME
    if age <= young_cutoff:
        if gp >= 10:
            return STAGE_YOUNG_NHL
        return STAGE_PROSPECT
    if age <= late_threshold + 2:
        return STAGE_YOUNG_NHL
    if age <= 24:
        return STAGE_YOUNG_NHL
    if ovr >= 76 and pot >= 78:
        return STAGE_EMERGING_CORE
    return STAGE_YOUNG_NHL


# ---------------------------------------------------------------------------
# NHL readiness — 0–100 call-up readiness estimate
# ---------------------------------------------------------------------------

def calculate_nhl_readiness_score(player: Any, context: Optional[Any] = None) -> float:
    """
    0–100 NHL readiness. Goalies face stricter gates; D mature slightly later.
  Attaches player.nhl_readiness when possible.
    """
    age = _get_player_age(player)
    ovr = _get_player_ovr_0_100(player)
    pot = _get_player_potential(player) * 100.0
    morale = _get_player_morale(player)
    gp = _get_games_played(player)
    role = _get_player_role(player)
    dev_type = _get_player_dev_type(player)
    goalie = _is_goalie(player)
    defense = _is_defense(player)

    score = ovr * 0.62 + (pot - ovr) * 0.18 + morale * 12.0
    score += min(8.0, gp * 0.12)

    if role and any(k in role for k in ("elite", "top_line", "top_4", "starter")):
        score += 6.0
    elif role and any(k in role for k in ("depth", "scratch", "press", "prospect")):
        score -= 4.0

    if dev_type in ("elite", "steal"):
        score += 4.0
    elif dev_type == "late_bloomer" and age >= 22:
        score += 3.0
    elif dev_type == "bust":
        score -= 6.0

    # Age curves — goalies get grace under 23; D slightly later
    if goalie:
        if age < 21:
            score -= 8.0
        elif age < 23:
            score -= 3.0
        elif age >= 30:
            score += 2.0
    elif defense:
        if age < 20:
            score -= 6.0
        elif age < 22:
            score -= 2.0
    else:
        if age < 19:
            score -= 7.0
        elif age < 20:
            score -= 3.0

    if context is not None:
        ctx_bonus = _safe_float(getattr(context, "readiness_bonus", 0), 0)
        score += ctx_bonus

    score = _clamp(score, 0.0, 100.0)
    _safe_setattr(player, "nhl_readiness", round(score, 1))
    return score


def update_player_nhl_eta(player: Any) -> int:
    """Recompute years-to-NHL-arrival from current readiness/ability.

    Distinct from potential (ceiling) and from nhl_readiness (0–100 now-score).
    Shortens when development accelerates; lengthens on stalls / young goalies.
    Established NHL players are stamped as 0 with a status label.
    """
    readiness = _safe_float(getattr(player, "nhl_readiness", 0), 0.0)
    if 0.0 < readiness <= 1.5:
        readiness *= 100.0
    age = _get_player_age(player)
    ovr = _get_player_ovr_0_100(player)
    gp = _get_games_played(player)
    goalie = _is_goalie(player)
    status = str(getattr(player, "status", "") or getattr(player, "prospect_status", "") or "").lower()

    if status in ("nhl", "active") and gp >= 40:
        years, label = 0, "NHL regular"
    elif status in ("nhl", "active") and gp >= 10:
        years, label = 0, "NHL depth"
    elif readiness >= 78 and age >= 18 and ovr >= 74:
        years, label = 0, "Now"
    elif readiness >= 70 or (ovr >= 72 and age >= 19):
        years, label = 1, "1Y"
    elif readiness >= 58 or ovr >= 66:
        years, label = 2, "2Y"
    elif readiness >= 48 or ovr >= 60:
        years, label = 3, "3Y"
    else:
        years, label = 4, "4Y+"

    if goalie and years < 4 and label not in ("NHL regular", "NHL depth"):
        years = min(4, int(years) + 1)
        label = {0: "Now", 1: "1Y", 2: "2Y", 3: "3Y"}.get(years, "4Y+")

    if age <= 17 and years < 2:
        years, label = 2, "2Y"
    if age <= 16:
        years, label = max(years, 3), "3Y"

    prev = getattr(player, "nhl_eta", None)
    try:
        prev_i = int(prev) if prev is not None else None
    except (TypeError, ValueError):
        prev_i = None
    # One-step drift per season so ETA does not teleport after a single pulse.
    if prev_i is not None and label not in ("NHL regular", "NHL depth"):
        if years < prev_i:
            years = max(years, prev_i - 1)
        elif years > prev_i:
            years = min(years, prev_i + 1)
        label = {0: "Now", 1: "1Y", 2: "2Y", 3: "3Y"}.get(years, "4Y+")

    _safe_setattr(player, "nhl_eta", int(years))
    _safe_setattr(player, "nhl_eta_label", str(label))
    return int(years)


# ---------------------------------------------------------------------------
# Development fit — environment / role match
# ---------------------------------------------------------------------------

def _development_fit_label(
    score: float,
    player: Any,
    career_stage: str,
    readiness: float,
) -> str:
    age = _get_player_age(player)
    gp = _get_games_played(player)
    role = _get_player_role(player)

    if score >= 85:
        return "Perfect Fit"
    if score >= 70:
        return "Good Fit"
    if age <= 22 and gp >= 40 and readiness < 55:
        return "Being Rushed"
    if career_stage == STAGE_PROSPECT and readiness < 45 and gp > 5:
        return "Wrong Level"
    if any(k in role for k in ("fourth", "depth", "scratch", "press")) and career_stage in (
        STAGE_YOUNG_NHL,
        STAGE_EMERGING_CORE,
    ):
        return "Needs Bigger Role"
    if str(getattr(player, "_dev_last_phase", "") or "") == "STALL":
        return "Stalled Risk"
    if score < 40:
        return "Wrong Level"
    if score < 55:
        return "Needs Bigger Role"
    return "Good Fit"


def calculate_development_fit_score(player: Any, context: Optional[Any] = None) -> float:
    """
    0–100 estimate of whether the player is in the right development environment.
    Attaches development_fit_score and development_fit_label.
    """
    age = _get_player_age(player)
    gp = _get_games_played(player)
    morale = _get_player_morale(player)
    role = _get_player_role(player)
    ovr = _get_player_ovr_0_100(player)
    pot = _get_player_potential(player) * 100.0
    headroom = max(0.0, pot - ovr)
    career_stage = str(
        getattr(player, "development_career_stage", "")
        or determine_development_career_stage(player)
    )
    dev_type = _get_player_dev_type(player)
    adj = _safe_int(getattr(player, "_nhl_adjustment_years_remaining", 0), 0)
    env_g = _safe_float(getattr(player, "_dev_env_growth_mult", 1.0), 1.0)

    score = 52.0
    score += headroom * 0.22
    score += morale * 18.0
    score += (env_g - 1.0) * 22.0

    if any(k in role for k in ("elite", "top_line", "top_4", "starter")):
        score += 12.0
    elif any(k in role for k in ("middle", "middle_6")):
        score += 6.0
    elif any(k in role for k in ("fourth", "depth", "scratch")):
        score -= 10.0

    if gp >= 55:
        score += 5.0
    elif gp < 15 and career_stage in (STAGE_YOUNG_NHL, STAGE_EMERGING_CORE):
        score -= 4.0

    if adj > 0:
        score -= adj * 4.0
    if dev_type == "bust":
        score -= 8.0
    elif dev_type in ("elite", "steal"):
        score += 4.0

    if career_stage == STAGE_PROSPECT and gp > 20:
        score -= 6.0
    if career_stage == STAGE_PRIME and gp < 30:
        score -= 3.0

    if context is not None:
        score += _safe_float(getattr(context, "fit_bonus", 0), 0)

    score = _clamp(score, 0.0, 100.0)
    readiness = _safe_float(getattr(player, "nhl_readiness", 0), 0)
    if readiness <= 0:
        readiness = calculate_nhl_readiness_score(player, context)

    label = _development_fit_label(score, player, career_stage, readiness)
    _safe_setattr(player, "development_fit_score", round(score, 1))
    _safe_setattr(player, "development_fit_label", label)
    return score


# ---------------------------------------------------------------------------
# Targeted attribute growth — position / playstyle aware
# ---------------------------------------------------------------------------

def _key_matches_any(key: str, patterns: Tuple[str, ...]) -> bool:
    kl = key.lower()
    return any(p in kl for p in patterns)


def _playstyle_bucket(player: Any) -> str:
    style = str(
        getattr(player, "playstyle", "")
        or getattr(player, "archetype", "")
        or getattr(player, "_generated_profile", "")
        or getattr(player, "player_type", "")
        or ""
    ).lower().replace(" ", "_").replace("-", "_")
    if not style:
        if _is_goalie(player):
            return "goalie"
        if _is_defense(player):
            return "two_way_defenseman"
        return "two_way"
    return style


def distribute_growth_by_player_type(
    player: Any,
    total_growth: float,
    phase: str = "NORMAL",
) -> Dict[str, float]:
    """
    Return rating-key → delta map. Fuzzy-matches keys; uniform fallback if no keys match.
    """
    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict) or not ratings:
        return {}

    style = _playstyle_bucket(player)
    goalie = _is_goalie(player)
    defense = _is_defense(player)

    # Pattern groups → relative weight
    groups: Dict[str, Tuple[str, ...]] = {
        "shooting": ("shot", "shoot", "accuracy", "wrist", "slap", "release"),
        "passing": ("pass", "vision", "playmak"),
        "off_aware": ("offensive_aware", "off_aware", "offense_iq", "creativity"),
        "puck": ("puck_control", "handling", "deke", "dangle"),
        "physical": ("strength", "balance", "physical", "body", "fight", "hit"),
        "defense": ("defensive_aware", "def_aware", "stick_check", "block", "poke"),
        "skating": ("speed", "accel", "skating", "agility", "edge"),
        "faceoff": ("faceoff", "draw"),
        "stamina": ("stamina", "endurance", "condition"),
        "discipline": ("disciplin", "penalty"),
        "consistency": ("consist", "poise", "clutch", "pressure"),
        "iq": ("iq", "awareness", "anticipation", "read"),
        "glove": ("glove", "catch"),
        "blocker": ("blocker", "pad"),
        "rebound": ("rebound", "recovery"),
        "positioning": ("position", "angle", "depth"),
    }

    # Style → group weights (higher = more growth share). Archetype beats position bucket.
    weight_map: Dict[str, float] = {}
    if goalie:
        weight_map = {
            "rebound": 1.25, "glove": 1.2, "blocker": 1.15, "positioning": 1.2,
            "consistency": 1.1, "iq": 0.9, "skating": 0.5,
        }
    elif "sniper" in style or "shooter" in style:
        weight_map = {"shooting": 1.3, "off_aware": 1.05, "puck": 0.95, "skating": 0.9}
        if defense:
            weight_map["defense"] = 0.75
    elif "playmaker" in style or "play_maker" in style:
        weight_map = {"passing": 1.3, "off_aware": 1.15, "puck": 1.1, "shooting": 0.85}
        if defense:
            weight_map["defense"] = 0.82
    elif "power" in style:
        weight_map = {"physical": 1.25, "puck": 1.05, "shooting": 0.95, "skating": 1.1}
    elif "grinder" in style or "enforcer" in style:
        weight_map = {
            "physical": 1.2, "defense": 1.1, "stamina": 1.1, "discipline": 1.05,
            "shooting": 0.8,
        }
    elif defense:
        if "offensive" in style:
            weight_map = {
                "passing": 1.2, "off_aware": 1.15, "skating": 1.1, "puck": 1.05,
                "shooting": 0.95, "defense": 0.85,
            }
        elif "defensive" in style or "stay_at_home" in style:
            weight_map = {
                "defense": 1.25, "physical": 1.1, "discipline": 1.05,
                "skating": 0.9, "shooting": 0.7,
            }
        else:
            weight_map = {
                "defense": 1.1, "skating": 1.05, "passing": 1.0, "off_aware": 0.95,
                "physical": 0.95,
            }
    else:
        weight_map = {
            "defense": 1.05, "off_aware": 1.05, "faceoff": 1.0, "skating": 1.0,
            "passing": 0.95,
        }

    key_weights: Dict[str, float] = {}
    for k in ratings.keys():
        w = 0.35  # baseline so every attribute gets some growth
        for group, patterns in groups.items():
            if _key_matches_any(k, patterns):
                w += weight_map.get(group, 0.65)
        key_weights[k] = w

    total_w = sum(key_weights.values()) or 1.0
    phase_up = str(phase or "NORMAL").upper()
    sign = -1.0 if phase_up == "REGRESSION" or total_growth < 0 else 1.0
    magnitude = abs(total_growth)

    if phase_up == "STALL":
        magnitude *= 0.12
    elif phase_up == "SPIKE":
        magnitude *= 1.0  # already baked into total_growth
    elif phase_up == "REGRESSION":
        magnitude = max(magnitude, 0.05)

    deltas: Dict[str, float] = {}
    for k, w in key_weights.items():
        share = w / total_w
        deltas[k] = sign * magnitude * share * len(ratings)

    if not deltas:
        per = total_growth / max(1, len(ratings))
        for k in ratings:
            deltas[k] = per

    return deltas


def _apply_rating_deltas(player: Any, deltas: Dict[str, float]) -> float:
    """Apply deltas to player.ratings; return net OVR-scale movement estimate."""
    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict):
        return 0.0
    applied = 0.0
    for k, d in deltas.items():
        if k not in ratings:
            continue
        before = _safe_float(ratings[k], 50)
        after = _clamp_rating(before + d)
        ratings[k] = after
        applied += after - before
    return applied / max(1, len(deltas))


# ---------------------------------------------------------------------------
# Prime refinement — small polish for established players
# ---------------------------------------------------------------------------

def apply_prime_refinement(player: Any, context: Optional[Any] = None) -> None:
    """
    Ages ~25–31: tiny consistency / awareness / poise bumps — not prospect-level growth.
    """
    stage = str(
        getattr(player, "development_career_stage", "")
        or determine_development_career_stage(player)
    )
    age = _get_player_age(player)
    if stage not in (STAGE_PRIME, STAGE_EMERGING_CORE) and not (25 <= age <= 31):
        return

    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict):
        return

    polish_patterns = (
        "consist", "poise", "aware", "disciplin", "leadership",
        "iqm", "clutch", "pressure", "composure",
    )
    bump = 0.035 + 0.015 * _get_player_morale(player)
    touched = 0
    for k in list(ratings.keys()):
        if _key_matches_any(k, polish_patterns):
            ratings[k] = _clamp_rating(_safe_float(ratings[k], 50) + bump)
            touched += 1
    if touched:
        _safe_setattr(player, "_prime_refinement_applied", True)


# ---------------------------------------------------------------------------
# Veteran decline profile — metadata for UI / narrative (does not replace regression.py)
# ---------------------------------------------------------------------------

def calculate_age_decline_profile(player: Any) -> str:
    """
    Classify likely decline style for veterans. Attaches player.decline_profile.
    """
    age = _get_player_age(player)
    if age < 30:
        profile = "graceful_aging"
    else:
        wear = 0.0
        health = getattr(player, "health", None)
        if health is not None:
            wear = _safe_float(getattr(health, "wear_and_tear", 0), 0)
        morale = _get_player_morale(player)
        gp = _get_games_played(player)
        chem = getattr(player, "chemistry_profile", None)
        leadership = 0.5
        if isinstance(chem, dict):
            leadership = _safe_float(chem.get("leadership", 50), 50) / 100.0

        inj_hist = []
        if health is not None and hasattr(health, "injury_history"):
            inj_hist = getattr(health, "injury_history", []) or []

        if age >= 36 and wear > 0.45:
            profile = "hard_decline"
        elif len(inj_hist) >= 3 or wear > 0.55:
            profile = "injury_decline"
        elif _is_goalie(player) and age >= 33:
            profile = "graceful_aging"
        elif leadership >= 0.72 and morale >= 0.55 and age >= 34:
            profile = "mentor_veteran"
        elif wear > 0.35 or gp > 75:
            profile = "physical_decline"
        elif age >= 33:
            profile = "skating_decline"
        else:
            profile = "graceful_aging"

    _safe_setattr(player, "decline_profile", profile)
    return profile


# ---------------------------------------------------------------------------
# Trend labels & recommendations
# ---------------------------------------------------------------------------

def _compute_development_trend(
    player: Any,
    dev_phase: str,
    net_growth: float,
    career_stage: str,
    dev_type: str,
) -> str:
    phase = str(dev_phase or "NORMAL").upper()
    bust_p = _safe_float(getattr(player, "_bust_pressure", 0), 0)
    if phase == "SPIKE" or net_growth >= 1.2:
        return "Breakout"
    if phase == "REGRESSION" or net_growth <= -0.8:
        return "Declining"
    if phase == "STALL":
        return "Stalled"
    if net_growth >= 0.35:
        return "Rising"
    if dev_type == "late_bloomer" and career_stage in (STAGE_YOUNG_NHL, STAGE_PROSPECT):
        return "Rising"
    if bust_p >= 0.5:
        return "Regression Risk"
    return "Stable"


def generate_development_recommendation(
    player: Any,
    career_stage: str,
    readiness: float,
    fit_score: float,
    trend: str,
    decline_profile: Optional[str] = None,
) -> str:
    age = _get_player_age(player)
    gp = _get_games_played(player)
    role = _get_player_role(player)
    fit_label = str(getattr(player, "development_fit_label", "") or "")
    goalie = _is_goalie(player)

    if trend in ("Declining", "Regression Risk"):
        return "Monitor regression risk"
    if fit_label == "Being Rushed" or (age <= 21 and gp >= 35 and readiness < 50):
        return "Sheltered NHL minutes recommended"
    if fit_label in ("Needs Bigger Role", "Stalled Risk"):
        return "Needs bigger role"
    if career_stage == STAGE_PROSPECT and readiness < 55:
        return "Keep in AHL top role"
    if readiness >= 72 and trend in ("Rising", "Breakout"):
        return "Ready for NHL call-up"
    if career_stage in (STAGE_PRIME, STAGE_EMERGING_CORE) and "power" in _playstyle_bucket(player):
        return "Increase special teams usage"
    if decline_profile in ("physical_decline", "injury_decline", "hard_decline") and age >= 32:
        return "Reduce veteran workload"
    if decline_profile == "mentor_veteran":
        return "Assign mentor role"
    if goalie and age < 23 and readiness < 60:
        return "Do not rush"
    if fit_label == "Wrong Level":
        return "Keep in AHL top role"
    return "Monitor progression"


def _report_headline(
    dev_phase: str,
    trend: str,
    career_stage: str,
    fit_label: str,
    readiness: float,
) -> str:
    phase = str(dev_phase or "NORMAL").upper()
    if phase == "SPIKE" or trend == "Breakout":
        return "AHL breakout has improved NHL readiness."
    if fit_label == "Being Rushed":
        return "Being rushed into NHL minutes too early."
    if fit_label == "Needs Bigger Role":
        return "Strong fit, but needs bigger offensive role."
    if trend == "Stalled" or phase == "STALL":
        return "Development stalled this cycle."
    if career_stage in (STAGE_PRIME, STAGE_EMERGING_CORE) and phase == "NORMAL":
        return "Prime player refined two-way details."
    if trend == "Declining":
        return "Veteran skating decline risk increasing."
    if readiness >= 70:
        return "NHL readiness trending up."
    return "Steady development year."


def _attach_structured_dev_report(
    player: Any,
    *,
    report_type: str,
    dev_phase: str,
    trend: str,
    career_stage: str,
    readiness: float,
    fit_score: float,
    fit_label: str,
    decline_profile: Optional[str],
    reason: str,
    recommendation: str,
) -> None:
    headline = _report_headline(dev_phase, trend, career_stage, fit_label, readiness)
    report = {
        "type": report_type,
        "headline": headline,
        "trend": trend,
        "career_stage": career_stage,
        "nhl_readiness": round(readiness, 1),
        "development_fit_score": round(fit_score, 1),
        "development_fit_label": fit_label,
        "decline_profile": decline_profile,
        "reason": reason,
        "recommendation": recommendation,
    }
    _safe_setattr(player, "_dev_report_pending", report)
    _safe_setattr(player, "development_trend", trend)
    _safe_setattr(player, "career_stage", career_stage)
    _safe_setattr(player, "recommendation", recommendation)

    pname = (
        getattr(player, "name", None)
        or getattr(getattr(player, "identity", None), "name", None)
        or "Player"
    )
    line = (
        f"PROSPECT DEVELOPMENT REPORT: {pname} trend={trend} stage={career_stage} "
        f"readiness={readiness:.0f} fit={fit_label} phase={dev_phase} — {headline}"
    )
    _safe_setattr(player, "_dev_report_pending_line", line)


# ---------------------------------------------------------------------------
# Legacy helpers (preserved for existing callers)
# ---------------------------------------------------------------------------

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
    ph = career_phase_for_age(_get_player_age(player))
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


def _clamp_rating(x: float, lo: int = 20, hi: int = 99) -> int:
    return int(max(lo, min(hi, round(x))))


def _ovr(player: Any) -> float:
    return _get_player_ovr(player)


def _age(player: Any) -> int:
    return _get_player_age(player)


def _morale(player: Any) -> float:
    return _get_player_morale(player)


def _potential(player: Any) -> float:
    return _get_player_potential(player)


def _development_rate(player: Any) -> float:
    rate = getattr(player, "development_rate", None)
    if rate is not None:
        return float(rate)
    career = getattr(player, "career", None)
    if career is not None and hasattr(career, "breakout_probability"):
        return 0.4 + 0.3 * float(career.breakout_probability)
    return 0.5


def _games_played(player: Any) -> int:
    return _get_games_played(player)


def _safe_attr_float(player: Any, keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        try:
            v = getattr(player, k, None)
        except Exception:
            v = None
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return float(default)


def _safe_attr_int(player: Any, keys: List[str], default: int = 0) -> int:
    for k in keys:
        try:
            v = getattr(player, k, None)
        except Exception:
            v = None
        if v is None:
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    return int(default)


def _assign_development_window(player: Any, rng: random.Random) -> str:
    cur = str(getattr(player, "_dev_window_type", "") or "").strip().lower()
    if cur:
        return cur
    dt = _get_player_dev_type(player)
    p = _potential(player)
    if dt == "late_bloomer":
        window = "late_bloomer"
    elif dt in ("elite", "steal") and p >= 0.86:
        window = rng.choices(["early_developer", "flash_prospect", "normal_developer"], [0.42, 0.22, 0.36], k=1)[0]
    elif dt == "slow":
        window = rng.choices(["long_project", "late_bloomer", "normal_developer"], [0.48, 0.30, 0.22], k=1)[0]
    else:
        window = rng.choices(
            ["early_developer", "normal_developer", "late_bloomer", "long_project", "raw_talent"],
            [0.20, 0.37, 0.22, 0.14, 0.07],
            k=1,
        )[0]
    setattr(player, "_dev_window_type", window)
    return window


def _development_window_multiplier(age: int, window: str) -> float:
    w = str(window or "").lower()
    if w == "early_developer":
        if age <= 21:
            return 1.18
        if age <= 24:
            return 1.0
        return 0.84
    if w == "normal_developer":
        if age <= 19:
            return 0.9
        if age <= 24:
            return 1.08
        return 0.92
    if w == "late_bloomer":
        if age <= 21:
            return 0.82
        if age <= 27:
            return 1.16
        return 0.94
    if w == "long_project":
        if age <= 20:
            return 0.88
        if age <= 28:
            return 1.07
        return 0.92
    if w == "flash_prospect":
        if age <= 21:
            return 1.22
        if age <= 24:
            return 0.96
        return 0.78
    if w == "raw_talent":
        if age <= 21:
            return 0.74
        if age <= 26:
            return 1.13
        return 0.94
    return 1.0


def _update_career_momentum(player: Any, *, net_growth: float, dev_phase: str) -> float:
    cur = float(getattr(player, "_career_momentum", 0.0) or 0.0)
    gp = _games_played(player)
    role = str(getattr(player, "role", "") or "").lower()
    injury_days = _safe_attr_int(player, ["injury_days", "injured_days", "_injury_days"], 0)
    role_stability = _safe_attr_float(player, ["role_stability", "_role_stability"], 0.5)
    toi_q = _safe_attr_float(player, ["toi_quality", "_toi_quality"], 0.5)
    scratches = _safe_attr_int(player, ["healthy_scratches", "_healthy_scratches"], 0)
    perf = _safe_attr_float(player, ["recent_performance_score", "points_signal", "production"], 0.5)

    delta = 0.0
    delta += max(-6.0, min(6.0, net_growth * 7.0))
    if dev_phase == "SPIKE":
        delta += 8.0
    elif dev_phase == "REGRESSION":
        delta -= 7.0
    elif dev_phase == "STALL":
        delta -= 3.0
    delta += max(-4.0, min(4.0, (perf - 0.5) * 12.0))
    delta += max(-3.0, min(3.0, (toi_q - 0.5) * 8.0))
    delta += max(-3.0, min(3.0, (role_stability - 0.5) * 8.0))
    if gp >= 70:
        delta += 1.6
    elif gp < 28:
        delta -= 2.6
    if scratches >= 15:
        delta -= 3.0
    if injury_days >= 35:
        delta -= 3.5
    if any(x in role for x in ("top_line", "top_4", "elite", "starter")):
        delta += 1.8
    elif any(x in role for x in ("depth", "scratch", "press", "fourth")):
        delta -= 1.6

    nxt = max(-100.0, min(100.0, cur * 0.92 + delta))
    setattr(player, "_career_momentum", nxt)
    return nxt


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
    n_teams = max(1, len(teams))
    players: List[Any] = []
    pos_counts = {"C": 0, "D": 0, "G": 0}
    tier_counts = {"85+": 0, "80+": 0, "75+": 0}
    for tm in teams:
        for p in getattr(tm, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            players.append(p)
            ovr100 = _get_player_ovr_0_100(p)
            if ovr100 >= 85.0:
                tier_counts["85+"] += 1
            if ovr100 >= 80.0:
                tier_counts["80+"] += 1
            if ovr100 >= 75.0:
                tier_counts["75+"] += 1
            pos = _get_player_position(p)
            if pos in ("C",):
                pos_counts["C"] += 1
            elif pos in ("D", "LD", "RD"):
                pos_counts["D"] += 1
            elif pos in ("G",):
                pos_counts["G"] += 1

    # Soft ecosystem correction (not forced balancing).
    # If the league is talent-starved, increase growth opportunity slightly.
    # If overloaded, tighten growth conversion and raise congestion.
    t85 = 2.0 * n_teams
    t80 = 11.0 * n_teams
    t75 = 20.0 * n_teams
    d85 = (t85 - float(tier_counts["85+"])) / max(1.0, t85)
    d80 = (t80 - float(tier_counts["80+"])) / max(1.0, t80)
    d75 = (t75 - float(tier_counts["75+"])) / max(1.0, t75)
    scarcity_index = 0.42 * d85 + 0.38 * d80 + 0.20 * d75
    scarcity_index = max(-0.35, min(0.35, scarcity_index))
    # Stronger overload damping — prevents decade-long mean-OVR inflation when
    # the league already has too many 80+ skaters (soak OVR_DRIFT).
    league_growth_mult = max(0.82, min(1.08, 1.0 + 0.24 * scarcity_index))
    league_var_mult = max(0.88, min(1.10, 1.0 + 0.14 * scarcity_index))

    # Positional opportunity pressure (scarcity -> easier NHL runway).
    targets = {
        "C": 4.1 * n_teams,
        "D": 7.4 * n_teams,
        "G": 2.4 * n_teams,
    }
    pos_opportunity_mult = {}
    for k, tgt in targets.items():
        gap = (tgt - float(pos_counts.get(k, 0))) / max(1.0, tgt)
        gap = max(-0.45, min(0.45, gap))
        pos_opportunity_mult[k] = max(0.90, min(1.12, 1.0 + 0.14 * gap))

    for team in teams:
        tid_v = getattr(team, "team_id", None)
        if tid_v is None:
            tid_v = getattr(team, "id", None)
        tid = str(tid_v) if tid_v is not None else ""
        if tid == "":
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
            # Contenders can still develop young players (coaching/linemates).
            # Opportunity vs environment is handled per-player below — do not
            # globally tax every rostered prospect with 0.925.
            g_mult *= 1.0
            v_mult *= 1.06
        g_mult *= 0.76 + 0.48 * pscore
        g_mult *= league_growth_mult
        v_mult *= league_var_mult
        g_mult = max(0.72, min(1.36, g_mult))
        v_mult = max(0.78, min(1.34, v_mult))
        roster = getattr(team, "roster", None) or []
        org_profile = "balanced"
        if window == "rebuild" and pscore >= 0.55:
            org_profile = "opportunity_driven"
        elif window == "contender" and pscore < 0.52:
            org_profile = "win_now_congested"
        elif pscore >= 0.62:
            org_profile = "development_strong"
        setattr(team, "org_development_profile", org_profile)
        top_sk = sorted(
            [x for x in roster if not getattr(x, "retired", False)],
            key=lambda x: _get_player_ovr_0_100(x),
            reverse=True,
        )[:5]
        star_count = sum(1 for x in top_sk if _get_player_ovr_0_100(x) >= 90.0)
        superstar_factor = max(0.0, min(1.0, star_count / 2.0))
        for p in roster:
            if getattr(p, "retired", False):
                continue
            ctx = getattr(p, "context", None)
            cid = str(getattr(ctx, "current_team_id", "") or "") if ctx is not None else ""
            if cid != tid:
                continue
            pos = _get_player_position(p)
            pos_key = "C" if pos == "C" else ("D" if pos in ("D", "LD", "RD") else ("G" if pos == "G" else ""))
            pos_mult = float(pos_opportunity_mult.get(pos_key, 1.0))
            teammate_boost = 1.0
            if pos_key in ("C", "D") and superstar_factor > 0:
                teammate_boost += 0.03 * superstar_factor
            elif pos_key not in ("G",) and superstar_factor > 0:
                teammate_boost += 0.05 * superstar_factor
            # Young players on contenders: coaching/linemates help, not hurt.
            age_p = _get_player_age(p)
            young_contender = 1.0
            if window == "contender" and age_p <= 23:
                young_contender = 1.06
            elif window == "contender" and age_p <= 26:
                young_contender = 1.02
            # Opportunity proxy from ice-time / role (separate from org environment).
            opp = _clamp(_ice_time_modifier(p), 0.70, 1.20)
            setattr(p, "_dev_env_team_window", window)
            setattr(p, "_dev_opportunity_score", round(float(opp), 3))
            setattr(
                p,
                "_dev_env_growth_mult",
                max(0.70, min(1.42, g_mult * pos_mult * teammate_boost * young_contender * (0.92 + 0.10 * (opp - 0.85)))),
            )
            setattr(p, "_dev_env_variance_mult", max(0.76, min(1.38, v_mult * (1.0 + (pos_mult - 1.0) * 0.55))))
            setattr(p, "_org_development_profile", org_profile)


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
    # Goalies: higher variance in development phases
    if age >= 18 and arch in ("HIGH_VARIANCE", "ELITE_CEILING_VOLATILE", "LATE_BLOOMER"):
        spike += 0.012
        reg += 0.008
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
    """Higher role / opportunity = higher development runway."""
    gp = _games_played(player)
    role = getattr(player, "role", None) or ""
    role_low = str(role).lower()
    if "elite" in role_low or "top_line" in role_low or "top_4" in role_low:
        base = 1.2
    elif "middle" in role_low:
        base = 1.0
    else:
        base = 0.85
    toi_q = _safe_attr_float(player, ["toi_quality", "_toi_quality", "avg_toi_quality"], 0.5)
    pp_u = _safe_attr_float(player, ["pp_usage", "_pp_usage"], 0.0)
    pk_u = _safe_attr_float(player, ["pk_usage", "_pk_usage"], 0.0)
    scratches = _safe_attr_int(player, ["healthy_scratches", "_healthy_scratches"], 0)
    role_stability = _safe_attr_float(player, ["role_stability", "_role_stability"], 0.5)
    base *= max(0.78, min(1.24, 0.82 + toi_q * 0.36))
    base *= max(0.86, min(1.16, 0.92 + (pp_u * 0.16 + pk_u * 0.08)))
    base *= max(0.80, min(1.12, 0.88 + role_stability * 0.24))
    if scratches >= 10:
        base *= 0.90
    if scratches >= 20:
        base *= 0.84
    if gp >= 70:
        return base * 1.0
    if gp >= 50:
        return base * 0.9
    if gp >= 30:
        return base * 0.75
    return base * 0.6


def _finalize_development_metadata(
    player: Any,
    *,
    dev_phase: str,
    net_growth: float,
    archetype: str,
    report_type: str,
    reason: str,
) -> None:
    """Attach scores, trend, structured report — safe even when no rating growth occurred."""
    career_stage = determine_development_career_stage(player)
    _safe_setattr(player, "development_career_stage", career_stage)
    readiness = calculate_nhl_readiness_score(player)
    fit_score = calculate_development_fit_score(player)
    fit_label = str(getattr(player, "development_fit_label", "") or "Good Fit")
    decline_profile = calculate_age_decline_profile(player)
    dev_type = _get_player_dev_type(player)
    trend = _compute_development_trend(player, dev_phase, net_growth, career_stage, dev_type)
    recommendation = generate_development_recommendation(
        player, career_stage, readiness, fit_score, trend, decline_profile
    )
    fail_reason = str(getattr(player, "_development_failure_reason", "") or "")
    if not fail_reason and dev_type == "bust":
        fail_reason = "translation_risk_bust_path"
    if fail_reason:
        reason = f"{reason} fail_reason={fail_reason}"
    _attach_structured_dev_report(
        player,
        report_type=report_type,
        dev_phase=dev_phase,
        trend=trend,
        career_stage=career_stage,
        readiness=readiness,
        fit_score=fit_score,
        fit_label=fit_label,
        decline_profile=decline_profile if career_stage in (STAGE_VETERAN, STAGE_DECLINING, STAGE_LATE_CAREER) else None,
        reason=reason,
        recommendation=recommendation,
    )


# --- Authoritative development profile + bounded seasonal growth ---------------

def resolve_development_profile(player: Any, context: Optional[Any] = None) -> Dict[str, Any]:
    """
    Resolve a backward-compatible development profile (0–1 scale).

    Priority: existing profile → explicit true/ceiling fields →
    tools/production/age/archetype → controlled fallback gap → safe default.
    Aliases player.potential to expected_ceiling.
    """
    from app.sim_engine.entities.player import (
        clamp01,
        normalize_rating,
        player_current_ovr_01,
    )

    current_ovr = clamp01(float(player_current_ovr_01(player)))
    age = _get_player_age(player)
    goalie = _is_goalie(player)
    arch = str(
        getattr(player, "_dev_archetype", "")
        or getattr(player, "archetype", "")
        or ""
    ).upper()
    dev_type = _get_player_dev_type(player)

    def _finalize(
        expected: float,
        maximum: float,
        *,
        development_rate: float,
        volatility: float,
        bust_risk: float,
        breakout_chance: float,
        decline_state: str,
        source: str,
        base: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        expected = clamp01(float(expected))
        maximum = clamp01(max(float(maximum), expected))
        if expected < current_ovr:
            expected = current_ovr
            maximum = max(maximum, expected)
        # Never let a sticky/inferred profile erase a higher true potential on the player.
        ratings = getattr(player, "ratings", None)
        for raw_pot in (
            getattr(player, "true_potential", None),
            getattr(player, "true_potential_score", None),
            getattr(player, "potential", None),
            (ratings.get("dev_potential") if isinstance(ratings, dict) else None),
            (ratings.get("true_potential") if isinstance(ratings, dict) else None),
        ):
            if raw_pot is None:
                continue
            try:
                pot01 = normalize_rating(raw_pot)
            except Exception:
                continue
            if pot01 > expected + 0.005:
                expected = pot01
                maximum = max(maximum, clamp01(min(0.99, expected + 0.03)))
                break
        profile: Dict[str, Any] = dict(base or {})
        profile.update(
            {
                "current_ovr": current_ovr,
                "expected_ceiling": expected,
                "maximum_ceiling": maximum,
                "development_rate": _clamp(float(development_rate), 0.15, 0.95),
                "volatility": _clamp(float(volatility), 0.05, 0.95),
                "bust_risk": _clamp(float(bust_risk), 0.0, 0.95),
                "breakout_chance": _clamp(float(breakout_chance), 0.0, 0.85),
                "decline_state": str(decline_state or "none"),
                "source": source,
            }
        )
        try:
            setattr(player, "development_profile", profile)
        except Exception:
            pass
        try:
            setattr(player, "potential", float(expected))
        except Exception:
            pass
        if isinstance(ratings, dict):
            from app.sim_engine.entities.player import display_rating

            ratings["dev_potential"] = float(display_rating(expected))
            if ratings.get("dev_ceiling") is None:
                ratings["dev_ceiling"] = float(display_rating(maximum))
        return profile

    def _decline_state_for(age_v: int) -> str:
        if age_v >= 36:
            return "late_decline"
        if age_v >= 33:
            return "early_decline"
        if age_v >= 30:
            return "aging"
        return "none"

    # 1) Existing valid profile
    existing = getattr(player, "development_profile", None)
    if isinstance(existing, dict) and existing.get("expected_ceiling") is not None:
        expected = normalize_rating(existing.get("expected_ceiling"))
        maximum = normalize_rating(
            existing.get("maximum_ceiling", max(expected, current_ovr + 0.03))
        )
        return _finalize(
            expected,
            maximum,
            development_rate=_safe_float(
                existing.get("development_rate"), _development_rate(player)
            ),
            volatility=_safe_float(existing.get("volatility"), 0.35),
            bust_risk=_safe_float(
                existing.get("bust_risk"),
                _safe_float(getattr(player, "_bust_pressure", 0.12), 0.12),
            ),
            breakout_chance=_safe_float(
                existing.get("breakout_chance"),
                _safe_float(getattr(player, "_steal_momentum", 0.08), 0.08),
            ),
            decline_state=str(existing.get("decline_state") or _decline_state_for(age)),
            source="existing_profile",
            base=existing,
        )

    # 2) Explicit true / ceiling fields on player or ratings
    explicit_expected = None
    explicit_maximum = None
    for key in (
        "expected_ceiling",
        "true_potential",
        "true_potential_score",
        "potential",
        "dev_potential",
    ):
        raw = getattr(player, key, None)
        if raw is None and isinstance(getattr(player, "ratings", None), dict):
            raw = player.ratings.get(key)
        if raw is not None:
            try:
                explicit_expected = normalize_rating(raw)
                break
            except Exception:
                continue
    for key in ("maximum_ceiling", "dev_ceiling", "true_ceiling", "ceiling"):
        raw = getattr(player, key, None)
        if raw is None and isinstance(getattr(player, "ratings", None), dict):
            raw = player.ratings.get(key)
        if raw is not None:
            try:
                explicit_maximum = normalize_rating(raw)
                break
            except Exception:
                continue
    if explicit_expected is not None:
        maximum = (
            explicit_maximum
            if explicit_maximum is not None
            else clamp01(min(0.98, explicit_expected + 0.04))
        )
        return _finalize(
            explicit_expected,
            maximum,
            development_rate=_development_rate(player),
            volatility=0.38 if arch in ("HIGH_VARIANCE", "ELITE_CEILING_VOLATILE") else 0.28,
            bust_risk=0.22 if dev_type == "bust" else _safe_float(getattr(player, "_bust_pressure", 0.12), 0.12),
            breakout_chance=0.18 if dev_type in ("elite", "steal") else 0.08,
            decline_state=_decline_state_for(age),
            source="explicit_fields",
        )

    # 3) Infer from tools / production / age / archetype (bounded gaps)
    tools = _safe_attr_float(
        player,
        ["tools_score", "tool_score", "athletic_tools", "raw_tools", "_tools_score"],
        -1.0,
    )
    production = _safe_attr_float(
        player,
        ["recent_performance_score", "points_signal", "production", "production_score"],
        0.5,
    )
    headroom = 0.055
    if tools >= 0.0:
        headroom += _clamp((tools - 0.5) * 0.08, -0.02, 0.05)
    headroom += _clamp((production - 0.5) * 0.04, -0.02, 0.03)
    if goalie:
        if age <= 21:
            headroom += 0.035
        elif age <= 24:
            headroom += 0.025
        elif age <= 27:
            headroom += 0.012
        else:
            headroom *= 0.35
    else:
        if age <= 19:
            headroom += 0.04
        elif age <= 21:
            headroom += 0.03
        elif age <= 24:
            headroom += 0.018
        elif age <= 26:
            headroom += 0.008
        else:
            headroom *= 0.3

    if arch in ("ELITE_CEILING_VOLATILE", "HIGH_VARIANCE"):
        headroom += 0.025
    elif arch == "FAST_RISER":
        headroom += 0.015
    elif arch == "SAFE_LOW_CEILING":
        headroom -= 0.015
    elif arch == "STALLED_DEVELOPER":
        headroom -= 0.02
    elif arch == "LATE_BLOOMER" and age >= 22:
        headroom += 0.02

    if dev_type == "bust":
        headroom *= 0.55
    elif dev_type in ("elite", "steal"):
        headroom *= 1.12
    elif dev_type == "slow":
        headroom *= 0.78

    # Allow real NHL-scale runway (up to ~18 OVR) when tools/age support it.
    # Higher ceilings still come from explicit potential / profile fields.
    headroom = _clamp(headroom, 0.015, 0.18)
    expected = clamp01(current_ovr + headroom)
    maximum = clamp01(min(0.98, expected + _clamp(0.02 + 0.03 * max(0.0, tools if tools >= 0 else 0.35), 0.015, 0.06)))
    vol = 0.42 if arch in ("HIGH_VARIANCE", "ELITE_CEILING_VOLATILE") else 0.30
    if tools >= 0.7:
        vol += 0.08
    return _finalize(
        expected,
        maximum,
        development_rate=_development_rate(player),
        volatility=vol,
        bust_risk=0.28 if dev_type == "bust" else 0.12 + (0.08 if arch == "STALLED_DEVELOPER" else 0.0),
        breakout_chance=0.16 if tools >= 0.65 or dev_type in ("elite", "steal") else 0.07,
        decline_state=_decline_state_for(age),
        source="inferred_tools_production",
    )


def calculate_season_growth_budget(
    player: Any,
    context: Optional[Any] = None,
    profile: Optional[Dict[str, Any]] = None,
    *,
    rng: Any = None,
    dev_phase: str = "NORMAL",
) -> float:
    """
    Bounded seasonal growth on the 0–1 OVR scale.

    Budgets are intended display-OVR targets (budget * 99 ≈ ΔOVR). Downstream
    ensure_displayed_ovr_delta corrects attribute dilution so the visible result
    lands near this target.
    """
    from app.sim_engine.entities.player import clamp01, normalize_rating_gap

    if not isinstance(profile, dict):
        profile = resolve_development_profile(player, context)
    if not isinstance(rng, random.Random):
        rng = random.Random()

    current = float(profile.get("current_ovr", _get_player_ovr(player)))
    expected = float(profile.get("expected_ceiling", current))
    maximum = float(profile.get("maximum_ceiling", max(expected, current)))
    gap_exp = float(normalize_rating_gap(current, expected))
    gap_max = float(normalize_rating_gap(current, maximum))

    # Meaningful season jumps for runway kids (display OVR ≈ budget * 99).
    # Age 18–20 with 10+ OVR gap: approach toward +4–6 before phase/noise.
    approach = min(0.072, gap_exp * 0.42 + 0.024)
    if gap_exp <= 0.012:
        approach = min(0.028, max(0.010, gap_max * 0.10 + 0.008))

    age = _get_player_age(player)
    morale = _get_player_morale(player)
    ice_mod = _ice_time_modifier(player)
    ice_mod = _clamp(ice_mod, 0.55, 1.22)
    scratches = _safe_attr_int(player, ["healthy_scratches", "_healthy_scratches"], 0)
    if scratches >= 20:
        ice_mod *= 0.55
    elif scratches >= 10:
        ice_mod *= 0.78

    rate = _clamp(float(profile.get("development_rate", _development_rate(player))), 0.15, 0.95)
    vol = _clamp(float(profile.get("volatility", 0.3)), 0.05, 0.95)
    env_g = _clamp(_safe_float(getattr(player, "_dev_env_growth_mult", 1.0), 1.0), 0.72, 1.28)
    nar = _clamp(_safe_float(getattr(player, "_narrative_prog_growth_mult", 1.0), 1.0), 0.75, 1.25)
    momentum = _clamp(_safe_float(getattr(player, "_dev_breakout_momentum", 0.0), 0.0), 0.0, 1.0)

    mod = 1.0
    mod *= 0.78 + 0.48 * morale
    mod *= ice_mod
    mod *= 0.70 + 0.70 * rate
    mod *= env_g
    mod *= nar
    mod *= 1.0 + 0.18 * momentum

    if age < 20:
        mod *= 1.38 if not _is_goalie(player) else 1.22
    elif age < 22:
        mod *= 1.26 if not _is_goalie(player) else 1.18
    elif age < 24:
        mod *= 1.14
    elif age <= 26:
        mod *= 1.04
    elif age <= 28:
        mod *= 0.88 if not _is_goalie(player) else 0.96
    elif age <= 31:
        mod *= 0.62
    else:
        mod *= 0.34

    if age <= 20 and gap_exp >= 0.10:
        mod *= 1.16
    elif age <= 23 and gap_exp >= 0.07:
        mod *= 1.10
    elif age <= 24 and gap_exp >= 0.06:
        mod *= 1.06
    elif age <= 26 and gap_exp >= 0.08:
        mod *= 1.04

    role = str(
        getattr(context, "role", None)
        if context is not None and getattr(context, "role", None) is not None
        else getattr(player, "role", "")
        or ""
    ).lower()
    if any(k in role for k in ("elite", "top_line", "top_4", "starter", "first")):
        mod *= 1.14
    elif any(k in role for k in ("second", "line2", "top_6")):
        mod *= 1.08
    elif any(k in role for k in ("fourth", "depth", "scratch", "press")):
        mod *= 0.88

    production = _safe_attr_float(
        player,
        ["recent_performance_score", "points_signal", "production", "production_score"],
        0.5,
    )
    # Overperformance vs current OVR: low-rated high-prod kids get extra runway.
    ovr100 = current * 99.0 if current <= 1.5 else current
    expected_prod = _clamp(0.28 + (ovr100 - 70.0) * 0.012, 0.30, 0.92)
    overperf = _clamp((production - expected_prod) * 1.35, -0.25, 0.45)
    mod *= _clamp(0.88 + (production - 0.5) * 0.42 + overperf * 0.55, 0.72, 1.42)

    injury_days = _safe_attr_int(player, ["injury_days", "injured_days", "_injury_days"], 0)
    if injury_days >= 40:
        mod *= 0.62
    elif injury_days >= 20:
        mod *= 0.78

    phase = str(dev_phase or "NORMAL").upper()
    noise = 1.0 + rng.uniform(-0.10, 0.14) * vol
    budget = approach * mod * noise

    if phase == "STALL":
        budget *= rng.uniform(0.28, 0.52) if gap_exp >= 0.06 else rng.uniform(0.10, 0.28)
    elif phase == "SPIKE":
        budget *= rng.uniform(1.55, 2.20)
        if gap_max > gap_exp and rng.random() < float(profile.get("breakout_chance", 0.08)):
            budget = min(0.11, budget + min(0.035, (gap_max - gap_exp) * 0.28))
    elif phase == "REGRESSION":
        budget = -abs(max(0.008, min(0.035, approach * rng.uniform(0.55, 1.15))))
    else:
        budget *= 0.95 + 0.12 * min(1.0, env_g)

    if budget > 0:
        # Allow strong / breakout years to hit design targets (+5–10 display).
        hard_cap = 0.110 if phase == "SPIKE" else 0.085
        gap_cap = max(0.028, gap_exp * 0.78 + 0.022) if gap_exp > 0 else 0.028
        # Near expected but below maximum: still allow breakout polish into max.
        if gap_exp <= 0.02 and gap_max >= 0.03:
            gap_cap = max(gap_cap, min(0.055, gap_max * 0.85 + 0.015))
        budget = min(budget, hard_cap, gap_cap)

        # Elite runway floors — +2 should not be the default for franchise kids.
        if age <= 20 and gap_exp >= 0.10 and phase in ("NORMAL", "SPIKE"):
            budget = max(budget, 0.045 if phase == "NORMAL" else 0.065)
        elif age <= 23 and gap_exp >= 0.07 and phase in ("NORMAL", "SPIKE"):
            budget = max(budget, 0.036 if phase == "NORMAL" else 0.055)
        elif gap_exp >= 0.04 and phase in ("NORMAL", "SPIKE"):
            budget = max(budget, 0.030 if phase == "NORMAL" else 0.045)
        elif gap_exp >= 0.06 and phase == "STALL":
            budget = max(budget, 0.016)
    else:
        budget = max(budget, -0.038)

    decline = str(profile.get("decline_state") or "none")
    if decline != "none" and budget > 0:
        budget *= 0.50 if decline == "late_decline" else 0.70

    return float(budget)


# Separate mid-season vs season-end pools (design §11).
_IN_SEASON_POOL_SHARE = 0.32
_SEASON_END_POOL_SHARE = 0.58


_META_RATING_KEYS = frozenset(
    {
        "dev_potential",
        "dev_ceiling",
        "potential",
        "overall",
        "ovr",
        "true_potential",
        "true_ceiling",
        "_generated_profile",
        "generated_profile",
    }
)


def allocate_growth_to_attributes(
    player: Any,
    growth_budget: float,
    role: Any = None,
    archetype: Any = None,
    phase: str = "NORMAL",
) -> Dict[str, float]:
    """Distribute seasonal growth budget (0–1 OVR scale) across attribute keys."""
    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict) or not ratings:
        return {}

    _ = role

    skill_keys = [k for k in ratings.keys() if str(k).lower() not in _META_RATING_KEYS]
    if not skill_keys:
        return {}

    full_ratings = ratings
    prev_arch = getattr(player, "archetype", None)
    try:
        if archetype is not None:
            setattr(player, "archetype", archetype)
        setattr(player, "ratings", {k: full_ratings[k] for k in skill_keys})
        n = max(1, len(skill_keys))
        total_growth_scale = float(growth_budget) * 99.0
        phase_up = str(phase or "NORMAL").upper()
        if phase_up == "REGRESSION":
            total_growth_scale = max(-0.55 * n, min(-0.06 * n, total_growth_scale))
        else:
            # Allow larger concentrated sprays so display OVR can move.
            total_growth_scale = min(float(total_growth_scale), max(10.0, 0.45 * n))

        seed = distribute_growth_by_player_type(
            player, 1.0 if total_growth_scale >= 0 else -1.0, phase=phase_up
        )
        if not seed:
            seed = {k: 1.0 for k in skill_keys}

        # Concentrate on fewer archetype-weighted skills (design §7).
        ranked = sorted(seed.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        focus_n = max(4, min(8, len(ranked)))
        focus = dict(ranked[:focus_n])
        weight_sum = sum(abs(float(v)) for v in focus.values()) or float(focus_n)
        target = abs(float(total_growth_scale))
        sign = -1.0 if total_growth_scale < 0 or phase_up == "REGRESSION" else 1.0
        deltas = {
            k: sign * target * (abs(float(v)) / weight_sum) for k, v in focus.items()
        }

        if phase_up == "REGRESSION":
            for k in list(deltas.keys()):
                deltas[k] = max(-5.0, min(-0.5, float(deltas[k])))
        else:
            for k in list(deltas.keys()):
                deltas[k] = min(9.0, max(0.8 if sign > 0 else -1.5, float(deltas[k])))
        return deltas
    finally:
        setattr(player, "ratings", full_ratings)
        if archetype is not None:
            if prev_arch is None:
                try:
                    delattr(player, "archetype")
                except Exception:
                    setattr(player, "archetype", prev_arch)
            else:
                setattr(player, "archetype", prev_arch)


def ensure_displayed_ovr_delta(
    player: Any,
    *,
    ovr_before_01: float,
    target_display_delta: float,
    rng: Any,
    phase: str = "NORMAL",
    archetype: Any = None,
    tolerance: float = 0.55,
    max_iters: int = 10,
) -> Dict[str, float]:
    """
    After attribute spray, correct toward the intended displayed OVR change.

    Preserves attribute-based OVR while ensuring budgets mean visible growth.
    """
    from app.sim_engine.entities.player import persist_recomputed_ovr, player_current_ovr_01

    if not isinstance(rng, random.Random):
        rng = random.Random()
    target = float(target_display_delta)
    if abs(target) < 0.35:
        return {}

    corrective: Dict[str, float] = {}
    ratings = getattr(player, "ratings", None)
    if not isinstance(ratings, dict) or not ratings:
        return {}

    for _ in range(max(1, int(max_iters))):
        cur = float(player_current_ovr_01(player))
        actual = (cur - float(ovr_before_01)) * 99.0
        shortfall = target - actual
        if target >= 0 and shortfall <= tolerance:
            break
        if target < 0 and shortfall >= -tolerance:
            break
        chunk_disp = min(abs(shortfall), 3.2)
        chunk_01 = (chunk_disp / 99.0) * (1.0 if shortfall > 0 else -1.0)
        chunk_01 *= 1.55 if shortfall > 0 else 1.20
        phase_use = phase if shortfall > 0 else "REGRESSION"
        deltas = allocate_growth_to_attributes(
            player,
            chunk_01,
            role=str(getattr(player, "role", "") or ""),
            archetype=archetype,
            phase=phase_use,
        )
        if not deltas:
            break
        ranked = sorted(deltas.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:5]
        if not ranked:
            break
        wsum = sum(abs(float(v)) for _, v in ranked) or 1.0
        scale = abs(float(chunk_01)) * 99.0
        sign = 1.0 if shortfall > 0 else -1.0
        focused = {k: sign * scale * (abs(float(v)) / wsum) for k, v in ranked}
        applied = apply_attribute_deltas(player, focused)
        for k, v in applied.items():
            corrective[k] = float(corrective.get(k, 0.0)) + float(v)
        persist_recomputed_ovr(player)
        if not applied:
            break

    # Last resort: category-wide bump so display OVR actually lands near target.
    # Raising all skill attrs by ~S moves category averages (and OVR) by ~S.
    cur = float(player_current_ovr_01(player))
    actual = (cur - float(ovr_before_01)) * 99.0
    shortfall = target - actual
    if (target >= 0 and shortfall > tolerance) or (target < 0 and shortfall < -tolerance):
        skill_keys = [k for k in ratings.keys() if str(k).lower() not in _META_RATING_KEYS]
        if skill_keys:
            bump = min(abs(shortfall) * 1.08, 9.0) * (1.0 if shortfall > 0 else -1.0)
            flat = {k: bump for k in skill_keys}
            applied = apply_attribute_deltas(player, flat)
            for k, v in applied.items():
                corrective[k] = float(corrective.get(k, 0.0)) + float(v)
            persist_recomputed_ovr(player)
            # Fine-tune once more if still short (diminishing).
            cur2 = float(player_current_ovr_01(player))
            actual2 = (cur2 - float(ovr_before_01)) * 99.0
            short2 = target - actual2
            if abs(short2) > tolerance and skill_keys:
                bump2 = min(abs(short2) * 1.05, 5.0) * (1.0 if short2 > 0 else -1.0)
                applied2 = apply_attribute_deltas(player, {k: bump2 for k in skill_keys})
                for k, v in applied2.items():
                    corrective[k] = float(corrective.get(k, 0.0)) + float(v)
                persist_recomputed_ovr(player)

    return corrective


def reevaluate_ceilings_from_performance(player: Any, rng: Any) -> Dict[str, Any]:
    """
    Raise projected / active / maximum ceilings when overperformance evidence is strong.

    Potential is an evolving evaluation — not a fixed destination (design §3–5, §13).
    """
    from app.sim_engine.entities.player import clamp01, display_rating, normalize_rating, persist_recomputed_ovr

    if not isinstance(rng, random.Random):
        rng = random.Random()

    profile = resolve_development_profile(player)
    age = _get_player_age(player)
    if age > 29:
        return {"applied": False, "reason": "outside_window"}

    current = float(profile.get("current_ovr", _get_player_ovr(player)))
    expected = float(profile.get("expected_ceiling", current))
    maximum = float(profile.get("maximum_ceiling", max(expected, current + 0.03)))
    prod = _safe_attr_float(
        player,
        ["production_score", "recent_performance_score", "points_signal", "production"],
        0.5,
    )
    morale = _get_player_morale(player)
    ovr100 = current * 99.0 if current <= 1.5 else current
    expected_prod = _clamp(0.28 + (ovr100 - 70.0) * 0.012, 0.30, 0.92)
    overperf = prod - expected_prod
    momentum = _clamp(_safe_float(getattr(player, "_dev_breakout_momentum", 0.0), 0.0), 0.0, 1.0)

    # Store multi-season momentum from this season's evidence.
    mom_delta = 0.0
    if overperf >= 0.12 and morale >= 0.55:
        mom_delta = 0.18 + min(0.22, overperf * 0.55)
    elif overperf >= 0.05:
        mom_delta = 0.08
    elif overperf <= -0.12:
        mom_delta = -0.14
    new_mom = _clamp(momentum + mom_delta, 0.0, 1.0)
    try:
        setattr(player, "_dev_breakout_momentum", new_mom)
    except Exception:
        pass

    if overperf < 0.08 and new_mom < 0.35:
        return {"applied": False, "reason": "insufficient_evidence", "momentum": new_mom}

    # Projected potential moves more than maximum (design §13 / §19).
    proj_delta = 0.0
    active_delta = 0.0
    max_delta = 0.0
    if overperf >= 0.22 and new_mom >= 0.45 and age <= 24:
        proj_delta = rng.uniform(0.025, 0.045)  # ~+2.5–4.5 pot
        active_delta = proj_delta + rng.uniform(0.005, 0.015)
        max_delta = rng.uniform(0.005, 0.020)
        label = "breakout"
    elif overperf >= 0.14 or new_mom >= 0.55:
        proj_delta = rng.uniform(0.015, 0.030)
        active_delta = proj_delta
        max_delta = rng.uniform(0.0, 0.012) if new_mom >= 0.5 else 0.0
        label = "strong"
    elif overperf >= 0.08:
        proj_delta = rng.uniform(0.008, 0.018)
        active_delta = proj_delta * 0.85
        max_delta = 0.0
        label = "solid"
    else:
        return {"applied": False, "reason": "borderline", "momentum": new_mom}

    # Low-pot stars: allow climbing toward maximum even when near expected.
    new_expected = clamp01(expected + proj_delta)
    new_active = clamp01(max(new_expected, expected + active_delta))
    new_max = clamp01(min(0.99, maximum + max_delta))
    new_max = max(new_max, new_active)

    profile["expected_ceiling"] = new_expected
    profile["maximum_ceiling"] = new_max
    profile["active_development_ceiling"] = new_active
    profile["breakout_chance"] = _clamp(
        float(profile.get("breakout_chance", 0.08)) + (0.04 if label == "breakout" else 0.015),
        0.05,
        0.55,
    )
    try:
        setattr(player, "development_profile", profile)
        setattr(player, "potential", float(new_expected))
        ratings = getattr(player, "ratings", None)
        if isinstance(ratings, dict):
            ratings["dev_potential"] = float(display_rating(new_expected))
            ratings["dev_ceiling"] = float(display_rating(new_max))
    except Exception:
        pass

    # Soft profile reclassification signal for UI / storylines.
    try:
        if label == "breakout" and new_mom >= 0.5:
            setattr(player, "_dev_profile_reclass_pending", True)
            setattr(
                player,
                "_dev_profile_reclass_note",
                f"Overperformance raised projected potential "
                f"{display_rating(expected):.0f}→{display_rating(new_expected):.0f}",
            )
    except Exception:
        pass

    return {
        "applied": True,
        "label": label,
        "expected_before": expected,
        "expected_after": new_expected,
        "maximum_before": maximum,
        "maximum_after": new_max,
        "momentum": new_mom,
        "overperformance": round(overperf, 3),
    }


def apply_attribute_deltas(player: Any, attribute_deltas: Dict[str, float]) -> Dict[str, float]:
    """Apply attribute deltas in-place; return {key: applied_delta} on display scale."""
    ratings = getattr(player, "ratings", None)
    applied: Dict[str, float] = {}
    if not ratings or not isinstance(ratings, dict) or not attribute_deltas:
        return applied
    for k, d in attribute_deltas.items():
        if k not in ratings or str(k).lower() in _META_RATING_KEYS:
            continue
        before = _safe_float(ratings[k], 50)
        # Keep sub-point precision so small seasonal budgets accumulate.
        after = _clamp(before + float(d), 20.0, 99.0)
        delta = after - before
        if abs(delta) < 1e-9:
            continue
        ratings[k] = after
        applied[k] = round(delta, 4)
    if applied:
        inval = getattr(player, "_invalidate_ovr_memo", None)
        if callable(inval):
            try:
                inval()
            except Exception:
                pass
    return applied


def apply_player_development(player: Any, rng: Any) -> None:
    """
    Apply young-player growth via bounded attribute budgets and seasonal ledger.
    Does not mutate overall/ovr except through persist_recomputed_ovr.
    """
    from app.sim_engine.entities.player import (
        normalize_rating_gap,
        persist_recomputed_ovr,
        player_current_ovr_01,
    )
    from app.sim_engine.progression.potential import ensure_development_ledger

    if not isinstance(rng, random.Random):
        rng = random.Random()

    season_id = getattr(player, "_active_dev_season", None) or "default"
    ledger = ensure_development_ledger(player, season_id)
    if ledger.get("development_applied"):
        return

    def _mark_ledger(
        *,
        ovr_before: float,
        ovr_after: float,
        attribute_deltas: Optional[Dict[str, float]] = None,
        source_path: str = "apply_player_development",
    ) -> None:
        ledger["ovr_before"] = round(float(ovr_before), 6)
        ledger["ovr_after"] = round(float(ovr_after), 6)
        ledger["attribute_deltas"] = dict(attribute_deltas or {})
        ledger["source_path"] = source_path
        ledger["development_applied"] = True
        try:
            setattr(player, "development_ledger", ledger)
        except Exception:
            pass

    ovr_before = float(player_current_ovr_01(player))
    profile = resolve_development_profile(player)
    potential = float(profile.get("expected_ceiling", ovr_before))
    archetype = str(getattr(player, "_dev_archetype", "") or "").strip()
    if not archetype:
        archetype = _lazy_assign_dev_archetype(player, potential, rng)
        setattr(player, "_dev_archetype", archetype)

    dev_type = str(getattr(player, "dev_type", "standard") or "standard").lower()
    age = _get_player_age(player)
    career_stage = determine_development_career_stage(player)
    _safe_setattr(player, "development_career_stage", career_stage)
    calculate_nhl_readiness_score(player)
    update_player_nhl_eta(player)
    calculate_development_fit_score(player)
    calculate_age_decline_profile(player)
    apply_prime_refinement(player)

    growth_eligible = age <= 29
    if not growth_eligible:
        if _is_goalie(player) and 24 <= age <= 31 and (dev_type == "late_bloomer" or rng.random() <= 0.28):
            growth_eligible = True
        elif 29 <= age <= 32 and rng.random() <= 0.22:
            growth_eligible = True

    if not growth_eligible:
        ovr_after = float(persist_recomputed_ovr(player))
        _mark_ledger(
            ovr_before=ovr_before,
            ovr_after=ovr_after,
            attribute_deltas={},
            source_path="apply_player_development:outside_window",
        )
        _finalize_development_metadata(
            player,
            dev_phase=str(getattr(player, "_dev_last_phase", "NORMAL") or "NORMAL"),
            net_growth=0.0,
            archetype=archetype,
            report_type="decline"
            if career_stage in (STAGE_VETERAN, STAGE_DECLINING, STAGE_LATE_CAREER)
            else "prime_refinement",
            reason="Outside primary development window.",
        )
        return

    curve_hint = str(
        getattr(player, "_pipeline_dev_curve", "")
        or getattr(player, "_dev_curve_hint", "")
        or "normal"
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

    prev_momentum = float(getattr(player, "_career_momentum", 0.0) or 0.0)
    role = str(getattr(player, "role", "") or "").lower()
    if prev_momentum >= 70.0 and dev_phase != "REGRESSION" and rng.random() < 0.22:
        dev_phase = "SPIKE"
    elif prev_momentum <= -70.0 and dev_phase != "SPIKE" and rng.random() < 0.24:
        dev_phase = "STALL"

    # Strong NHL seasons should not roll a dead-stall year when runway remains.
    prod = _safe_attr_float(
        player,
        ["production_score", "recent_performance_score", "points_signal", "production"],
        0.5,
    )
    gap_now = float(normalize_rating_gap(ovr_before, potential))
    if age <= 26 and gap_now >= 0.04 and prod >= 0.78:
        if dev_phase == "STALL":
            dev_phase = "SPIKE" if prod >= 0.88 and rng.random() < 0.55 else "NORMAL"
        elif dev_phase == "NORMAL" and prod >= 0.88 and rng.random() < 0.42:
            dev_phase = "SPIKE"
    if age <= 24 and gap_now >= 0.08 and prod >= 0.85 and dev_phase == "REGRESSION":
        # Elite young producers rarely take a full regression year.
        if rng.random() < 0.65:
            dev_phase = "NORMAL"

    if dev_phase in ("NORMAL", "SPIKE") and age <= 27:
        breakout_p = float(profile.get("breakout_chance", 0.08)) * 0.35
        if any(k in role for k in ("top_line", "top_4", "second", "line2")):
            breakout_p += 0.01
        if prev_momentum >= 35.0:
            breakout_p += 0.01
        if rng.random() < min(0.08, breakout_p):
            dev_phase = "SPIKE"
            setattr(player, "_development_breakout_event", True)

    setattr(player, "_dev_last_phase", dev_phase)
    if adj > 0:
        setattr(player, "_nhl_adjustment_years_remaining", max(0, adj - 1))

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

    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict) or not ratings:
        ovr_after = float(persist_recomputed_ovr(player))
        _mark_ledger(
            ovr_before=ovr_before,
            ovr_after=ovr_after,
            attribute_deltas={},
            source_path="apply_player_development:no_ratings",
        )
        _finalize_development_metadata(
            player,
            dev_phase=dev_phase,
            net_growth=0.0,
            archetype=archetype,
            report_type="growth",
            reason="No ratings container on player.",
        )
        return

    budget = calculate_season_growth_budget(
        player, None, profile, rng=rng, dev_phase=dev_phase
    )
    # Soft stall when already at expected — still allow a small real step unless
    # the player is truly capped (gap ~0).
    gap_now = float(normalize_rating_gap(ovr_before, potential))
    if gap_now <= 0.004 and dev_phase not in ("REGRESSION", "SPIKE"):
        budget = min(max(budget, 0.012), 0.022)
    elif gap_now <= 0.02 and dev_phase == "NORMAL" and budget > 0:
        budget = max(budget, 0.018)

    # Season-end uses its own pool (65–75%). Mid-season pulses do NOT claw this back.
    if budget > 0:
        budget = float(budget) * float(_SEASON_END_POOL_SHARE)
    elif budget < 0:
        budget = float(budget) * float(_SEASON_END_POOL_SHARE)

    # Live OVR before this pass (for displayed-target correction). Mid-season gains
    # already accrued; season-end targets additional movement from the end pool.
    live_before = float(player_current_ovr_01(player))

    # Ledger "before" should reflect season-start OVR when available (cumulative report).
    try:
        start_disp = getattr(player, "season_start_ovr", None)
        if start_disp is not None:
            ovr_before = float(start_disp) / 99.0 if float(start_disp) > 1.5 else float(start_disp)
    except Exception:
        pass

    target_display = float(budget) * 99.0
    deltas = allocate_growth_to_attributes(
        player,
        budget,
        role=role,
        archetype=archetype,
        phase=dev_phase,
    )
    applied = apply_attribute_deltas(player, deltas)
    persist_recomputed_ovr(player)
    corrective = ensure_displayed_ovr_delta(
        player,
        ovr_before_01=live_before,
        target_display_delta=target_display,
        rng=rng,
        phase=dev_phase,
        archetype=archetype,
    )
    for k, v in corrective.items():
        applied[k] = float(applied.get(k, 0.0)) + float(v)

    # Performance can raise projected / active / max ceilings (design §4–5).
    try:
        reevaluate_ceilings_from_performance(player, rng)
    except Exception:
        pass

    ovr_after = float(persist_recomputed_ovr(player))
    _mark_ledger(
        ovr_before=ovr_before,
        ovr_after=ovr_after,
        attribute_deltas=applied,
        source_path="apply_player_development",
    )

    net_growth = (ovr_after - ovr_before) * 99.0
    report_type = "growth"
    if dev_phase == "SPIKE":
        report_type = "breakout"
    elif abs(target_display) >= 5.0 and net_growth >= 4.5:
        report_type = "breakout"
    elif dev_phase == "STALL":
        report_type = "stall"
    elif dev_phase == "REGRESSION":
        report_type = "regression"

    env_g = float(getattr(player, "_dev_env_growth_mult", 1.0) or 1.0)
    tw = str(getattr(player, "_dev_env_team_window", "") or "")
    reason = (
        f"archetype={archetype} growth_phase={dev_phase} age={age} "
        f"budget_01={budget:.4f} target_dOVR={target_display:.1f} "
        f"env_window={tw or 'n/a'} env_growth_x={env_g:.2f}"
    )
    _finalize_development_metadata(
        player,
        dev_phase=dev_phase,
        net_growth=net_growth,
        archetype=archetype,
        report_type=report_type,
        reason=reason,
    )
    # Refresh readiness/ETA from post-growth ability so timelines move with evidence.
    calculate_nhl_readiness_score(player)
    update_player_nhl_eta(player)
    mom = _update_career_momentum(player, net_growth=net_growth, dev_phase=dev_phase)
    setattr(player, "career_momentum", round(mom, 1))

    if dev_phase in ("STALL", "REGRESSION") and float(getattr(player, "_bust_pressure", 0.0) or 0.0) >= 0.42:
        if not getattr(player, "_development_failure_reason", None):
            setattr(
                player,
                "_development_failure_reason",
                rng.choice(
                    [
                        "skating_never_translated",
                        "nhl_pace_processing_limit",
                        "injury_disruption",
                        "offense_did_not_translate",
                        "confidence_collapse",
                        "poor_role_opportunity",
                        "defensive_reads_stalled",
                    ]
                ),
            )

    pname = (
        getattr(player, "name", None)
        or getattr(getattr(player, "identity", None), "name", None)
        or "Player"
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

    try:
        setattr(player, "_in_season_growth_spent_01", 0.0)
    except Exception:
        pass


def apply_in_season_development_pulse(
    player: Any,
    rng: Any,
    *,
    pulse_fraction: float = 0.12,
) -> float:
    """
    Mid-season growth from the dedicated in-season pool (~30% of annual).

    Does not consume the season-end pool / year-end ledger.
    Returns display-OVR delta applied this pulse.
    """
    from app.sim_engine.entities.player import (
        normalize_rating_gap,
        persist_recomputed_ovr,
        player_current_ovr_01,
    )

    if player is None or getattr(player, "retired", False):
        return 0.0
    if not isinstance(rng, random.Random):
        rng = random.Random()

    age = _get_player_age(player)
    ovr_before_01 = float(player_current_ovr_01(player))
    profile = resolve_development_profile(player)
    potential = float(profile.get("expected_ceiling", ovr_before_01))
    gap = float(normalize_rating_gap(ovr_before_01, potential))
    prod = _safe_attr_float(
        player,
        ["production_score", "recent_performance_score", "points_signal", "production"],
        0.5,
    )

    if age >= 32 or (age >= 29 and gap <= 0.015):
        if rng.random() > 0.42:
            return 0.0
        phase = "REGRESSION"
        frac = pulse_fraction * rng.uniform(0.55, 1.05)
    elif age <= 27 and gap >= 0.025:
        if rng.random() < 0.08 and prod < 0.42:
            phase = "STALL"
        elif prod >= 0.82 and gap >= 0.05 and rng.random() < 0.35:
            phase = "SPIKE"
        else:
            phase = "NORMAL"
        frac = pulse_fraction * rng.uniform(0.85, 1.25)
        if age <= 22 and gap >= 0.06:
            frac *= 1.25
    else:
        if rng.random() < 0.35:
            return 0.0
        phase = "NORMAL" if prod >= 0.48 else "STALL"
        frac = pulse_fraction * rng.uniform(0.55, 0.95)

    annual = calculate_season_growth_budget(
        player, None, profile, rng=rng, dev_phase=phase
    )
    # Pulses draw only from the in-season share of the annual budget.
    season_pool = abs(float(annual)) * float(_IN_SEASON_POOL_SHARE)
    pulse_budget = season_pool * float(frac) * (1.0 if annual >= 0 else -1.0)
    if abs(pulse_budget) < 0.0020:
        if age <= 24 and gap >= 0.05 and phase != "REGRESSION":
            pulse_budget = 0.0035 if phase != "SPIKE" else 0.0055
        else:
            return 0.0

    spent = float(getattr(player, "_in_season_growth_spent_01", 0.0) or 0.0)
    # Cap cumulative mid-season spend to the in-season pool only.
    pool_cap = max(0.012, season_pool)
    if pulse_budget > 0 and spent + pulse_budget > pool_cap:
        pulse_budget = max(0.0, pool_cap - spent)
    if pulse_budget == 0.0:
        return 0.0

    archetype = str(getattr(player, "_dev_archetype", "") or getattr(player, "archetype", "") or "")
    deltas = allocate_growth_to_attributes(
        player,
        pulse_budget,
        role=str(getattr(player, "role", "") or ""),
        archetype=archetype or None,
        phase=phase,
    )
    if not deltas:
        return 0.0
    apply_attribute_deltas(player, deltas)
    persist_recomputed_ovr(player)
    ensure_displayed_ovr_delta(
        player,
        ovr_before_01=ovr_before_01,
        target_display_delta=float(pulse_budget) * 99.0,
        rng=rng,
        phase=phase,
        archetype=archetype or None,
        tolerance=0.4,
        max_iters=6,
    )
    ovr_after_01 = float(persist_recomputed_ovr(player))
    delta_disp = (ovr_after_01 - ovr_before_01) * 99.0
    try:
        setattr(player, "_in_season_growth_spent_01", spent + abs(float(pulse_budget)))
        accum = float(getattr(player, "_in_season_ovr_delta_accum", 0.0) or 0.0)
        setattr(player, "_in_season_ovr_delta_accum", accum + delta_disp)
    except Exception:
        pass
    return float(delta_disp)
