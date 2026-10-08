"""Emergent player identity — body maturation, play style, and archetype from how they play."""
from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

from app.sim_engine.generation.prospect_body import (
    apply_body_tradeoffs_to_ratings,
    generate_realistic_weight_kg,
)
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)


def _stable_hash(value):
    """Process-stable replacement for built-in hash() (salted per process via PYTHONHASHSEED)."""
    import hashlib as _hl

    return int.from_bytes(_hl.sha256(str(value).encode("utf-8")).digest()[:8], "big", signed=True)


def _pos_key(position: Any) -> str:
    p = str(getattr(position, "value", position) or "").upper()
    if p == "G":
        return "G"
    if p in ("D", "LD", "RD", "LHD", "RHD"):
        return "D"
    return "F"


# (rating key, patterns) -> whether the key belongs to that pattern group. Rating keys come
# from a fixed schema, so this replaces a substring scan per key on every call.
_KEY_PATTERN_MATCH: Dict[Tuple[str, Tuple[str, ...]], bool] = {}


def _key_in_patterns(key: Any, patterns: Tuple[str, ...]) -> bool:
    ks = str(key)
    ck = (ks, patterns)
    hit = _KEY_PATTERN_MATCH.get(ck)
    if hit is None:
        if len(_KEY_PATTERN_MATCH) > 20000:
            _KEY_PATTERN_MATCH.clear()
        kl = ks.lower()
        hit = (not ks.startswith("_")) and any(p in kl for p in patterns)
        _KEY_PATTERN_MATCH[ck] = hit
    return hit


def _rating_avg(player: Any, patterns: Tuple[str, ...]) -> float:
    ratings = getattr(player, "ratings", None) or {}
    if not isinstance(ratings, dict):
        return 50.0
    vals = []
    for key, val in ratings.items():
        if _key_in_patterns(key, patterns):
            try:
                vals.append(float(val))
            except (TypeError, ValueError):
                continue
    return sum(vals) / len(vals) if vals else 50.0


def _player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    if ident is not None:
        try:
            return int(getattr(ident, "age", 0) or 0)
        except (TypeError, ValueError):
            pass
    try:
        return int(getattr(player, "age", 18) or 18)
    except (TypeError, ValueError):
        return 18


def _player_height_weight(player: Any) -> Tuple[int, int]:
    ident = getattr(player, "identity", None)
    h = int(getattr(ident, "height_cm", 0) or getattr(player, "height_cm", 0) or 0)
    w = int(getattr(ident, "weight_kg", 0) or getattr(player, "weight_kg", 0) or 0)
    return h, w


def extract_season_stats(player: Any, stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalize season stat line from explicit dict or player fields."""
    row = dict(stats or {})
    gp = int(row.get("gp", 0) or getattr(player, "gp", 0) or getattr(player, "games_played", 0) or 0)
    goals = int(row.get("goals", row.get("g", 0)) or 0)
    assists = int(row.get("assists", row.get("a", 0)) or 0)
    points = int(row.get("points", row.get("pts", goals + assists)) or 0)
    out = {"gp": gp, "goals": goals, "assists": assists, "points": points}
    if gp > 0:
        out["ppg"] = float(points) / float(gp)
    else:
        out["ppg"] = float(row.get("ppg", 0) or 0)
    ps = getattr(player, "_prospect_season_stats", None)
    if isinstance(ps, dict) and not gp:
        gp2 = int(ps.get("gp", ps.get("games_played", 0)) or 0)
        if gp2 > 0:
            g2 = int(ps.get("goals", ps.get("g", 0)) or 0)
            a2 = int(ps.get("assists", ps.get("a", 0)) or 0)
            p2 = int(ps.get("points", ps.get("pts", g2 + a2)) or 0)
            out.update({"gp": gp2, "goals": g2, "assists": a2, "points": p2, "ppg": p2 / gp2})
    return out


def _sync_body_snapshot(player: Any) -> None:
    h, w = _player_height_weight(player)
    try:
        setattr(player, "_body_height_prev", int(h))
        setattr(player, "_body_weight_prev", int(w))
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def apply_yearly_body_maturation(player: Any, rng: Optional[random.Random] = None) -> bool:
    """
    Annual body change — height spurts for teens, muscle fill-out through the early 20s,
    maintenance / lean-down for veterans. Returns True if height or weight changed.
    """
    if rng is None:
        seed = abs(_stable_hash(str(getattr(player, "id", "") or getattr(player, "rng_seed", "")))) & 0xFFFFFFFF
        rng = random.Random(seed)

    ident = getattr(player, "identity", None)
    if ident is None:
        return False

    age = _player_age(player)
    if age >= 36:
        return False

    h_before = int(getattr(ident, "height_cm", 0) or 0)
    w_before = int(getattr(ident, "weight_kg", 0) or 0)
    if h_before <= 0:
        return False

    grew_cm = 0
    if age <= 21:
        if age <= 14:
            p_spurt, max_cm = 0.22, 7
        elif age == 15:
            p_spurt, max_cm = 0.18, 6
        elif age == 16:
            p_spurt, max_cm = 0.12, 5
        elif age == 17:
            p_spurt, max_cm = 0.08, 4
        elif age <= 19:
            p_spurt, max_cm = 0.05, 3
        else:
            p_spurt, max_cm = 0.03, 2

        dev_arch = str(getattr(player, "_dev_archetype", "") or "").upper()
        if "LATE_BLOOMER" in dev_arch and age >= 15:
            p_spurt += 0.05

        if rng.random() < p_spurt:
            grew_cm = int(max(1, round(1 + rng.random() * (max_cm - 1))))
        elif age <= 16 and rng.random() < 0.55:
            grew_cm = 1 if rng.random() < 0.75 else 2
    elif age <= 25 and rng.random() < 0.06:
        grew_cm = 1

    if grew_cm > 0:
        ident.height_cm = int(min(210, h_before + grew_cm))

    phys = _rating_avg(player, ("strength", "physical", "balance", "endurance", "checking"))
    playstyle = str(getattr(player, "playstyle", "") or "").lower()

    if age <= 21:
        base_gain = 0.6 + 1.8 * rng.random()
        base_gain += (phys - 50.0) * 0.025
        base_gain += 0.55 * grew_cm
        if age <= 17:
            base_gain *= 0.82
        elif age >= 20:
            base_gain *= 1.12
    elif age <= 25:
        base_gain = 0.35 + 1.2 * rng.random() + (phys - 50.0) * 0.018
        base_gain += 0.35 * grew_cm
    elif age <= 29:
        base_gain = 0.15 + 0.55 * rng.random() + (phys - 50.0) * 0.01
    else:
        base_gain = rng.uniform(-0.8, 0.35)
        if "power" not in playstyle and "grinder" not in playstyle:
            base_gain -= 0.25

    if "power" in playstyle or "grinder" in playstyle:
        base_gain += 0.65 if age <= 25 else 0.25
    elif "playmaker" in playstyle or "mobile" in playstyle or "scoring" in playstyle:
        base_gain += 0.12 if age <= 23 else 0.0

    ident.weight_kg = int(max(45, min(125, round(w_before + base_gain))))

    realistic = generate_realistic_weight_kg(
        int(ident.height_cm), getattr(ident, "position", "C"), age=age
    )
    frame_floor = realistic - 12
    if ident.weight_kg < frame_floor:
        ident.weight_kg = int(min(125, max(ident.weight_kg + 2, frame_floor)))

    changed = int(ident.height_cm) != h_before or int(ident.weight_kg) != w_before
    if changed:
        apply_body_tradeoffs_to_ratings(player, rng)
        try:
            setattr(player, "_body_maturation_changed", True)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    return changed


def infer_playstyle_from_identity(player: Any, stats: Optional[Dict[str, Any]] = None) -> str:
    """Derive playstyle bucket from production + frame + skill lean."""
    st = extract_season_stats(player, stats)
    gp = int(st.get("gp", 0) or 0)
    goals = int(st.get("goals", 0) or 0)
    assists = int(st.get("assists", 0) or 0)
    ppg = float(st.get("ppg", 0) or 0)
    pos = _pos_key(getattr(getattr(player, "identity", None), "position", getattr(player, "position", "C")))
    h_cm, w_kg = _player_height_weight(player)

    shot = _rating_avg(player, ("shot", "shoot", "accuracy", "finishing"))
    pass_r = _rating_avg(player, ("pass", "playmak", "vision"))
    skate = _rating_avg(player, ("speed", "accel", "skating", "agility"))
    defense = _rating_avg(player, ("def", "stick_check", "position", "block"))
    phys = _rating_avg(player, ("strength", "physical", "checking", "balance"))

    if pos == "G":
        rebound = _rating_avg(player, ("rebound", "recovery"))
        if pass_r >= 62 or _rating_avg(player, ("puck", "handling")) >= 65:
            return "puck_moving_goalie"
        if rebound >= 65 or skate >= 60:
            return "athletic_goalie"
        return "butterfly_goalie"

    big_frame = h_cm >= 193 or w_kg >= 98
    lean_frame = h_cm <= 178 and w_kg <= 78

    if gp >= 8:
        if pos == "D":
            if goals + assists >= 8 and pass_r >= shot + 4:
                return "offensive_defenseman"
            if ppg <= 0.38 and defense >= shot:
                return "defensive_defenseman"
            if defense >= 58 and ppg >= 0.42:
                return "two_way_defenseman"
        else:
            if goals >= assists * 1.2 and goals >= 8:
                return "sniper"
            if assists >= goals * 1.25 and assists >= 10:
                return "playmaker"
            if big_frame and phys >= 55 and goals >= 6:
                return "power_forward"
            if ppg <= 0.42 and defense >= 55:
                return "grinder"
            if ppg >= 0.75:
                return "scoring_forward"

    # Relative skill lean, not absolute cutoffs. Junior ratings sit around 40-55, so
    # the old "shot >= 58" style thresholds almost never fired and ~95% of kids fell
    # through to "two_way" / "mobile_defenseman". Each player is judged against his
    # OWN profile, with a small stable per-player tiebreak so near-neutral kids still
    # spread across the full menu of styles.
    return _relative_style_pick(player, pos, shot, pass_r, skate, defense, phys, big_frame, lean_frame)


def infer_skill_archetype(player: Any, playstyle: str, stats: Optional[Dict[str, Any]] = None) -> str:
    """Map playstyle + frame to engine archetype string."""
    pos = _pos_key(getattr(getattr(player, "identity", None), "position", getattr(player, "position", "C")))
    style = str(playstyle or "").lower()
    h_cm, w_kg = _player_height_weight(player)

    if pos == "G":
        if "puck" in style:
            return "HYBRID_G"
        if "athletic" in style:
            return "BUTTERFLY_G"
        return "BALANCED_G"

    if pos == "D":
        if "offensive" in style:
            return "OFFENSIVE_D"
        if "defensive" in style or "shutdown" in style or "stay" in style:
            return "DEFENSIVE_D"
        if "mobile" in style:
            return "OFFENSIVE_D" if _rating_avg(player, ("pass", "shot")) >= 58 else "TWO_WAY"
        return "TWO_WAY"

    if "sniper" in style or "shooter" in style:
        return "SNIPER"
    if "playmaker" in style or "distributor" in style:
        return "PLAYMAKER"
    if "power" in style or (h_cm >= 188 and w_kg >= 92):
        return "POWER_FORWARD"
    if "grinder" in style or "checker" in style or "energy" in style:
        return "GRINDER"
    if "two_way" in style or "two-way" in style:
        return "TWO_WAY_F"
    if "scoring" in style:
        st = extract_season_stats(player, stats)
        if int(st.get("goals", 0) or 0) >= int(st.get("assists", 0) or 0):
            return "SNIPER"
        return "PLAYMAKER"
    return "TWO_WAY_F"


def refresh_player_identity(
    player: Any,
    rng: Optional[random.Random] = None,
    stats: Optional[Dict[str, Any]] = None,
    *,
    min_gp: int = 8,
) -> Dict[str, Any]:
    """
    Re-evaluate playstyle + archetype from current body, skills, and production.
    Can shift an established archetype when play or frame changes.
    """
    st = extract_season_stats(player, stats)
    gp = int(st.get("gp", 0) or 0)
    body_shift = bool(getattr(player, "_body_maturation_changed", False))
    committed = bool(getattr(player, "_identity_committed", False))

    if gp < min_gp and not body_shift and committed:
        return {"changed": False, "reason": "insufficient_signal"}

    prev_style = str(getattr(player, "playstyle", "") or "")
    prev_arch = str(getattr(player, "archetype", "") or "")

    playstyle = infer_playstyle_from_identity(player, st)
    archetype = infer_skill_archetype(player, playstyle, st)

    style_changed = prev_style and prev_style != playstyle
    arch_changed = prev_arch and prev_arch != str(archetype)

    try:
        setattr(player, "playstyle", playstyle)
        setattr(player, "player_type", playstyle)
        setattr(player, "archetype", archetype)
        setattr(player, "_identity_committed", True)
        setattr(player, "_identity_provisional", False)
        setattr(player, "_prospect_cached_playstyle_bucket", None)
        if arch_changed:
            setattr(player, "_identity_archetype_prev", prev_arch)
            setattr(player, "_archetype_shifted", True)
        if style_changed:
            setattr(player, "_identity_playstyle_prev", prev_style)
        setattr(player, "_body_maturation_changed", False)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    chem = getattr(player, "chemistry_profile", None)
    if isinstance(chem, dict):
        chem["playstyle"] = playstyle
    elif chem is None:
        try:
            setattr(player, "chemistry_profile", {"playstyle": playstyle})
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    _sync_body_snapshot(player)
    return {
        "changed": True,
        "playstyle": playstyle,
        "archetype": archetype,
        "playstyle_changed": style_changed,
        "archetype_changed": arch_changed,
        "gp": gp,
    }


def commit_prospect_identity(
    player: Any,
    rng: Optional[random.Random] = None,
    stats: Optional[Dict[str, Any]] = None,
    *,
    min_gp: int = 8,
) -> bool:
    """Backward-compatible wrapper."""
    result = refresh_player_identity(player, rng=rng, stats=stats, min_gp=min_gp)
    return bool(result.get("changed"))


def progress_season_body_and_identity(
    player: Any,
    rng: Optional[random.Random] = None,
    stats: Optional[Dict[str, Any]] = None,
    *,
    min_gp: int = 8,
) -> Dict[str, Any]:
    """Run annual body maturation, then refresh role identity (may shift archetype)."""
    if not getattr(player, "_body_height_prev", None):
        _sync_body_snapshot(player)
    body_changed = apply_yearly_body_maturation(player, rng)
    identity = refresh_player_identity(player, rng=rng, stats=stats, min_gp=min_gp)
    identity["body_changed"] = body_changed
    return identity


def spawn_youth_baseline_profile(position: Any, rng: Optional[random.Random] = None) -> str:
    """Youth rating shape at spawn.

    Every junior used to spawn with the neutral two-way shape, so whole draft classes
    read as two-way forwards / two-way D. Kids now spawn with a real (mild) lean drawn
    from a balanced menu; play and the yearly identity refresh still reshape it.
    """
    pos = _pos_key(position)
    if pos == "G":
        return "balanced_g"
    if rng is None:
        if pos == "D":
            return "two_way_d"
        return "two_way"
    if pos == "D":
        return rng.choices(
            ["offensive_d", "defensive_d", "two_way_d", "enforcer_d"],
            weights=[0.32, 0.30, 0.30, 0.08],
        )[0]
    return rng.choices(
        ["sniper", "playmaker", "power_forward", "two_way", "grinder"],
        weights=[0.24, 0.24, 0.20, 0.20, 0.12],
    )[0]

# Calibrated so a neutral class lands near: F sniper 22 / playmaker 22 / power 18 /
# two-way 20 / grinder 10 / speed 8 %; D offensive 30 / defensive 30 / two-way 25 /
# mobile 15 %.
_F_STYLE_BIAS = {"sniper": 0.6, "playmaker": -0.9, "power_forward": 0.0, "two_way": -0.3, "grinder": -0.9, "scoring_forward": -0.4}
_D_STYLE_BIAS = {"offensive_defenseman": 0.3, "defensive_defenseman": -0.9, "two_way_defenseman": 1.3, "mobile_defenseman": -0.3}


def _style_noise(player: Any, key: str) -> float:
    import hashlib
    import math

    ident = getattr(player, "identity", None)
    pid = str(getattr(player, "id", None) or getattr(player, "player_id", None) or getattr(ident, "name", None) or "")
    raw = hashlib.sha256(f"style|{pid}|{key}".encode()).hexdigest()
    u1 = max(1e-9, int(raw[:12], 16) / float(1 << 48))
    u2 = int(raw[12:24], 16) / float(1 << 48)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _relative_style_pick(
    player: Any, pos: str, shot: float, pass_r: float, skate: float, defense: float, phys: float,
    big_frame: bool, lean_frame: bool,
) -> str:
    m = (shot + pass_r + skate + defense + phys) / 5.0
    ts, tp, tk, td, tf = shot - m, pass_r - m, skate - m, defense - m, phys - m
    sd = 2.2  # stable per-player noise, rating points
    if pos == "D":
        scores = {
            "offensive_defenseman": 0.5 * (ts + tp) + _D_STYLE_BIAS["offensive_defenseman"],
            "defensive_defenseman": td + 0.4 * tf + (2.0 if big_frame else 0.0) + _D_STYLE_BIAS["defensive_defenseman"],
            "two_way_defenseman": -0.35 * (abs(td) + abs(ts) + abs(tp)) + 1.0 + _D_STYLE_BIAS["two_way_defenseman"],
            "mobile_defenseman": tk + (1.0 if lean_frame else 0.0) + _D_STYLE_BIAS["mobile_defenseman"],
        }
    else:
        scores = {
            "sniper": ts + _F_STYLE_BIAS["sniper"],
            "playmaker": tp + _F_STYLE_BIAS["playmaker"],
            "power_forward": tf + 0.4 * ts + (2.0 if big_frame else 0.0) + _F_STYLE_BIAS["power_forward"],
            "two_way": td - 0.2 * abs(ts - tp) + 0.5 + _F_STYLE_BIAS["two_way"],
            "grinder": 0.6 * td + 0.6 * tf + _F_STYLE_BIAS["grinder"],
            "scoring_forward": tk + (1.0 if lean_frame else 0.0) + _F_STYLE_BIAS["scoring_forward"],
        }
    best = max(scores, key=lambda k: scores[k] + sd * _style_noise(player, k))
    return best
