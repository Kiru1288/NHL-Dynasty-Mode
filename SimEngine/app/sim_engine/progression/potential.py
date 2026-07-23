# app/sim_engine/progression/potential.py
"""
Dynamic potential drift: breakout, stagnate, bust.

Canonical scale for player.potential is 0.0–1.0.
All drift must go through apply_potential_drift so seasons cannot compound.
"""

from typing import Any, Dict, Optional


def _age(player: Any) -> int:
    identity = getattr(player, "identity", None)
    if identity is not None and hasattr(identity, "age"):
        return int(identity.age)
    return int(getattr(player, "age", 26))


def _ovr(player: Any) -> float:
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
        return 0.5


def _morale(player: Any) -> float:
    psych = getattr(player, "psych", None)
    if psych is not None and hasattr(psych, "morale"):
        return float(psych.morale)
    return float(getattr(player, "morale", 0.5))


def _potential(player: Any) -> float:
    from app.sim_engine.entities.player import normalize_rating

    p = getattr(player, "potential", None)
    if p is not None:
        return float(normalize_rating(p))
    ratings = getattr(player, "ratings", None)
    if isinstance(ratings, dict):
        for key in ("dev_potential", "potential", "dev_ceiling"):
            if ratings.get(key) is not None:
                return float(normalize_rating(ratings.get(key)))
    return float(normalize_rating(_ovr(player) * 1.05))


def ensure_development_ledger(player: Any, season_id: Any) -> Dict[str, Any]:
    """Persistent per-player seasonal development ledger (lazy-init for legacy saves)."""
    sid = str(season_id if season_id is not None else getattr(player, "_active_dev_season", None) or "default")
    ledger = getattr(player, "development_ledger", None)
    if not isinstance(ledger, dict):
        ledger = {}
    if str(ledger.get("season") or "") != sid:
        hist = getattr(player, "development_history", None)
        if not isinstance(hist, list):
            hist = []
            try:
                setattr(player, "development_history", hist)
            except Exception:
                hist = []
        if ledger.get("season") is not None:
            try:
                hist.append(dict(ledger))
            except Exception:
                pass
        ledger = {
            "season": sid,
            "development_applied": False,
            "aging_applied": False,
            "potential_drift_applied": False,
            "breakout_or_bust_applied": False,
            "source_path": None,
            "attribute_deltas": {},
            "ovr_before": None,
            "ovr_after": None,
            "potential_drift_reason": None,
        }
    try:
        setattr(player, "development_ledger", ledger)
    except Exception:
        pass
    return ledger


def apply_potential_drift(
    player: Any,
    event: str,
    season_context: Optional[Dict[str, Any]] = None,
    *,
    rng: Any = None,
    delta_01: Optional[float] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Centralized true-potential drift. One ordinary resolution per season unless force.

    Updates player.potential (0–1) and mirrors ratings.dev_potential when present.
    Bust reduces expected ceiling first; maximum moves only on significant events.
    """
    from app.sim_engine.entities.player import clamp01, display_rating, normalize_rating

    ctx = season_context if isinstance(season_context, dict) else {}
    sid = ctx.get("season") or getattr(player, "_active_dev_season", None) or "default"
    ledger = ensure_development_ledger(player, sid)

    ordinary = str(event or "ordinary").lower() in ("ordinary", "annual", "update", "stagnate", "")
    if ordinary and ledger.get("potential_drift_applied") and not force:
        return {"applied": False, "reason": "already_applied", "event": event}

    before = _potential(player)
    profile = getattr(player, "development_profile", None)
    if not isinstance(profile, dict):
        profile = {}

    expected = normalize_rating(profile.get("expected_ceiling", before))
    maximum = normalize_rating(profile.get("maximum_ceiling", max(before, expected + 0.03)))
    if expected > maximum:
        expected, maximum = maximum, expected

    delta = float(delta_01) if delta_01 is not None else 0.0
    reason = str(event or "ordinary")

    if delta_01 is None and rng is not None:
        age = _age(player)
        ovr = _ovr(player)
        morale = _morale(player)
        arch = str(getattr(player, "_dev_archetype", "") or "").upper()
        bp = float(getattr(player, "_bust_pressure", 0.08) or 0.08)
        sm = float(getattr(player, "_steal_momentum", 0.06) or 0.06)
        nar_g = float(getattr(player, "_narrative_prog_growth_mult", 1.0) or 1.0)
        nar_d = float(getattr(player, "_narrative_decline_p_mult", 1.0) or 1.0)
        p_break = 0.078
        p_bust = 0.10
        if arch == "HIGH_VARIANCE":
            p_break += 0.035
            p_bust += 0.028
        elif arch == "SAFE_LOW_CEILING":
            p_break -= 0.022
            p_bust += 0.012
        elif arch == "LATE_BLOOMER" and 22 <= age <= 26:
            p_break += 0.04
        elif arch == "ELITE_CEILING_VOLATILE":
            p_break += 0.02
            p_bust += 0.035
        p_break *= max(0.72, min(1.28, nar_g))
        p_bust *= max(0.75, min(1.32, nar_d))
        if sm >= 0.5:
            p_break += 0.04 * (sm - 0.5)
        if bp >= 0.45:
            p_bust += 0.05 * (bp - 0.45)

        if age < 24 and morale >= 0.6 and before < 0.90 and rng.random() < min(0.22, p_break):
            delta = rng.choice([0.01, 0.02])
            reason = "breakout"
        elif age < 23 and morale < 0.4 and ovr < 0.70 and rng.random() < min(0.22, p_bust):
            delta = rng.choice([-0.01, -0.02])
            reason = "bust"
        elif age < 24 and bp > 0.5 and rng.random() < 0.06 + 0.14 * (bp - 0.5):
            delta = rng.choice([-0.01, -0.02])
            reason = "bust_pressure"
        elif 23 <= age <= 27 and morale < 0.45 and rng.random() < 0.06:
            delta = -0.01
            reason = "stagnate"

    if abs(delta) < 1e-9:
        ledger["potential_drift_applied"] = True
        ledger["potential_drift_reason"] = "no_change"
        return {"applied": True, "delta": 0.0, "reason": "no_change", "before": before, "after": before}

    if delta < 0:
        expected = clamp01(expected + delta)
        if reason.startswith("bust") or abs(delta) >= 0.02:
            maximum = clamp01(max(expected, maximum + delta * 0.35))
    else:
        expected = clamp01(expected + delta)
        if reason == "breakout" and abs(delta) >= 0.02:
            maximum = clamp01(min(0.98, maximum + delta * 0.5))
        maximum = max(maximum, expected)

    after = clamp01(max(_ovr(player), expected))
    try:
        setattr(player, "potential", after)
    except Exception:
        pass
    ratings = getattr(player, "ratings", None)
    if isinstance(ratings, dict):
        ratings["dev_potential"] = float(display_rating(after))
        if reason == "breakout" and abs(delta) >= 0.02:
            ratings["dev_ceiling"] = float(display_rating(maximum))

    profile["expected_ceiling"] = after
    profile["maximum_ceiling"] = maximum
    profile["current_ovr"] = _ovr(player)
    try:
        setattr(player, "development_profile", profile)
    except Exception:
        pass

    if reason in ("breakout", "bust", "bust_pressure"):
        ledger["breakout_or_bust_applied"] = True
    ledger["potential_drift_applied"] = True
    ledger["potential_drift_reason"] = reason

    hist = getattr(player, "development_history", None)
    if isinstance(hist, list):
        hist.append({
            "season": sid,
            "kind": "potential_drift",
            "reason": reason,
            "before": round(before, 4),
            "after": round(after, 4),
            "delta": round(delta, 4),
        })

    return {
        "applied": True,
        "delta": round(delta, 4),
        "reason": reason,
        "before": before,
        "after": after,
        "expected_ceiling": after,
        "maximum_ceiling": maximum,
    }


def update_player_potential(player: Any, rng: Any) -> None:
    """
    Dynamic potential: breakout / stagnate / bust via apply_potential_drift.
    Idempotent per season.
    """
    apply_potential_drift(
        player,
        "ordinary",
        {"season": getattr(player, "_active_dev_season", None)},
        rng=rng,
    )
