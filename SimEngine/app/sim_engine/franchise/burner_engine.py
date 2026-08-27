"""GM burner account — risk, exposure, and investigation loop."""

from __future__ import annotations

import random
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.franchise.storyline_engine import (
    MARKET_MEDIA_PROFILES,
    MEDIA_REPORTERS,
    _REPORTER_BY_ID,
    _apply_storyline_effects,
    _clamp,
    _market_key_for_team,
    _market_profile_for_team,
    _u_current_meta,
    apply_fan_engagement_delta,
)

RISKY_WORD_WEIGHTS: Dict[str, int] = {
    "trade": 16, "traded": 16, "shop": 18, "shopping": 18, "dump": 20, "deal": 10,
    "fire": 22, "fired": 22, "quit": 18, "resign": 18, "coach": 10, "bench": 12,
    "owner": 16, "ownership": 16, "cheap": 14, "gm": 14, "management": 12,
    "lazy": 16, "selfish": 18, "washed": 16, "overpaid": 14, "embarrassing": 15,
    "tank": 18, "tanking": 18, "soft": 10, "choke": 14, "choked": 14, "clown": 16,
    "disgrace": 18, "garbage": 14, "joke": 12,
}

BURNER_COOLDOWN_DAYS = 180
LEE_INVESTIGATION_THRESHOLD = 55.0
LEE_INVESTIGATION_EXPOSE = 92.0


def _ensure_burner_account(session: Any) -> Dict[str, Any]:
    acct = getattr(session, "gm_burner_account", None)
    if not isinstance(acct, dict):
        acct = {
            "handle": "",
            "created_day": 0,
            "posts": [],
            "suspicion_score": 0.0,
            "exposed": False,
        }
        session.gm_burner_account = acct
    return acct


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", str(text or "").lower())


def _contextual_risk_words(session: Any) -> Dict[str, int]:
    extra: Dict[str, int] = {}
    for sl in list(getattr(session, "active_cause_storylines", None) or [])[-20:]:
        headline = str(sl.get("headline") or "").lower()
        summary = str(sl.get("summary") or sl.get("description") or "").lower()
        blob = f"{headline} {summary}"
        if "trade" in blob:
            extra.update({"trade": 24, "traded": 24, "shop": 26, "deal": 18})
        if "coach" in blob or "hot seat" in blob:
            extra.update({"fire": 28, "coach": 18, "bench": 16})
        if "contract" in blob:
            extra.update({"overpaid": 20, "cheap": 18, "deal": 14})
        pname = str(sl.get("player_name") or "").lower()
        if pname:
            for part in pname.split():
                if len(part) > 3:
                    extra[part] = max(extra.get(part, 0), 12)
    return extra


def compute_burner_risk(session: Any, post_text: str, market_key: str) -> int:
    market = MARKET_MEDIA_PROFILES.get(market_key, MARKET_MEDIA_PROFILES["default"])
    weights = {**RISKY_WORD_WEIGHTS, **_contextual_risk_words(session)}
    word_score = sum(weights.get(w.lower().strip(".,!?"), 0) for w in _tokenize(post_text))
    length_penalty = 8 if len(post_text) > 200 else 0
    base = 22 + word_score + length_penalty
    acct = _ensure_burner_account(session)
    suspicion_bump = int(float(acct.get("suspicion_score") or 0) * 0.08)
    return int(_clamp(base * float(market.get("pressure_mult") or 1.0) + suspicion_bump, 6, 94))


def preview_burner_risk(session: Any, post_text: str, market_key: str) -> Dict[str, Any]:
    risk = compute_burner_risk(session, post_text, market_key)
    market = MARKET_MEDIA_PROFILES.get(market_key, MARKET_MEDIA_PROFILES["default"])
    return {
        "risk": risk,
        "market_key": market_key,
        "market_label": market.get("label"),
        "risk_band": "low" if risk < 35 else "mid" if risk < 60 else "high",
    }


def _generate_burner_handle(session: Any, rng: random.Random) -> str:
    day, _, _ = _u_current_meta(session)
    return f"@RinkInsider{rng.randint(100, 9999)}{day % 97}"


def _can_create_burner(session: Any) -> bool:
    acct = _ensure_burner_account(session)
    if not acct.get("exposed"):
        return not acct.get("handle")
    day, _, _ = _u_current_meta(session)
    created = int(acct.get("created_day") or 0)
    return (day - created) >= BURNER_COOLDOWN_DAYS


def ensure_burner_handle(session: Any, rng: Optional[random.Random] = None) -> str:
    acct = _ensure_burner_account(session)
    if acct.get("handle"):
        return str(acct["handle"])
    if not _can_create_burner(session) and acct.get("exposed"):
        return ""
    r = rng or random.Random()
    day, _, _ = _u_current_meta(session)
    acct["handle"] = _generate_burner_handle(session, r)
    acct["created_day"] = day
    if acct.get("exposed"):
        acct["suspicion_score"] = max(float(acct.get("suspicion_score") or 0), 18.0)
        acct["exposed"] = False
    session.gm_burner_account = acct
    return str(acct["handle"])


def _tick_lee_investigation(session: Any, risk: int) -> None:
    acct = _ensure_burner_account(session)
    inv = getattr(session, "gm_burner_investigation", None)
    suspicion = float(acct.get("suspicion_score") or 0)
    if suspicion < LEE_INVESTIGATION_THRESHOLD:
        return
    day, _, _ = _u_current_meta(session)
    if not isinstance(inv, dict) or not inv.get("reporter_id"):
        session.gm_burner_investigation = {
            "reporter_id": "lee",
            "reporter_name": _REPORTER_BY_ID.get("lee", MEDIA_REPORTERS[5])["name"],
            "progress": 0.0,
            "started_day": day,
        }
        inv = session.gm_burner_investigation
    bump = 2.5 + (risk * 0.06) + (suspicion * 0.02)
    inv["progress"] = min(100.0, float(inv.get("progress") or 0) + bump)
    session.gm_burner_investigation = inv


def _apply_burner_exposure(session: Any, result: Dict[str, Any]) -> None:
    text = str(result.get("text") or "").lower()
    utid = str(getattr(session, "user_team_id") or "")
    acct = _ensure_burner_account(session)
    acct["exposed"] = True
    severity = "minor"
    effects = {"media_pressure": 8, "fan_confidence": -6, "owner_patience": -5}
    if any(w in text for w in ("fire", "fired", "bench", "lazy", "selfish", "washed", "clown")):
        severity = "major"
        effects = {"media_pressure": 18, "fan_confidence": -14, "owner_patience": -12, "team_morale": -8}
    elif any(w in text for w in ("trade", "shop", "dump", "tank")):
        severity = "trade"
        effects = {"media_pressure": 14, "fan_confidence": -10, "owner_patience": -8}
    _apply_storyline_effects(session, utid, "", effects)
    apply_fan_engagement_delta(session, utid, -0.12 * (1.0 if severity == "minor" else 2.0), source="burner_exposed")
    try:
        from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433

        headline = "GM burner account linked to franchise social post" if severity != "major" else "Burner scandal erupts around front office"
        if severity == "trade":
            headline = "Anonymous account stokes trade chaos — investigation opened"
        _record_storyline(
            session,
            {
                "headline": headline,
                "summary": f"Investigative desk traced activity to an account matching internal patterns. Risk score {result.get('risk')}.",
                "team_id": utid,
                "category": "conduct",
                "type": "burner_exposure",
                "cause_type": "TRADE_DEMAND" if severity == "major" else "GM_JOB_SECURITY",
                "priority": "HIGH" if severity != "minor" else "MEDIUM",
                "heat": 72 if severity == "major" else 58,
                "reporter_id": "lee",
                "reporter_name": _REPORTER_BY_ID.get("lee", MEDIA_REPORTERS[5])["name"],
                "knowledge_type": "fact",
            },
        )
    except Exception:
        pass
    result["outcome"] = f"Exposure ({severity}): media heat and owner patience dropped."
    inv = dict(getattr(session, "gm_burner_investigation", None) or {})
    inv["progress"] = 100.0
    session.gm_burner_investigation = inv


def _apply_burner_success(session: Any, result: Dict[str, Any]) -> None:
    utid = str(getattr(session, "user_team_id") or "")
    risk = int(result.get("risk") or 0)
    scale = risk / 100.0
    effects = {
        "fan_confidence": int(4 + scale * 10),
        "media_pressure": int(-2 - scale * 4),
        "team_morale": int(2 + scale * 6),
    }
    _apply_storyline_effects(session, utid, "", effects)
    apply_fan_engagement_delta(session, utid, 0.04 + scale * 0.08, source="burner_success")
    result["outcome"] = f"Post landed cleanly. Fan pulse ticked up (risk {risk})."


def submit_burner_post(session: Any, text: str, market_key: str, rng: Optional[random.Random] = None) -> Dict[str, Any]:
    post_text = str(text or "").strip()
    if not post_text:
        raise ValueError("Post text required")
    r = rng or random.Random()
    ensure_burner_handle(session, r)
    risk = compute_burner_risk(session, post_text, market_key)
    acct = _ensure_burner_account(session)
    acct["suspicion_score"] = min(100.0, float(acct.get("suspicion_score") or 0) + risk * 0.12)
    inv = dict(getattr(session, "gm_burner_investigation", None) or {})
    inv_progress = float(inv.get("progress") or 0)
    caught = (r.random() * 100 < risk) or float(acct.get("suspicion_score") or 0) >= LEE_INVESTIGATION_EXPOSE or inv_progress >= 98.0
    day, iso, _ = _u_current_meta(session)
    result = {
        "text": post_text,
        "risk": risk,
        "market_key": market_key,
        "caught": caught,
        "day": day,
        "calendar_iso": iso,
        "handle": acct.get("handle"),
    }
    if caught:
        _apply_burner_exposure(session, result)
    else:
        _apply_burner_success(session, result)
    posts = list(acct.get("posts") or [])
    posts.append(dict(result))
    acct["posts"] = posts[-20:]
    session.gm_burner_account = acct
    _tick_lee_investigation(session, risk)
    return result


def burner_state_payload(session: Any) -> Dict[str, Any]:
    acct = _ensure_burner_account(session)
    utid = str(getattr(session, "user_team_id") or "")
    market_key = _market_key_for_team(session, utid) if utid else "default"
    market = _market_profile_for_team(session, utid) if utid else MARKET_MEDIA_PROFILES["default"]
    inv = dict(getattr(session, "gm_burner_investigation", None) or {})
    return {
        **acct,
        "default_market_key": market_key,
        "default_market_label": market.get("label"),
        "investigation": inv,
        "can_post": bool(acct.get("handle")) or _can_create_burner(session),
        "risky_words": dict(RISKY_WORD_WEIGHTS),
    }


def tick_burner_investigation_daily(session: Any) -> None:
    """Called from narrative daily pass — passive investigation progress."""
    acct = _ensure_burner_account(session)
    if float(acct.get("suspicion_score") or 0) < LEE_INVESTIGATION_THRESHOLD:
        return
    inv = dict(getattr(session, "gm_burner_investigation", None) or {})
    if not inv:
        return
    inv["progress"] = min(100.0, float(inv.get("progress") or 0) + 1.2)
    session.gm_burner_investigation = inv
