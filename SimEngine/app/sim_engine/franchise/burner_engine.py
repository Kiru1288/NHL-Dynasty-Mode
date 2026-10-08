"""GM burner account — risk, exposure, and investigation loop."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

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
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

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


def _storyline_name_parts(session: Any) -> Dict[str, int]:
    """Name tokens of players in live storylines. Mentioning one makes a post easier to
    trace, but a name on its own is not criticism (bug U1)."""
    names: Dict[str, int] = {}
    for sl in list(getattr(session, "active_cause_storylines", None) or [])[-20:]:
        pname = str(sl.get("player_name") or "").lower()
        for part in pname.split():
            if len(part) > 3:
                names[part] = 12
    return names


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
    return extra


POSITIVE_WORDS = frozenset({
    "love", "great", "proud", "believe", "best", "elite", "clutch", "win", "wins", "hype", "underrated",
    "legend", "beast", "goat", "special", "future", "rebuild", "patience", "trust", "process", "playoffs",
    "cup", "loyal", "heart", "warrior", "sniper", "wall", "class", "leader", "captain",
})


def post_tone(session: Any, post_text: str) -> Dict[str, Any]:
    """What kind of post this is: hype for the club, criticism of someone, or noise."""
    weights = {**RISKY_WORD_WEIGHTS, **_contextual_risk_words(session)}
    words = _tokenize(post_text)
    neg = sum(weights.get(w, 0) for w in words)
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    names = _storyline_name_parts(session)
    name_hits = sorted({w for w in words if w in names})
    # A named storyline player raises the stakes of a negative post a lot, and makes
    # a positive one a little easier to trace.
    trace = 0
    if name_hits:
        if neg > 0:
            neg += sum(names[w] for w in name_hits)
        else:
            trace = 4 * len(name_hits)
    targets = sorted({w for w in words if w in ("coach", "bench", "owner", "ownership", "gm", "management")})
    trade_talk = any(w in ("trade", "traded", "shop", "shopping", "dump", "deal") for w in words)
    if neg > 0:
        tone = "trade_talk" if trade_talk and not targets else "criticism"
    elif pos > 0:
        tone = "hype"
    else:
        tone = "noise"
    return {"tone": tone, "neg": neg, "pos": pos, "targets": targets, "words": len(words), "trace": trace, "names": name_hits}


def compute_burner_risk(session: Any, post_text: str, market_key: str) -> int:
    market = MARKET_MEDIA_PROFILES.get(market_key, MARKET_MEDIA_PROFILES["default"])
    t = post_tone(session, post_text)
    length_penalty = 8 if len(post_text) > 200 else 0
    # A plain fan-style post is hard to trace; inside information and attacks are not.
    base = {"hype": 6, "noise": 8, "trade_talk": 14, "criticism": 12}[t["tone"]] + t["neg"] + t.get("trace", 0) + length_penalty
    acct = _ensure_burner_account(session)
    suspicion_bump = int(float(acct.get("suspicion_score") or 0) * 0.10)
    # An account that does nothing but hype the club starts to look like team PR.
    if t["tone"] == "hype":
        suspicion_bump += 4 * max(0, _recent_hype_count(session) - 1)
    return int(_clamp(base * float(market.get("pressure_mult") or 1.0) + suspicion_bump, 3, 94))


def catch_probability(risk: int) -> float:
    """Risk is how traceable the post is; getting caught on any single post is rarer."""
    return round(min(0.65, (max(0, risk) / 100.0) ** 1.6 * 0.62), 3)


HYPE_WINDOW_DAYS = 7


def _team_losing(session: Any) -> bool:
    """True when the user's club has played 8+ games and lost more than it won (bug U2:
    StandingsTable has no .get, the records live on .records)."""
    try:
        utid = str(getattr(session, "user_team_id", "") or "")
        st = getattr(session, "standings", None)
        recs = getattr(st, "records", None)
        if recs is None and isinstance(st, dict):
            recs = st
        rec = (recs or {}).get(utid)
        if rec is None:
            return False
        if isinstance(rec, dict):
            w = int(rec.get("w") or rec.get("wins") or 0)
            l = int(rec.get("l") or rec.get("losses") or 0) + int(rec.get("otl") or 0)
        else:
            w = int(getattr(rec, "wins", 0) or 0)
            l = int(getattr(rec, "losses", 0) or 0) + int(getattr(rec, "otl", 0) or 0)
        return (w + l) >= 8 and w < l
    except Exception:
        return False


def _recent_hype_count(session: Any) -> int:
    acct = getattr(session, "gm_burner_account", None) or {}
    day, _, _ = _u_current_meta(session)
    return sum(
        1 for p in list(acct.get("posts") or [])
        if p.get("tone") == "hype" and not p.get("caught") and day - int(p.get("day") or -999) < HYPE_WINDOW_DAYS
    )


def _projected_effects(session: Any, tone: Dict[str, Any], risk: int) -> Dict[str, Dict[str, int]]:
    scale = risk / 100.0
    losing = _team_losing(session)
    if tone["tone"] == "hype":
        # Diminishing returns (bug U3): full effect once a week, then fading to nothing.
        n = _recent_hype_count(session)
        ok = {"fan_confidence": [3, 2, 1][n] if n < 3 else 0, "team_morale": 1 if n == 0 else 0}
    elif tone["tone"] == "noise":
        ok = {"fan_confidence": 1}
    elif tone["tone"] == "trade_talk":
        ok = {"media_pressure": int(3 + scale * 6), "team_morale": int(-1 - scale * 3), "fan_confidence": 2 if losing else -1}
    else:
        ok = {"media_pressure": int(2 + scale * 5), "team_morale": int(-2 - scale * 4), "fan_confidence": int(3 + scale * 6) if losing else -2}
    if tone["tone"] in ("criticism", "trade_talk") and any(w in tone["targets"] for w in ("coach", "bench")):
        bad = {"media_pressure": 18, "fan_confidence": -14, "owner_patience": -12, "team_morale": -8}
    elif tone["tone"] == "trade_talk":
        bad = {"media_pressure": 14, "fan_confidence": -10, "owner_patience": -8}
    elif tone["tone"] == "criticism":
        bad = {"media_pressure": 12, "fan_confidence": -8, "owner_patience": -7, "team_morale": -4}
    else:
        bad = {"media_pressure": 5, "fan_confidence": -3, "owner_patience": -2}
    return {"ok": ok, "bad": bad}


def _fx_text(fx: Dict[str, int]) -> str:
    names = {"fan_confidence": "fan confidence", "media_pressure": "media pressure", "team_morale": "room morale", "owner_patience": "owner patience"}
    parts = [f"{names.get(k, k)} {v:+d}" for k, v in fx.items() if v]
    return ", ".join(parts) or "no real effect"


def preview_burner_risk(session: Any, post_text: str, market_key: str) -> Dict[str, Any]:
    risk = compute_burner_risk(session, post_text, market_key)
    market = MARKET_MEDIA_PROFILES.get(market_key, MARKET_MEDIA_PROFILES["default"])
    tone = post_tone(session, post_text)
    fx = _projected_effects(session, tone, risk)
    return {
        "risk": risk,
        "catch_pct": round(catch_probability(risk) * 100),
        "tone": tone["tone"],
        "targets": tone["targets"],
        "if_lands": _fx_text(fx["ok"]),
        "if_caught": _fx_text(fx["bad"]),
        "market_key": market_key,
        "market_label": market.get("label"),
        "risk_band": "low" if risk < 35 else "mid" if risk < 60 else "high",
    }


def _generate_burner_handle(session: Any, rng: random.Random) -> str:
    day, _, _ = _u_current_meta(session)
    return f"@RinkInsider{rng.randint(100, 9999)}{day % 97}"


def _days_until_new_burner(session: Any) -> int:
    acct = _ensure_burner_account(session)
    if not acct.get("exposed"):
        return 0
    day, _, _ = _u_current_meta(session)
    since = int(acct.get("exposed_day", acct.get("created_day")) or 0)
    if day < since:  # calendar rolled over to a new season: the offseason covers it
        return 0
    return max(0, BURNER_COOLDOWN_DAYS - (day - since))


def _can_create_burner(session: Any) -> bool:
    acct = _ensure_burner_account(session)
    if not acct.get("exposed"):
        return True
    return _days_until_new_burner(session) <= 0


def ensure_burner_handle(session: Any, rng: Optional[random.Random] = None) -> str:
    acct = _ensure_burner_account(session)
    if acct.get("handle") and not acct.get("exposed"):
        return str(acct["handle"])
    if not _can_create_burner(session):
        return ""
    r = rng or random.Random()
    day, _, _ = _u_current_meta(session)
    acct["handle"] = _generate_burner_handle(session, r)
    acct["created_day"] = day
    if acct.get("exposed"):
        # Fresh account: the old investigation is closed, but the press still remembers.
        acct["suspicion_score"] = 18.0
        acct["exposed"] = False
        acct.pop("exposed_day", None)
        session.gm_burner_investigation = {}
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


def _apply_burner_exposure(session: Any, result: Dict[str, Any], fx: Dict[str, int], tone: Dict[str, Any]) -> None:
    utid = str(getattr(session, "user_team_id") or "")
    acct = _ensure_burner_account(session)
    acct["exposed"] = True
    day_now, _, _ = _u_current_meta(session)
    acct["exposed_day"] = day_now
    acct["burned_handles"] = (list(acct.get("burned_handles") or []) + [acct.get("handle")])[-5:]
    severity = "major" if fx.get("media_pressure", 0) >= 16 else "trade" if tone["tone"] == "trade_talk" else "minor"
    _apply_storyline_effects(session, utid, "", dict(fx))
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
                "summary": f"An investigative desk traced {acct.get('handle') or 'an anonymous account'} back to the front office. The post: \u201c{str(result.get('text') or '')[:140]}\u201d",
                "team_id": utid,
                "category": "conduct",
                "type": "burner_exposure",
                "cause_type": "GM_JOB_SECURITY",
                "priority": "HIGH" if severity != "minor" else "MEDIUM",
                "heat": 72 if severity == "major" else 58,
                "reporter_id": "lee",
                "reporter_name": _REPORTER_BY_ID.get("lee", MEDIA_REPORTERS[5])["name"],
                "knowledge_type": "fact",
            },
        )
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    result["outcome"] = f"Exposed ({severity}): {_fx_text(fx)}. The account is burned; a new one can be made in {BURNER_COOLDOWN_DAYS} days."
    inv = dict(getattr(session, "gm_burner_investigation", None) or {})
    inv["progress"] = 100.0
    session.gm_burner_investigation = inv


def _apply_burner_success(session: Any, result: Dict[str, Any], fx: Dict[str, int], tone: Dict[str, Any]) -> None:
    utid = str(getattr(session, "user_team_id") or "")
    _apply_storyline_effects(session, utid, "", dict(fx))
    delta = {"hype": 0.05, "noise": 0.01, "trade_talk": 0.03, "criticism": 0.03}[tone["tone"]]
    if tone["tone"] == "hype":
        delta *= 1.0 / (1 + _recent_hype_count(session))
    apply_fan_engagement_delta(session, utid, delta, source="burner_success")
    result["outcome"] = f"Post landed ({tone['tone'].replace('_', ' ')}): {_fx_text(fx)}."


def _publish_to_feed(session: Any, result: Dict[str, Any], tone: Dict[str, Any]) -> None:
    try:
        from services.social_feed_engine import publish_external_post  # noqa: WPS433

        publish_external_post(
            session,
            handle=str(result.get("handle") or "@anon"),
            name="Anonymous",
            text=str(result.get("text") or ""),
            kind="burner",
            sentiment={"hype": 0.6, "noise": 0.0, "trade_talk": -0.3, "criticism": -0.6}[tone["tone"]],
            controversy=min(1.0, float(result.get("risk") or 0) / 80.0),
        )
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    if not result.get("caught"):
        try:
            from services.player_social_engine import react_to_gm_burner  # noqa: WPS433

            react_to_gm_burner(session, str(result.get("text") or ""), tone["tone"])
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)


def submit_burner_post(session: Any, text: str, market_key: str, rng: Optional[random.Random] = None) -> Dict[str, Any]:
    post_text = str(text or "").strip()
    if not post_text:
        raise ValueError("Write something first.")
    if len(post_text) > 280:
        raise ValueError("Posts are capped at 280 characters.")
    r = rng or random.Random()
    if not ensure_burner_handle(session, r):
        raise ValueError(f"Your burner was exposed. You can open a new one in {_days_until_new_burner(session)} days.")
    risk = compute_burner_risk(session, post_text, market_key)
    tone = post_tone(session, post_text)
    fxs = _projected_effects(session, tone, risk)
    acct = _ensure_burner_account(session)
    acct["suspicion_score"] = min(100.0, float(acct.get("suspicion_score") or 0) + risk * 0.10)
    inv = dict(getattr(session, "gm_burner_investigation", None) or {})
    inv_progress = float(inv.get("progress") or 0)
    caught = (
        r.random() < catch_probability(risk)
        or float(acct.get("suspicion_score") or 0) >= LEE_INVESTIGATION_EXPOSE
        or inv_progress >= 98.0
    )
    day, iso, _ = _u_current_meta(session)
    result = {
        "text": post_text,
        "risk": risk,
        "tone": tone["tone"],
        "market_key": market_key,
        "caught": caught,
        "day": day,
        "calendar_iso": iso,
        "handle": acct.get("handle"),
    }
    if caught:
        _apply_burner_exposure(session, result, fxs["bad"], tone)
    else:
        _apply_burner_success(session, result, fxs["ok"], tone)
    _publish_to_feed(session, result, tone)
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
        "can_post": (bool(acct.get("handle")) and not acct.get("exposed")) or _can_create_burner(session),
        "days_until_new_account": _days_until_new_burner(session),
        "recent_hype_posts": _recent_hype_count(session),
        "risky_words": {**RISKY_WORD_WEIGHTS, **_contextual_risk_words(session)},
        "storyline_names": sorted(_storyline_name_parts(session).keys()),
    }


def tick_burner_investigation_daily(session: Any) -> None:
    """Called from narrative daily pass — passive investigation progress."""
    acct = _ensure_burner_account(session)
    # Suspicion cools off when the account goes quiet.
    acct["suspicion_score"] = max(0.0, float(acct.get("suspicion_score") or 0) - 0.35)
    if float(acct.get("suspicion_score") or 0) < LEE_INVESTIGATION_THRESHOLD:
        return
    inv = dict(getattr(session, "gm_burner_investigation", None) or {})
    if not inv:
        return
    inv["progress"] = min(100.0, float(inv.get("progress") or 0) + 1.2)
    session.gm_burner_investigation = inv
