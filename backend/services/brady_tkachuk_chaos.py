"""
Brady Tkachuk chaos mode (local franchise joke / house rule).

Forces 55 OVR, CANCER name tag, deeply negative trade value, rehab/drinking
storyline focus, and a soft teammate rating hit while he is on the roster.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

BRADY_NHL_ID = 8480801
BRADY_TARGET_OVR = 55.0 / 99.0  # display 55
CANCER_SUFFIX = " ☢ CANCER"
TEAMMATE_RATING_SCALE = 0.945  # ~5.5% attribute hit for teammates
TRADE_VALUE_TOTAL = -42.0


def is_brady_tkachuk(player: Any) -> bool:
    if player is None:
        return False
    if bool(getattr(player, "brady_tkachuk_chaos", False)):
        return True
    try:
        nhl_id = int(getattr(player, "nhl_player_id", 0) or 0)
    except Exception:
        nhl_id = 0
    if nhl_id == BRADY_NHL_ID:
        return True
    try:
        ext = str(getattr(player, "external_player_id", "") or "")
        if ext.isdigit() and int(ext) == BRADY_NHL_ID:
            return True
    except Exception:
        pass
    ident = getattr(player, "identity", None)
    name = str(getattr(ident, "name", "") or getattr(player, "name", "") or "").lower()
    clean = name.replace(CANCER_SUFFIX.lower(), "").strip()
    return "brady" in clean and "tkachuk" in clean


def _ensure_cancer_name(player: Any) -> None:
    ident = getattr(player, "identity", None)
    if ident is None:
        return
    raw = str(getattr(ident, "name", "") or "").strip()
    if not raw:
        return
    if "CANCER" in raw.upper():
        return
    # Strip prior suffix variants then stamp the giant tag.
    base = raw
    for token in ("☢ CANCER", "[CANCER]", "CANCER"):
        base = base.replace(token, "").replace(token.lower(), "")
    base = " ".join(base.split()).strip()
    ident.name = f"{base}{CANCER_SUFFIX}"


def force_brady_overall(player: Any) -> float:
    """Pin attribute-derived OVR to 55 and stamp chaos flags."""
    from services.real_nhl_roster_importer import align_attribute_ovr_to_target

    _ensure_cancer_name(player)
    setattr(player, "brady_tkachuk_chaos", True)
    setattr(player, "locker_room_cancer", True)
    setattr(player, "locker_room_disruptor", True)
    setattr(player, "name_tags", ["CANCER"])
    setattr(player, "display_name_tag", "CANCER")
    setattr(player, "_systemic_trade_value_mult", -8.0)
    setattr(player, "real_nhl_target_ovr", round(BRADY_TARGET_OVR, 4))
    # Kill upside so development / league floors can't rescue him.
    try:
        setattr(player, "potential", BRADY_TARGET_OVR)
    except Exception:
        pass
    try:
        psych = getattr(player, "psych", None)
        if psych is not None:
            for attr, val in (
                ("morale", 0.18),
                ("confidence", 0.22),
                ("locker_room_fit", 0.05),
                ("coach_relationship", 0.12),
                ("role_satisfaction", 0.15),
            ):
                if hasattr(psych, attr):
                    setattr(psych, attr, val)
    except Exception:
        pass
    try:
        traits = getattr(player, "traits", None)
        if traits is not None:
            for attr, val in (
                ("volatility", 0.95),
                ("ego", 0.92),
                ("coachability", 0.08),
                ("work_ethic", 0.12),
                ("loyalty", 0.15),
            ):
                if hasattr(traits, attr):
                    setattr(traits, attr, val)
    except Exception:
        pass
    return float(align_attribute_ovr_to_target(player, BRADY_TARGET_OVR, rounds=14))


def degrade_teammates_for_brady(team: Any, *, scale: float = TEAMMATE_RATING_SCALE) -> int:
    """Soft attribute hit on depth players sharing Brady's dressing room.

    Franchise / first-line talents are exempt — the house-rule joke must not
    silently destroy Reinhart / Bobrovsky-tier teammates after a real-NHL import.
    """
    from app.sim_engine.engine import _scale_player_ratings
    from app.sim_engine.entities.player import persist_recomputed_ovr

    roster = list(getattr(team, "roster", None) or [])
    touched = 0
    for p in roster:
        if is_brady_tkachuk(p):
            continue
        if getattr(p, "in_minors", False) or getattr(p, "buried", False):
            continue
        if bool(getattr(p, "_brady_teammate_hit", False)):
            continue
        try:
            tgt = float(getattr(p, "real_nhl_target_ovr", 0) or 0)
            fn = getattr(p, "ovr", None)
            cur = float(fn()) if callable(fn) else float(tgt or 0)
            if tgt >= 0.84 or cur >= 0.84:
                continue
        except Exception:
            pass
        try:
            _scale_player_ratings(p, float(scale))
            persist_recomputed_ovr(p)
            setattr(p, "_brady_teammate_hit", True)
            touched += 1
        except Exception:
            continue
    return touched


def apply_brady_chaos_to_league(teams: List[Any]) -> Dict[str, Any]:
    """After real-NHL import: pin Brady, stamp CANCER, drag teammates."""
    found = None
    host = None
    for team in teams or []:
        for p in list(getattr(team, "roster", None) or []) + list(getattr(team, "ahl_roster", None) or []):
            if is_brady_tkachuk(p):
                found = p
                host = team
                break
        if found is not None:
            break
    if found is None:
        return {"ok": False, "found": False}
    force_brady_overall(found)
    tm_hit = degrade_teammates_for_brady(host) if host is not None else 0
    return {
        "ok": True,
        "found": True,
        "player_id": str(getattr(found, "id", "") or ""),
        "name": str(getattr(getattr(found, "identity", None), "name", "") or ""),
        "ovr": round(BRADY_TARGET_OVR * 99.0, 1),
        "teammates_hit": tm_hit,
        "team": str(
            getattr(host, "abbreviation", None)
            or getattr(host, "abbr", None)
            or getattr(host, "city", "")
            or ""
        ),
    }


def brady_trade_value_override(player: Any) -> Optional[Dict[str, Any]]:
    if not is_brady_tkachuk(player):
        return None
    return {
        "total": TRADE_VALUE_TOTAL,
        "base": TRADE_VALUE_TOTAL,
        "context_mod": 0.0,
        "tier": "negative",
        "brady_tkachuk_chaos": True,
        "risk_flags": [
            "Locker-room CANCER",
            "Active substance / rehab storyline",
            "Negative asset — clubs pay to move him",
        ],
        "contract_flags": ["Toxic asset"],
        "explain": [
            "House rule: Brady Tkachuk is a negative trade asset",
            "CANCER tag depresses every offer sheet",
        ],
        "retained_pct_supported": True,
    }


def brady_storyline_events(*, team_abbr: str = "OTT", calendar_iso: str = "") -> List[Dict[str, Any]]:
    """Opening-night / ongoing Brady rehab + drinking arcs."""
    day = calendar_iso or "2025-09-15"
    abbr = (team_abbr or "OTT").upper()
    name = f"Brady Tkachuk{CANCER_SUFFIX}"
    return [
        {
            "type": "locker_room",
            "tone": "crisis",
            "priority": "CRITICAL",
            "date": day,
            "calendar_iso": day,
            "team": abbr,
            "team_abbrev": abbr,
            "player_name": name,
            "players": [name],
            "headline": f"{name} spotted stumbling out of ByWard Market at 3 a.m. — teammates furious",
            "details": (
                "Multiple sources say the captain is drinking himself into a hole. "
                "Locker-room trust is collapsing. Front office reviewing intervention options."
            ),
            "summary": "Captain nightlife spiral; chemistry cratering.",
            "stable_key": f"brady:drinking:{abbr}:{day}",
            "cause": "Off-ice drinking spiral",
            "effect_summary": "Morale crater; teammates play worse; trade value deeply negative.",
        },
        {
            "type": "conduct",
            "tone": "crisis",
            "priority": "CRITICAL",
            "date": day,
            "calendar_iso": day,
            "team": abbr,
            "team_abbrev": abbr,
            "player_name": name,
            "players": [name],
            "headline": f"{name} checks himself into the NHLPA Player Assistance / drug rehab program",
            "details": (
                "After another weekend binge, Brady entered the league's substance program. "
                "He remains rostered as a 55-OVR locker-room CANCER while the arc plays out."
            ),
            "summary": "Voluntary rehab admission; focal franchise storyline.",
            "stable_key": f"brady:rehab:{abbr}:{day}",
            "cause": "Substance issues / self check-in",
            "effect_summary": "Focal storyline pressure; availability and leadership compromised.",
        },
        {
            "type": "locker_room",
            "tone": "negative",
            "priority": "HIGH",
            "date": day,
            "calendar_iso": day,
            "team": abbr,
            "team_abbrev": abbr,
            "player_name": name,
            "players": [name],
            "headline": f"Veterans say {name} is a dressing-room CANCER — 'he makes everyone worse'",
            "details": (
                "Anonymous teammates report that Brady's spiral is dragging attribute focus "
                "and compete level across the roster. Opposing GMs price him as a negative asset."
            ),
            "summary": "Teammate attribute hit active while Brady is on the roster.",
            "stable_key": f"brady:cancer-tag:{abbr}:{day}",
            "cause": "Locker-room cancer label",
            "effect_summary": "Teammates rated down; trade market treats him as toxic.",
        },
    ]


def inject_brady_storylines(session: Any, *, team_abbr: str = "OTT") -> int:
    """Push Brady arcs into the franchise storyline feed."""
    cal = ""
    try:
        idx = int(getattr(session, "calendar_index", 0) or 0)
        days = list(getattr(session, "nhl_calendar", None) or [])
        if 0 <= idx < len(days):
            cal = str(days[idx].get("iso") or days[idx].get("date") or "")
    except Exception:
        cal = ""
    n = 0
    events = getattr(session, "storyline_events", None)
    if events is None:
        session.storyline_events = []
        events = session.storyline_events
    for ev in brady_storyline_events(team_abbr=team_abbr, calendar_iso=cal):
        try:
            # Prefer franchise normalizer when available (avoids circular import at module load).
            try:
                from services import franchise_sim as _fs

                _fs._record_storyline(session, ev)
            except Exception:
                events.append(ev)
            n += 1
        except Exception:
            continue
    try:
        notes = getattr(session, "notifications", None)
        if isinstance(notes, list):
            notes.append(
                f"STORYLINE FOCUS: Brady Tkachuk{CANCER_SUFFIX} — drinking spiral + rehab check-in. "
                "Teammates playing worse. Trade value negative."
            )
    except Exception:
        pass
    return n


def display_name_with_cancer_tag(name: str, player: Any = None) -> str:
    raw = str(name or "").strip()
    if player is not None and not is_brady_tkachuk(player):
        return raw
    if player is None and "tkachuk" not in raw.lower():
        return raw
    if "CANCER" in raw.upper():
        return raw
    base = raw
    for token in ("☢ CANCER", "[CANCER]", "CANCER"):
        base = base.replace(token, "")
    base = " ".join(base.split()).strip() or "Brady Tkachuk"
    return f"{base}{CANCER_SUFFIX}"
