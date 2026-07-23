"""
Draft rights engine: league-specific NHL rights independent of active club affiliation.

Drafting transfers NHL rights only. The player keeps current_team_id / current_league_id
on his junior, NCAA, or European club while nhl_rights_team_id records organizational claim.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

RIGHTS_STATUS = (
    "drafted_unsigned",
    "exclusive_rights",
    "indefinite_european_rights",
    "college_rights",
    "rights_expiring",
    "rights_relinquished",
    "draft_reentry",
    "unrestricted_free_agent",
    "signed",
)

RIGHTS_TYPE = (
    "chl_exclusive",
    "ncaa_college",
    "european_exclusive",
    "european_indefinite",
    "ushl_junior",
    "other_amateur",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _league_code(block: Optional[Dict], entry: Optional[Dict] = None) -> str:
    code = str((block or {}).get("league_code") or (entry or {}).get("league_code") or "").upper()
    if not code:
        code = str((entry or {}).get("league") or (entry or {}).get("league_name") or "").upper()
    return code


def development_path_for(entry: Optional[Dict] = None, block: Optional[Dict] = None) -> str:
    code = _league_code(block, entry)
    if code.startswith("NCAA") or "NCAA" in code:
        return "NCAA"
    if code.startswith("EU_") or any(x in code for x in ("SHL", "LIIGA", "DEL", "KHL", "CZE", "SVK", "SUI")):
        return "Europe"
    if code.startswith("CHL") or code in ("OHL", "WHL", "QMJHL", "USHL"):
        return "Junior"
    return "Junior"


def infer_rights_type(dev_path: str, league_code: str = "") -> str:
    code = (league_code or "").upper()
    path = (dev_path or "").upper()
    if path == "NCAA" or code.startswith("NCAA") or "NCAA" in code:
        return "ncaa_college"
    if path == "EUROPE" or code.startswith("EU_"):
        # First-year exclusive European claim; later passes may convert to indefinite.
        return "european_exclusive"
    if code == "USHL":
        return "ushl_junior"
    if path == "JUNIOR" or code.startswith("CHL") or code in ("OHL", "WHL", "QMJHL"):
        return "chl_exclusive"
    return "other_amateur"


def _rights_status_for_type(rights_type: str) -> str:
    if rights_type == "ncaa_college":
        return "college_rights"
    if rights_type in ("european_exclusive", "european_indefinite"):
        return "indefinite_european_rights" if rights_type == "european_indefinite" else "exclusive_rights"
    if rights_type in ("chl_exclusive", "ushl_junior"):
        return "exclusive_rights"
    return "drafted_unsigned"


def compute_rights_window(
    *,
    rights_type: str,
    draft_year: int,
    player_age: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Approximate CBA-style windows (game abstraction, not literal CBA counsel).

    CHL-style: exclusive claim through draft_year + 2 (June 1 signing deadline style).
    NCAA: retained while enrolled; model expiry at draft_year + 4 (typical graduation window).
    European exclusive: draft_year + 2, then may become indefinite or re-entry.
    """
    dy = int(draft_year)
    age = int(player_age or 18)
    if rights_type == "ncaa_college":
        expiry_year = dy + max(2, 4 - max(0, age - 18))
        return {
            "rights_expiry_year": expiry_year,
            "rights_signing_deadline": f"{expiry_year}-08-15",
            "rights_expiry_date": f"{expiry_year}-08-15",
            "rights_is_exclusive": False,
            "rights_can_reenter_draft": True,
        }
    if rights_type in ("chl_exclusive", "ushl_junior"):
        expiry_year = dy + 2
        return {
            "rights_expiry_year": expiry_year,
            "rights_signing_deadline": f"{expiry_year}-06-01",
            "rights_expiry_date": f"{expiry_year}-06-01",
            "rights_is_exclusive": True,
            "rights_can_reenter_draft": True,
        }
    if rights_type == "european_exclusive":
        expiry_year = dy + 2
        return {
            "rights_expiry_year": expiry_year,
            "rights_signing_deadline": f"{expiry_year}-06-15",
            "rights_expiry_date": f"{expiry_year}-06-15",
            "rights_is_exclusive": True,
            "rights_can_reenter_draft": True,
        }
    if rights_type == "european_indefinite":
        return {
            "rights_expiry_year": None,
            "rights_signing_deadline": None,
            "rights_expiry_date": None,
            "rights_is_exclusive": True,
            "rights_can_reenter_draft": False,
        }
    expiry_year = dy + 2
    return {
        "rights_expiry_year": expiry_year,
        "rights_signing_deadline": f"{expiry_year}-06-01",
        "rights_expiry_date": f"{expiry_year}-06-01",
        "rights_is_exclusive": True,
        "rights_can_reenter_draft": True,
    }


def build_draft_rights_fields(
    *,
    team_id: str,
    draft_year: int,
    block: Optional[Dict] = None,
    entry: Optional[Dict] = None,
    player: Any = None,
    acquired_date: Optional[str] = None,
) -> Dict[str, Any]:
    code = _league_code(block, entry)
    dev_path = development_path_for(entry, block)
    rights_type = infer_rights_type(dev_path, code)
    age = None
    if player is not None:
        try:
            age = int(getattr(player, "age", None) or (entry or {}).get("age") or 18)
        except Exception:
            age = int((entry or {}).get("age") or 18)
    else:
        age = int((entry or {}).get("age") or 18)
    window = compute_rights_window(rights_type=rights_type, draft_year=draft_year, player_age=age)
    status = _rights_status_for_type(rights_type)
    return {
        "rights_type": rights_type,
        "rights_acquired_date": acquired_date or _now_iso(),
        "rights_expiry_date": window.get("rights_expiry_date"),
        "rights_expiry_year": window.get("rights_expiry_year"),
        "rights_signing_deadline": window.get("rights_signing_deadline"),
        "rights_team_id": str(team_id),
        "rights_status": status,
        "rights_source_league": code or dev_path,
        "rights_is_exclusive": bool(window.get("rights_is_exclusive")),
        "rights_can_reenter_draft": bool(window.get("rights_can_reenter_draft")),
        "development_path": dev_path,
        "nhl_rights_team_id": str(team_id),
        "organizational_status": "unsigned_drafted",
        "signed_status": "unsigned",
    }


def affiliation_snapshot(
    player: Any,
    block: Optional[Dict] = None,
    tm: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Capture active club/league without overwriting NHL rights ownership."""
    current_team_id = None
    if tm is not None:
        current_team_id = str(tm.get("team_id") or tm.get("id") or tm.get("name") or "")
    if not current_team_id:
        current_team_id = str(
            getattr(player, "current_team_id", None)
            or getattr(player, "team_id", None)
            or getattr(player, "dev_team_id", None)
            or ""
        )
    current_league_id = None
    if block is not None:
        current_league_id = str(block.get("league_code") or block.get("id") or block.get("name") or "")
    if not current_league_id:
        current_league_id = str(
            getattr(player, "current_league_id", None)
            or getattr(player, "league_id", None)
            or getattr(player, "dev_league_id", None)
            or ""
        )
    return {
        "current_team_id": current_team_id or None,
        "current_league_id": current_league_id or None,
    }


def apply_draft_rights(
    player: Any,
    *,
    nhl_team_id: str,
    draft_year: int,
    pick_meta: Optional[Dict[str, Any]] = None,
    block: Optional[Dict] = None,
    tm: Optional[Dict] = None,
    entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assign NHL draft rights while leaving the player on his development club.

    Does NOT set team_id to the NHL organization. Does NOT remove from junior/NCAA/EU roster.
    """
    pick_meta = pick_meta or {}
    entry = entry or {}
    aff = affiliation_snapshot(player, block, tm)
    rights = build_draft_rights_fields(
        team_id=nhl_team_id,
        draft_year=draft_year,
        block=block,
        entry=entry,
        player=player,
    )
    overall = int(pick_meta.get("overall_pick") or 1)

    setattr(player, "drafted", True)
    setattr(player, "drafted_by", str(nhl_team_id))
    setattr(player, "draft_team_id", str(nhl_team_id))
    setattr(player, "draft_year", int(draft_year))
    setattr(player, "draft_round", int(pick_meta.get("round") or 1))
    setattr(player, "draft_pick_number", int(pick_meta.get("pick_in_round") or 1))
    setattr(player, "draft_overall_pick", overall)

    setattr(player, "current_team_id", aff["current_team_id"])
    setattr(player, "current_league_id", aff["current_league_id"])
    # Keep active club id as the development team — never the NHL org.
    if aff["current_team_id"]:
        setattr(player, "team_id", aff["current_team_id"])
    setattr(player, "dev_team_id", aff["current_team_id"])
    setattr(player, "dev_league_id", aff["current_league_id"])

    for k, v in rights.items():
        setattr(player, k, v)

    setattr(player, "prospect_status", "org_prospect")
    setattr(player, "post_draft_league", rights["development_path"])
    setattr(player, "entry_level_contract_eligible", True)
    setattr(player, "status", "prospect")
    setattr(player, "elc_slide_eligible", True)
    setattr(player, "elc_slide_years_remaining", 1)
    setattr(player, "nhl_games_played_this_season", int(getattr(player, "nhl_games_played_this_season", 0) or 0))
    setattr(player, "contract_burned", False)

    return {**aff, **rights}


def available_rights_actions(player: Any) -> List[Dict[str, Any]]:
    """Legal decision options for Prospect Rights — never invent illegal paths."""
    signed = str(getattr(player, "signed_status", "unsigned") or "unsigned").lower()
    status = str(getattr(player, "rights_status", "") or "").lower()
    path = str(getattr(player, "development_path", "") or getattr(player, "post_draft_league", "") or "").upper()
    rights_type = str(getattr(player, "rights_type", "") or "").lower()
    age = _safe_player_age(player)
    actions: List[Dict[str, Any]] = []
    if signed == "signed" or status == "signed":
        return [{"id": "already_signed", "label": "Already signed", "enabled": False}]
    if status in ("rights_relinquished", "unrestricted_free_agent", "draft_reentry"):
        return [{"id": "no_rights", "label": "No active rights", "enabled": False}]

    elc_ok = bool(getattr(player, "entry_level_contract_eligible", True))
    if elc_ok:
        actions.append({
            "id": "sign_elc",
            "label": "Sign to ELC",
            "enabled": True,
            "starts_elc": True,
            "uses_contract_slot": True,
            "can_slide": bool(getattr(player, "elc_slide_eligible", False)),
        })
    actions.append({"id": "keep_unsigned", "label": "Keep unsigned", "enabled": True})
    if "NCAA" in path or rights_type == "ncaa_college":
        actions.append({"id": "keep_college", "label": "Keep in college", "enabled": True})
    elif "EUROPE" in path or rights_type.startswith("european"):
        actions.append({"id": "keep_europe", "label": "Keep in Europe", "enabled": True})
    else:
        actions.append({"id": "return_junior", "label": "Return to junior", "enabled": True})
    # AHL only when age/path supports a pro assignment (CHL return rules otherwise).
    ahl_ok = age >= 20 or "EUROPE" in path or rights_type.startswith("european")
    actions.append({
        "id": "assign_ahl",
        "label": "Assign to AHL",
        "enabled": ahl_ok,
        "blocked_reason": None if ahl_ok else "Not AHL-eligible under junior return rules",
    })
    actions.append({"id": "invite_camp", "label": "Invite to training camp", "enabled": True})
    actions.append({"id": "delay", "label": "Delay decision", "enabled": True})
    expiry = getattr(player, "rights_expiry_year", None)
    if expiry is not None:
        actions.append({
            "id": "allow_expire",
            "label": "Allow rights to expire",
            "enabled": True,
            "warning": "Relinquishes organizational claim at deadline",
        })
    return actions


def _safe_player_ovr(player: Any) -> float:
    """Resolve overall when it may be a property, method, or missing."""
    for key in ("overall", "ovr", "current_ovr", "true_ovr"):
        raw = getattr(player, key, None)
        if raw is None:
            continue
        if callable(raw) and not isinstance(raw, (int, float)):
            try:
                raw = raw()
            except TypeError:
                continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def _safe_player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    for raw in (
        getattr(ident, "age", None) if ident is not None else None,
        getattr(player, "age", None),
    ):
        if raw is None:
            continue
        if callable(raw) and not isinstance(raw, (int, float)):
            try:
                raw = raw()
            except TypeError:
                continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return 18


def development_environment_assessment(player: Any) -> Dict[str, Any]:
    """Public development-environment grade from observable signals only."""
    path = str(getattr(player, "development_path", "") or "").upper()
    age = _safe_player_age(player)
    ovr = _safe_player_ovr(player)
    reasons: List[str] = []
    grade = "acceptable"
    if ovr >= 70 and age <= 21:
        grade = "ideal"
        reasons.append("Ready for meaningful pro minutes")
    elif ovr >= 64:
        grade = "good"
        reasons.append("Tools project a usable role with ice time")
    elif ovr > 0 and ovr < 55 and "NCAA" not in path and "JUNIOR" not in path:
        grade = "risky"
        reasons.append("Current ability may be overmatched")
    elif age <= 18 and ("JUNIOR" in path or "NCAA" in path):
        grade = "good"
        reasons.append("Age-appropriate development league")
    elif age >= 21 and ("JUNIOR" in path):
        grade = "poor"
        reasons.append("May have outgrown junior competition")
    else:
        reasons.append("Standard developmental placement")
    return {"grade": grade, "reasons": reasons[:3], "path": path or None}


def rights_card_payload(player: Any) -> Dict[str, Any]:
    expiry = getattr(player, "rights_expiry_year", None)
    path = str(getattr(player, "development_path", "") or getattr(player, "post_draft_league", "") or "")
    status = str(getattr(player, "rights_status", "") or "")
    signed = str(getattr(player, "signed_status", "unsigned") or "unsigned")
    env = development_environment_assessment(player)
    actions = available_rights_actions(player)
    recommended = next((a for a in actions if a.get("enabled") and a.get("id") != "allow_expire"), None)
    return {
        "rights_through": expiry,
        "rights_status": status,
        "rights_type": getattr(player, "rights_type", None),
        "rights_signing_deadline": getattr(player, "rights_signing_deadline", None),
        "returning_to": path,
        "expected_role": getattr(player, "expected_role", None) or "Org prospect",
        "elc_decision": "Signed" if signed == "signed" else "Unsigned",
        "eta": getattr(player, "nhl_eta", None),
        "organizational_status": getattr(player, "organizational_status", None),
        "current_league_id": getattr(player, "current_league_id", None),
        "current_team_id": getattr(player, "current_team_id", None),
        "nhl_rights_team_id": getattr(player, "nhl_rights_team_id", None)
        or getattr(player, "rights_team_id", None),
        "elc_slide_eligible": bool(getattr(player, "elc_slide_eligible", False)),
        "elc_slide_years_remaining": getattr(player, "elc_slide_years_remaining", None),
        "nhl_games_played_this_season": getattr(player, "nhl_games_played_this_season", 0),
        "contract_burned": bool(getattr(player, "contract_burned", False)),
        "entry_level_contract_eligible": bool(getattr(player, "entry_level_contract_eligible", True)),
        "available_actions": actions,
        "development_environment": env,
        "recommended_action": (recommended or {}).get("id"),
        "recommended_label": (recommended or {}).get("label"),
        "path_visual": _path_visual(path),
    }


def _path_visual(path: str) -> List[str]:
    p = (path or "").upper()
    if "NCAA" in p:
        return ["College", "AHL", "NHL"]
    if "EUROPE" in p:
        return ["Europe", "AHL", "NHL"]
    return ["Junior", "AHL", "NHL"]


def _iter_org_prospects(league: Any) -> List[Tuple[Any, Any]]:
    out: List[Tuple[Any, Any]] = []
    for team in getattr(league, "teams", None) or []:
        for p in list(getattr(team, "prospect_pool", None) or []):
            out.append((team, p))
        for entry in list(getattr(team, "reserve_list", None) or []):
            if not isinstance(entry, dict):
                continue
            # Prefer live player from org pools; registry resolution happens upstream.
            ref = entry.get("player_ref")
            if ref is not None and all(ref is not x for _, x in out):
                out.append((team, ref))
    return out


def process_draft_rights_deadlines(
    session: Any,
    league: Any,
    season_year: int,
) -> Dict[str, Any]:
    """
    Evaluate CHL/NCAA/European rights expiry, re-entry eligibility, relinquishment, and UFA.
    """
    sy = int(season_year)
    notifications: List[Dict[str, Any]] = []
    expired: List[Dict[str, Any]] = []
    reentry: List[Dict[str, Any]] = []
    extended_eu: List[Dict[str, Any]] = []
    reviewed = 0

    from services.draft_player_registry import get_player, ensure_players_by_id

    ensure_players_by_id(league)

    for team in getattr(league, "teams", None) or []:
        tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
        pool = list(getattr(team, "prospect_pool", None) or [])
        reserve = list(getattr(team, "reserve_list", None) or [])
        reserve_by_id = {str(e.get("player_id")): e for e in reserve if isinstance(e, dict)}

        for player in list(pool):
            pid = str(getattr(player, "id", "") or "")
            if not pid:
                continue
            if str(getattr(player, "signed_status", "") or "").lower() == "signed":
                setattr(player, "rights_status", "signed")
                continue
            if str(getattr(player, "rights_status", "") or "") in (
                "rights_relinquished",
                "unrestricted_free_agent",
                "draft_reentry",
            ):
                continue

            reviewed += 1
            rtype = str(getattr(player, "rights_type", "") or "")
            expiry_year = getattr(player, "rights_expiry_year", None)
            deadline = str(getattr(player, "rights_signing_deadline", "") or "")

            if expiry_year is not None and int(expiry_year) - sy <= 1 and int(expiry_year) >= sy:
                setattr(player, "rights_status", "rights_expiring")
                notifications.append({
                    "type": "signing_deadline",
                    "team_id": tid,
                    "player_id": pid,
                    "player_name": str(getattr(player, "name", "") or pid),
                    "rights_expiry_year": int(expiry_year),
                    "rights_signing_deadline": deadline,
                    "message": f"Rights expire {deadline or expiry_year}",
                })

            if expiry_year is None or int(expiry_year) > sy:
                continue

            # Expired
            can_reenter = bool(getattr(player, "rights_can_reenter_draft", True))
            if rtype == "european_exclusive":
                # Convert to indefinite European exclusive claim rather than auto-UFA.
                setattr(player, "rights_type", "european_indefinite")
                setattr(player, "rights_status", "indefinite_european_rights")
                setattr(player, "rights_expiry_year", None)
                setattr(player, "rights_expiry_date", None)
                setattr(player, "rights_signing_deadline", None)
                setattr(player, "rights_can_reenter_draft", False)
                extended_eu.append({"player_id": pid, "team_id": tid})
                continue

            if can_reenter:
                setattr(player, "rights_status", "draft_reentry")
                setattr(player, "organizational_status", "draft_reentry")
                reentry.append({"player_id": pid, "team_id": tid, "name": getattr(player, "name", "")})
            else:
                setattr(player, "rights_status", "unrestricted_free_agent")
                setattr(player, "organizational_status", "ufa")

            setattr(player, "nhl_rights_team_id", None)
            setattr(player, "rights_team_id", None)
            setattr(player, "rights_is_exclusive", False)
            setattr(player, "prospect_status", "free_agent")

            # Drop from org membership but leave active development affiliation intact.
            try:
                pool.remove(player)
                team.prospect_pool = pool
            except Exception:
                pass
            if pid in reserve_by_id:
                try:
                    team.reserve_list = [e for e in reserve if str(e.get("player_id")) != pid]
                except Exception:
                    pass

            expired.append({
                "player_id": pid,
                "former_team_id": tid,
                "outcome": "draft_reentry" if can_reenter else "unrestricted_free_agent",
            })
            notifications.append({
                "type": "rights_relinquished" if not can_reenter else "draft_reentry",
                "team_id": tid,
                "player_id": pid,
                "player_name": str(getattr(player, "name", "") or pid),
                "message": (
                    f"{getattr(player, 'name', pid)} eligible for draft re-entry"
                    if can_reenter
                    else f"{getattr(player, 'name', pid)} became an unrestricted free agent"
                ),
            })

            # Sync reserve row if still present elsewhere via registry
            _ = get_player(league, pid)

    payload = {
        "season_year": sy,
        "reviewed": reviewed,
        "expired": expired,
        "reentry_eligible": reentry,
        "european_indefinite": extended_eu,
        "notifications": notifications,
        "processed_at": _now_iso(),
    }
    try:
        session.draft_rights_review_payload = payload
    except Exception:
        pass
    return payload


def should_cpu_auto_sign_elc(
    player: Any,
    team: Any,
    *,
    season_year: int,
    league: Any = None,
) -> Tuple[bool, str]:
    """
    Automatic ELC only with a clear organizational reason — not blanket roster cleanup.
    """
    if str(getattr(player, "signed_status", "") or "").lower() == "signed":
        return False, "already_signed"
    if not bool(getattr(player, "entry_level_contract_eligible", False)):
        return False, "not_elc_eligible"

    expiry = getattr(player, "rights_expiry_year", None)
    status = str(getattr(player, "rights_status", "") or "")
    readiness_raw = getattr(player, "nhl_readiness", None) or getattr(player, "overall", None)
    if readiness_raw is None:
        # `ovr` is a method on real player entities — resolve it like the rest of
        # the codebase instead of calling float() on a bound method.
        ov = getattr(player, "ovr", None)
        if callable(ov):
            try:
                ov = ov()
            except Exception:
                ov = None
        readiness_raw = ov
    try:
        readiness = float(readiness_raw if readiness_raw is not None else 50)
    except (TypeError, ValueError):
        readiness = 50.0
    if readiness <= 1.5:  # 0–1 scale from ovr() → 0–99 scale
        readiness *= 99.0
    age = int(getattr(player, "age", 18) or 18)
    path = str(getattr(player, "development_path", "") or "")

    # Rights about to expire / marked expiring
    if status == "rights_expiring" or (expiry is not None and int(expiry) <= int(season_year) + 1):
        return True, "rights_expiration_risk"

    # Clearly NHL-ready and not in multi-year NCAA return
    if readiness >= 72 and path != "NCAA":
        return True, "prospect_readiness"

    # Older juniors / European pros ready to turn pro
    if age >= 21 and readiness >= 64 and path in ("Junior", "Europe"):
        return True, "age_and_readiness"

    # Cap / slots: do not auto-sign if org is near contract limit
    try:
        from services.contract_economy import validate_contract_slots

        slots = validate_contract_slots(team, league, additional=1)
        if not slots.get("ok"):
            return False, "contract_slots_full"
    except Exception:
        pass

    return False, "no_clear_reason"
