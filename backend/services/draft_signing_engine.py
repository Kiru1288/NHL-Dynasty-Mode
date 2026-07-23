"""
Prospect ELC signing decisions — players do not always accept immediately.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional


def _rng(parts: Any) -> float:
    raw = ":".join(str(p) for p in parts) if isinstance(parts, (list, tuple)) else str(parts)
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def evaluate_elc_signing_decision(
    player: Any,
    team: Any,
    *,
    season_year: int,
    promote_to_nhl: bool = False,
    development_promise: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return acceptance likelihood and a deterministic accept/decline for this season context.
    """
    pid = str(getattr(player, "id", "") or "")
    tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
    age = int(getattr(player, "age", 18) or 18)
    path = str(getattr(player, "development_path", "") or "")
    readiness = float(getattr(player, "nhl_readiness", None) or getattr(player, "overall", None) or 55)
    expiry = getattr(player, "rights_expiry_year", None)
    relationship = float(getattr(player, "org_relationship", None) or 0.55)
    ncaa = bool(getattr(player, "ncaa_commitment", False) or path.upper() == "NCAA")
    eu_contract = bool(getattr(player, "european_contract", False) or path.upper() == "EUROPE")
    willingness = getattr(player, "willingness_to_sign", None)

    score = 0.55
    reasons = []

    if willingness is False:
        score -= 0.35
        reasons.append("prospect signaled low willingness to sign")
    elif willingness is True:
        score += 0.15
        reasons.append("prospect is motivated to turn pro")

    if ncaa and age <= 21 and not promote_to_nhl:
        score -= 0.22
        reasons.append("NCAA return remains attractive")
    if eu_contract and readiness < 70:
        score -= 0.18
        reasons.append("European contract still in play")

    if expiry is not None and int(expiry) <= int(season_year) + 1:
        score += 0.25
        reasons.append("rights deadline pressure")

    if readiness >= 72:
        score += 0.18
        reasons.append("NHL readiness supports turning pro")
    elif readiness < 58:
        score -= 0.12
        reasons.append("not yet ready for pro hockey")

    if development_promise in ("top_six_track", "ahl_featured", "protected_role"):
        score += 0.12
        reasons.append("strong development promise")
    elif development_promise == "depth_only":
        score -= 0.08
        reasons.append("unclear path")

    score += (relationship - 0.5) * 0.3
    score += (_rng((pid, tid, season_year, "sign")) - 0.5) * 0.12

    # Contract-slot awareness
    try:
        from services.contract_economy import validate_contract_slots

        slots = validate_contract_slots(team, getattr(team, "_league_ref", None), additional=1)
        if not slots.get("ok"):
            return {
                "ok": False,
                "accepted": False,
                "reason": slots.get("reason") or "contract_slots_full",
                "score": round(score, 3),
                "reasons": reasons,
                "contract_slots": slots,
            }
        slot_payload = slots
    except Exception:
        slot_payload = {}

    accepted = score >= 0.48
    if ncaa and age <= 20 and readiness < 68 and (expiry is None or int(expiry) > season_year + 1):
        accepted = False
        reasons.append("chose to remain in NCAA")

    return {
        "ok": True,
        "accepted": accepted,
        "score": round(max(0.0, min(1.0, score)), 3),
        "reasons": reasons,
        "path": path,
        "elc_slide_eligible": bool(getattr(player, "elc_slide_eligible", True)),
        "elc_slide_years_remaining": getattr(player, "elc_slide_years_remaining", 1),
        "contract_slots": slot_payload,
        "decision": "accept" if accepted else "decline",
    }


def attempt_sign_elc_with_decision(
    session: Any,
    player: Any,
    team: Any,
    *,
    season_year: int,
    promote_to_nhl: bool = False,
    development_promise: Optional[str] = None,
) -> Dict[str, Any]:
    decision = evaluate_elc_signing_decision(
        player,
        team,
        season_year=season_year,
        promote_to_nhl=promote_to_nhl,
        development_promise=development_promise,
    )
    if not decision.get("accepted"):
        return {
            "ok": False,
            "signed": False,
            "reason": "prospect_declined",
            "decision": decision,
        }

    from services.contract_economy import sign_elc

    league = getattr(getattr(session, "sim", None), "league", None)
    result = sign_elc(player, team, league, season_year, promote_to_nhl=promote_to_nhl)
    if result.get("ok"):
        try:
            setattr(player, "rights_status", "signed")
            setattr(player, "organizational_status", "signed")
            setattr(player, "signed_status", "signed")
            if bool(getattr(player, "elc_slide_eligible", False)):
                setattr(player, "contract_burned", False)
        except Exception:
            pass
    return {**result, "signed": bool(result.get("ok")), "decision": decision}
