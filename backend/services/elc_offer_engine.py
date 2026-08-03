"""
Structured ELC offer construction, legal term validation, preview, and submit.

Currency: millions of USD (suffix _m), matching contract_economy.
Authority: builds offers consumed by contract_economy.assign_elc_contract.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Optional, Tuple

# League ELC rules (millions). Keep aligned with contract_economy.ELC_AAV_M.
ELC_AAV_M = 0.95
ELC_MINOR_SALARY_M = 0.085
ELC_MAX_SIGNING_BONUS_TOTAL_M = 0.285  # ~$95k × 3
ELC_MAX_SCHEDULE_A_M = 0.2125
ELC_MAX_SCHEDULE_B_M = 0.2125
ELC_SLIDE_GAMES_THRESHOLD = 10
CONTRACT_SCHEMA_VERSION = 2

TEMPLATE_IDS = (
    "standard_elc",
    "maximum_bonus_elc",
    "no_bonus_elc",
    "two_year_bridge",
)


def _pid(player: Any) -> str:
    return str(getattr(player, "id", None) or getattr(player, "player_id", "") or "")


def _age(player: Any) -> int:
    try:
        return int(getattr(player, "age", 18) or 18)
    except (TypeError, ValueError):
        return 18


def _path(player: Any) -> str:
    return str(
        getattr(player, "development_path", None)
        or getattr(player, "post_draft_league", None)
        or getattr(player, "current_league_id", None)
        or ""
    ).upper()


def _rng(parts: Any) -> float:
    raw = ":".join(str(p) for p in parts) if isinstance(parts, (list, tuple)) else str(parts)
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def validation_result(
    *,
    allowed: bool,
    error_code: str = "",
    title: str = "",
    user_message: str = "",
    technical_message: str = "",
    blocking_reasons: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "allowed": bool(allowed),
        "ok": bool(allowed),
        "error_code": error_code or None,
        "title": title or None,
        "user_message": user_message or None,
        "technical_message": technical_message or None,
        "blocking_reasons": list(blocking_reasons or []),
        "warnings": list(warnings or []),
        **extra,
    }


def dollars_from_m(value_m: float) -> int:
    return int(round(float(value_m or 0.0) * 1_000_000))


def format_money_display(value_m: float) -> str:
    d = dollars_from_m(value_m)
    if d >= 1_000_000:
        return f"${d / 1_000_000:.3f}M".replace(".000M", "M").replace(".950M", ".95M")
    return f"${d:,}"


def legal_elc_terms(player: Any, season_year: int) -> Dict[str, Any]:
    """Player-specific legal ELC structure — frontend must not invent terms."""
    age = _age(player)
    path = _path(player)
    signed = str(getattr(player, "signed_status", "unsigned") or "").lower() == "signed"
    elc_ok = bool(getattr(player, "entry_level_contract_eligible", True)) and not signed
    try:
        from services.contract_economy import has_true_elc_contract, has_active_contract

        if has_true_elc_contract(player) or has_active_contract(player):
            elc_ok = False
    except Exception:
        pass

    # Simplified CBA-style: age 18–21 typically 3 years; older bridge often 1–2.
    if age <= 20:
        allowed_years = [3]
        recommended = 3
        term_reason = "Standard entry-level term for under-21 signees"
    elif age == 21:
        allowed_years = [2, 3]
        recommended = 3
        term_reason = "Age 21 may take a 2- or 3-year ELC"
    elif age <= 24:
        allowed_years = [1, 2]
        recommended = 2
        term_reason = "Older prospect ELC / bridge-style term"
    else:
        allowed_years = [1]
        recommended = 1
        term_reason = "Maximum entry-level style term for this age"

    ncaa = "NCAA" in path
    europe = "EUROPE" in path or "KHL" in path or "SHL" in path or "LIIGA" in path
    junior = any(x in path for x in ("JUNIOR", "CHL", "OHL", "WHL", "QMJHL", "USHL"))
    ahl_ok = age >= 20 or europe
    slide_eligible = bool(getattr(player, "elc_slide_eligible", age <= 20 and not europe))
    if age >= 22:
        slide_eligible = False

    return {
        "elc_eligible": elc_ok,
        "legal_terms": allowed_years if elc_ok else [],
        "recommended_term": recommended if elc_ok else None,
        "required_term": allowed_years[0] if elc_ok and len(allowed_years) == 1 else None,
        "term_reason": term_reason,
        "aav_m": ELC_AAV_M,
        "aav_display": format_money_display(ELC_AAV_M),
        "minor_salary_m": ELC_MINOR_SALARY_M,
        "minor_salary_display": format_money_display(ELC_MINOR_SALARY_M),
        "max_signing_bonus_total_m": ELC_MAX_SIGNING_BONUS_TOTAL_M,
        "max_schedule_a_m": ELC_MAX_SCHEDULE_A_M,
        "max_schedule_b_m": ELC_MAX_SCHEDULE_B_M,
        "slide_eligible": slide_eligible,
        "slide_games_threshold": ELC_SLIDE_GAMES_THRESHOLD,
        "slide_years_remaining": getattr(player, "elc_slide_years_remaining", 1 if slide_eligible else 0),
        "can_return_junior": junior and age < 21,
        "can_assign_ahl": ahl_ok,
        "can_remain_europe": europe,
        "can_remain_ncaa": ncaa,
        "assignment_options": _assignment_options(player, age, path, junior, europe, ncaa, ahl_ok),
        "contract_start_season": int(season_year),
        "bonus_descriptions": {
            "signing_bonus": "Guaranteed payment included in the contract structure",
            "schedule_a": "Individual rookie achievement bonuses",
            "schedule_b": "Major league-wide achievement bonuses",
        },
    }


def _assignment_options(
    player: Any, age: int, path: str, junior: bool, europe: bool, ncaa: bool, ahl_ok: bool
) -> List[Dict[str, Any]]:
    opts: List[Dict[str, Any]] = []
    if junior and age < 21:
        opts.append({"id": "return_junior", "label": "Return to junior", "enabled": True})
    if ncaa:
        opts.append({"id": "keep_college", "label": "Remain NCAA", "enabled": True})
    if europe:
        opts.append({"id": "keep_europe", "label": "Remain in Europe", "enabled": True})
    opts.append({
        "id": "assign_ahl",
        "label": "Assign to AHL",
        "enabled": ahl_ok,
        "blocked_reason": None if ahl_ok else "Not AHL-eligible under junior return rules",
    })
    opts.append({"id": "invite_camp", "label": "Invite to training camp", "enabled": True})
    opts.append({"id": "keep_unsigned", "label": "Keep unsigned", "enabled": True})
    return opts


def _year_arrays(years: int, nhl_m: float, minor_m: float, sb_total_m: float) -> Dict[str, List[float]]:
    y = max(1, int(years))
    per_sb = round(float(sb_total_m) / y, 4) if sb_total_m else 0.0
    return {
        "nhl_salary_by_year_m": [round(nhl_m, 4)] * y,
        "minor_salary_by_year_m": [round(minor_m, 4)] * y,
        "signing_bonus_by_year_m": [per_sb] * y,
    }


def build_offer_from_template(
    player: Any,
    *,
    season_year: int,
    template_id: str,
    assignment_plan: Optional[str] = None,
    development_promise: Optional[str] = None,
    term_years: Optional[int] = None,
) -> Dict[str, Any]:
    legal = legal_elc_terms(player, season_year)
    tid = str(template_id or "standard_elc").strip().lower()
    if tid not in TEMPLATE_IDS:
        tid = "standard_elc"

    years_pool = list(legal.get("legal_terms") or [3])
    if tid == "two_year_bridge":
        years = 2 if 2 in years_pool else (1 if 1 in years_pool else years_pool[0])
        schedule_a = round(ELC_MAX_SCHEDULE_A_M * 0.5, 4)
        schedule_b = 0.0
        sb_total = round(ELC_MAX_SIGNING_BONUS_TOTAL_M * (years / 3.0) * 0.5, 4)
        label = "Two-Year Bridge ELC"
        summary = f"{years} Years · Standard bonuses"
    elif tid == "maximum_bonus_elc":
        years = 3 if 3 in years_pool else years_pool[-1]
        schedule_a = ELC_MAX_SCHEDULE_A_M
        schedule_b = ELC_MAX_SCHEDULE_B_M
        sb_total = round(ELC_MAX_SIGNING_BONUS_TOTAL_M * (years / 3.0), 4)
        label = "Maximum Bonus ELC"
        summary = f"{years} Years · Schedule A · Schedule B · Max signing bonus"
    elif tid == "no_bonus_elc":
        years = 2 if 2 in years_pool else years_pool[0]
        schedule_a = 0.0
        schedule_b = 0.0
        sb_total = 0.0
        label = "No Bonus ELC"
        summary = f"{years} Years · No bonuses"
    else:
        years = int(legal.get("recommended_term") or years_pool[-1])
        schedule_a = round(ELC_MAX_SCHEDULE_A_M * 0.55, 4)
        schedule_b = 0.0
        sb_total = round(ELC_MAX_SIGNING_BONUS_TOTAL_M * (years / 3.0) * 0.7, 4)
        label = "Standard ELC"
        summary = f"{years} Years · Schedule A · Signing bonus"

    if term_years is not None and int(term_years) in years_pool:
        years = int(term_years)

    arrays = _year_arrays(years, ELC_AAV_M, ELC_MINOR_SALARY_M, sb_total)
    assign = assignment_plan or _default_assignment(legal)
    promise = development_promise
    if promise is None and tid == "maximum_bonus_elc":
        promise = "ahl_featured"
    elif promise is None and tid == "no_bonus_elc":
        promise = "depth_only"

    bonus_conditions = _default_bonus_conditions(schedule_a, schedule_b)
    offer = {
        "offer_id": str(uuid.uuid4()),
        "template_id": tid,
        "label": label,
        "summary": summary,
        "player_id": _pid(player),
        "term_years": years,
        "aav_m": ELC_AAV_M,
        "aav_display": format_money_display(ELC_AAV_M),
        "cap_hit_m": ELC_AAV_M,
        "minor_salary_m": ELC_MINOR_SALARY_M,
        "signing_bonus_total_m": round(sb_total, 4),
        "signing_bonus_display": format_money_display(sb_total),
        "schedule_a_bonus_m": schedule_a,
        "schedule_a_display": format_money_display(schedule_a) if schedule_a else "None",
        "schedule_b_bonus_m": schedule_b,
        "schedule_b_display": format_money_display(schedule_b) if schedule_b else "None",
        "performance_bonus_m": round(schedule_a + schedule_b, 4),
        "maximum_performance_bonus_m": round(schedule_a + schedule_b, 4),
        "contract_start_season": int(season_year),
        "expiry_season": int(season_year) + years,
        "slide_eligible": bool(legal.get("slide_eligible")),
        "slide_games_threshold": ELC_SLIDE_GAMES_THRESHOLD,
        "development_promise": promise,
        "assignment_plan": assign,
        "is_two_way": True,
        "is_entry_level": True,
        "bonus_conditions": bonus_conditions,
        "bonus_descriptions": legal.get("bonus_descriptions"),
        "schema_version": CONTRACT_SCHEMA_VERSION,
        **arrays,
        "yearly_preview": [
            {
                "year": i + 1,
                "season": f"{season_year + i}-{str(season_year + i + 1)[-2:]}",
                "nhl_salary_m": arrays["nhl_salary_by_year_m"][i],
                "minor_salary_m": arrays["minor_salary_by_year_m"][i],
                "signing_bonus_m": arrays["signing_bonus_by_year_m"][i],
                "max_bonus_m": round((schedule_a + schedule_b) / years, 4) if years else 0.0,
                "cap_hit_m": ELC_AAV_M,
            }
            for i in range(years)
        ],
    }
    return offer


def _default_assignment(legal: Dict[str, Any]) -> str:
    for opt in legal.get("assignment_options") or []:
        if opt.get("enabled") and opt.get("id") in ("return_junior", "keep_college", "keep_europe"):
            return str(opt["id"])
    return "invite_camp"


def _default_bonus_conditions(schedule_a: float, schedule_b: float) -> Dict[str, Any]:
    cond_a = []
    cond_b = []
    if schedule_a > 0:
        cond_a = [
            {"id": "games_played", "label": "Games played", "threshold": 40, "amount_m": round(schedule_a * 0.25, 4)},
            {"id": "points", "label": "Points", "threshold": 40, "amount_m": round(schedule_a * 0.25, 4)},
            {"id": "goals", "label": "Goals", "threshold": 15, "amount_m": round(schedule_a * 0.2, 4)},
            {"id": "all_rookie", "label": "All-Rookie recognition", "threshold": 1, "amount_m": round(schedule_a * 0.3, 4)},
        ]
    if schedule_b > 0:
        cond_b = [
            {"id": "calder", "label": "Calder Trophy", "threshold": 1, "amount_m": round(schedule_b * 0.5, 4)},
            {"id": "hart_finish", "label": "Hart Trophy finish", "threshold": 1, "amount_m": round(schedule_b * 0.5, 4)},
        ]
    return {"schedule_a": cond_a, "schedule_b": cond_b}


def list_offer_templates(player: Any, season_year: int) -> List[Dict[str, Any]]:
    legal = legal_elc_terms(player, season_year)
    if not legal.get("elc_eligible"):
        return []
    out = []
    for tid in TEMPLATE_IDS:
        if tid == "two_year_bridge" and 2 not in (legal.get("legal_terms") or []) and 1 not in (legal.get("legal_terms") or []):
            continue
        offer = build_offer_from_template(player, season_year=season_year, template_id=tid)
        out.append({
            "template_id": tid,
            "label": offer["label"],
            "summary": offer["summary"],
            "term_years": offer["term_years"],
            "aav_display": offer["aav_display"],
            "signing_bonus_display": offer["signing_bonus_display"],
            "schedule_a_display": offer["schedule_a_display"],
            "schedule_b_display": offer["schedule_b_display"],
            "slide_eligible": offer["slide_eligible"],
            "offer": offer,
        })
    return out


def validate_offer(player: Any, offer: Dict[str, Any], season_year: int) -> Dict[str, Any]:
    legal = legal_elc_terms(player, season_year)
    warnings: List[str] = []
    blocking: List[str] = []
    if not legal.get("elc_eligible"):
        blocking.append("Player is not ELC eligible")
    years = int(offer.get("term_years") or 0)
    if years not in (legal.get("legal_terms") or []):
        blocking.append(f"Term {years} years is not legal for this prospect")
    aav = float(offer.get("aav_m") or 0)
    if abs(aav - ELC_AAV_M) > 0.01:
        blocking.append(f"ELC AAV must be {format_money_display(ELC_AAV_M)}")
    sb = float(offer.get("signing_bonus_total_m") or 0)
    max_sb = float(legal.get("max_signing_bonus_total_m") or 0) * (years / 3.0)
    if sb > max_sb + 0.001:
        blocking.append("Signing bonus exceeds maximum for this term")
    sa = float(offer.get("schedule_a_bonus_m") or 0)
    sb_b = float(offer.get("schedule_b_bonus_m") or 0)
    if sa > float(legal.get("max_schedule_a_m") or 0) + 0.001:
        blocking.append("Schedule A exceeds maximum")
    if sb_b > float(legal.get("max_schedule_b_m") or 0) + 0.001:
        blocking.append("Schedule B exceeds maximum")
    assign = str(offer.get("assignment_plan") or "")
    assign_ids = {o["id"]: o for o in legal.get("assignment_options") or []}
    if assign and assign in assign_ids and not assign_ids[assign].get("enabled", True):
        blocking.append(assign_ids[assign].get("blocked_reason") or "Assignment not legal")
    if offer.get("slide_eligible") and not legal.get("slide_eligible"):
        warnings.append("Slide requested but prospect is not slide-eligible; slide will be cleared")
    return validation_result(
        allowed=not blocking,
        error_code="elc_offer_invalid" if blocking else "",
        title="ELC offer invalid" if blocking else "ELC offer valid",
        user_message="; ".join(blocking) if blocking else "Offer is legal",
        technical_message="; ".join(blocking),
        blocking_reasons=blocking,
        warnings=warnings,
        legal_terms=legal,
    )


def agent_wants(player: Any, legal: Dict[str, Any]) -> List[Dict[str, Any]]:
    wants = []
    age = _age(player)
    readiness = float(getattr(player, "nhl_readiness", None) or getattr(player, "overall", None) or 55)
    wants.append({"id": "signing_bonus", "label": "Signing bonus", "priority": "high" if age <= 20 else "med"})
    if readiness >= 68:
        wants.append({"id": "nhl_opportunity", "label": "NHL opportunity", "priority": "high"})
    else:
        wants.append({"id": "ahl_role", "label": "Featured AHL role", "priority": "high"})
    if legal.get("can_return_junior"):
        wants.append({"id": "junior_path", "label": "Junior development path", "priority": "med"})
    wants.append({"id": "performance_bonuses", "label": "Performance bonuses", "priority": "med"})
    return wants[:5]


def evaluate_offer_acceptance(
    player: Any,
    team: Any,
    offer: Dict[str, Any],
    *,
    season_year: int,
    prior_rejects: int = 0,
) -> Dict[str, Any]:
    legal = legal_elc_terms(player, season_year)
    age = _age(player)
    path = _path(player)
    readiness = float(getattr(player, "nhl_readiness", None) or getattr(player, "overall", None) or 55)
    relationship = float(getattr(player, "org_relationship", None) or 0.55)
    expiry = getattr(player, "rights_expiry_year", None)
    willingness = getattr(player, "willingness_to_sign", None)

    score = 0.52
    positives: List[str] = []
    concerns: List[str] = []

    sb = float(offer.get("signing_bonus_total_m") or 0)
    sa = float(offer.get("schedule_a_bonus_m") or 0)
    sbb = float(offer.get("schedule_b_bonus_m") or 0)
    if sb >= ELC_MAX_SIGNING_BONUS_TOTAL_M * 0.55:
        score += 0.08
        positives.append("strong signing bonus")
    elif sb <= 0.01:
        score -= 0.07
        concerns.append("no signing bonus")
    if sa > 0:
        score += 0.05
        positives.append("Schedule A bonuses")
    if sbb > 0:
        score += 0.06
        positives.append("Schedule B bonuses")
    if sa <= 0 and sbb <= 0:
        score -= 0.04
        concerns.append("no performance bonuses")

    promise = offer.get("development_promise")
    if promise in ("top_six_track", "ahl_featured", "protected_role"):
        score += 0.1
        positives.append("development promise")
    elif promise == "depth_only":
        score -= 0.06
        concerns.append("depth-only promise")

    assign = str(offer.get("assignment_plan") or "")
    if assign == "assign_ahl" and readiness < 62:
        score -= 0.08
        concerns.append("AHL assignment may be early")
    if assign == "return_junior" and readiness >= 70:
        score -= 0.05
        concerns.append("wants pro opportunity")
    if assign in ("return_junior", "keep_college", "keep_europe") and readiness < 65:
        score += 0.04
        positives.append("assignment matches development stage")

    if willingness is False:
        score -= 0.3
        concerns.append("low willingness to sign")
    elif willingness is True:
        score += 0.12
        positives.append("motivated to turn pro")

    if "NCAA" in path and age <= 21 and assign != "keep_college":
        score -= 0.15
        concerns.append("NCAA return remains attractive")
    if ("EUROPE" in path) and readiness < 70 and assign != "keep_europe":
        score -= 0.12
        concerns.append("European contract still in play")

    if expiry is not None and int(expiry) <= int(season_year) + 1:
        score += 0.2
        positives.append("rights deadline pressure")

    if readiness >= 72:
        score += 0.14
        positives.append("NHL readiness supports turning pro")
    elif readiness < 58:
        score -= 0.1
        concerns.append("not yet ready for pro hockey")

    score += (relationship - 0.5) * 0.28
    score -= min(0.2, prior_rejects * 0.07)
    if prior_rejects:
        concerns.append("prior offers rejected")
    score += (_rng((_pid(player), season_year, offer.get("template_id"), "accept")) - 0.5) * 0.1

    # Slots
    slot_payload: Dict[str, Any] = {}
    try:
        from services.contract_economy import validate_contract_slots

        slots = validate_contract_slots(team, getattr(team, "_league_ref", None), additional=1)
        slot_payload = slots
        if not slots.get("ok"):
            return {
                "ok": False,
                "accepted": False,
                "acceptance_probability": 0.0,
                "decision": "blocked",
                "main_positive": None,
                "main_concern": slots.get("reason") or "contract_slots_full",
                "agent_wants": agent_wants(player, legal),
                "counter_offer": None,
                "reasons": [slots.get("reason") or "contract slots full"],
                "positives": positives,
                "concerns": concerns,
                "contract_slots": slots,
                "score": round(score, 3),
            }
    except Exception:
        pass

    score = max(0.0, min(1.0, score))
    if "NCAA" in path and age <= 20 and readiness < 68 and (expiry is None or int(expiry) > season_year + 1):
        score = min(score, 0.42)
        concerns.append("chose to remain in NCAA")

    decision = "accepted" if score >= 0.48 else "rejected"
    counter = None
    if decision == "rejected" and score >= 0.38:
        decision = "countered"
        counter = {
            "request": "more_bonuses" if sa + sbb < ELC_MAX_SCHEDULE_A_M else "development_promise",
            "preferred_template": "maximum_bonus_elc",
            "preferred_assignment": "return_junior" if legal.get("can_return_junior") else assign,
            "message": "Agent wants stronger bonuses or a clearer development path",
        }
    elif decision == "rejected" and score >= 0.44:
        decision = "considering"

    return {
        "ok": True,
        "accepted": decision == "accepted",
        "acceptance_probability": round(score, 3),
        "acceptance_pct": int(round(score * 100)),
        "decision": decision,
        "main_positive": positives[0] if positives else None,
        "main_concern": concerns[0] if concerns else None,
        "agent_wants": agent_wants(player, legal),
        "counter_offer": counter,
        "reasons": (positives + concerns)[:6],
        "positives": positives[:4],
        "concerns": concerns[:4],
        "contract_slots": slot_payload,
        "score": round(score, 3),
        "outlook_label": (
            "Strong offer"
            if score >= 0.72
            else "Competitive offer"
            if score >= 0.55
            else "Below expectations"
            if score >= 0.42
            else "Likely to decline"
        ),
    }


def offer_to_contract_dict(offer: Dict[str, Any], season_year: int) -> Dict[str, Any]:
    years = int(offer.get("term_years") or 3)
    sa = float(offer.get("schedule_a_bonus_m") or 0)
    sbb = float(offer.get("schedule_b_bonus_m") or 0)
    sb = float(offer.get("signing_bonus_total_m") or 0)
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "type": "ELC",
        "contract_type": "ELC",
        "years": years,
        "years_remaining": years,
        "aav_m": ELC_AAV_M,
        "cap_hit_m": ELC_AAV_M,
        "base_salary_m": ELC_AAV_M,
        "salary_m": ELC_AAV_M,
        "signing_bonus_m": round(sb, 4),
        "performance_bonus_m": round(sa + sbb, 4),
        "schedule_a_bonus_m": round(sa, 4),
        "schedule_b_bonus_m": round(sbb, 4),
        "maximum_performance_bonus_m": round(sa + sbb, 4),
        "nhl_salary_by_year_m": list(offer.get("nhl_salary_by_year_m") or [ELC_AAV_M] * years),
        "minor_salary_by_year_m": list(offer.get("minor_salary_by_year_m") or [ELC_MINOR_SALARY_M] * years),
        "signing_bonus_by_year_m": list(offer.get("signing_bonus_by_year_m") or [0.0] * years),
        "bonus_conditions": offer.get("bonus_conditions") or {},
        "earned_bonuses_m": 0.0,
        "rights_status": "RFA",
        "rights": "RFA",
        "expiry_status": "RFA",
        "expiry_year": int(season_year) + years,
        "effective_season": int(season_year),
        "two_way": True,
        "is_entry_level": True,
        "can_slide": bool(offer.get("slide_eligible")),
        "slide_eligible": bool(offer.get("slide_eligible")),
        "slide_games_threshold": int(offer.get("slide_games_threshold") or ELC_SLIDE_GAMES_THRESHOLD),
        "slide_years_used": 0,
        "slide_triggered": False,
        "development_promise": offer.get("development_promise"),
        "assignment_plan": offer.get("assignment_plan"),
        "offer_template_id": offer.get("template_id"),
        "negotiation_id": offer.get("offer_id"),
        "source": "elc_offer",
    }


def _history(session: Any) -> List[Dict[str, Any]]:
    hist = getattr(session, "elc_negotiation_history", None)
    if not isinstance(hist, list):
        session.elc_negotiation_history = []
        hist = session.elc_negotiation_history
    return hist


def prior_reject_count(session: Any, player_id: str) -> int:
    n = 0
    for row in _history(session):
        if str(row.get("player_id")) == str(player_id) and row.get("result") in ("rejected", "countered"):
            n += 1
    return n


def record_negotiation(session: Any, row: Dict[str, Any]) -> None:
    hist = _history(session)
    hist.append(row)
    # Cap history length
    if len(hist) > 400:
        del hist[:-400]
    session.elc_negotiation_history = hist


def preview_elc_offer(
    session: Any,
    player: Any,
    team: Any,
    *,
    season_year: int,
    template_id: str = "standard_elc",
    offer: Optional[Dict[str, Any]] = None,
    assignment_plan: Optional[str] = None,
    development_promise: Optional[str] = None,
    term_years: Optional[int] = None,
) -> Dict[str, Any]:
    from services.contract_economy import validate_contract_slots, get_team_cap_snapshot_full

    built = offer or build_offer_from_template(
        player,
        season_year=season_year,
        template_id=template_id,
        assignment_plan=assignment_plan,
        development_promise=development_promise,
        term_years=term_years,
    )
    valid = validate_offer(player, built, season_year)
    league = getattr(getattr(session, "sim", None), "league", None)
    slots = validate_contract_slots(team, league, additional=1)
    used = int(slots.get("contract_slots_used") or 0)
    limit = int(slots.get("contract_slots_limit") or 50)
    snap = {}
    try:
        snap = get_team_cap_snapshot_full(team, league, getattr(session, "sim", None), season_year=season_year)
    except Exception:
        snap = {}
    acceptance = evaluate_offer_acceptance(
        player,
        team,
        built,
        season_year=season_year,
        prior_rejects=prior_reject_count(session, _pid(player)),
    )
    signing_result = {
        "contract_starts": built.get("contract_start_season"),
        "cap_impact_m": ELC_AAV_M if built.get("assignment_plan") not in ("keep_unsigned",) else 0.0,
        "cap_impact_display": f"+{format_money_display(ELC_AAV_M)}",
        "roster": "Reserve / developmental",
        "assignment": built.get("assignment_plan"),
        "rights": "Protected via ELC",
        "contract_slots_after": f"{used + 1}/{limit}" if slots.get("ok") else f"{used}/{limit}",
        "slide_eligible": built.get("slide_eligible"),
    }
    return {
        "ok": True,
        "validation": valid,
        "offer": built,
        "legal_terms": valid.get("legal_terms") or legal_elc_terms(player, season_year),
        "acceptance": acceptance,
        "cap_preview": {
            "projected_cap_hit_m": ELC_AAV_M,
            "usable_cap_space_m": snap.get("usable_cap_space_m"),
            "bonus_reserve_m": snap.get("bonus_reserve_m"),
            "bonus_exposure_m": float(built.get("performance_bonus_m") or 0),
        },
        "slot_preview": {
            "contract_slots_used": used,
            "contract_slots_limit": limit,
            "after_signing": used + 1 if slots.get("ok") else used,
            "ok": bool(slots.get("ok")),
            "reason": slots.get("reason"),
        },
        "signing_result": signing_result,
        "templates": list_offer_templates(player, season_year),
    }


def submit_elc_offer(
    session: Any,
    player: Any,
    team: Any,
    *,
    season_year: int,
    template_id: str = "standard_elc",
    offer: Optional[Dict[str, Any]] = None,
    assignment_plan: Optional[str] = None,
    development_promise: Optional[str] = None,
    term_years: Optional[int] = None,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    # Idempotency
    if idempotency_key:
        for row in _history(session):
            if row.get("idempotency_key") == idempotency_key and row.get("result") == "accepted":
                return {"ok": True, "signed": True, "idempotent": True, "contract": row.get("contract")}

    preview = preview_elc_offer(
        session,
        player,
        team,
        season_year=season_year,
        template_id=template_id,
        offer=offer,
        assignment_plan=assignment_plan,
        development_promise=development_promise,
        term_years=term_years,
    )
    built = preview["offer"]
    valid = preview["validation"]
    if not valid.get("allowed"):
        return {
            "ok": False,
            "signed": False,
            "reason": valid.get("user_message") or "invalid_offer",
            "validation": valid,
            "preview": preview,
        }

    acceptance = preview["acceptance"]
    hist_row = {
        "offer_id": built.get("offer_id"),
        "idempotency_key": idempotency_key,
        "player_id": _pid(player),
        "team_id": str(getattr(team, "team_id", None) or getattr(team, "id", "") or ""),
        "submitted_at": f"{season_year}-offseason",
        "terms": {
            "template_id": built.get("template_id"),
            "term_years": built.get("term_years"),
            "aav_m": built.get("aav_m"),
            "signing_bonus_total_m": built.get("signing_bonus_total_m"),
            "schedule_a_bonus_m": built.get("schedule_a_bonus_m"),
            "schedule_b_bonus_m": built.get("schedule_b_bonus_m"),
            "assignment_plan": built.get("assignment_plan"),
            "development_promise": built.get("development_promise"),
        },
        "acceptance_probability": acceptance.get("acceptance_probability"),
        "result": acceptance.get("decision"),
        "reason": acceptance.get("main_concern"),
    }

    if not acceptance.get("accepted"):
        record_negotiation(session, hist_row)
        return {
            "ok": False,
            "signed": False,
            "reason": "prospect_declined",
            "decision": acceptance,
            "counter_offer": acceptance.get("counter_offer"),
            "preview": preview,
            "negotiation": hist_row,
        }

    from services.contract_economy import assign_elc_from_offer, apply_post_elc_assignment

    # Snapshot for rollback
    prev_contract = getattr(player, "contract", None)
    prev_signed = getattr(player, "signed_status", None)
    prev_elc = getattr(player, "entry_level_contract_eligible", None)
    prev_rights = getattr(player, "rights_status", None)
    prev_status = getattr(player, "organizational_status", None)
    prev_prospect = getattr(player, "prospect_status", None)

    result = assign_elc_from_offer(player, team, getattr(getattr(session, "sim", None), "league", None), season_year, built)
    if not result.get("ok"):
        # restore
        try:
            player.contract = prev_contract
            player.signed_status = prev_signed
            player.entry_level_contract_eligible = prev_elc
            player.rights_status = prev_rights
            player.organizational_status = prev_status
            player.prospect_status = prev_prospect
        except Exception:
            pass
        hist_row["result"] = "failed"
        hist_row["reason"] = result.get("reason")
        record_negotiation(session, hist_row)
        return {**result, "signed": False, "preview": preview, "negotiation": hist_row}

    # Persist promise + slide attrs
    try:
        setattr(player, "development_promise", built.get("development_promise"))
        setattr(player, "development_promise_season", season_year)
        setattr(player, "elc_slide_eligible", bool(built.get("slide_eligible")))
        setattr(player, "elc_slide_years_remaining", 1 if built.get("slide_eligible") else 0)
        setattr(player, "slide_games_threshold", built.get("slide_games_threshold") or ELC_SLIDE_GAMES_THRESHOLD)
        setattr(player, "organizational_status", "signed_unassigned")
        setattr(player, "prospect_status", "signed_prospect")
        setattr(player, "rights_status", "signed")
        setattr(player, "signed_status", "signed")
    except Exception:
        pass

    assign_res = apply_post_elc_assignment(
        session, player, team, built.get("assignment_plan"), season_year=season_year
    )
    hist_row["result"] = "accepted"
    hist_row["contract"] = result.get("contract")
    hist_row["assignment"] = assign_res
    record_negotiation(session, hist_row)

    # The contract itself is signed even if the *requested* developmental placement
    # (e.g. assign_ahl) failed — surface that distinctly so callers/UI don't report a
    # clean "signed to AHL" when the player actually landed elsewhere.
    return {
        "ok": True,
        "signed": True,
        "contract": result.get("contract"),
        "decision": acceptance,
        "assignment": assign_res,
        "assignment_ok": bool(assign_res.get("ok", True)),
        "preview": preview,
        "negotiation": hist_row,
    }


def process_elc_slides(session: Any, season_year: int) -> Dict[str, Any]:
    """Year-end: if signed ELC slide-eligible and GP under threshold, extend term (do not burn year)."""
    league = getattr(getattr(session, "sim", None), "league", None)
    slid: List[Dict[str, Any]] = []
    burned: List[Dict[str, Any]] = []
    if league is None:
        return {"slid": [], "burned": [], "count": 0}

    for team in list(getattr(league, "teams", None) or []):
        for p in list(getattr(team, "prospect_pool", None) or []) + list(getattr(team, "roster", None) or []):
            c = getattr(p, "contract", None)
            if not isinstance(c, dict):
                continue
            if str(c.get("contract_type") or c.get("type") or "").upper() != "ELC":
                continue
            if not c.get("slide_eligible") and not getattr(p, "elc_slide_eligible", False):
                continue
            used = int(c.get("slide_years_used") or 0)
            if used >= 1:
                continue
            gp = int(getattr(p, "nhl_games_played_this_season", 0) or c.get("games_played_this_season") or 0)
            threshold = int(c.get("slide_games_threshold") or ELC_SLIDE_GAMES_THRESHOLD)
            if gp < threshold:
                # Slide: do not decrement — bump expiry, mark slid
                c["slide_triggered"] = True
                c["slide_years_used"] = used + 1
                c["effective_season"] = int(season_year) + 1
                c["expiry_year"] = int(c.get("expiry_year") or season_year) + 1
                c["years_remaining"] = int(c.get("years_remaining") or c.get("years") or 3)
                try:
                    setattr(p, "elc_slide_years_remaining", 0)
                    setattr(p, "contract_burned", False)
                    setattr(p, "nhl_games_played_this_season", 0)
                except Exception:
                    pass
                slid.append({"player_id": _pid(p), "gp": gp, "expiry_year": c.get("expiry_year")})
            else:
                c["slide_triggered"] = False
                c["slide_eligible"] = False
                try:
                    setattr(p, "elc_slide_eligible", False)
                    setattr(p, "contract_burned", True)
                except Exception:
                    pass
                burned.append({"player_id": _pid(p), "gp": gp})
    return {"slid": slid, "burned": burned, "count": len(slid) + len(burned)}
