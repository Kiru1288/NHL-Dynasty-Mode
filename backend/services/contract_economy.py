"""
Franchise contract & cap economy — single source of truth for cap snapshots,
contract generation, signing, RFA rights, buyouts, waivers, CPU FA, and valuation.

# Source of truth: contract_economy (franchise API + UI).
# SimEngine/app/sim_engine/entities/contract.py (PCDS negotiate_contract) is retained
# for the standalone sim runner only — not wired into franchise negotiation.
# DESIGN: user/CPU parity on the FA wire = skill matters, not info asymmetry. Intended.
"""

from __future__ import annotations

import logging
import math
import random
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.economy.cap_engine import (
    buried_cap_hit_millions,
    calculate_team_cap_snapshot,
    can_sign_player,
    can_trade_cap_fit,
    normalize_money_to_millions,
    player_cap_hit_millions,
    validate_team_cap_compliance,
)

CONTRACT_SCHEMA_VERSION = 2
OFFER_SHEET_ELIGIBLE_AAV_CEILING_M = 3.613  # NHL Group-2 offer-sheet threshold (millions)
OFFER_SHEET_MATCH_WINDOW_DAYS = 7
DEFAULT_MNTC_TEAM_COUNT = 10


def compute_prorated_cap_hit_m(
    aav_m: float,
    years: int,
    signing_bonus_m: float = 0.0,
) -> float:
    """Cap hit = (AAV × years + signing_bonus) / years — signing bonus prorates into cap."""
    yrs = max(1, int(years or 1))
    total = float(aav_m or 0.0) * yrs + float(signing_bonus_m or 0.0)
    return round(total / yrs, 3)


def _offer_ntc_mode(offer: Dict[str, Any]) -> str:
    """Return FULL | MODIFIED | NONE from offer payload."""
    raw = str(offer.get("ntc_mode") or "").upper()
    if raw in ("FULL", "MODIFIED", "MNTC", "M-NTC"):
        return "MODIFIED" if raw in ("MODIFIED", "MNTC", "M-NTC") else "FULL"
    if bool(offer.get("m_ntc") or offer.get("mntc") or offer.get("modified_ntc")):
        return "MODIFIED"
    if bool(offer.get("ntc") or offer.get("no_trade_clause")):
        return "FULL"
    return "NONE"


def _contract_ntc_fields_from_offer(offer: Dict[str, Any]) -> Dict[str, Any]:
    mode = _offer_ntc_mode(offer)
    nmc = bool(offer.get("nmc") or offer.get("no_move_clause"))
    teams = list(offer.get("ntc_teams") or offer.get("approved_trade_teams") or [])
    return {
        "ntc_mode": "NONE" if nmc else mode,
        "ntc": mode in ("FULL", "MODIFIED") and not nmc,
        "nmc": nmc,
        "no_trade_clause": mode in ("FULL", "MODIFIED") and not nmc,
        "no_move_clause": nmc,
        "modified_no_trade_teams": int(offer.get("modified_no_trade_teams") or DEFAULT_MNTC_TEAM_COUNT)
        if mode == "MODIFIED"
        else 0,
        "ntc_teams": teams,
        "approved_trade_teams": teams,
        "clause_type": "NMC" if nmc else ("M-NTC" if mode == "MODIFIED" else "NTC" if mode == "FULL" else "None"),
    }


def rfa_offer_sheet_eligible(player: Any, entry: Optional[Dict[str, Any]] = None) -> bool:
    """Group-2 style gate: age ≥ 20 and prior salary below offer-sheet compensation floor."""
    age = _player_age(player)
    if age < 20:
        return False
    prev = 0.0
    if isinstance(entry, dict):
        prev = float(entry.get("previous_aav_m") or entry.get("qualifying_offer_aav_m") or 0.0)
    if prev <= 0:
        prev = float(player_cap_hit_millions(player) or 0.0)
    if prev <= 0:
        prev = float(compute_market_value(player, None) or LEAGUE_MINIMUM_AAV_M)
    return prev < OFFER_SHEET_ELIGIBLE_AAV_CEILING_M


def _emit_contract_storyline(session: Any, text: str, *, kind: str = "contracts", severity: str = "medium") -> None:
    if session is None or not text:
        return
    try:
        events = getattr(session, "storyline_events", None)
        if not isinstance(events, list):
            session.storyline_events = []
            events = session.storyline_events
        events.append({
            "id": f"contract-{abs(hash(text)) & 0xFFFFFFFF:08x}",
            "kind": kind,
            "severity": severity,
            "text": text,
            "headline": text[:120],
        })
        session.storyline_events = events[-200:]
    except Exception:
        pass
_NON_NHL_SPC_TYPES = frozenset({
    "AHL", "ECHL", "AHL_ECHL", "PTO", "ATO", "TRYOUT",
    "AHL_ONLY", "ECHL_ONLY", "AHLONLY", "ECHLONLY", "MINORS", "MINOR",
})
_NHL_SPC_ALIASES = {
    "SPC": "STANDARD",
    "NHL": "STANDARD",
    "NHL_SPC": "STANDARD",
    "ONE_WAY": "STANDARD",
    "TWO_WAY": "STANDARD",
    "TWOWAY": "STANDARD",
    "ONEWAY": "STANDARD",
    "ENTRY_LEVEL": "ELC",
    "ENTRYLEVEL": "ELC",
}


def _normalize_contract_type_token(raw: Any) -> str:
    s = str(raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    if s in _NHL_SPC_ALIASES:
        return _NHL_SPC_ALIASES[s]
    if s in ("AHL_ONLY", "AHLONLY"):
        return "AHL"
    if s in ("ECHL_ONLY", "ECHLONLY"):
        return "ECHL"
    return s or "STANDARD"


LEAGUE_MINIMUM_AAV_M = 0.775
ELC_AAV_M = 0.95
CONTRACT_SLOTS_LIMIT = 50
CAP_SAFE_CORE_TOP_N = 6
CAP_SAFE_STAR_OVR = 88.0

BAD_CONTRACT_TYPES = (
    "overpaid_depth",
    "aging_star_term",
    "goalie_gamble",
    "panic_ufa",
    "clause_veteran",
    "injury_risk",
    "declining_expensive",
    "middle_six_top_six_pay",
)

BAD_TAG_MAP = {
    "overpaid_depth": "Bad Deal",
    "aging_star_term": "Heavy Term",
    "goalie_gamble": "Bad Deal",
    "panic_ufa": "Bad Deal",
    "clause_veteran": "Clause Locked",
    "injury_risk": "Buyout Risk",
    "declining_expensive": "Cap Casualty",
    "middle_six_top_six_pay": "Bad Deal",
}


# ---------------------------------------------------------------------------
# Player / team helpers (self-contained to avoid circular imports)
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj: Any, key: str, value: Any) -> None:
    if obj is None:
        return
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _player_age(player: Any) -> int:
    ident = _get(player, "identity", None)
    try:
        return int(_get(ident, "age", 0) or _get(player, "age", 0) or 27)
    except (TypeError, ValueError):
        return 27


def _player_ovr(player: Any) -> float:
    fn = _get(player, "ovr", None)
    try:
        v = float(fn() if callable(fn) else fn or 0)
    except Exception:
        return 0.0
    return v * 99.0 if v <= 1.5 else v


def _player_potential(player: Any) -> float:
    ratings = _get(player, "ratings", None) or {}
    if isinstance(ratings, dict):
        try:
            return float(ratings.get("dev_potential", 0) or 0)
        except Exception:
            pass
    try:
        return float(_get(player, "potential", 0) or 0)
    except Exception:
        return 0.0


def _player_pos(player: Any) -> str:
    try:
        from services.roster_compliance import position_code

        return position_code(player) or "C"
    except Exception:
        ident = _get(player, "identity", None)
        pos = _get(ident, "position", None) if ident else None
        if pos is not None and hasattr(pos, "value"):
            pos = pos.value
        return str(pos or _get(player, "position", "C") or "C").upper()


def _player_name(player: Any) -> str:
    ident = _get(player, "identity", None)
    return str(_get(ident, "name", "") or _get(player, "name", "") or "Unknown")


def _player_id(player: Any) -> str:
    return str(_get(player, "id", "") or _get(player, "player_id", "") or "")


def _active_roster(team: Any) -> List[Any]:
    """Active NHL roster — excludes retired, buried/minors, IR, and LTIR."""
    try:
        from services.roster_compliance import iter_active_nhl_roster

        return iter_active_nhl_roster(team)
    except Exception:
        out = []
        for p in _get(team, "roster", None) or []:
            if _get(p, "retired", False):
                continue
            if _get(p, "is_buried", False) or _get(p, "buried", False) or _get(p, "in_minors", False):
                continue
            out.append(p)
        return out


def _all_rostered(team: Any) -> List[Any]:
    return [p for p in (_get(team, "roster", None) or []) if not _get(p, "retired", False)]


ELC_AAV_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Money normalization — single hydration path for legacy saves
# ---------------------------------------------------------------------------

MONEY_FIELDS = (
    "aav_m", "cap_hit_m", "base_salary_m", "salary_m",
    "signing_bonus_m", "performance_bonus_m", "buyout_penalty_m",
    "schedule_a_bonus_m", "schedule_b_bonus_m", "maximum_performance_bonus_m",
    "earned_bonuses_m", "minor_salary_m",
)
def normalize_money_m(value: Any) -> float:
    return normalize_money_to_millions(value)


def normalize_contract_dict(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        src = dict(raw)
    else:
        src = {k: getattr(raw, k, None) for k in (
            "contract_type", "type", "years", "years_remaining", "aav_m", "cap_hit_m",
            "base_salary_m", "salary_m", "signing_bonus_m", "performance_bonus_m",
            "buyout_penalty_m", "rights_status", "rights", "expiry_status", "expiry_year",
            "no_trade_clause", "no_move_clause", "no_movement_clause", "ntc", "nmc", "two_way", "source",
            "ntc_mode", "ntc_teams", "modified_no_trade_teams", "approved_trade_teams",
            "bad_contract_type", "bad_contract_score", "aav", "cap_hit", "salary_aav",
            "term", "term_remaining", "remaining_years",
        ) if hasattr(raw, k) or k in ("aav", "cap_hit")}

    out: Dict[str, Any] = {}
    for legacy, modern in (("aav", "aav_m"), ("cap_hit", "cap_hit_m"), ("salary", "salary_m")):
        if modern not in src or not src.get(modern):
            if src.get(legacy):
                src[modern] = src[legacy]

    for field in MONEY_FIELDS:
        if field in src and src[field] is not None:
            out[field] = round(max(0.0, normalize_money_m(src[field])), 3)

    aav = out.get("aav_m") or out.get("cap_hit_m") or 0.0
    bonus = float(out.get("signing_bonus_m") or src.get("signing_bonus_m") or 0.0)
    yrs_for_cap = max(1, int(out.get("years") or out.get("years_remaining") or src.get("years") or 1))
    if aav > 0:
        out.setdefault("aav_m", aav)
        prorated = compute_prorated_cap_hit_m(aav, yrs_for_cap, bonus)
        out.setdefault("cap_hit_m", prorated if bonus > 0 else aav)
        out.setdefault("base_salary_m", aav)
        out.setdefault("salary_m", aav)
    else:
        out.setdefault("aav_m", 0.0)
        out.setdefault("cap_hit_m", 0.0)
        out.setdefault("base_salary_m", 0.0)
        out.setdefault("salary_m", 0.0)
    out.setdefault("signing_bonus_m", 0.0)
    out.setdefault("performance_bonus_m", 0.0)
    out.setdefault("buyout_penalty_m", 0.0)
    out.setdefault("schedule_a_bonus_m", float(out.get("schedule_a_bonus_m") or 0.0))
    out.setdefault("schedule_b_bonus_m", float(out.get("schedule_b_bonus_m") or 0.0))
    out.setdefault(
        "maximum_performance_bonus_m",
        float(
            out.get("maximum_performance_bonus_m")
            or (out["schedule_a_bonus_m"] + out["schedule_b_bonus_m"])
            or out.get("performance_bonus_m")
            or 0.0
        ),
    )
    out.setdefault("earned_bonuses_m", float(out.get("earned_bonuses_m") or 0.0))
    out.setdefault("minor_salary_m", float(out.get("minor_salary_m") or 0.0))

    ctype_raw = str(src.get("contract_type") or src.get("type") or "STANDARD")
    ctype = _normalize_contract_type_token(ctype_raw)
    out["type"] = ctype
    out["contract_type"] = ctype
    # Explicit SPC flag — slot eligibility must not depend only on salary fields.
    if "is_nhl_spc" in src and src.get("is_nhl_spc") is not None:
        out["is_nhl_spc"] = bool(src.get("is_nhl_spc"))
    else:
        out["is_nhl_spc"] = ctype not in _NON_NHL_SPC_TYPES and ctype != ""
    out["nhl_spc"] = out["is_nhl_spc"]
    out["standard_player_contract"] = bool(out["is_nhl_spc"])

    yrs_rem = src.get("years_remaining")
    if yrs_rem is None:
        yrs_rem = src.get("term_remaining", src.get("remaining_years", src.get("term", src.get("years"))))
    if yrs_rem is None:
        out["years_remaining"] = 0
        out["years"] = max(0, int(src.get("years") or 0))
    else:
        try:
            out["years_remaining"] = max(0, int(yrs_rem))
            out["years"] = max(0, int(src.get("years") or out["years_remaining"] or 0))
        except (TypeError, ValueError):
            out["years_remaining"] = 0
            out["years"] = 0

    out["rights"] = str(src.get("rights_status") or src.get("rights") or "UFA").upper()
    out["rights_status"] = out["rights"]
    out["expiry_status"] = str(src.get("expiry_status") or out["rights"]).upper()
    out["expiry_year"] = int(src.get("expiry_year") or 0)
    out["ntc"] = bool(src.get("no_trade_clause") or src.get("ntc"))
    out["nmc"] = bool(src.get("no_move_clause") or src.get("no_movement_clause") or src.get("nmc"))
    ntc_mode = str(src.get("ntc_mode") or "").upper()
    if not ntc_mode:
        if out["nmc"]:
            ntc_mode = "NONE"
        elif int(src.get("modified_no_trade_teams") or 0) > 0 or src.get("clause_type") in ("M-NTC", "MNTC"):
            ntc_mode = "MODIFIED"
        elif out["ntc"]:
            ntc_mode = "FULL"
        else:
            ntc_mode = "NONE"
    out["ntc_mode"] = ntc_mode
    out["modified_no_trade_teams"] = int(src.get("modified_no_trade_teams") or 0)
    if ntc_mode == "MODIFIED" and out["modified_no_trade_teams"] <= 0:
        out["modified_no_trade_teams"] = DEFAULT_MNTC_TEAM_COUNT
    teams = list(src.get("ntc_teams") or src.get("approved_trade_teams") or [])
    out["ntc_teams"] = teams
    out["approved_trade_teams"] = teams
    out["ntc"] = ntc_mode in ("FULL", "MODIFIED") and not out["nmc"]
    out["no_trade_clause"] = out["ntc"]
    out["no_move_clause"] = out["nmc"]
    out["two_way"] = bool(src.get("two_way", False))
    out["is_one_way"] = bool(src.get("is_one_way", not out["two_way"]))
    out["is_two_way"] = out["two_way"]
    out["is_entry_level"] = bool(src.get("is_entry_level", ctype == "ELC"))
    out["source"] = str(src.get("source") or "generated")
    out["schema_version"] = int(src.get("schema_version") or CONTRACT_SCHEMA_VERSION)
    # Preserve structured year arrays / bonus schedules when present
    for key in (
        "nhl_salary_by_year_m",
        "minor_salary_by_year_m",
        "signing_bonus_by_year_m",
        "performance_bonus_by_year_m",
        "bonus_conditions",
        "development_promise",
        "assignment_plan",
        "offer_template_id",
        "negotiation_id",
        "clause_type",
        "effective_season",
        "can_slide",
        "slide_eligible",
        "slide_years_used",
        "slide_triggered",
        "slide_games_threshold",
    ):
        if key in src and src[key] is not None:
            out[key] = src[key]
    if src.get("bad_contract_type"):
        out["bad_contract_type"] = str(src["bad_contract_type"])
    _apply_contract_type_truth(out)
    return out


def _apply_contract_type_truth(c: Dict[str, Any]) -> None:
    """ELC label requires ELC_AAV_M cap hit; fake labels are relabeled."""
    cap = float(c.get("cap_hit_m") or c.get("aav_m") or 0.0)
    ctype = str(c.get("type") or c.get("contract_type") or "STANDARD").upper()
    true_elc_cap = cap > 0 and abs(cap - ELC_AAV_M) <= ELC_AAV_TOLERANCE
    if ctype == "ELC":
        if true_elc_cap:
            c["cap_hit_m"] = ELC_AAV_M
            c["aav_m"] = ELC_AAV_M
            c["base_salary_m"] = ELC_AAV_M
            c["salary_m"] = ELC_AAV_M
            c["type"] = "ELC"
            c["contract_type"] = "ELC"
        else:
            rights = str(c.get("rights_status") or c.get("rights") or "RFA").upper()
            relabel = "RFA_BRIDGE" if "RFA" in rights else "STANDARD"
            c["type"] = relabel
            c["contract_type"] = relabel


def normalize_contract_payload(player_or_contract: Any) -> Dict[str, Any]:
    """Canonical read path for dict contracts and _GeneratedContract objects."""
    if player_or_contract is None:
        return {}
    if _get(player_or_contract, "contract", None) is not None:
        raw = _get(player_or_contract, "contract", None)
    elif isinstance(player_or_contract, dict) and (
        "cap_hit_m" in player_or_contract or "contract_type" in player_or_contract or "type" in player_or_contract
    ):
        raw = player_or_contract
    elif hasattr(player_or_contract, "contract_type") or hasattr(player_or_contract, "cap_hit_m"):
        raw = player_or_contract
    else:
        raw = player_or_contract
    return normalize_contract_dict(raw) if raw is not None else {}


# ---------------------------------------------------------------------------
# Canonical contract accessors (screens / sim must use these)
# ---------------------------------------------------------------------------

def get_contract_cap_hit(contract: Any, season: Optional[int] = None) -> float:
    c = normalize_contract_payload(contract)
    ctype = str(c.get("contract_type") or c.get("type") or "").upper()
    if ctype in ("AHL", "ECHL", "AHL_ECHL", "PTO", "ATO", "TRYOUT"):
        return float(c.get("cap_hit_m") or 0.0)
    aav = float(c.get("aav_m") or 0.0)
    bonus = float(c.get("signing_bonus_m") or 0.0)
    yrs = max(1, int(c.get("years_remaining") or c.get("years") or 1))
    if bonus > 0 and aav > 0:
        return compute_prorated_cap_hit_m(aav, yrs, bonus)
    if "cap_hit_m" in c and c.get("cap_hit_m") is not None:
        return float(c.get("cap_hit_m") or 0.0)
    return aav


def get_contract_nhl_salary(contract: Any, season: Optional[int] = None) -> float:
    c = normalize_contract_payload(contract)
    arr = c.get("nhl_salary_by_year_m")
    if isinstance(arr, list) and arr:
        idx = 0
        if season is not None and c.get("effective_season"):
            idx = max(0, int(season) - int(c["effective_season"]))
        return float(arr[min(idx, len(arr) - 1)])
    return float(c.get("salary_m") or c.get("base_salary_m") or c.get("aav_m") or 0.0)


def get_contract_minor_salary(contract: Any, season: Optional[int] = None) -> float:
    c = normalize_contract_payload(contract)
    arr = c.get("minor_salary_by_year_m")
    if isinstance(arr, list) and arr:
        idx = 0
        if season is not None and c.get("effective_season"):
            idx = max(0, int(season) - int(c["effective_season"]))
        return float(arr[min(idx, len(arr) - 1)])
    return float(c.get("minor_salary_m") or 0.0)


def get_contract_bonus_total(contract: Any, season: Optional[int] = None) -> float:
    c = normalize_contract_payload(contract)
    return float(
        c.get("maximum_performance_bonus_m")
        or (float(c.get("schedule_a_bonus_m") or 0) + float(c.get("schedule_b_bonus_m") or 0))
        or c.get("performance_bonus_m")
        or 0.0
    )


def get_contract_years_remaining(contract: Any, season: Optional[int] = None) -> int:
    c = normalize_contract_payload(contract)
    return int(c.get("years_remaining") or 0)


def get_contract_clause(contract: Any, season: Optional[int] = None) -> Dict[str, Any]:
    c = normalize_contract_payload(contract)
    return {
        "clause_type": c.get("clause_type") or ("NMC" if c.get("nmc") else "NTC" if c.get("ntc") else "None"),
        "ntc": bool(c.get("ntc")),
        "nmc": bool(c.get("nmc")),
        "season": season,
    }


def is_contract_active(contract: Any, season: Optional[int] = None) -> bool:
    c = normalize_contract_payload(contract)
    if not c:
        return False
    if int(c.get("years_remaining") or 0) <= 0 and float(c.get("aav_m") or 0) <= 0:
        return False
    if season is not None and c.get("expiry_year"):
        try:
            if int(season) >= int(c["expiry_year"]):
                return False
        except (TypeError, ValueError):
            pass
    return True


def does_contract_use_contract_slot(contract: Any, season: Optional[int] = None) -> bool:
    """NHL SPC eligibility — prefer explicit category; never infer solely from salary.

    Salary validation is separate: a malformed AAV does not drop an explicit SPC.
    """
    c = normalize_contract_payload(contract)
    if not c:
        return False
    ctype = _normalize_contract_type_token(c.get("contract_type") or c.get("type") or "")
    if ctype in _NON_NHL_SPC_TYPES:
        return False
    # Explicit flags win over inferred type aliases.
    if c.get("is_nhl_spc") is False or c.get("nhl_spc") is False or c.get("standard_player_contract") is False:
        return False
    if c.get("is_nhl_spc") is True or c.get("nhl_spc") is True or c.get("standard_player_contract") is True:
        return is_contract_active(c, season)
    # Legacy: typed NHL deal without flag — active years gate; AAV is advisory only.
    if not is_contract_active(c, season):
        return False
    yrs = int(c.get("years_remaining") or 0)
    if yrs <= 0:
        return False
    return True


def uses_nhl_contract_slot(player: Any, season: Optional[int] = None) -> bool:
    """NHL SPC consumes a 50-contract reserve slot regardless of NHL/AHL/ECHL assignment.

    One-way and two-way NHL deals both count. Pure AHL/ECHL/PTO deals do not.
    Retained-salary records are not SPCs and never count here.
    """
    if player is None or bool(_get(player, "retired", False)):
        return False
    if str(_get(player, "signed_status", "") or "").lower() == "unsigned":
        return False
    return does_contract_use_contract_slot(player, season)


def _org_player_dedupe_key(player: Any) -> str:
    """Stable dedupe key. Missing IDs use object identity — never collapse many Nones to one."""
    pid = _player_id(player)
    if pid:
        return f"id:{pid}"
    return f"obj:{id(player)}"


def iter_org_contract_players(team: Any) -> List[Any]:
    """Unique org players across roster / AHL / ECHL / prospect_pool.

    prospect_pool may mix rights-only and signed development players; slot counting
    still filters via uses_nhl_contract_slot. Containers are ownership/assignment lists;
    flags like roster_location must be synced to the list that holds the player.
    """
    seen: set = set()
    out: List[Any] = []
    for attr in ("roster", "ahl_roster", "echl_roster", "prospect_pool"):
        for p in list(_get(team, attr, None) or []):
            if bool(_get(p, "retired", False)):
                continue
            key = _org_player_dedupe_key(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def get_contract_display_summary(contract: Any, season: Optional[int] = None) -> Dict[str, Any]:
    c = normalize_contract_payload(contract)
    return {
        "type": c.get("contract_type") or c.get("type"),
        "years_remaining": get_contract_years_remaining(c, season),
        "aav_m": float(c.get("aav_m") or 0),
        "cap_hit_m": get_contract_cap_hit(c, season),
        "nhl_salary_m": get_contract_nhl_salary(c, season),
        "minor_salary_m": get_contract_minor_salary(c, season),
        "signing_bonus_m": float(c.get("signing_bonus_m") or 0),
        "performance_bonus_m": get_contract_bonus_total(c, season),
        "two_way": bool(c.get("two_way")),
        "is_entry_level": bool(c.get("is_entry_level") or str(c.get("type") or "").upper() == "ELC"),
        "clause": get_contract_clause(c, season),
        "schema_version": c.get("schema_version"),
    }


def get_player_contract_status(player: Any, season: Optional[int] = None) -> Dict[str, Any]:
    c = normalize_contract_payload(player)
    return {
        "has_contract": bool(c) and is_contract_active(c, season),
        "signed_status": str(getattr(player, "signed_status", "") or ""),
        "rights_status": str(getattr(player, "rights_status", "") or c.get("rights_status") or ""),
        "organizational_status": str(getattr(player, "organizational_status", "") or ""),
        "contract": get_contract_display_summary(c, season) if c else None,
        "elc_eligible": bool(getattr(player, "entry_level_contract_eligible", False)),
        "slide_eligible": bool(getattr(player, "elc_slide_eligible", False) or c.get("slide_eligible")),
    }


def contract_validation_result(
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


def _contract_cap_hit_m(player: Any) -> float:
    c = normalize_contract_payload(player)
    if c.get("cap_hit_m", 0) > 0:
        return float(c["cap_hit_m"])
    try:
        return max(0.0, float(player_cap_hit_millions(player)))
    except Exception:
        return 0.0


def _contract_years_remaining(player: Any) -> int:
    c = normalize_contract_payload(player)
    if not c:
        return 0
    try:
        return max(0, int(c.get("years_remaining") or 0))
    except (TypeError, ValueError):
        return 0


def has_active_contract(player: Any) -> bool:
    c = normalize_contract_payload(player)
    if not c:
        return False
    yrs = _contract_years_remaining(player)
    if yrs <= 0:
        return False
    cap = _contract_cap_hit_m(player)
    if cap <= 0:
        return False
    ctype = str(c.get("type") or c.get("contract_type") or "").upper()
    if not ctype:
        return False
    if str(_get(player, "signed_status", "") or "").lower() == "unsigned":
        return False
    return True


def has_true_elc_contract(player: Any) -> bool:
    if not has_active_contract(player):
        return False
    c = normalize_contract_payload(player)
    if str(c.get("type") or c.get("contract_type") or "").upper() != "ELC":
        return False
    cap = _contract_cap_hit_m(player)
    return cap > 0 and abs(cap - ELC_AAV_M) <= ELC_AAV_TOLERANCE


def has_elc_contract(player: Any) -> bool:
    """Alias — only true ELC deals at ELC_AAV_M."""
    return has_true_elc_contract(player)


def _clear_expired_contract(player: Any) -> None:
    try:
        player.contract = None
        player.cap_hit_m = 0.0
        player.aav_m = 0.0
    except Exception:
        pass


def _strip_malformed_contract(player: Any) -> None:
    if _get(player, "contract", None) is not None and not has_active_contract(player):
        _clear_expired_contract(player)


def hydrate_player_contract(player: Any) -> None:
    """Normalize legacy contract money on load. Idempotent."""
    _strip_malformed_contract(player)
    c = _get(player, "contract", None)
    if c is None:
        hit = player_cap_hit_millions(player)
        if hit <= 0:
            return
        normalized = normalize_contract_dict({
            "aav_m": hit, "cap_hit_m": hit,
            "years_remaining": max(0, int(_get(player, "years_remaining", 0) or 0)),
        })
    else:
        normalized = normalize_contract_dict(c)

    try:
        if isinstance(c, dict):
            c.clear()
            c.update(normalized)
        else:
            for k, v in normalized.items():
                try:
                    setattr(c, k, v)
                except Exception:
                    pass
    except Exception:
        try:
            player.contract = normalized
        except Exception:
            pass

    try:
        player.cap_hit_m = normalized["cap_hit_m"]
        player.aav_m = normalized["aav_m"]
    except Exception:
        pass


def apply_contract_to_player(player: Any, contract: Dict[str, Any], season_year: int) -> None:
    normalized = normalize_contract_dict(contract)
    yrs = int(normalized.get("years_remaining") or normalized.get("years") or 1)
    if not normalized.get("expiry_year"):
        normalized["expiry_year"] = int(season_year) + yrs

    existing = _get(player, "contract", None)
    if isinstance(existing, dict):
        existing.clear()
        existing.update(normalized)
    elif existing is not None:
        for k, v in normalized.items():
            try:
                setattr(existing, k, v)
            except Exception:
                pass
    else:
        try:
            player.contract = normalized
        except Exception:
            pass

    try:
        player.cap_hit_m = normalized["cap_hit_m"]
        player.aav_m = normalized["aav_m"]
        player.rights_status = normalized["rights_status"]
    except Exception:
        pass

    # Extension / re-sign replaces the pending July-1 burn — clear exclusivity flags
    # so the player is not force-expired into free agency on market open.
    try:
        setattr(player, "pending_july1_expiry", False)
    except Exception:
        pass
    try:
        setattr(player, "ufa_exclusive", False)
    except Exception:
        pass
    if isinstance(normalized, dict):
        normalized.pop("pending_july1_expiry", None)
    existing_after = _get(player, "contract", None)
    if isinstance(existing_after, dict):
        existing_after.pop("pending_july1_expiry", None)

def get_team_cap_snapshot_full(
    team: Any,
    league: Any = None,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
    calendar_cursor: int = 0,
    regular_season_last_index: int = 192,
) -> Dict[str, Any]:
    season_label = None
    if season_year is not None:
        season_label = f"{int(season_year)}-{(int(season_year) + 1) % 100:02d}"

    raw = calculate_team_cap_snapshot(
        team,
        league=league,
        sim=sim,
        season_label=season_label,
        calendar_cursor=calendar_cursor,
        regular_season_last_index=regular_season_last_index,
    )

    roster = _all_rostered(team)
    slots_used = _count_team_contract_slots(team)

    projected_next = None
    hist = _get(league, "cap_history", None) or []
    if hist:
        projected_next = float(_get(hist[-1], "upperLimit", 0) or 0)
    elif raw.get("upperLimit"):
        projected_next = round(float(raw["upperLimit"]) * 1.03, 3)

    return {
        "upper_limit_m": float(raw.get("upperLimit") or 0),
        "lower_limit_m": float(raw.get("lowerLimit") or 0),
        "total_cap_hit_m": float(raw.get("totalCapHit") or 0),
        "active_roster_cap_hit_m": float(raw.get("activeRosterCapHit") or 0),
        "buried_cap_hit_m": float(raw.get("buriedCapHit") or 0),
        "retained_salary_m": float(raw.get("retainedSalary") or 0),
        "buyout_cap_hit_m": float(raw.get("buyoutCapHit") or 0),
        "bonus_overage_m": float(raw.get("bonusOverage") or 0),
        "bonus_reserve_m": float(raw.get("performanceBonusReserve") or 0),
        "other_dead_cap_m": float(raw.get("otherDeadCap") or 0),
        "ltir_pool_m": float(raw.get("ltirPool") or 0),
        "real_cap_space_m": float(raw.get("realCapSpace") or 0),
        "usable_cap_space_m": float(raw.get("usableCapSpace") or raw.get("capSpace") or 0),
        "offseason_space_m": float(raw.get("realCapSpace") or 0),
        "projected_next_year_upper_limit_m": projected_next,
        "contract_slots_used": int(slots_used),
        "contract_slots_limit": CONTRACT_SLOTS_LIMIT,
        # Alias for UI clarity — same integer as contract_slots_used (API-compatible).
        "nhl_spcs_used": int(slots_used),
        "nhl_spcs_limit": CONTRACT_SLOTS_LIMIT,
        "active_roster_count": int(raw.get("activeRosterCount") or 0),
        "active_roster_max": int(raw.get("activeRosterMax") or 23),
        "warnings": list(raw.get("warnings") or []),
        "_raw": raw,
    }


def sync_team_cap_fields(team: Any, league: Any, sim: Any = None, **kwargs) -> Dict[str, Any]:
    snap = get_team_cap_snapshot_full(team, league, sim, **kwargs)
    try:
        team.salary_cap_m = snap["upper_limit_m"]
        team.salary_cap = snap["upper_limit_m"]
        team.cap_limit = snap["upper_limit_m"]
        team.total_cap_hit = snap["total_cap_hit_m"]
        team.cap_hit = snap["total_cap_hit_m"]
        team.total_salary = snap["total_cap_hit_m"]
        team.cap_space = snap["usable_cap_space_m"]
        team.cap_space_m = snap["usable_cap_space_m"]
        team.cap_snapshot = snap
    except Exception:
        pass
    return snap


def sync_all_team_cap_fields(league: Any, sim: Any = None, **kwargs) -> int:
    """Refresh usable cap space on every club — required so CPU FA bidding works."""
    n = 0
    for team in list(_get(league, "teams", None) or []):
        try:
            sync_team_cap_fields(team, league, sim, **kwargs)
            n += 1
        except Exception:
            continue
    return n


def team_cap_snapshot_legacy_compat(snap: Dict[str, Any]) -> Dict[str, float]:
    """Backward-compatible {salary_cap, cap_hit, cap_space} for old payloads."""
    return {
        "salary_cap": round(snap["upper_limit_m"], 3),
        "cap_hit": round(snap["total_cap_hit_m"], 3),
        "cap_space": round(snap["usable_cap_space_m"], 3),
    }


# ---------------------------------------------------------------------------
# Valuation — team importance, peer gap, fair AAV
# ---------------------------------------------------------------------------

def compute_team_importance_score(player: Any, team: Any) -> float:
    cache = _get(team, "_bootstrap_contract_ovr_cache", None)
    player_key = id(player)
    if isinstance(cache, dict):
        ovr = float((_get(cache, "by_player_id", None) or {}).get(player_key, _player_ovr(player)))
    else:
        ovr = _player_ovr(player)
    pos = _player_pos(player)
    roster = _active_roster(team)
    if not roster:
        return 0.5

    if isinstance(cache, dict):
        ovrs = list(_get(cache, "all_ovrs_desc", None) or [])
    else:
        ovrs = sorted((_player_ovr(p) for p in roster), reverse=True)
    rank = next((i for i, v in enumerate(ovrs) if v <= ovr + 0.01), len(ovrs))
    rank_score = 1.0 - (rank / max(1, len(ovrs)))

    next_best = ovrs[1] if len(ovrs) > 1 and ovrs[0] >= ovr - 0.01 else (ovrs[0] if ovrs else 0)
    if ovrs and ovrs[0] >= ovr - 0.01 and len(ovrs) > 1:
        next_best = ovrs[1]
    elif ovrs:
        next_best = max((v for v in ovrs if v < ovr - 0.01), default=ovrs[-1])
    gap = max(0.0, ovr - next_best)
    gap_score = min(1.0, gap / 12.0)

    if isinstance(cache, dict):
        pos_ovr_map = _get(cache, "pos_ovrs_desc", None) or {}
        pos_ovrs = list(pos_ovr_map.get(pos, []))
    else:
        pos_ovrs = sorted((_player_ovr(p) for p in roster if _player_pos(p) == pos or (pos == "D" and _player_pos(p) == "D")), reverse=True)
    pos_gap = 0.0
    if pos_ovrs:
        pos_next = pos_ovrs[1] if pos_ovrs[0] >= ovr - 0.01 and len(pos_ovrs) > 1 else pos_ovrs[0]
        pos_gap = max(0.0, ovr - pos_next)
    pos_gap_score = min(1.0, pos_gap / 10.0)

    stats = _get(player, "season_stats", None) or {}
    pts = float(stats.get("pts", 0) or 0) if isinstance(stats, dict) else 0.0
    prod_score = min(1.0, pts / 80.0)

    return round(min(1.0, 0.35 * rank_score + 0.30 * gap_score + 0.20 * pos_gap_score + 0.15 * prod_score), 3)


def compute_peer_gap_score(player: Any, team: Any) -> float:
    cache = _get(team, "_bootstrap_contract_ovr_cache", None)
    player_key = id(player)
    if isinstance(cache, dict):
        ovr = float((_get(cache, "by_player_id", None) or {}).get(player_key, _player_ovr(player)))
    else:
        ovr = _player_ovr(player)
    roster = _active_roster(team)
    if len(roster) < 2:
        return 0.0
    if isinstance(cache, dict):
        ovrs = list(_get(cache, "all_ovrs_desc", None) or [])
    else:
        ovrs = sorted((_player_ovr(p) for p in roster), reverse=True)
    if ovrs[0] >= ovr - 0.01:
        second = ovrs[1] if len(ovrs) > 1 else ovrs[0]
    else:
        second = ovrs[0]
    return round(min(15.0, max(0.0, ovr - second)), 2)


def _player_production_score(player: Any) -> float:
    """League-wide on-ice results in [0,1]. 0.5 = neutral / unknown so rating-
    driven value is unaffected when no stats exist (keeps bootstrap stable)."""
    stats = _get(player, "season_stats", None) or {}
    if not isinstance(stats, dict) or not stats:
        return 0.5
    if _player_pos(player) == "G":
        sv = stats.get("save_pct") or stats.get("sv_pct")
        if sv:
            try:
                svf = float(sv)
                if svf > 1.5:
                    svf /= 100.0
                return max(0.0, min(1.0, (svf - 0.880) / (0.930 - 0.880)))
            except (TypeError, ValueError):
                pass
        return 0.5
    pts = float(stats.get("pts", stats.get("points", 0)) or 0)
    gp = float(stats.get("gp", stats.get("games_played", 0)) or 0)
    if gp > 0:
        ppg = pts / gp
        prod = (ppg - 0.20) / (0.90 - 0.20)
    else:
        prod = pts / 80.0
    return max(0.0, min(1.0, prod))


def _player_experience_factor(player: Any) -> float:
    """Professional experience confidence in [0,1] — scales how much of a
    potential premium the market will actually pay (item 16)."""
    stats = _get(player, "season_stats", None) or {}
    seasons = _get(player, "pro_seasons", None) or _get(player, "nhl_seasons", None)
    if seasons:
        try:
            return max(0.0, min(1.0, float(seasons) / 4.0))
        except (TypeError, ValueError):
            pass
    if isinstance(stats, dict):
        gp = float(stats.get("career_gp", stats.get("nhl_gp", 0)) or 0)
        if gp > 0:
            return max(0.0, min(1.0, gp / 200.0))
    age = _player_age(player)
    return max(0.0, min(1.0, (age - 18) / 6.0))


def _player_negotiation_profile(player: Any) -> Dict[str, float]:
    """Deterministic, hidden per-player personality (item 10) plus a small
    fixed interest variance (item 5). Stable across calls for the same id."""
    seed = abs(hash(("nego", _player_id(player)))) & 0xFFFFFFFF
    r = random.Random(seed)
    return {
        "security_pref": round(r.uniform(0.0, 1.0), 3),
        "gamble_pref": round(r.uniform(0.0, 1.0), 3),
        "loyalty": round(r.uniform(0.0, 1.0), 3),
        "competitiveness": round(r.uniform(0.0, 1.0), 3),
        "variance": round(r.uniform(-0.05, 0.05), 4),
    }


def compute_market_value(player: Any, league: Any = None) -> float:
    """League-wide market value — independent of any negotiating team (item 3).

    Rating-driven baseline, shifted modestly by production (item 15) with a
    potential premium gated by professional experience (item 16). This is the
    'value 1: market value' of the five-value model.
    """
    ovr = _player_ovr(player)
    age = _player_age(player)
    pos = _player_pos(player)
    pot = _player_potential(player)

    # Depth / replacement players hug the league minimum; mid-tier ramps faster,
    # stars pay the premium. Old flat (ovr-58)*0.14 put OVR 74 at ~$3M.
    if ovr < 70:
        base = LEAGUE_MINIMUM_AAV_M + max(0.0, ovr - 55.0) * 0.035
    elif ovr < 78:
        base = 1.15 + max(0.0, ovr - 70.0) * 0.18
    else:
        base = LEAGUE_MINIMUM_AAV_M + max(0.0, ovr - 58.0) * 0.12 + max(0.0, ovr - 82.0) * 0.42
    # Production shifts value +/-~15% so results matter without overriding rating.
    base *= 0.85 + 0.30 * _player_production_score(player)
    # Only pay a youth-potential premium when there is some pro proof.
    if pot > 80 and age <= 24:
        base *= 1.0 + 0.08 * _player_experience_factor(player)
    if pos == "G" and ovr < 88:
        base *= 0.88
    if age >= 35:
        base *= 0.72
    elif age >= 32:
        base *= 0.84
    elif age <= 22:
        exp = _player_experience_factor(player)
        youth_ceiling = 3.2 + max(0.0, ovr - 80.0) * 0.55
        base = min(base, youth_ceiling) * (0.80 + 0.20 * exp)
    # True depth / AHL call-ups accept near-min money (CHEAP board, veterans, etc.).
    if ovr < 75:
        base = min(base, LEAGUE_MINIMUM_AAV_M + 0.85 + max(0.0, ovr - 65.0) * 0.10)
    return round(max(LEAGUE_MINIMUM_AAV_M, base), 3)


def compute_fair_aav(player: Any, team: Optional[Any] = None, league: Any = None) -> float:
    """Fair value is league-wide and independent of the negotiating team
    (items 2 & 3). Team-specific leverage now lives in compute_player_demand and
    is applied exactly once. `team` is accepted for backward compatibility but
    intentionally no longer changes the result."""
    return compute_market_value(player, league)


def compute_bad_contract_score(player: Any, team: Optional[Any] = None) -> float:
    aav = player_cap_hit_millions(player)
    if aav <= 0:
        return 0.0
    fair = compute_fair_aav(player, team)
    yrs = _contract_years_remaining(player)
    age = _player_age(player)
    overpay = aav - fair
    ratio = aav / max(0.75, fair)
    term_risk = 1.0
    if yrs >= 5 and age >= 30:
        term_risk = 1.25
    elif yrs >= 4 and age >= 32:
        term_risk = 1.35
    score = max(0.0, (overpay / max(0.5, fair)) * term_risk * (ratio - 1.0))
    return round(min(2.0, score), 3)


def compute_contract_tags(player: Any, team: Optional[Any] = None) -> List[str]:
    tags: List[str] = []
    c = normalize_contract_dict(_get(player, "contract", None) or {})
    aav = player_cap_hit_millions(player) or c.get("aav_m", 0)
    ovr = _player_ovr(player)
    yrs = _contract_years_remaining(player)
    age = _player_age(player)
    rights = c.get("rights_status", "UFA")

    if yrs <= 1:
        tags.append("Expiring")
    if c.get("pending_july1_expiry") or getattr(player, "pending_july1_expiry", False):
        tags.append("Extension Window")
    if c.get("type") == "ELC" or c.get("contract_type") == "ELC":
        tags.append("ELC Value")
    if c.get("type") == "RFA_BRIDGE":
        tags.append("Bridge Deal")

    bad_type = c.get("bad_contract_type")
    if bad_type and bad_type in BAD_TAG_MAP:
        tag = BAD_TAG_MAP[bad_type]
        if tag not in tags:
            tags.append(tag)
    elif compute_bad_contract_score(player, team) >= 0.35:
        tags.append("Bad Deal")

    fair = compute_fair_aav(player, team)
    if 0 < aav < fair * 0.82 and ovr >= 74:
        tags.append("Bargain")
    if ovr >= 88 and yrs >= 3:
        tags.append("Core Value")
    if c.get("nmc") or c.get("no_move_clause"):
        tags.append("Clause Locked")
    elif c.get("ntc") or c.get("no_trade_clause"):
        tags.append("Clause Locked")
    if rights == "RFA":
        tags.append("RFA")
    return tags


# ---------------------------------------------------------------------------
# Contract term generation — team-aware, bad contracts at bootstrap
# ---------------------------------------------------------------------------

def generate_contract_terms(
    player: Any,
    team: Optional[Any],
    league: Any,
    rng: random.Random,
    *,
    max_aav_m: Optional[float] = None,
    allow_bad: bool = False,
    context: str = "bootstrap",
) -> Tuple[float, int, Dict[str, Any]]:
    ovr = _player_ovr(player)
    age = _player_age(player)
    pos = _player_pos(player)
    pot = _player_potential(player)
    fair = compute_fair_aav(player, team, league)
    importance = compute_team_importance_score(player, team) if team else 0.5
    gap = compute_peer_gap_score(player, team) if team else 0.0

    # Bootstrap/history contracts reflect what the signing team paid, so team
    # leverage is applied locally here. compute_fair_aav is now league-wide
    # (items 2 & 3); negotiation demand applies its own single leverage.
    if team is not None:
        fair = fair * (1.0 + 0.12 * importance + 0.04 * min(gap, 10.0))

    spread = 0.18 + 0.08 * importance
    aav = fair * rng.uniform(1.0 - spread * 0.5, 1.0 + spread)

    if ovr >= 94:
        aav = max(aav, rng.uniform(11.0, 14.0))
    elif ovr >= 90:
        aav = max(aav, rng.uniform(8.5, 12.0))
    elif ovr >= 86:
        aav = max(aav, rng.uniform(6.0, 9.5))

    if gap >= 8.0 and ovr >= 86:
        aav *= rng.uniform(1.05, 1.18)
    if gap <= 3.0 and ovr >= 90:
        aav *= rng.uniform(0.92, 1.02)

    if pos == "G" and ovr < 90:
        aav *= 0.9
    if age <= 22 and ovr < 86:
        aav = min(aav, rng.uniform(0.85, 3.4))
    if age >= 35:
        aav *= rng.uniform(0.72, 0.95)

    meta: Dict[str, Any] = {}
    bad_type = None
    if allow_bad and context == "bootstrap" and rng.random() < 0.07:
        bad_type = _pick_bad_contract_type(player, team, rng)
        aav = _apply_bad_contract_premium(aav, fair, bad_type, rng)
        meta["bad_contract_type"] = bad_type

    aav = max(LEAGUE_MINIMUM_AAV_M, round(aav, 3))
    if max_aav_m is not None:
        ceiling = max(0.0, float(max_aav_m))
        if ceiling < LEAGUE_MINIMUM_AAV_M:
            aav = round(max(0.0, ceiling), 3) or LEAGUE_MINIMUM_AAV_M
        else:
            aav = min(aav, ceiling)

    years = _term_for_profile(ovr, age, pot, importance, rng)
    meta["team_importance_score"] = importance
    meta["peer_gap_score"] = gap
    meta["fair_aav_m"] = fair
    return aav, years, meta


def _term_for_profile(ovr: float, age: int, pot: float, importance: float, rng: random.Random) -> int:
    if age <= 22:
        return rng.randint(2, 3)
    if age <= 26:
        base = rng.randint(2, 8) if ovr >= 84 else rng.randint(1, 5)
        if importance >= 0.75 and ovr >= 84:
            base = min(8, base + rng.randint(0, 2))
        return base
    if age <= 30:
        return rng.randint(2, 7) if ovr >= 86 else rng.randint(1, 4)
    if age <= 34:
        return rng.randint(1, 3)
    return 1 if rng.random() < 0.75 else 2


def _pick_bad_contract_type(player: Any, team: Optional[Any], rng: random.Random) -> str:
    ovr = _player_ovr(player)
    age = _player_age(player)
    pos = _player_pos(player)
    weights: List[Tuple[str, float]] = []
    if ovr < 78:
        weights.append(("overpaid_depth", 3.0))
        weights.append(("middle_six_top_six_pay", 2.5))
    if age >= 32 and ovr >= 84:
        weights.append(("aging_star_term", 2.5))
    if pos == "G":
        weights.append(("goalie_gamble", 2.0))
    if age >= 30:
        weights.append(("declining_expensive", 2.0))
        weights.append(("clause_veteran", 1.5))
    weights.append(("panic_ufa", 1.0))
    weights.append(("injury_risk", 1.0))
    total = sum(w for _, w in weights)
    roll = rng.random() * total
    for kind, w in weights:
        roll -= w
        if roll <= 0:
            return kind
    return weights[0][0]


def _apply_bad_contract_premium(aav: float, fair: float, bad_type: str, rng: random.Random) -> float:
    mult = {
        "overpaid_depth": rng.uniform(1.35, 1.75),
        "aging_star_term": rng.uniform(1.15, 1.35),
        "goalie_gamble": rng.uniform(1.20, 1.50),
        "panic_ufa": rng.uniform(1.25, 1.55),
        "clause_veteran": rng.uniform(1.10, 1.30),
        "injury_risk": rng.uniform(1.15, 1.40),
        "declining_expensive": rng.uniform(1.20, 1.45),
        "middle_six_top_six_pay": rng.uniform(1.30, 1.65),
    }.get(bad_type, 1.25)
    return max(aav, fair * mult)


def contract_type_and_rights(age: int, ovr: float, *, true_elc: bool = False) -> Tuple[str, str]:
    if true_elc:
        return ("ELC", "RFA")
    if age <= 23 and ovr < 82:
        return ("RFA_BRIDGE", "RFA")
    if age < 27:
        return ("STANDARD", "RFA" if age < 25 else "UFA")
    return ("STANDARD", "UFA")


def build_contract_for_player(
    player: Any,
    team: Optional[Any],
    league: Any,
    season_year: int,
    rng: random.Random,
    *,
    max_aav_m: Optional[float] = None,
    allow_bad: bool = False,
    override_aav: Optional[float] = None,
    override_years: Optional[int] = None,
    contract_type: Optional[str] = None,
) -> Dict[str, Any]:
    age = _player_age(player)
    ovr = _player_ovr(player)
    aav, years, meta = generate_contract_terms(
        player, team, league, rng, max_aav_m=max_aav_m, allow_bad=allow_bad,
    )
    if override_aav is not None:
        aav = round(float(override_aav), 3)
    if override_years is not None:
        years = max(1, int(override_years))

    ctype, rights = contract_type_and_rights(age, ovr)
    if contract_type:
        ctype = str(contract_type).upper()
    if ctype == "ELC":
        aav = ELC_AAV_M
        ctype = "ELC"

    ntc = bool(ovr >= 88 and age >= 28 and rng.random() < 0.55)
    nmc = bool(ovr >= 92 and age >= 29 and rng.random() < 0.40)
    if meta.get("bad_contract_type") == "clause_veteran":
        ntc = True if rng.random() < 0.7 else ntc

    two_way = bool(ovr < 72 and age <= 25)
    contract = {
        "type": ctype,
        "contract_type": ctype,
        "years": years,
        "years_remaining": years,
        "aav_m": aav,
        "cap_hit_m": aav,
        "base_salary_m": aav,
        "salary_m": aav,
        "signing_bonus_m": 0.0,
        "performance_bonus_m": 0.0,
        "buyout_penalty_m": 0.0,
        "rights_status": rights,
        "rights": rights,
        "expiry_status": rights,
        "expiry_year": int(season_year) + years,
        "ntc": ntc,
        "nmc": nmc,
        "no_trade_clause": ntc,
        "no_move_clause": nmc,
        "two_way": two_way,
        "source": "bootstrap" if allow_bad else "generated",
        **{k: v for k, v in meta.items() if k not in ("fair_aav_m",)},
    }
    return normalize_contract_dict(contract)


def build_elc_contract(
    season_year: int,
    performance_bonus_m: float = 0.0,
    *,
    offer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if offer:
        try:
            from services.elc_offer_engine import offer_to_contract_dict

            return normalize_contract_dict(offer_to_contract_dict(offer, season_year))
        except Exception:
            pass
    return normalize_contract_dict({
        "type": "ELC",
        "contract_type": "ELC",
        "years": 3,
        "years_remaining": 3,
        "aav_m": ELC_AAV_M,
        "cap_hit_m": ELC_AAV_M,
        "base_salary_m": ELC_AAV_M,
        "salary_m": ELC_AAV_M,
        "signing_bonus_m": 0.0,
        "performance_bonus_m": performance_bonus_m,
        "schedule_a_bonus_m": 0.0,
        "schedule_b_bonus_m": 0.0,
        "minor_salary_m": 0.085,
        "two_way": True,
        "is_entry_level": True,
        "rights_status": "RFA",
        "rights": "RFA",
        "expiry_status": "RFA",
        "expiry_year": int(season_year) + 3,
        "effective_season": int(season_year),
        "source": "elc",
        "schema_version": CONTRACT_SCHEMA_VERSION,
    })


# ---------------------------------------------------------------------------
# Prospect / ELC pipeline
# ---------------------------------------------------------------------------

def _ensure_reserve_list(team: Any) -> List[Dict[str, Any]]:
    reserve = _get(team, "reserve_list", None)
    if not isinstance(reserve, list):
        try:
            team.reserve_list = []
        except Exception:
            pass
        return getattr(team, "reserve_list", [])
    return reserve


def _contract_type(player: Any) -> str:
    c = normalize_contract_payload(player)
    return str(c.get("contract_type") or c.get("type") or "")


def is_elc_eligible(player: Any) -> bool:
    if not bool(_get(player, "entry_level_contract_eligible", False)):
        return False
    if str(_get(player, "signed_status", "") or "").lower() == "signed":
        return False
    if has_true_elc_contract(player):
        return False
    if has_active_contract(player):
        return False
    return True


def _count_team_contract_slots(team: Any) -> int:
    """Count distinct NHL SPCs across the organization (assignment-agnostic)."""
    return sum(1 for p in iter_org_contract_players(team) if uses_nhl_contract_slot(p))


def validate_contract_slots(team: Any, league: Any, *, additional: int = 1) -> Dict[str, Any]:
    used = _count_team_contract_slots(team)
    limit = CONTRACT_SLOTS_LIMIT
    if used + max(0, int(additional)) > limit:
        return {
            "ok": False,
            "reason": f"Contract slot limit reached ({used}/{limit})",
            "contract_slots_used": used,
            "contract_slots_limit": limit,
        }
    return {"ok": True, "contract_slots_used": used, "contract_slots_limit": limit}


def add_to_reserve_list(
    team: Any,
    player: Any,
    *,
    draft_year: Optional[int] = None,
    draft_overall: Optional[int] = None,
    added_season: Optional[int] = None,
) -> Dict[str, Any]:
    """Store stable identifiers only — never live player_ref objects."""
    reserve = _ensure_reserve_list(team)
    pid = _player_id(player)
    tid = str(_get(team, "team_id", None) or _get(team, "id", "") or "")
    rights_fields = {
        "rights_team_id": str(_get(player, "nhl_rights_team_id", None) or _get(player, "rights_team_id", None) or tid),
        "signed_status": str(_get(player, "signed_status", "unsigned") or "unsigned"),
        "rights_status": _get(player, "rights_status", None),
        "rights_type": _get(player, "rights_type", None),
        "rights_expiry_year": _get(player, "rights_expiry_year", None),
        "rights_signing_deadline": _get(player, "rights_signing_deadline", None),
        "current_team_id": _get(player, "current_team_id", None),
        "current_league_id": _get(player, "current_league_id", None),
        "organizational_status": _get(player, "organizational_status", None),
    }
    for entry in reserve:
        if str(entry.get("player_id", "")) == pid:
            entry.update({
                "name": _player_name(player),
                "position": _player_pos(player),
                "draft_year": draft_year or entry.get("draft_year"),
                "draft_overall_pick": draft_overall or entry.get("draft_overall_pick"),
                "entry_level_contract_eligible": True,
                **rights_fields,
            })
            entry.pop("player_ref", None)
            return entry
    entry = {
        "player_id": pid,
        "name": _player_name(player),
        "position": _player_pos(player),
        "draft_year": draft_year,
        "draft_overall_pick": draft_overall,
        "added_season": added_season,
        "entry_level_contract_eligible": True,
        **rights_fields,
    }
    reserve.append(entry)
    return entry


def remove_from_reserve_list(team: Any, player_id: str) -> Optional[Dict[str, Any]]:
    reserve = _ensure_reserve_list(team)
    pid = str(player_id)
    for idx, entry in enumerate(reserve):
        if str(entry.get("player_id", "")) == pid:
            return reserve.pop(idx)
    return None


def find_prospect_on_team(team: Any, player_id: str) -> Optional[Any]:
    pid = str(player_id)
    for entry in _ensure_reserve_list(team):
        if str(entry.get("player_id", "")) == pid:
            ref = entry.get("player_ref")
            if ref is not None:
                return ref
    for p in _get(team, "prospect_pool", None) or []:
        if _player_id(p) == pid:
            return p
    for p in _all_rostered(team):
        if _player_id(p) == pid:
            return p
    return None


def assign_elc_contract(
    player: Any,
    team: Any,
    league: Any,
    season_year: int,
    *,
    offer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not is_elc_eligible(player) and not has_true_elc_contract(player):
        return contract_validation_result(
            allowed=False,
            error_code="not_elc_eligible",
            title="Not ELC eligible",
            user_message="Player is not ELC eligible",
            blocking_reasons=["Player is not ELC eligible"],
        )
    if has_true_elc_contract(player):
        return {
            "ok": True,
            "allowed": True,
            "contract": normalize_contract_dict(_get(player, "contract", None) or {}),
            "already_signed": True,
        }

    slot_check = validate_contract_slots(team, league, additional=1)
    if not slot_check.get("ok"):
        return contract_validation_result(
            allowed=False,
            error_code="contract_slots_full",
            title="Contract slots full",
            user_message=slot_check.get("reason") or "No contract slots available",
            blocking_reasons=[slot_check.get("reason") or "No contract slots available"],
            projected_contract_slots=slot_check,
        )

    contract = build_elc_contract(season_year, offer=offer)
    on_roster = player in _all_rostered(team)
    if on_roster:
        check = _validate_sign_cap(team, contract["aav_m"], league)
        if not check.get("ok"):
            return contract_validation_result(
                allowed=False,
                error_code="cap_exceeded",
                title="Cap exceeded",
                user_message=check.get("reason") or "Signing exceeds cap",
                blocking_reasons=[check.get("reason") or "Signing exceeds cap"],
                projected_cap_hit=contract["aav_m"],
                snapshot=check.get("snapshot"),
            )

    apply_contract_to_player(player, contract, season_year)
    try:
        player.signed_status = "signed"
        player.entry_level_contract_eligible = False
        player.rights_status = "RFA"
        player.prospect_status = "signed_prospect"
        player.organizational_status = "signed_unassigned"
        if contract.get("development_promise"):
            player.development_promise = contract.get("development_promise")
        if contract.get("slide_eligible") or contract.get("can_slide"):
            player.elc_slide_eligible = True
            player.elc_slide_years_remaining = max(
                1, int(getattr(player, "elc_slide_years_remaining", 1) or 1)
            )
            player.contract_burned = False
    except Exception:
        pass

    reserve = _ensure_reserve_list(team)
    for entry in reserve:
        if str(entry.get("player_id", "")) == _player_id(player):
            entry["signed_status"] = "signed"
            entry["entry_level_contract_eligible"] = False
            entry["organizational_status"] = "signed_unassigned"
            break

    # NHL CBA: signing a bonus-eligible ELC reserves the maximum potential
    # Schedule A/B bonus against THIS season's cap immediately (it's not free
    # cap space just because the bonus hasn't been earned yet). Fund the club's
    # bonus reserve here so usable_cap_space_m reflects that exposure now,
    # instead of the whole bonus silently becoming "overage" against next
    # season's cap once it's actually earned (see apply_earned_bonuses_to_team_cap).
    bonus_exposure_m = float(
        contract.get("maximum_performance_bonus_m")
        or contract.get("performance_bonus_m")
        or 0
    )
    if bonus_exposure_m > 0:
        try:
            current_reserve = float(getattr(team, "performance_bonus_reserve_m", 0) or 0)
            new_reserve = round(current_reserve + bonus_exposure_m, 4)
            team.performance_bonus_reserve_m = new_reserve
            team.performance_bonus_reserve = new_reserve
            team.bonus_reserve_m = new_reserve
        except Exception:
            pass

    sync_team_cap_fields(team, league)
    return {
        "ok": True,
        "allowed": True,
        "contract": contract,
        "projected_contract_slots": {
            "used": int(slot_check.get("contract_slots_used") or 0) + 1,
            "limit": slot_check.get("contract_slots_limit"),
        },
    }


def assign_elc_from_offer(
    player: Any,
    team: Any,
    league: Any,
    season_year: int,
    offer: Dict[str, Any],
) -> Dict[str, Any]:
    return assign_elc_contract(player, team, league, season_year, offer=offer)


def apply_post_elc_assignment(
    session: Any,
    player: Any,
    team: Any,
    assignment_plan: Optional[str],
    *,
    season_year: int,
) -> Dict[str, Any]:
    """Apply developmental assignment after ELC sign — validates eligibility.

    This is the ONLY place a freshly-signed ELC player should land on the AHL roster.
    Every branch other than "assign_ahl" defensively clears any prior AHL placement so
    an assignment change (or a later NHL promotion) can never leave a player on both
    the AHL roster (in_minors=True) and an NHL-facing status at the same time.
    """
    plan = str(assignment_plan or "invite_camp").strip()
    age = _player_age(player)
    path = str(
        getattr(player, "development_path", None)
        or getattr(player, "post_draft_league", None)
        or ""
    ).upper()
    result: Dict[str, Any] = {"assignment": plan, "ok": True}

    from services.draft_rights_engine import move_prospect_to_ahl, remove_prospect_from_ahl

    if plan == "assign_ahl":
        if age < 20 and "EUROPE" not in path:
            return {
                "ok": False,
                "assignment": plan,
                "reason": "Not AHL-eligible under junior return rules",
                "organizational_status": getattr(player, "organizational_status", None),
            }
        try:
            setattr(player, "development_path", "AHL")
            setattr(player, "organizational_status", "signed_ahl")
            setattr(player, "current_league_id", getattr(player, "ahl_league_id", None) or "AHL")
        except Exception:
            pass
        moved = False
        try:
            moved = bool(
                move_prospect_to_ahl(getattr(getattr(session, "sim", None), "league", None), player, team)
            )
        except Exception:
            moved = False
        result["organizational_status"] = "signed_ahl"
        result["in_minors"] = bool(getattr(player, "in_minors", False))
        if not moved:
            result["ok"] = False
            result["reason"] = "Failed to place player on the AHL roster"
        return _mirror_post_elc_assignment_to_reserve(team, player, result)

    if plan == "return_junior":
        try:
            setattr(player, "organizational_status", "signed_junior")
            if "JUNIOR" not in path and "CHL" not in path and "OHL" not in path:
                setattr(player, "development_path", path or "JUNIOR")
        except Exception:
            pass
        result["organizational_status"] = "signed_junior"
    elif plan == "keep_college":
        try:
            setattr(player, "organizational_status", "signed_ncaa")
            setattr(player, "development_path", "NCAA")
        except Exception:
            pass
        result["organizational_status"] = "signed_ncaa"
    elif plan == "keep_europe":
        try:
            setattr(player, "organizational_status", "overseas_loan")
            setattr(player, "development_path", "EUROPE")
        except Exception:
            pass
        result["organizational_status"] = "overseas_loan"
    elif plan == "invite_camp":
        try:
            setattr(player, "camp_invite", True)
            setattr(player, "camp_invite_season", int(season_year))
            setattr(player, "organizational_status", "signed_unassigned")
        except Exception:
            pass
        result["organizational_status"] = "signed_unassigned"
        result["camp_invite"] = True
    else:
        try:
            setattr(player, "organizational_status", "signed_unassigned")
        except Exception:
            pass
        result["organizational_status"] = "signed_unassigned"

    try:
        remove_prospect_from_ahl(team, player)
    except Exception:
        pass
    result["in_minors"] = bool(getattr(player, "in_minors", False))
    return _mirror_post_elc_assignment_to_reserve(team, player, result)


def _mirror_post_elc_assignment_to_reserve(team: Any, player: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    for entry in _ensure_reserve_list(team):
        if str(entry.get("player_id", "")) == _player_id(player):
            entry["organizational_status"] = result.get("organizational_status")
            entry["signed_status"] = "signed"
            break
    return result


def auto_sign_elc_on_promotion(
    player: Any,
    team: Any,
    league: Any,
    season_year: int,
) -> Dict[str, Any]:
    if has_true_elc_contract(player):
        return {"ok": True, "skipped": True, "reason": "already_true_elc"}
    if has_active_contract(player):
        return {"ok": True, "skipped": True, "reason": "already_signed"}

    _strip_malformed_contract(player)

    age = _player_age(player)
    if age > 26:
        return {"ok": True, "skipped": True, "reason": "too_old_for_elc"}

    try:
        player.entry_level_contract_eligible = True
        player.signed_status = "unsigned"
    except Exception:
        pass

    result = assign_elc_contract(player, team, league, season_year)
    if result.get("ok"):
        return result

    fallback = build_elc_contract(season_year)
    apply_contract_to_player(player, fallback, season_year)
    try:
        player.signed_status = "signed"
        player.entry_level_contract_eligible = False
        player.rights_status = "RFA"
    except Exception:
        pass
    sync_team_cap_fields(team, league)
    return {"ok": True, "contract": fallback, "fallback": True, "reason": result.get("reason")}


def install_prospect_contract_hooks(league: Any) -> None:
    try:
        league._on_nhl_roster_promotion = auto_sign_elc_on_promotion
        league._on_roster_make_room = make_roster_room_for_promotion
    except Exception:
        pass


def promote_prospect_to_nhl(
    player: Any,
    team: Any,
    league: Any,
    season_year: int,
    *,
    auto_elc: bool = True,
) -> Dict[str, Any]:
    roster = list(_get(team, "roster", None) or [])
    if player in roster:
        if auto_elc and is_elc_eligible(player):
            return assign_elc_contract(player, team, league, season_year)
        return {"ok": True, "already_on_roster": True}

    snap = get_team_cap_snapshot_full(team, league, season_year=season_year)
    if int(snap.get("active_roster_count", 0)) >= int(snap.get("active_roster_max", 23)):
        return {"ok": False, "reason": "Active roster is full"}

    pool = list(_get(team, "prospect_pool", None) or [])
    if player in pool:
        pool.remove(player)
        team.prospect_pool = pool

    roster.append(player)
    team.roster = roster
    try:
        player.status = "nhl"
        player.prospect_status = "nhl"
    except Exception:
        pass
    # A player moving onto the active NHL roster can never simultaneously be an AHL
    # assignment — clear any stale minors placement so the two systems can't conflict
    # (e.g. a signing flow that set assignment_plan=assign_ahl before also promoting).
    try:
        from services.draft_rights_engine import remove_prospect_from_ahl

        remove_prospect_from_ahl(team, player)
    except Exception:
        pass

    if auto_elc:
        elc = auto_sign_elc_on_promotion(player, team, league, season_year)
        if not elc.get("ok"):
            return elc

    sync_team_cap_fields(team, league)
    return {"ok": True, "player_id": _player_id(player), "promoted": True}


def run_prospect_promotion_pass(session: Any) -> Dict[str, Any]:
    """
    Promote ready org prospects. Auto-ELC only when there is a clear organizational reason
    (rights risk, readiness, age) — never blanket-sign every eligible prospect.
    """
    sim = getattr(session, "sim", None)
    league = getattr(sim, "league", None) if sim else None
    if sim is None or league is None:
        return {"promoted": 0, "elc_signed": 0, "elc_skipped": 0}

    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    rng = getattr(sim, "rng", None) or random.Random(season_year)

    try:
        from services.draft_rights_engine import process_draft_rights_deadlines, should_cpu_auto_sign_elc

        process_draft_rights_deadlines(session, league, season_year)
    except Exception:
        should_cpu_auto_sign_elc = None  # type: ignore

    for team in _get(league, "teams", None) or []:
        for p in list(_get(team, "prospect_pool", None) or []):
            if bool(_get(p, "entry_level_contract_eligible", False)):
                add_to_reserve_list(team, p, added_season=season_year)

    promoted = 0
    elc_signed = 0
    elc_skipped = 0
    try:
        _, ages, _ = sim._run_prospect_promotion(rng, season_year)
        promoted = len(ages or [])
    except Exception:
        for team in _get(league, "teams", None) or []:
            for p in list(_get(team, "prospect_pool", None) or []):
                age = _player_age(p)
                yrs = int(_get(p, "development_years_remaining", 2) or 0)
                if yrs > 0 and age < 22:
                    continue
                # Promote without forcing ELC; signing evaluated separately.
                result = promote_prospect_to_nhl(p, team, league, season_year, auto_elc=False)
                if result.get("promoted"):
                    promoted += 1

    user_tid = str(getattr(session, "user_team_id", "") or "")
    for team in _get(league, "teams", None) or []:
        tid = str(_get(team, "team_id", None) or _get(team, "id", "") or "")
        # Never auto-sign the user's unsigned prospects — GM decision.
        if tid and tid == user_tid:
            continue
        targets = list(_get(team, "prospect_pool", None) or []) + list(_all_rostered(team))
        seen: set = set()
        for p in targets:
            pid = _player_id(p)
            if not pid or pid in seen:
                continue
            seen.add(pid)
            if not is_elc_eligible(p):
                continue
            if should_cpu_auto_sign_elc is not None:
                ok, reason = should_cpu_auto_sign_elc(p, team, season_year=season_year, league=league)
                if not ok:
                    elc_skipped += 1
                    continue
            else:
                reason = "fallback"
            res = assign_elc_contract(p, team, league, season_year)
            if res.get("ok") and not res.get("already_signed"):
                elc_signed += 1
                try:
                    setattr(p, "rights_status", "signed")
                    setattr(p, "organizational_status", "signed")
                    setattr(p, "signed_status", "signed")
                except Exception:
                    pass
                _ = reason

    wire = _ensure_waiver_wire(league)
    if any(not e.get("cleared") and not e.get("claimed_by") for e in wire):
        run_waiver_claim_pass(league, sim, season_year=season_year, user_team_id=user_tid)
        resolve_cleared_waivers(league, sim, season_year=season_year)

    return {"promoted": promoted, "elc_signed": elc_signed, "elc_skipped": elc_skipped}


# ---------------------------------------------------------------------------
# RFA rights
# ---------------------------------------------------------------------------

def _ensure_rfa_rights_list(team: Any) -> List[Dict[str, Any]]:
    rights = _get(team, "rfa_rights", None)
    if not isinstance(rights, list):
        try:
            team.rfa_rights = []
        except Exception:
            pass
        return getattr(team, "rfa_rights", [])
    return rights


def qualifying_offer_aav(previous_aav_m: float) -> float:
    """
    NHL CBA qualifying-offer salary brackets (millions of previous NHL salary).
    Article 10 schedule (modern CBA):
    - Under $1.000M: 110%
    - $1.000M–$1.499M: 105%
    - $1.500M and above: 100%
    Always floored at LEAGUE_MINIMUM_AAV_M.
    """
    prev = max(0.0, float(previous_aav_m or 0.0))
    if prev < 1.0:
        qo = prev * 1.10
    elif prev < 1.5:
        qo = prev * 1.05
    else:
        qo = prev * 1.00
    return round(max(LEAGUE_MINIMUM_AAV_M, qo), 3)


# Literal NHL offer-sheet compensation grid (AAV thresholds in millions).
# Source: CBA Article 10 / offer-sheet compensation schedule (original picks).
OFFER_SHEET_COMPENSATION_GRID: List[Dict[str, Any]] = [
    {
        "aav_floor_m": 14.051,
        "tier": "1st_1st_1st_1st_1st",
        "rounds": [1, 1, 1, 1, 1],
        "label": "Five 1st-round picks",
    },
    {
        "aav_floor_m": 10.628,
        "tier": "1st_1st_1st_1st",
        "rounds": [1, 1, 1, 1],
        "label": "Four 1st-round picks",
    },
    {
        "aav_floor_m": 8.503,
        "tier": "1st_1st_1st_2nd",
        "rounds": [1, 1, 1, 2],
        "label": "Three 1sts + 2nd",
    },
    {
        "aav_floor_m": 6.378,
        "tier": "1st_1st_1st_3rd",
        "rounds": [1, 1, 1, 3],
        "label": "Three 1sts + 3rd",
    },
    {
        "aav_floor_m": 4.784,
        "tier": "1st_1st_2nd_3rd",
        "rounds": [1, 1, 2, 3],
        "label": "Two 1sts + 2nd + 3rd",
    },
    {
        "aav_floor_m": 3.613,
        "tier": "1st_2nd_3rd",
        "rounds": [1, 2, 3],
        "label": "1st + 2nd + 3rd",
    },
    {
        "aav_floor_m": 2.761,
        "tier": "1st_3rd",
        "rounds": [1, 3],
        "label": "1st + 3rd",
    },
    {
        "aav_floor_m": 2.082,
        "tier": "2nd_3rd",
        "rounds": [2, 3],
        "label": "2nd + 3rd",
    },
    {
        "aav_floor_m": 1.488,
        "tier": "2nd",
        "rounds": [2],
        "label": "2nd-round pick",
    },
    {
        "aav_floor_m": 1.190,
        "tier": "3rd",
        "rounds": [3],
        "label": "3rd-round pick",
    },
]


def offer_sheet_compensation_tier(aav_m: float) -> Dict[str, Any]:
    """
    NHL offer-sheet compensation grid by AAV (millions).
    Returns tier id + required draft-pick rounds (original picks).
    """
    aav = float(aav_m or 0.0)
    for row in OFFER_SHEET_COMPENSATION_GRID:
        if aav >= float(row["aav_floor_m"]):
            return {
                "tier": row["tier"],
                "rounds": list(row["rounds"]),
                "label": row["label"],
                "aav_floor_m": float(row["aav_floor_m"]),
            }
    return {
        "tier": "none",
        "rounds": [],
        "label": "No compensation",
        "aav_floor_m": 0.0,
    }


def create_rfa_rights_entry(
    player: Any,
    team: Any,
    season_year: int,
) -> Dict[str, Any]:
    prev = player_cap_hit_millions(player) or LEAGUE_MINIMUM_AAV_M
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    return {
        "player_id": _player_id(player),
        "name": _player_name(player),
        "position": _player_pos(player),
        "age": _player_age(player),
        "overall": round(_player_ovr(player)),
        "original_team_id": tid,
        "rights_team_id": tid,
        "qualifying_offer_required": True,
        "qualified": False,
        "arbitration_eligible": _player_age(player) >= 20,
        "offer_sheet_eligible": rfa_offer_sheet_eligible(player, {"previous_aav_m": prev}),
        "status": "RFA_RIGHTS",
        "expiry_year": int(season_year) + 1,
        "previous_aav_m": round(prev, 3),
        "qualifying_offer_aav_m": qualifying_offer_aav(prev),
        "arbitration_filed": False,
        "team_offer_m": None,
        "player_ask_m": None,
        "award_aav_m": None,
        "award_years": None,
        "player_ref": player,
    }


def add_rfa_rights(team: Any, player: Any, season_year: int, league: Any = None) -> Dict[str, Any]:
    rights_list = _ensure_rfa_rights_list(team)
    pid = _player_id(player)
    for r in rights_list:
        if str(r.get("player_id", "")) == pid:
            if r.get("player_ref") is None:
                r["player_ref"] = player
            return r
    entry = create_rfa_rights_entry(player, team, season_year)
    rights_list.append(entry)
    try:
        team.rfa_rights = rights_list
    except Exception:
        pass
    # Keep the off-roster RFA discoverable for demand / negotiate / qualify.
    try:
        from services.draft_player_registry import register_player

        if league is not None:
            register_player(league, player)
        else:
            # Best-effort: registry lives on the league hanging off team when present.
            lg = _get(team, "league", None)
            if lg is not None:
                register_player(lg, player)
    except Exception:
        pass
    return entry


def serialize_rfa_rights(team: Any) -> List[Dict[str, Any]]:
    out = []
    for r in _ensure_rfa_rights_list(team):
        row = {k: v for k, v in r.items() if k != "player_ref"}
        out.append(row)
    return out


def find_rfa_rights(team: Any, player_id: str) -> Optional[Dict[str, Any]]:
    for r in _ensure_rfa_rights_list(team):
        if str(r.get("player_id", "")) == str(player_id):
            return r
    return None


def resolve_rfa_player(entry: Optional[Dict[str, Any]], league: Any = None) -> Optional[Any]:
    """Resolve the live player object for an RFA-rights entry.

    Rights holders are off-roster and only kept via `player_ref`. After saves/reloads
    or registry rebuilds that miss rights, fall back to the league player registry and
    re-attach the ref so qualify / negotiate / arbitration keep working.
    """
    if not isinstance(entry, dict):
        return None
    player = entry.get("player_ref")
    if player is not None:
        return player
    pid = str(entry.get("player_id") or "")
    if not pid or league is None:
        return None
    try:
        from services.draft_player_registry import get_player, register_player

        player = get_player(league, pid)
        if player is not None:
            entry["player_ref"] = player
            register_player(league, player)
    except Exception:
        player = None
    return player


def remove_rfa_rights(team: Any, player_id: str) -> None:
    rights_list = _ensure_rfa_rights_list(team)
    team.rfa_rights = [r for r in rights_list if str(r.get("player_id", "")) != str(player_id)]


def clear_rfa_rights_for_player(league: Any, player_id: str, *, prefer_team: Any = None) -> None:
    """Drop RFA-rights ghosts after a player is signed (any club)."""
    pid = str(player_id or "")
    if not pid:
        return
    if prefer_team is not None:
        remove_rfa_rights(prefer_team, pid)
    if league is None:
        return
    for t in list(_get(league, "teams", None) or []):
        if prefer_team is not None and t is prefer_team:
            continue
        remove_rfa_rights(t, pid)


# ---------------------------------------------------------------------------
# Clause enforcement
# ---------------------------------------------------------------------------

def has_nmc(player: Any) -> bool:
    c = normalize_contract_payload(player)
    return bool(c.get("nmc") or c.get("no_move_clause") or c.get("no_movement_clause"))


def has_ntc(player: Any) -> bool:
    c = normalize_contract_payload(player)
    return bool(c.get("ntc") or c.get("no_trade_clause"))


def _modified_ntc_team_count(player: Any) -> int:
    c = normalize_contract_payload(player)
    raw = getattr(player, "contract", None)
    clauses = getattr(raw, "clauses", None) if raw is not None else None
    mntc = 0
    if clauses is not None:
        mntc = int(getattr(clauses, "modifiedNoTradeTeams", 0) or 0)
        clause_type = str(getattr(clauses, "clause_type", "") or "").lower()
        if mntc <= 0 and clause_type in ("m-ntc", "mntc"):
            mntc = max(mntc, int(getattr(clauses, "trade_list_size", 10) or 10))
    elif raw is not None:
        mntc = int(getattr(raw, "modified_no_trade_teams", 0) or 0)
    if mntc <= 0:
        mntc = int(c.get("modified_no_trade_teams") or 0)
    return max(0, mntc)


def _approved_trade_team_ids(player: Any) -> List[str]:
    approved = getattr(player, "approved_trade_teams", None)
    if approved:
        return [str(x) for x in approved]
    raw = getattr(player, "contract", None)
    if raw is not None:
        ap = getattr(raw, "approved_trade_teams", None)
        if ap:
            return [str(x) for x in ap]
    c = normalize_contract_payload(player)
    ap = c.get("approved_trade_teams")
    if isinstance(ap, (list, tuple, set)):
        return [str(x) for x in ap]
    return []


def can_waive_or_bury(player: Any) -> Tuple[bool, str]:
    if has_nmc(player):
        return False, "No-move clause blocks waiver/bury"
    return True, "ok"


def can_trade_player(player: Any, to_team_id: str = "") -> Tuple[bool, str]:
    if has_nmc(player):
        return False, "No-move clause blocks trade"
    if has_ntc(player):
        return False, "No-trade clause — consent required"
    mntc = _modified_ntc_team_count(player)
    if mntc > 0 and to_team_id:
        approved = _approved_trade_team_ids(player)
        if approved and str(to_team_id) not in approved:
            return False, "Modified NTC — destination not on approved list"
    return True, "ok"


def can_buyout_player(player: Any) -> Tuple[bool, str]:
    if is_buyout_protected(player, None):
        return False, "Player is buyout protected"
    if has_nmc(player):
        return False, "No-move clause complicates buyout"
    return True, "ok"


# ---------------------------------------------------------------------------
# Buyout / waive / bury estimates
# ---------------------------------------------------------------------------

def estimate_buyout(player: Any) -> Dict[str, Any]:
    aav = player_cap_hit_millions(player)
    yrs = _contract_years_remaining(player)
    if aav <= 0 or yrs <= 0:
        return {"total_cost_m": 0, "years": 0, "annual_penalty_m": 0, "cap_savings_m": 0, "warning": "No contract"}
    age = _player_age(player)
    penalty_years = min(yrs, 2 if age >= 26 else yrs)
    total_penalty = aav * penalty_years * 0.67
    annual = round(total_penalty / max(1, penalty_years), 3)
    return {
        "total_cost_m": round(total_penalty, 3),
        "years": penalty_years,
        "annual_penalty_m": annual,
        "cap_savings_m": round(max(0.0, aav - annual), 3),
        "warning": "NMC may block" if has_nmc(player) else "",
    }


def estimate_bury_savings(player: Any) -> float:
    hit = player_cap_hit_millions(player)
    buried = buried_cap_hit_millions(player)
    return round(max(0.0, hit - buried), 3)


# ---------------------------------------------------------------------------
# Waivers, bury/unbury, cap consequences (Phase 3)
# ---------------------------------------------------------------------------
# Simplified waiver approximation:
# - Exempt: unsigned prospects, young/ELC/low-NHL-GP players
# - Required: established NHL-contract players assigned NHL -> minors
# - Buried cap relief uses cap_engine formula (min + bonus ~= $1.15M relief)

BURY_RELIEF_CAP_M = 1.15


def _player_nhl_games(player: Any) -> int:
    for src in (
        _get(player, "season_stats", None),
        _get(player, "career_stats", None),
        _get(player, "stats", None),
    ):
        if isinstance(src, dict):
            for key in ("gp", "games_played", "nhl_gp"):
                v = src.get(key)
                if v is not None:
                    try:
                        return max(0, int(v))
                    except (TypeError, ValueError):
                        pass
    try:
        return max(0, int(_get(player, "nhl_gp", _get(player, "games_played", 0)) or 0))
    except (TypeError, ValueError):
        return 0


def _has_signed_nhl_contract(player: Any) -> bool:
    return has_active_contract(player)


def _gp_trusted(player: Any) -> bool:
    """True when GP stat is explicitly tracked (non-zero or pro_seasons set)."""
    gp = _player_nhl_games(player)
    if gp > 0:
        return True
    for key in ("pro_seasons", "nhl_seasons", "seasons_pro"):
        v = _get(player, key, None)
        if v is not None:
            try:
                return int(v) >= 0
            except (TypeError, ValueError):
                pass
    return False


def is_waiver_exempt(player: Any, team: Any = None, league: Any = None) -> bool:
    """
    Simplified NHL waiver approximation.
    two_way flag is cosmetic and does not grant exemption.
    Missing bootstrap GP is not treated as zero pro experience.
    """
    if bool(_get(player, "waiver_exempt", False)):
        return True
    if str(_get(player, "signed_status", "") or "").lower() == "unsigned":
        return True
    if _get(player, "entry_level_contract_eligible", False) and not has_active_contract(player):
        return True
    if not has_active_contract(player):
        return True

    age = _player_age(player)
    gp = _player_nhl_games(player)

    if has_true_elc_contract(player):
        if age <= 21 and (not _gp_trusted(player) or gp < 60):
            return True
        if age <= 23 and _gp_trusted(player) and gp > 0 and gp < 25:
            return True
        return False

    if age <= 20:
        return True

    ctype = str(_contract_type(player) or "").upper()
    if ctype in ("RFA_BRIDGE",):
        if age <= 22 and (not _gp_trusted(player) or gp < 25):
            return True
        return False

    # Signed standard NHL contract
    if age >= 23:
        return False
    if age >= 21:
        if _gp_trusted(player) and gp > 0 and gp < 40:
            return True
        return False
    return True


def get_waiver_status(player: Any, team: Any = None, league: Any = None) -> str:
    if is_waiver_exempt(player, team, league):
        return "exempt"
    ws = str(_get(player, "waiver_status", "") or "").lower()
    if ws in ("on_waivers", "waiver_wire", "pending"):
        return "on_waivers"
    if ws == "cleared":
        return "cleared"
    if _get(player, "is_buried", False) or _get(player, "buried", False):
        return "buried"
    return "required"


def is_waiver_required_for_assignment(
    player: Any,
    from_roster: str = "nhl",
    to_roster: str = "minors",
    league: Any = None,
) -> bool:
    if str(from_roster).lower() != "nhl" or str(to_roster).lower() not in ("minors", "ahl", "buried"):
        return False
    if is_waiver_exempt(player, None, league):
        return False
    return get_waiver_status(player, None, league) in ("required", "on_waivers")


def calculate_buried_cap_hit(player: Any, league: Any = None) -> float:
    """Cap hit still charged while buried (after partial relief)."""
    _ = league
    if not (_get(player, "is_buried", False) or _get(player, "buried", False) or _get(player, "in_minors", False)):
        return player_cap_hit_millions(player)
    return buried_cap_hit_millions(player)


def get_team_buried_cap_total(team: Any, league: Any = None) -> float:
    from app.sim_engine.economy.cap_engine import team_buried_cap_hit_millions
    _ = league
    return round(float(team_buried_cap_hit_millions(team)), 3)


def get_team_active_cap_total(team: Any, league: Any = None) -> float:
    from app.sim_engine.economy.cap_engine import team_active_roster_cap_hit_millions
    _ = league
    return round(float(team_active_roster_cap_hit_millions(team)), 3)


def get_team_bury_relief_total(team: Any) -> float:
    relief = 0.0
    for p in _all_rostered(team):
        if not (_get(p, "is_buried", False) or _get(p, "buried", False)):
            continue
        hit = player_cap_hit_millions(p)
        relief += max(0.0, hit - buried_cap_hit_millions(p))
    return round(relief, 3)


def bury_player_contract(team: Any, player: Any, league: Any = None, *, skip_waiver_check: bool = False) -> Dict[str, Any]:
    if has_nmc(player):
        return {"ok": False, "reason": "No-move clause blocks bury"}
    if not skip_waiver_check and is_waiver_required_for_assignment(player, "nhl", "minors", league):
        return {"ok": False, "reason": "waiver_required", "requires_waivers": True}

    try:
        player.is_buried = True
        player.buried = True
        player.in_minors = True
        player.waiver_status = "buried"
        player.roster_location = "minors"
    except Exception:
        pass
    sync_team_cap_fields(team, league)
    return {
        "ok": True,
        "player_id": _player_id(player),
        "bury_savings_m": estimate_bury_savings(player),
        "buried_cap_hit_m": calculate_buried_cap_hit(player, league),
    }


def unbury_player_contract(team: Any, player: Any, league: Any = None) -> Dict[str, Any]:
    snap = get_team_cap_snapshot_full(team, league)
    added = player_cap_hit_millions(player) - buried_cap_hit_millions(player)
    if snap["usable_cap_space_m"] - max(0.0, added) < -0.01:
        return {"ok": False, "reason": "Insufficient cap to recall"}
    if int(snap.get("active_roster_count", 0)) >= int(snap.get("active_roster_max", 23)):
        return {"ok": False, "reason": "Active roster full"}

    try:
        player.is_buried = False
        player.buried = False
        player.in_minors = False
        player.waiver_status = None
        player.roster_location = "nhl"
    except Exception:
        pass
    sync_team_cap_fields(team, league)
    return {"ok": True, "player_id": _player_id(player)}


def _ensure_waiver_wire(league: Any) -> List[Dict[str, Any]]:
    wire = _get(league, "waiver_wire", None)
    if not isinstance(wire, list):
        try:
            league.waiver_wire = []
        except Exception:
            pass
        return getattr(league, "waiver_wire", [])
    return wire


def _append_waiver_history(league: Any, entry: Dict[str, Any]) -> None:
    history = _get(league, "waiver_history", None)
    if not isinstance(history, list):
        try:
            league.waiver_history = []
        except Exception:
            return
        history = getattr(league, "waiver_history", [])
    history.append(dict(entry))


def is_core_player_protected(player: Any, team: Any, league: Any = None) -> bool:
    _ = league
    if has_nmc(player):
        return True
    ovr = _player_ovr(player)
    age = _player_age(player)
    if ovr >= 86.0:
        return True
    if has_true_elc_contract(player) and age <= 22:
        return True
    pot = _player_potential(player)
    if age <= 23 and pot >= 78:
        return True
    if team is not None:
        pos = _position_bucket(player)
        if pos == "G" and ovr >= 82:
            peers = [_player_ovr(p) for p in _active_roster(team) if _position_bucket(p) == "G"]
            if peers and ovr >= max(peers) - 1.0:
                return True
    return False


def is_waiver_protected_from_compliance(player: Any, team: Any, league: Any = None) -> bool:
    return is_core_player_protected(player, team, league)


def is_buyout_protected(player: Any, team: Any, league: Any = None) -> bool:
    if has_true_elc_contract(player):
        return True
    if is_core_player_protected(player, team, league):
        return True
    age = _player_age(player)
    if has_true_elc_contract(player) and age <= 24:
        return True
    return False


def is_compliance_protected(player: Any, team: Any) -> bool:
    return is_waiver_protected_from_compliance(player, team)


def expose_player_to_waivers(
    team: Any,
    player: Any,
    league: Any,
    *,
    reason: str = "roster",
    season_year: Optional[int] = None,
) -> Dict[str, Any]:
    ok, block_reason = can_waive_or_bury(player)
    if not ok:
        return {"ok": False, "reason": block_reason}
    if is_compliance_protected(player, team):
        return {"ok": False, "reason": "compliance_protected"}
    if is_waiver_exempt(player, team, league):
        return {"ok": False, "reason": "waiver_exempt_use_bury"}

    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    wire = _ensure_waiver_wire(league)
    pid = _player_id(player)
    for entry in wire:
        if str(entry.get("player_id", "")) == pid and not entry.get("cleared"):
            return {"ok": False, "reason": "already_on_waivers"}

    ovr = _player_ovr(player)
    pos = _position_bucket(player)
    entry = {
        "player_id": pid,
        "name": _player_name(player),
        "position": pos,
        "overall": round(ovr),
        "ovr_band": ovr_band(ovr),
        "cap_hit_m": round(player_cap_hit_millions(player), 3),
        "years_remaining": _contract_years_remaining(player),
        "original_team_id": tid,
        "season_year": int(season_year or 0),
        "reason": reason,
        "waiver_status": "on_waivers",
        "claimed_by": None,
        "cleared": False,
        "player_ref": player,
    }
    wire.append(entry)
    _append_waiver_history(league, entry)

    roster = list(_get(team, "roster", None) or [])
    if player in roster:
        roster.remove(player)
        team.roster = roster

    try:
        player.waiver_status = "on_waivers"
        player.on_waiver_wire = True
        player.active_roster = False
    except Exception:
        pass
    sync_team_cap_fields(team, league)
    return {"ok": True, "waiver_entry": entry}


def make_roster_room_for_promotion(
    team: Any,
    incoming_player: Any,
    league: Any,
    season_year: int,
) -> Dict[str, Any]:
    """Make one active roster slot before NHL promotion (bury exempt, else waive depth)."""
    _ = incoming_player
    if len(_active_roster(team)) < 23:
        return {"ok": True, "action": "none"}

    candidates = sorted(
        [p for p in _active_roster(team) if not is_compliance_protected(p, team)],
        key=lambda p: (_player_ovr(p), -player_cap_hit_millions(p)),
    )
    for p in candidates:
        if is_waiver_exempt(p, team, league):
            res = bury_player_contract(team, p, league, skip_waiver_check=True)
            if res.get("ok"):
                return {"ok": True, "action": "buried", "player_id": _player_id(p)}
    for p in candidates:
        if is_waiver_exempt(p, team, league):
            continue
        if _player_ovr(p) >= 80:
            continue
        res = expose_player_to_waivers(team, p, league, reason="promotion_room", season_year=season_year)
        if res.get("ok"):
            return {"ok": True, "action": "waived", "player_id": _player_id(p)}
    return {"ok": False, "reason": "no_movable_player"}


def score_waiver_claim_fit(
    team: Any,
    player: Any,
    needs_ctx: Dict[str, Any],
    league: Any,
) -> float:
    pos = _position_bucket(player)
    ovr = _player_ovr(player)
    need = float(needs_ctx["need_score"].get(pos, 0))
    cap = float(needs_ctx.get("cap_space_m", 0))
    hit = player_cap_hit_millions(player)

    if cap < hit:
        return 0.0
    if needs_ctx["overload"].get(pos) and ovr < needs_ctx["best_ovr"].get(pos, 0) + 3:
        return 0.0
    if ovr < 68 and need < 0.45:
        return 0.0

    score = 0.10 + (ovr / 99.0) * 0.35 + need * 0.40
    if pos == "G" and needs_ctx["counts"].get("G", 0) <= 1:
        score += 0.20
    if ovr >= needs_ctx["best_ovr"].get(pos, 0) + 2:
        score += 0.10
    return max(0.0, min(1.2, score))


def _waiver_claim_priority(team: Any) -> float:
    for key in ("points", "pts", "wins", "w"):
        v = _get(team, key, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "0")
    try:
        return float(int(tid))
    except ValueError:
        return 999.0


def _transfer_waiver_player(player: Any, from_team: Any, to_team: Any, league: Any) -> None:
    if from_team is not None:
        roster = list(_get(from_team, "roster", None) or [])
        if player in roster:
            roster.remove(player)
            from_team.roster = roster
        sync_team_cap_fields(from_team, league)

    roster_to = list(_get(to_team, "roster", None) or [])
    if player not in roster_to:
        roster_to.append(player)
        to_team.roster = roster_to

    tid = str(_get(to_team, "team_id", "") or _get(to_team, "id", "") or "")
    try:
        player.team_id = tid
        player.current_team_id = tid
        player.waiver_status = None
        player.on_waiver_wire = False
        player.is_buried = False
        player.buried = False
        player.in_minors = False
        player.roster_location = "nhl"
    except Exception:
        pass
    sync_team_cap_fields(to_team, league)


def run_waiver_claim_pass(
    league: Any,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
    user_team_id: str = "",
) -> Dict[str, Any]:
    wire = _ensure_waiver_wire(league)
    pending = [e for e in wire if not e.get("cleared") and not e.get("claimed_by")]
    claims: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    teams = list(_get(league, "teams", None) or [])
    claim_order = sorted(
        [t for t in teams if str(_get(t, "team_id", "") or _get(t, "id", "")) != str(user_team_id)],
        key=_waiver_claim_priority,
    )

    for entry in list(pending):
        player = entry.get("player_ref")
        if player is None:
            continue
        orig_tid = str(entry.get("original_team_id", ""))
        orig_team = next(
            (t for t in teams if str(_get(t, "team_id", "") or _get(t, "id", "")) == orig_tid),
            None,
        )

        best_score = 0.0
        best_team = None
        for team in claim_order:
            tid = str(_get(team, "team_id", "") or _get(team, "id", ""))
            if tid == orig_tid:
                continue
            ctx = evaluate_team_position_needs(team, league, sim, season_year=season_year)
            if ctx["slots_remaining"] <= 0:
                continue
            if ctx["cap_space_m"] < float(entry.get("cap_hit_m", 0)):
                continue
            fit = score_waiver_claim_fit(team, player, ctx, league)
            if fit >= 0.42 and fit > best_score:
                best_score = fit
                best_team = team

        if best_team is None:
            failures.append({"player_id": entry.get("player_id"), "reason": "no_claimer"})
            continue

        claim_tid = str(_get(best_team, "team_id", "") or _get(best_team, "id", ""))
        _transfer_waiver_player(player, orig_team, best_team, league)
        entry["claimed_by"] = claim_tid
        entry["waiver_status"] = "claimed"
        claims.append({
            "player_id": entry.get("player_id"),
            "from_team_id": orig_tid,
            "to_team_id": claim_tid,
            "position": entry.get("position"),
            "overall": entry.get("overall"),
            "ovr_band": entry.get("ovr_band"),
            "fit_score": round(best_score, 3),
        })

    return {"claims": claims, "failures": failures, "pending_after": len([e for e in wire if not e.get("cleared") and not e.get("claimed_by")])}


def resolve_cleared_waivers(league: Any, sim: Any = None, *, season_year: Optional[int] = None) -> Dict[str, Any]:
    wire = _ensure_waiver_wire(league)
    cleared: List[Dict[str, Any]] = []
    buried_after: List[Dict[str, Any]] = []
    teams = {str(_get(t, "team_id", "") or _get(t, "id", "")): t for t in (_get(league, "teams", None) or [])}

    for entry in wire:
        if entry.get("claimed_by") or entry.get("cleared"):
            continue
        player = entry.get("player_ref")
        if player is None:
            continue
        tid = str(entry.get("original_team_id", ""))
        team = teams.get(tid)
        entry["cleared"] = True
        entry["waiver_status"] = "cleared"
        try:
            player.waiver_status = "cleared"
            player.on_waiver_wire = False
        except Exception:
            pass
        cleared.append({
            "player_id": entry.get("player_id"),
            "team_id": tid,
            "position": entry.get("position"),
            "overall": entry.get("overall"),
            "ovr_band": entry.get("ovr_band"),
        })
        if team is not None:
            roster = list(_get(team, "roster", None) or [])
            if player not in roster:
                roster.append(player)
                team.roster = roster
        if team is not None and player in (_get(team, "roster", None) or []):
            res = bury_player_contract(team, player, league, skip_waiver_check=True)
            if res.get("ok"):
                buried_after.append({"player_id": entry.get("player_id"), "team_id": tid})

    return {"cleared": cleared, "buried_after_clear": buried_after}


def identify_buyout_candidates(
    team: Any,
    league: Any,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in _all_rostered(team):
        if is_compliance_protected(p, team):
            continue
        if is_buyout_protected(p, team, league):
            continue
        if has_true_elc_contract(p):
            continue
        if _player_age(p) <= 24:
            continue
        aav = player_cap_hit_millions(p)
        if aav <= 1.25:
            continue
        ovr = _player_ovr(p)
        fair = compute_fair_aav(p, team, league)
        bad = compute_bad_contract_score(p, team)
        age = _player_age(p)
        if ovr >= 84 and bad < 0.18:
            continue
        fit_penalty = max(0.0, (aav - fair) / max(0.1, fair)) + bad
        if age >= 30:
            fit_penalty += 0.15
        if fit_penalty < 0.12:
            continue
        ok, _ = can_buyout_player(p)
        if not ok:
            continue
        est = estimate_buyout(p)
        out.append({
            "player_id": _player_id(p),
            "name": _player_name(p),
            "overall": round(ovr),
            "ovr_band": ovr_band(ovr),
            "aav_m": round(aav, 3),
            "fair_aav_m": round(fair, 3),
            "bad_score": round(bad, 3),
            "buyout_score": round(fit_penalty, 3),
            "cap_savings_m": est.get("cap_savings_m", 0),
            "player_ref": p,
        })
    out.sort(key=lambda r: -float(r.get("buyout_score", 0)))
    return out[:5]


def execute_cpu_buyout(team: Any, player: Any, league: Any, season_year: int) -> Dict[str, Any]:
    return execute_buyout(team, player, league, season_year)


def run_cpu_buyout_pass(session: Any, *, max_buyouts: int = 8) -> Dict[str, Any]:
    league = getattr(session.sim, "league", None)
    sim = session.sim
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    executed: List[Dict[str, Any]] = []

    for team in _get(league, "teams", None) or []:
        if len(executed) >= max_buyouts:
            break
        tid = str(_get(team, "team_id", "") or _get(team, "id", ""))
        if tid == user_tid:
            continue
        snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
        if snap["usable_cap_space_m"] >= -0.01:
            continue
        for cand in identify_buyout_candidates(team, league, sim, season_year=season_year):
            if len(executed) >= max_buyouts:
                break
            player = cand.get("player_ref")
            if player is None:
                continue
            res = execute_cpu_buyout(team, player, league, season_year)
            if res.get("ok"):
                executed.append({
                    "team_id": tid,
                    "player_id": cand.get("player_id"),
                    "overall": cand.get("overall"),
                    "ovr_band": cand.get("ovr_band"),
                    "aav_m": cand.get("aav_m"),
                    "cap_savings_m": res.get("saved_aav_m"),
                    "buyout": res.get("buyout"),
                })
                snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
                if snap["usable_cap_space_m"] >= -0.01:
                    break
    return {"buyouts": executed, "count": len(executed)}


def run_cap_compliance_pipeline(session: Any, *, include_buyouts: bool = True) -> Dict[str, Any]:
    """
    Cap order: exempt down -> bury -> waive -> claim pass -> clear/bury -> buyout
    -> recalc cap -> cap casualty trades -> final sync.
    """
    league = getattr(session.sim, "league", None)
    sim = session.sim
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    user_tid = str(getattr(session, "user_team_id", "") or "")

    report: Dict[str, Any] = {
        "exempt_sent_down": [],
        "buried": [],
        "waived": [],
        "claims": [],
        "cleared": [],
        "buyouts": [],
        "cap_casualty_trades": [],
    }

    teams = list(_get(league, "teams", None) or [])
    for team in teams:
        tid = str(_get(team, "team_id", "") or _get(team, "id", ""))

        def _needs_relief() -> bool:
            snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
            return snap["usable_cap_space_m"] < -0.01 or len(_active_roster(team)) > 23

        def _cap_over_m() -> float:
            snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
            return max(0.0, -float(snap["usable_cap_space_m"]))

        def _waive_veterans(*, max_rounds: int, cap_pressure: bool) -> None:
            for _ in range(max_rounds):
                if not _needs_relief():
                    break
                candidates = sorted(
                    [p for p in _active_roster(team) if not is_compliance_protected(p, team)],
                    key=lambda p: (-player_cap_hit_millions(p), _player_ovr(p)),
                )
                moved = False
                over = _cap_over_m()
                ovr_limit = 84.0 if cap_pressure and over >= 0.5 else 82.0
                for p in candidates:
                    if is_waiver_exempt(p, team, league):
                        continue
                    if _player_ovr(p) >= ovr_limit:
                        continue
                    res = expose_player_to_waivers(
                        team, p, league, reason="cap_compliance", season_year=season_year,
                    )
                    if res.get("ok"):
                        report["waived"].append({"team_id": tid, "player_id": _player_id(p), **res})
                        moved = True
                        break
                if not moved:
                    break

        # 0) Roster limit trim (NHL 23-man cap)
        for _ in range(10):
            if len(_active_roster(team)) <= 23:
                break
            candidates = sorted(
                [p for p in _active_roster(team) if not is_compliance_protected(p, team)],
                key=lambda p: (_player_ovr(p), -player_cap_hit_millions(p)),
            )
            moved = False
            for p in candidates:
                if is_waiver_exempt(p, team, league):
                    res = bury_player_contract(team, p, league, skip_waiver_check=True)
                    if res.get("ok"):
                        report["exempt_sent_down"].append({"team_id": tid, "player_id": _player_id(p), **res})
                        moved = True
                        break
                if _player_ovr(p) >= 80:
                    continue
                res = expose_player_to_waivers(team, p, league, reason="roster_trim", season_year=season_year)
                if res.get("ok"):
                    report["waived"].append({"team_id": tid, "player_id": _player_id(p), **res})
                    moved = True
                    break
            if not moved:
                break

        # 1) Exempt depth down without waivers
        for _ in range(8):
            if not _needs_relief():
                break
            candidates = sorted(
                [p for p in _active_roster(team) if not is_compliance_protected(p, team)],
                key=lambda p: (_player_ovr(p), -player_cap_hit_millions(p)),
            )
            moved = False
            for p in candidates:
                if not is_waiver_exempt(p, team, league):
                    continue
                res = bury_player_contract(team, p, league, skip_waiver_check=True)
                if res.get("ok"):
                    report["exempt_sent_down"].append({"team_id": tid, "player_id": _player_id(p), **res})
                    moved = True
                    break
            if not moved:
                break

        cap_pressure = _cap_over_m() >= 0.01
        if cap_pressure:
            _waive_veterans(max_rounds=4, cap_pressure=True)

        # 2) Bury replacement-level players (cleared or exempt-only already handled)
        for _ in range(6):
            if not _needs_relief():
                break
            candidates = sorted(
                [p for p in _active_roster(team) if not is_compliance_protected(p, team)],
                key=lambda p: (_player_ovr(p), -player_cap_hit_millions(p)),
            )
            moved = False
            for p in candidates:
                if is_waiver_required_for_assignment(p, "nhl", "minors", league):
                    continue
                if player_cap_hit_millions(p) <= LEAGUE_MINIMUM_AAV_M + 0.05:
                    continue
                res = bury_player_contract(team, p, league, skip_waiver_check=True)
                if res.get("ok"):
                    report["buried"].append({"team_id": tid, "player_id": _player_id(p), **res})
                    moved = True
                    break
            if not moved:
                break

        # 3) Waive veterans if still over cap / roster
        _waive_veterans(max_rounds=4, cap_pressure=cap_pressure)

    claim_pass = run_waiver_claim_pass(league, sim, season_year=season_year, user_team_id=user_tid)
    report["claims"] = claim_pass.get("claims") or []

    clear_pass = resolve_cleared_waivers(league, sim, season_year=season_year)
    report["cleared"] = clear_pass.get("cleared") or []
    report["buried"].extend(clear_pass.get("buried_after_clear") or [])

    if include_buyouts:
        buyout_pass = run_cpu_buyout_pass(session)
        report["buyouts"] = buyout_pass.get("buyouts") or []
        still_over = any(
            get_team_cap_snapshot_full(t, league, sim, season_year=season_year)["usable_cap_space_m"] < -0.01
            for t in teams
        )
        if still_over:
            extra = run_cpu_buyout_pass(session, max_buyouts=12)
            report["buyouts"].extend(extra.get("buyouts") or [])

    for team in teams:
        sync_team_cap_fields(team, league, sim, season_year=season_year)

    cap_pass = run_cpu_cap_casualty_trade_pass(session)
    report["cap_casualty_trades"] = cap_pass.get("cap_casualty_trades") or []

    for team in teams:
        sync_team_cap_fields(team, league, sim, season_year=season_year)

    return report


# ---------------------------------------------------------------------------
# Offer evaluation
# ---------------------------------------------------------------------------

def _logistic(x: float) -> float:
    """Numerically safe logistic squashing to (0,1)."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _demand_context_multiplier(context: str) -> float:
    c = str(context or "").lower()
    if c == "ufa":
        return 1.05
    if c == "rfa":
        return 0.98
    return 1.0  # re_sign / offer — your own player, no open-market premium


def compute_player_demand(
    player: Any,
    team: Any,
    league: Any = None,
    *,
    context: str = "ufa",
    days_on_market: int = 0,
    offer_count: int = 0,
) -> Dict[str, Any]:
    """Value 2 of the five-value model: player demand = league market value
    adjusted ONCE by team leverage, market context, and hidden personality.

    Team importance / peer-gap are applied here and nowhere else (item 2), so a
    star is not charged a premium on top of an already-inflated baseline.
    ``days_on_market`` / ``offer_count`` apply bounded cold-market flexibility so
    unsigned players do not hold identical July asks into September.
    """
    market = compute_market_value(player, league)
    importance = compute_team_importance_score(player, team) if team is not None else 0.5
    gap = compute_peer_gap_score(player, team) if team is not None else 0.0
    prof = _player_negotiation_profile(player)

    leverage = 1.0 + 0.10 * importance + 0.035 * min(gap, 10.0)
    want = market * leverage * _demand_context_multiplier(context)
    # Bet-on-self players push AAV up; security-minded shade it down for term.
    want *= 1.0 + 0.05 * (prof["gamble_pref"] - prof["security_pref"] * 0.6)
    want = max(LEAGUE_MINIMUM_AAV_M, round(want, 3))

    ovr = _player_ovr(player)
    age = _player_age(player)
    pot = _player_potential(player)
    term_rng = random.Random(abs(hash(("term", _player_id(player)))) & 0xFFFFFFFF)
    base_years = _term_for_profile(ovr, age, pot, importance, term_rng)
    yr_shift = int(round(prof["security_pref"] * 1.5 - prof["gamble_pref"] * 1.5))
    want_years = max(1, min(8, base_years + yr_shift))

    # Minimum he will actually sign for — loyal / patient players flex lower.
    # Solid NHL depth (not stars) accept more below-market AAV so prove-it /
    # short-money deals can close instead of endless counters.
    floor_ratio = 0.90 - 0.06 * prof["loyalty"]
    if ovr < 76:
        floor_ratio = min(floor_ratio, 0.78 - 0.06 * prof["loyalty"])
    elif ovr < 84:
        floor_ratio = min(floor_ratio, 0.84 - 0.07 * prof["loyalty"])
    min_acceptable = max(LEAGUE_MINIMUM_AAV_M, round(want * floor_ratio, 3))

    days = max(0, int(days_on_market or 0))
    offers = max(0, int(offer_count or 0))
    ctx_lower = str(context or "").lower()
    # Cold-market decay for UFAs; RFAs soften slower (no open bidding war).
    decay_start = 5 if ovr < 84 else 8
    is_ufa_ctx = ctx_lower in ("ufa", "free_agency", "")
    is_rfa_ctx = ctx_lower == "rfa"
    if days >= decay_start and is_ufa_ctx:
        # Cold-market floor drop: fringe → near min; elite floors stay elevated.
        if ovr < 70:
            decay = 0.88 if offers == 0 else 0.94
            floor_m = LEAGUE_MINIMUM_AAV_M
        elif ovr < 76:
            decay = 0.90 if offers == 0 else 0.95
            floor_m = LEAGUE_MINIMUM_AAV_M
        elif ovr < 82:
            decay = 0.92 if offers == 0 else 0.955
            floor_m = max(LEAGUE_MINIMUM_AAV_M, market * 0.58)
        elif ovr < 88:
            decay = 0.95 if offers == 0 else 0.97
            floor_m = max(LEAGUE_MINIMUM_AAV_M, market * 0.72)
        else:
            decay = 0.97 if offers == 0 else 0.985
            floor_m = max(LEAGUE_MINIMUM_AAV_M, market * 0.80)
        want = max(floor_m, round(want * decay, 3))
        min_acceptable = max(floor_m, round(min_acceptable * decay, 3))
        if days >= 20 and ovr < 82:
            want_years = min(want_years, 2 if ovr < 76 else 3)
        if days >= 25 and ovr >= 86 and offers <= 1:
            want_years = min(want_years, 4)
            want = max(floor_m, round(want * 0.97, 3))
            min_acceptable = max(floor_m, round(min_acceptable * 0.96, 3))

    # RFAs exempt from full UFA decay — compensation rights limit bidding pressure.
    # Apply 0.5× UFA decay rate starting day 10 unsigned.
    if is_rfa_ctx and days >= 10:
        rfa_decay = 0.985 if offers == 0 else 0.992
        if ovr < 76:
            rfa_decay = 0.97 if offers == 0 else 0.98
        elif ovr < 82:
            rfa_decay = 0.978 if offers == 0 else 0.985
        # Half-strength vs UFA path
        blend = 1.0 - (1.0 - rfa_decay) * 0.5
        floor_m = max(LEAGUE_MINIMUM_AAV_M, market * (0.65 if ovr < 76 else 0.72))
        want = max(floor_m, round(want * blend, 3))
        min_acceptable = max(floor_m, round(min_acceptable * blend, 3))

    return {
        "market_value_m": market,
        "want_aav_m": want,
        "min_acceptable_aav_m": min_acceptable,
        "want_years": want_years,
        "importance": round(importance, 3),
        "peer_gap": round(gap, 2),
        "profile": prof,
        "days_on_market": days,
        "offer_count": offers,
    }


def evaluate_contract_offer(
    player: Any,
    team: Any,
    offer: Dict[str, Any],
    league: Any = None,
    *,
    context: str = "ufa",
) -> Dict[str, Any]:
    """Value 4 of the five-value model: does the player LIKE this offer?

    Interest is continuous and weights salary, term, age fit, relationship/stay
    interest, NTC/NMC clauses, and signing bonus. Cap space is NOT part of interest.
    """
    aav_m = normalize_money_m(offer.get("aav_m") or offer.get("aav") or 0)
    years = max(1, int(offer.get("years") or offer.get("term") or 1))
    bonus_m = normalize_money_m(offer.get("signing_bonus_m") or offer.get("signing_bonus") or 0)
    cap_hit_m = compute_prorated_cap_hit_m(aav_m, years, bonus_m)
    ntc_mode = _offer_ntc_mode(offer)
    nmc = bool(offer.get("nmc"))
    ntc = ntc_mode in ("FULL", "MODIFIED") and not nmc
    age = _player_age(player)
    ovr = _player_ovr(player)

    demand = compute_player_demand(
        player,
        team,
        league,
        context=context,
        days_on_market=int(offer.get("days_on_market") or getattr(player, "days_on_market", 0) or 0),
        offer_count=int(offer.get("offer_count") or getattr(player, "fa_offer_count", 0) or 0),
    )
    want_aav = demand["want_aav_m"]
    want_years = demand["want_years"]
    min_acceptable = demand["min_acceptable_aav_m"]
    prof = demand["profile"]
    security = float(prof["security_pref"])
    gamble = float(prof["gamble_pref"])
    loyalty = float(prof["loyalty"])
    importance = float(demand.get("importance") or 0.5)

    try:
        morale = float(getattr(player, "morale", None) or getattr(player, "happiness", 70) or 70)
    except Exception:
        morale = 70.0

    # Preferred protection: stars / security-minded want real clauses.
    # Late-market unsigned stars soften clause demands rather than stay unsigned forever.
    days_unsigned = int(offer.get("days_on_market") or getattr(player, "days_on_market", 0) or 0)
    if days_unsigned >= 25 and ovr >= 88:
        preferred_clause = "NTC" if years >= 5 and security >= 0.7 else "None"
    elif years >= 5 and ovr >= 88 and security >= 0.45:
        preferred_clause = "NMC"
    elif years >= 4 and ovr >= 84 and security >= 0.55:
        preferred_clause = "NTC"
    elif years >= 5 and ovr >= 82 and security >= 0.75:
        preferred_clause = "NTC"
    elif years >= 4 and 80 <= ovr < 88 and 0.40 <= security < 0.65:
        preferred_clause = "M-NTC"
    else:
        preferred_clause = "None"

    # --- Continuous interest (0..100) ---
    r = aav_m / max(0.25, want_aav)
    salary_component = 80.0 * (_logistic((r - 0.90) * 9.0) - 0.5)
    term_gap = years - want_years
    term_component = 10.0 * math.tanh(term_gap * (0.5 + 0.5 * security))
    fit_component = 0.0
    if age >= 32:
        fit_component -= max(0, years - 3) * (2.5 + 3.0 * (1.0 - security))
    if age <= 24 and years >= 4:
        fit_component += 6.0
    variance_component = prof["variance"] * 100.0

    # Stay-interest / relationship — High/Med/Low on the board maps here.
    relationship_component = (
        (loyalty - 0.5) * 14.0
        + (morale - 55.0) * 0.18
        + (importance - 0.5) * 10.0
    )
    try:
        session = offer.get("_franchise_session") or offer.get("_session")
        pid = str(getattr(player, "id", "") or getattr(player, "player_id", "") or "")
        tid = str(getattr(team, "team_id", "") or getattr(team, "id", "") or "")
        if session is not None and pid:
            from app.sim_engine.franchise.storyline_engine import (  # noqa: WPS433
                _u_sync_player_entities,
                build_human_dossier_payload,
            )

            entities = _u_sync_player_entities(session)
            entity = entities.get(pid) or {}
            life = entity.get("life") or {}
            city_attachment = float(life.get("city_attachment") or 40)
            home_owned = bool(life.get("home_owned"))
            relationship_component += (city_attachment - 50) * 0.08
            if home_owned:
                relationship_component += 4.0
            if float((entity.get("state") or {}).get("gm_trust", 65)) >= 68:
                relationship_component += 3.0
    except Exception:
        pass
    stay_interest = max(
        0.0,
        min(
            100.0,
            50.0 + (loyalty - 0.5) * 40.0 + (morale - 50.0) * 0.45 + importance * 20.0,
        ),
    )
    if stay_interest >= 70:
        stay_label = "High"
    elif stay_interest < 45:
        stay_label = "Low"
    else:
        stay_label = "Medium"

    # NTC / NMC — heavy weight when the player cares about protection.
    clause_component = 0.0
    clause_note = "Clauses not a priority"
    if preferred_clause == "NMC":
        if nmc:
            clause_component += 16.0 + 8.0 * security
            clause_note = "NMC matches demand"
        elif ntc:
            clause_component += 5.0
            clause_note = "NTC helps, still wants NMC"
        else:
            clause_component -= 12.0 * (0.55 + security)
            clause_note = "Missing expected NMC"
    elif preferred_clause == "NTC":
        if nmc:
            clause_component += 14.0 + 4.0 * security
            clause_note = "NMC exceeds ask"
        elif ntc_mode == "FULL":
            clause_component += 11.0 + 5.0 * security
            clause_note = "NTC matches demand"
        elif ntc_mode == "MODIFIED":
            clause_component += 7.0 + 4.0 * security
            clause_note = "M-NTC partial match"
        else:
            clause_component -= 8.0 * (0.45 + security)
            clause_note = "Missing expected NTC"
    elif preferred_clause == "M-NTC":
        if nmc:
            clause_component += 14.0 + 4.0 * security
            clause_note = "NMC exceeds ask"
        elif ntc_mode == "FULL":
            clause_component += 12.0 + 4.0 * security
            clause_note = "Full NTC exceeds M-NTC ask"
        elif ntc_mode == "MODIFIED":
            clause_component += 8.0 + 6.0 * security
            clause_note = "M-NTC matches demand"
        else:
            clause_component -= 6.0 * (0.40 + security)
            clause_note = "Missing expected M-NTC"
    else:
        if nmc:
            clause_component += 4.0 + 2.0 * security
            clause_note = "NMC is a sweetener"
        elif ntc_mode == "FULL":
            clause_component += 2.5
            clause_note = "NTC is a sweetener"
        elif ntc_mode == "MODIFIED":
            clause_component += 3.5 + 1.5 * security
            clause_note = "M-NTC is a sweetener"

    # Signing bonus — cash-upfront lever; high revenue clubs can buy down AAV.
    bonus_component = 0.0
    total_value = max(aav_m * years, 0.25)
    bonus_share = bonus_m / total_value if bonus_m > 0 else 0.0
    if bonus_m > 0:
        bonus_component = min(28.0, 72.0 * bonus_share) * (0.50 + 0.50 * gamble)
        if r < 1.0:
            # Massive bonuses let players accept a lower AAV.
            gap_bridge = min(18.0, (bonus_m / max(want_aav, 0.5)) * 9.0) * max(0.0, 1.08 - r)
            bonus_component += gap_bridge
        if bonus_share >= 0.12:
            bonus_component += 4.0 + 3.0 * gamble
    elif gamble >= 0.6 and want_aav >= 4.0:
        bonus_component -= 3.5 * gamble

    interest = (
        48.0
        + salary_component
        + term_component
        + fit_component
        + relationship_component
        + clause_component
        + bonus_component
        + variance_component
    )
    interest = max(0.0, min(100.0, interest))

    # Bonus / loyalty flexes the cash floor so AAV can sit under market want.
    effective_min = min_acceptable
    if bonus_share >= 0.08:
        soft = 0.90 - min(0.10, bonus_share * 0.40) - (0.02 * gamble)
        effective_min = max(
            LEAGUE_MINIMUM_AAV_M,
            round(min_acceptable * soft, 3),
        )
    elif stay_interest >= 70 and (ntc or nmc or bonus_m > 0):
        effective_min = max(
            LEAGUE_MINIMUM_AAV_M,
            round(min_acceptable * (0.94 - 0.03 * loyalty), 3),
        )

    meets_floor = aav_m >= effective_min
    # Solid NHL players take slightly-under-ask cash if interest is otherwise good.
    if ovr < 84 and r >= 0.85:
        interest = min(100.0, interest + 14.0)
    if (not meets_floor) and ovr < 84 and r >= 0.85 and interest >= 52.0:
        meets_floor = True
    if ovr < 76:
        accept_cut = 52.0
    elif ovr < 83:
        accept_cut = 54.0
    elif ovr < 87:
        accept_cut = 57.0
    else:
        accept_cut = 60.0
    accepted = interest >= accept_cut and meets_floor
    instant_accept = accepted and interest >= (80.0 if ovr < 84 else 88.0)

    if accepted and instant_accept:
        reason = "Accepted immediately"
    elif accepted:
        reason = "Considering — will decide after reviewing the market"
    elif not meets_floor:
        reason = "Wants higher AAV"
    elif preferred_clause != "None" and not ntc and not nmc:
        reason = f"Wants {preferred_clause} protection"
    elif years < want_years - 1:
        reason = "Wants more term"
    else:
        reason = "Wants more money"

    # Days until a non-instant accept resolves (1–5). Stronger interest = sooner.
    if accepted and not instant_accept:
        resolve_days = max(1, min(5, int(round((88.0 - interest) / 7.0)) + 1))
    else:
        resolve_days = 0

    counter_years = max(years, want_years)
    term_relief = 0.0
    if counter_years > years:
        term_relief = min(0.08, 0.02 * (counter_years - years) * (0.5 + security))
    counter_aav = round(max(effective_min, want_aav * (1.0 - term_relief)), 3)
    counter_aav = max(counter_aav, round(aav_m * 1.02, 3))
    counter_ntc = bool(ntc or preferred_clause in ("NTC", "NMC", "M-NTC"))
    counter_nmc = bool(nmc or preferred_clause == "NMC")
    counter_ntc_mode = "FULL" if counter_ntc and preferred_clause != "M-NTC" else (
        "MODIFIED" if preferred_clause == "M-NTC" else ("FULL" if counter_ntc else "NONE")
    )
    counter_bonus = 0.0
    if bonus_m <= 0 and gamble >= 0.55 and want_aav >= 3.5:
        counter_bonus = round(min(want_aav * 0.35, want_aav * years * 0.08), 3)

    projected: Optional[float] = None
    try:
        snap = get_team_cap_snapshot_full(team, league)
        projected = round(float(snap["usable_cap_space_m"]) - cap_hit_m, 3)
    except Exception:
        projected = None

    variance_raw = float(prof.get("variance") or 0.0)
    if variance_raw >= 0.025:
        agent_mood = "Agent is firm on ask."
    elif variance_raw <= -0.025:
        agent_mood = "Agent seemed flexible."
    else:
        agent_mood = "Agent confidence unknown."

    risk_tags: List[str] = []
    if compute_bad_contract_score(player, team) >= 0.3:
        risk_tags.append("Overpay risk")
    if years >= 5 and age >= 30:
        risk_tags.append("Term risk")
    if nmc:
        risk_tags.append("Full NMC")
    elif ntc:
        risk_tags.append("NTC")
    elif ntc_mode == "MODIFIED":
        risk_tags.append("M-NTC")

    return {
        "accepted": bool(accepted),
        "instant_accept": bool(instant_accept),
        "pending_decision": bool(accepted and not instant_accept),
        "resolve_days": int(resolve_days),
        "interest": round(interest, 1),
        "accept_cut": round(accept_cut, 1),
        "interest_range_low": round(max(0.0, interest - 5.0), 1),
        "interest_range_high": round(min(100.0, interest + 5.0), 1),
        "agent_mood": agent_mood,
        "stay_interest": round(stay_interest, 1),
        "stay_label": stay_label,
        "reason": reason,
        "aav_m": round(aav_m, 3),
        "cap_hit_m": round(cap_hit_m, 3),
        "signing_bonus_m": round(bonus_m, 3),
        "counter_offer": {
            "aav_m": counter_aav,
            "years": counter_years,
            "ntc": counter_ntc,
            "nmc": counter_nmc,
            "ntc_mode": counter_ntc_mode,
            "signing_bonus_m": counter_bonus,
            "cap_hit_m": compute_prorated_cap_hit_m(counter_aav, counter_years, counter_bonus),
        },
        "risk_tags": risk_tags,
        "projected_cap_after_m": projected,
        "market_value_m": demand["market_value_m"],
        "want_aav_m": want_aav,
        "min_acceptable_aav_m": effective_min,
        "want_years": want_years,
        "preferred_clause": preferred_clause,
        "clause_note": clause_note,
        "components": {
            "salary": round(salary_component, 1),
            "term": round(term_component, 1),
            "fit": round(fit_component, 1),
            "relationship": round(relationship_component, 1),
            "clauses": round(clause_component, 1),
            "signing_bonus": round(bonus_component, 1),
        },
    }


# ---------------------------------------------------------------------------
# Team needs & CPU signing intelligence
# ---------------------------------------------------------------------------

CPU_POSITIONS = ("C", "LW", "RW", "LD", "RD", "G")
CPU_POSITION_MIN = {"C": 3, "LW": 3, "RW": 3, "LD": 3, "RD": 3, "G": 2}
CPU_POSITION_OVERLOAD = {"C": 5, "LW": 5, "RW": 5, "LD": 6, "RD": 6, "G": 3}
CPU_SIGN_MIN_FIT_SCORE = 0.38


def ovr_band(ovr: float) -> str:
    o = float(ovr)
    if o >= 90:
        return "90+"
    if o >= 85:
        return "85-89"
    if o >= 80:
        return "80-84"
    if o >= 75:
        return "75-79"
    if o >= 70:
        return "70-74"
    return "under-70"


def _position_bucket(player: Any) -> str:
    pos = _player_pos(player)
    if pos in CPU_POSITIONS:
        return pos
    if pos == "D":
        hand = str(
            _get(_get(player, "identity", None), "shoots", None)
            or _get(player, "shoots", "L")
            or "L"
        ).upper()
        return "LD" if hand == "L" else "RD"
    if pos in ("W",):
        return "LW"
    return "C"


def _team_position_snapshot(team: Any) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
    counts = {k: 0 for k in CPU_POSITIONS}
    quality_sum = {k: 0.0 for k in CPU_POSITIONS}
    best_ovr = {k: 0.0 for k in CPU_POSITIONS}

    for p in _active_roster(team):
        bucket = _position_bucket(p)
        ovr = _player_ovr(p)
        counts[bucket] = counts.get(bucket, 0) + 1
        quality_sum[bucket] = quality_sum.get(bucket, 0.0) + ovr
        best_ovr[bucket] = max(best_ovr.get(bucket, 0.0), ovr)

    avg_ovr = {k: quality_sum[k] / max(1, counts[k]) for k in CPU_POSITIONS}
    return counts, avg_ovr, best_ovr


def compute_priority_extension_reserve_m(
    team: Any,
    league: Any = None,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
) -> float:
    """Cap dollars reserved for impending core extensions / own RFAs before open-market spend.

    Purpose: prevent clubs from blowing future flexibility on marginal UFAs, then
    discovering they cannot afford their own stars. Bound so rebuilders and depth
    clubs do not hoard the entire cap for fringe expirings.
    """
    reserved = 0.0
    core_hits = 0
    for p in _all_rostered(team):
        yrs = _contract_years_remaining(p)
        if yrs != 1:
            continue
        ovr = _player_ovr(p)
        if ovr < 82:
            continue
        # Only true core / franchise priorities create hard reserves.
        if ovr < 86 and core_hits >= 2:
            continue
        fair = compute_market_value(p, league)
        # Reserve a discounted share — not the full ask — so the club can still
        # fill holes while protecting space for the re-sign.
        share = 0.85 if ovr >= 90 else (0.70 if ovr >= 86 else 0.45)
        reserved += fair * share
        core_hits += 1
        if core_hits >= 3:
            break

    # Qualified RFA rights that still need contracts.
    for entry in _ensure_rfa_rights_list(team):
        if entry.get("status") == "RELEASED":
            continue
        ovr = float(entry.get("overall") or 0)
        if ovr < 80:
            continue
        qo = float(entry.get("qualifying_offer_aav_m") or LEAGUE_MINIMUM_AAV_M)
        reserved += qo * (0.90 if ovr >= 86 else 0.55)

    snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
    usable = max(0.0, float(snap.get("usable_cap_space_m", 0) or 0))
    # Never reserve more than 45% of usable space or the club cannot function.
    return round(min(reserved, usable * 0.45), 3)


def get_team_competitive_window(
    team: Any,
    league: Any,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
) -> str:
    raw = str(_get(team, "gm_window", _get(team, "window", "")) or "").lower()
    aliases = {
        "contend": "contender",
        "contender": "contender",
        "win_now": "contender",
        "rebuild": "rebuilder",
        "rebuilder": "rebuilder",
        "tank": "rebuilder",
        "bubble": "bubble",
        "balanced": "bubble",
    }
    if raw in aliases:
        window = aliases[raw]
    else:
        roster = _active_roster(team)
        avg = sum(_player_ovr(p) for p in roster) / max(1, len(roster)) if roster else 75.0
        if avg >= 84.0:
            window = "contender"
        elif avg <= 76.0:
            window = "rebuilder"
        else:
            window = "bubble"

    snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
    # Real teams frequently sit with $1-2M of "tight but workable" room and still
    # chase free agents (they just can't overpay). A too-generous cap_strapped
    # trigger here previously mislabeled most of the league as broke every
    # summer, which — combined with cpu_signing_blocked's luxury gate — starved
    # the CPU free-agent market down to a handful of active clubs. Only flag a
    # team as genuinely cap_strapped once it's below roughly one league-minimum
    # contract of breathing room.
    if float(snap.get("usable_cap_space_m", 0)) < 1.0:
        return "cap_strapped"
    slots_used = int(snap.get("contract_slots_used", 0))
    slots_limit = int(snap.get("contract_slots_limit", CONTRACT_SLOTS_LIMIT))
    if slots_limit - slots_used <= 2 and float(snap.get("usable_cap_space_m", 0)) < 3.0:
        return "cap_strapped"
    return window


def evaluate_team_position_needs(
    team: Any,
    league: Any,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
) -> Dict[str, Any]:
    counts, avg_ovr, best_ovr = _team_position_snapshot(team)
    need_score: Dict[str, float] = {}
    overload: Dict[str, bool] = {}
    signed_prospect_counts = {k: 0 for k in CPU_POSITIONS}

    for p in _get(team, "prospect_pool", None) or []:
        if not (_get(p, "contract", None) and _contract_years_remaining(p) > 0):
            continue
        bucket = _position_bucket(p)
        signed_prospect_counts[bucket] = signed_prospect_counts.get(bucket, 0) + 1

    for pos in CPU_POSITIONS:
        n = counts.get(pos, 0)
        min_t = CPU_POSITION_MIN[pos]
        over_t = CPU_POSITION_OVERLOAD[pos]
        overload[pos] = n >= over_t
        if pos == "G":
            avg_g = avg_ovr.get("G", 0.0)
            if n == 0:
                need_score[pos] = 0.95
            elif n == 1:
                need_score[pos] = 0.42 if avg_g < 78 else 0.18
            elif n >= 2:
                need_score[pos] = 0.04 if avg_g >= 76 else 0.22
        elif n < min_t:
            need_score[pos] = min(1.0, 0.45 + (min_t - n) * 0.22)
        elif n == min_t:
            avg = avg_ovr.get(pos, 0.0)
            need_score[pos] = 0.55 if avg < 76 else 0.35
        elif overload[pos]:
            need_score[pos] = 0.05
        else:
            need_score[pos] = max(0.08, 0.42 - (n - min_t) * 0.12)

        if signed_prospect_counts.get(pos, 0) >= 2 and need_score[pos] > 0.2:
            need_score[pos] = max(0.1, need_score[pos] - 0.15)

    primary_needs = sorted(CPU_POSITIONS, key=lambda p: (-need_score[p], counts.get(p, 0)))

    snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
    slots_used = int(snap.get("contract_slots_used", 0))
    slots_limit = int(snap.get("contract_slots_limit", CONTRACT_SLOTS_LIMIT))
    usable = float(snap.get("usable_cap_space_m", 0))
    reserve = compute_priority_extension_reserve_m(
        team, league, sim, season_year=season_year
    )
    spendable = max(0.0, usable - reserve)

    return {
        "counts": counts,
        "avg_ovr": avg_ovr,
        "best_ovr": best_ovr,
        "need_score": need_score,
        "overload": overload,
        "primary_needs": primary_needs,
        "signed_prospect_counts": signed_prospect_counts,
        "window": get_team_competitive_window(team, league, sim, season_year=season_year),
        "cap_space_m": usable,
        "extension_reserve_m": reserve,
        "spendable_cap_space_m": spendable,
        "slots_remaining": max(0, slots_limit - slots_used),
        "roster_count": len(_active_roster(team)),
    }


def compute_team_needs(team: Any) -> Dict[str, str]:
    """Legacy high/medium/low need labels — kept for contract office compatibility."""
    ctx = evaluate_team_position_needs(team, None)

    def level(pos: str) -> str:
        score = float(ctx["need_score"].get(pos, 0))
        if score >= 0.55:
            return "high"
        if score >= 0.30:
            return "medium"
        return "low"

    return {pos: level(pos) for pos in ("C", "LW", "RW", "LD", "RD", "G")}


def cpu_signing_blocked(
    team: Any,
    player: Any,
    ctx: Dict[str, Any],
    offer_aav: float,
) -> Optional[str]:
    pos = _position_bucket(player)
    ovr = _player_ovr(player)
    counts = ctx["counts"]
    best = ctx["best_ovr"].get(pos, 0.0)
    window = ctx["window"]
    need = float(ctx["need_score"].get(pos, 0))
    spendable = float(ctx.get("spendable_cap_space_m", ctx.get("cap_space_m", 0)) or 0)

    if ctx["overload"].get(pos) and ovr < best + 3.0 and ovr < 88:
        return "position_overload"
    if pos == "G" and counts.get("G", 0) >= 2 and ovr < 82:
        return "goalie_overload"
    if pos == "G" and counts.get("G", 0) >= 3:
        return "goalie_overload_hard"
    # Replacement-level: never chase without a real hole.
    if ovr < 70 and need < 0.45:
        return "fringe_no_need"
    if ovr < 76 and need < 0.28 and ovr < best + 1.0:
        return "depth_no_upgrade"
    if window == "rebuilder" and _player_age(player) >= 33 and offer_aav >= 4.5 and ovr < 84:
        return "rebuilder_old_expensive"
    # Cap-strapped clubs still fill out a bottom-six/AHL taxi squad every summer —
    # only reject genuinely extravagant depth overpays, not a normal $1-1.5M
    # depth deal, or the whole strapped half of the league goes silent on FAs.
    if window == "cap_strapped" and offer_aav > LEAGUE_MINIMUM_AAV_M + 0.75 and ovr < 76:
        return "cap_strapped_luxury"
    if ctx["slots_remaining"] <= 3 and ovr < 72 and need < 0.35:
        return "slot_pressure_non_need"
    # Do not spend reserved extension money on marginal open-market adds.
    # Emergency holes (no goalie, severe need) may still use reserved dollars.
    if (
        ovr < 82
        and offer_aav > spendable + 1e-6
        and need < 0.55
        and not (pos == "G" and counts.get("G", 0) == 0)
    ):
        return "extension_reserve"
    return None


def score_free_agent_fit(
    team: Any,
    player: Any,
    ctx: Dict[str, Any],
    offer_aav: float,
    years: int,
    league: Any,
) -> Tuple[float, List[str]]:
    pos = _position_bucket(player)
    ovr = _player_ovr(player)
    age = _player_age(player)
    window = ctx["window"]
    need = float(ctx["need_score"].get(pos, 0))
    counts = ctx["counts"]
    best = ctx["best_ovr"].get(pos, 0.0)
    reasons: List[str] = []

    score = 0.12 + (ovr / 99.0) * 0.28 + need * 0.42

    if ovr >= best + 5:
        score += 0.14
        reasons.append("clear_upgrade")
    elif ovr >= best + 1:
        score += 0.06
        reasons.append("roster_upgrade")

    if ctx["overload"].get(pos):
        # Elite clear upgrades still move the needle — depth-chart congestion is not
        # a veto on a 90+ defenseman just because a club already carries five RD.
        if ovr >= 88 or ovr >= best + 4:
            score -= 0.08
            reasons.append("upgrade_despite_depth")
        else:
            score -= 0.50
            reasons.append("overload_penalty")
    elif counts.get(pos, 0) >= CPU_POSITION_OVERLOAD[pos] - 1 and ovr < best + 2:
        score -= 0.22
        reasons.append("near_overload")

    if pos == "G":
        if counts.get("G", 0) == 0:
            score += 0.28
            reasons.append("goalie_emergency")
        elif counts.get("G", 0) == 1 and need >= 0.35:
            score += 0.10
            reasons.append("backup_goalie_need")
        elif counts.get("G", 0) >= 2:
            score -= 0.55
            reasons.append("third_goalie_penalty")

    if window == "contender":
        if ovr >= 82 and need >= 0.35:
            score += 0.10
            reasons.append("contender_need_fit")
        if age > 34 and years > 2:
            score -= 0.18
            reasons.append("contender_term_risk")
        if ovr < 78 and need < 0.40:
            score -= 0.12
            reasons.append("contender_luxury_depth")
    elif window == "rebuilder":
        if age >= 32 and offer_aav >= 4.0:
            score -= 0.35
            reasons.append("rebuilder_vet_overpay")
        if age <= 27 and ovr >= 74:
            score += 0.08
            reasons.append("rebuilder_youth")
        if years > 3 and age >= 30:
            score -= 0.20
            reasons.append("rebuilder_long_term")
    elif window == "cap_strapped":
        if offer_aav > LEAGUE_MINIMUM_AAV_M + 0.05:
            score -= 0.25
            reasons.append("cap_strapped_cost")
        if need < 0.45:
            score -= 0.20
            reasons.append("cap_strapped_non_essential")
    else:
        if need >= 0.50:
            score += 0.08
            reasons.append("bubble_hole_fill")
        if years >= 5 and ovr < 82:
            score -= 0.15
            reasons.append("bubble_long_deal")

    fair = compute_fair_aav(player, team, league)
    if offer_aav > fair * 1.15 and ovr < 80:
        score -= 0.12
        reasons.append("overpay_depth")

    cap_after = float(ctx["cap_space_m"]) - offer_aav
    if cap_after < 0:
        score -= 0.40
        reasons.append("cap_negative")
    elif cap_after < 1.5 and need < 0.45:
        score -= 0.10
        reasons.append("cap_tight_non_need")

    return max(0.0, min(1.25, score)), reasons


def validate_post_fa_roster_shape(team: Any, league: Any, sim: Any = None) -> List[Dict[str, Any]]:
    """Return warning rows for suspicious post-FA roster shapes."""
    ctx = evaluate_team_position_needs(team, league, sim)
    issues: List[Dict[str, Any]] = []
    tid = str(_get(team, "team_id", "") or _get(team, "id", ""))

    counts = ctx["counts"]
    if counts.get("G", 0) >= 4:
        issues.append({"team_id": tid, "code": "goalie_overload", "counts": dict(counts)})
    for pos in ("C", "LW", "RW"):
        if counts.get(pos, 0) >= 7:
            issues.append({"team_id": tid, "code": f"{pos.lower()}_overload", "counts": dict(counts)})
    for pos in ("LD", "RD"):
        if counts.get(pos, 0) >= 6:
            issues.append({"team_id": tid, "code": f"{pos.lower()}_overload", "counts": dict(counts)})

    primary = ctx["primary_needs"][0] if ctx["primary_needs"] else "C"
    if float(ctx["need_score"].get(primary, 0)) >= 0.55 and counts.get(primary, 0) < CPU_POSITION_MIN[primary]:
        issues.append({"team_id": tid, "code": "unmet_primary_need", "position": primary, "counts": dict(counts)})

    low_ovr = sum(1 for p in _active_roster(team) if _player_ovr(p) < 70)
    if low_ovr >= 6 and ctx["window"] in ("contender", "bubble"):
        issues.append({"team_id": tid, "code": "too_many_fringe", "fringe_count": low_ovr})

    return issues


# ---------------------------------------------------------------------------
# Contract actions
# ---------------------------------------------------------------------------

def _validate_sign_cap(team: Any, aav_m: float, league: Any, *, player: Any = None) -> Dict[str, Any]:
    return can_sign_player(team, aav_m, league=league, player=player)


def _find_player_in_league(league: Any, player_id: str) -> Tuple[Optional[Any], Optional[Any]]:
    pid = str(player_id)
    for p in _get(league, "free_agents", None) or []:
        if _player_id(p) == pid:
            return p, None
    # Overseas/minor-league unsigned players are also part of the signable pool.
    for p in _get(league, "overseas_free_agents", None) or []:
        if _player_id(p) == pid:
            return p, None
    for team in _get(league, "teams", None) or []:
        for p in _all_rostered(team):
            if _player_id(p) == pid:
                return p, team
        for r in _ensure_rfa_rights_list(team):
            if str(r.get("player_id", "")) == pid:
                player = resolve_rfa_player(r, league)
                if player is not None:
                    return player, team
    # Last resort: registry (covers RFA refs that lost their team list attachment).
    try:
        from services.draft_player_registry import get_player

        player = get_player(league, pid)
        if player is not None:
            return player, None
    except Exception:
        pass
    return None, None


def _remove_from_unsigned_pools(league: Any, player: Any) -> None:
    """Purge a newly-signed player from every unsigned pool so no duplicate copy survives.

    Covers the NHL free-agent pool, the overseas free-agent pool, and any external
    development-league team roster the player may have been assigned to. Also clears the
    overseas/free-agent assignment markers so the player is no longer treated as unsigned.
    """
    pid = _player_id(player)

    fa_pool = list(_get(league, "free_agents", None) or [])
    league.free_agents = [p for p in fa_pool if _player_id(p) != pid]

    overseas = list(_get(league, "overseas_free_agents", None) or [])
    if overseas:
        league.overseas_free_agents = [p for p in overseas if _player_id(p) != pid]

    for block in _get(league, "development_leagues", None) or []:
        if not isinstance(block, dict):
            continue
        for tm in block.get("teams") or []:
            players = tm.get("players")
            if isinstance(players, list) and any(_player_id(p) == pid for p in players):
                tm["players"] = [p for p in players if _player_id(p) != pid]

    meta = getattr(player, "_franchise_assignment", None)
    if isinstance(meta, dict):
        meta.pop("overseas", None)
        meta.pop("overseas_league", None)
        meta["level"] = "nhl_signed"


# ---------------------------------------------------------------------------
# Minor-league / tryout contracts (do not consume NHL 50-slot by default)
# ---------------------------------------------------------------------------

AHL_MIN_SALARY_M = 0.055
ECHL_MIN_SALARY_M = 0.025
PTO_DEFAULT_DAYS = 25


def build_ahl_contract(season_year: int, *, aav_m: float = 0.085, years: int = 1, echl_salary_m: float = 0.04) -> Dict[str, Any]:
    aav = max(AHL_MIN_SALARY_M, float(aav_m or AHL_MIN_SALARY_M))
    y = max(1, min(3, int(years or 1)))
    return normalize_contract_dict({
        "type": "AHL",
        "contract_type": "AHL",
        "years": y,
        "years_remaining": y,
        "aav_m": aav,
        "cap_hit_m": 0.0,  # No NHL cap hit
        "base_salary_m": aav,
        "salary_m": aav,
        "minor_salary_m": aav,
        "nhl_salary_by_year_m": [0.0] * y,
        "minor_salary_by_year_m": [aav] * y,
        "echl_salary_m": float(echl_salary_m or ECHL_MIN_SALARY_M),
        "two_way": False,
        "is_entry_level": False,
        "rights_status": "AHL",
        "expiry_year": int(season_year) + y,
        "effective_season": int(season_year),
        "source": "ahl_spc",
        "schema_version": CONTRACT_SCHEMA_VERSION,
    })


def build_ahl_echl_two_way_contract(
    season_year: int,
    *,
    ahl_salary_m: float = 0.085,
    echl_salary_m: float = 0.04,
    years: int = 1,
) -> Dict[str, Any]:
    ahl = max(AHL_MIN_SALARY_M, float(ahl_salary_m or AHL_MIN_SALARY_M))
    echl = max(ECHL_MIN_SALARY_M, float(echl_salary_m or ECHL_MIN_SALARY_M))
    y = max(1, min(2, int(years or 1)))
    return normalize_contract_dict({
        "type": "AHL_ECHL",
        "contract_type": "AHL_ECHL",
        "years": y,
        "years_remaining": y,
        "aav_m": ahl,
        "cap_hit_m": 0.0,
        "salary_m": ahl,
        "minor_salary_m": ahl,
        "echl_salary_m": echl,
        "nhl_salary_by_year_m": [0.0] * y,
        "minor_salary_by_year_m": [ahl] * y,
        "assignment_levels": ["AHL", "ECHL"],
        "two_way": True,
        "rights_status": "AHL",
        "expiry_year": int(season_year) + y,
        "effective_season": int(season_year),
        "source": "ahl_echl_two_way",
        "schema_version": CONTRACT_SCHEMA_VERSION,
    })


def build_echl_contract(season_year: int, *, aav_m: float = 0.03, years: int = 1) -> Dict[str, Any]:
    aav = max(ECHL_MIN_SALARY_M, float(aav_m or ECHL_MIN_SALARY_M))
    y = max(1, min(2, int(years or 1)))
    return normalize_contract_dict({
        "type": "ECHL",
        "contract_type": "ECHL",
        "years": y,
        "years_remaining": y,
        "aav_m": aav,
        "cap_hit_m": 0.0,
        "salary_m": aav,
        "minor_salary_m": aav,
        "echl_salary_m": aav,
        "nhl_salary_by_year_m": [0.0] * y,
        "minor_salary_by_year_m": [aav] * y,
        "assignment_levels": ["ECHL"],
        "rights_status": "ECHL",
        "expiry_year": int(season_year) + y,
        "effective_season": int(season_year),
        "source": "echl_spc",
        "schema_version": CONTRACT_SCHEMA_VERSION,
    })


def build_pto_contract(season_year: int, *, days: int = PTO_DEFAULT_DAYS) -> Dict[str, Any]:
    d = max(7, min(45, int(days or PTO_DEFAULT_DAYS)))
    return normalize_contract_dict({
        "type": "PTO",
        "contract_type": "PTO",
        "years": 0,
        "years_remaining": 0,
        "aav_m": 0.0,
        "cap_hit_m": 0.0,
        "salary_m": 0.0,
        "pto_days": d,
        "pto_expires_season": int(season_year),
        "is_tryout": True,
        "rights_status": "PTO",
        "expiry_year": int(season_year),
        "effective_season": int(season_year),
        "source": "professional_tryout",
        "schema_version": CONTRACT_SCHEMA_VERSION,
    })


def sign_minor_or_tryout_contract(
    player: Any,
    team: Any,
    league: Any,
    season_year: int,
    offer: Dict[str, Any],
) -> Dict[str, Any]:
    """Sign AHL / ECHL / AHL-ECHL two-way / PTO — no NHL slot / no NHL cap hit."""
    category = str(offer.get("contract_category") or offer.get("contract_type") or "").lower()
    years = max(1, int(offer.get("years") or 1))
    aav = normalize_money_m(offer.get("aav_m") or offer.get("minor_salary_m") or 0)

    if category in ("pto", "tryout", "professional_tryout"):
        contract = build_pto_contract(season_year, days=int(offer.get("pto_days") or PTO_DEFAULT_DAYS))
        apply_contract_to_player(player, contract, season_year)
        try:
            player.signed_status = "pto"
            player.organizational_status = "tryout"
            player.status = "tryout"
        except Exception:
            pass
        return {"ok": True, "allowed": True, "contract": contract, "contract_category": "pto"}

    if category in ("echl",):
        contract = build_echl_contract(season_year, aav_m=aav or ECHL_MIN_SALARY_M, years=years)
        dest = "ECHL"
    elif category in ("ahl_echl_two_way", "ahl_echl", "ahl/echl"):
        contract = build_ahl_echl_two_way_contract(
            season_year,
            ahl_salary_m=aav or AHL_MIN_SALARY_M,
            echl_salary_m=normalize_money_m(offer.get("echl_salary_m") or ECHL_MIN_SALARY_M),
            years=years,
        )
        dest = "AHL"
    else:
        contract = build_ahl_contract(
            season_year,
            aav_m=aav or AHL_MIN_SALARY_M,
            years=years,
            echl_salary_m=normalize_money_m(offer.get("echl_salary_m") or ECHL_MIN_SALARY_M),
        )
        dest = "AHL"
        category = "ahl"

    # Supersede lower agreements
    apply_contract_to_player(player, contract, season_year)
    try:
        player.signed_status = "signed"
        player.entry_level_contract_eligible = False
        player.organizational_status = f"signed_{dest.lower()}"
        player.development_path = dest
        player.current_league_id = dest
        player.status = "minor"
    except Exception:
        pass

    # Place on affiliate / prospect pool — not NHL roster
    pool = list(_get(team, "prospect_pool", None) or [])
    roster = list(_get(team, "roster", None) or [])
    if player in roster:
        roster.remove(player)
        team.roster = roster
    if player not in pool:
        pool.append(player)
        team.prospect_pool = pool

    # Remove from FA pool if present
    try:
        fa = list(_get(league, "free_agents", None) or [])
        pid = _player_id(player)
        league.free_agents = [p for p in fa if _player_id(p) != pid]
    except Exception:
        pass

    sync_team_cap_fields(team, league)
    return {
        "ok": True,
        "allowed": True,
        "contract": contract,
        "contract_category": category,
        "assignment": dest,
        "uses_nhl_slot": False,
        "nhl_cap_hit_m": 0.0,
    }


def sign_player_to_team(
    player: Any,
    team: Any,
    league: Any,
    season_year: int,
    offer: Dict[str, Any],
) -> Dict[str, Any]:
    category = str(offer.get("contract_category") or offer.get("contract_type") or "").lower()
    if category in (
        "ahl",
        "echl",
        "ahl_echl_two_way",
        "ahl_echl",
        "ahl/echl",
        "pto",
        "tryout",
        "professional_tryout",
    ):
        return sign_minor_or_tryout_contract(player, team, league, season_year, offer)

    aav_m = round(normalize_money_m(offer.get("aav_m") or offer.get("aav") or 0), 3)
    years = max(1, int(offer.get("years") or offer.get("term") or 1))
    bonus_m = round(normalize_money_m(offer.get("signing_bonus_m") or offer.get("signing_bonus") or 0), 3)
    if bonus_m > 0:
        from services.franchise_offseason import (
            SIGNING_BONUS_REVENUE_FLOOR_M,
            signing_bonus_max_pct_for_revenue,
        )

        revenue_m = None
        try:
            revenue_m = float(getattr(team, "revenue_m", None) or getattr(team, "annual_revenue_m", None) or 0) or None
        except Exception:
            revenue_m = None
        if revenue_m is None:
            try:
                from services.league_operations import calculate_team_revenue
                session = offer.get("_session")
                if session is not None:
                    tid = str(_get(team, "team_id", "") or _get(team, "id", ""))
                    row = calculate_team_revenue(session, team, tid, is_user=False)
                    revenue_m = float(row.get("revenue") or row.get("revenue_m") or 0) or None
            except Exception:
                revenue_m = None
        if revenue_m is None:
            return {"ok": False, "reason": "Team revenue unavailable — cannot validate signing bonus"}
        if revenue_m < SIGNING_BONUS_REVENUE_FLOOR_M:
            return {
                "ok": False,
                "reason": (
                    f"Signing bonus not allowed — NHL revenue eligibility requires "
                    f"${SIGNING_BONUS_REVENUE_FLOOR_M:.0f}M (club at ${revenue_m:.1f}M)"
                ),
                "revenue_m": revenue_m,
                "floor_m": SIGNING_BONUS_REVENUE_FLOOR_M,
            }
        total_value = aav_m * years
        max_pct = signing_bonus_max_pct_for_revenue(revenue_m)
        if total_value > 0 and (bonus_m / total_value) > max_pct + 1e-6:
            return {
                "ok": False,
                "reason": f"Signing bonus exceeds revenue-based flexibility ({max_pct:.0%} of contract value)",
                "max_bonus_pct": max_pct,
                "revenue_m": revenue_m,
            }

    already_contracted = (
        _player_id(player) in {_player_id(p) for p in _all_rostered(team)}
        and has_active_contract(player)
    )
    slot_check = validate_contract_slots(team, league, additional=0 if already_contracted else 1)
    if not slot_check.get("ok"):
        return {
            "ok": False,
            "status": "invalid",
            "reason": slot_check.get("reason"),
            "contract_slots": slot_check,
        }
    check = _validate_sign_cap(team, compute_prorated_cap_hit_m(aav_m, years, bonus_m), league, player=player)
    if not check.get("ok"):
        return {
            "ok": False,
            "status": "invalid",
            "reason": check.get("reason"),
            "snapshot": check.get("snapshot"),
        }

    if offer.get("evaluate_only"):
        evaluation = evaluate_contract_offer(
            player, team, offer, league, context=str(offer.get("context") or "ufa")
        )
        return {
            "ok": True,
            "status": "evaluated",
            "evaluation": evaluation,
            "player_response": {
                "status": "evaluated",
                "interest": evaluation.get("interest"),
                "stay_interest": evaluation.get("stay_interest"),
                "stay_label": evaluation.get("stay_label"),
                "want_aav_m": evaluation.get("want_aav_m"),
                "want_years": evaluation.get("want_years"),
                "preferred_clause": evaluation.get("preferred_clause"),
                "clause_note": evaluation.get("clause_note"),
                "feedback": _offer_feedback_label(evaluation, aav_m, years),
                "counter_cap_hit": (evaluation.get("counter_offer") or {}).get("aav_m"),
                "counter_term": (evaluation.get("counter_offer") or {}).get("years"),
                "instant_accept": evaluation.get("instant_accept"),
                "pending_decision": evaluation.get("pending_decision"),
                "resolve_days": evaluation.get("resolve_days"),
            },
        }

    eval_result = evaluate_contract_offer(player, team, offer, league, context=str(offer.get("context") or "ufa"))
    session = offer.get("_session")
    pid = _player_id(player)

    if session is not None and pid:
        neg_map = getattr(session, "resign_negotiations", None)
        if not isinstance(neg_map, dict):
            session.resign_negotiations = {}
            neg_map = session.resign_negotiations
        entry = neg_map.get(pid) if isinstance(neg_map.get(pid), dict) else None
        if entry is None:
            entry = {
                "negotiation_id": f"nego-{pid}",
                "player_id": pid,
                "status": "open",
                "current_round": 0,
                "team_offers": [],
                "player_counters": [],
            }
            neg_map[pid] = entry
        entry["current_round"] = int(entry.get("current_round") or 0) + 1
        entry.setdefault("team_offers", []).append(
            {
                "round": entry["current_round"],
                "aav_m": aav_m,
                "years": years,
                "ntc": bool(offer.get("ntc")),
                "nmc": bool(offer.get("nmc")),
                "signing_bonus_m": bonus_m,
                "interest": eval_result.get("interest"),
            }
        )

    if not eval_result.get("accepted") and not offer.get("force"):
        interest = float(eval_result.get("interest") or 0)
        status = "rejected" if interest < 40.0 else "countered"
        counter = eval_result.get("counter_offer") or {}
        if session is not None and pid and status == "countered":
            neg_map = getattr(session, "resign_negotiations", {}) or {}
            entry = neg_map.get(pid) if isinstance(neg_map.get(pid), dict) else None
            if entry is not None:
                entry.setdefault("player_counters", []).append(
                    {
                        "round": entry.get("current_round"),
                        "aav_m": counter.get("aav_m"),
                        "years": counter.get("years"),
                        "ntc": counter.get("ntc"),
                        "nmc": counter.get("nmc"),
                        "signing_bonus_m": counter.get("signing_bonus_m"),
                    }
                )
                entry["status"] = "countered"
                entry["pending_offer"] = None
        elif session is not None and pid and status == "rejected":
            neg_map = getattr(session, "resign_negotiations", {}) or {}
            entry = neg_map.get(pid) if isinstance(neg_map.get(pid), dict) else None
            if entry is not None:
                entry["status"] = "rejected"
                entry["pending_offer"] = None
        if session is not None and pid and str(offer.get("context") or "") in ("re_sign", "extension", ""):
            try:
                from services.franchise_offseason import upsert_resign_phase_outcome

                upsert_resign_phase_outcome(
                    session,
                    player_id=pid,
                    phase_status=status,
                    name=_player_name(player),
                    last_offer={
                        "aav_m": aav_m,
                        "years": years,
                        "ntc": bool(offer.get("ntc")),
                        "nmc": bool(offer.get("nmc")),
                        "signing_bonus_m": bonus_m,
                    },
                    terms={
                        "aav_m": counter.get("aav_m") if status == "countered" else None,
                        "years": counter.get("years") if status == "countered" else None,
                    } if status == "countered" else None,
                    reason=str(eval_result.get("reason") or ""),
                )
            except Exception:
                pass
        return {
            "ok": False,
            "status": status,
            "reason": eval_result.get("reason"),
            "evaluation": eval_result,
            "player_response": {
                "status": status,
                "reason": eval_result.get("reason"),
                "interest": eval_result.get("interest"),
                "stay_interest": eval_result.get("stay_interest"),
                "stay_label": eval_result.get("stay_label"),
                "counter_cap_hit": counter.get("aav_m"),
                "counter_term": counter.get("years"),
                "counter_ntc": counter.get("ntc"),
                "counter_nmc": counter.get("nmc"),
                "counter_signing_bonus_m": counter.get("signing_bonus_m"),
                "want_aav_m": eval_result.get("want_aav_m"),
                "want_years": eval_result.get("want_years"),
                "preferred_clause": eval_result.get("preferred_clause"),
                "clause_note": eval_result.get("clause_note"),
                "accept_cut": eval_result.get("accept_cut"),
                "agent_mood": eval_result.get("agent_mood"),
                "cap_hit_m": eval_result.get("cap_hit_m"),
                "aav_m": eval_result.get("aav_m"),
                "feedback": _offer_feedback_label(eval_result, aav_m, years),
            },
            "negotiation": (getattr(session, "resign_negotiations", {}) or {}).get(pid) if session else None,
        }

    # Competitive but not insane: hold pending until day sim resolves (unless force/resolve).
    if (
        eval_result.get("pending_decision")
        and not offer.get("force")
        and not offer.get("resolve_pending")
        and session is not None
    ):
        resolve_days = max(1, int(eval_result.get("resolve_days") or 2))
        neg_map = getattr(session, "resign_negotiations", None)
        if not isinstance(neg_map, dict):
            session.resign_negotiations = {}
            neg_map = session.resign_negotiations
        entry = neg_map.get(pid) if isinstance(neg_map.get(pid), dict) else {
            "negotiation_id": f"nego-{pid}",
            "player_id": pid,
            "team_offers": [],
            "player_counters": [],
            "current_round": 1,
        }
        entry["status"] = "pending"
        entry["pending_offer"] = {
            "aav_m": aav_m,
            "years": years,
            "ntc": bool(offer.get("ntc")),
            "nmc": bool(offer.get("nmc")),
            "signing_bonus_m": bonus_m,
            "two_way": bool(offer.get("two_way")),
            "contract_category": offer.get("contract_category") or offer.get("contract_type"),
            "context": str(offer.get("context") or "re_sign"),
            "interest": eval_result.get("interest"),
            "resolve_days": resolve_days,
            "days_held": 0,
            "submitted_window_day": int(getattr(session, "own_fa_window_day", 0) or 0),
        }
        neg_map[pid] = entry
        try:
            from services.franchise_offseason import invalidate_offseason_decision_payloads, upsert_resign_phase_outcome
            invalidate_offseason_decision_payloads(session, reason="pending_offer")
            upsert_resign_phase_outcome(
                session,
                player_id=pid,
                phase_status="pending",
                name=_player_name(player),
                last_offer={
                    "aav_m": aav_m,
                    "years": years,
                    "ntc": bool(offer.get("ntc")),
                    "nmc": bool(offer.get("nmc")),
                    "signing_bonus_m": bonus_m,
                    "resolve_days": resolve_days,
                },
                reason=str(eval_result.get("reason") or ""),
            )
        except Exception:
            pass
        return {
            "ok": True,
            "status": "pending",
            "reason": eval_result.get("reason"),
            "evaluation": eval_result,
            "player_response": {
                "status": "pending",
                "reason": eval_result.get("reason"),
                "interest": eval_result.get("interest"),
                "stay_interest": eval_result.get("stay_interest"),
                "stay_label": eval_result.get("stay_label"),
                "resolve_days": resolve_days,
                "feedback": "Offer is on the table — Sim Day to hear back",
                "preferred_clause": eval_result.get("preferred_clause"),
                "clause_note": eval_result.get("clause_note"),
                "accept_cut": eval_result.get("accept_cut"),
                "agent_mood": eval_result.get("agent_mood"),
                "cap_hit_m": eval_result.get("cap_hit_m"),
                "aav_m": eval_result.get("aav_m"),
            },
            "negotiation": entry,
        }

    if session is not None and pid:
        neg_map = getattr(session, "resign_negotiations", {}) or {}
        entry = neg_map.get(pid)
        if isinstance(entry, dict):
            entry["status"] = "accepted"
            entry["pending_offer"] = None

    contract = normalize_contract_dict({
        "type": offer.get("type") or "STANDARD",
        "years": years,
        "years_remaining": years,
        "aav_m": aav_m,
        "cap_hit_m": compute_prorated_cap_hit_m(aav_m, years, bonus_m),
        "base_salary_m": aav_m,
        "salary_m": aav_m,
        "signing_bonus_m": bonus_m,
        "rights_status": offer.get("rights") or "UFA",
        "expiry_year": int(season_year) + years,
        "two_way": bool(offer.get("two_way")),
        "source": "signed",
        **_contract_ntc_fields_from_offer(offer),
    })
    # Signed-state and playoff-eligibility are tracked separately (default eligible).
    playoff_eligible = bool(offer.get("playoff_eligible", True))
    contract["playoff_eligible"] = playoff_eligible
    if offer.get("signed_day") is not None:
        contract["signed_day"] = int(offer.get("signed_day") or 0)
    apply_contract_to_player(player, contract, season_year)
    try:
        player.playoff_eligible = playoff_eligible
    except Exception:
        pass

    _remove_from_unsigned_pools(league, player)
    # Negotiated re-sign of an RFA must clear rights or the ghost "Rights" row
    # stays on the Contract Table with a blank ask / stale QO path.
    clear_rfa_rights_for_player(league, _player_id(player), prefer_team=team)

    roster = list(_get(team, "roster", None) or [])
    if player not in roster:
        roster.append(player)
        team.roster = roster

    # FA / re-sign desks put players on the active NHL list — clear any leftover
    # minors/IR flags that would leave Roster Check pretending the club is short.
    try:
        player.is_buried = False
        player.buried = False
        player.in_minors = False
        player.on_ir = False
        player.on_ltir = False
        player.organizational_status = "signed"
    except Exception:
        pass

    try:
        player.team_id = str(_get(team, "team_id", "") or _get(team, "id", ""))
    except Exception:
        pass

    if session is not None and pid and str(offer.get("context") or "") in ("re_sign", "extension", "ufa", ""):
        try:
            from services.franchise_offseason import upsert_resign_phase_outcome

            upsert_resign_phase_outcome(
                session,
                player_id=pid,
                phase_status="accepted",
                name=_player_name(player),
                terms={
                    "aav_m": aav_m,
                    "years": years,
                    "expiry_year": contract.get("expiry_year"),
                    "ntc": bool(offer.get("ntc")),
                    "nmc": bool(offer.get("nmc")),
                },
                last_offer={
                    "aav_m": aav_m,
                    "years": years,
                    "ntc": bool(offer.get("ntc")),
                    "nmc": bool(offer.get("nmc")),
                    "signing_bonus_m": bonus_m,
                },
            )
        except Exception:
            pass

    sync_team_cap_fields(team, league)
    return {
        "ok": True,
        "status": "accepted",
        "player_id": _player_id(player),
        "contract": contract,
        "evaluation": eval_result,
        "final_term": years,
        "final_cap_hit": contract.get("cap_hit_m"),
        "expiry_year": contract.get("expiry_year"),
        "contract_type": contract.get("contract_type") or contract.get("type"),
    }


def _offer_feedback_label(evaluation: Dict[str, Any], aav_m: float, years: int) -> str:
    """Descriptive negotiation feedback — no hidden formula leakage."""
    interest = float(evaluation.get("interest") or 0)
    want_aav = float(evaluation.get("want_aav_m") or 0) or 0.01
    want_years = int(evaluation.get("want_years") or 1)
    reason = str(evaluation.get("reason") or "")
    clause_note = str(evaluation.get("clause_note") or "")
    if evaluation.get("instant_accept"):
        return "Slam dunk — signing immediately"
    if evaluation.get("pending_decision"):
        days = int(evaluation.get("resolve_days") or 1)
        return f"On the table — decision in about {days} day{'s' if days != 1 else ''}"
    if interest >= 75:
        return "Strong offer"
    if interest >= 62:
        return "Competitive"
    if "NMC" in reason or "NTC" in reason or "protection" in reason.lower():
        return clause_note or "Clause protection is a concern"
    if "term" in reason.lower() or years < want_years - 1:
        return "Term is a concern"
    if aav_m < want_aav * 0.85:
        return "Salary is a concern"
    if interest < 40:
        return "Player is unwilling to negotiate"
    return "Below expectations"


def qualify_rfa(team: Any, player_id: str, league: Any, season_year: int) -> Dict[str, Any]:
    entry = find_rfa_rights(team, player_id)
    if not entry:
        return {"ok": False, "reason": "RFA rights not found"}
    player = resolve_rfa_player(entry, league)
    if player is None:
        return {"ok": False, "reason": "Player reference missing"}

    qo = float(entry.get("qualifying_offer_aav_m") or qualifying_offer_aav(entry.get("previous_aav_m") or 0) or LEAGUE_MINIMUM_AAV_M)
    # A qualifying offer becomes a real contract, so it consumes a contract slot — the
    # player is currently held in rfa_rights (off-roster) and re-enters the 50-contract
    # count once signed. Validate the slot before the cap so a full org is told why.
    slot_check = validate_contract_slots(team, league, additional=1)
    if not slot_check.get("ok"):
        return {"ok": False, "reason": slot_check.get("reason"), "contract_slots": slot_check}
    check = _validate_sign_cap(team, qo, league)
    if not check.get("ok"):
        return {"ok": False, "reason": check.get("reason")}

    contract = normalize_contract_dict({
        "type": "STANDARD",
        "years": 1,
        "years_remaining": 1,
        "aav_m": qo,
        "cap_hit_m": qo,
        "rights_status": "RFA",
        "expiry_year": int(season_year) + 1,
        "source": "qualifying_offer",
    })
    apply_contract_to_player(player, contract, season_year)
    entry["qualified"] = True
    remove_rfa_rights(team, player_id)

    roster = list(_get(team, "roster", None) or [])
    if player not in roster:
        roster.append(player)
        team.roster = roster

    sync_team_cap_fields(team, league)
    return {"ok": True, "player_id": player_id, "contract": contract}


def release_rfa_rights(team: Any, player_id: str, league: Any, session: Any = None) -> Dict[str, Any]:
    entry = find_rfa_rights(team, player_id)
    if not entry:
        return {"ok": False, "reason": "RFA rights not found"}
    player = resolve_rfa_player(entry, league)
    name = _player_name(player) if player is not None else str(
        entry.get("name") if isinstance(entry, dict) else player_id
    )
    snapshot = {
        "player_id": str(player_id),
        "name": name,
        "position": _player_pos(player) if player is not None else (entry.get("position") if isinstance(entry, dict) else None),
        "overall": round(_player_ovr(player)) if player is not None else None,
        "contract_status": "rfa_rights",
        "expiry_status": "RFA",
        "previous_aav_m": entry.get("previous_aav_m") if isinstance(entry, dict) else None,
        "qualifying_offer_aav_m": entry.get("qualifying_offer_aav_m") if isinstance(entry, dict) else None,
    }
    remove_rfa_rights(team, player_id)
    if player is not None:
        fa_pool = list(_get(league, "free_agents", None) or [])
        if player not in fa_pool:
            fa_pool.append(player)
            league.free_agents = fa_pool
        try:
            player.rights_status = "UFA"
        except Exception:
            pass
    if session is not None:
        try:
            from services.franchise_offseason import upsert_resign_phase_outcome

            upsert_resign_phase_outcome(
                session,
                player_id=str(player_id),
                phase_status="released",
                snapshot_row=snapshot,
                name=name,
                reason="walked_away",
            )
        except Exception:
            pass
    return {"ok": True, "player_id": player_id, "status": "released"}


def execute_buyout(team: Any, player: Any, league: Any, season_year: int) -> Dict[str, Any]:
    ok, reason = can_buyout_player(player)
    if not ok:
        return {"ok": False, "reason": reason}
    est = estimate_buyout(player)
    aav = player_cap_hit_millions(player)

    buyouts = _get(team, "buyout_cap_hits", None)
    if not isinstance(buyouts, list):
        buyouts = []
    for i in range(int(est["years"])):
        yr = int(season_year) + i
        label = f"{yr}-{(yr + 1) % 100:02d}"
        buyouts.append({
            "season": label,
            "amount_m": est["annual_penalty_m"],
            "player_id": _player_id(player),
            "player_name": _player_name(player),
        })
    _set(team, "buyout_cap_hits", buyouts)

    roster = list(_get(team, "roster", None) or [])
    if player in roster:
        roster.remove(player)
        _set(team, "roster", roster)

    fa_pool = list(_get(league, "free_agents", None) or [])
    if player not in fa_pool:
        fa_pool.append(player)
        league.free_agents = fa_pool

    try:
        player.contract = None
        player.cap_hit_m = 0
    except Exception:
        pass

    sync_team_cap_fields(team, league)
    return {"ok": True, "buyout": est, "saved_aav_m": aav}


def execute_waive(team: Any, player: Any, league: Any, *, season_year: Optional[int] = None) -> Dict[str, Any]:
    return expose_player_to_waivers(team, player, league, reason="manual_waive", season_year=season_year)


def execute_bury(team: Any, player: Any, league: Any) -> Dict[str, Any]:
    if is_waiver_required_for_assignment(player, "nhl", "minors", league):
        cleared = str(_get(player, "waiver_status", "") or "").lower() == "cleared"
        if not cleared:
            return {"ok": False, "reason": "waiver_required", "requires_waivers": True}
    return bury_player_contract(team, player, league, skip_waiver_check=True)


def sign_elc(
    player: Any,
    team: Any,
    league: Any,
    season_year: int,
    *,
    promote_to_nhl: bool = False,
    offer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not is_elc_eligible(player):
        if has_true_elc_contract(player):
            return {"ok": True, "contract": normalize_contract_dict(_get(player, "contract", None) or {}), "already_signed": True}
        return {"ok": False, "reason": "Player is not ELC eligible"}

    result = assign_elc_contract(player, team, league, season_year, offer=offer)
    if not result.get("ok"):
        return result

    # Keep signed prospects on reserve list unless promoted to NHL.
    if promote_to_nhl:
        remove_from_reserve_list(team, _player_id(player))
        roster = list(_get(team, "roster", None) or [])
        if player not in roster:
            snap = get_team_cap_snapshot_full(team, league, season_year=season_year)
            if int(snap.get("active_roster_count", 0)) >= int(snap.get("active_roster_max", 23)):
                return {"ok": False, "reason": "ELC signed but active roster is full", "contract": result.get("contract")}
            pool = list(_get(team, "prospect_pool", None) or [])
            if player in pool:
                pool.remove(player)
                team.prospect_pool = pool
            roster.append(player)
            team.roster = roster
            try:
                player.organizational_status = "signed_nhl"
            except Exception:
                pass

    sync_team_cap_fields(team, league)
    return {"ok": True, "contract": result.get("contract")}


def execute_offer_sheet(
    offering_team: Any,
    rights_team: Any,
    player: Any,
    league: Any,
    season_year: int,
    offer: Dict[str, Any],
    *,
    session: Any = None,
) -> Dict[str, Any]:
    aav_m = normalize_money_m(offer.get("aav_m") or 0)
    years = max(1, int(offer.get("years") or 1))
    entry = find_rfa_rights(rights_team, _player_id(player))
    if not entry:
        return {"ok": False, "reason": "RFA rights not found"}
    if not rfa_offer_sheet_eligible(player, entry):
        return {
            "ok": False,
            "reason": (
                f"Player not offer-sheet eligible (age ≥ 20 and prior AAV < "
                f"${OFFER_SHEET_ELIGIBLE_AAV_CEILING_M:.3f}M required)"
            ),
        }

    cap_hit = compute_prorated_cap_hit_m(aav_m, years, float(offer.get("signing_bonus_m") or 0))
    check = _validate_sign_cap(offering_team, cap_hit, league)
    if not check.get("ok"):
        return {"ok": False, "reason": check.get("reason")}

    tier_info = offer_sheet_compensation_tier(aav_m)
    filed_day = int(
        offer.get("filed_day")
        or getattr(session, "fa_market_day", None)
        or getattr(session, "calendar_days_finished", 0)
        or 0
    )
    sheet_id = str(offer.get("offer_sheet_id") or f"os-{ _player_id(player) }-{filed_day}")
    pending = {
        "offer_sheet_id": sheet_id,
        "player_id": _player_id(player),
        "player_name": _player_name(player),
        "offering_team_id": str(_get(offering_team, "team_id", "") or _get(offering_team, "id", "")),
        "rights_team_id": str(_get(rights_team, "team_id", "") or _get(rights_team, "id", "")),
        "aav_m": aav_m,
        "years": years,
        "compensation_tier": tier_info["tier"],
        "compensation_rounds": list(tier_info.get("rounds") or []),
        "compensation_label": tier_info.get("label"),
        "match_deadline_days": OFFER_SHEET_MATCH_WINDOW_DAYS,
        "filed_day": filed_day,
        "expires_day": filed_day + OFFER_SHEET_MATCH_WINDOW_DAYS,
        "status": "pending",
    }
    entry["offer_sheet_pending"] = dict(pending)
    entry["offer_sheet_eligible"] = True
    sheets = list(_get(league, "pending_offer_sheets", None) or [])
    sheets.append(pending)
    league.pending_offer_sheets = sheets
    if session is not None:
        off_name = str(_get(offering_team, "name", "") or _get(offering_team, "city", "") or "A club")
        _emit_contract_storyline(
            session,
            f"{off_name} files offer sheet on {_player_name(player)} — {years}y/${aav_m:.2f}M AAV",
            kind="offer_sheet",
            severity="high",
        )
    return {"ok": True, "offer_sheet": pending}


def compute_arbitration_award(
    player: Any,
    team: Any,
    league: Any,
    *,
    team_offer_m: float,
    player_ask_m: float,
    season_year: int,
) -> Tuple[float, int]:
    """Arbitration award: production×0.35 + market×0.30 + age_fit×0.20 + cap_situation×0.15.

    Result is clamped between team_offer and player_ask. Younger players on reasonable
    awards may receive two-year deals; older / rich awards stay one year.
    """
    team_offer = float(team_offer_m or LEAGUE_MINIMUM_AAV_M)
    player_ask = float(player_ask_m or team_offer)
    lo, hi = (team_offer, player_ask) if player_ask >= team_offer else (player_ask, team_offer)
    market = compute_market_value(player, league)
    prod = _player_production_score(player)
    age = _player_age(player)
    age_fit = max(0.0, min(1.0, 1.0 - abs(age - 26) / 12.0))
    try:
        snap = get_team_cap_snapshot_full(team, league)
        usable = float(snap.get("usable_cap_space_m", 0) or 0)
        cap_situation = max(0.0, min(1.0, usable / max(1.0, hi)))
    except Exception:
        cap_situation = 0.5
    midpoint = (lo + hi) / 2.0
    base = (
        0.35 * (lo + (hi - lo) * prod)
        + 0.30 * max(lo, min(hi, market))
        + 0.20 * (lo + (hi - lo) * age_fit)
        + 0.15 * (lo + (hi - lo) * cap_situation)
    )
    var = random.Random(
        abs(hash(("arb", _player_id(player), int(season_year)))) & 0xFFFFFFFF
    ).uniform(-0.03, 0.03)
    award = round(min(hi, max(lo, base * (1.0 + var))), 3)
    years = 2 if (age <= 27 and award <= market * 1.1) else 1
    return award, years


def execute_arbitration_file(team: Any, player_id: str, player_ask_m: float) -> Dict[str, Any]:
    entry = find_rfa_rights(team, player_id)
    if not entry or not entry.get("arbitration_eligible"):
        return {"ok": False, "reason": "Not arbitration eligible"}
    entry["arbitration_filed"] = True
    entry["player_ask_m"] = round(normalize_money_m(player_ask_m), 3)
    entry["team_offer_m"] = round(float(entry.get("qualifying_offer_aav_m") or LEAGUE_MINIMUM_AAV_M), 3)
    return {"ok": True, "arbitration": {k: entry.get(k) for k in ("player_id", "team_offer_m", "player_ask_m", "arbitration_filed")}}


def execute_arbitration_settle(team: Any, player_id: str, league: Any, season_year: int) -> Dict[str, Any]:
    entry = find_rfa_rights(team, player_id)
    if not entry or not entry.get("arbitration_filed"):
        return {"ok": False, "reason": "Arbitration not filed"}
    player = resolve_rfa_player(entry, league)
    if player is None:
        return {"ok": False, "reason": "Player missing"}

    team_offer = float(entry.get("team_offer_m") or LEAGUE_MINIMUM_AAV_M)
    player_ask = float(entry.get("player_ask_m") or team_offer)
    award, years = compute_arbitration_award(
        player, team, league,
        team_offer_m=team_offer,
        player_ask_m=player_ask,
        season_year=season_year,
    )
    entry["award_aav_m"] = award
    entry["award_years"] = years

    contract = normalize_contract_dict({
        "type": "STANDARD", "years": years, "years_remaining": years,
        "aav_m": award, "cap_hit_m": award, "rights_status": "RFA",
        "expiry_year": int(season_year) + years, "source": "arbitration",
    })
    apply_contract_to_player(player, contract, season_year)
    remove_rfa_rights(team, player_id)
    roster = list(_get(team, "roster", None) or [])
    if player not in roster:
        roster.append(player)
        team.roster = roster
    sync_team_cap_fields(team, league)
    return {"ok": True, "award_aav_m": award, "award_years": years}


# ---------------------------------------------------------------------------
# Contract expiry / tick
# ---------------------------------------------------------------------------

def handle_player_contract_expiry(
    player: Any,
    team: Any,
    league: Any,
    season_year: int,
    *,
    defer_july1_ufa: bool = False,
    force_expire: bool = False,
) -> str:
    """Returns 'kept', 'ufa', or 'rfa_rights'.

    When ``defer_july1_ufa`` is set (salary-cap stage), final-year UFAs stay on the
    roster with ``pending_july1_expiry`` so clubs can still negotiate extensions
    through the re-sign desk until Free Agency / July 1. RFAs still expire into
    rights immediately so qualifying offers can run. Call again with
    ``force_expire=True`` (or ``expire_pending_july1_contracts``) to burn them.
    """
    c = _get(player, "contract", None)
    if c is None and not has_active_contract(player):
        return "kept"

    if force_expire:
        if isinstance(c, dict):
            try:
                c["years_remaining"] = 0
                c.pop("pending_july1_expiry", None)
            except Exception:
                pass
        elif c is not None:
            try:
                if hasattr(c, "years_remaining"):
                    c.years_remaining = 0
            except Exception:
                pass
        try:
            setattr(player, "pending_july1_expiry", False)
        except Exception:
            pass
    else:
        yrs_before = _contract_years_remaining(player)
        norm_peek = normalize_contract_payload(player)
        rights_peek = str(
            norm_peek.get("rights_status") or _get(player, "rights_status", "UFA") or "UFA"
        ).upper()
        ctype_peek = str(norm_peek.get("type") or norm_peek.get("contract_type") or "").upper()
        is_rfa = "RFA" in rights_peek or ctype_peek == "RFA_BRIDGE"

        # Keep final-year UFAs signed until July 1 / Free Agency for extensions.
        if defer_july1_ufa and yrs_before == 1 and not is_rfa:
            if isinstance(c, dict):
                try:
                    c["pending_july1_expiry"] = True
                    c["years_remaining"] = 1
                except Exception:
                    pass
            try:
                setattr(player, "pending_july1_expiry", True)
            except Exception:
                pass
            return "kept"

        if c is not None and hasattr(c, "tick_year"):
            try:
                c.tick_year()
            except Exception:
                pass
        elif isinstance(c, dict):
            try:
                c["years_remaining"] = max(0, int(c.get("years_remaining", 0)) - 1)
            except (TypeError, ValueError):
                c["years_remaining"] = 0

    yrs = _contract_years_remaining(player)
    if yrs > 0:
        return "kept"

    norm = normalize_contract_payload(player)
    rights = str(norm.get("rights_status") or _get(player, "rights_status", "UFA") or "UFA").upper()
    ctype = str(norm.get("type") or norm.get("contract_type") or "").upper()
    if "RFA" in rights or ctype == "RFA_BRIDGE":
        add_rfa_rights(team, player, season_year, league)
        _clear_expired_contract(player)
        try:
            setattr(player, "pending_july1_expiry", False)
        except Exception:
            pass
        return "rfa_rights"

    fa_pool = list(_get(league, "free_agents", None) or [])
    if player not in fa_pool:
        fa_pool.append(player)
        league.free_agents = fa_pool
    _clear_expired_contract(player)
    try:
        player.rights_status = "UFA"
    except Exception:
        pass
    try:
        setattr(player, "pending_july1_expiry", False)
    except Exception:
        pass
    # Home-team exclusive window: former club gets first crack on the re-sign desk
    # before open free agency. Tag so the desk / market can find them.
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    try:
        if tid:
            player.ufa_from_team_id = tid
            player.previous_nhl_team_id = tid
            player.ufa_exclusive = True
    except Exception:
        pass
    return "ufa"


def expire_pending_july1_contracts(session: Any) -> Dict[str, Any]:
    """Burn deferred final-year UFAs when Free Agency / July 1 opens."""
    if bool(getattr(session, "july1_contracts_expired", False)):
        return {
            "expired_ufas": [],
            "expired_rfas": [],
            "skipped": True,
            "reason": "july1_already_expired",
        }

    sim = getattr(session, "sim", None)
    league = getattr(sim, "league", None) if sim is not None else None
    if league is None:
        return {"expired_ufas": [], "expired_rfas": [], "ok": False, "reason": "no_league"}

    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    expired_ufas: List[Dict[str, Any]] = []
    expired_rfas: List[Dict[str, Any]] = []

    def _is_pending(player: Any) -> bool:
        if bool(getattr(player, "pending_july1_expiry", False)):
            return True
        c = _get(player, "contract", None)
        return bool(isinstance(c, dict) and c.get("pending_july1_expiry"))

    for team in list(getattr(league, "teams", None) or []):
        for attr in ("roster", "ahl_roster", "echl_roster"):
            roster = list(getattr(team, attr, None) or [])
            if not roster:
                continue
            kept: List[Any] = []
            for p in roster:
                if getattr(p, "retired", False):
                    continue
                if not _is_pending(p):
                    kept.append(p)
                    continue
                outcome = handle_player_contract_expiry(
                    p, team, league, season_year, force_expire=True
                )
                if outcome == "kept":
                    kept.append(p)
                    continue
                row = {
                    "player_id": _player_id(p),
                    "name": _player_name(p),
                    "team_id": str(_get(team, "team_id", "") or ""),
                    "outcome": outcome,
                }
                if outcome == "rfa_rights":
                    expired_rfas.append(row)
                else:
                    expired_ufas.append(row)
            setattr(team, attr, kept)

    # Refresh cached cap mirrors league-wide now that deferred UFAs/RFAs are
    # actually off the books, so usable_cap_space_m (and any code still
    # reading the team.cap_space / cap_snapshot mirrors) reflects the freed
    # room immediately rather than a stale pre-July-1 snapshot.
    for team in list(getattr(league, "teams", None) or []):
        try:
            sync_team_cap_fields(team, league, sim, season_year=season_year)
        except Exception:
            continue

    try:
        session.july1_contracts_expired = True
    except Exception:
        pass
    return {
        "expired_ufas": expired_ufas,
        "expired_rfas": expired_rfas,
        "expired_ufa_count": len(expired_ufas),
        "expired_rfa_count": len(expired_rfas),
    }


# ---------------------------------------------------------------------------
# Bootstrap cap compliance (creation only)
# ---------------------------------------------------------------------------

def _protected_core_ids(roster: List[Any]) -> set:
    if not roster:
        return set()
    by_ovr = sorted(roster, key=lambda p: -_player_ovr(p))
    protected = {id(p) for p in by_ovr[:CAP_SAFE_CORE_TOP_N]}
    for p in roster:
        if _player_ovr(p) >= CAP_SAFE_STAR_OVR:
            protected.add(id(p))
    return protected


def _record_bootstrap_trim(
    league: Any,
    team_id: str,
    player_id: str,
    old_aav_m: float,
    new_aav_m: float,
    reason: str,
) -> None:
    log = _get(league, "_bootstrap_cap_trim_log", None)
    if not isinstance(log, list):
        try:
            league._bootstrap_cap_trim_log = []
        except Exception:
            return
        log = getattr(league, "_bootstrap_cap_trim_log", [])
    log.append({
        "team_id": team_id,
        "player_id": player_id,
        "old_aav_m": round(float(old_aav_m), 3),
        "new_aav_m": round(float(new_aav_m), 3),
        "reduction_m": round(max(0.0, float(old_aav_m) - float(new_aav_m)), 3),
        "reason": reason,
        "true_elc_skipped": False,
    })


def fix_league_contract_truth(league: Any) -> Dict[str, int]:
    """Normalize and repair contract labels across the league."""
    fixed = 0
    stripped = 0
    for team in _get(league, "teams", None) or []:
        for p in list(_all_rostered(team)) + list(_get(team, "prospect_pool", None) or []):
            before_type = str(_contract_type(p) or "").upper()
            hydrate_player_contract(p)
            if _get(p, "contract", None) is not None and not has_active_contract(p):
                _strip_malformed_contract(p)
                stripped += 1
            after_type = str(_contract_type(p) or "").upper()
            if before_type == "ELC" and after_type != "ELC":
                fixed += 1
    return {"fake_elc_relabeled": fixed, "malformed_stripped": stripped}


def rebalance_team_cap_at_bootstrap(
    team: Any,
    league: Any,
    season_year: int,
    rng: random.Random,
) -> bool:
    snap = get_team_cap_snapshot_full(team, league, season_year=season_year)
    cap_m = snap["upper_limit_m"]
    if snap["usable_cap_space_m"] >= -1e-6:
        return False

    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    roster = _active_roster(team)
    protected = _protected_core_ids(roster)
    changed = False

    def payroll() -> float:
        return get_team_cap_snapshot_full(team, league, season_year=season_year)["total_cap_hit_m"]

    over = payroll() - cap_m

    try:
        from services.real_nhl_contracts import is_real_nhl_contract
    except Exception:
        def is_real_nhl_contract(_p: Any) -> bool:  # type: ignore
            return False

    if getattr(league, "real_nhl_import_meta", None):
        # Preserve authentic Spotrac / R4 AAVs for Real NHL franchises.
        sync_team_cap_fields(team, league, season_year=season_year)
        return False

    trim_order = sorted(roster, key=lambda p: (_player_ovr(p), -_player_age(p)))
    for p in trim_order:
        if over <= 1e-6:
            break
        if id(p) in protected:
            continue
        if has_true_elc_contract(p) or is_real_nhl_contract(p):
            continue
        cur = player_cap_hit_millions(p)
        if cur <= LEAGUE_MINIMUM_AAV_M + 1e-6:
            continue
        cut = min(cur - LEAGUE_MINIMUM_AAV_M, over * rng.uniform(0.55, 0.95))
        new_aav = max(LEAGUE_MINIMUM_AAV_M, round(cur - cut, 3))
        c = normalize_contract_dict(_get(p, "contract", None) or {})
        c["aav_m"] = c["cap_hit_m"] = new_aav
        apply_contract_to_player(p, c, season_year)
        _record_bootstrap_trim(league, tid, _player_id(p), cur, new_aav, "over_cap_trim")
        changed = True
        over = payroll() - cap_m

    for p in roster:
        if id(p) in protected:
            continue
        if has_true_elc_contract(p) or is_real_nhl_contract(p):
            continue
        c = normalize_contract_dict(_get(p, "contract", None) or {})
        if c.get("bad_contract_type") and payroll() > cap_m:
            cur = player_cap_hit_millions(p)
            new_aav = max(LEAGUE_MINIMUM_AAV_M, round(cur * 0.92, 3))
            if new_aav < cur:
                c["aav_m"] = c["cap_hit_m"] = new_aav
                apply_contract_to_player(p, c, season_year)
                _record_bootstrap_trim(league, tid, _player_id(p), cur, new_aav, "bad_contract_trim")
                changed = True

    if payroll() > cap_m:
        non_core = [
            p for p in roster
            if id(p) not in protected
            and not has_true_elc_contract(p)
            and not is_real_nhl_contract(p)
        ]
        core_pay = sum(player_cap_hit_millions(p) for p in roster if id(p) in protected)
        budget = max(0.0, cap_m - core_pay)
        rest = sum(player_cap_hit_millions(p) for p in non_core)
        if non_core and rest > budget and rest > 0:
            factor = max(0.35, budget / rest)
            for p in non_core:
                if has_true_elc_contract(p) or is_real_nhl_contract(p):
                    continue
                cur = player_cap_hit_millions(p)
                new_aav = max(LEAGUE_MINIMUM_AAV_M, round(cur * factor, 3))
                c = normalize_contract_dict(_get(p, "contract", None) or {})
                c["aav_m"] = c["cap_hit_m"] = new_aav
                apply_contract_to_player(p, c, season_year)
                _record_bootstrap_trim(league, tid, _player_id(p), cur, new_aav, "proportional_trim")
                changed = True

    sync_team_cap_fields(team, league, season_year=season_year)
    return changed


def validate_franchise_cap_at_start(league: Any, season_year: int) -> List[str]:
    issues = []
    for team in _get(league, "teams", None) or []:
        snap = get_team_cap_snapshot_full(team, league, season_year=season_year)
        if snap["usable_cap_space_m"] < -0.01:
            issues.append(f"Team {_get(team, 'team_id', '?')} over cap: {snap['usable_cap_space_m']}")
    return issues


# ---------------------------------------------------------------------------
# CPU free agency
# ---------------------------------------------------------------------------

def _cpu_team_revenue_m(team: Any, session: Any = None) -> float:
    try:
        rev = float(getattr(team, "revenue_m", None) or getattr(team, "annual_revenue_m", None) or 0.0)
        if rev > 0:
            return rev
    except Exception:
        pass
    if session is not None:
        try:
            from services.league_operations import calculate_team_revenue

            tid = str(_get(team, "team_id", "") or _get(team, "id", ""))
            row = calculate_team_revenue(session, team, tid, is_user=False)
            return float(row.get("revenue") or row.get("revenue_m") or 0.0)
        except Exception:
            pass
    return 0.0


def _cpu_bonus_usage_rate(revenue_m: float) -> float:
    """Probability CPU proposes signing bonus on high-revenue clubs."""
    rev = float(revenue_m or 0.0)
    if rev >= 230:
        return 0.25
    if rev >= 210:
        return 0.12
    if rev >= 190:
        return 0.02
    return 0.0


def _cpu_negotiate_offer(
    team: Any,
    player: Any,
    league: Any,
    start_aav: float,
    years: int,
    ctx: Dict[str, Any],
    *,
    context: str = "ufa",
    max_rounds: int = 3,
    ceiling: Optional[float] = None,
    session: Any = None,
) -> Tuple[bool, float, int]:
    """Negotiate a CPU offer through the SAME acceptance system the user faces
    (item 1). Players can refuse; the team responds with limited counter rounds,
    chasing the player's counter only within a sane budget ceiling. Returns
    (agreed, final_aav, final_years). Ceiling is compared against prorated cap hit."""
    aav = float(start_aav)
    ntc_mode = "NONE"
    nmc = False
    signing_bonus_m = 0.0
    rng = random.Random(abs(hash(("cpu_nego", _player_id(player), _get(team, "team_id", "")))) & 0xFFFFFFFF)

    if ceiling is None:
        space = max(
            0.0,
            float(ctx.get("spendable_cap_space_m", ctx.get("cap_space_m", 0)) or 0),
        )
        if str(context or "").lower() in ("re_sign", "rfa", "extension"):
            space = max(space, float(ctx.get("cap_space_m", 0) or 0))
        market = float(compute_market_value(player, league) or LEAGUE_MINIMUM_AAV_M)
        ovr = float(_player_ovr(player))
        space_frac = 0.98 if ovr >= 88 else (0.90 if ovr >= 82 else 0.75)
        ceiling = min(space * space_frac, max(market * 1.22, float(start_aav) * 1.12))
        ceiling = max(ceiling, min(space * 0.99, LEAGUE_MINIMUM_AAV_M))

    def _cap_hit(a: float, y: int, b: float) -> float:
        return compute_prorated_cap_hit_m(a, y, b)

    revenue_m = _cpu_team_revenue_m(team, session)
    bonus_rate = _cpu_bonus_usage_rate(revenue_m)
    if bonus_rate > 0 and rng.random() < bonus_rate:
        try:
            from services.franchise_offseason import signing_bonus_max_pct_for_revenue

            max_pct = signing_bonus_max_pct_for_revenue(revenue_m)
            total = max(aav * years, 0.25)
            signing_bonus_m = round(min(total * max_pct * 0.45, total * max_pct), 3)
            if _cap_hit(aav, years, signing_bonus_m) > ceiling + 1e-6:
                signing_bonus_m = 0.0
        except Exception:
            signing_bonus_m = 0.0

    rounds = max(1, int(max_rounds))
    if float(_player_ovr(player)) >= 88:
        rounds = max(rounds, 5)
    days_on_market = int(ctx.get("days_on_market") or getattr(player, "days_on_market", 0) or 0)
    offer_count = int(ctx.get("offer_count") or getattr(player, "fa_offer_count", 0) or 0)

    for _ in range(rounds):
        offer_payload = {
            "aav_m": aav,
            "years": years,
            "context": context,
            "ntc_mode": ntc_mode,
            "ntc": ntc_mode in ("FULL", "MODIFIED"),
            "nmc": nmc,
            "signing_bonus_m": signing_bonus_m,
            "days_on_market": days_on_market,
            "offer_count": offer_count,
        }
        ev = evaluate_contract_offer(player, team, offer_payload, league, context=context)
        if ev.get("accepted"):
            return True, round(aav, 3), years
        co = ev.get("counter_offer") or {}
        nxt_aav = float(co.get("aav_m") or aav)
        nxt_years = int(co.get("years") or years)
        co_mode = str(co.get("ntc_mode") or "").upper()
        if co_mode in ("FULL", "MODIFIED", "MNTC", "M-NTC"):
            ntc_mode = "MODIFIED" if co_mode in ("MODIFIED", "MNTC", "M-NTC") else "FULL"
        elif co.get("ntc") or co.get("no_trade_clause"):
            ntc_mode = "FULL"
        if co.get("nmc") or co.get("no_move_clause"):
            nmc = True
            ntc_mode = "NONE"
        try:
            signing_bonus_m = max(signing_bonus_m, float(co.get("signing_bonus_m") or 0))
        except Exception:
            pass
        nxt_cap = _cap_hit(nxt_aav, nxt_years, signing_bonus_m)
        space = max(0.0, float(ctx.get("cap_space_m", 0) or 0))
        if nxt_cap > ceiling + 1e-6:
            if nxt_years > years:
                years = min(nxt_years, years + 1)
            if _cap_hit(aav, years, signing_bonus_m) < ceiling:
                aav = min(ceiling, max(aav, nxt_aav * 0.98))
                if _cap_hit(aav, years, signing_bonus_m) <= ceiling:
                    continue
            if nxt_cap <= space * 0.99:
                aav = nxt_aav
                years = max(years, nxt_years)
                ceiling = max(ceiling, nxt_cap)
                continue
            return False, round(aav, 3), years
        aav = nxt_aav
        years = max(years, min(nxt_years, years + 1))

    offer_payload = {
        "aav_m": aav,
        "years": years,
        "context": context,
        "ntc_mode": ntc_mode,
        "ntc": ntc_mode in ("FULL", "MODIFIED"),
        "nmc": nmc,
        "signing_bonus_m": signing_bonus_m,
        "days_on_market": days_on_market,
        "offer_count": offer_count,
    }
    ev = evaluate_contract_offer(player, team, offer_payload, league, context=context)
    return bool(ev.get("accepted")), round(aav, 3), years


def run_cpu_prospect_rights_pass(session: Any) -> Dict[str, Any]:
    """CPU teams decide ELC vs keep-unsigned for high-priority prospects (idempotent)."""
    sim = session.sim
    league = getattr(sim, "league", None)
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    actions: List[Dict[str, Any]] = []
    if league is None:
        return {"actions": [], "count": 0}

    profiles = getattr(session, "cpu_franchise_profiles", None)
    if not isinstance(profiles, dict):
        session.cpu_franchise_profiles = {}
        profiles = session.cpu_franchise_profiles

    for team in list(getattr(league, "teams", None) or []):
        tid = str(_get(team, "team_id", "") or _get(team, "id", ""))
        if not tid or tid == user_tid:
            continue
        prof = profiles.get(tid) if isinstance(profiles.get(tid), dict) else {}
        if prof.get("prospect_rights_complete"):
            continue
        slots = validate_contract_slots(team, league, additional=0)
        open_slots = max(0, CONTRACT_SLOTS_LIMIT - int(slots.get("contract_slots_used") or 0))
        signed_here = 0
        for p in list(getattr(team, "prospect_pool", None) or []):
            if str(getattr(p, "signed_status", "unsigned") or "").lower() == "signed":
                continue
            if not bool(getattr(p, "entry_level_contract_eligible", True)):
                continue
            age = _player_age(p)
            ovr = _player_ovr(p)
            expiry = getattr(p, "rights_expiry_year", None)
            urgent = expiry is not None and int(expiry or 9999) <= season_year + 1
            # Contenders / urgent rights / NHL-ready tools → sign a small subset.
            should_sign = False
            if open_slots <= 0 or signed_here >= 2:
                should_sign = False
            elif urgent and ovr >= 58:
                should_sign = True
            elif age >= 20 and ovr >= 64:
                should_sign = True
            elif age <= 18 and ovr < 62:
                should_sign = False
            if should_sign:
                try:
                    from services.elc_offer_engine import submit_elc_offer

                    res = submit_elc_offer(
                        session,
                        p,
                        team,
                        season_year=season_year,
                        template_id="standard_elc" if ovr < 68 else "maximum_bonus_elc",
                        assignment_plan="return_junior" if age < 20 else "assign_ahl",
                    )
                except Exception:
                    res = sign_elc(p, team, league, season_year, promote_to_nhl=False)
                if res.get("ok") and res.get("signed") and not res.get("already_signed"):
                    signed_here += 1
                    open_slots = max(0, open_slots - 1)
                    actions.append({
                        "team_id": tid,
                        "player_id": _player_id(p),
                        "action": "sign_elc",
                        "template": "standard_elc" if ovr < 68 else "maximum_bonus_elc",
                    })
                elif res.get("reason") == "prospect_declined":
                    actions.append({
                        "team_id": tid,
                        "player_id": _player_id(p),
                        "action": "elc_declined",
                    })
        prof = dict(prof)
        prof["prospect_rights_complete"] = True
        profiles[tid] = prof

    session.cpu_franchise_profiles = profiles
    return {"actions": actions, "count": len(actions)}


def run_cpu_free_agency(session: Any, max_signings: int = 40) -> Dict[str, Any]:
    sim = session.sim
    league = getattr(sim, "league", None)
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    rng = sim.rng
    fa_pool = list(_get(league, "free_agents", None) or [])
    signings: List[Dict[str, Any]] = []
    post_fa_issues: List[Dict[str, Any]] = []

    teams = list(_get(league, "teams", None) or [])
    user_tid = str(getattr(session, "user_team_id", "") or "")
    cpu_teams = [
        t for t in teams
        if str(_get(t, "team_id", "") or _get(t, "id", "")) != user_tid
    ]
    rng.shuffle(cpu_teams)

    per_team_cap = max(1, min(4, max_signings // max(1, len(cpu_teams))))
    team_sign_counts: Dict[str, int] = {}
    team_goalie_signed: Dict[str, bool] = {}

    while len(signings) < max_signings and fa_pool:
        made_signing = False
        for team in cpu_teams:
            if len(signings) >= max_signings:
                break
            tid = str(_get(team, "team_id", "") or _get(team, "id", ""))
            if team_sign_counts.get(tid, 0) >= per_team_cap:
                continue

            ctx = evaluate_team_position_needs(team, league, sim, season_year=season_year)
            if ctx["cap_space_m"] < LEAGUE_MINIMUM_AAV_M * 2:
                continue
            if ctx["slots_remaining"] <= 0:
                continue

            window = ctx["window"]
            scored: List[Tuple[float, Any, float, int, List[str]]] = []

            for player in fa_pool:
                if player not in fa_pool:
                    continue
                pos = _position_bucket(player)
                if pos == "G" and team_goalie_signed.get(tid):
                    continue

                fair = compute_fair_aav(player, team, league)
                need = float(ctx["need_score"].get(pos, 0))

                discount = 0.90 if window == "rebuilder" else (1.02 if need >= 0.55 else 0.96)
                if window == "cap_strapped":
                    discount = 0.88
                # Empty net: pay fair value — do not lowball the only available goalie.
                if pos == "G" and int(ctx["counts"].get("G", 0) or 0) == 0:
                    discount = max(discount, 1.06)
                offer_aav = round(
                    min(fair * rng.uniform(0.90, 1.06) * discount, ctx["cap_space_m"] * 0.35),
                    3,
                )
                if pos == "G" and int(ctx["counts"].get("G", 0) or 0) == 0:
                    offer_aav = round(
                        min(max(offer_aav, fair * 0.98), max(0.0, float(ctx["cap_space_m"])) * 0.95),
                        3,
                    )
                if offer_aav < LEAGUE_MINIMUM_AAV_M:
                    continue

                blocked = cpu_signing_blocked(team, player, ctx, offer_aav)
                if blocked:
                    continue

                _, years, _ = generate_contract_terms(player, team, league, rng, context="ufa")
                if window == "rebuilder":
                    years = min(years, 3)
                elif window == "cap_strapped":
                    years = min(years, 2)
                if window == "contender" and _player_age(player) > 34:
                    years = min(years, 2)

                fit, reasons = score_free_agent_fit(team, player, ctx, offer_aav, years, league)
                if fit < CPU_SIGN_MIN_FIT_SCORE:
                    continue
                scored.append((fit, player, offer_aav, years, reasons))

            if not scored:
                continue

            scored.sort(key=lambda row: -row[0])
            # Emergency: never let a 0-goalie club skip the goalie because a winger
            # happened to clear negotiation first.
            if int(ctx["counts"].get("G", 0) or 0) == 0:
                scored.sort(
                    key=lambda row: (
                        0 if _position_bucket(row[1]) == "G" else 1,
                        -row[0],
                    )
                )

            # Work down the best-fit targets: the player must actually ACCEPT
            # (through negotiation), so a team can be turned down and move on to
            # the next-best fit instead of force-signing its favourite (item 1).
            signed_player = None
            for fit, player, offer_aav, years, reasons in scored[:5]:
                rounds = 5 if _position_bucket(player) == "G" and int(ctx["counts"].get("G", 0) or 0) == 0 else 3
                agreed, final_aav, final_years = _cpu_negotiate_offer(
                    team, player, league, offer_aav, years, ctx, context="ufa",
                    max_rounds=rounds,
                )
                if not agreed:
                    continue
                result = sign_player_to_team(
                    player,
                    team,
                    league,
                    season_year,
                    {
                        "aav_m": final_aav,
                        "years": final_years,
                        "context": "ufa",
                        "force": True,
                    },
                )
                if not result.get("ok"):
                    continue
                signed_player = (fit, player, final_aav, final_years, reasons)
                break

            if signed_player is None:
                continue
            fit, player, offer_aav, years, reasons = signed_player

            if player in fa_pool:
                fa_pool.remove(player)

            ovr = _player_ovr(player)
            pos = _position_bucket(player)
            signings.append({
                "team_id": tid,
                "player_id": _player_id(player),
                "aav_m": offer_aav,
                "years": years,
                "position": pos,
                "overall": round(ovr),
                "ovr_band": ovr_band(ovr),
                "fit_score": round(fit, 3),
                "window": window,
                "fit_reasons": reasons[:4],
            })
            team_sign_counts[tid] = team_sign_counts.get(tid, 0) + 1
            if pos == "G":
                team_goalie_signed[tid] = True
            made_signing = True

        if not made_signing:
            break

    league.free_agents = fa_pool

    for team in cpu_teams:
        post_fa_issues.extend(validate_post_fa_roster_shape(team, league, sim))

    return {
        "signings": signings,
        "count": len(signings),
        "post_fa_issues": post_fa_issues,
    }


def run_cpu_in_season_free_agency(
    session: Any,
    *,
    max_signings: int = 2,
    min_fit: float = 0.55,
    cooldown_days: int = 20,
) -> Dict[str, Any]:
    """Controlled in-season CPU free-agent activity at a transaction checkpoint.

    Reuses the same need/fit/negotiation path as the offseason market but with tight
    guardrails so the CPU reacts to genuine holes without flooding the league:
      * at most `max_signings` league-wide per checkpoint,
      * a per-team cooldown so a team cannot sign repeatedly on consecutive checkpoints,
      * only when a team has a real positional need (high need score or an unfilled hole),
      * players must ACCEPT through negotiation — no force-sign of a rejected offer.
    """
    sim = session.sim
    league = getattr(sim, "league", None)
    if league is None:
        return {"signings": [], "count": 0}
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    rng = sim.rng
    day = int(getattr(session, "calendar_days_finished", 0) or 0)

    # Idempotency: never run the same season/day checkpoint twice (guards double-finalize
    # of a calendar day and save/reload re-advancing an already-processed day).
    marker = [int(season_year), int(day)]
    if list(getattr(session, "_last_cpu_fa_checkpoint", []) or []) == marker:
        return {"signings": [], "count": 0, "skipped": "already_processed"}

    cooldowns: Dict[str, int] = getattr(session, "_cpu_fa_cooldowns", None) or {}
    fa_pool = list(_get(league, "free_agents", None) or [])
    if not fa_pool:
        session._last_cpu_fa_checkpoint = marker
        return {"signings": [], "count": 0}

    user_tid = str(getattr(session, "user_team_id", "") or "")
    cpu_teams = [
        t for t in (_get(league, "teams", None) or [])
        if str(_get(t, "team_id", "") or _get(t, "id", "")) != user_tid
    ]
    rng.shuffle(cpu_teams)

    signings: List[Dict[str, Any]] = []
    for team in cpu_teams:
        if len(signings) >= max_signings:
            break
        tid = str(_get(team, "team_id", "") or _get(team, "id", ""))
        if day - int(cooldowns.get(tid, -9999)) < cooldown_days:
            continue

        ctx = evaluate_team_position_needs(team, league, sim, season_year=season_year)
        top_need = max(ctx["need_score"].values()) if ctx.get("need_score") else 0.0
        counts = ctx.get("counts") or {}
        emergency_pos: Optional[str] = None
        for pos in CPU_POSITIONS:
            n = int(counts.get(pos, 0) or 0)
            if n == 0:
                emergency_pos = pos
                break
            if pos == "G" and n == 1:
                emergency_pos = "G"
                break

        if ctx["cap_space_m"] < LEAGUE_MINIMUM_AAV_M * 2 or ctx["slots_remaining"] <= 0:
            continue
        if emergency_pos is None and top_need < 0.6:
            continue

        window = ctx["window"]
        scored: List[Tuple[float, Any, float, int, List[str]]] = []
        for player in fa_pool:
            pos = _position_bucket(player)
            if emergency_pos is None:
                if float(ctx["need_score"].get(pos, 0)) < 0.55:
                    continue
            elif pos != emergency_pos:
                continue
            fair = compute_fair_aav(player, team, league)
            offer_aav = round(min(fair * rng.uniform(0.92, 1.03), ctx["cap_space_m"] * 0.30), 3)
            if offer_aav < LEAGUE_MINIMUM_AAV_M:
                continue
            if cpu_signing_blocked(team, player, ctx, offer_aav) and emergency_pos is None:
                continue
            _, years, _ = generate_contract_terms(player, team, league, rng, context="ufa")
            fit, reasons = score_free_agent_fit(team, player, ctx, offer_aav, years, league)
            if fit < min_fit and emergency_pos is None:
                continue
            scored.append((fit, player, offer_aav, years, reasons))

        if not scored:
            continue
        scored.sort(key=lambda row: -row[0])

        for fit, player, offer_aav, years, reasons in scored[:4]:
            agreed, final_aav, final_years = _cpu_negotiate_offer(
                team, player, league, offer_aav, years, ctx, context="ufa", session=session,
            )
            if not agreed:
                continue
            result = sign_player_to_team(
                player, team, league, season_year,
                {"aav_m": final_aav, "years": final_years, "context": "ufa", "force": True},
            )
            if not result.get("ok"):
                continue
            if player in fa_pool:
                fa_pool.remove(player)
            cooldowns[tid] = day
            if emergency_pos is not None:
                team_label = str(_get(team, "name", "") or _get(team, "city", "") or tid)
                _emit_contract_storyline(
                    session,
                    f"Emergency signing: {team_label} adds {_player_name(player)} at {emergency_pos}",
                    kind="emergency_fa",
                    severity="high",
                )
            signings.append({
                "team_id": tid,
                "player_id": _player_id(player),
                "aav_m": final_aav,
                "years": final_years,
                "position": _position_bucket(player),
                "fit_score": round(fit, 3),
                "window": window,
                "fit_reasons": reasons[:3],
            })
            break

    session._cpu_fa_cooldowns = cooldowns
    session._last_cpu_fa_checkpoint = marker
    if signings:
        league.free_agents = fa_pool
    return {"signings": signings, "count": len(signings)}


def run_cpu_rfa_decisions(session: Any) -> Dict[str, Any]:
    """Resolve every CPU team's restricted free agents each offseason.

    Without this pass, expiring RFAs are stripped off CPU rosters into `rfa_rights`
    limbo and never come back — young, cost-controlled talent silently disappears
    season after season. Here each CPU team decides, through the SAME acceptance
    system the user faces, whether to re-sign each RFA (a bridge/qualifying-style
    deal) or walk away and let him reach unrestricted free agency. The user team is
    intentionally skipped so the human still manages his own RFAs in the contract
    office.
    """
    sim = session.sim
    league = getattr(sim, "league", None)
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    rng = getattr(sim, "rng", None) or random.Random(0)
    user_tid = str(getattr(session, "user_team_id", "") or "")

    re_signed: List[Dict[str, Any]] = []
    walked: List[Dict[str, Any]] = []

    for team in _get(league, "teams", None) or []:
        tid = str(_get(team, "team_id", "") or _get(team, "id", ""))
        if tid == user_tid:
            continue
        rights = list(_ensure_rfa_rights_list(team))
        if not rights:
            continue

        # Re-evaluate need/cap once per team; the snapshot inside sign_player_to_team
        # keeps the hard cap/slot gates authoritative between individual signings.
        ctx = evaluate_team_position_needs(team, league, sim, season_year=season_year)
        # Best RFAs first so scarce cap/slots go to the players worth keeping.
        rights.sort(key=lambda r: -float(r.get("overall") or 0))

        for entry in rights:
            player = entry.get("player_ref")
            pid = str(entry.get("player_id", "") or "")
            if player is None:
                # Orphaned rights row (player object lost) — drop it so it can't
                # accumulate forever.
                remove_rfa_rights(team, pid)
                continue

            ovr = _player_ovr(player)
            pos = _position_bucket(player)
            qo = float(entry.get("qualifying_offer_aav_m") or LEAGUE_MINIMUM_AAV_M)
            demand = compute_player_demand(player, team, league, context="rfa")
            market = float(demand["market_value_m"])

            snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
            cap_space = float(snap.get("usable_cap_space_m", 0) or 0)
            slots_left = int(snap.get("contract_slots_limit", CONTRACT_SLOTS_LIMIT)) - int(
                snap.get("contract_slots_used", 0)
            )

            # A team pays the retention premium for its own RFA (the leverage flows to
            # the club, but a valuable young player still commands a real deal). Meet a
            # reasonable term and open near his demand, bounded by cap space; the hard
            # cap/slot check inside sign_player_to_team stays authoritative.
            want = float(demand["want_aav_m"])
            want_years = int(demand["want_years"])
            years = max(2, min(want_years, 6))
            retain_ceiling = min(cap_space * 0.9, want * 1.08)
            start_aav = round(max(qo, min(want, retain_ceiling)), 3)

            overload = bool(ctx["overload"].get(pos)) and ovr < ctx["best_ovr"].get(pos, 0.0) + 2.0
            unaffordable = cap_space < start_aav or slots_left <= 0
            # Walk away from cheap depth pieces the team has no room/need for; keep
            # anyone with real value if it can be afforded.
            if slots_left <= 0 or (overload and ovr < 78) or (unaffordable and ovr < 80):
                release_rfa_rights(team, pid, league)
                walked.append({
                    "team_id": tid, "player_id": pid, "overall": round(ovr),
                    "position": pos, "reason": "no_room" if unaffordable else "surplus",
                })
                continue

            spendable = float(ctx.get("spendable_cap_space_m", cap_space) or cap_space)
            two_year_cost = start_aav * min(2, years)

            # Depth RFAs: 1-year qualifying offer instead of a multi-year chase.
            if ovr < 80 or (ovr < 82 and two_year_cost > spendable * 0.85):
                qo_result = qualify_rfa(team, pid, league, season_year)
                if qo_result.get("ok"):
                    re_signed.append({
                        "team_id": tid, "player_id": pid, "overall": round(ovr),
                        "position": pos, "aav_m": qo, "years": 1, "method": "qualifying_offer",
                    })
                else:
                    release_rfa_rights(team, pid, league)
                    walked.append({
                        "team_id": tid, "player_id": pid, "overall": round(ovr),
                        "position": pos, "reason": str(qo_result.get("reason") or "qo_failed"),
                    })
                continue

            neg_rounds = 5 if ovr >= 88 else (3 if ovr >= 82 else 2)
            agreed, final_aav, final_years = _cpu_negotiate_offer(
                team, player, league, start_aav, years, ctx, context="rfa",
                ceiling=retain_ceiling, max_rounds=neg_rounds, session=session,
            )
            if agreed:
                result = sign_player_to_team(
                    player, team, league, season_year,
                    {
                        "aav_m": final_aav,
                        "years": final_years,
                        "context": "rfa",
                        "rights": "RFA",
                        "force": True,
                    },
                )
                if result.get("ok"):
                    remove_rfa_rights(team, pid)
                    re_signed.append({
                        "team_id": tid, "player_id": pid, "overall": round(ovr),
                        "position": pos, "aav_m": final_aav, "years": final_years,
                        "method": "negotiated",
                    })
                else:
                    release_rfa_rights(team, pid, league)
                    walked.append({
                        "team_id": tid, "player_id": pid, "overall": round(ovr),
                        "position": pos, "reason": str(result.get("reason") or "sign_failed"),
                    })
                continue

            # Failed negotiation → salary arbitration for valuable RFAs.
            if _player_age(player) >= 20 and ovr >= 82:
                team_ask = round(market * 0.90, 3)
                player_ask = round(want, 3)
                award, award_years = compute_arbitration_award(
                    player, team, league,
                    team_offer_m=team_ask,
                    player_ask_m=player_ask,
                    season_year=season_year,
                )
                result = sign_player_to_team(
                    player, team, league, season_year,
                    {
                        "aav_m": award,
                        "years": award_years,
                        "context": "rfa",
                        "rights": "RFA",
                        "force": True,
                    },
                )
                if result.get("ok"):
                    remove_rfa_rights(team, pid)
                    re_signed.append({
                        "team_id": tid, "player_id": pid, "overall": round(ovr),
                        "position": pos, "aav_m": award, "years": award_years,
                        "method": "arbitration",
                    })
                    continue

            release_rfa_rights(team, pid, league)
            walked.append({
                "team_id": tid, "player_id": pid, "overall": round(ovr),
                "position": pos, "reason": "no_agreement",
            })

    return {
        "re_signed": re_signed,
        "walked": walked,
        "re_signed_count": len(re_signed),
        "walked_count": len(walked),
    }


def run_cpu_own_ufa_resign(session: Any) -> Dict[str, Any]:
    """CPU clubs re-sign their own exclusive UFAs before Opening Day opens the market.

    Without this, every expired UFA (including 90+ OVR stars) dumps into the open
    FA pool the moment exclusivity clears — Kucherov/Makar/Crosby style free agents
    on day 1.
    """
    sim = session.sim
    league = getattr(sim, "league", None)
    if league is None:
        return {"re_signed": [], "walked": [], "re_signed_count": 0, "walked_count": 0}
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    team_by_id = getattr(session, "team_by_id", None) or {}

    def _resolve_team(tid: str) -> Any:
        t = team_by_id.get(str(tid)) if isinstance(team_by_id, dict) else None
        if t is not None:
            return t
        for cand in _get(league, "teams", None) or []:
            cid = str(_get(cand, "team_id", "") or _get(cand, "id", ""))
            if cid == str(tid):
                return cand
        return None

    # Group exclusive UFAs by former club.
    by_team: Dict[str, List[Any]] = {}
    for p in list(_get(league, "free_agents", None) or []):
        if _get(p, "retired", False):
            continue
        from_tid = str(
            getattr(p, "ufa_from_team_id", None)
            or getattr(p, "previous_nhl_team_id", None)
            or ""
        )
        if not from_tid or from_tid == user_tid:
            continue
        exclusive = bool(getattr(p, "ufa_exclusive", False))
        if not exclusive and bool(getattr(session, "free_agency_open", False)):
            continue
        if not exclusive:
            # Pre-open: treat tagged former club as exclusive home rights.
            try:
                setattr(p, "ufa_exclusive", True)
            except Exception:
                pass
        by_team.setdefault(from_tid, []).append(p)

    re_signed: List[Dict[str, Any]] = []
    walked: List[Dict[str, Any]] = []
    fa_pool = list(_get(league, "free_agents", None) or [])

    for tid, players in by_team.items():
        team = _resolve_team(tid)
        if team is None:
            continue
        ctx = evaluate_team_position_needs(team, league, sim, season_year=season_year)
        players.sort(key=lambda p: -float(_player_ovr(p)))

        for player in players:
            pid = _player_id(player)
            ovr = _player_ovr(player)
            pos = _position_bucket(player)
            demand = compute_player_demand(player, team, league, context="re_sign")
            want = float(demand["want_aav_m"])
            want_years = int(demand["want_years"])
            years = max(1, min(want_years, 7 if ovr >= 88 else 5))

            snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
            cap_space = float(snap.get("usable_cap_space_m", 0) or 0)
            slots_left = int(snap.get("contract_slots_limit", CONTRACT_SLOTS_LIMIT)) - int(
                snap.get("contract_slots_used", 0)
            )
            # Franchise priorities get nearly all remaining room; depth does not.
            retain_frac = 0.98 if ovr >= 88 else (0.92 if ovr >= 84 else 0.85)
            retain_ceiling = min(cap_space * retain_frac, want * 1.12)
            start_aav = round(max(LEAGUE_MINIMUM_AAV_M, min(want, retain_ceiling)), 3)

            overload = bool(ctx["overload"].get(pos)) and ovr < ctx["best_ovr"].get(pos, 0.0) + 2.0
            unaffordable = cap_space < start_aav or slots_left <= 0

            # No force-keep: stars can walk when the club has no room, is overloaded,
            # or cannot meet the ask — same agency players have on the open market.
            # Franchise players (88+) are never walked solely for "surplus".
            if slots_left <= 0 or (overload and ovr < 80) or (unaffordable and ovr < 84):
                try:
                    setattr(player, "ufa_exclusive", False)
                except Exception:
                    pass
                walked.append({
                    "team_id": tid, "player_id": pid, "overall": round(ovr),
                    "position": pos, "reason": "no_room" if unaffordable else "surplus",
                })
                continue

            if unaffordable:
                # Offer what space allows; player may still reject via negotiation.
                years = min(years, 2 if ovr < 90 else 3)
                affordable = max(LEAGUE_MINIMUM_AAV_M, max(0.0, cap_space) * 0.98)
                start_aav = round(max(LEAGUE_MINIMUM_AAV_M, min(want * 0.92, affordable or want * 0.85)), 3)

            # Losing / tiny-market clubs: elite UFAs often prefer testing free agency.
            # Contenders and bubbles always attempt a serious offer first.
            window = str(ctx.get("window") or getattr(team, "gm_window", "") or "").lower()
            prof = demand.get("profile") or _player_negotiation_profile(player)
            loyalty = float(prof.get("loyalty") or 0.5)
            competitiveness = float(prof.get("competitiveness") or 0.5)
            if ovr >= 86 and window in ("rebuilder", "rebuild", "tank", "retool"):
                rebuilder_rounds = 3 if (loyalty >= 0.6 or competitiveness >= 0.7) else 1
                if cap_space >= max(LEAGUE_MINIMUM_AAV_M * 2, want * 0.55) and slots_left > 0:
                    years = min(years, 3)
                    start_aav = round(min(want * 0.90, cap_space * 0.90), 3)
                    agreed, final_aav, final_years = _cpu_negotiate_offer(
                        team, player, league, start_aav, years, ctx, context="re_sign",
                        ceiling=max(start_aav, min(cap_space * 0.95, want)),
                        max_rounds=rebuilder_rounds,
                        session=session,
                    )
                    if agreed:
                        result = sign_player_to_team(
                            player, team, league, season_year,
                            {
                                "aav_m": final_aav,
                                "years": final_years,
                                "context": "re_sign",
                                "rights": "UFA",
                                "force": True,
                            },
                        )
                        if result.get("ok"):
                            fa_pool = [p for p in fa_pool if _player_id(p) != pid]
                            try:
                                setattr(player, "ufa_exclusive", False)
                            except Exception:
                                pass
                            re_signed.append({
                                "team_id": tid, "player_id": pid, "name": _player_name(player),
                                "overall": round(ovr), "position": pos,
                                "aav_m": final_aav, "years": final_years,
                                "note": "rebuilder_retention_attempt",
                            })
                            continue
                try:
                    setattr(player, "ufa_exclusive", False)
                except Exception:
                    pass
                walked.append({
                    "team_id": tid, "player_id": pid, "overall": round(ovr),
                    "position": pos, "reason": "wants_contender",
                })
                continue

            agreed, final_aav, final_years = _cpu_negotiate_offer(
                team, player, league, start_aav, years, ctx, context="re_sign",
                ceiling=max(retain_ceiling, start_aav),
                max_rounds=5 if ovr >= 88 else 3,
                session=session,
            )
            if not agreed:
                try:
                    setattr(player, "ufa_exclusive", False)
                except Exception:
                    pass
                walked.append({
                    "team_id": tid, "player_id": pid, "overall": round(ovr),
                    "position": pos, "reason": "no_agreement",
                })
                continue

            # Honour the negotiated outcome — do not force-sign a rejected deal.
            result = sign_player_to_team(
                player, team, league, season_year,
                {
                    "aav_m": final_aav,
                    "years": final_years,
                    "context": "re_sign",
                    "rights": "UFA",
                    "force": True,  # execute terms already accepted in negotiate
                },
            )
            if not result.get("ok"):
                try:
                    setattr(player, "ufa_exclusive", False)
                except Exception:
                    pass
                walked.append({
                    "team_id": tid, "player_id": pid, "overall": round(ovr),
                    "position": pos, "reason": str(result.get("reason") or "sign_failed"),
                })
                continue

            # Remove from FA pool if still listed.
            fa_pool = [p for p in fa_pool if _player_id(p) != pid]
            try:
                setattr(player, "ufa_exclusive", False)
            except Exception:
                pass
            re_signed.append({
                "team_id": tid, "player_id": pid, "name": _player_name(player),
                "overall": round(ovr), "position": pos,
                "aav_m": final_aav, "years": final_years,
            })

    try:
        league.free_agents = fa_pool
    except Exception:
        pass

    return {
        "re_signed": re_signed,
        "walked": walked,
        "re_signed_count": len(re_signed),
        "walked_count": len(walked),
    }


def _apply_offer_sheet_decision(
    session: Any,
    sheet: Dict[str, Any],
    *,
    rights_team: Any,
    offering_team: Any,
    player: Any,
    decision: str,
    season_year: int,
    league: Any,
) -> Dict[str, Any]:
    """Match or decline a pending offer sheet (any rights-holding team)."""
    pid = str(sheet.get("player_id", "") or "")
    aav_m = float(sheet.get("aav_m") or 0)
    years = max(1, int(sheet.get("years") or 1))
    cap_hit = compute_prorated_cap_hit_m(aav_m, years, float(sheet.get("signing_bonus_m") or 0))
    contract = normalize_contract_dict({
        "type": "STANDARD",
        "years": years,
        "years_remaining": years,
        "aav_m": aav_m,
        "cap_hit_m": cap_hit,
        "rights_status": "RFA",
        "expiry_year": int(season_year) + years,
        "source": "offer_sheet",
        "is_offer_sheet": True,
    })
    decision_l = str(decision or "").lower()
    if decision_l in ("match", "matched"):
        slot_ok = validate_contract_slots(rights_team, league, additional=1)
        if not slot_ok.get("ok"):
            return {"ok": False, "reason": slot_ok.get("reason")}
        check = _validate_sign_cap(rights_team, cap_hit, league, player=player)
        if not check.get("ok"):
            return {"ok": False, "reason": check.get("reason")}
        apply_contract_to_player(player, contract, season_year)
        remove_rfa_rights(rights_team, pid)
        roster = list(_get(rights_team, "roster", None) or [])
        if player not in roster:
            roster.append(player)
            rights_team.roster = roster
        sync_team_cap_fields(rights_team, league)
        sheet["status"] = "matched"
        return {"ok": True, "outcome": "matched", "contract": contract}

    apply_contract_to_player(player, contract, season_year)
    remove_rfa_rights(rights_team, pid)
    for cand in _get(league, "teams", None) or []:
        r = list(_get(cand, "roster", None) or [])
        if player in r:
            r.remove(player)
            cand.roster = r
    offering_roster = list(_get(offering_team, "roster", None) or [])
    if player not in offering_roster:
        offering_roster.append(player)
        offering_team.roster = offering_roster
    try:
        player.team_id = str(_get(offering_team, "team_id", "") or _get(offering_team, "id", ""))
    except Exception:
        pass
    comp = _record_offer_sheet_compensation(
        league,
        offering_team,
        rights_team,
        {"tier": sheet.get("compensation_tier"), "rounds": sheet.get("compensation_rounds")},
        season_year,
    )
    sync_team_cap_fields(offering_team, league)
    sync_team_cap_fields(rights_team, league)
    sheet["status"] = "signed_away"
    return {"ok": True, "outcome": "signed_away", "compensation": comp}


def run_cpu_offer_sheet_pass(session: Any, *, max_sheets: int = 4) -> Dict[str, Any]:
    """CPU teams occasionally file offer sheets on eligible RFAs held by other clubs."""
    sim = session.sim
    league = getattr(sim, "league", None)
    if league is None:
        return {"filed": [], "count": 0}
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    rng = getattr(sim, "rng", None) or random.Random(0)
    filed: List[Dict[str, Any]] = []
    team_sheet_counts: Dict[str, int] = {}

    for offering_team in _get(league, "teams", None) or []:
        if len(filed) >= max_sheets:
            break
        off_tid = str(_get(offering_team, "team_id", "") or _get(offering_team, "id", ""))
        if not off_tid or off_tid == user_tid:
            continue
        if team_sheet_counts.get(off_tid, 0) >= 2:
            continue
        ctx = evaluate_team_position_needs(offering_team, league, sim, season_year=season_year)
        window = str(ctx.get("window") or "").lower()
        if window in ("rebuilder", "rebuild", "tank"):
            file_chance = 0.02
        elif window == "contender":
            file_chance = 0.12
        else:
            file_chance = 0.08
        if rng.random() > file_chance:
            continue

        candidates: List[Tuple[float, Any, Any, Dict[str, Any]]] = []
        for rights_team in _get(league, "teams", None) or []:
            rtid = str(_get(rights_team, "team_id", "") or _get(rights_team, "id", ""))
            if not rtid or rtid == off_tid:
                continue
            for entry in list(_ensure_rfa_rights_list(rights_team)):
                player = entry.get("player_ref")
                if player is None:
                    continue
                ovr = _player_ovr(player)
                if ovr < 80 or ovr > 86:
                    continue
                if not rfa_offer_sheet_eligible(player, entry):
                    continue
                pos = _position_bucket(player)
                need = float(ctx["need_score"].get(pos, 0))
                if need < 0.35:
                    continue
                market = compute_market_value(player, league)
                candidates.append((need + ovr * 0.01, player, rights_team, entry))

        if not candidates:
            continue
        candidates.sort(key=lambda row: -row[0])
        _, player, rights_team, entry = candidates[0]
        market = compute_market_value(player, league)
        years = 4 if _player_ovr(player) >= 84 else 3
        aav_m = round(min(market * rng.uniform(1.02, 1.12), float(ctx.get("cap_space_m", 0) or 0) * 0.92), 3)
        if aav_m < LEAGUE_MINIMUM_AAV_M:
            continue
        res = execute_offer_sheet(
            offering_team, rights_team, player, league, season_year,
            {"aav_m": aav_m, "years": years},
            session=session,
        )
        if res.get("ok"):
            team_sheet_counts[off_tid] = team_sheet_counts.get(off_tid, 0) + 1
            filed.append({"team_id": off_tid, "player_id": _player_id(player), "aav_m": aav_m, "years": years})

    return {"filed": filed, "count": len(filed)}


def tick_offer_sheets(
    session: Any,
    *,
    current_day: Optional[int] = None,
    force_finalize: bool = False,
) -> Dict[str, Any]:
    """Advance offer-sheet match clocks; auto-resolve CPU holders and expired user windows."""
    sim = session.sim
    league = getattr(sim, "league", None)
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    team_by_id = getattr(session, "team_by_id", None) or {}
    day = int(
        current_day
        if current_day is not None
        else getattr(session, "fa_market_day", None)
        or getattr(session, "calendar_days_finished", 0)
        or 0
    )

    def _team(tid: str) -> Any:
        t = team_by_id.get(str(tid)) if isinstance(team_by_id, dict) else None
        if t is not None:
            return t
        for cand in _get(league, "teams", None) or []:
            if str(_get(cand, "team_id", "") or _get(cand, "id", "")) == str(tid):
                return cand
        return None

    sheets = list(_get(league, "pending_offer_sheets", None) or [])
    resolved: List[Dict[str, Any]] = []
    still_pending: List[Dict[str, Any]] = []

    for sheet in sheets:
        if str(sheet.get("status") or "pending") != "pending":
            still_pending.append(sheet)
            continue
        pid = str(sheet.get("player_id", "") or "")
        rights_team = _team(sheet.get("rights_team_id", ""))
        offering_team = _team(sheet.get("offering_team_id", ""))
        aav_m = round(normalize_money_m(sheet.get("aav_m") or 0), 3)
        years = max(1, int(sheet.get("years") or 1))
        filed_day = int(sheet.get("filed_day") or 0)
        expires_day = int(sheet.get("expires_day") or (filed_day + OFFER_SHEET_MATCH_WINDOW_DAYS))
        days_elapsed = max(0, day - filed_day)

        if rights_team is None or offering_team is None:
            sheet["status"] = "void"
            resolved.append({**sheet, "outcome": "void"})
            continue

        entry = find_rfa_rights(rights_team, pid)
        player = entry.get("player_ref") if entry else None
        if entry is None or player is None:
            sheet["status"] = "void"
            resolved.append({**sheet, "outcome": "void_no_rights"})
            continue

        rights_tid = str(_get(rights_team, "team_id", "") or _get(rights_team, "id", ""))
        is_user_rights = rights_tid == user_tid

        if is_user_rights and not force_finalize:
            sheet["days_remaining"] = max(0, expires_day - day)
            if days_elapsed >= 4 and not sheet.get("pressure_emitted"):
                pname = sheet.get("player_name") or _player_name(player)
                _emit_contract_storyline(
                    session,
                    f"Offer sheet decision due soon: match or decline {_player_name(player)} "
                    f"({sheet.get('compensation_label', 'compensation owed')})",
                    kind="offer_sheet",
                    severity="high",
                )
                sheet["pressure_emitted"] = True
            if day < expires_day:
                still_pending.append(sheet)
                continue
            # Day 7+: auto-decline if user has not acted.
            decision = "decline"
        elif is_user_rights and force_finalize:
            decision = "decline"
        else:
            if not force_finalize and days_elapsed < 4:
                sheet["days_remaining"] = max(0, expires_day - day)
                still_pending.append(sheet)
                continue
            if days_elapsed >= 4 and not sheet.get("pressure_emitted"):
                _emit_contract_storyline(
                    session,
                    f"{_get(rights_team, 'name', 'Rights holder')} weighing offer sheet on {_player_name(player)}",
                    kind="offer_sheet",
                )
                sheet["pressure_emitted"] = True
            market = compute_market_value(player, league)
            cap_ok = can_sign_player(rights_team, aav_m, league=league, player=player).get("ok")
            slot_ok = validate_contract_slots(rights_team, league, additional=1).get("ok")
            worth_matching = aav_m <= market * 1.15
            decision = "match" if (cap_ok and slot_ok and worth_matching) else "decline"

        if decision == "match":
            if is_user_rights:
                outcome = resolve_user_offer_sheet(session, player_id=pid, decision="match")
            else:
                outcome = _apply_offer_sheet_decision(
                    session, sheet,
                    rights_team=rights_team, offering_team=offering_team, player=player,
                    decision="match", season_year=season_year, league=league,
                )
                sheets_list = list(_get(league, "pending_offer_sheets", None) or [])
                league.pending_offer_sheets = [s for s in sheets_list if s is not sheet]
            if outcome.get("ok"):
                sheet["status"] = "matched"
                resolved.append({**sheet, "outcome": "matched"})
            else:
                still_pending.append(sheet)
            continue

        if is_user_rights:
            outcome = resolve_user_offer_sheet(session, player_id=pid, decision="decline")
        else:
            outcome = _apply_offer_sheet_decision(
                session, sheet,
                rights_team=rights_team, offering_team=offering_team, player=player,
                decision="decline", season_year=season_year, league=league,
            )
            sheets_list = list(_get(league, "pending_offer_sheets", None) or [])
            league.pending_offer_sheets = [s for s in sheets_list if s is not sheet]
        sheet["status"] = "signed_away" if outcome.get("ok") else "void"
        resolved.append({**sheet, "outcome": sheet["status"], "auto": not is_user_rights or day >= expires_day})

    league.pending_offer_sheets = still_pending
    return {"resolved": resolved, "count": len(resolved), "pending": len(still_pending), "day": day}


def resolve_offer_sheets(session: Any) -> Dict[str, Any]:
    """Finalize all pending offer sheets (roster-check stage)."""
    return tick_offer_sheets(session, force_finalize=True)


def resolve_user_offer_sheet(
    session: Any,
    *,
    player_id: str,
    decision: str,
) -> Dict[str, Any]:
    """User match or decline of a pending offer sheet against their RFA."""
    league = getattr(getattr(session, "sim", None), "league", None)
    user_team = session.team_by_id.get(str(session.user_team_id))
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    if league is None or user_team is None:
        return {"ok": False, "reason": "League/team missing"}
    sheets = list(_get(league, "pending_offer_sheets", None) or [])
    pid = str(player_id)
    user_tid = str(_get(user_team, "team_id", "") or _get(user_team, "id", ""))
    idx = next(
        (
            i
            for i, s in enumerate(sheets)
            if str(s.get("player_id")) == pid and str(s.get("rights_team_id")) == user_tid
        ),
        None,
    )
    if idx is None:
        return {"ok": False, "reason": "No pending offer sheet for this player"}
    sheet = sheets[idx]
    offering_team = session.team_by_id.get(str(sheet.get("offering_team_id") or ""))
    player, _ = _find_player_in_league(league, pid)
    if player is None:
        player = (find_rfa_rights(user_team, pid) or {}).get("player_ref")
    if player is None or offering_team is None:
        return {"ok": False, "reason": "Offer sheet parties missing"}

    aav_m = float(sheet.get("aav_m") or 0)
    years = max(1, int(sheet.get("years") or 1))
    contract = normalize_contract_dict({
        "type": "STANDARD",
        "years": years,
        "years_remaining": years,
        "aav_m": aav_m,
        "cap_hit_m": aav_m,
        "rights_status": "RFA",
        "expiry_year": int(season_year) + years,
        "source": "offer_sheet",
        "is_offer_sheet": True,
    })
    decision_l = str(decision or "").lower()
    if decision_l in ("match", "matched"):
        slot_ok = validate_contract_slots(user_team, league, additional=1)
        if not slot_ok.get("ok"):
            return {"ok": False, "reason": slot_ok.get("reason"), "contract_slots": slot_ok}
        check = _validate_sign_cap(user_team, aav_m, league, player=player)
        if not check.get("ok"):
            return {"ok": False, "reason": check.get("reason")}
        apply_contract_to_player(player, contract, season_year)
        remove_rfa_rights(user_team, pid)
        roster = list(_get(user_team, "roster", None) or [])
        if player not in roster:
            roster.append(player)
            user_team.roster = roster
        sync_team_cap_fields(user_team, league)
        sheet["status"] = "matched"
        sheets.pop(idx)
        league.pending_offer_sheets = sheets
        return {"ok": True, "outcome": "matched", "contract": contract, "offer_sheet": sheet}

    # Decline → player to offering team + compensation
    apply_contract_to_player(player, contract, season_year)
    remove_rfa_rights(user_team, pid)
    for cand in _get(league, "teams", None) or []:
        r = list(_get(cand, "roster", None) or [])
        if player in r:
            r.remove(player)
            cand.roster = r
    offering_roster = list(_get(offering_team, "roster", None) or [])
    if player not in offering_roster:
        offering_roster.append(player)
        offering_team.roster = offering_roster
    comp = _record_offer_sheet_compensation(
        league,
        offering_team,
        user_team,
        {"tier": sheet.get("compensation_tier"), "rounds": sheet.get("compensation_rounds")},
        season_year,
    )
    sync_team_cap_fields(offering_team, league)
    sheet["status"] = "declined"
    sheets.pop(idx)
    league.pending_offer_sheets = sheets
    return {
        "ok": True,
        "outcome": "declined",
        "contract": contract,
        "offer_sheet": sheet,
        "compensation": comp,
    }


def _record_offer_sheet_compensation(
    league: Any,
    offering_team: Any,
    rights_team: Any,
    tier: str,
    season_year: int,
) -> Dict[str, Any]:
    """Charge offer-sheet draft-pick compensation from the offering team to the
    original team. Transfers picks via the pick registry when available; always
    records the compensation on the league for auditing / the draft engine."""
    # Support both legacy tier labels and the expanded CBA-style grid.
    tier_id = "none"
    rounds: List[Any] = []
    if isinstance(tier, dict):
        tier_id = str(tier.get("tier") or "none")
        rounds = list(tier.get("rounds") or [])
    else:
        tier_id = str(tier or "none")

    if not rounds:
        legacy = {"1st+3rd": [1, 3], "2nd": [2], "none": [], "1st_3rd": [1, 3]}
        if tier_id in legacy:
            rounds = list(legacy[tier_id])
        else:
            for probe_aav in (14.1, 10.7, 8.6, 6.4, 4.8, 3.7, 2.8, 2.1, 1.5, 1.2, 0.0):
                info = offer_sheet_compensation_tier(probe_aav)
                if info.get("tier") == tier_id:
                    rounds = list(info.get("rounds") or [])
                    break
    tier = tier_id
    off_tid = str(_get(offering_team, "team_id", "") or _get(offering_team, "id", ""))
    rights_tid = str(_get(rights_team, "team_id", "") or _get(rights_team, "id", ""))
    comp = {
        "offering_team_id": off_tid,
        "rights_team_id": rights_tid,
        "rounds": rounds,
        "season_year": int(season_year) + 1,
        "transferred": [],
    }
    if rounds:
        try:
            from app.sim_engine.trades.trade_pick_registry import (
                get_team_owned_picks,
                transfer_pick,
            )

            owned = get_team_owned_picks(league, off_tid)
            for rnd in rounds:
                # Prefer the offering team's own earliest future pick in that round.
                match = next(
                    (
                        p for p in owned
                        if int(p.get("round", 0)) == rnd
                        and int(p.get("year", 0)) >= int(season_year)
                        and str(p.get("pick_id", "")) not in {t["pick_id"] for t in comp["transferred"]}
                    ),
                    None,
                )
                if match is None:
                    continue
                transfer_pick(league, str(match["pick_id"]), rights_tid)
                comp["transferred"].append({
                    "round": rnd,
                    "year": int(match.get("year", 0)),
                    "pick_id": str(match["pick_id"]),
                })
        except Exception:
            pass
    log = _get(league, "offer_sheet_compensation_log", None)
    if not isinstance(log, list):
        try:
            league.offer_sheet_compensation_log = []
            log = league.offer_sheet_compensation_log
        except Exception:
            log = None
    if isinstance(log, list):
        log.append(comp)
    return comp


# ---------------------------------------------------------------------------
# Cap casualty trades (Phase 3c) — last-resort cap relief via realistic trades
# ---------------------------------------------------------------------------

CAP_CASUALTY_MIN_OVERAGE_M = 0.35
CAP_CASUALTY_MIN_CLEAR_M = 0.75


def _team_id(team: Any) -> str:
    return str(_get(team, "team_id", "") or _get(team, "id", "") or "")


def _team_by_id_from_league(league: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for t in _get(league, "teams", None) or []:
        tid = _team_id(t)
        if tid:
            out[tid] = t
    return out


def _append_cap_casualty_log(league: Any, record: Dict[str, Any]) -> None:
    log = _get(league, "cap_casualty_trades", None)
    if not isinstance(log, list):
        try:
            league.cap_casualty_trades = []
        except Exception:
            return
        log = getattr(league, "cap_casualty_trades", [])
    log.append(dict(record))


def get_team_cap_pressure_context(
    team: Any,
    league: Any,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Full cap-pressure snapshot for cap-casualty decisioning."""
    needs = evaluate_team_position_needs(team, league, sim, season_year=season_year)
    snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
    overage = max(0.0, -float(snap.get("usable_cap_space_m", 0)))
    tid = _team_id(team)
    protected_ids = [
        _player_id(p) for p in _active_roster(team)
        if is_cap_casualty_trade_protected(p, team, league, season_year=season_year)
    ]
    movable = identify_cap_casualty_candidates(
        team, league, sim, season_year=season_year, include_protected=False,
    )
    return {
        "team_id": tid,
        "cap_overage_m": round(overage, 3),
        "usable_cap_space_m": round(float(snap.get("usable_cap_space_m", 0)), 3),
        "total_cap_hit_m": round(float(snap.get("total_cap_hit_m", 0)), 3),
        "active_cap_hit_m": round(float(snap.get("active_roster_cap_hit_m", 0)), 3),
        "buried_cap_hit_m": round(float(snap.get("buried_cap_hit_m", 0)), 3),
        "contract_slots_used": int(snap.get("contract_slots_used", 0)),
        "contract_slots_limit": int(snap.get("contract_slots_limit", CONTRACT_SLOTS_LIMIT)),
        "roster_count": int(needs.get("roster_count", 0)),
        "window": needs.get("window"),
        "need_score": dict(needs.get("need_score") or {}),
        "counts": dict(needs.get("counts") or {}),
        "overload": dict(needs.get("overload") or {}),
        "protected_player_ids": protected_ids,
        "movable_candidates": len(movable),
        "internal_relief_remaining": _team_has_internal_cap_relief(team, league, sim, season_year),
    }


def _team_has_internal_cap_relief(
    team: Any,
    league: Any,
    sim: Any = None,
    season_year: Optional[int] = None,
    *,
    include_buyout: bool = True,
) -> bool:
    """True if a safe bury/waiver/buyout move could still relieve cap pressure."""
    snap = get_team_cap_snapshot_full(team, league, sim, season_year=season_year)
    overage = max(0.0, -float(snap.get("usable_cap_space_m", 0)))
    if overage < 0.01:
        return False
    min_relief = min(overage, max(0.35, overage * 0.25))

    for p in _active_roster(team):
        if is_compliance_protected(p, team):
            continue
        if has_nmc(p) or has_true_elc_contract(p):
            continue
        hit = player_cap_hit_millions(p)
        if is_waiver_exempt(p, team, league):
            savings = estimate_bury_savings(p)
            if savings >= min_relief:
                return True
        elif _player_ovr(p) < 82 and hit >= min_relief * 0.5:
            return True

    if include_buyout:
        for cand in identify_buyout_candidates(team, league, sim, season_year=season_year):
            if float(cand.get("cap_savings_m", 0) or 0) >= min_relief:
                return True
    return False


def needs_cap_casualty_trade(
    team: Any,
    league: Any,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
    after_compliance: bool = True,
) -> bool:
    ctx = get_team_cap_pressure_context(team, league, sim, season_year=season_year)
    if ctx["cap_overage_m"] < CAP_CASUALTY_MIN_OVERAGE_M:
        return False
    if _team_has_internal_cap_relief(
        team, league, sim, season_year, include_buyout=not after_compliance,
    ):
        return False
    if ctx["movable_candidates"] <= 0:
        return False
    best = identify_cap_casualty_candidates(team, league, sim, season_year=season_year)
    if not best:
        return False
    top_hit = float(best[0].get("cap_hit_m", 0) or 0)
    if top_hit < min(CAP_CASUALTY_MIN_CLEAR_M, ctx["cap_overage_m"] * 0.5):
        return False
    return True


def is_cap_casualty_trade_protected(
    player: Any,
    team: Any,
    league: Any = None,
    *,
    season_year: Optional[int] = None,
) -> bool:
    _ = season_year
    if has_nmc(player) or has_ntc(player):
        return True
    if has_true_elc_contract(player):
        return True
    if is_core_player_protected(player, team, league):
        return True
    ovr = _player_ovr(player)
    age = _player_age(player)
    pot = _player_potential(player)
    if age <= 23 and pot >= 76 and ovr >= 74:
        return True
    draft_year = int(_get(player, "draft_year", 0) or _get(player, "draftYear", 0) or 0)
    if draft_year and season_year and (int(season_year) - draft_year) <= 2 and ovr >= 72:
        return True
    if team is not None:
        pos = _position_bucket(player)
        counts = evaluate_team_position_needs(team, league, season_year=season_year)["counts"]
        if counts.get(pos, 0) <= CPU_POSITION_MIN.get(pos, 1):
            return True
    signed_year = int(_get(player, "signed_year", 0) or _get(_get(player, "contract", None), "signed_year", 0) or 0)
    if signed_year and season_year and int(season_year) - signed_year <= 0:
        bad = compute_bad_contract_score(player, team)
        if bad < 0.35:
            return True
    return False


def _score_cap_casualty_candidate(
    player: Any,
    team: Any,
    league: Any,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
) -> float:
    if is_cap_casualty_trade_protected(player, team, league, season_year=season_year):
        return -999.0
    ok, _ = can_trade_player(player)
    if not ok:
        return -999.0
    if not has_active_contract(player):
        return -999.0

    pos = _position_bucket(player)
    ctx = evaluate_team_position_needs(team, league, sim, season_year=season_year)
    counts = ctx["counts"]
    best = ctx["best_ovr"].get(pos, 0.0)
    ovr = _player_ovr(player)
    hit = player_cap_hit_millions(player)
    bad = compute_bad_contract_score(player, team)
    fair = compute_fair_aav(player, team, league)
    age = _player_age(player)
    yrs = _contract_years_remaining(player)

    score = bad * 2.2 + hit * 0.12
    if hit > fair * 1.12:
        score += (hit - fair) * 0.18
    if age >= 32:
        score += 0.18
    if yrs >= 4 and bad >= 0.2:
        score += 0.12
    if counts.get(pos, 0) > CPU_POSITION_MIN.get(pos, 1) + 1:
        score += 0.22
    if ovr < best - 4 and counts.get(pos, 0) >= CPU_POSITION_MIN.get(pos, 1) + 1:
        score += 0.25
    if pos == "G" and counts.get("G", 0) >= 2 and ovr < best:
        score += 0.30
    if counts.get(pos, 0) <= CPU_POSITION_MIN.get(pos, 1):
        score -= 1.5
    if ovr >= 84:
        score -= 0.8
    if hit < 2.5:
        score *= 0.35
    return round(score, 4)


def identify_cap_casualty_candidates(
    team: Any,
    league: Any = None,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
    include_protected: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in _active_roster(team):
        if not include_protected and is_cap_casualty_trade_protected(p, team, league, season_year=season_year):
            continue
        score = _score_cap_casualty_candidate(p, team, league, sim, season_year=season_year)
        if score < 0.05:
            continue
        ok, reason = can_trade_player(p)
        pos = _position_bucket(p)
        ovr = _player_ovr(p)
        hit = player_cap_hit_millions(p)
        out.append({
            "player_id": _player_id(p),
            "name": _player_name(p),
            "position": pos,
            "overall": round(ovr),
            "ovr_band": ovr_band(ovr),
            "cap_hit_m": round(hit, 3),
            "aav_m": round(hit, 3),
            "bad_score": round(compute_bad_contract_score(p, team), 3),
            "fair_aav_m": round(compute_fair_aav(p, team, league), 3),
            "candidate_score": score,
            "tradeable": ok,
            "trade_block_reason": reason if not ok else "",
            "has_ntc": has_ntc(p),
            "has_nmc": has_nmc(p),
            "years_remaining": _contract_years_remaining(p),
            "player_ref": p,
        })
    out.sort(key=lambda r: (-float(r.get("candidate_score", 0)), -float(r.get("cap_hit_m", 0))))
    return out[:8]


def identify_cap_casualty_teams(
    league: Any,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
    user_team_id: str = "",
) -> List[Dict[str, Any]]:
    teams_out: List[Dict[str, Any]] = []
    for team in _get(league, "teams", None) or []:
        tid = _team_id(team)
        if user_team_id and tid == str(user_team_id):
            continue
        if not needs_cap_casualty_trade(team, league, sim, season_year=season_year):
            continue
        ctx = get_team_cap_pressure_context(team, league, sim, season_year=season_year)
        teams_out.append(ctx)
    teams_out.sort(key=lambda r: -float(r.get("cap_overage_m", 0)))
    return teams_out


def score_cap_casualty_trade_partner(
    buyer: Any,
    candidate: Dict[str, Any],
    league: Any,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
) -> float:
    player = candidate.get("player_ref")
    if player is None:
        return 0.0
    pos = str(candidate.get("position") or _position_bucket(player))
    ovr = float(candidate.get("overall", 0) or _player_ovr(player))
    hit = float(candidate.get("cap_hit_m", 0) or player_cap_hit_millions(player))
    bad = float(candidate.get("bad_score", 0) or compute_bad_contract_score(player, buyer))
    ctx = evaluate_team_position_needs(buyer, league, sim, season_year=season_year)
    cap = float(ctx.get("cap_space_m", 0))
    window = str(ctx.get("window") or "bubble")

    if cap < hit + 0.05:
        return 0.0
    if ctx["slots_remaining"] <= 0:
        return 0.0
    if ctx["overload"].get(pos) and ovr < ctx["best_ovr"].get(pos, 0) + 2:
        return 0.0
    if window == "cap_strapped" and hit > LEAGUE_MINIMUM_AAV_M + 0.2:
        return 0.0

    need = float(ctx["need_score"].get(pos, 0))
    score = 0.08 + need * 0.45 + (ovr / 99.0) * 0.22
    if ovr >= ctx["best_ovr"].get(pos, 0) + 1:
        score += 0.12
    if pos == "G" and ctx["counts"].get("G", 0) <= 1:
        score += 0.25

    if window == "contender":
        if need >= 0.35 and ovr >= 78 and bad < 0.35:
            score += 0.18
        elif bad >= 0.35:
            score -= 0.35
    elif window == "rebuilder":
        if bad >= 0.25 and cap >= hit + 2.0:
            score += 0.20
        elif bad < 0.2 and ovr >= 80:
            score -= 0.15
    else:
        if need >= 0.40 and bad < 0.30:
            score += 0.10
        elif bad >= 0.40:
            score -= 0.20

    if bad >= 0.35 and window != "rebuilder":
        score -= 0.25
    return max(0.0, min(1.35, round(score, 4)))


def find_cap_casualty_trade_partners(
    league: Any,
    seller: Any,
    candidate: Dict[str, Any],
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
    user_team_id: str = "",
) -> List[Dict[str, Any]]:
    seller_tid = _team_id(seller)
    scored: List[Dict[str, Any]] = []
    for team in _get(league, "teams", None) or []:
        tid = _team_id(team)
        if tid == seller_tid:
            continue
        if user_team_id and tid == str(user_team_id):
            continue
        fit = score_cap_casualty_trade_partner(
            team, candidate, league, sim, season_year=season_year,
        )
        if fit >= 0.42:
            scored.append({
                "team_id": tid,
                "partner_score": fit,
                "window": evaluate_team_position_needs(team, league, sim, season_year=season_year).get("window"),
                "cap_space_m": evaluate_team_position_needs(team, league, sim, season_year=season_year).get("cap_space_m"),
            })
    scored.sort(key=lambda r: -float(r.get("partner_score", 0)))
    return scored[:6]


def _select_team_trade_pick(
    team: Any,
    league: Any,
    *,
    min_round: int = 3,
    max_round: int = 7,
    prefer_latest_year: bool = True,
) -> Optional[Dict[str, Any]]:
    from services.franchise_paths import ensure_simengine_path

    ensure_simengine_path()
    from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry, get_team_owned_picks

    ensure_draft_pick_registry(league)
    picks = get_team_owned_picks(league, _team_id(team))
    eligible = [
        p for p in picks
        if min_round <= int(p.get("round", 0) or 0) <= max_round and not p.get("resolved")
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda r: (
            -int(r.get("year", 0)) if prefer_latest_year else int(r.get("year", 0)),
            -int(r.get("round", 0)),
        )
    )
    return dict(eligible[0])


def build_cap_casualty_trade_package(
    seller: Any,
    buyer: Any,
    candidate: Dict[str, Any],
    league: Any,
    sim: Any = None,
    *,
    season_year: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    player = candidate.get("player_ref")
    if player is None:
        raise ValueError("Cap casualty candidate missing player_ref")
    pid = str(candidate.get("player_id") or _player_id(player))
    seller_tid = _team_id(seller)
    buyer_tid = _team_id(buyer)
    bad = float(candidate.get("bad_score", 0) or compute_bad_contract_score(player, seller))
    ovr = float(candidate.get("overall", 0) or _player_ovr(player))
    hit = float(candidate.get("cap_hit_m", 0) or player_cap_hit_millions(player))
    fair = float(candidate.get("fair_aav_m", 0) or compute_fair_aav(player, seller, league))
    buyer_window = evaluate_team_position_needs(
        buyer, league, sim, season_year=season_year,
    ).get("window")

    assets: Dict[str, List[Dict[str, Any]]] = {
        buyer_tid: [{"type": "player", "id": pid, "team": seller_tid}],
        seller_tid: [],
    }

    overpaid = bad >= 0.30 or hit > fair * 1.15
    useful = ovr >= 80 and bad < 0.25
    # Mild AAV overpays on still-useful NHL players are "buyer pays" deals, not
    # sweetened dumps. Only treat as overpaid dump when the contract is actually bad.
    if useful and bad < 0.25:
        overpaid = bad >= 0.30

    if overpaid:
        rnd = 3 if bad >= 0.45 else 4 if bad >= 0.35 else 5
        if buyer_window == "rebuilder":
            rnd = min(rnd, 3)
        pick = _select_team_trade_pick(seller, league, min_round=rnd, max_round=7)
        if pick:
            assets[buyer_tid].append({
                "type": "pick",
                "id": pick["pick_id"],
                "team": seller_tid,
                "year": pick.get("year"),
                "round": pick.get("round"),
            })
    elif useful:
        rnd = 2 if ovr >= 84 else 3 if ovr >= 81 else 4
        pick = _select_team_trade_pick(buyer, league, min_round=rnd, max_round=6)
        if pick:
            assets[seller_tid].append({
                "type": "pick",
                "id": pick["pick_id"],
                "team": buyer_tid,
                "year": pick.get("year"),
                "round": pick.get("round"),
            })

    return assets


def execute_cap_casualty_trade(
    league: Any,
    seller: Any,
    buyer: Any,
    assets_by_team: Dict[str, List[Dict[str, Any]]],
    reason: str,
    *,
    sim: Any = None,
    season_year: Optional[int] = None,
    team_by_id: Optional[Dict[str, Any]] = None,
    user_team_id: str = "",
) -> Dict[str, Any]:
    from services.franchise_paths import ensure_simengine_path

    ensure_simengine_path()
    from app.sim_engine.trades.trade_evaluator import evaluate_trade_package
    from app.sim_engine.trades.trade_executor import execute_validated_trade
    from app.sim_engine.trades.trade_pick_registry import (
        audit_pick_registry_integrity,
        ensure_draft_pick_registry,
        reconcile_pick_registry_consistency,
    )

    seller_tid = _team_id(seller)
    buyer_tid = _team_id(buyer)
    team_by_id = dict(team_by_id or _team_by_id_from_league(league))
    season_year = int(season_year or getattr(league, "current_season", 0) or 2025)
    from app.sim_engine.trades.trade_pick_registry import upcoming_draft_year

    draft_year = upcoming_draft_year(season_year)
    ensure_draft_pick_registry(league, start_year=draft_year, years_ahead=4)

    seller_before = get_team_cap_snapshot_full(seller, league, sim, season_year=season_year)
    buyer_before = get_team_cap_snapshot_full(buyer, league, sim, season_year=season_year)

    ctx = {
        "sim": sim,
        "league": league,
        "team_by_id": team_by_id,
        "season_year": season_year,
        "draft_year": draft_year,
        "season_is_calendar": True,
        "use_upcoming_draft_year": True,
        "cap_casualty_trade": True,
    }

    evaluation = evaluate_trade_package(
        dict(assets_by_team or {}),
        league=league,
        team_by_id=team_by_id,
        context=ctx,
        user_team_id=user_team_id or None,
    )
    if not evaluation.get("can_execute"):
        return {
            "ok": False,
            "reason": "; ".join(str(r) for r in (evaluation.get("rejection_reasons") or ["validation_failed"])),
        }

    try:
        exec_result = execute_validated_trade(
            evaluation,
            league=league,
            team_by_id=team_by_id,
            context=ctx,
            user_team_id=user_team_id or None,
        )
    except ValueError as exc:
        return {"ok": False, "reason": str(exc)}

    reconcile_pick_registry_consistency(league)
    audit = audit_pick_registry_integrity(league, start_year=draft_year, years_ahead=4)
    critical = [
        e for e in (audit.get("errors") or [])
        if any(
            token in str(e)
            for token in (
                "owned_pick_ids mismatch",
                "Duplicate pick IDs",
                "Missing current_owner_team_id",
                "Registry/list desync",
            )
        )
    ]
    if critical:
        return {"ok": False, "reason": f"pick_registry_mismatch: {critical[0]}"}

    sync_team_cap_fields(seller, league, sim, season_year=season_year)
    sync_team_cap_fields(buyer, league, sim, season_year=season_year)
    seller_after = get_team_cap_snapshot_full(seller, league, sim, season_year=season_year)
    buyer_after = get_team_cap_snapshot_full(buyer, league, sim, season_year=season_year)

    moved_players = list((exec_result.get("history_record") or {}).get("moved_players") or [])
    moved_picks = list((exec_result.get("history_record") or {}).get("moved_picks") or [])
    cap_cleared = round(
        max(0.0, float(seller_after["usable_cap_space_m"]) - float(seller_before["usable_cap_space_m"])),
        3,
    )

    record = {
        "trade_type": "cap_casualty",
        "trade_id": exec_result.get("trade_id"),
        "seller_team_id": seller_tid,
        "buyer_team_id": buyer_tid,
        "players_moved": moved_players,
        "picks_moved": moved_picks,
        "assets_by_team": assets_by_team,
        "cap_cleared_m": cap_cleared,
        "seller_cap_before_m": round(float(seller_before["usable_cap_space_m"]), 3),
        "seller_cap_after_m": round(float(seller_after["usable_cap_space_m"]), 3),
        "buyer_cap_before_m": round(float(buyer_before["usable_cap_space_m"]), 3),
        "buyer_cap_after_m": round(float(buyer_after["usable_cap_space_m"]), 3),
        "reason": reason,
        "season_year": season_year,
        "solved_cap": seller_after["usable_cap_space_m"] >= -0.01,
    }
    _append_cap_casualty_log(league, record)
    return {"ok": True, "record": record, "execution": exec_result}


def run_cpu_cap_casualty_trade_pass(session: Any, *, max_trades: int = 12) -> Dict[str, Any]:
    league = getattr(session.sim, "league", None)
    if league is None:
        return {"cap_casualty_trades": [], "count": 0}
    sim = session.sim
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    team_by_id = dict(getattr(session, "team_by_id", None) or _team_by_id_from_league(league))
    executed: List[Dict[str, Any]] = []

    pressured = identify_cap_casualty_teams(
        league, sim, season_year=season_year, user_team_id=user_tid,
    )
    for team_ctx in pressured:
        if len(executed) >= max_trades:
            break
        seller = team_by_id.get(str(team_ctx.get("team_id", "")))
        if seller is None:
            continue
        for cand in identify_cap_casualty_candidates(seller, league, sim, season_year=season_year):
            if not cand.get("tradeable"):
                continue
            partners = find_cap_casualty_trade_partners(
                league, seller, cand, sim, season_year=season_year, user_team_id=user_tid,
            )
            traded = False
            for partner in partners:
                buyer = team_by_id.get(str(partner.get("team_id", "")))
                if buyer is None:
                    continue
                try:
                    package = build_cap_casualty_trade_package(
                        seller, buyer, cand, league, sim, season_year=season_year,
                    )
                except ValueError:
                    continue
                res = execute_cap_casualty_trade(
                    league,
                    seller,
                    buyer,
                    package,
                    reason="cap_compliance_last_resort",
                    sim=sim,
                    season_year=season_year,
                    team_by_id=team_by_id,
                    user_team_id=user_tid,
                )
                if res.get("ok"):
                    executed.append(res.get("record") or {})
                    traded = True
                    break
            if traded:
                snap = get_team_cap_snapshot_full(seller, league, sim, season_year=season_year)
                if snap["usable_cap_space_m"] >= -0.01:
                    break

    return {"cap_casualty_trades": executed, "count": len(executed)}


# ---------------------------------------------------------------------------
# Contract office payload
# ---------------------------------------------------------------------------

def build_own_ufa_resign_row(player: Any, team: Any, season_year: int, league: Any = None) -> Dict[str, Any]:
    """Serialize a just-expired UFA for the re-sign desk (off-roster, exclusive to former club)."""
    age = _player_age(player)
    ovr = round(_player_ovr(player))
    pot = round(_player_potential(player))
    fair = compute_fair_aav(player, team, league)
    pid = _player_id(player)
    ask = round(max(LEAGUE_MINIMUM_AAV_M, fair * 1.04), 3)
    years = 3 if ovr >= 86 else (2 if ovr >= 78 else 1)
    if age >= 34:
        years = min(years, 2)
    row = {
        "player_id": pid,
        "playerId": pid,
        "id": pid,
        "name": _player_name(player),
        "position": _player_pos(player),
        "age": age,
        "overall": ovr,
        "ovr": ovr,
        "potential": pot,
        "role": _player_pos(player),
        "aav_m": ask,
        "cap_hit_m": ask,
        "current_cap_hit": ask,
        "years_remaining": 0,
        "years": 0,
        "expiry_status": "UFA",
        "rights": "UFA",
        "rights_status": "UFA",
        "contract_status": "own_ufa",
        "own_ufa": True,
        "ufa_exclusive": True,
        "can_negotiate": True,
        "can_qualify": False,
        "can_release_rights": True,
        "can_buyout": False,
        "can_waive": False,
        "can_bury": False,
        "player_ask_aav_m": ask,
        "requested_cap_hit": ask,
        "requested_term": years,
        "fair_aav_m": round(fair, 3),
        "interest_label": "High",
        "interest_level": "High",
        "tags": ["Own UFA", "Exclusive"],
        "extension_eligible": True,
        "negotiation_state": "open",
        "previous_team_id": str(
            getattr(player, "ufa_from_team_id", None)
            or getattr(player, "previous_nhl_team_id", None)
            or _get(team, "team_id", "")
            or ""
        ),
    }
    try:
        from app.sim_engine.generation.player_headshots import merge_headshot_into_row

        row = merge_headshot_into_row(row, player)
    except Exception:
        pass
    row["available_actions"] = contract_row_available_actions(row)
    # Own UFAs negotiate a new deal (not a mid-contract extension).
    for act in row["available_actions"]:
        if act.get("id") == "negotiate_extension":
            act["id"] = "negotiate"
            act["label"] = "Negotiate Contract"
    if not any(a.get("id") == "walk_away" for a in row["available_actions"]):
        row["available_actions"].insert(
            1, {"id": "walk_away", "label": "Walk Away", "enabled": True}
        )
    return row


def _clause_label(c: Dict[str, Any]) -> str:
    if c.get("nmc"):
        return "NMC"
    if c.get("ntc"):
        return "NTC"
    return "None"


def build_contract_row(player: Any, team: Any, season_year: int, league: Any = None) -> Dict[str, Any]:
    hydrate_player_contract(player)
    c = normalize_contract_dict(_get(player, "contract", None) or {})
    aav = c.get("aav_m") or player_cap_hit_millions(player)
    yrs = _contract_years_remaining(player)
    age = _player_age(player)
    ovr = round(_player_ovr(player))
    pot = round(_player_potential(player))
    tags = compute_contract_tags(player, team)
    fair = compute_fair_aav(player, team, league)
    bad_score = compute_bad_contract_score(player, team)
    importance = compute_team_importance_score(player, team)
    gap = compute_peer_gap_score(player, team)
    buyout = estimate_buyout(player)
    waive_ok, _ = can_waive_or_bury(player)
    trade_ok, _ = can_trade_player(player)
    buyout_ok, _ = can_buyout_player(player)

    value_label = "Fair"
    if "Bargain" in tags:
        value_label = "Bargain"
    elif "Bad Deal" in tags or bad_score >= 0.35:
        value_label = "Bad"

    ext_aav = round(fair * 1.02, 3)
    ext_years = max(1, min(5, yrs))

    pending_july1 = bool(
        c.get("pending_july1_expiry") or getattr(player, "pending_july1_expiry", False)
    )
    # Prefer stored expiry; fall back to season + remaining years.
    try:
        expiry_year = int(c.get("expiry_year") or 0)
    except (TypeError, ValueError):
        expiry_year = 0
    if expiry_year <= 0:
        expiry_year = int(season_year) + max(int(yrs), 0)
    # Re-sign / Extensions desk is the current free-agency class only
    # (this summer's expirings), not players still owed another season.
    fa_class_year = int(season_year) + 1
    in_re_sign_class = bool(pending_july1) or (yrs <= 1 and expiry_year <= fa_class_year)

    pid = _player_id(player)
    row = {
        "player_id": pid,
        "playerId": pid,
        "id": pid,
        "name": _player_name(player),
        "position": _player_pos(player),
        "age": age,
        "overall": ovr,
        "ovr": ovr,
        "potential": pot,
        "role": _player_pos(player),
        "aav_m": round(aav, 3),
        "aav": round(aav, 3),
        "cap_hit_m": round(aav, 3),
        "capHit": round(aav, 3),
        "cap_hit": round(aav, 3),
        "years_remaining": yrs,
        "yearsRemaining": yrs,
        "expiry_year": expiry_year,
        "expiryYear": expiry_year,
        "expiry_status": c.get("rights_status", "UFA"),
        "expiryStatus": c.get("rights_status", "UFA"),
        "expiry_type": c.get("rights_status", "UFA"),
        "rights_status": c.get("rights_status", "UFA"),
        "contractType": c.get("contract_type", "STANDARD"),
        "clause_label": _clause_label(c),
        "clauseLabel": _clause_label(c),
        "ntc": c.get("ntc", False),
        "nmc": c.get("nmc", False),
        "tags": tags,
        "contract_value_score": value_label,
        "team_importance_score": importance,
        "peer_gap_score": gap,
        "bad_contract_score": bad_score,
        "trade_value_impact": round(-bad_score * 2.0, 2),
        "extension_estimate": {"likelyAav": ext_aav, "likelyTerm": ext_years, "risk": "Medium" if bad_score > 0.3 else "Low"},
        "buyout_estimate": {
            "totalCost": buyout["total_cost_m"],
            "years": buyout["years"],
            "annualPenalty": buyout["annual_penalty_m"],
            "capSavings": buyout["cap_savings_m"],
            "warning": buyout.get("warning", ""),
        },
        "waiver_eligible": waive_ok,
        "bury_savings_m": estimate_bury_savings(player) if not _get(player, "is_buried", False) else 0,
        "two_way": bool(c.get("two_way")),
        "entry_level": str(c.get("contract_type") or c.get("type") or "").upper() in ("ELC", "ENTRY_LEVEL"),
        "no_trade_clause": bool(c.get("ntc")),
        "no_move_clause": bool(c.get("nmc")),
        "extension_eligible": in_re_sign_class,
        "pending_july1_expiry": pending_july1,
        "contract_status": "expiring" if in_re_sign_class else ("signed" if yrs > 0 else "expiring"),
        "can_negotiate": in_re_sign_class,
        "can_buyout": buyout_ok and yrs > 0,
        "can_waive": waive_ok,
        "can_trade": trade_ok,
        "can_qualify": False,
        "can_release_rights": False,
        "contract": c,
    }
    try:
        from app.sim_engine.generation.player_headshots import merge_headshot_into_row

        row = merge_headshot_into_row(row, player)
    except Exception:
        pass
    row["available_actions"] = contract_row_available_actions(row)
    return row


def contract_row_available_actions(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Explicit UI actions from backend flags — frontend must not invent eligibility."""
    actions: List[Dict[str, Any]] = []
    if row.get("can_negotiate"):
        actions.append({"id": "negotiate_extension", "label": "Negotiate Extension", "enabled": True})
    if row.get("can_qualify"):
        actions.append({
            "id": "qualify_rfa",
            "label": "Submit Qualifying Offer",
            "enabled": True,
            "qualifying_offer_aav_m": row.get("qualifying_offer_aav_m"),
        })
    if row.get("can_release_rights"):
        actions.append({"id": "walk_away", "label": "Walk Away", "enabled": True})
    if row.get("can_buyout"):
        actions.append({"id": "buyout", "label": "Buyout", "enabled": True})
    if row.get("can_waive"):
        actions.append({"id": "waive", "label": "Waive", "enabled": True})
    if row.get("arbitration_eligible") and not row.get("arbitration_filed"):
        actions.append({"id": "arbitration_file", "label": "File Arbitration", "enabled": True})
    if row.get("arbitration_filed") and not row.get("award_aav_m"):
        actions.append({"id": "arbitration_settle", "label": "Settle Arbitration", "enabled": True})
    if row.get("offer_sheet_pending"):
        actions.append({"id": "match_offer_sheet", "label": "Match Offer Sheet", "enabled": True})
        actions.append({"id": "decline_offer_sheet", "label": "Decline Offer Sheet", "enabled": True})
    actions.append({"id": "view_dossier", "label": "View Player Dossier", "enabled": True})
    if not any(a["id"] in ("negotiate_extension", "qualify_rfa", "walk_away") for a in actions):
        reason = row.get("ineligible_reason") or (
            "Under contract" if int(row.get("years_remaining") or 0) > 1 else "No contract actions available"
        )
        row.setdefault("ineligible_reason", reason)
    return actions


def build_rfa_rights_row(entry: Dict[str, Any], season_year: int, league: Any = None) -> Dict[str, Any]:
    player = resolve_rfa_player(entry, league)
    prev = entry.get("previous_aav_m")
    qo = entry.get("qualifying_offer_aav_m")
    if qo is None and prev is not None:
        qo = qualifying_offer_aav(prev)
    row = {
        "player_id": entry.get("player_id"),
        "name": entry.get("name"),
        "position": entry.get("position"),
        "age": entry.get("age"),
        "overall": entry.get("overall"),
        "status": entry.get("status"),
        "contract_status": "rfa_rights",
        "rights_status": "RFA",
        "expiry_status": "RFA",
        "expiry_type": "RFA",
        "qualifying_offer_aav_m": qo,
        "previous_aav_m": prev,
        "aav_m": prev if prev is not None else qo,
        "cap_hit_m": prev if prev is not None else qo,
        "years_remaining": 0,
        "arbitration_eligible": entry.get("arbitration_eligible"),
        "offer_sheet_eligible": entry.get("offer_sheet_eligible"),
        "arbitration_filed": entry.get("arbitration_filed"),
        "player_ask_m": entry.get("player_ask_m"),
        "team_offer_m": entry.get("team_offer_m"),
        "award_aav_m": entry.get("award_aav_m"),
        "can_qualify": not entry.get("qualified"),
        "can_file_arbitration": bool(entry.get("arbitration_eligible")) and not entry.get("arbitration_filed"),
        "can_release_rights": True,
        "can_negotiate": True,
        "extension_eligible": True,
        "qualifying_offer_eligible": not entry.get("qualified"),
        "tags": ["RFA Rights"],
    }
    if player is not None:
        live = build_contract_row(player, None, season_year, league)
        # Keep identity / ratings from the live player, but do not let a cleared
        # expired contract wipe QO / rights status / action flags.
        for key in (
            "name", "position", "age", "overall", "ovr", "potential", "role",
            "morale", "playerId", "id",
        ):
            if live.get(key) is not None:
                row[key] = live[key]
        row["can_qualify"] = not entry.get("qualified")
        row["can_release_rights"] = True
        row["can_negotiate"] = True
        row["extension_eligible"] = True
        row["contract_status"] = "rfa_rights"
        row["rights_status"] = "RFA"
        row["expiry_status"] = "RFA"
        row["expiry_type"] = "RFA"
        row["qualifying_offer_eligible"] = not entry.get("qualified")
        row["qualifying_offer_aav_m"] = qo
        row["previous_aav_m"] = prev
        if prev is not None:
            row["aav_m"] = prev
            row["cap_hit_m"] = prev
            row["current_cap_hit"] = prev
        elif qo is not None:
            row["aav_m"] = qo
            row["cap_hit_m"] = qo
            row["current_cap_hit"] = qo
        row["years_remaining"] = 0
        tags = list(live.get("tags") or [])
        if "RFA Rights" not in tags:
            tags = ["RFA Rights"] + tags
        row["tags"] = tags
    row["available_actions"] = contract_row_available_actions(row)
    return row


def _resolve_session_user_team(session: Any) -> Any:
    """Resolve the user's team from session maps with id-type / case fallbacks."""
    raw_uid = getattr(session, "user_team_id", None)
    tid = str(raw_uid or "")
    by_id = getattr(session, "team_by_id", None) or {}
    team = by_id.get(tid)
    if team is None and raw_uid is not None:
        team = by_id.get(raw_uid)
    if team is None and tid:
        tid_l = tid.lower()
        for key, tm in by_id.items():
            if str(key).lower() == tid_l:
                return tm
        league = getattr(getattr(session, "sim", None), "league", None)
        for tm in list(getattr(league, "teams", None) or []):
            cand = str(_get(tm, "team_id", None) or _get(tm, "id", "") or "")
            if cand and cand.lower() == tid_l:
                return tm
    return team


def build_contract_office(session: Any) -> Dict[str, Any]:
    """Read-only contract ledger + live cap snapshot for Cap Ledger / FA / Re-Sign.

    Does NOT rewrite roster AAVs. Bootstrap headroom healing belongs only in
    franchise-start / league contract generation — never after a user signing.
    """
    from services.franchise_sim import _display_team, _build_free_agent_row

    sim = session.sim
    league = getattr(sim, "league", None)
    user_team = _resolve_session_user_team(session)
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    cal_cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    last_idx = int(getattr(session, "nhl_regular_season_last_index", 192) or 192)

    # Keep team.cap_space / cap_hit mirrors in sync with the live roster, but do
    # not mutate any player contracts here (except one-time affiliate SPC backfill
    # for saves that predate org-wide 50-contract filling).
    if user_team is not None:
        try:
            from services.franchise_sim import _ensure_team_affiliate_nhl_spcs

            if not bool(getattr(user_team, "_affiliate_spcs_ensured", False)):
                seed = abs(hash(f"aff_backfill|{session.user_team_id}|{season_year}")) & 0xFFFFFFFF
                _ensure_team_affiliate_nhl_spcs(user_team, season_year, __import__("random").Random(seed))
                try:
                    user_team._affiliate_spcs_ensured = True
                except Exception:
                    pass
            # Also backfill other orgs once per session so CPU trades/slots stay consistent
            if not bool(getattr(session, "_league_affiliate_spcs_ensured", False)):
                for tm in list(getattr(league, "teams", None) or []):
                    if tm is user_team:
                        continue
                    tid = str(getattr(tm, "team_id", None) or getattr(tm, "id", "") or "")
                    tseed = abs(hash(f"aff_backfill|{tid}|{season_year}")) & 0xFFFFFFFF
                    _ensure_team_affiliate_nhl_spcs(tm, season_year, __import__("random").Random(tseed))
                session._league_affiliate_spcs_ensured = True
            # League-wide usable space — FA Wire / CPU bids must not use stale mirrors.
            sync_all_team_cap_fields(
                league,
                sim,
                season_year=season_year,
                calendar_cursor=cal_cursor,
                regular_season_last_index=last_idx,
            )
        except Exception:
            pass

    cap_snapshot = get_team_cap_snapshot_full(
        user_team, league, sim,
        season_year=season_year,
        calendar_cursor=cal_cursor,
        regular_season_last_index=last_idx,
    ) if user_team else {}

    contracts: List[Dict[str, Any]] = []
    if user_team:
        for p in _all_rostered(user_team):
            if _get(p, "is_buried", False):
                continue
            contracts.append(build_contract_row(p, user_team, season_year, league))

    contracts.sort(key=lambda r: (
        -(r.get("bad_contract_score") or 0),
        0 if (r.get("years_remaining") or 0) <= 1 else 1,
        -(r.get("overall") or 0),
        -(r.get("aav_m") or 0),
    ))

    expiring = [r for r in contracts if int(r.get("years_remaining") or 0) <= 1]
    # Just-expired UFAs from this club live in the FA pool but belong on the re-sign desk
    # until open free agency (home-team exclusive window).
    own_expired_ufas: List[Dict[str, Any]] = []
    user_tid = str(getattr(session, "user_team_id", "") or "")
    fa_open = bool(getattr(session, "free_agency_open", False))
    if user_team and league is not None and user_tid:
        for p in list(_get(league, "free_agents", None) or []):
            if _get(p, "retired", False):
                continue
            from_tid = str(
                getattr(p, "ufa_from_team_id", None)
                or getattr(p, "previous_nhl_team_id", None)
                or ""
            )
            exclusive = bool(getattr(p, "ufa_exclusive", False))
            if from_tid == user_tid and exclusive:
                try:
                    own_expired_ufas.append(
                        build_own_ufa_resign_row(p, user_team, season_year, league)
                    )
                except Exception:
                    continue
            elif from_tid == user_tid and not fa_open:
                # Legacy: tagged former team but flag missing — still give home desk rights.
                try:
                    setattr(p, "ufa_exclusive", True)
                    own_expired_ufas.append(
                        build_own_ufa_resign_row(p, user_team, season_year, league)
                    )
                except Exception:
                    continue
        # Seed desk list so pending_ufa includes Batherson-type home UFAs.
        # Also merge into `contracts` so the left table / selection can open them
        # (expiring-only left the queue counting them with an empty right panel).
        seen_exp = {str(r.get("player_id") or "") for r in expiring}
        seen_c = {str(r.get("player_id") or "") for r in contracts}
        for row in own_expired_ufas:
            pid = str(row.get("player_id") or "")
            if not pid:
                continue
            if pid not in seen_exp:
                expiring.append(row)
                seen_exp.add(pid)
            if pid not in seen_c:
                contracts.append(row)
                seen_c.add(pid)

    rfa_rights = serialize_rfa_rights(user_team) if user_team else []
    rfa_rows = [
        build_rfa_rights_row(r, season_year, league)
        for r in _ensure_rfa_rights_list(user_team)
    ] if user_team else []
    try:
        user_tid = str(session.user_team_id)
        pending_by_pid = {
            str(s.get("player_id")): s
            for s in list(_get(league, "pending_offer_sheets", None) or [])
            if str(s.get("status") or "pending") == "pending"
            and str(s.get("rights_team_id")) == user_tid
        }
        for row in rfa_rows:
            hit = pending_by_pid.get(str(row.get("player_id")))
            if hit:
                row["offer_sheet_pending"] = True
                row["pending_offer_sheet"] = hit
                row["offer_sheet_aav_m"] = hit.get("aav_m")
                row["offer_sheet_compensation"] = hit.get("compensation_label") or hit.get("compensation_tier")
                filed = int(hit.get("filed_day") or 0)
                expires = int(hit.get("expires_day") or (filed + OFFER_SHEET_MATCH_WINDOW_DAYS))
                day_now = int(getattr(session, "fa_market_day", 0) or getattr(session, "calendar_days_finished", 0) or 0)
                row["offer_sheet_days_remaining"] = max(0, expires - day_now)
    except Exception:
        pass

    buyout_candidates = []
    cap_casualty = []
    if user_team:
        for p in _active_roster(user_team):
            est = estimate_buyout(p)
            if est["total_cost_m"] > 0 and _contract_years_remaining(p) > 0:
                ok, _ = can_buyout_player(p)
                if ok:
                    buyout_candidates.append({"player_id": _player_id(p), "name": _player_name(p), **est})
        cap_casualty = [
            {k: v for k, v in row.items() if k != "player_ref"}
            for row in identify_cap_casualty_candidates(
                user_team, league, sim, season_year=season_year,
            )
        ]

    # Free-agent pool — available for in-season signing (not gated by the offseason stage).
    free_agents: List[Dict[str, Any]] = []
    slot_info: Dict[str, Any] = {}
    stage = str(getattr(session, "offseason_stage", "") or "")
    awaiting_july1 = (
        not fa_open
        and str(getattr(session, "phase", "") or "").lower() == "offseason"
        and stage in ("re_sign", "salary_cap", "draft", "draft_combine", "awards", "retirements")
    )
    if user_team is not None and league is not None:
        try:
            from app.sim_engine.league_hierarchy_bootstrap import ensure_overseas_fa_pool

            rng = getattr(sim, "rng", None)
            if rng is not None:
                ensure_overseas_fa_pool(league, rng, min_count=120, min_goalies=12)
        except Exception:
            pass
        seen_fa: set = set()
        for pool_attr in ("free_agents", "overseas_free_agents"):
            for p in _get(league, pool_attr, None) or []:
                if _get(p, "retired", False):
                    continue
                pid = _player_id(p)
                if pid in seen_fa:
                    continue
                # Exclusive home-team UFAs stay off the open board until FA opens.
                # Overseas + leftover summer UFAs always remain visible.
                is_overseas = pool_attr == "overseas_free_agents" or bool(
                    (getattr(p, "_franchise_assignment", None) or {}).get("overseas")
                )
                if awaiting_july1 and (not is_overseas) and bool(getattr(p, "ufa_exclusive", False)):
                    continue
                seen_fa.add(pid)
                try:
                    free_agents.append(_build_free_agent_row(p, season_year, session))
                except Exception:
                    continue
        free_agents.sort(key=lambda r: -float(r.get("ovr") or 0))
        # Full open market — every unsigned FA, not a high-OVR sample.
        slot_check = validate_contract_slots(user_team, league, additional=0)
        used_slots = int(slot_check.get("contract_slots_used") or 0)
        slot_info = {
            "used": used_slots,
            "limit": CONTRACT_SLOTS_LIMIT,
            "open": max(0, CONTRACT_SLOTS_LIMIT - used_slots),
        }

    cap_history = list(_get(league, "cap_history", None) or [])
    projection = dict(cap_history[-1]) if cap_history else {}
    if projection:
        projection["projectionSource"] = "league"
    elif cap_snapshot.get("upper_limit_m"):
        projection = {
            "upperLimit": round(cap_snapshot["upper_limit_m"] * 1.03, 1),
            "projectionSource": "projected",
        }

    legacy_team = team_cap_snapshot_legacy_compat(cap_snapshot) if cap_snapshot else {}

    pending_sheets = []
    offer_sheet_targets: List[Dict[str, Any]] = []
    try:
        user_tid = str(session.user_team_id)
        for sheet in list(_get(league, "pending_offer_sheets", None) or []):
            if str(sheet.get("status") or "pending") != "pending":
                continue
            if str(sheet.get("rights_team_id")) == user_tid or str(sheet.get("offering_team_id")) == user_tid:
                pending_sheets.append(dict(sheet))
        # League RFAs eligible for user offer sheets (other clubs only)
        for t in list(_get(league, "teams", None) or []):
            tid = str(_get(t, "team_id", "") or _get(t, "id", "") or "")
            if not tid or tid == user_tid:
                continue
            for entry in _ensure_rfa_rights_list(t):
                if not entry.get("offer_sheet_eligible"):
                    continue
                if entry.get("qualified") is False and entry.get("status") == "RELEASED":
                    continue
                qo = float(entry.get("qualifying_offer_aav_m") or LEAGUE_MINIMUM_AAV_M)
                # Default sheet preview at ~120% of QO for compensation peek
                preview_aav = round(max(qo * 1.2, qo + 0.25), 3)
                tier = offer_sheet_compensation_tier(preview_aav)
                offer_sheet_targets.append({
                    "player_id": entry.get("player_id"),
                    "name": entry.get("name"),
                    "position": entry.get("position"),
                    "age": entry.get("age"),
                    "overall": entry.get("overall"),
                    "rights_team_id": tid,
                    "previous_aav_m": entry.get("previous_aav_m"),
                    "qualifying_offer_aav_m": qo,
                    "offer_sheet_eligible": True,
                    "suggested_aav_m": preview_aav,
                    "compensation_preview": tier,
                })
        offer_sheet_targets.sort(key=lambda r: -(float(r.get("overall") or 0)))
        offer_sheet_targets = offer_sheet_targets[:40]
    except Exception:
        pending_sheets = []
        offer_sheet_targets = []

    return {
        "ok": True,
        "team_id": str(session.user_team_id),
        "season": f"{season_year}-{(season_year + 1) % 100:02d}",
        "cap_snapshot": cap_snapshot,
        "team": {
            "id": str(session.user_team_id),
            "name": _display_team(user_team) if user_team else str(session.user_team_id),
            "salary_cap": legacy_team.get("salary_cap", 0),
            "cap_hit": legacy_team.get("cap_hit", 0),
            "cap_space": legacy_team.get("cap_space", 0),
            "cap_warnings": cap_snapshot.get("warnings", []),
            "needs": compute_team_needs(user_team) if user_team else {},
        },
        "cap": cap_snapshot.get("_raw", {}),
        "team_cap": cap_snapshot,
        "capHistory": cap_history,
        "nextYearProjection": projection,
        "contracts": contracts,
        "free_agents": free_agents,
        "contract_slots": slot_info,
        "expiring": expiring,
        "own_expired_ufas": own_expired_ufas,
        "rfa_rights": rfa_rows,
        "buyout_candidates": buyout_candidates,
        "cap_casualty_candidates": cap_casualty,
        "pending_offer_sheets": pending_sheets,
        "offer_sheet_targets": offer_sheet_targets,
        "offer_sheet_compensation_grid": [
            {k: v for k, v in row.items()} for row in OFFER_SHEET_COMPENSATION_GRID
        ],
        "warnings": cap_snapshot.get("warnings", []),
        "summary": {
            "expiringDeals": len(expiring),
            "ufaCount": sum(1 for r in expiring if r.get("expiry_status") == "UFA"),
            "rfaCount": sum(1 for r in expiring if r.get("expiry_status") == "RFA") + len(rfa_rows),
            "clauseProtected": sum(1 for r in contracts if r.get("clause_label") != "None"),
            "badContracts": sum(1 for r in contracts if "Bad Deal" in (r.get("tags") or [])),
            "capWarnings": cap_snapshot.get("warnings", []),
            "capCommitted": cap_snapshot.get("total_cap_hit_m", 0),
            "pendingOfferSheets": len(pending_sheets),
            "offerSheetTargets": len(offer_sheet_targets),
        },
    }


def get_cached_contract_office(session: Any) -> Dict[str, Any]:
    rev = int(getattr(session, "_stats_revision", 0) or 0)
    cached = getattr(session, "_cached_contract_office_payload", None)
    if isinstance(cached, dict) and int(cached.get("revision", -1)) == rev:
        payload = cached.get("payload")
        if isinstance(payload, dict):
            return dict(payload)
    payload = build_contract_office(session)
    session._cached_contract_office_payload = {"revision": rev, "payload": payload}
    return dict(payload) if isinstance(payload, dict) else {}


def _percentile_rank(values: List[float], value: float, *, higher_is_better: bool = True) -> Optional[int]:
    """Stats Central–style percentile: 100 = best in pool."""
    clean = [float(v) for v in values if v is not None]
    if not clean or value is None:
        return None
    try:
        target = float(value)
    except (TypeError, ValueError):
        return None
    if len(clean) == 1:
        return 100
    ordered = sorted(clean, reverse=higher_is_better)
    # Prefer exact match index; otherwise nearest
    try:
        idx = ordered.index(target)
    except ValueError:
        idx = min(range(len(ordered)), key=lambda i: abs(ordered[i] - target))
    return int(round((1.0 - (idx / (len(ordered) - 1))) * 100.0))


def _team_label_for_id(session: Any, team_id: Any) -> str:
    tid = str(team_id or "")
    if not tid:
        return ""
    team = (getattr(session, "team_by_id", None) or {}).get(tid)
    if team is None and tid.isdigit():
        try:
            team = (getattr(session, "team_by_id", None) or {}).get(int(tid))
        except Exception:
            team = None
    if team is None:
        return tid
    return str(
        _get(team, "name", None)
        or _get(team, "team_name", None)
        or _get(team, "abbreviation", None)
        or tid
    )


def _attach_free_agent_stats_central(session: Any, player: Any, detail: Dict[str, Any]) -> Dict[str, Any]:
    """Attach NHL ledger WAR / percentiles / prior clubs for the FA signing desk."""
    pid = str(detail.get("player_id") or detail.get("id") or _player_id(player) or "")
    age = detail.get("age")
    if age is None:
        age = _player_age(player)
        detail["age"] = age
    if detail.get("potential") is None:
        pot = getattr(player, "ratings", None) or {}
        try:
            detail["potential"] = int(float(pot.get("dev_potential", 0) or 0)) or None
        except Exception:
            detail["potential"] = None

    previous_teams: List[Dict[str, Any]] = []
    cur_team = detail.get("current_team") or detail.get("previous_team")
    cur_league = detail.get("current_league") or detail.get("previous_season_league")
    if cur_team or cur_league:
        previous_teams.append({
            "team": cur_team or "Unsigned",
            "league": cur_league,
            "season": detail.get("season"),
            "label": "Most recent",
            "stats": detail.get("season_stats") or {},
        })
    prev_stats = detail.get("previous_season_stats")
    if isinstance(prev_stats, dict) and prev_stats:
        previous_teams.append({
            "team": detail.get("previous_season_league") or "Prior club",
            "league": detail.get("previous_season_league"),
            "season": (detail.get("season") or 0) - 1 if detail.get("season") else None,
            "label": "Prior season",
            "stats": prev_stats,
        })

    meta = getattr(player, "_franchise_assignment", None) or {}
    for key in ("club", "previous_club", "nhl_team", "last_nhl_team"):
        club = meta.get(key)
        if club and not any(str(t.get("team") or "") == str(club) for t in previous_teams):
            previous_teams.append({"team": str(club), "league": meta.get("league"), "label": "Former"})

    career = getattr(player, "career_stats", None) or {}
    if isinstance(career, dict):
        seasons = career.get("seasons") or career.get("by_season") or career.get("history")
        if isinstance(seasons, list):
            for row in seasons[-4:]:
                if not isinstance(row, dict):
                    continue
                previous_teams.append({
                    "team": row.get("team") or row.get("team_name") or row.get("club"),
                    "league": row.get("league"),
                    "season": row.get("season") or row.get("year"),
                    "label": "Career",
                    "stats": {
                        "gp": row.get("gp"),
                        "g": row.get("g") or row.get("goals"),
                        "a": row.get("a") or row.get("assists"),
                        "pts": row.get("pts") or row.get("points"),
                        "save_pct": row.get("save_pct"),
                        "gaa": row.get("gaa"),
                    },
                })

    detail["previous_teams"] = previous_teams[:6]

    raw_ledger = dict((getattr(session, "player_season_stats", None) or {}).get(pid) or {})
    if not raw_ledger:
        # Offseason may still hold completed-season snapshot under alternate keys
        for alt in (getattr(session, "preseason_player_stats_snapshot", None) or {},):
            if isinstance(alt, dict) and pid in alt:
                raw_ledger = dict(alt.get(pid) or {})
                break

    try:
        from app.sim_engine.generation.player_analytics import (  # noqa: WPS433
            enrich_player_row,
            enrich_player_rows,
            is_goalie_row,
        )

        pool_raw = [
            dict(r)
            for r in list((getattr(session, "player_season_stats", None) or {}).values())
            if isinstance(r, dict) and str(r.get("stat_scope") or "regular_season") == "regular_season"
        ]
        if not pool_raw and raw_ledger:
            pool_raw = [raw_ledger]

        enriched_pool = enrich_player_rows(pool_raw) if pool_raw else []
        mine = next(
            (r for r in enriched_pool if str(r.get("player_id") or r.get("id") or "") == pid),
            None,
        )
        if mine is None and raw_ledger:
            mine = enrich_player_row(raw_ledger)

        if mine:
            is_g = is_goalie_row(mine)
            peer_pool = [r for r in enriched_pool if is_goalie_row(r) == is_g] or [mine]
            metric_defs = (
                [("WAR", "war", True), ("SV%", "sv_pct", True), ("GAA", "gaa", False), ("GSAx", "gsax", True)]
                if is_g
                else [
                    ("WAR", "war", True),
                    ("G", "g", True),
                    ("A", "a", True),
                    ("PTS", "pts", True),
                    ("xGF%", "xgf_pct", True),
                    ("CF%", "cf_pct", True),
                ]
            )
            percentiles: List[Dict[str, Any]] = []
            for label, key, hib in metric_defs:
                vals = []
                for row in peer_pool:
                    try:
                        if row.get(key) is not None:
                            vals.append(float(row.get(key)))
                    except (TypeError, ValueError):
                        continue
                pct = _percentile_rank(vals, mine.get(key), higher_is_better=hib)
                percentiles.append({
                    "label": label,
                    "key": key,
                    "value": mine.get(key),
                    "percentile": pct,
                })

            nhl_line = {
                "gp": mine.get("gp"),
                "g": mine.get("g") or mine.get("goals"),
                "a": mine.get("a") or mine.get("assists"),
                "pts": mine.get("pts") or mine.get("points"),
                "points": mine.get("pts") or mine.get("points"),
                "plus_minus": mine.get("plus_minus"),
                "toi": mine.get("toi_per_game") or mine.get("avg_toi") or mine.get("toi"),
                "xgf_pct": mine.get("xgf_pct"),
                "cf_pct": mine.get("cf_pct"),
                "war": mine.get("war"),
                "sv_pct": mine.get("sv_pct") or mine.get("save_pct"),
                "save_pct": mine.get("sv_pct") or mine.get("save_pct"),
                "gaa": mine.get("gaa"),
                "gsax": mine.get("gsax"),
                "wins": mine.get("wins"),
                "shutouts": mine.get("shutouts"),
                "is_goalie": is_g,
            }
            detail["stats_central"] = {
                "source": "nhl_ledger",
                "war": mine.get("war"),
                "war_valid": mine.get("war_valid"),
                "xgf_pct": mine.get("xgf_pct"),
                "cf_pct": mine.get("cf_pct"),
                "league_rank_war": mine.get("league_rank_war") or mine.get("league_rank_goalie_watr"),
                "league_rank_pts": mine.get("league_rank_pts"),
                "percentiles": percentiles,
                "pool_size": len(peer_pool),
                "season_stats": nhl_line,
            }
            # Prefer real NHL ledger over overseas projection when the player logged NHL games
            if int(mine.get("gp") or 0) > 0:
                detail["season_stats"] = {**(detail.get("season_stats") or {}), **nhl_line}
                detail["stat_source"] = "nhl_ledger"
                detail["stat_projected"] = False
                detail["war"] = mine.get("war")
                tid = mine.get("team_id")
                if tid:
                    label = _team_label_for_id(session, tid)
                    detail["nhl_team"] = label
                    if label and not any(str(t.get("team") or "") == label for t in previous_teams):
                        previous_teams.insert(0, {
                            "team": label,
                            "league": "NHL",
                            "season": detail.get("season"),
                            "label": "NHL season",
                            "stats": nhl_line,
                        })
                        detail["previous_teams"] = previous_teams[:6]
    except Exception:
        logging.getLogger(__name__).exception("FA stats-central attach failed for %s", pid)

    return detail


def build_free_agent_detail(session: Any, player_id: str) -> Dict[str, Any]:
    """Full detail for a single free agent (loaded on demand so the office list stays light)."""
    from services.franchise_sim import _build_free_agent_row

    sim = session.sim
    league = getattr(sim, "league", None)
    user_team = session.team_by_id.get(str(session.user_team_id))
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    pid = str(player_id)

    player = None
    for pool_attr in ("free_agents", "overseas_free_agents"):
        for p in _get(league, pool_attr, None) or []:
            if _player_id(p) == pid:
                player = p
                break
        if player is not None:
            break
    if player is None:
        return {"ok": False, "reason": "Player not found"}

    detail = _build_free_agent_row(player, season_year, session, detail=True)
    detail = _attach_free_agent_stats_central(session, player, detail)
    cap_space = None
    open_slots = None
    if user_team is not None:
        snap = get_team_cap_snapshot_full(user_team, league, sim, season_year=season_year)
        cap_space = snap.get("usable_cap_space_m")
        slot_check = validate_contract_slots(user_team, league, additional=0)
        open_slots = max(0, CONTRACT_SLOTS_LIMIT - int(slot_check.get("contract_slots_used") or 0))
    return {"ok": True, "free_agent": detail, "cap_space_m": cap_space, "open_contract_slots": open_slots}


# ---------------------------------------------------------------------------
# Route handlers (called from main.py)
# ---------------------------------------------------------------------------

def handle_contract_action(session: Any, action: str, body: Dict[str, Any]) -> Dict[str, Any]:
    sim = session.sim
    league = getattr(sim, "league", None)
    user_team = session.team_by_id.get(str(session.user_team_id))
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    player_id = str(body.get("player_id") or body.get("playerId") or "")
    body = dict(body or {})
    body["_session"] = session
    body["_franchise_session"] = session

    if user_team is None:
        return {"ok": False, "reason": "User team not found"}

    player, owner_team = _find_player_in_league(league, player_id)

    result: Dict[str, Any]
    if action == "qualify-rfa":
        result = qualify_rfa(user_team, player_id, league, season_year)
    elif action == "release-rights":
        result = release_rfa_rights(user_team, player_id, league, session=session)
    elif action == "sign-free-agent":
        if player is None:
            result = {"ok": False, "reason": "Player not found"}
        elif owner_team is not None:
            result = {"ok": False, "reason": "Player already signed"}
        else:
            body.pop("force", None)
            body.setdefault("context", "ufa")
            phase = str(getattr(session, "phase", "") or "").lower()
            body["playoff_eligible"] = phase not in ("playoffs", "playoff_ready", "post_cup")
            body["signed_day"] = int(getattr(session, "calendar_cursor", 0) or 0)
            result = sign_player_to_team(player, user_team, league, season_year, body)
            try:
                from services.fa_market_engine import mark_fa_player_signed, record_user_fa_offer

                status = str(result.get("status") or "")
                if status in ("accepted", "pending"):
                    record_user_fa_offer(
                        session,
                        player_id=player_id,
                        aav_m=float(body.get("aav_m") or 0),
                        years=int(body.get("years") or 1),
                        ntc=bool(body.get("ntc")),
                        nmc=bool(body.get("nmc")),
                        status="accepted" if status == "accepted" else "pending",
                    )
                if status == "accepted":
                    mark_fa_player_signed(session, player_id)
            except Exception:
                pass
            # Always refresh FA board after a user offer so signed players leave the list.
            try:
                from services.franchise_offseason import _open_free_agency

                if result.get("ok") and str(result.get("status") or "") != "evaluated":
                    refreshed_fa = _open_free_agency(session, force=False)
                    market = refreshed_fa.get("free_agency_market") or {}
                    if str(result.get("status") or "") == "accepted":
                        wire = list(market.get("market_news") or [])
                        wire.append({
                            "kind": "signing",
                            "text": (
                                f"Your club signs "
                                f"{getattr(player, 'name', None) or player_id} · "
                                f"{body.get('aav_m')}M × {body.get('years')}y"
                            ),
                        })
                        market["market_news"] = wire[-24:]
                        session.free_agency_market_payload = market
                    elif str(result.get("status") or "") == "pending":
                        wire = list(market.get("market_news") or [])
                        wire.append({
                            "kind": "offer",
                            "text": (
                                f"Your offer is on the table for "
                                f"{getattr(player, 'name', None) or player_id} — Sim Day to hear back"
                            ),
                        })
                        market["market_news"] = wire[-24:]
                        session.free_agency_market_payload = market
                    result["free_agency_market"] = market
                    result["free_agents"] = refreshed_fa.get("free_agents") or market.get("free_agents")
            except Exception:
                pass
            if str(result.get("status") or "") == "accepted":
                try:
                    from services.franchise_offseason import invalidate_offseason_decision_payloads

                    invalidate_offseason_decision_payloads(session, reason="fa_sign")
                    result["roster_cleanup"] = getattr(session, "roster_cleanup_payload", None) or {}
                except Exception:
                    pass
    elif action in ("re-sign", "offer"):
        if player is None:
            player = _find_player_in_league(league, player_id)[0]
        if player is None:
            result = {"ok": False, "status": "invalid", "reason": "Player not found"}
        else:
            body.setdefault("context", "re_sign")
            result = sign_player_to_team(player, user_team, league, season_year, body)
    else:
        # Preserve remaining action branches via original flow below.
        result = None

    if result is None:
        # Fall through to the rest of the original handler by re-entering specific branches.
        pass
    elif result.get("ok"):
        if result.get("status") != "evaluated":
            try:
                from services.franchise_offseason import (
                    invalidate_offseason_decision_payloads,
                    _prepare_resign_payload,
                )
                invalidate_offseason_decision_payloads(session, reason=action)
                refreshed = _prepare_resign_payload(session, force=True)
                result["re_sign"] = refreshed.get("re_sign") or refreshed.get("contracts")
                result["contracts"] = refreshed.get("contracts") or refreshed.get("re_sign")
            except Exception:
                pass
        return result
    elif result is not None:
        # Countered / rejected — still return negotiation state + refreshed board.
        try:
            from services.franchise_offseason import _prepare_resign_payload
            refreshed = _prepare_resign_payload(session, force=True)
            result["re_sign"] = refreshed.get("re_sign") or refreshed.get("contracts")
            result["contracts"] = refreshed.get("contracts") or refreshed.get("re_sign")
        except Exception:
            pass
        return result

    # --- Remaining actions (buyout / waive / bury / elc / offer-sheet / arbitration) ---
    if action == "buyout":
        if player is None or owner_team != user_team:
            p = next((p for p in _active_roster(user_team) if _player_id(p) == player_id), None)
            if p is None:
                return {"ok": False, "reason": "Player not found on roster"}
            player = p
        result = execute_buyout(user_team, player, league, season_year)
    elif action == "waive":
        p = player if owner_team == user_team else next(
            (p for p in _active_roster(user_team) if _player_id(p) == player_id), None
        )
        if p is None:
            return {"ok": False, "reason": "Player not found"}
        result = execute_waive(user_team, p, league)
    elif action == "bury":
        p = player if owner_team == user_team else next(
            (p for p in _active_roster(user_team) if _player_id(p) == player_id), None
        )
        if p is None:
            return {"ok": False, "reason": "Player not found"}
        result = execute_bury(user_team, p, league)
    elif action == "sign-elc":
        p = find_prospect_on_team(user_team, player_id)
        if p is None:
            return {"ok": False, "reason": "Prospect not found"}
        # Prefer structured offer submit path
        from services.elc_offer_engine import submit_elc_offer, build_offer_from_template

        template_id = str(body.get("template_id") or body.get("offer_template") or "standard_elc")
        offer_body = body.get("offer") if isinstance(body.get("offer"), dict) else None
        if offer_body is None and body.get("development_promise"):
            offer_body = build_offer_from_template(
                p,
                season_year=season_year,
                template_id=template_id,
                development_promise=body.get("development_promise"),
                assignment_plan=body.get("assignment_plan"),
                term_years=body.get("term_years"),
            )
        result = submit_elc_offer(
            session,
            p,
            user_team,
            season_year=season_year,
            template_id=template_id,
            offer=offer_body,
            assignment_plan=body.get("assignment_plan"),
            development_promise=body.get("development_promise"),
            term_years=body.get("term_years"),
            idempotency_key=body.get("idempotency_key") or body.get("offer_id"),
        )
    elif action == "preview-elc-offer":
        p = find_prospect_on_team(user_team, player_id)
        if p is None:
            return {"ok": False, "reason": "Prospect not found"}
        from services.elc_offer_engine import preview_elc_offer

        return preview_elc_offer(
            session,
            p,
            user_team,
            season_year=season_year,
            template_id=str(body.get("template_id") or "standard_elc"),
            offer=body.get("offer") if isinstance(body.get("offer"), dict) else None,
            assignment_plan=body.get("assignment_plan"),
            development_promise=body.get("development_promise"),
            term_years=body.get("term_years"),
        )
    elif action == "submit-elc-offer":
        p = find_prospect_on_team(user_team, player_id)
        if p is None:
            return {"ok": False, "reason": "Prospect not found"}
        from services.elc_offer_engine import submit_elc_offer

        result = submit_elc_offer(
            session,
            p,
            user_team,
            season_year=season_year,
            template_id=str(body.get("template_id") or "standard_elc"),
            offer=body.get("offer") if isinstance(body.get("offer"), dict) else None,
            assignment_plan=body.get("assignment_plan"),
            development_promise=body.get("development_promise"),
            term_years=body.get("term_years"),
            idempotency_key=body.get("idempotency_key") or body.get("offer_id"),
        )
    elif action == "prospect-rights":
        p = find_prospect_on_team(user_team, player_id)
        if p is None:
            return {"ok": False, "reason": "Prospect not found"}
        from services.draft_rights_engine import apply_prospect_rights_decision

        action_id = str(
            body.get("action_id") or body.get("decision") or body.get("rights_action") or ""
        ).strip()
        # Signing via prospect-rights with template routes through offer engine
        if action_id == "sign_elc":
            from services.elc_offer_engine import submit_elc_offer

            result = submit_elc_offer(
                session,
                p,
                user_team,
                season_year=season_year,
                template_id=str(body.get("template_id") or "standard_elc"),
                offer=body.get("offer") if isinstance(body.get("offer"), dict) else None,
                assignment_plan=body.get("assignment_plan"),
                development_promise=body.get("development_promise"),
                term_years=body.get("term_years"),
                idempotency_key=body.get("idempotency_key"),
            )
        else:
            result = apply_prospect_rights_decision(
                session,
                p,
                user_team,
                action_id,
                season_year=season_year,
            )
    elif action == "evaluate-elc":
        p = find_prospect_on_team(user_team, player_id)
        if p is None:
            return {"ok": False, "reason": "Prospect not found"}
        from services.elc_offer_engine import preview_elc_offer

        return preview_elc_offer(
            session,
            p,
            user_team,
            season_year=season_year,
            template_id=str(body.get("template_id") or "standard_elc"),
            offer=body.get("offer") if isinstance(body.get("offer"), dict) else None,
            assignment_plan=body.get("assignment_plan"),
            development_promise=body.get("development_promise"),
            term_years=body.get("term_years"),
        )
    elif action == "offer-sheet":
        rights_team = session.team_by_id.get(str(body.get("rights_team_id", "")))
        if rights_team is None:
            return {"ok": False, "reason": "Offer sheet targets missing"}
        if player is None:
            entry = find_rfa_rights(rights_team, player_id)
            player = resolve_rfa_player(entry, league)
        if player is None:
            return {"ok": False, "reason": "Offer sheet targets missing"}
        result = execute_offer_sheet(user_team, rights_team, player, league, season_year, body, session=session)
    elif action in ("match-offer-sheet", "decline-offer-sheet"):
        result = resolve_user_offer_sheet(
            session,
            player_id=player_id,
            decision="match" if action == "match-offer-sheet" else "decline",
        )
    elif action == "arbitration-file":
        result = execute_arbitration_file(user_team, player_id, float(body.get("player_ask_m") or 0))
    elif action == "arbitration-settle":
        result = execute_arbitration_settle(user_team, player_id, league, season_year)
    else:
        return {"ok": False, "reason": f"Unknown contract action: {action}"}

    if result.get("ok"):
        try:
            from services.franchise_offseason import invalidate_offseason_decision_payloads
            invalidate_offseason_decision_payloads(session, reason=action)
        except Exception:
            pass
    return result


def _fill_candidate_sort_key(player: Any) -> float:
    return -_player_ovr(player)


def _strip_player_from_all_org_lists(league: Any, player: Any) -> None:
    """Remove a player from every NHL/AHL/ECHL/prospect list before reassignment."""
    pid = _player_id(player)
    if not pid:
        return
    for tm in list(_get(league, "teams", None) or []):
        for attr in ("roster", "ahl_roster", "echl_roster", "prospect_pool"):
            lst = list(_get(tm, attr, None) or [])
            cleaned = [p for p in lst if _player_id(p) != pid]
            if len(cleaned) != len(lst):
                setattr(tm, attr, cleaned)


def _demote_surplus_for_roster_fill(team: Any, need: str, league: Any) -> bool:
    """Free an NHL roster slot by demoting surplus non-needed depth to AHL.

    Clubs can sit at 23 with illegal composition (e.g. 11F/9D/3G). Position
    floors must win: demote the weakest surplus D/G/F above its minimum, then
    the fill pass can recall the missing bucket.
    """
    from services.roster_compliance import (
        MIN_DEFENSE,
        MIN_FORWARDS,
        MIN_GOALIES,
        is_active_nhl_roster_player,
        position_bucket,
    )

    mins = {"F": MIN_FORWARDS, "D": MIN_DEFENSE, "G": MIN_GOALIES}
    roster = list(_get(team, "roster", None) or [])
    active = [p for p in roster if is_active_nhl_roster_player(p)]
    counts = {"F": 0, "D": 0, "G": 0}
    for p in active:
        b = position_bucket(p)
        if b in counts:
            counts[b] += 1

    surplus_buckets = [
        b for b in ("F", "D", "G")
        if b != need and counts[b] > mins[b]
    ]
    if not surplus_buckets:
        return False

    # Prefer demoting from the bucket with the largest surplus, lowest OVR.
    surplus_buckets.sort(key=lambda b: (-(counts[b] - mins[b]), b))
    target_bucket = surplus_buckets[0]
    candidates = [p for p in active if position_bucket(p) == target_bucket]
    if not candidates:
        return False
    pick = min(candidates, key=_player_ovr)
    pid = _player_id(pick)
    remaining = [p for p in roster if not (p is pick or _player_id(p) == pid)]
    if len(remaining) == len(roster):
        return False
    try:
        pick.in_minors = True
        pick.is_buried = False
        pick.buried = False
        pick.roster_location = "ahl"
        ahl = list(_get(team, "ahl_roster", None) or [])
        ahl.append(pick)
        setattr(team, "roster", remaining)
        setattr(team, "ahl_roster", ahl)
    except Exception:
        return False
    sync_team_cap_fields(team, league)
    return True


def _recall_affiliate_player(team: Any, player: Any, league: Any, source_attr: str) -> bool:
    """Move an affiliate SPC onto the NHL roster. Returns False if the lists reject it."""
    source = list(_get(team, source_attr, None) or [])
    pid = _player_id(player)
    remaining = [p for p in source if _player_id(p) != pid or p is not player]
    if len(remaining) == len(source):
        return False
    roster = list(_get(team, "roster", None) or [])
    try:
        player.in_minors = False
        player.is_buried = False
        player.buried = False
        player.waiver_status = None
        player.roster_location = "nhl"
        roster.append(player)
        setattr(team, "roster", roster)
        setattr(team, source_attr, remaining)
    except Exception:
        return False
    sync_team_cap_fields(team, league)
    return True


def run_roster_fill_pass(session: Any, *, teams: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Recall affiliate SPCs until every club meets the active-roster floor.

    The compliance pipeline can only shed salary, so a club that lost players to
    expiry, waivers or buyouts enters the new season below 20 men while its AHL
    affiliate sits full. Roster legality outranks cap comfort here: the cap
    pipeline runs on either side of this pass and takes relief where it can.

    Also repairs illegal composition at the 23-man ceiling (e.g. 11F/9D/3G) by
    demoting surplus depth before recalling the missing position.
    """
    from services.roster_compliance import (
        ACTIVE_ROSTER_MAX,
        ACTIVE_ROSTER_MIN,
        MIN_DEFENSE,
        MIN_FORWARDS,
        MIN_GOALIES,
        is_retired,
        position_bucket,
        summarize_team_roster_capacity,
    )

    league = _get(getattr(session, "sim", None), "league", None)
    team_list = list(teams if teams is not None else (_get(league, "teams", None) or []))
    recalls: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []

    for team in team_list:
        tid = str(_get(team, "team_id", "") or _get(team, "id", ""))
        for _ in range(ACTIVE_ROSTER_MAX * 3):
            cap = summarize_team_roster_capacity(team)
            # Position floors first — a club without two goalies cannot ice a game.
            if int(cap["goalies"]) < MIN_GOALIES:
                need = "G"
            elif int(cap["defense"]) < MIN_DEFENSE:
                need = "D"
            elif int(cap["forwards"]) < MIN_FORWARDS:
                need = "F"
            elif int(cap["nhl_count"]) < ACTIVE_ROSTER_MIN:
                need = "ANY"
            else:
                break

            # At the 23-man ceiling with illegal composition: demote surplus, then recall.
            if int(cap["nhl_count"]) >= ACTIVE_ROSTER_MAX:
                if need == "ANY":
                    break
                if _demote_surplus_for_roster_fill(team, need, league):
                    recalls.append({
                        "team_id": tid,
                        "player_id": None,
                        "source": "demote_surplus",
                        "need": need,
                    })
                    continue
                unresolved.append({
                    "team_id": tid,
                    "need": need,
                    "active": int(cap["nhl_count"]),
                    "composition": cap["composition"],
                    "reason": "full_roster_no_surplus",
                })
                break

            def _matches(p: Any) -> bool:
                if p is None or is_retired(p):
                    return False
                return need == "ANY" or position_bucket(p) == need

            # Cheapest legal source first: players already on the NHL list who
            # were buried for cap relief, then affiliate SPCs.
            buried = [
                p for p in (_get(team, "roster", None) or [])
                if _matches(p) and (_get(p, "is_buried", False) or _get(p, "in_minors", False))
            ]
            moved = False
            if buried:
                buried.sort(key=_fill_candidate_sort_key)
                pick = buried[0]
                try:
                    pick.is_buried = False
                    pick.buried = False
                    pick.in_minors = False
                    pick.waiver_status = None
                    pick.roster_location = "nhl"
                    sync_team_cap_fields(team, league)
                    moved = True
                    recalls.append({
                        "team_id": tid,
                        "player_id": _player_id(pick),
                        "source": "buried",
                        "need": need,
                    })
                except Exception:
                    moved = False

            if not moved:
                for attr in ("ahl_roster", "echl_roster"):
                    pool = [
                        p for p in (_get(team, attr, None) or [])
                        if _matches(p) and uses_nhl_contract_slot(p)
                    ]
                    if not pool:
                        # Floor legality outranks two-way bookkeeping — promote any match.
                        pool = [p for p in (_get(team, attr, None) or []) if _matches(p)]
                    if not pool:
                        continue
                    pool.sort(key=_fill_candidate_sort_key)
                    if _recall_affiliate_player(team, pool[0], league, attr):
                        moved = True
                        recalls.append({
                            "team_id": tid,
                            "player_id": _player_id(pool[0]),
                            "source": attr,
                            "need": need,
                        })
                        break

            if not moved:
                # Last resort: sign a cheap free agent so clubs can ice a legal roster.
                try:
                    fas = list(_get(league, "free_agents", None) or [])
                    fa_pool = [p for p in fas if _matches(p)]
                    fa_pool.sort(key=_fill_candidate_sort_key)
                    sy = int(_get(session, "season_calendar_year", 2025) or 2025)
                    for pick in fa_pool[:8]:
                        contract = {
                            "type": "STANDARD",
                            "contract_type": "STANDARD",
                            "years": 1,
                            "years_remaining": 1,
                            "aav_m": float(LEAGUE_MINIMUM_AAV_M),
                            "cap_hit_m": float(LEAGUE_MINIMUM_AAV_M),
                            "base_salary_m": float(LEAGUE_MINIMUM_AAV_M),
                            "salary_m": float(LEAGUE_MINIMUM_AAV_M),
                            "start_year": sy,
                            "expiry_year": sy + 1,
                            "rights_status": "UFA",
                            "rights": "UFA",
                        }
                        apply_contract_to_player(pick, contract, sy)
                        pick.roster_location = "nhl"
                        pick.in_minors = False
                        pick.is_buried = False
                        # FA pool can retain stale refs still listed on another club.
                        _strip_player_from_all_org_lists(league, pick)
                        roster = list(_get(team, "roster", None) or [])
                        if not any(_player_id(p) == _player_id(pick) for p in roster):
                            roster.append(pick)
                        setattr(team, "roster", roster)
                        try:
                            fas.remove(pick)
                        except ValueError:
                            fas = [p for p in fas if _player_id(p) != _player_id(pick)]
                        setattr(league, "free_agents", fas)
                        _remove_from_unsigned_pools(league, pick)
                        sync_team_cap_fields(team, league)
                        moved = True
                        recalls.append({
                            "team_id": tid,
                            "player_id": _player_id(pick),
                            "source": "free_agent",
                            "need": need,
                        })
                        break
                except Exception:
                    moved = False

            if not moved:
                cap = summarize_team_roster_capacity(team)
                unresolved.append({
                    "team_id": tid,
                    "need": need,
                    "active": int(cap["nhl_count"]),
                    "composition": cap["composition"],
                })
                break

    # Safety: one NHL roster seat per player id across the league.
    seen: Dict[str, Any] = {}
    for team in team_list:
        roster = list(_get(team, "roster", None) or [])
        kept: List[Any] = []
        changed = False
        for p in roster:
            pid = _player_id(p)
            if pid and pid in seen:
                changed = True
                continue
            if pid:
                seen[pid] = team
            kept.append(p)
        if changed:
            setattr(team, "roster", kept)
            sync_team_cap_fields(team, league)

    return {
        "recalls": recalls,
        "recall_count": len(recalls),
        "unresolved": unresolved,
        "teams_filled": len({r["team_id"] for r in recalls if r.get("team_id")}),
    }


def run_cap_compliance_before_season(session: Any, *, allow_rebalance: bool = False) -> Dict[str, Any]:
    pipeline = run_cap_compliance_pipeline(session, include_buyouts=True)
    actions = []
    for key in ("exempt_sent_down", "buried", "waived", "claims", "cleared", "buyouts"):
        rows = pipeline.get(key) or []
        if rows:
            actions.append({"action": key, "count": len(rows)})
    fill = run_roster_fill_pass(session)
    if fill.get("recall_count"):
        actions.append({"action": "affiliate_recalls", "count": fill["recall_count"]})
    if allow_rebalance:
        league = getattr(session.sim, "league", None)
        season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
        for team in _get(league, "teams", None) or []:
            snap = get_team_cap_snapshot_full(team, league, session.sim, season_year=season_year)
            if snap["usable_cap_space_m"] < -0.01:
                rebalance_team_cap_at_bootstrap(team, league, season_year, session.sim.rng)
                actions.append({"team_id": _get(team, "team_id", ""), "action": "bootstrap_trim"})
    return {"actions": actions, "pipeline": pipeline, "roster_fill": fill}
