"""Single source of truth for NHL active-roster compliance and capacity.

Active roster excludes retired, buried/minors, IR, and LTIR. Scratches still count.
Position codes unwrap Position enums ("Position.C" must never leak into comparisons).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

ACTIVE_ROSTER_MAX = 23
ACTIVE_ROSTER_MIN = 20
MIN_FORWARDS = 12
MIN_DEFENSE = 6
MIN_GOALIES = 2

FORWARD_CODES = frozenset({"C", "LW", "RW", "W", "F"})
DEFENSE_CODES = frozenset({"D", "LD", "RD", "LHD", "RHD"})
GOALIE_CODES = frozenset({"G"})


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def position_code(player: Any) -> str:
    """Normalize player position to "C"/"LW"/"RW"/"D"/"G" (etc.)."""
    pos = _get(player, "position", None)
    if pos is None:
        ident = _get(player, "identity", None)
        pos = _get(ident, "position", None) if ident is not None else None
    if pos is None and isinstance(player, dict):
        pos = player.get("position")
    return str(getattr(pos, "value", pos) or "").strip().upper()


def position_bucket(player: Any) -> str:
    """Coarse bucket: F, D, G, or OTHER."""
    code = position_code(player)
    if code in FORWARD_CODES:
        return "F"
    if code in DEFENSE_CODES:
        return "D"
    if code in GOALIE_CODES:
        return "G"
    return "OTHER"


def _injury_status_token(player: Any) -> str:
    health = _get(player, "health", None)
    raw = (
        _get(player, "injury_status", None)
        or _get(health, "injury_status", None)
        or _get(player, "status", None)
        or ""
    )
    return str(getattr(raw, "value", raw) or "").strip().upper()


def is_retired(player: Any) -> bool:
    return bool(_get(player, "retired", False))


def is_buried_or_minors(player: Any) -> bool:
    return bool(
        _get(player, "is_buried", False)
        or _get(player, "buried", False)
        or _get(player, "in_minors", False)
    )


def is_on_ir(player: Any) -> bool:
    if bool(_get(player, "on_ir", False) or _get(player, "is_ir", False) or _get(player, "ir", False)):
        return True
    status = _injury_status_token(player)
    if status in ("IR", "INJURED_RESERVE", "INJURED-RESERVE"):
        return True
    # Plain INJURED / OUT without IR flag still occupies an active spot unless
    # the club has formally placed the player on IR/LTIR.
    return False


def is_on_ltir(player: Any) -> bool:
    if bool(
        _get(player, "on_ltir", False)
        or _get(player, "is_ltir", False)
        or _get(player, "ltir", False)
        or _get(player, "excluded_from_cap_while_ltir", False)
    ):
        return True
    status = _injury_status_token(player)
    return status in ("LTIR", "LONG_TERM_IR", "LONG-TERM IR")


def is_active_nhl_roster_player(player: Any) -> bool:
    """True when the player occupies one of the club's 23 active NHL spots."""
    if player is None or is_retired(player):
        return False
    if is_buried_or_minors(player):
        return False
    if is_on_ir(player) or is_on_ltir(player):
        return False
    return True


def iter_team_nhl_roster(team: Any) -> List[Any]:
    return list(_get(team, "roster", None) or [])


def iter_active_nhl_roster(team: Any) -> List[Any]:
    return [p for p in iter_team_nhl_roster(team) if is_active_nhl_roster_player(p)]


def _player_on_injured_reserve_list(team: Any, player: Any) -> bool:
    ir_list = list(_get(team, "injured_reserve", None) or [])
    if not ir_list or player is None:
        return False
    pid = str(_get(player, "id", "") or _get(player, "player_id", "") or "")
    if not pid:
        return player in ir_list
    for entry in ir_list:
        if entry is player:
            return True
        eid = str(_get(entry, "id", "") or _get(entry, "player_id", "") or "")
        if eid and eid == pid:
            return True
    return False


def summarize_team_roster_capacity(team: Any) -> Dict[str, Any]:
    """Roster + composition capacity for Trade Hub / Roster Check / Cap Ledger."""
    roster = iter_team_nhl_roster(team)
    active: List[Any] = []
    ir_players: List[Any] = []
    ltir_players: List[Any] = []
    buried = 0

    for p in roster:
        if is_retired(p):
            continue
        if is_buried_or_minors(p):
            buried += 1
            continue
        on_ir = is_on_ir(p) or _player_on_injured_reserve_list(team, p)
        on_ltir = is_on_ltir(p)
        if on_ltir:
            ltir_players.append(p)
            continue
        if on_ir:
            ir_players.append(p)
            continue
        active.append(p)

    forwards = sum(1 for p in active if position_bucket(p) == "F")
    defense = sum(1 for p in active if position_bucket(p) == "D")
    goalies = sum(1 for p in active if position_bucket(p) == "G")
    nhl_count = len(active)
    nhl_available = max(0, ACTIVE_ROSTER_MAX - nhl_count)
    ahl_count = len([p for p in (_get(team, "ahl_roster", None) or []) if not is_retired(p)])
    echl_count = len([p for p in (_get(team, "echl_roster", None) or []) if not is_retired(p)])

    return {
        "nhl_count": nhl_count,
        "nhl_max": ACTIVE_ROSTER_MAX,
        "nhl_min": ACTIVE_ROSTER_MIN,
        "nhl_available": nhl_available,
        "forwards": forwards,
        "defense": defense,
        "goalies": goalies,
        "composition": f"{forwards}F · {defense}D · {goalies}G",
        "ahl_count": ahl_count,
        "echl_count": echl_count,
        "active_roster_count": nhl_count,
        "active_roster_max": ACTIVE_ROSTER_MAX,
        "ir_count": len(ir_players),
        "ltir_count": len(ltir_players),
        "buried_count": buried,
        "raw_roster_count": len(roster),
        "min_forwards": MIN_FORWARDS,
        "min_defense": MIN_DEFENSE,
        "min_goalies": MIN_GOALIES,
    }


def summarize_team_contract_slots(team: Any, *, league: Any = None) -> Dict[str, Any]:
    """SPC / 50-slot usage — delegates to contract_economy so rules stay unified."""
    from services.contract_economy import (
        CONTRACT_SLOTS_LIMIT,
        validate_contract_slots,
        _count_team_contract_slots,
    )

    used = int(_count_team_contract_slots(team))
    limit = int(CONTRACT_SLOTS_LIMIT)
    slots = validate_contract_slots(team, league, additional=0) if team is not None else {}
    used = int(slots.get("contract_slots_used") or used)
    limit = int(slots.get("contract_slots_limit") or limit)
    return {
        "used": used,
        "limit": limit,
        "available": max(0, limit - used),
        "nhl_spcs_used": used,
        "ok": used <= limit,
    }


def evaluate_roster_compliance(
    team: Any,
    *,
    league: Any = None,
    sim: Any = None,
    season_year: Optional[int] = None,
    cap_snap: Optional[Dict[str, Any]] = None,
    cap_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Full compliance evaluation used by Roster Check / generate-next-season."""
    from services.contract_economy import get_team_cap_snapshot_full

    capacity = summarize_team_roster_capacity(team)
    slots = summarize_team_contract_slots(team, league=league)

    blocking: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    nhl_count = int(capacity["nhl_count"])
    forwards = int(capacity["forwards"])
    defense = int(capacity["defense"])
    goalies = int(capacity["goalies"])

    if nhl_count > ACTIVE_ROSTER_MAX:
        blocking.append({
            "code": "roster_max",
            "message": f"Active NHL roster over limit ({nhl_count}/{ACTIVE_ROSTER_MAX})",
            "route": "roster",
        })
    if nhl_count < ACTIVE_ROSTER_MIN:
        blocking.append({
            "code": "roster_min",
            "message": f"Active NHL roster under minimum ({nhl_count}/{ACTIVE_ROSTER_MIN})",
            "route": "free_agency",
        })
    if forwards < MIN_FORWARDS:
        blocking.append({
            "code": "forward_depth",
            "message": f"Need at least {MIN_FORWARDS} forwards ({forwards})",
            "route": "free_agency",
        })
    if defense < MIN_DEFENSE:
        blocking.append({
            "code": "defense_depth",
            "message": f"Need at least {MIN_DEFENSE} defensemen ({defense})",
            "route": "free_agency",
        })
    if goalies < MIN_GOALIES:
        blocking.append({
            "code": "goalie_depth",
            "message": f"Need at least {MIN_GOALIES} goalies ({goalies})",
            "route": "free_agency",
        })

    resolved_cap = dict(cap_snap or {})
    if cap_error:
        blocking.append({
            "code": "cap_check_failed",
            "message": f"Cap check failed: {cap_error}",
            "route": "cap_ledger",
        })
    else:
        if not resolved_cap and team is not None:
            try:
                year = int(season_year or 2025)
                resolved_cap = get_team_cap_snapshot_full(team, league, sim, season_year=year) or {}
            except Exception as exc:
                blocking.append({
                    "code": "cap_check_failed",
                    "message": f"Cap check failed: {exc}",
                    "route": "cap_ledger",
                })
                resolved_cap = {}
        if resolved_cap:
            try:
                space = float(resolved_cap.get("usable_cap_space_m") or 0)
            except (TypeError, ValueError):
                space = 0.0
            if space < -0.01:
                blocking.append({
                    "code": "cap_over",
                    "message": f"Over salary cap by ${abs(space):.2f}M",
                    "route": "cap_ledger",
                })

    if not slots.get("ok", True):
        used = int(slots.get("used") or 0)
        limit = int(slots.get("limit") or 50)
        blocking.append({
            "code": "contract_slots",
            "message": f"Contract slots exceeded ({used}/{limit})",
            "route": "cap_ledger",
        })

    # Soft warnings for near-miss depth after hard mins are met.
    if forwards >= MIN_FORWARDS and forwards < MIN_FORWARDS + 2:
        warnings.append({
            "code": "forward_thin",
            "message": f"Forward depth thin ({forwards})",
            "route": "free_agency",
        })
    if defense >= MIN_DEFENSE and defense < MIN_DEFENSE + 1:
        warnings.append({
            "code": "defense_thin",
            "message": f"Defense depth thin ({defense})",
            "route": "free_agency",
        })
    if int(capacity.get("ir_count") or 0) or int(capacity.get("ltir_count") or 0):
        warnings.append({
            "code": "injury_reserve",
            "message": (
                f"IR {capacity.get('ir_count', 0)} · LTIR {capacity.get('ltir_count', 0)} "
                f"(excluded from active roster)"
            ),
            "route": "roster",
        })

    valid = len(blocking) == 0
    return {
        "valid": valid,
        "blocking": blocking,
        "warnings": warnings,
        "capacity": capacity,
        "contract_slots": slots,
        "cap_snapshot": resolved_cap,
        "nhl_roster_count": nhl_count,
        "forward_count": forwards,
        "defense_count": defense,
        "goalie_count": goalies,
        "ir_count": int(capacity.get("ir_count") or 0),
        "ltir_count": int(capacity.get("ltir_count") or 0),
        "contract_slots_used": int(slots.get("used") or 0),
        "contract_slots_limit": int(slots.get("limit") or 50),
        "payroll_m": resolved_cap.get("total_cap_hit_m"),
        "cap_space_m": resolved_cap.get("usable_cap_space_m"),
        "blocking_reasons": [b["message"] for b in blocking],
        "warning_reasons": [w["message"] for w in warnings],
        "issues": [b["message"] for b in blocking],
        "warning_messages": [w["message"] for w in warnings],
    }
