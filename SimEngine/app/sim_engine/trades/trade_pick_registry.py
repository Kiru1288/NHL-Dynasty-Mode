"""
League-wide draft pick ownership registry for season trades.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.sim_engine.trades.trade_asset import canonical_pick_id, _team_slug


def _safe_str(x: Any, default: str = "") -> str:
    return str(x) if x is not None else default


def upcoming_draft_year(season_calendar_year: int) -> int:
    """NHL Entry Draft year for a franchise season labeled by calendar year.

    Example: 2025-26 season (season_calendar_year=2025) → June 2026 draft.
    """
    return int(season_calendar_year) + 1


def tradeable_draft_year(season_calendar_year: int, *, draft_completed: bool = False) -> int:
    """Earliest draft year still available to trade.

    Once the Entry Draft for a year finishes, that year's picks are consumed and the
    trade market moves to the following draft class.
    """
    dy = upcoming_draft_year(int(season_calendar_year))
    return dy + 1 if draft_completed else dy


def draft_year_from_context(
    context: Optional[Dict[str, Any]] = None,
    *,
    league: Any = None,
) -> int:
    """Resolve the upcoming Entry Draft year from trade context / league."""
    ctx = context or {}
    if ctx.get("tradeable_draft_year") is not None:
        try:
            return int(ctx["tradeable_draft_year"])
        except (TypeError, ValueError):
            pass
    raw = ctx.get("draft_year")
    if raw is not None:
        try:
            dy = int(raw)
            if ctx.get("draft_completed"):
                return dy + 1
            return dy
        except (TypeError, ValueError):
            pass
    sy = ctx.get("season_year")
    if sy is None and league is not None:
        sy = getattr(league, "current_season", None) or getattr(league, "season_year", None)
    if sy is None:
        sy = 2025
    completed = bool(ctx.get("draft_completed") or getattr(league, "draft_completed", False))
    # Franchise contexts mark calendar seasons explicitly; otherwise treat as legacy draft-year anchor.
    if ctx.get("season_is_calendar") or ctx.get("use_upcoming_draft_year"):
        return tradeable_draft_year(int(sy), draft_completed=completed)
    return int(sy) + (1 if completed else 0)


def _get_registry(league: Any) -> Dict[str, Dict[str, Any]]:
    reg = getattr(league, "draft_pick_registry", None)
    if not isinstance(reg, dict):
        reg = {}
        setattr(league, "draft_pick_registry", reg)
    return reg


def _get_team_pick_ids(team: Any) -> List[str]:
    ids = getattr(team, "owned_pick_ids", None)
    if not isinstance(ids, list):
        ids = []
        setattr(team, "owned_pick_ids", ids)
    return ids


def _entity_team_id(t: Any) -> Any:
    # team_id=0 is valid; only fall back to "id" when team_id is truly absent.
    tid = getattr(t, "team_id", None)
    if tid is None:
        tid = getattr(t, "id", None)
    return tid


def _team_ids_from_league(league: Any) -> List[str]:
    teams = list(getattr(league, "teams", None) or [])
    out: List[str] = []
    for t in teams:
        tid = _safe_str(_entity_team_id(t))
        if tid:
            out.append(tid)
    return out


def prune_expired_unresolved_picks(league: Any, *, before_year: int) -> int:
    """Mark unresolved picks with year < before_year as resolved (past drafts)."""
    if league is None:
        return 0
    before = int(before_year)
    reg = _get_registry(league)
    n = 0
    for _pid, row in reg.items():
        if not isinstance(row, dict) or bool(row.get("resolved")):
            continue
        try:
            y = int(row.get("year") or 0)
        except (TypeError, ValueError):
            continue
        if y < before:
            row["resolved"] = True
            row["resolved_reason"] = "past_draft"
            n += 1
    if n:
        setattr(league, "draft_pick_registry", reg)
        reconcile_pick_registry_consistency(league)
    return n


def retire_draft_year_picks(
    league: Any,
    *,
    draft_year: int,
    reason: str = "draft_completed",
) -> int:
    """Mark every unresolved pick for draft_year as resolved and drop from owned lists.

    Call after the Entry Draft finishes (or for any leftover slots) so Trade Hub and
    trade rules can no longer surface that year's capital.
    """
    if league is None:
        return 0
    dy = int(draft_year)
    reg = _get_registry(league)
    n = 0
    for _pid, row in reg.items():
        if not isinstance(row, dict) or bool(row.get("resolved")):
            continue
        try:
            y = int(row.get("year") or 0)
        except (TypeError, ValueError):
            continue
        if y != dy:
            continue
        row["resolved"] = True
        row["resolved_reason"] = str(reason or "draft_completed")
        n += 1
    if n:
        setattr(league, "draft_pick_registry", reg)
    reconcile_pick_registry_consistency(league)
    # Ensure the next draft class exists for trading immediately after the draft.
    ensure_draft_pick_registry(league, start_year=dy + 1, years_ahead=4)
    return n


def migrate_calendar_year_picks_to_draft_year(
    league: Any,
    *,
    season_calendar_year: int,
    draft_year: Optional[int] = None,
) -> int:
    """Copy ownership from phantom calendar-year picks onto upcoming draft-year slots.

    Older trade paths minted/traded picks keyed by season_calendar_year, but Entry Draft
    consumes season_calendar_year + 1. Without migration, traded picks never appear on the board.
    """
    if league is None:
        return 0
    sy = int(season_calendar_year)
    dy = int(draft_year) if draft_year is not None else upcoming_draft_year(sy)
    if dy != sy + 1:
        return 0

    reg = _get_registry(league)
    migrated = 0
    for pid, row in list(reg.items()):
        if not isinstance(row, dict) or bool(row.get("resolved")):
            continue
        try:
            y = int(row.get("year") or 0)
            rnd = int(row.get("round") or 0)
        except (TypeError, ValueError):
            continue
        if y != sy or rnd < 1:
            continue
        orig = _safe_str(row.get("original_team_id"))
        owner = _safe_str(row.get("current_owner_team_id")) or orig
        if not orig:
            row["resolved"] = True
            row["resolved_reason"] = "invalid_calendar_pick"
            continue
        new_id = canonical_pick_id(dy, rnd, orig)
        target = reg.get(new_id)
        if not isinstance(target, dict):
            reg[new_id] = {
                "pick_id": new_id,
                "year": dy,
                "round": rnd,
                "original_team_id": orig,
                "current_owner_team_id": owner,
                "protection": row.get("protection"),
                "conditions": row.get("conditions"),
                "resolved": False,
                "migrated_from": _safe_str(pid),
            }
            migrated += 1
        else:
            target_owner = _safe_str(target.get("current_owner_team_id")) or orig
            # Move traded ownership onto the draft-year pick when it is still with the original club.
            if owner != orig and target_owner == orig:
                target["current_owner_team_id"] = owner
                if row.get("protection") and not target.get("protection"):
                    target["protection"] = row.get("protection")
                if row.get("conditions") and not target.get("conditions"):
                    target["conditions"] = row.get("conditions")
                target["migrated_from"] = _safe_str(pid)
                migrated += 1
        row["resolved"] = True
        row["resolved_reason"] = "migrated_to_draft_year"
        row["migrated_to"] = new_id

    if migrated or any(
        isinstance(r, dict) and r.get("resolved_reason") == "migrated_to_draft_year"
        for r in reg.values()
    ):
        setattr(league, "draft_pick_registry", reg)
        reconcile_pick_registry_consistency(league)
    return migrated


def ensure_franchise_pick_registry(
    league: Any,
    *,
    season_calendar_year: int,
    years_ahead: int = 4,
    rounds: int = 7,
    draft_completed: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Mint picks from the upcoming draft year, migrate mis-year ownership, prune past drafts."""
    sy = int(season_calendar_year)
    dy = upcoming_draft_year(sy)
    migrate_calendar_year_picks_to_draft_year(league, season_calendar_year=sy, draft_year=dy)
    if draft_completed:
        retire_draft_year_picks(league, draft_year=dy, reason="draft_completed")
        start = dy + 1
    else:
        start = dy
    reg = ensure_draft_pick_registry(league, start_year=start, years_ahead=years_ahead, rounds=rounds)
    prune_expired_unresolved_picks(league, before_year=start)
    reconcile_pick_registry_consistency(league)
    return reg


def ensure_draft_pick_registry(
    league: Any,
    *,
    start_year: Optional[int] = None,
    years_ahead: int = 4,
    rounds: int = 7,
) -> Dict[str, Dict[str, Any]]:
    """Initialize pick registry and team ownership lists if missing."""
    if league is None:
        return {}

    reg = _get_registry(league)
    team_ids = _team_ids_from_league(league)
    if not team_ids:
        return reg

    if start_year is None:
        # Prefer an explicit draft year on the league; else assume calendar season → upcoming draft.
        explicit = getattr(league, "draft_year", None)
        if explicit is not None:
            start_year = int(explicit)
        else:
            cal = int(getattr(league, "current_season", 0) or getattr(league, "season_year", 0) or 2025)
            start_year = upcoming_draft_year(cal) if getattr(league, "season_is_calendar", True) else cal
    start_year = int(start_year)

    for tid in team_ids:
        team = _find_team(league, tid)
        owned = _get_team_pick_ids(team) if team is not None else []

        for offset in range(years_ahead):
            year = start_year + offset
            for rnd in range(1, rounds + 1):
                pick_id = canonical_pick_id(year, rnd, tid)
                existing = reg.get(pick_id)
                if isinstance(existing, dict):
                    if (
                        int(existing.get("year", 0)) != int(year)
                        or int(existing.get("round", 0)) != int(rnd)
                        or _safe_str(existing.get("original_team_id")) != tid
                    ):
                        raise ValueError(f"Draft pick id collision detected for {pick_id}")
                    continue
                reg[pick_id] = {
                    "pick_id": pick_id,
                    "year": year,
                    "round": rnd,
                    "original_team_id": tid,
                    "current_owner_team_id": tid,
                    "protection": None,
                    "conditions": None,
                    "resolved": False,
                }
                if pick_id not in owned:
                    owned.append(pick_id)

        if team is not None:
            setattr(team, "owned_pick_ids", owned)

    setattr(league, "draft_pick_registry", reg)
    reconcile_pick_registry_consistency(league)
    return reg


def _find_team(league: Any, team_id: str) -> Any:
    tid = _safe_str(team_id)
    for t in getattr(league, "teams", None) or []:
        if _safe_str(_entity_team_id(t)) == tid:
            return t
    return None


def get_pick_by_id(league: Any, pick_id: str) -> Optional[Dict[str, Any]]:
    if league is None:
        return None
    reg = _get_registry(league)
    row = reg.get(_safe_str(pick_id))
    return dict(row) if isinstance(row, dict) else None


def validate_pick_ownership(league: Any, pick_id: str, source_team_id: str) -> bool:
    row = get_pick_by_id(league, pick_id)
    if not row:
        return False
    if bool(row.get("resolved")):
        return False
    return _safe_str(row.get("current_owner_team_id")) == _safe_str(source_team_id)


def get_team_owned_picks(
    league: Any,
    team_id: str,
    *,
    min_year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ensure_draft_pick_registry(league)
    reconcile_pick_registry_consistency(league)
    reg = _get_registry(league)
    tid = _safe_str(team_id)
    team = _find_team(league, tid)
    owned_ids = list(_get_team_pick_ids(team)) if team is not None else []
    out: List[Dict[str, Any]] = []
    floor = int(min_year) if min_year is not None else None
    for pid in owned_ids:
        row = reg.get(pid)
        if not isinstance(row, dict) or row.get("resolved"):
            continue
        if floor is not None:
            try:
                if int(row.get("year") or 0) < floor:
                    continue
            except (TypeError, ValueError):
                continue
        out.append(dict(row))
    out.sort(key=lambda r: (int(r.get("year", 0)), int(r.get("round", 0)), str(r.get("pick_id", ""))))
    return out


def sync_owned_pick_ids_from_registry(league: Any) -> Dict[str, int]:
    """Rebuild every team's owned_pick_ids from registry current_owner_team_id."""
    return reconcile_pick_registry_consistency(league)


def transfer_pick(league: Any, pick_id: str, new_owner_team_id: str) -> Dict[str, Any]:
    reg = _get_registry(league)
    pid = _safe_str(pick_id)
    row = reg.get(pid)
    if not row:
        raise ValueError(f"Pick not found in registry: {pid}")
    if bool(row.get("resolved")):
        raise ValueError(f"Cannot transfer resolved pick: {pid}")

    old_owner = _safe_str(row.get("current_owner_team_id"))
    new_owner = _safe_str(new_owner_team_id)
    if old_owner == new_owner:
        return dict(row)

    old_team = _find_team(league, old_owner)
    if old_team is not None:
        owned = _get_team_pick_ids(old_team)
        if pid not in owned:
            raise ValueError(
                f"Registry/list desync for pick {pid}: registry owner {old_owner} "
                "does not have pick in owned_pick_ids"
            )
        owned.remove(pid)
        setattr(old_team, "owned_pick_ids", owned)

    new_team = _find_team(league, new_owner)
    if new_team is None:
        raise ValueError(f"New owner team not found: {new_owner}")
    owned = _get_team_pick_ids(new_team)
    if pid not in owned:
        owned.append(pid)
        setattr(new_team, "owned_pick_ids", sorted(set(_safe_str(x) for x in owned)))

    row["current_owner_team_id"] = new_owner
    reg[pid] = row
    setattr(league, "draft_pick_registry", reg)

    try:
        import logging

        logging.getLogger(__name__).info(
            "PICK TRANSFER pick_id=%s year=%s round=%s original=%s from=%s to=%s",
            pid,
            row.get("year"),
            row.get("round"),
            row.get("original_team_id"),
            old_owner,
            new_owner,
        )
    except Exception:
        pass

    return dict(row)


def reconcile_pick_registry_consistency(league: Any) -> Dict[str, int]:
    """Rebuild team owned_pick_ids from unresolved registry ownership rows."""
    reg = _get_registry(league)
    if not reg:
        return {"rows": 0, "teams": 0}
    team_ids = _team_ids_from_league(league)
    owned_by_team: Dict[str, List[str]] = {tid: [] for tid in team_ids}
    for pid, row in reg.items():
        if not isinstance(row, dict):
            continue
        if bool(row.get("resolved")):
            continue
        owner = _safe_str(row.get("current_owner_team_id"))
        if not owner:
            continue
        owned_by_team.setdefault(owner, []).append(_safe_str(pid))
    for tid, picks in owned_by_team.items():
        team = _find_team(league, tid)
        if team is None:
            continue
        deduped = sorted(set(picks))
        setattr(team, "owned_pick_ids", deduped)
    return {"rows": len(reg), "teams": len(owned_by_team)}


def audit_pick_registry_integrity(
    league: Any,
    *,
    start_year: Optional[int] = None,
    years_ahead: int = 4,
    rounds: int = 7,
) -> Dict[str, Any]:
    """Validate registry invariants and return machine-readable failures."""
    ensure_draft_pick_registry(league, start_year=start_year, years_ahead=years_ahead, rounds=rounds)
    reg = _get_registry(league)
    team_ids = _team_ids_from_league(league)
    errors: List[str] = []
    warnings: List[str] = []
    owner_to_ids: Dict[str, List[str]] = {}
    seen_ids: set[str] = set()
    required = (
        "pick_id",
        "year",
        "round",
        "original_team_id",
        "current_owner_team_id",
        "resolved",
    )

    for pid, row in reg.items():
        if pid in seen_ids:
            errors.append(f"Duplicate pick_id found in registry: {pid}")
        seen_ids.add(pid)
        if not isinstance(row, dict):
            errors.append(f"Registry row is not a dict for pick_id={pid}")
            continue
        for key in required:
            if key not in row:
                errors.append(f"Missing required field {key} for pick_id={pid}")
        if _safe_str(row.get("pick_id")) != _safe_str(pid):
            errors.append(f"pick_id key mismatch for {pid} (row has {_safe_str(row.get('pick_id'))})")
        try:
            yr = int(row.get("year", 0))
            rnd = int(row.get("round", 0))
            if yr <= 0:
                errors.append(f"Invalid year for pick_id={pid}: {row.get('year')}")
            if rnd < 1 or rnd > int(rounds):
                errors.append(f"Invalid round for pick_id={pid}: {row.get('round')}")
        except Exception:
            errors.append(f"Non-numeric year/round for pick_id={pid}")
        owner = _safe_str(row.get("current_owner_team_id"))
        if not owner:
            errors.append(f"Missing current_owner_team_id for pick_id={pid}")
            continue
        if owner not in team_ids:
            warnings.append(f"Owner team id not found in league for pick_id={pid}: {owner}")
        if not bool(row.get("resolved")):
            owner_to_ids.setdefault(owner, []).append(_safe_str(pid))

    for tid in team_ids:
        team = _find_team(league, tid)
        owned_ids = list(_get_team_pick_ids(team)) if team is not None else []
        expected = sorted(set(owner_to_ids.get(tid, [])))
        actual = sorted(set(_safe_str(x) for x in owned_ids))
        if expected != actual:
            errors.append(
                f"owned_pick_ids mismatch for team={tid}: expected={len(expected)} actual={len(actual)}"
            )
        dup_count = len(owned_ids) - len(set(_safe_str(x) for x in owned_ids))
        if dup_count > 0:
            errors.append(f"Duplicate pick IDs in owned_pick_ids for team={tid}: {dup_count}")

    if start_year is not None:
        base = int(start_year)
        horizon = {base + off for off in range(max(0, int(years_ahead)))}
        # Count resolved + unresolved rows. Mid-draft selections mark picks resolved;
        # uniqueness still requires exactly one row per (year, round, original team).
        slot_counts: Dict[tuple[int, int, str], int] = {}
        for pid, row in reg.items():
            if not isinstance(row, dict):
                continue
            try:
                y = int(row.get("year", 0))
                r = int(row.get("round", 0))
            except (TypeError, ValueError):
                continue
            orig = _safe_str(row.get("original_team_id"))
            if y in horizon and 1 <= r <= int(rounds) and orig:
                key = (y, r, orig)
                slot_counts[key] = slot_counts.get(key, 0) + 1
                if _safe_str(row.get("pick_id")) != _safe_str(pid):
                    errors.append(f"pick_id mismatch in registry row for {pid}")
        for tid in team_ids:
            for offset in range(max(0, int(years_ahead))):
                year = base + offset
                for rnd in range(1, int(rounds) + 1):
                    key = (year, rnd, tid)
                    count = slot_counts.get(key, 0)
                    if count != 1:
                        errors.append(
                            f"Registry expected 1 pick for original_team={tid} year={year} round={rnd}, found {count}"
                        )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "registry_count": len(reg),
        "team_count": len(team_ids),
    }


def _pick_display(row: Dict[str, Any]) -> str:
    year = int(row.get("year", 0))
    rnd = int(row.get("round", 0))
    orig = _team_slug(_safe_str(row.get("original_team_id")))
    return f"{year} {orig} Rd {rnd}"


def team_owns_own_first(
    league: Any,
    team_id: str,
    *,
    draft_year: int,
) -> Dict[str, Any]:
    """Return whether team_id owns its own unprotected first-round pick for draft_year."""
    ensure_draft_pick_registry(league, start_year=max(2020, int(draft_year) - 1), years_ahead=4)
    tid = _safe_str(team_id)
    pick_id = canonical_pick_id(int(draft_year), 1, tid)
    row = get_pick_by_id(league, pick_id)
    if not row:
        return {
            "owns_own_first": False,
            "owns_protected_first": False,
            "pick_ownership_reason": "unknown",
            "pick_id": pick_id,
        }

    orig = _safe_str(row.get("original_team_id"))
    owner = _safe_str(row.get("current_owner_team_id"))
    protection = row.get("protection")
    has_protection = protection not in (None, "", "none", "unprotected")

    if orig == tid and owner == tid:
        if has_protection:
            return {
                "owns_own_first": True,
                "owns_protected_first": True,
                "pick_ownership_reason": "protected_pick",
                "pick_id": pick_id,
                "protection": protection,
            }
        return {
            "owns_own_first": True,
            "owns_protected_first": False,
            "pick_ownership_reason": "owns_own_first",
            "pick_id": pick_id,
        }

    if orig == tid and owner != tid:
        return {
            "owns_own_first": False,
            "owns_protected_first": False,
            "pick_ownership_reason": "pick_traded",
            "pick_id": pick_id,
            "current_owner_team_id": owner,
        }

    if owner == tid and orig != tid:
        return {
            "owns_own_first": False,
            "owns_protected_first": False,
            "pick_ownership_reason": "pick_traded",
            "pick_id": pick_id,
            "original_team_id": orig,
        }

    return {
        "owns_own_first": False,
        "owns_protected_first": False,
        "pick_ownership_reason": "unknown",
        "pick_id": pick_id,
    }


def serialize_team_picks(
    league: Any,
    team_id: str,
    *,
    value_hint_fn: Optional[Any] = None,
    min_year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = get_team_owned_picks(league, team_id, min_year=min_year)
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = {
            "pick_id": row.get("pick_id"),
            "id": row.get("pick_id"),
            "year": row.get("year"),
            "round": row.get("round"),
            "draftYear": row.get("year"),
            "original_team_id": row.get("original_team_id"),
            "current_owner_team_id": row.get("current_owner_team_id"),
            "owner": row.get("current_owner_team_id"),
            "display": _pick_display(row),
            "label": _pick_display(row),
            "protection": row.get("protection"),
            "conditions": row.get("conditions"),
            "resolved": False,
        }
        if value_hint_fn is not None:
            try:
                item["value_hint"] = round(float(value_hint_fn(row)), 1)
            except Exception:
                item["value_hint"] = None
        out.append(item)
    return out
