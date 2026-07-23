from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_money_to_millions(value: Any) -> float:
    raw = _safe_float(value, 0.0)
    if abs(raw) > 250.0:
        return raw / 1_000_000.0
    return raw


def _season_label_from_year(year: Optional[int]) -> Optional[str]:
    if year is None:
        return None
    start = int(year)
    end = (start + 1) % 100
    return f"{start}-{end:02d}"


def _season_keys(season_label: Optional[str]) -> List[str]:
    out: List[str] = []
    if season_label:
        out.append(str(season_label))
    return out


def player_cap_hit_millions(player: Any) -> float:
    for key in ("cap_hit_m", "contract_aav_m", "aav_m", "salary_m"):
        v = normalize_money_to_millions(_get(player, key, 0))
        if v > 0:
            return v

    c = _get(player, "contract", None)
    if c is not None:
        for key in ("cap_hit_m", "cap_hit", "aav_m", "aav", "salary_aav"):
            v = normalize_money_to_millions(_get(c, key, 0))
            if v > 0:
                return v
    return 0.0


def buried_cap_hit_millions(player: Any, league_minimum_m: float = 0.775, burying_bonus_m: float = 0.375) -> float:
    cap_hit = player_cap_hit_millions(player)
    if cap_hit <= 0:
        return 0.0

    buried_flag = bool(_get(player, "is_buried", False) or _get(player, "buried", False) or _get(player, "in_minors", False))
    if not buried_flag:
        return 0.0

    relief = float(league_minimum_m) + float(burying_bonus_m)
    return max(0.0, cap_hit - relief)


def _iter_team_roster(team: Any) -> List[Any]:
    roster = _get(team, "roster", None) or []
    try:
        return list(roster)
    except Exception:
        return []


def _is_active_roster_player(player: Any) -> bool:
    return (
        not bool(_get(player, "retired", False))
        and not bool(_get(player, "is_buried", False))
        and not bool(_get(player, "buried", False))
        and not bool(_get(player, "in_minors", False))
    )


def team_active_roster_cap_hit_millions(team: Any) -> float:
    total = 0.0
    for p in _iter_team_roster(team):
        if not _is_active_roster_player(p):
            continue
        total += player_cap_hit_millions(p)
    return max(0.0, total)


def team_buried_cap_hit_millions(team: Any) -> float:
    total = 0.0
    for p in _iter_team_roster(team):
        if bool(_get(p, "retired", False)):
            continue
        total += buried_cap_hit_millions(p)
    return max(0.0, total)


def _sum_money_records_millions(records: Any, season_label: Optional[str] = None) -> float:
    if records is None:
        return 0.0
    keys = _season_keys(season_label)

    if isinstance(records, dict):
        for amount_key in ("amount_m", "cap_hit_m", "amount", "value", "cap_hit"):
            if amount_key in records:
                return normalize_money_to_millions(records.get(amount_key, 0))
        if keys:
            total = 0.0
            hit = False
            for k in keys:
                if k in records:
                    hit = True
                    total += _sum_money_records_millions(records.get(k), None)
            if hit:
                return total
        total = 0.0
        for v in records.values():
            total += _sum_money_records_millions(v, None)
        return total

    if isinstance(records, (list, tuple, set)):
        if keys:
            total = 0.0
            for item in records:
                if isinstance(item, dict):
                    try:
                        rem = item.get("seasons_remaining", item.get("remaining_seasons"))
                        if rem is not None and int(rem) <= 0:
                            continue
                    except (TypeError, ValueError):
                        pass
                    item_season = str(item.get("season", "") or "")
                    if item_season and item_season not in keys:
                        continue
                    total += normalize_money_to_millions(
                        item.get("amount_m", item.get("cap_hit_m", item.get("amount", item.get("value", 0))))
                    )
                else:
                    total += normalize_money_to_millions(item)
            return total

        # No season filter: a cap snapshot covers exactly ONE season, so season-tagged
        # rows (multi-year buyouts, retained salary) must NOT be summed across every
        # future season — that double-counts a buyout's annual hit once per remaining
        # year. Count season-less rows always, then add only the single worst season's
        # tagged total. Untagged numeric lists behave exactly as before.
        seasonless_total = 0.0
        per_season: Dict[str, float] = {}
        for item in records:
            if isinstance(item, dict):
                try:
                    rem = item.get("seasons_remaining", item.get("remaining_seasons"))
                    if rem is not None and int(rem) <= 0:
                        continue
                except (TypeError, ValueError):
                    pass
                amt = normalize_money_to_millions(
                    item.get("amount_m", item.get("cap_hit_m", item.get("amount", item.get("value", 0))))
                )
                item_season = str(item.get("season", "") or "")
                if item_season:
                    per_season[item_season] = per_season.get(item_season, 0.0) + amt
                else:
                    seasonless_total += amt
            else:
                seasonless_total += normalize_money_to_millions(item)
        worst_season = max(per_season.values()) if per_season else 0.0
        return seasonless_total + worst_season

    return normalize_money_to_millions(records)


def team_retained_salary_millions(team: Any, season_label: Optional[str] = None) -> float:
    for key in ("retained_salary", "retained_salaries", "retained_salary_records"):
        val = _get(team, key, None)
        if val is not None:
            return max(0.0, _sum_money_records_millions(val, season_label))
    return 0.0


def team_buyout_cap_hit_millions(team: Any, season_label: Optional[str] = None) -> float:
    for key in ("buyout_cap_hits", "buyout_cap_hit", "buyouts"):
        val = _get(team, key, None)
        if val is not None:
            return max(0.0, _sum_money_records_millions(val, season_label))
    return 0.0


def team_bonus_overage_millions(team: Any, season_label: Optional[str] = None) -> float:
    for key in ("bonus_overage", "bonus_overages"):
        val = _get(team, key, None)
        if val is not None:
            return max(0.0, _sum_money_records_millions(val, season_label))
    return 0.0


def team_performance_bonus_reserve_millions(team: Any) -> float:
    for key in ("performance_bonus_reserve_m", "performance_bonus_reserve", "bonus_reserve_m", "bonus_reserve"):
        val = _get(team, key, None)
        if val is not None:
            return max(0.0, normalize_money_to_millions(val))
    return 0.0


def team_ltir_pool_millions(team: Any) -> float:
    for key in ("ltir_pool_m", "ltir_pool", "ltir_relief_m", "ltir_relief"):
        val = _get(team, key, None)
        if val is not None:
            return max(0.0, normalize_money_to_millions(val))
    return 0.0


def _league_cap_bounds_millions(league: Any, sim: Any) -> Dict[str, float]:
    # First choice: explicit league attrs.
    if league is not None:
        upper = normalize_money_to_millions(
            _get(league, "salary_cap_m", _get(league, "salary_cap", _get(league, "upper_limit_m", 0)))
        )
        floor = normalize_money_to_millions(
            _get(league, "cap_floor_m", _get(league, "cap_floor", _get(league, "lower_limit_m", 0)))
        )
        if upper > 0:
            if floor <= 0:
                floor = upper * 0.74
            return {"upper": upper, "lower": floor}

        econ = _get(league, "economics", None)
        upper = normalize_money_to_millions(_get(econ, "salary_cap", 0))
        floor = normalize_money_to_millions(_get(econ, "cap_floor", 0))
        if upper > 0:
            if floor <= 0:
                floor = upper * 0.74
            return {"upper": upper, "lower": floor}

    # Fallback: sim context economics payload.
    econ = {}
    if sim is not None:
        try:
            econ = (_get(_get(sim, "league", None), "get_league_context", lambda: {})() or {}).get("economics") or {}
        except Exception:
            econ = {}
    upper = normalize_money_to_millions(econ.get("salary_cap", 92_000_000.0))
    lower = normalize_money_to_millions(econ.get("cap_floor", upper * 0.74))
    if lower <= 0:
        lower = upper * 0.74
    return {"upper": upper, "lower": lower}


def _retained_record_active(item: Any) -> bool:
    if not isinstance(item, dict):
        return True
    try:
        rem = item.get("seasons_remaining", item.get("remaining_seasons"))
        if rem is not None and int(rem) <= 0:
            return False
    except (TypeError, ValueError):
        pass
    return True


def cleanup_expired_retained_salary_records(team: Any) -> int:
    """Drop retention rows whose contract term has expired. Returns rows removed."""
    records = _get(team, "retained_salary_records", None)
    if not isinstance(records, list):
        return 0
    kept = [r for r in records if _retained_record_active(r)]
    removed = len(records) - len(kept)
    if removed:
        setattr(team, "retained_salary_records", kept)
    return removed


def decrement_retained_salary_seasons(team: Any) -> int:
    """Decrement seasons_remaining on active retention rows at season rollover."""
    records = _get(team, "retained_salary_records", None)
    if not isinstance(records, list):
        return 0
    expired = 0
    kept: List[Any] = []
    for item in records:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        try:
            rem = int(item.get("seasons_remaining", item.get("remaining_seasons", 1)) or 1)
        except (TypeError, ValueError):
            rem = 1
        rem -= 1
        if rem <= 0:
            expired += 1
            continue
        row = dict(item)
        row["seasons_remaining"] = rem
        kept.append(row)
    setattr(team, "retained_salary_records", kept)
    return expired


def cleanup_league_retained_salary_records(league: Any) -> Dict[str, int]:
    """Run retention cleanup across all teams in a league."""
    report = {"teams": 0, "removed": 0, "decremented": 0}
    for team in list(getattr(league, "teams", None) or []):
        report["teams"] += 1
        report["removed"] += cleanup_expired_retained_salary_records(team)
        report["decremented"] += decrement_retained_salary_seasons(team)
        report["removed"] += cleanup_expired_retained_salary_records(team)
    return report


def _retained_slots_used(team: Any, season_label: Optional[str] = None) -> int:
    records = _get(team, "retained_salary_records", None)
    if records is None:
        records = _get(team, "retained_salaries", None)
    if records is None:
        records = _get(team, "retained_salary", None)
    if records is None:
        return 0
    if isinstance(records, dict):
        rows = records.get(season_label) if season_label and season_label in records else records
        if isinstance(rows, dict):
            return len(rows)
        if isinstance(rows, (list, tuple, set)):
            return len(list(rows))
        return 1
    if isinstance(records, (list, tuple, set)):
        return len([r for r in records if _retained_record_active(r)])
    return 1


def calculate_team_cap_snapshot(
    team: Any,
    league: Any = None,
    sim: Any = None,
    season_label: Optional[str] = None,
    calendar_cursor: int = 0,
    regular_season_last_index: int = 192,
) -> Dict[str, Any]:
    bounds = _league_cap_bounds_millions(league, sim)
    upper_limit_m = max(0.0, bounds["upper"])
    lower_limit_m = max(0.0, bounds["lower"])

    active_m = team_active_roster_cap_hit_millions(team)
    if active_m <= 0.0:
        active_m = max(0.0, normalize_money_to_millions(_get(team, "total_cap_hit", 0.0)))
    buried_m = team_buried_cap_hit_millions(team)
    retained_m = team_retained_salary_millions(team, season_label=season_label)
    buyout_m = team_buyout_cap_hit_millions(team, season_label=season_label)
    bonus_overage_m = team_bonus_overage_millions(team, season_label=season_label)
    bonus_reserve_m = team_performance_bonus_reserve_millions(team)
    ltir_pool_m = team_ltir_pool_millions(team)

    other_dead_m = max(0.0, normalize_money_to_millions(_get(team, "other_dead_cap_m", _get(team, "other_dead_cap", 0.0))))
    effective_limit_m = upper_limit_m + ltir_pool_m
    total_m = (
        active_m
        + buried_m
        + retained_m
        + buyout_m
        + bonus_overage_m
        + bonus_reserve_m
        + other_dead_m
    )

    real_cap_space_m = upper_limit_m - total_m
    usable_cap_space_m = effective_limit_m - total_m
    floor_space_m = total_m - lower_limit_m

    total_regular_days = max(1, int(regular_season_last_index))
    day_idx = max(0, int(calendar_cursor))
    days_remaining = max(0, total_regular_days - day_idx)
    remaining_pct = max(0.05, float(days_remaining) / float(total_regular_days))
    projected_deadline_space_m = max(0.0, usable_cap_space_m / remaining_pct)

    roster = _iter_team_roster(team)
    active_roster_players = [p for p in roster if _is_active_roster_player(p)]
    active_roster_count = len(active_roster_players)
    retained_slots_used = _retained_slots_used(team, season_label=season_label)
    retained_slots_max = 3

    is_using_ltir = bool(ltir_pool_m > 0 and total_m > upper_limit_m)
    dead_cap_total = buried_m + retained_m + buyout_m + other_dead_m

    warnings: List[str] = []
    if usable_cap_space_m < 0:
        warnings.append("Over Cap")
    if floor_space_m < 0:
        warnings.append("Below Floor")
    if 0 <= real_cap_space_m <= 2.5:
        warnings.append("Near Cap")
    if is_using_ltir:
        warnings.append("Using LTIR")
    if active_roster_count >= 23:
        warnings.append("Roster Full")
    if active_roster_count > 23:
        warnings.append("Too Many Players")
    if dead_cap_total > 5.0:
        warnings.append("Dead Cap Heavy")
    if bonus_reserve_m > real_cap_space_m and bonus_reserve_m > 0:
        warnings.append("Bonus Overage Risk")

    return {
        "season": season_label,
        "upperLimit": round(float(upper_limit_m), 3),
        "lowerLimit": round(float(lower_limit_m), 3),
        "activeRosterCapHit": round(float(active_m), 3),
        "buriedCapHit": round(float(buried_m), 3),
        "retainedSalary": round(float(retained_m), 3),
        "buyoutCapHit": round(float(buyout_m), 3),
        "bonusOverage": round(float(bonus_overage_m), 3),
        "performanceBonusReserve": round(float(bonus_reserve_m), 3),
        "otherDeadCap": round(float(other_dead_m), 3),
        "ltirPool": round(float(ltir_pool_m), 3),
        "effectiveCapLimit": round(float(effective_limit_m), 3),
        "totalCapHit": round(float(total_m), 3),
        "realCapSpace": round(float(real_cap_space_m), 3),
        "usableCapSpace": round(float(usable_cap_space_m), 3),
        "capSpace": round(float(usable_cap_space_m), 3),
        "floorSpace": round(float(floor_space_m), 3),
        "projectedDeadlineSpace": round(float(projected_deadline_space_m), 3),
        "isUsingLTIR": is_using_ltir,
        "activeRosterCount": active_roster_count,
        "activeRosterMax": 23,
        "retainedSlotsUsed": retained_slots_used,
        "retainedSlotsMax": retained_slots_max,
        "warnings": warnings,
    }


def _round_to_half_million(value_m: float) -> float:
    return round(value_m * 2.0) / 2.0


_CAP_MOVEMENT_RULES = (
    ("normal_increase", 0.72, 1.0, 3.5, "Standard growth", "League revenue nudged the cap higher."),
    ("strong_increase", 0.18, 3.5, 6.5, "Revenue jump", "League revenue pushed the cap higher."),
    ("dramatic_increase", 0.06, 6.5, 10.5, "Media revenue boost", "A major revenue spike lifted the cap."),
    ("flat_cap", 0.03, 0.0, 0.0, "Flat cap year", "League held the cap steady."),
    ("decrease", 0.01, 1.0, 3.0, "League revenue dip", "League revenue pressure lowered the cap."),
)


def _pick_cap_movement(rng: Any) -> tuple:
    roll = float(rng.random()) if rng is not None and hasattr(rng, "random") else 0.5
    cursor = 0.0
    for rule in _CAP_MOVEMENT_RULES:
        movement_type, prob, low, high, label, reason = rule
        cursor += float(prob)
        if roll < cursor:
            if movement_type == "flat_cap":
                return movement_type, 0.0, label, reason
            if movement_type == "decrease":
                delta = float(rng.uniform(low, high)) if rng is not None and hasattr(rng, "uniform") else low
                return movement_type, -abs(delta), label, reason
            delta = float(rng.uniform(low, high)) if rng is not None and hasattr(rng, "uniform") else (low + high) / 2.0
            if movement_type == "dramatic_increase" and rng is not None and hasattr(rng, "random"):
                if float(rng.random()) < 0.35:
                    return movement_type, delta, "CBA adjustment", "A new CBA clause moved the cap."
            return movement_type, delta, label, reason
    movement_type, _, low, high, label, reason = _CAP_MOVEMENT_RULES[0]
    delta = float(rng.uniform(low, high)) if rng is not None and hasattr(rng, "uniform") else 2.0
    return movement_type, delta, label, reason


def advance_league_salary_cap(league: Any, rng: Any, season_year: Optional[int] = None) -> Dict[str, Any]:
    previous_cap = normalize_money_to_millions(
        _get(league, "salary_cap_m", _get(league, "salary_cap", _get(_get(league, "economics", None), "salary_cap", 92.0)))
    )
    if previous_cap <= 0:
        previous_cap = 92.0

    movement_type, delta_m, movement_label, movement_reason = _pick_cap_movement(rng)
    if movement_type == "flat_cap":
        next_cap = previous_cap
        change = 0.0
    else:
        next_cap = max(30.0, _round_to_half_million(previous_cap + float(delta_m)))
        change = round(next_cap - previous_cap, 3)

    floor = _round_to_half_million(next_cap * 0.74)
    cap_change_percent = round((change / previous_cap) * 100.0, 1) if previous_cap > 0 else 0.0
    direction = "flat" if abs(change) < 1e-6 else ("up" if change > 0 else "down")

    label = _season_label_from_year(season_year)
    row = {
        "season": label,
        "previous_cap": round(float(previous_cap), 3),
        "upperLimit": round(float(next_cap), 3),
        "current_cap": round(float(next_cap), 3),
        "lowerLimit": round(float(floor), 3),
        "direction": direction,
        "change": round(float(change), 3),
        "cap_change": round(float(change), 3),
        "cap_change_percent": cap_change_percent,
        "movement_type": movement_type,
        "movement_label": movement_label,
        "movement_reason": movement_reason,
    }

    setattr(league, "salary_cap_m", float(next_cap))
    setattr(league, "cap_floor_m", float(floor))
    setattr(league, "cap_growth_rate", float(change / previous_cap if previous_cap > 0 else 0.0))

    history = _get(league, "cap_history", None)
    if not isinstance(history, list):
        history = []
    history.append(row)
    setattr(league, "cap_history", history)

    econ = _get(league, "economics", None)
    if econ is not None:
        try:
            setattr(econ, "salary_cap", float(next_cap))
            setattr(econ, "cap_floor", float(floor))
            setattr(econ, "cap_growth_rate", float(getattr(league, "cap_growth_rate")))
        except Exception:
            pass

    return row


def _player_on_active_roster(team: Any, player: Any) -> bool:
    if player is None:
        return False
    pid = str(_get(player, "id", "") or _get(player, "player_id", "") or "")
    if not pid:
        return False
    for p in _iter_team_roster(team):
        if not _is_active_roster_player(p):
            continue
        if str(_get(p, "id", "") or _get(p, "player_id", "") or "") == pid:
            return True
    return False


def can_sign_player(
    team: Any,
    contract_aav_m: float,
    league: Any = None,
    *,
    player: Any = None,
) -> Dict[str, Any]:
    snap = calculate_team_cap_snapshot(team, league=league)
    needed = max(0.0, float(contract_aav_m))
    roster_add = 0 if _player_on_active_roster(team, player) else 1
    projected_roster_count = snap["activeRosterCount"] + roster_add

    if projected_roster_count > snap["activeRosterMax"]:
        return {
            "ok": False,
            "reason": "Active roster would exceed maximum",
            "snapshot": snap,
        }

    if snap["usableCapSpace"] < needed:
        return {
            "ok": False,
            "reason": "Insufficient usable cap space",
            "snapshot": snap,
        }

    return {"ok": True, "reason": "ok", "snapshot": snap}


def can_recall_player(team: Any, player: Any, league: Any = None) -> Dict[str, Any]:
    snap = calculate_team_cap_snapshot(team, league=league)
    added = player_cap_hit_millions(player) - buried_cap_hit_millions(player)
    ok = (snap["usableCapSpace"] - max(0.0, added)) >= 0 and snap["activeRosterCount"] < snap["activeRosterMax"]
    reason = "ok" if ok else "Recall would break cap or roster limit"
    return {"ok": bool(ok), "reason": str(reason), "snapshot": snap}


CONTRACT_SLOTS_LIMIT = 50


def _player_has_active_contract(player: Any) -> bool:
    yrs = 0
    for obj in (player, getattr(player, "contract", None)):
        if obj is None:
            continue
        for key in ("years_remaining", "term_remaining", "remaining_years", "term"):
            val = _get(obj, key, None)
            if val is None:
                continue
            try:
                yrs = max(yrs, int(val))
            except (TypeError, ValueError):
                continue
    if yrs <= 0:
        return False
    return player_cap_hit_millions(player) > 0 or yrs > 0


def count_team_contract_slots(team: Any) -> int:
    """Count distinct players under contract on NHL roster + prospect_pool."""
    seen: set = set()
    count = 0
    for p in list(getattr(team, "roster", None) or []):
        if bool(_get(p, "retired", False)):
            continue
        pid = str(_get(p, "id", "") or "")
        if not pid or pid in seen:
            continue
        if _player_has_active_contract(p):
            count += 1
            seen.add(pid)
    for p in list(getattr(team, "prospect_pool", None) or []):
        pid = str(_get(p, "id", "") or "")
        if not pid or pid in seen:
            continue
        if _player_has_active_contract(p):
            count += 1
            seen.add(pid)
    return count


def projected_contract_slots_after_trade(
    team: Any,
    outgoing_players: Iterable[Any],
    incoming_players: Iterable[Any],
) -> int:
    used = count_team_contract_slots(team)
    outgoing = list(outgoing_players or [])
    incoming = list(incoming_players or [])
    out_ids = {str(_get(p, "id", "") or "") for p in outgoing}
    delta = 0
    for p in outgoing:
        if _player_has_active_contract(p):
            delta -= 1
    for p in incoming:
        pid = str(_get(p, "id", "") or "")
        if pid in out_ids:
            continue
        if _player_has_active_contract(p):
            delta += 1
    return used + delta


def can_trade_contract_slots_fit(
    team: Any,
    outgoing_players: Iterable[Any],
    incoming_players: Iterable[Any],
) -> Dict[str, Any]:
    used = count_team_contract_slots(team)
    projected = projected_contract_slots_after_trade(team, outgoing_players, incoming_players)
    ok = projected <= CONTRACT_SLOTS_LIMIT
    return {
        "ok": ok,
        "reason": "ok" if ok else f"Trade would exceed contract slot limit ({projected}/{CONTRACT_SLOTS_LIMIT})",
        "contract_slots_used": used,
        "projected_contract_slots": projected,
        "contract_slots_limit": CONTRACT_SLOTS_LIMIT,
    }


def can_trade_cap_fit(
    team: Any,
    outgoing_players: Iterable[Any],
    incoming_players: Iterable[Any],
    retained_added_m: float = 0.0,
    league: Any = None,
    *,
    incoming_retained_pct: Optional[Dict[str, float]] = None,
    calendar_cursor: int = 0,
    regular_season_last_index: int = 192,
    deadline_phase: float = 0.0,
    season_label: Optional[str] = None,
) -> Dict[str, Any]:
    snap = calculate_team_cap_snapshot(
        team,
        league=league,
        season_label=season_label,
        calendar_cursor=int(calendar_cursor or 0),
        regular_season_last_index=int(regular_season_last_index or 192),
    )
    outgoing = list(outgoing_players or [])
    incoming = list(incoming_players or [])
    retained_map = dict(incoming_retained_pct or {})

    total_days = max(1, int(regular_season_last_index or 192))
    day_idx = max(0, int(calendar_cursor or 0))
    days_remaining = max(1, total_days - day_idx)
    remaining_pct = max(0.05, float(days_remaining) / float(total_days))
    # Mid-season acquisitions only apply remaining-season cap burden; stricter after deadline.
    prorate_factor = remaining_pct if day_idx > 0 else 1.0
    if float(deadline_phase or 0.0) > 0.72:
        prorate_factor = min(1.0, prorate_factor + (1.0 - prorate_factor) * 0.65)

    out_full_m = sum(player_cap_hit_millions(p) for p in outgoing)
    in_full_m = 0.0
    for p in incoming:
        pid = str(_get(p, "id", "") or "")
        cap_hit = player_cap_hit_millions(p)
        retained_pct = max(0.0, min(50.0, float(retained_map.get(pid, 0.0))))
        in_full_m += cap_hit * (1.0 - retained_pct / 100.0)

    out_m = out_full_m * prorate_factor
    in_m = in_full_m * prorate_factor
    retained_added = max(0.0, float(retained_added_m))
    delta = in_m - out_m + retained_added
    full_delta = in_full_m - out_full_m + retained_added
    projected_roster_count = snap["activeRosterCount"] - len(outgoing) + len(incoming)

    upper_limit = float(snap.get("upperLimit", 0.0))
    effective_limit = float(snap.get("effectiveCapLimit", upper_limit))
    projected_total_cap = float(snap.get("totalCapHit", 0.0)) + full_delta
    projected_cap_space = float(snap["usableCapSpace"]) - delta
    projected_deadline_space = float(snap.get("projectedDeadlineSpace", snap["usableCapSpace"])) - delta

    if projected_roster_count > snap["activeRosterMax"]:
        return {
            "ok": False,
            "reason": "Trade would exceed active roster maximum",
            "snapshot": snap,
            "projectedCapSpace": round(projected_cap_space, 3),
            "projectedDeadlineSpace": round(projected_deadline_space, 3),
            "projectedRosterCount": projected_roster_count,
            "capDelta": round(delta, 3),
            "capDeltaFull": round(full_delta, 3),
            "prorationFactor": round(prorate_factor, 4),
            "ltirReliefUsed": False,
        }

    ok = projected_cap_space >= -0.001
    ltir_relief = False
    reason = "ok"
    if not ok:
        if projected_total_cap <= effective_limit + 0.02 and float(snap.get("ltirPool", 0.0)) > 0.001:
            ok = True
            ltir_relief = True
            reason = "ok_with_ltir"
        elif projected_deadline_space >= -0.001 and day_idx > 0 and float(deadline_phase or 0.0) < 0.72:
            ok = True
            reason = "ok_with_accrual"
        else:
            reason = "Trade would exceed usable cap space"

    if not ok:
        return {
            "ok": False,
            "reason": reason,
            "snapshot": snap,
            "projectedCapSpace": round(projected_cap_space, 3),
            "projectedDeadlineSpace": round(projected_deadline_space, 3),
            "projectedRosterCount": projected_roster_count,
            "capDelta": round(delta, 3),
            "capDeltaFull": round(full_delta, 3),
            "prorationFactor": round(prorate_factor, 4),
            "ltirReliefUsed": False,
            "projectedTotalCapHit": round(projected_total_cap, 3),
        }

    return {
        "ok": True,
        "reason": reason,
        "snapshot": snap,
        "projectedCapSpace": round(projected_cap_space, 3),
        "projectedDeadlineSpace": round(projected_deadline_space, 3),
        "projectedRosterCount": projected_roster_count,
        "capDelta": round(delta, 3),
        "capDeltaFull": round(full_delta, 3),
        "prorationFactor": round(prorate_factor, 4),
        "ltirReliefUsed": ltir_relief,
        "projectedTotalCapHit": round(projected_total_cap, 3),
    }


def can_activate_ltir_player(team: Any, player: Any, league: Any = None) -> Dict[str, Any]:
    snap = calculate_team_cap_snapshot(team, league=league)
    player_cap = player_cap_hit_millions(player)

    projected_ltir_pool = max(0.0, snap["ltirPool"] - player_cap)
    projected_effective_limit = snap["upperLimit"] + projected_ltir_pool

    projected_total = snap["totalCapHit"]
    if bool(_get(player, "excluded_from_cap_while_ltir", False)):
        projected_total += player_cap

    projected_usable_space = projected_effective_limit - projected_total
    ok = projected_usable_space >= 0

    return {
        "ok": bool(ok),
        "reason": "ok" if ok else "Activation would exceed usable cap limit",
        "snapshot": snap,
        "projectedLtirPool": round(projected_ltir_pool, 3),
        "projectedEffectiveLimit": round(projected_effective_limit, 3),
        "projectedTotalCapHit": round(projected_total, 3),
        "projectedUsableCapSpace": round(projected_usable_space, 3),
    }


def validate_team_cap_compliance(team: Any, league: Any = None) -> Dict[str, Any]:
    snap = calculate_team_cap_snapshot(team, league=league)
    if snap["usableCapSpace"] < 0:
        return {"ok": False, "reason": "Team is over the usable cap limit", "snapshot": snap}
    if snap["floorSpace"] < 0:
        return {"ok": False, "reason": "Team is below the cap floor", "snapshot": snap}
    if snap["activeRosterCount"] > snap["activeRosterMax"]:
        return {"ok": False, "reason": "Active roster exceeds maximum", "snapshot": snap}
    if snap["retainedSlotsUsed"] > snap["retainedSlotsMax"]:
        return {"ok": False, "reason": "Retained salary slots exceed maximum", "snapshot": snap}
    return {"ok": True, "reason": "ok", "snapshot": snap}
