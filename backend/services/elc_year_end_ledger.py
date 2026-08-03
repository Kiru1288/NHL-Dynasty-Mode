"""
Year-end ELC Schedule A/B bonus evaluation + development-promise enforcement.

Consumes franchise session.player_season_stats (gp/g/a/pts) — never invents stats.
Currency: millions (_m).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _pid(player: Any) -> str:
    return str(getattr(player, "id", None) or getattr(player, "player_id", "") or "")


def _season_row(session: Any, player: Any) -> Dict[str, Any]:
    stats = getattr(session, "player_season_stats", None) or {}
    row = stats.get(_pid(player)) if isinstance(stats, dict) else None
    if isinstance(row, dict):
        return row
    # Fallbacks from player attrs stamped during development
    return {
        "gp": int(getattr(player, "games_played", 0) or getattr(player, "nhl_games_played_this_season", 0) or 0),
        "g": int(getattr(player, "goals", 0) or 0),
        "a": int(getattr(player, "assists", 0) or 0),
        "pts": int(getattr(player, "points", 0) or 0),
    }


def _nudge_morale(player: Any, delta_01: float) -> None:
    """delta_01 is on 0–1 psych scale (e.g. +0.04)."""
    try:
        psych = getattr(player, "psych", None)
        if psych is not None and hasattr(psych, "morale"):
            cur = float(getattr(psych, "morale", 0.5) or 0.5)
            setattr(psych, "morale", max(0.0, min(1.0, cur + float(delta_01))))
            return
    except Exception:
        pass
    try:
        cur = float(getattr(player, "morale", 50) or 50)
        if cur <= 1.5:
            setattr(player, "morale", max(0.0, min(1.0, cur + float(delta_01))))
        else:
            setattr(player, "morale", max(0.0, min(100.0, cur + float(delta_01) * 100.0)))
    except Exception:
        pass
    try:
        rel = float(getattr(player, "org_relationship", 0.55) or 0.55)
        setattr(player, "org_relationship", max(0.0, min(1.0, rel + float(delta_01) * 0.5)))
    except Exception:
        pass


def _condition_met(cond: Dict[str, Any], stats: Dict[str, Any], awards: Optional[Dict[str, Any]] = None) -> bool:
    cid = str(cond.get("id") or "")
    thr = float(cond.get("threshold") or 0)
    awards = awards or {}
    if cid == "games_played":
        return float(stats.get("gp") or 0) >= thr
    if cid == "goals":
        return float(stats.get("g") or 0) >= thr
    if cid == "assists":
        return float(stats.get("a") or 0) >= thr
    if cid == "points":
        pts = stats.get("pts")
        if pts is None:
            pts = float(stats.get("g") or 0) + float(stats.get("a") or 0)
        return float(pts or 0) >= thr
    if cid == "all_rookie":
        return bool(awards.get("all_rookie") or awards.get("all_rookie_team"))
    if cid == "calder":
        return bool(awards.get("calder") or awards.get("calder_trophy"))
    if cid == "hart_finish":
        return bool(awards.get("hart_finish") or awards.get("hart"))
    return False


def evaluate_contract_performance_bonuses(
    session: Any,
    player: Any,
    *,
    season_year: int,
    awards: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate Schedule A/B conditions on an ELC (or any contract with bonus_conditions).
    Records earned amounts on the contract; does not invent conditions.
    """
    c = getattr(player, "contract", None)
    if not isinstance(c, dict):
        return {"ok": False, "reason": "no_contract"}
    conditions = c.get("bonus_conditions") or {}
    if not isinstance(conditions, dict):
        return {"ok": True, "earned_m": 0.0, "items": [], "skipped": True}

    stats = _season_row(session, player)
    earned_items: List[Dict[str, Any]] = []
    total = 0.0
    for schedule_key in ("schedule_a", "schedule_b"):
        for cond in list(conditions.get(schedule_key) or []):
            if not isinstance(cond, dict):
                continue
            met = _condition_met(cond, stats, awards)
            amount = float(cond.get("amount_m") or 0)
            if met and amount > 0:
                total += amount
                earned_items.append({
                    "schedule": schedule_key,
                    "id": cond.get("id"),
                    "label": cond.get("label"),
                    "amount_m": round(amount, 4),
                    "stats": {k: stats.get(k) for k in ("gp", "g", "a", "pts")},
                })

    prior = float(c.get("earned_bonuses_m") or 0)
    # Track this season separately then accumulate
    season_key = f"earned_bonuses_{season_year}_m"
    c[season_key] = round(total, 4)
    c["earned_bonuses_m"] = round(prior + total, 4)
    c["last_bonus_evaluation"] = {
        "season_year": season_year,
        "earned_m": round(total, 4),
        "items": earned_items,
        "stats": {k: stats.get(k) for k in ("gp", "g", "a", "pts")},
    }
    # Cap accounting: performance bonuses feed bonus reserve / overage via existing
    # contract fields — bump performance_bonus exposure already on contract.
    max_bonus = float(c.get("maximum_performance_bonus_m") or c.get("performance_bonus_m") or 0)
    if total > 0 and max_bonus > 0:
        c["performance_bonus_paid_m"] = round(float(c.get("performance_bonus_paid_m") or 0) + total, 4)

    return {
        "ok": True,
        "player_id": _pid(player),
        "earned_m": round(total, 4),
        "items": earned_items,
        "stats": stats,
    }


def _resolve_promise(player: Any) -> Optional[str]:
    promise = getattr(player, "development_promise", None)
    if promise:
        return str(promise)
    c = getattr(player, "contract", None)
    if isinstance(c, dict) and c.get("development_promise"):
        return str(c.get("development_promise"))
    return None


def _promise_progress(
    promise: str,
    *,
    gp: int,
    status: str,
    path: str,
    pace_factor: float = 1.0,
) -> Tuple[bool, str, str]:
    """
    Returns (on_track_or_honoured, reason, status_label).
    pace_factor scales GP thresholds (0.5 at midseason ≈ half-season pace).
    """
    pace = max(0.25, float(pace_factor))
    ahl_like = "AHL" in path or "signed_ahl" in status or "ahl" in status
    nhl_like = "NHL" in path or "signed_nhl" in status or status.startswith("nhl")

    if promise in ("ahl_featured", "protected_role"):
        thr_ahl = int(40 * pace)
        thr_nhl = int(20 * pace)
        ok = (ahl_like and gp >= thr_ahl) or gp >= thr_nhl
        return ok, ("featured_minutes" if ok else "insufficient_role_minutes"), ("on_track" if ok else "at_risk")
    if promise == "top_six_track":
        thr_full = int(45 * pace)
        thr_nhl = int(25 * pace)
        ok = gp >= thr_full or (gp >= thr_nhl and nhl_like)
        return ok, ("top_six_track_ok" if ok else "top_six_track_missed"), ("on_track" if ok else "at_risk")
    if promise == "depth_only":
        return True, "depth_promise_met", "on_track"
    thr = int(30 * pace)
    ok = gp >= thr
    return ok, ("generic_promise" if ok else "generic_promise_missed"), ("on_track" if ok else "at_risk")


def evaluate_development_promise(
    session: Any,
    player: Any,
    *,
    season_year: int,
) -> Dict[str, Any]:
    """
    Year-end: check whether the organization honoured an ELC development promise.
    Affects morale / org_relationship. Uses season GP + assignment proxies.
    """
    promise = _resolve_promise(player)
    if not promise:
        return {"ok": True, "skipped": True}

    promise_season = getattr(player, "development_promise_season", None)
    if promise_season is not None and int(promise_season) > int(season_year):
        return {"ok": True, "skipped": True, "reason": "promise_not_due"}

    stats = _season_row(session, player)
    gp = int(stats.get("gp") or 0)
    status = str(getattr(player, "organizational_status", "") or "").lower()
    path = str(getattr(player, "development_path", "") or "").upper()
    honoured, reason, _label = _promise_progress(promise, gp=gp, status=status, path=path, pace_factor=1.0)

    if honoured:
        _nudge_morale(player, 0.035)
        try:
            setattr(player, "development_promise_honoured", True)
            setattr(player, "development_promise_result", reason)
        except Exception:
            pass
    else:
        _nudge_morale(player, -0.055)
        try:
            setattr(player, "development_promise_honoured", False)
            setattr(player, "development_promise_result", reason)
            friction = float(getattr(player, "negotiation_friction", 0) or 0)
            setattr(player, "negotiation_friction", min(1.0, friction + 0.12))
        except Exception:
            pass

    return {
        "ok": True,
        "player_id": _pid(player),
        "promise": promise,
        "honoured": honoured,
        "reason": reason,
        "gp": gp,
        "morale_delta": 0.035 if honoured else -0.055,
        "phase": "year_end",
    }


def evaluate_development_promise_midseason(
    session: Any,
    player: Any,
    *,
    season_year: int,
) -> Dict[str, Any]:
    """
    Mid-season enforcement: soft morale nudge from pace toward the promise.
    Does not finalize honour/fail — only marks on_track / at_risk.
    """
    promise = _resolve_promise(player)
    if not promise:
        return {"ok": True, "skipped": True}

    promise_season = getattr(player, "development_promise_season", None)
    if promise_season is not None and int(promise_season) > int(season_year):
        return {"ok": True, "skipped": True, "reason": "promise_not_due"}

    # Skip if already finalized this season
    if getattr(player, "development_promise_honoured", None) is not None and getattr(
        player, "development_promise_result", None
    ):
        prior = str(getattr(player, "development_promise_result", "") or "")
        if prior and not prior.startswith("midseason_"):
            return {"ok": True, "skipped": True, "reason": "already_finalized"}

    stats = _season_row(session, player)
    gp = int(stats.get("gp") or 0)
    status = str(getattr(player, "organizational_status", "") or "").lower()
    path = str(getattr(player, "development_path", "") or "").upper()
    on_track, reason, label = _promise_progress(
        promise, gp=gp, status=status, path=path, pace_factor=0.5
    )

    delta = 0.018 if on_track else -0.028
    _nudge_morale(player, delta)
    try:
        setattr(
            player,
            "development_promise_midseason",
            {
                "season_year": int(season_year),
                "promise": promise,
                "on_track": on_track,
                "status": label,
                "reason": f"midseason_{reason}",
                "gp": gp,
                "morale_delta": delta,
            },
        )
        setattr(player, "development_promise_midseason_status", label)
    except Exception:
        pass

    return {
        "ok": True,
        "player_id": _pid(player),
        "promise": promise,
        "on_track": on_track,
        "status": label,
        "reason": f"midseason_{reason}",
        "gp": gp,
        "morale_delta": delta,
        "phase": "midseason",
    }


def apply_earned_bonuses_to_team_cap(
    team: Any,
    earned_m: float,
    *,
    season_year: int,
) -> Dict[str, Any]:
    """
    Season-stat payout → bonus reserve drawdown + next-season overage.
    Earned bonuses consume performance_bonus_reserve_m first; remainder becomes
    bonus_overage charged against the following season's cap.

    Always releases the season's reserve, even when nothing was earned. The
    reserve was funded at signing time as a *maximum-exposure* cap hold
    (assign_elc_contract); once the season's bonus window closes, whatever
    wasn't actually earned must be freed back to usable cap space rather than
    silently squatting on it forever — otherwise every unearned ELC bonus
    permanently taxes the club's cap in every future season.
    """
    earned = max(0.0, float(earned_m or 0))
    if team is None:
        return {"earned_m": 0.0, "from_reserve_m": 0.0, "overage_m": 0.0}

    reserve = float(
        getattr(team, "performance_bonus_reserve_m", None)
        or getattr(team, "performance_bonus_reserve", None)
        or getattr(team, "bonus_reserve_m", None)
        or 0
    )
    from_reserve = min(reserve, earned)
    overage = round(max(0.0, earned - from_reserve), 4)
    # Release the full reserve regardless of how much was earned — it was only
    # ever a hold against this season's potential bonus payout.
    new_reserve = 0.0
    try:
        team.performance_bonus_reserve_m = new_reserve
        team.performance_bonus_reserve = new_reserve
        team.bonus_reserve_m = new_reserve
    except Exception:
        pass

    if overage > 0:
        next_label = f"{int(season_year) + 1}-{(int(season_year) + 2) % 100:02d}"
        records = getattr(team, "bonus_overage", None)
        if not isinstance(records, list):
            records = list(getattr(team, "bonus_overages", None) or [])
        records.append({
            "season": next_label,
            "amount_m": overage,
            "source": "performance_bonus_payout",
            "from_season": int(season_year),
        })
        try:
            team.bonus_overage = records
            team.bonus_overages = records
        except Exception:
            pass

    return {
        "earned_m": round(earned, 4),
        "from_reserve_m": round(from_reserve, 4),
        "overage_m": overage,
        "reserve_remaining_m": new_reserve,
    }


def run_midseason_contract_ledger(session: Any, *, season_year: int) -> Dict[str, Any]:
    """Mid-season development-promise enforcement across all orgs (morale only)."""
    league = getattr(getattr(session, "sim", None), "league", None)
    promise_rows: List[Dict[str, Any]] = []
    if league is None:
        return {"promises": [], "count": 0, "phase": "midseason"}

    for team in list(getattr(league, "teams", None) or []):
        pool = list(getattr(team, "roster", None) or []) + list(getattr(team, "prospect_pool", None) or [])
        for p in pool:
            c = getattr(p, "contract", None)
            if getattr(p, "development_promise", None) or (
                isinstance(c, dict) and c.get("development_promise")
            ):
                pr = evaluate_development_promise_midseason(session, p, season_year=season_year)
                if not pr.get("skipped"):
                    promise_rows.append(pr)

    session.midseason_contract_ledger = {
        "season_year": season_year,
        "promises": promise_rows,
    }
    return {
        "promises": promise_rows,
        "promise_count": len(promise_rows),
        "count": len(promise_rows),
        "phase": "midseason",
    }


def run_year_end_contract_ledger(session: Any, *, season_year: int) -> Dict[str, Any]:
    """Evaluate bonuses + promises for all org players with ELC / promise state."""
    league = getattr(getattr(session, "sim", None), "league", None)
    bonus_rows: List[Dict[str, Any]] = []
    promise_rows: List[Dict[str, Any]] = []
    team_cap_rows: List[Dict[str, Any]] = []
    if league is None:
        return {"bonuses": [], "promises": [], "count": 0}

    awards_map = {}
    try:
        awards_payload = getattr(session, "awards_payload", None) or {}
        for row in list(awards_payload.get("winners") or awards_payload.get("awards") or []):
            if not isinstance(row, dict):
                continue
            pid = str(row.get("player_id") or row.get("winner_player_id") or "")
            name = str(row.get("name") or row.get("award") or "").lower()
            if not pid:
                continue
            awards_map.setdefault(pid, {})
            if "calder" in name:
                awards_map[pid]["calder"] = True
            if "all-rookie" in name or "all_rookie" in name or "rookie" in name:
                awards_map[pid]["all_rookie"] = True
            if "hart" in name:
                awards_map[pid]["hart_finish"] = True
    except Exception:
        awards_map = {}

    for team in list(getattr(league, "teams", None) or []):
        team_earned = 0.0
        pool = list(getattr(team, "roster", None) or []) + list(getattr(team, "prospect_pool", None) or [])
        for p in pool:
            c = getattr(p, "contract", None)
            if isinstance(c, dict) and (
                c.get("bonus_conditions")
                or str(c.get("type") or c.get("contract_type") or "").upper() == "ELC"
            ):
                res = evaluate_contract_performance_bonuses(
                    session, p, season_year=season_year, awards=awards_map.get(_pid(p))
                )
                if res.get("earned_m"):
                    bonus_rows.append(res)
                    team_earned += float(res.get("earned_m") or 0)
            if getattr(p, "development_promise", None) or (
                isinstance(c, dict) and c.get("development_promise")
            ):
                pr = evaluate_development_promise(session, p, season_year=season_year)
                if not pr.get("skipped"):
                    promise_rows.append(pr)
        team_reserve = float(
            getattr(team, "performance_bonus_reserve_m", None)
            or getattr(team, "performance_bonus_reserve", None)
            or getattr(team, "bonus_reserve_m", None)
            or 0
        )
        if team_earned > 0 or team_reserve > 0:
            cap_res = apply_earned_bonuses_to_team_cap(team, team_earned, season_year=season_year)
            cap_res["team_id"] = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
            team_cap_rows.append(cap_res)

    session.year_end_contract_ledger = {
        "season_year": season_year,
        "bonuses": bonus_rows,
        "promises": promise_rows,
        "team_cap": team_cap_rows,
    }
    return {
        "bonuses": bonus_rows,
        "promises": promise_rows,
        "team_cap": team_cap_rows,
        "bonus_count": len(bonus_rows),
        "promise_count": len(promise_rows),
        "count": len(bonus_rows) + len(promise_rows),
    }
