"""
Unsigned prospect development while players remain outside the NHL organization.

Drafted-but-unsigned players keep developing on junior / NCAA / European clubs.
Uses the same core attribute development engine as signed paths, with contextual mods.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional


def _rng(parts: Any) -> float:
    raw = ":".join(str(p) for p in parts) if isinstance(parts, (list, tuple)) else str(parts)
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _league_quality(league_id: str) -> float:
    code = str(league_id or "").upper()
    if code.startswith("NCAA") or "NCAA" in code:
        return 0.78
    if code in ("OHL", "WHL", "QMJHL") or code.startswith("CHL"):
        return 0.72
    if code == "USHL":
        return 0.62
    if code.startswith("EU_") or any(x in code for x in ("SHL", "LIIGA", "DEL", "KHL")):
        return 0.80
    return 0.65


def develop_unsigned_prospect(
    player: Any,
    *,
    season_year: int,
    nhl_team: Any = None,
    request_league_transfer: bool = False,
) -> Dict[str, Any]:
    if str(getattr(player, "signed_status", "unsigned") or "unsigned").lower() == "signed":
        return {"ok": False, "reason": "already_signed"}

    # Same core engine as org development — contextual modifiers only.
    from app.sim_engine.entities.player import (
        display_rating,
        normalize_rating,
        normalize_rating_gap,
        persist_recomputed_ovr,
        player_current_ovr_01,
    )
    from app.sim_engine.progression.development import (
        apply_attribute_deltas,
        allocate_growth_to_attributes,
        calculate_season_growth_budget,
        resolve_development_profile,
    )
    from app.sim_engine.progression.potential import ensure_development_ledger

    sid = int(season_year)
    try:
        setattr(player, "_active_dev_season", sid)
        setattr(player, "_dev_source_path", "unsigned_prospect")
    except Exception:
        pass

    ledger = ensure_development_ledger(player, sid)
    if ledger.get("development_applied"):
        return {
            "ok": True,
            "skipped": True,
            "reason": "already_developed_this_season",
            "player_id": str(getattr(player, "id", "") or ""),
            "delta": 0.0,
            "overall": float(display_rating(player_current_ovr_01(player))),
            "nhl_readiness": float(getattr(player, "nhl_readiness", 0) or 0),
        }

    age = int(getattr(player, "age", None) or getattr(getattr(player, "identity", None), "age", 18) or 18)
    path = str(getattr(player, "development_path", "") or getattr(player, "current_league_id", "") or "")
    league_id = str(getattr(player, "current_league_id", "") or path)
    lq = _league_quality(league_id)

    ice = float(getattr(player, "ice_time_quality", None) or getattr(player, "line_role_score", None) or 0.55)
    coach = float(getattr(player, "coaching_quality", None) or 0.55)
    production = float(getattr(player, "ppg", None) or getattr(player, "production_score", None) or 0.5)
    injured = bool(getattr(player, "injured", False) or getattr(player, "injury_flag", False))
    tourney = float(getattr(player, "tournament_boost", None) or 0.0)

    plan = 0.55
    if nhl_team is not None:
        plan = float(
            getattr(nhl_team, "prospect_pipeline_score", None)
            or getattr(nhl_team, "development_plan_score", None)
            or 0.55
        )

    # Org / coaching modify efficiency with diminishing returns — not raw OVR bonuses.
    org_mod = 0.86 + 0.22 * min(1.0, plan) - 0.04 * max(0.0, plan - 0.85)
    coach_mod = 0.90 + 0.16 * min(1.0, coach)
    # Production is relative context for budget, not a direct attribute factory.
    prod_mod = 0.92 + 0.12 * max(0.0, min(1.0, production))
    if tourney:
        # Tournament cannot outweigh season context.
        prod_mod = min(1.06, prod_mod + min(0.04, tourney * 0.05))

    try:
        setattr(player, "_league_quality_mod", 0.88 + 0.2 * lq)
        setattr(player, "_dev_env_growth_mult", org_mod * coach_mod)
    except Exception:
        pass

    profile = resolve_development_profile(
        player,
        {
            "league_quality": lq,
            "ppg": production,
            "production_adjusted_score": production,
        },
    )
    ovr_before = float(profile["current_ovr"])
    ledger["ovr_before"] = round(ovr_before, 4)

    ctx = {
        "participation_mod": max(0.4, min(1.05, ice)),
        "league_quality_mod": 0.88 + 0.2 * lq,
        "org_dev_mod": max(0.78, min(1.1, org_mod * coach_mod)),
        "morale_mod": 1.0,
        "injured": injured,
        "overmatched": lq >= 0.78 and ovr_before < 0.55,
        "underchallenged": lq <= 0.65 and ovr_before > 0.7,
        "bench_or_scratch": ice < 0.35,
    }
    if request_league_transfer:
        ctx["league_quality_mod"] = min(1.08, float(ctx["league_quality_mod"]) * 1.05)

    budget = calculate_season_growth_budget(player, ctx) * prod_mod
    # Deterministic jitter from franchise-stable hash (no unseeded random).
    budget *= 0.92 + 0.16 * _rng((getattr(player, "id", ""), season_year, "g"))

    ratings = getattr(player, "ratings", None)
    delta_display = 0.0
    if isinstance(ratings, dict) and ratings:
        # Seeded RNG for attribute allocation phase labels only.
        seed = int(_rng((getattr(player, "id", ""), season_year, "alloc")) * 1e9)
        rng = random.Random(seed)
        phase = "NORMAL"
        if _rng((getattr(player, "id", ""), season_year, "phase")) < 0.08:
            phase = "STALL"
            budget *= 0.2
        deltas = allocate_growth_to_attributes(player, budget, phase=phase)
        applied = apply_attribute_deltas(player, deltas)
        ledger["attribute_deltas"] = {
            k: round(float(v), 4) for k, v in list(applied.items())[:24]
        }
        ovr_after = player_current_ovr_01(player)
        delta_display = (ovr_after - ovr_before) * 99.0
    else:
        # Legacy fixture without attributes: nudge toward expected via readiness/overall
        # mirrors only — still ledger-gated and ceiling-bounded (no gap*0.35 dump).
        gap = normalize_rating_gap(ovr_before, profile["expected_ceiling"])
        step = min(budget, max(0.004, gap * 0.10))
        new_ovr01 = min(float(profile["expected_ceiling"]) - 0.005, ovr_before + step)
        try:
            setattr(player, "overall", float(display_rating(new_ovr01)))
        except Exception:
            pass
        try:
            if not callable(getattr(type(player), "ovr", None)):
                setattr(player, "ovr", float(new_ovr01))
        except Exception:
            pass
        ovr_after = normalize_rating(getattr(player, "overall", new_ovr01))
        delta_display = (ovr_after - ovr_before) * 99.0

    readiness = float(getattr(player, "nhl_readiness", None) or display_rating(ovr_before))
    if readiness > 1.5:
        ready01 = normalize_rating(readiness)
    else:
        ready01 = normalize_rating(readiness)
    new_ready01 = min(0.92, ready01 + max(0.0, (ovr_after - ovr_before)) * 0.7 + (0.004 if lq > 0.75 else 0.0))
    setattr(player, "nhl_readiness", float(display_rating(new_ready01)))
    setattr(player, "last_development_year", int(season_year))
    setattr(player, "last_development_delta", round(delta_display, 3))
    if request_league_transfer:
        setattr(player, "requested_league_transfer", True)

    if bool(getattr(player, "elc_slide_eligible", False)):
        setattr(player, "elc_slide_years_remaining", int(getattr(player, "elc_slide_years_remaining", 1) or 1))

    ledger["development_applied"] = True
    ledger["source_path"] = "unsigned_prospect"
    ledger["ovr_after"] = round(float(ovr_after), 4)

    try:
        persist_recomputed_ovr(player)
    except Exception:
        pass

    attr_out = {}
    if isinstance(ledger.get("attribute_deltas"), dict):
        for k, v in ledger["attribute_deltas"].items():
            try:
                fv = round(float(v), 2)
            except Exception:
                continue
            if abs(fv) < 0.01:
                continue
            attr_out[str(k)] = fv

    return {
        "ok": True,
        "player_id": str(getattr(player, "id", "") or ""),
        "delta": round(delta_display, 3),
        "overall": float(display_rating(ovr_after)),
        "previous_overall": float(display_rating(ovr_before)),
        "nhl_readiness": float(display_rating(new_ready01)),
        "league_id": league_id,
        "path": path,
        "attribute_deltas": attr_out,
        "development_trend": str(getattr(player, "development_trend", "") or ""),
    }


def run_unsigned_prospect_development_pass(session: Any, *, season_year: Optional[int] = None) -> Dict[str, Any]:
    sy = int(season_year or getattr(session, "season_calendar_year", 2025) or 2025)
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return {"developed": 0, "results": []}

    results: List[Dict[str, Any]] = []
    for team in getattr(league, "teams", None) or []:
        for p in list(getattr(team, "prospect_pool", None) or []):
            if str(getattr(p, "signed_status", "unsigned") or "unsigned").lower() == "signed":
                continue
            if not bool(getattr(p, "drafted", False)) and not getattr(p, "nhl_rights_team_id", None):
                continue
            if str(getattr(p, "status", "") or "").lower() in ("nhl", "active"):
                continue
            transfer = bool(getattr(p, "requested_league_transfer", False))
            try:
                setattr(p, "_active_dev_season", sy)
            except Exception:
                pass
            res = develop_unsigned_prospect(p, season_year=sy, nhl_team=team, request_league_transfer=transfer)
            if res.get("ok"):
                results.append({**res, "rights_team_id": str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")})

    payload = {
        "season_year": sy,
        "developed": len([r for r in results if not r.get("skipped")]),
        "results": results[:80],
        "avg_delta": round(sum(r.get("delta", 0) for r in results) / len(results), 3) if results else 0.0,
    }
    try:
        session.unsigned_development_payload = payload
    except Exception:
        pass
    return payload
