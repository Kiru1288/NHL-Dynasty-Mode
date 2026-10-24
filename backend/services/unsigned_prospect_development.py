"""
Unsigned prospect development while players remain outside the NHL organization.

Drafted-but-unsigned players keep developing on junior / NCAA / European clubs.
Uses the same core attribute development engine as signed paths, with contextual mods.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)


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
        assign_prospect_realization,
        calculate_season_growth_budget,
        nudge_prospect_realization,
        resolve_development_profile,
    )
    from app.sim_engine.progression.potential import ensure_development_ledger

    sid = int(season_year)
    try:
        setattr(player, "_active_dev_season", sid)
        setattr(player, "_dev_source_path", "unsigned_prospect")
        assign_prospect_realization(player)
        nudge_prospect_realization(player, sid)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

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

    int(getattr(player, "age", None) or getattr(getattr(player, "identity", None), "age", 18) or 18)
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
        _swallowed_log.debug("suppressed exception", exc_info=True)

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
    # In-season pulses already paid out most of this year's growth: pay only what is still owed.
    # If those pulses never moved the ratings, the leftover is the whole gap-scaled year.
    try:
        from app.sim_engine.progression.development import prospect_offseason_leftover, prospect_year_payment

        year_cap = float(prospect_year_payment(player))
        _owed = prospect_offseason_leftover(player)
        if _owed is not None:
            budget = _owed * (0.92 + 0.16 * _rng((getattr(player, "id", ""), season_year, "left")))
        else:
            # Pulses never paid the ratings. The year is the gap to potential,
            # not the small raw budget.
            budget = year_cap * (0.92 + 0.16 * _rng((getattr(player, "id", ""), season_year, "full")))
        pool_spent = float(getattr(player, "_pool_inseason_spent", 0.0) or 0.0)
        if pool_spent > 0.0 and not isinstance(getattr(player, "ratings", None), dict):
            budget = max(0.0, year_cap - pool_spent)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    ratings = getattr(player, "ratings", None)
    delta_display = 0.0
    if isinstance(ratings, dict) and ratings:
        # Seeded RNG for attribute allocation phase labels only.
        seed = int(_rng((getattr(player, "id", ""), season_year, "alloc")) * 1e9)
        phase = "NORMAL"
        if _rng((getattr(player, "id", ""), season_year, "phase")) < 0.08:
            phase = "STALL"
            budget *= 0.2
        deltas = allocate_growth_to_attributes(player, budget, phase=phase)
        applied = apply_attribute_deltas(player, deltas)
        alloc_rng = random.Random(seed)
        try:
            from app.sim_engine.progression.development import ensure_displayed_ovr_delta

            ensure_displayed_ovr_delta(
                player,
                ovr_before_01=ovr_before,
                target_display_delta=float(budget) * 99.0,
                rng=alloc_rng,
                phase=phase,
                tolerance=0.45,
            )
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        ledger["attribute_deltas"] = {
            k: round(float(v), 4) for k, v in list(applied.items())[:24]
        }
        ovr_after = player_current_ovr_01(player)
        delta_display = (ovr_after - ovr_before) * 99.0
    else:
        # Legacy fixture without attributes: nudge toward expected via readiness/overall
        # mirrors only — still ledger-gated and ceiling-bounded (no gap*0.35 dump).
        gap = normalize_rating_gap(ovr_before, profile["expected_ceiling"])
        step = min(max(budget, 0.0), max(0.004, gap * 0.45))
        new_ovr01 = min(float(profile["expected_ceiling"]) - 0.005, ovr_before + step)
        try:
            setattr(player, "overall", float(display_rating(new_ovr01)))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        try:
            if not callable(getattr(type(player), "ovr", None)):
                setattr(player, "ovr", float(new_ovr01))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        ovr_after = normalize_rating(getattr(player, "overall", new_ovr01))
        delta_display = (ovr_after - ovr_before) * 99.0
        band = getattr(player, "draft_value_range", None)
        if isinstance(band, (list, tuple)) and len(band) >= 2:
            try:
                lo, hi = float(band[0]), float(band[1])
                new_lo = min(lo + step, max(lo, new_ovr01))
                player.draft_value_range = (new_lo, max(hi, new_lo + 0.02))
            except (TypeError, ValueError):
                pass

    readiness = float(getattr(player, "nhl_readiness", None) or display_rating(ovr_before))
    if readiness > 1.5:
        ready01 = normalize_rating(readiness)
    else:
        ready01 = normalize_rating(readiness)
    new_ready01 = min(0.92, ready01 + max(0.0, (ovr_after - ovr_before)) * 0.7 + (0.004 if lq > 0.75 else 0.0))
    # Canonical 0–100 readiness + ETA recompute (not a soft OVR-scaled display).
    try:
        from app.sim_engine.progression.development import (
            calculate_nhl_readiness_score,
            update_player_nhl_eta,
        )

        calculate_nhl_readiness_score(player)
        update_player_nhl_eta(player)
        readiness_out = float(getattr(player, "nhl_readiness", display_rating(new_ready01)) or 0)
    except Exception:
        setattr(player, "nhl_readiness", float(display_rating(new_ready01)))
        readiness_out = float(display_rating(new_ready01))
    setattr(player, "last_development_year", int(season_year))
    setattr(player, "last_development_delta", round(delta_display, 3))
    if request_league_transfer:
        setattr(player, "requested_league_transfer", True)

    if bool(getattr(player, "elc_slide_eligible", False)):
        setattr(player, "elc_slide_years_remaining", int(getattr(player, "elc_slide_years_remaining", 1) or 1))

    ledger["development_applied"] = True
    ledger["source_path"] = "unsigned_prospect"
    ledger["ovr_after"] = round(float(ovr_after), 4)

    # Potential moves on its own track from the season's real production (not tied to OVR growth).
    potential_review: Dict[str, Any] = {}
    try:
        from app.sim_engine.progression.development import apply_prospect_potential_review

        potential_review = apply_prospect_potential_review(
            player, random.Random(int(_rng((getattr(player, "id", ""), season_year, "pot")) * 1e9)), sid
        )
    except Exception:
        potential_review = {}

    try:
        persist_recomputed_ovr(player)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

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
        "nhl_readiness": float(readiness_out),
        "nhl_eta": int(getattr(player, "nhl_eta", 4) or 4),
        "league_id": league_id,
        "path": path,
        "attribute_deltas": attr_out,
        "development_trend": str(getattr(player, "development_trend", "") or ""),
        "potential_delta": float(potential_review.get("display_delta", 0.0) or 0.0)
        if potential_review.get("applied")
        else 0.0,
    }


def _unsigned_rights_id(player: Any) -> str:
    return str(
        getattr(player, "nhl_rights_team_id", None)
        or getattr(player, "rights_team_id", None)
        or ""
    )


def _consider_unsigned(player: Any) -> bool:
    if player is None or getattr(player, "retired", False):
        return False
    if str(getattr(player, "signed_status", "unsigned") or "unsigned").lower() == "signed":
        return False
    rights = _unsigned_rights_id(player)
    if not bool(getattr(player, "drafted", False)) and not rights:
        return False
    if str(getattr(player, "status", "") or "").lower() in ("nhl", "active"):
        return False
    return True


def run_unsigned_prospect_development_pass(session: Any, *, season_year: Optional[int] = None) -> Dict[str, Any]:
    sy = int(season_year or getattr(session, "season_calendar_year", 2025) or 2025)
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return {"developed": 0, "results": []}

    teams_by_id: Dict[str, Any] = {}
    for team in getattr(league, "teams", None) or []:
        for key in (getattr(team, "team_id", None), getattr(team, "id", None)):
            if key:
                teams_by_id[str(key)] = team

    results: List[Dict[str, Any]] = []
    seen: set = set()

    def _run_one(player: Any, nhl_team: Any) -> None:
        pid = str(getattr(player, "id", "") or getattr(player, "player_id", "") or "")
        mark = pid.lower() if pid else str(id(player))
        if mark in seen or not _consider_unsigned(player):
            return
        seen.add(mark)
        try:
            from services.franchise_offseason import _dev_stamp_season_production

            _dev_stamp_season_production(session, player)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        transfer = bool(getattr(player, "requested_league_transfer", False))
        try:
            setattr(player, "_active_dev_season", sy)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        res = develop_unsigned_prospect(
            player,
            season_year=sy,
            nhl_team=nhl_team,
            request_league_transfer=transfer,
        )
        if res.get("ok"):
            rights = _unsigned_rights_id(player) or str(
                getattr(nhl_team, "team_id", None) or getattr(nhl_team, "id", "") or ""
            )
            results.append({**res, "rights_team_id": rights})

    for team in getattr(league, "teams", None) or []:
        for p in list(getattr(team, "prospect_pool", None) or []):
            _run_one(p, team)
    # Rights-held juniors live on the CHL / NCAA / Europe clubs, not in prospect_pool.
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                rights = _unsigned_rights_id(p)
                if not rights:
                    continue
                _run_one(p, teams_by_id.get(rights))

    payload = {
        "season_year": sy,
        "developed": len([r for r in results if not r.get("skipped")]),
        "results": results[:80],
        "avg_delta": round(sum(r.get("delta", 0) for r in results) / len(results), 3) if results else 0.0,
    }
    try:
        session.unsigned_development_payload = payload
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return payload
