#!/usr/bin/env python3
"""Baseline contract-economy diagnostic for elite UFAs, weak bidding, September signing."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
for p in (str(BACKEND), str(ROOT / "SimEngine"), str(ROOT / "SimEngine" / "app")):
    if p not in sys.path:
        sys.path.insert(0, p)

from services import franchise_sim  # noqa: E402
from services.contract_economy import (  # noqa: E402
    LEAGUE_MINIMUM_AAV_M,
    _player_id,
    _player_name,
    _player_ovr,
    compute_market_value,
    compute_player_demand,
    sign_player_to_team,
    sync_all_team_cap_fields,
)
from services.fa_market_engine import (  # noqa: E402
    _ask_for_player,
    ensure_fa_market_book,
    tick_free_agency_market,
)
from services.franchise_offseason import (  # noqa: E402
    _open_free_agency,
    _tick_league_contracts,
    build_free_agency_desk,
)


def _band(ovr: float) -> str:
    if ovr >= 90:
        return "90+"
    if ovr >= 85:
        return "85-89"
    if ovr >= 80:
        return "80-84"
    if ovr >= 70:
        return "70-79"
    if ovr >= 60:
        return "60-69"
    return "under-60"


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    session = franchise_sim.start_franchise(
        team_query="Colorado Avalanche",
        head_coach_name="Diag Coach",
        coach_archetype="balanced",
        seed=seed,
        player_universe="real_nhl",
        games_per_team=82,
    )
    league = session.sim.league
    season = int(session.season_calendar_year)
    report: dict = {"seed": seed, "season": season}

    # Depth demand spot-check
    depth_vals = []
    for t in league.teams:
        for p in list(getattr(t, "roster", None) or []):
            ovr = _player_ovr(p)
            if 55 <= ovr <= 68:
                depth_vals.append(compute_market_value(p, league))
    report["depth_market_value"] = {
        "n": len(depth_vals),
        "avg": round(sum(depth_vals) / max(1, len(depth_vals)), 3),
        "min": round(min(depth_vals), 3) if depth_vals else None,
        "max": round(max(depth_vals), 3) if depth_vals else None,
        "above_2m": sum(1 for v in depth_vals if v >= 2.0),
        "league_min": LEAGUE_MINIMUM_AAV_M,
    }

    _tick_league_contracts(session)
    opened = _open_free_agency(session, force=True)
    report["fa_open_ok"] = bool(opened.get("ok") or session.free_agency_open)
    fa_pool = list(getattr(league, "free_agents", None) or [])
    report["fa_pool_size"] = len(fa_pool)

    elites_day0 = []
    for p in fa_pool:
        ovr = _player_ovr(p)
        if ovr < 86:
            continue
        demand = compute_player_demand(p, None, league, context="ufa")
        elites_day0.append({
            "name": _player_name(p),
            "ovr": round(ovr),
            "ask": _ask_for_player(p, league),
            "want": demand["want_aav_m"],
            "min": demand["min_acceptable_aav_m"],
            "from": getattr(p, "ufa_from_team_id", None),
        })
    report["elite_ufas_day0"] = sorted(elites_day0, key=lambda x: -x["ovr"])[:25]

    # Tick FA market ~30 days (August/September secondary market)
    for _ in range(30):
        tick_free_agency_market(session, days=1)

    book = ensure_fa_market_book(session)
    unsigned_elite = []
    weak_multi_offers = []
    offer_dist = Counter()
    for pid, e in (book.get("entries") or {}).items():
        if e.get("state") == "signed":
            continue
        ovr = float(e.get("overall") or 0)
        offers = int(e.get("offer_count") or 0)
        offer_dist[_band(ovr)] += offers
        if ovr >= 86:
            unsigned_elite.append({
                "name": e.get("name"),
                "ovr": round(ovr),
                "ask": e.get("ask_aav_m"),
                "best": e.get("best_offer_m"),
                "offers": offers,
                "state": e.get("state"),
                "days": e.get("days_on_market"),
                "reason": e.get("reason"),
            })
        if 50 <= ovr <= 65 and offers >= 3:
            weak_multi_offers.append({
                "name": e.get("name"),
                "ovr": round(ovr),
                "offers": offers,
                "best": e.get("best_offer_m"),
                "ask": e.get("ask_aav_m"),
                "state": e.get("state"),
            })

    report["after_30_days"] = {
        "fa_pool_size": len(list(getattr(league, "free_agents", None) or [])),
        "unsigned_elite": sorted(unsigned_elite, key=lambda x: -x["ovr"])[:30],
        "unsigned_elite_count": len(unsigned_elite),
        "weak_50_65_with_3plus_offers": sorted(
            weak_multi_offers, key=lambda x: -x["offers"]
        )[:20],
        "weak_multi_offer_count": len(weak_multi_offers),
        "offers_by_band": dict(offer_dist),
        "market_day": book.get("day"),
    }

    # Simulate September / preseason desk + user signing eligibility
    session.phase = "preseason"
    session.season_phase = "preseason"
    session.offseason_stage = "preseason_start"
    # Keep free_agency_open False to mimic post-FA stage transition
    session.free_agency_open = False
    desk = build_free_agency_desk(session)
    report["preseason_desk"] = {
        "market_status": desk.get("market_status"),
        "market_phase": desk.get("market_phase"),
        "available_count": desk.get("available_count"),
        "empty_reason": desk.get("empty_reason"),
    }

    # Try user signing an unsigned FA in preseason
    user_team = session.team_by_id.get(str(session.user_team_id))
    sync_all_team_cap_fields(league, session.sim, season_year=season)
    candidate = None
    for p in list(getattr(league, "free_agents", None) or []):
        ovr = _player_ovr(p)
        if 70 <= ovr <= 78 and not getattr(p, "retired", False):
            candidate = p
            break
    sign_result = None
    if candidate is not None and user_team is not None:
        demand = compute_player_demand(candidate, user_team, league, context="ufa")
        sign_result = sign_player_to_team(
            candidate,
            user_team,
            league,
            season,
            {
                "aav_m": demand["want_aav_m"],
                "years": max(1, min(2, int(demand["want_years"]))),
                "context": "ufa",
                "_session": session,
            },
        )
    report["preseason_user_sign"] = {
        "player": _player_name(candidate) if candidate else None,
        "ovr": round(_player_ovr(candidate)) if candidate else None,
        "result_ok": (sign_result or {}).get("ok"),
        "status": (sign_result or {}).get("status"),
        "reason": (sign_result or {}).get("reason"),
    }

    # Regular season gate check
    session.phase = "regular_season"
    session.season_phase = "regular_season"
    desk2 = build_free_agency_desk(session)
    report["regular_season_desk"] = {
        "market_status": desk2.get("market_status"),
        "available_count": desk2.get("available_count"),
    }

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
