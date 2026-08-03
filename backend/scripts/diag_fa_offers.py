"""Diagnostic: why do only a subset of CPU teams generate FA offers on day 1?"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
SIM = ROOT / "SimEngine" / "app"
for p in (str(BACKEND), str(SIM), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from services import franchise_sim  # noqa: E402
from services.contract_economy import evaluate_team_position_needs  # noqa: E402


def main() -> None:
    session = franchise_sim.start_franchise(
        team_query="Ottawa Senators",
        head_coach_name="Diag Coach",
        coach_archetype="balanced",
        seed=777,
        player_universe="real_nhl",
        games_per_team=82,
    )
    league = session.sim.league
    season_year = int(session.season_calendar_year)

    from services.franchise_offseason import _tick_league_contracts, _open_free_agency

    _tick_league_contracts(session)
    _open_free_agency(session, force=True)

    from services.fa_market_engine import tick_free_agency_market

    tick = tick_free_agency_market(session, days=1, max_signings_per_day=10, max_offers_per_day=400)
    offers = tick.get("offers") or []
    print(f"fa_pool size after open: {len(getattr(league, 'free_agents', None) or [])}")
    print(f"total offers day 1: {len(offers)}")

    from collections import Counter
    c = Counter(o.get("team_id") for o in offers)

    user_tid = str(session.user_team_id)
    rows = []
    for t in league.teams:
        tid = str(getattr(t, "team_id", "") or getattr(t, "id", ""))
        if tid == user_tid:
            continue
        ctx = evaluate_team_position_needs(t, league, session.sim, season_year=season_year)
        rows.append((
            getattr(t, "abbreviation", None) or "?",
            round(ctx["cap_space_m"], 2),
            ctx["slots_remaining"],
            ctx["window"],
            c.get(tid, 0),
        ))
    rows.sort(key=lambda r: r[4])
    print(f"{'team':6}{'cap_space':>10}{'slots':>7}{'window':>14}{'offers':>8}")
    for r in rows:
        print(f"{r[0]:6}{r[1]:10.2f}{r[2]:7d}{r[3]:>14}{r[4]:8d}")


if __name__ == "__main__":
    main()
