"""Diagnostic: exercise contract-tick -> free-agency-open pipeline directly and
inspect cap snapshots + FA offer spread across all 32 teams."""
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
from services.contract_economy import get_team_cap_snapshot_full  # noqa: E402


def cap_table(league, sim, season_year, label):
    rows = []
    for t in league.teams:
        snap = get_team_cap_snapshot_full(t, league, sim, season_year=season_year)
        rows.append((
            getattr(t, "abbreviation", None) or getattr(t, "abbrev", "?"),
            snap["usable_cap_space_m"],
            snap["total_cap_hit_m"],
            snap["active_roster_count"],
        ))
    rows.sort(key=lambda r: r[1])
    print(f"\n--- {label} ---")
    under_1m = sum(1 for r in rows if r[1] < 1.0)
    print(f"teams with <$1M usable space: {under_1m} / {len(rows)}")
    for r in rows[:8]:
        print(f"  {r[0]:6} usable={r[1]:7.2f} total={r[2]:7.2f} roster={r[3]}")
    print("  ...")
    for r in rows[-5:]:
        print(f"  {r[0]:6} usable={r[1]:7.2f} total={r[2]:7.2f} roster={r[3]}")
    return {r[0]: r[1] for r in rows}


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

    before = cap_table(league, session.sim, season_year, "BEFORE any contract tick")

    from services.franchise_offseason import _tick_league_contracts

    tick_result = _tick_league_contracts(session)
    print(f"\ntick: expired_ufas={len(tick_result['expired_ufas'])} expired_rfas={len(tick_result['expired_rfas'])}")

    after_tick = cap_table(league, session.sim, season_year, "AFTER tick (defer_july1_ufa=True, before FA opens)")

    # Simulate the exclusive window elapsing without any user re-signing action.
    from services.franchise_offseason import _open_free_agency

    fa_result = _open_free_agency(session, force=True)
    print(f"\nfa_open ok={fa_result.get('ok', True)} reason={fa_result.get('reason')}")

    after_fa_open = cap_table(league, session.sim, season_year, "AFTER free agency opens (July1 burn + CPU re-signs)")

    # Tick the market forward several days and see offer spread.
    from services.fa_market_engine import tick_free_agency_market

    tick = tick_free_agency_market(session, days=5, max_signings_per_day=10, max_offers_per_day=400)
    offers = tick.get("offers") or []
    teams_making_offers = sorted(set(o.get("team_id") for o in offers))
    print(f"\noffers made over 5 FA days: {len(offers)}; distinct teams making offers: {len(teams_making_offers)}")
    from collections import Counter
    c = Counter(o.get("team_abbrev") for o in offers)
    print("offers per team (top 10):", c.most_common(10))
    print("offers per team (bottom 10):", c.most_common()[-10:])
    print(f"signings this tick: {len(tick.get('signings') or [])}")

    after_market = cap_table(league, session.sim, season_year, "AFTER 5 days of FA market ticking")

    ott = getattr(league, "teams", [])
    ott_team = session.team_by_id.get(str(session.user_team_id))
    print(f"\nUser team (Ottawa) usable space: before={before.get('OTT')} after_tick={after_tick.get('OTT')} after_fa_open={after_fa_open.get('OTT')} after_market={after_market.get('OTT')}")


if __name__ == "__main__":
    main()
