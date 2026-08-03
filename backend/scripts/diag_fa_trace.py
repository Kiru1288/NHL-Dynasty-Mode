"""Trace why a specific CPU team makes zero FA offers."""
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
from services.contract_economy import (  # noqa: E402
    evaluate_team_position_needs,
    compute_fair_aav,
    cpu_signing_blocked,
    _player_ovr,
    _position_bucket,
)
from services.fa_market_engine import _serious_cpu_offer_aav, _is_exclusive_home_ufa  # noqa: E402


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

    target_abbr = "WSH"
    team = next(t for t in league.teams if (getattr(t, "abbreviation", None) or "") == target_abbr)
    ctx = evaluate_team_position_needs(team, league, session.sim, season_year=season_year)
    print(f"{target_abbr} ctx: cap_space={ctx['cap_space_m']:.2f} slots_remaining={ctx['slots_remaining']} window={ctx['window']}")
    print("need_score:", {k: round(v, 2) for k, v in ctx["need_score"].items()})
    print("overload:", ctx["overload"])

    fa_pool = [p for p in list(getattr(league, "free_agents", None) or []) if not _is_exclusive_home_ufa(p)]
    print(f"fa_pool (non-exclusive): {len(fa_pool)}")

    rng = session.sim.rng
    reasons = {}
    checked = 0
    offers_would_make = 0
    for player in fa_pool:
        ovr = float(_player_ovr(player))
        pos = _position_bucket(player)
        if ctx["slots_remaining"] <= 0:
            reasons["no_slots"] = reasons.get("no_slots", 0) + 1
            continue
        if ctx["cap_space_m"] < 0.775 * 1.02:
            reasons["no_cap_space"] = reasons.get("no_cap_space", 0) + 1
            continue
        checked += 1
        need = float(ctx["need_score"].get(pos, 0))
        window = ctx["window"]
        discount = 0.94 if window == "rebuilder" else (1.04 if need >= 0.45 else 0.98)
        if window == "cap_strapped":
            discount = 0.90
        if ovr >= 88:
            discount = max(discount, 0.96)
        fair = compute_fair_aav(player, team, league)
        offer_aav = _serious_cpu_offer_aav(
            fair=float(fair), ovr=ovr, cap_space_m=float(ctx["cap_space_m"]), discount=discount, rng=rng,
        )
        if offer_aav is None:
            reasons["offer_none"] = reasons.get("offer_none", 0) + 1
            continue
        block = cpu_signing_blocked(team, player, ctx, offer_aav)
        if block:
            reasons[f"blocked_{block}"] = reasons.get(f"blocked_{block}", 0) + 1
            continue
        offers_would_make += 1
        reasons["would_offer"] = reasons.get("would_offer", 0) + 1

    print(f"checked (passed slots+cap gate): {checked}")
    print("reasons breakdown:", reasons)
    print(f"would actually make offers to: {offers_would_make} players (before fit-score / per-team-cap / dup checks)")


if __name__ == "__main__":
    main()
