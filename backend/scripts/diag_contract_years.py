"""Diagnostic: distribution of years_remaining across the whole real-NHL league at bootstrap."""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
SIM = ROOT / "SimEngine" / "app"
for p in (str(BACKEND), str(SIM), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from services import franchise_sim  # noqa: E402
from services.contract_economy import _contract_years_remaining, _get  # noqa: E402


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

    yrs_counter = Counter()
    no_contract = 0
    total_players = 0
    ott_players = []
    for team in league.teams:
        abbr = getattr(team, "abbreviation", None) or "?"
        for attr in ("roster", "ahl_roster", "echl_roster"):
            for p in list(getattr(team, attr, None) or []):
                total_players += 1
                c = getattr(p, "contract", None)
                if c is None:
                    no_contract += 1
                    yrs_counter["NONE"] += 1
                else:
                    yrs = _contract_years_remaining(p)
                    yrs_counter[yrs] += 1
                if abbr == "OTT" and attr == "roster":
                    name = getattr(getattr(p, "identity", None), "name", None) or getattr(p, "name", "?")
                    aav = getattr(p, "aav_m", None) or getattr(p, "cap_hit_m", None)
                    yrs = _contract_years_remaining(p)
                    ott_players.append((str(name), yrs, aav, getattr(p, "real_nhl_contract", False)))

    print(f"total players (roster+ahl+echl) across league: {total_players}")
    print(f"no contract at all: {no_contract}")
    print("years_remaining distribution:", dict(sorted(((str(k), v) for k, v in yrs_counter.items()), key=lambda x: str(x[0]))))

    print("\n--- Ottawa NHL roster ---")
    for name, yrs, aav, real in sorted(ott_players, key=lambda x: -(x[2] or 0)):
        print(f"  {name:28} yrs_remaining={yrs} aav={aav} real_contract={real}")


if __name__ == "__main__":
    main()
