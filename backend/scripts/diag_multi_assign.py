"""Diagnose PLAYER_MULTI_ASSIGN after seed 1042 season rollover."""
from __future__ import annotations

import time
from collections import defaultdict

from services.franchise_entry_draft import complete_entry_draft, initialize_entry_draft
from services.franchise_offseason import (
    OFFSEASON_STAGES,
    _run_roster_cleanup,
    advance_season_phase,
    continue_offseason,
    generate_next_season,
)
from services.franchise_sim import advance_franchise_bulk, start_franchise


def _pid(p):
    return str(getattr(p, "player_id", None) or getattr(p, "id", "") or "")


def find_nhl_dups(session):
    seen = {}
    dups = []
    for tm in (session.team_by_id or {}).values():
        tid = str(getattr(tm, "team_id", None) or getattr(tm, "id", "") or "")
        for p in getattr(tm, "roster", None) or []:
            pid = _pid(p)
            if not pid:
                continue
            if pid in seen and seen[pid] != tid:
                dups.append((pid, seen[pid], tid, getattr(p, "name", None) or getattr(p, "full_name", None)))
            else:
                seen[pid] = tid
    return dups


def main():
    t0 = time.time()
    s = start_franchise(
        team_query="Buffalo Sabres",
        head_coach_name="Diag",
        coach_archetype="rebuild",
        seed=1042,
    )
    print("start", round(time.time() - t0, 1))
    while True:
        phase = str(getattr(s, "phase", "") or "")
        if phase not in ("regular", "preseason"):
            break
        if phase == "regular" and bool(getattr(s, "regular_season_complete", False)):
            break
        advance_franchise_bulk(s, mode="season" if phase == "regular" else "days", count=21 if phase == "preseason" else 1, auto_resolve_decisions=True)
    print("bulk", getattr(s, "phase", None), round(time.time() - t0, 1), "dups", len(find_nhl_dups(s)))
    for _ in range(20):
        ph = str(getattr(s, "phase", "") or "")
        if ph in ("offseason", "preseason"):
            break
        if ph == "playoff_ready":
            advance_season_phase(s, target="playoffs")
        else:
            advance_season_phase(s)
    print("phase", getattr(s, "phase"), getattr(s, "offseason_stage"), round(time.time() - t0, 1))
    for _ in range(len(OFFSEASON_STAGES) + 12):
        ph = str(getattr(s, "phase", "") or "")
        st = str(getattr(s, "offseason_stage", "") or "")
        if ph == "preseason":
            break
        if ph == "post_cup":
            continue_offseason(s)
            continue
        if st == "draft" or (
            not getattr(s, "draft_completed", False)
            and getattr(s, "draft_lottery_done", False)
            and st in ("draft", "")
        ):
            initialize_entry_draft(s)
            complete_entry_draft(s)
        if st == "roster_cleanup" or getattr(s, "next_important_event", "") == "generate_next_season":
            _run_roster_cleanup(s, force=True)
            print("pre_gen_dups", find_nhl_dups(s)[:5])
            generate_next_season(s)
            break
        continue_offseason(s, from_stage=st or None)
    print("after", getattr(s, "phase"), getattr(s, "season_calendar_year"), round(time.time() - t0, 1))
    dups = find_nhl_dups(s)
    print("nhl_dups", len(dups))
    for row in dups[:8]:
        print("DUP", row)
    # broader org locations for first dup
    if dups:
        pid = dups[0][0]
        locs = defaultdict(list)
        for tm in (s.sim.league.teams or []):
            tid = str(getattr(tm, "team_id", None) or getattr(tm, "id", "") or "")
            for attr in ("roster", "ahl_roster", "echl_roster", "prospect_pool"):
                for p in getattr(tm, attr, None) or []:
                    if _pid(p) == pid:
                        locs[pid].append((tid, attr))
        for p in getattr(s.sim.league, "free_agents", None) or []:
            if _pid(p) == pid:
                locs[pid].append(("FA", "free_agents"))
        print("LOC", dict(locs))


if __name__ == "__main__":
    main()
