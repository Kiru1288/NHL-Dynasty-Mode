"""League-wide lift for rights-held prospect ceilings.

Drafted kids (junior / college / Europe, rights held by an NHL club) were rated with
conservative ceilings, so pipelines looked thin and prospects carried little value.
Every rights-held prospect gets one lift (idempotent per player): the gap to 99
closes by a quarter, with a +2 floor. Order on the board is preserved.
"""

from __future__ import annotations

from typing import Any, Iterable

_FLAG = "_rights_pot_boost_v1"


def _iter_rights_players(league: Any) -> Iterable[Any]:
    for team in list(getattr(league, "teams", None) or []):
        for p in list(getattr(team, "prospect_pool", None) or []):
            yield p
    for block in list(getattr(league, "development_leagues", None) or []):
        for tm in list((block or {}).get("teams") or []) if isinstance(block, dict) else []:
            for p in list((tm or {}).get("players") or []) if isinstance(tm, dict) else []:
                if getattr(p, "nhl_rights_team_id", None) or getattr(p, "rights_team_id", None):
                    yield p


def boosted_potential(pot99: float) -> float:
    return min(99.0, float(pot99) + max(2.0, (99.0 - float(pot99)) * 0.25))


def boost_rights_prospect_potential(session: Any) -> int:
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return 0
    try:
        from app.sim_engine.entities.player import player_current_ovr_01
        from app.sim_engine.progression.potential import (
            read_player_potential99,
            write_player_potential99,
        )
    except Exception:
        return 0
    n = 0
    seen: set = set()
    for p in _iter_rights_players(league):
        if id(p) in seen or getattr(p, _FLAG, False) or getattr(p, "retired", False):
            continue
        seen.add(id(p))
        try:
            pot = read_player_potential99(p)
            ovr = float(player_current_ovr_01(p)) * 99.0
            if pot <= 0:
                continue
            new = max(boosted_potential(pot), ovr + 3.0)
            write_player_potential99(p, new)
            setattr(p, _FLAG, True)
            n += 1
        except Exception:
            continue
    return n
