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


_REPAIR_FLAG = "_pot_chapter_repair_v1"


def repair_rated_potential(session: Any) -> int:
    """One-time fix for saves where a ratings-txt player's potential was rebuilt from hidden parts.

    The legacy sync averaged ceiling with bust/breakout odds into ``dev_potential``, so
    AHL kids and fringe players showed ceilings several points below the ratings file
    while their potential chapter still held the rated value. Lift them back to it.
    """
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return 0
    try:
        from app.sim_engine.progression.potential import write_player_potential99
    except Exception:
        return 0
    n = 0
    # Second pass, every load: any prospect whose shown ceiling fell under its rated
    # chapter (a stale copy written back by in-season development) is restored.
    for p in _iter_rights_players(league):
        try:
            chapters = (getattr(p, "attribute_profile", None) or {}).get("chapters") or {}
            rated = float(chapters.get("potential") or 0)
            shown = float((getattr(p, "ratings", None) or {}).get("dev_potential") or 0)
            if rated > 0 and shown < rated - 0.5:
                write_player_potential99(p, rated)
                n += 1
        except Exception:
            continue
    for team in list(getattr(league, "teams", None) or []):
        for attr in ("roster", "ahl_roster", "echl_roster", "prospect_pool", "injured_reserve"):
            for p in list(getattr(team, attr, None) or []):
                if getattr(p, _REPAIR_FLAG, False) or not getattr(p, "_dynasty_entry_key", None):
                    continue
                try:
                    setattr(p, _REPAIR_FLAG, True)
                    chapters = (getattr(p, "attribute_profile", None) or {}).get("chapters") or {}
                    rated = float(chapters.get("potential") or 0)
                    shown = float((getattr(p, "ratings", None) or {}).get("dev_potential") or 0)
                    if rated > 0 and shown < rated - 0.5:
                        write_player_potential99(p, rated)
                        n += 1
                except Exception:
                    continue
    return n
