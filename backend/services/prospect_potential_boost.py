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
    # Potential is the ceiling already on the card. A blanket lift, even +2,
    # put the same green number on every rights-held prospect.
    return float(pot99)


def boost_rights_prospect_potential(session: Any) -> int:
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return 0
    try:
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
            if pot <= 0:
                continue
            new = boosted_potential(pot)
            if new > pot + 0.05:
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
        if getattr(p, "_pot_pyramid_v1", False):
            continue
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
                if getattr(p, _REPAIR_FLAG, False) or getattr(p, "_pot_pyramid_v1", False) or not getattr(p, "_dynasty_entry_key", None):
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


_PYRAMID_FLAG = "_pot_pyramid_v1"
_PYRAMID_SLOTS = (94.0, 88.0, 88.0, 84.0, 84.0, 84.0)


def _player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    raw = getattr(ident, "age", None) if ident is not None else None
    if raw is None:
        raw = getattr(player, "age", 21)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 21


def _player_ovr(player: Any) -> float:
    try:
        from app.sim_engine.entities.player import display_rating, player_current_ovr_01

        return float(display_rating(player_current_ovr_01(player)))
    except Exception:
        return 70.0


def _rights_groups(league: Any) -> dict:
    groups: dict = {}

    def _add(player: Any, team_id: Any) -> None:
        if player is None or getattr(player, "retired", False) or not team_id:
            return
        groups.setdefault(str(team_id), []).append(player)

    for team in list(getattr(league, "teams", None) or []):
        tid = getattr(team, "team_id", None) or getattr(team, "id", None)
        for attr in ("prospect_pool", "ahl_roster", "echl_roster"):
            for player in list(getattr(team, attr, None) or []):
                _add(player, tid)
    for block in list(getattr(league, "development_leagues", None) or []):
        teams = (block or {}).get("teams") or [] if isinstance(block, dict) else []
        for tm in teams:
            for player in list((tm or {}).get("players") or []) if isinstance(tm, dict) else []:
                rights = getattr(player, "nhl_rights_team_id", None) or getattr(player, "rights_team_id", None)
                _add(player, rights)
    return groups


def rebalance_prospect_ceilings(session: Any) -> int:
    """One high-end prospect per club. The rest of a rights list is not a 99.

    A club has one kid who can be in the low 90s, a couple in the high 80s,
    and a long list in the low 80s. The old lifts put the whole list at 93-99.
    """
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return 0
    try:
        from app.sim_engine.progression.potential import (
            read_player_potential99,
            write_player_potential99,
        )
    except Exception:
        return 0
    n = 0
    for players in _rights_groups(league).values():
        unique = []
        seen: set = set()
        for player in players:
            if id(player) in seen or getattr(player, _PYRAMID_FLAG, False):
                continue
            seen.add(id(player))
            unique.append(player)
        ranked = []
        for player in unique:
            try:
                pot = float(read_player_potential99(player))
            except Exception:
                pot = 0.0
            ranked.append((pot, _player_ovr(player), player))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        for index, (pot, ovr, player) in enumerate(ranked):
            try:
                age = _player_age(player)
                room = 14.0 if age <= 19 else (10.0 if age <= 21 else 7.0)
                slot = _PYRAMID_SLOTS[index] if index < len(_PYRAMID_SLOTS) else 80.0
                tier = str(getattr(player, "pipeline_tier", "") or "").lower()
                if index == 0 and tier == "transcendent":
                    ceiling = min(pot, 96.0)
                else:
                    ceiling = min(pot, slot, ovr + room)
                ceiling = max(ovr, ceiling)
                if pot > ceiling + 0.5:
                    write_player_potential99(player, ceiling)
                    profile = getattr(player, "development_profile", None)
                    if isinstance(profile, dict):
                        profile["expected_ceiling"] = ceiling / 99.0
                        profile["maximum_ceiling"] = min(0.99, ceiling / 99.0 + 0.02)
                    ratings = getattr(player, "ratings", None)
                    if isinstance(ratings, dict) and ratings.get("dev_ceiling") is not None:
                        ratings["dev_ceiling"] = float(min(99.0, ceiling + 2.0))
                    try:
                        setattr(player, "season_start_potential", int(round(ceiling)))
                    except Exception:
                        pass
                    n += 1
                setattr(player, _PYRAMID_FLAG, True)
            except Exception:
                continue
    return n
