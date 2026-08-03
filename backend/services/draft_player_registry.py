"""
League-wide player registry keyed by stable player_id.

Development teams, NHL orgs, prospect pools, and rights records all resolve the same entity
through league.players_by_id rather than searching rosters ad hoc.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def ensure_players_by_id(league: Any) -> Dict[str, Any]:
    reg = getattr(league, "players_by_id", None)
    if not isinstance(reg, dict):
        try:
            league.players_by_id = {}
        except Exception:
            pass
        reg = getattr(league, "players_by_id", None)
        if not isinstance(reg, dict):
            return {}
    return reg


def register_player(league: Any, player: Any) -> Optional[str]:
    if player is None:
        return None
    pid = str(getattr(player, "id", "") or "")
    if not pid:
        return None
    reg = ensure_players_by_id(league)
    reg[pid] = player
    return pid


def get_player(league: Any, player_id: str) -> Any:
    pid = str(player_id or "")
    if not pid:
        return None
    reg = ensure_players_by_id(league)
    hit = reg.get(pid)
    if hit is not None:
        return hit
    # Lazy fill from known pools
    rebuild_players_by_id(league, only_if_missing=True)
    return reg.get(pid)


def _iter_known_players(league: Any) -> Iterable[Any]:
    for team in getattr(league, "teams", None) or []:
        for attr in (
            "roster",
            "prospect_pool",
            "prospects",
            "ahl_roster",
            "farm_roster",
            "reserve_players",
        ):
            for p in getattr(team, attr, None) or []:
                yield p
        for entry in getattr(team, "reserve_list", None) or []:
            if isinstance(entry, dict) and entry.get("player_ref") is not None:
                yield entry["player_ref"]
        for entry in getattr(team, "rfa_rights", None) or []:
            if isinstance(entry, dict) and entry.get("player_ref") is not None:
                yield entry["player_ref"]
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                yield p
    for p in getattr(league, "global_player_pool", None) or []:
        yield p
    for pool_attr in ("free_agents", "overseas_free_agents"):
        for p in getattr(league, pool_attr, None) or []:
            yield p


def rebuild_players_by_id(league: Any, *, only_if_missing: bool = False) -> Dict[str, Any]:
    reg = ensure_players_by_id(league)
    if only_if_missing and reg:
        # Still add any new ids without wiping existing mappings
        for p in _iter_known_players(league):
            pid = str(getattr(p, "id", "") or "")
            if pid and pid not in reg:
                reg[pid] = p
        return reg
    for p in _iter_known_players(league):
        register_player(league, p)
    return reg


def find_development_home(league: Any, player_id: str):
    """Return (player, development_block, development_team_dict) if still on a development roster."""
    pid = str(player_id or "")
    player = get_player(league, pid)
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if str(getattr(p, "id", "") or "") == pid:
                    return p, block, tm
    if player is not None:
        return player, None, None
    return None, None, None
