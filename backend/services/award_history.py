"""A player's awards history for the dossier: real NHL trophies plus everything won (or
nearly won) inside this save.

* NHL: ``player.nhl_awards`` from the NHL API player landing (stored at import). Saves
  imported before that existed fetch the landing once on first view and cache it on the
  player; generated players have no NHL record and say so.
* Save: ``player.career_awards`` (winners, written at each awards night by
  ``apply_career_award_history``) and ``player.player_award_history["finishes"]`` (top-3
  placings), plus the Legacy Hall season rows for anything the player object lost.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def _find_player(session: Any, pid: str) -> Any:
    try:
        from services.franchise_sim import _iter_league_players_for_aging
        for p in _iter_league_players_for_aging(getattr(session.sim, "league", None)):
            if str(getattr(p, "id", "")) == str(pid):
                return p
    except Exception:
        log.debug("player lookup failed", exc_info=True)
    return None


def _nhl_id(p: Any) -> Optional[int]:
    for key in ("nhl_player_id", "nhl_id"):
        v = getattr(p, key, None) or getattr(getattr(p, "identity", None), key, None)
        try:
            if v:
                return int(v)
        except (TypeError, ValueError):
            continue
    return None


def nhl_awards_for(p: Any, *, fetch: bool = True) -> Dict[str, Any]:
    if p is None:
        return {"status": "unknown", "awards": []}
    if getattr(p, "nhl_awards_fetched", False):
        return {"status": "ok", "awards": list(getattr(p, "nhl_awards", None) or [])}
    nid = _nhl_id(p)
    if nid is None:
        return {"status": "no_nhl_record", "awards": []}
    if not fetch:
        return {"status": "not_fetched", "awards": []}
    try:
        from services.real_nhl_roster_importer import _PLAYER_LANDING_CACHE, _fetch_player_landing, parse_landing_awards

        landing = _PLAYER_LANDING_CACHE.get(nid) or _fetch_player_landing(nid)
    except Exception:
        landing = None
    if not landing:
        return {"status": "unavailable", "awards": []}
    awards = parse_landing_awards(landing)
    try:
        p.nhl_awards = awards
        p.nhl_awards_fetched = True
    except Exception:
        pass
    return {"status": "ok", "awards": awards}


def player_awards_history(session: Any, pid: str) -> Dict[str, Any]:
    p = _find_player(session, pid)
    sim: List[Dict[str, Any]] = []
    seen = set()
    for a in list(getattr(p, "career_awards", None) or []) if p is not None else []:
        if not isinstance(a, dict):
            continue
        key = a.get("award_result_id") or f"{a.get('award_id')}:{a.get('season')}"
        if key in seen:
            continue
        seen.add(key)
        selection = str(a.get("award_id") or "").startswith("all_star") or a.get("award_id") in ("jennings", "stanley")
        sim.append({
            "award_id": a.get("award_id"), "award_name": a.get("award_name") or a.get("name"),
            # All-Star teams and shared trophies are selections, not placings
            "season": a.get("season"), "finish": 1 if selection else int(a.get("finish") or 1), "team_name": a.get("team_name"),
            "stat_line": a.get("winning_stats") or {}, "source": "save",
        })
    hist = getattr(p, "player_award_history", None) if p is not None else None
    for f in (hist or {}).get("finishes") or [] if isinstance(hist, dict) else []:
        key = f"{f.get('award_id')}:{f.get('season')}:{f.get('finish')}"
        if key in seen or int(f.get("finish") or 0) == 1:
            continue
        seen.add(key)
        sim.append({**{k: f.get(k) for k in ("award_id", "award_name", "season", "finish", "team_name")}, "source": "save"})
    # Legacy Hall keeps winners per season even if the player object was rebuilt.
    lg = getattr(session, "legacy", None)
    if isinstance(lg, dict):
        for a in ((lg.get("players") or {}).get(str(pid)) or {}).get("awards") or []:
            key = f"legacy:{a.get('award')}:{a.get('season')}"
            if any(x.get("award_name") == a.get("award") and str(x.get("season")) == str(a.get("season")) for x in sim):
                continue
            sim.append({"award_name": a.get("award"), "season": a.get("season"), "finish": 1, "source": "save"})
    sim.sort(key=lambda a: (str(a.get("season") or ""), -int(a.get("finish") or 1)), reverse=True)
    nhl = nhl_awards_for(p)
    trophies = sum(1 for a in sim if a.get("finish") == 1) + len(nhl["awards"])
    return {
        "player_id": str(pid),
        "found": p is not None,
        "name": getattr(getattr(p, "identity", None), "name", None) or getattr(p, "name", None),
        "save_awards": sim,
        "nhl_awards": nhl["awards"],
        "nhl_status": nhl["status"],
        "trophy_count": trophies,
        "note": {
            "no_nhl_record": "Generated player: no real NHL record.",
            "unavailable": "The NHL API didn't answer; real trophies will load next time.",
            "unknown": "Player not found in this league.",
        }.get(nhl["status"]),
    }
