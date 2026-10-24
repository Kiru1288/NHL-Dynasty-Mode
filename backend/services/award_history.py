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
        from app.sim_engine.systems.chemistry import _canonical_player_id, _canonical_player_id_from_string  # noqa: WPS433
        from services.franchise_sim import _iter_league_players_for_aging

        want = _canonical_player_id_from_string(pid)
        raw = str(pid or "").strip()
        for p in _iter_league_players_for_aging(getattr(session.sim, "league", None)):
            canon = _canonical_player_id(p) or str(getattr(p, "id", "") or "")
            if canon == want or str(getattr(p, "id", "") or "") == raw:
                return p
    except Exception:
        log.debug("player lookup failed", exc_info=True)
    return None


def _nhl_id(p: Any) -> Optional[int]:
    try:
        from services.real_nhl_roster_importer import nhl_player_id_from_obj  # noqa: WPS433

        return nhl_player_id_from_obj(p)
    except Exception:
        log.debug("nhl id resolve failed", exc_info=True)
        return None


def _draft_payload_from_player(p: Any) -> Dict[str, Any]:
    ident = getattr(p, "identity", None)
    dy = getattr(p, "draft_year", None)
    if dy is None and ident is not None:
        dy = getattr(ident, "draft_year", None)
    dr = getattr(p, "draft_round", None)
    if dr is None and ident is not None:
        dr = getattr(ident, "draft_round", None)
    overall = getattr(p, "draft_overall_pick", None)
    if overall is None and ident is not None:
        overall = getattr(ident, "draft_pick", None)
    try:
        dy_i = int(dy) if dy is not None and int(dy) > 0 else None
    except (TypeError, ValueError):
        dy_i = None
    try:
        dr_i = int(dr) if dr is not None and int(dr) > 0 else None
    except (TypeError, ValueError):
        dr_i = None
    try:
        ov_i = int(overall) if overall is not None and int(overall) > 0 else None
    except (TypeError, ValueError):
        ov_i = None
    team_id = (
        getattr(p, "drafted_by_team_id", None)
        or getattr(p, "draft_team_id", None)
        or getattr(p, "drafted_by", None)
    )
    return {
        "drafted": bool(getattr(p, "drafted", False) or dy_i or ov_i),
        "undrafted": bool(getattr(p, "undrafted", False)) and not (dy_i or ov_i),
        "draft_year": dy_i,
        "draft_round": dr_i,
        "draft_overall_pick": ov_i,
        "draft_team_abbr": getattr(p, "draft_team_abbr", None),
        "drafted_by_team_id": team_id,
        "draft_team_id": getattr(p, "draft_team_id", None),
    }


def nhl_awards_for(p: Any, *, fetch: bool = True, league: Any = None) -> Dict[str, Any]:
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
        from services.real_nhl_roster_importer import ensure_player_nhl_landing_data

        if not ensure_player_nhl_landing_data(p, fetch=True, league=league):
            return {"status": "unavailable", "awards": []}
    except Exception:
        return {"status": "unavailable", "awards": []}
    return {"status": "ok", "awards": list(getattr(p, "nhl_awards", None) or [])}


def player_career_history(session: Any, pid: str, *, fetch: bool = True) -> Dict[str, Any]:
    """Pre-franchise NHL seasons for dossiers (NHL API landing, cached on the player)."""
    p = _find_player(session, pid)
    if p is None:
        return {"found": False, "player_id": str(pid), "career_seasons": [], "status": "unknown"}
    league = getattr(getattr(session, "sim", None), "league", None)
    if fetch:
        try:
            from services.real_nhl_roster_importer import ensure_player_nhl_landing_data

            if bool(getattr(p, "real_nhl_import", False)):
                ensure_player_nhl_landing_data(p, fetch=True, league=league)
            else:
                from services.real_nhl_roster_importer import player_missing_nhl_career_seasons

                if player_missing_nhl_career_seasons(p):
                    ensure_player_nhl_landing_data(p, fetch=True, league=league)
        except Exception:
            log.debug("career landing fetch failed", exc_info=True)
    cs = getattr(p, "career_stats", None)
    seasons: List[Dict[str, Any]] = []
    totals: Dict[str, Any] = {}
    try:
        from services.season_carryover import _franchise_first_season
    except Exception:
        _franchise_first_season = lambda session, season_year: int(season_year)  # type: ignore

    cur_sy = int(getattr(session, "season_calendar_year", 0) or 0)
    franchise_sy = int(_franchise_first_season(session, cur_sy) or cur_sy or 0)
    franchise_label = ""
    if franchise_sy > 0:
        try:
            from services.real_nhl_roster_importer import franchise_season_label_from_start_year

            franchise_label = franchise_season_label_from_start_year(franchise_sy)
        except Exception:
            franchise_label = f"{franchise_sy}-{str(franchise_sy + 1)[-2:]}"
    if isinstance(cs, dict):
        raw = cs.get("seasons") or cs.get("by_season") or cs.get("history") or []
        if isinstance(raw, list):
            seasons = [dict(s) for s in raw if isinstance(s, dict)][-24:]
        t = cs.get("totals") or cs.get("career") or cs.get("nhl")
        if isinstance(t, dict):
            totals = dict(t)
    if franchise_sy > 0:
        try:
            from services.real_nhl_roster_importer import drop_nhl_import_from_franchise_open

            seasons = drop_nhl_import_from_franchise_open(seasons, franchise_sy)
        except Exception:
            log.debug("franchise season strip failed", exc_info=True)
    if p is not None and franchise_sy > 0 and isinstance(getattr(p, "career_stats", None), dict):
        try:
            from services.real_nhl_roster_importer import drop_nhl_import_from_franchise_open

            cs_clean = dict(getattr(p, "career_stats") or {})
            cs_clean["seasons"] = drop_nhl_import_from_franchise_open(
                list(cs_clean.get("seasons") or []), franchise_sy
            )
            setattr(p, "career_stats", cs_clean)
        except Exception:
            log.debug("player career_stats strip failed", exc_info=True)
    status = "ok" if seasons else ("not_fetched" if not fetch else "empty")
    if not seasons and bool(getattr(p, "real_nhl_import", False)):
        imp = getattr(p, "real_nhl_import_stats", None) or {}
        if isinstance(imp, dict) and int(imp.get("gp") or imp.get("gamesPlayed") or 0) > 0:
            status = "partial_import_stats_only"
    draft = _draft_payload_from_player(p)
    nhl_awards: List[Dict[str, Any]] = []
    nhl_awards_status = "not_fetched"
    if p is not None:
        if getattr(p, "nhl_awards_fetched", False):
            nhl_awards_status = "ok"
            nhl_awards = list(getattr(p, "nhl_awards", None) or [])
            if franchise_sy > 0:
                try:
                    from services.real_nhl_roster_importer import drop_nhl_awards_from_franchise_open

                    nhl_awards = drop_nhl_awards_from_franchise_open(nhl_awards, franchise_sy)
                    setattr(p, "nhl_awards", nhl_awards)
                except Exception:
                    log.debug("nhl awards strip failed", exc_info=True)
        elif _nhl_id(p) is None:
            nhl_awards_status = "no_nhl_record"
        elif fetch:
            nhl_awards_status = "unavailable"
    return {
        "found": True,
        "player_id": str(pid),
        "name": getattr(getattr(p, "identity", None), "name", None) or getattr(p, "name", None),
        "career_seasons": seasons,
        "career_totals": totals,
        "status": status,
        "franchise_season_year": franchise_sy or None,
        "franchise_season_label": franchise_label or None,
        "nhl_awards": nhl_awards,
        "nhl_awards_status": nhl_awards_status,
        **draft,
    }


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
    try:
        from services.season_carryover import _franchise_first_season
    except Exception:
        _franchise_first_season = lambda session, season_year: int(season_year)  # type: ignore

    cur_sy = int(getattr(session, "season_calendar_year", 0) or 0)
    franchise_sy = int(_franchise_first_season(session, cur_sy) or cur_sy or 0)
    league = getattr(getattr(session, "sim", None), "league", None)
    nhl = nhl_awards_for(p, fetch=True, league=league)
    nhl_awards = list(nhl.get("awards") or [])
    if franchise_sy > 0 and p is not None:
        try:
            from services.real_nhl_roster_importer import drop_nhl_awards_from_franchise_open

            nhl_awards = drop_nhl_awards_from_franchise_open(nhl_awards, franchise_sy)
            if getattr(p, "nhl_awards_fetched", False):
                setattr(p, "nhl_awards", nhl_awards)
        except Exception:
            log.debug("awards history nhl strip failed", exc_info=True)
    trophies = sum(1 for a in sim if a.get("finish") == 1) + len(nhl_awards)
    career_seasons: List[Dict[str, Any]] = []
    career_totals: Dict[str, Any] = {}
    cs = getattr(p, "career_stats", None) if p is not None else None
    if isinstance(cs, dict):
        raw = cs.get("seasons") or cs.get("by_season") or cs.get("history") or []
        if isinstance(raw, list):
            career_seasons = [dict(s) for s in raw if isinstance(s, dict)][-24:]
        t = cs.get("totals") or cs.get("career") or cs.get("nhl")
        if isinstance(t, dict):
            career_totals = dict(t)
    if franchise_sy > 0:
        try:
            from services.real_nhl_roster_importer import drop_nhl_import_from_franchise_open

            career_seasons = drop_nhl_import_from_franchise_open(career_seasons, franchise_sy)
        except Exception:
            log.debug("awards history career strip failed", exc_info=True)
    out: Dict[str, Any] = {
        "player_id": str(pid),
        "found": p is not None,
        "name": getattr(getattr(p, "identity", None), "name", None) or getattr(p, "name", None),
        "save_awards": sim,
        "nhl_awards": nhl_awards,
        "nhl_status": nhl["status"],
        "career_seasons": career_seasons,
        "career_totals": career_totals,
        "trophy_count": trophies,
        "note": {
            "no_nhl_record": "Generated player: no real NHL record.",
            "unavailable": "The NHL API didn't answer; real trophies will load next time.",
            "unknown": "Player not found in this league.",
        }.get(nhl["status"]),
    }
    if p is not None:
        out.update(_draft_payload_from_player(p))
    return out
