"""Bridge franchise storyline / universe modifiers into live engine.py stat allocation.

Also feeds the game stats ledger back into morale and storyline beats.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _player_id(player: Any) -> str:
    return str(getattr(player, "id", "") or "")


def _roster_modifiers(team: Any) -> Dict[str, Dict[str, float]]:
    from app.sim_engine.franchise.storyline_conduct import get_player_stat_allocation_modifiers  # noqa: WPS433

    out: Dict[str, Dict[str, float]] = {}
    for player in getattr(team, "roster", None) or []:
        pid = _player_id(player)
        if not pid:
            continue
        mods = get_player_stat_allocation_modifiers(player)
        if mods:
            out[pid] = mods
    return out


def player_stat_map_from_box(box: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Flatten home/away skater + goalie rows from a completed stats-ledger box."""
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(box, dict):
        return out
    for key in ("home_skaters", "away_skaters"):
        for row in list(box.get(key) or []):
            if not isinstance(row, dict):
                continue
            pid = str(row.get("player_id") or row.get("id") or "")
            if not pid:
                continue
            g = int(row.get("g") or row.get("goals") or 0)
            a = int(row.get("a") or row.get("assists") or 0)
            out[pid] = {
                "goals": g,
                "assists": a,
                "points": g + a,
                "sog": int(row.get("sog") or 0),
                "pim": int(row.get("pim") or 0),
                "hits": int(row.get("hit") or row.get("hits") or 0),
                "blocks": int(row.get("blk") or row.get("blocks") or 0),
                "toi_sec": int(row.get("toi_sec") or 0),
                "xgf": float(row.get("xgf") or 0),
                "xga": float(row.get("xga") or 0),
                "position": str(row.get("position") or ""),
                "is_goalie": False,
            }
    for key in ("home_goalie", "away_goalie"):
        row = box.get(key)
        if not isinstance(row, dict):
            continue
        pid = str(row.get("player_id") or row.get("id") or "")
        if not pid:
            continue
        ga = int(row.get("ga") or row.get("goals_against") or 0)
        saves = int(row.get("saves") or row.get("sv") or 0)
        sa = int(row.get("sa") or row.get("shots_against") or 0) or (saves + ga)
        out[pid] = {
            "goals": 0,
            "assists": int(row.get("a") or 0),
            "points": int(row.get("a") or 0),
            "ga": ga,
            "saves": saves,
            "sa": sa,
            "shutout": bool(row.get("so") or row.get("shutout") or (ga == 0 and sa >= 1)),
            "is_goalie": True,
            "position": "G",
        }
    for key in ("home_goalies", "away_goalies"):
        for row in list(box.get(key) or []):
            if not isinstance(row, dict):
                continue
            pid = str(row.get("player_id") or row.get("id") or "")
            if not pid or pid in out:
                continue
            ga = int(row.get("ga") or 0)
            saves = int(row.get("saves") or 0)
            sa = int(row.get("sa") or 0) or (saves + ga)
            out[pid] = {
                "goals": 0,
                "assists": int(row.get("a") or 0),
                "points": int(row.get("a") or 0),
                "ga": ga,
                "saves": saves,
                "sa": sa,
                "shutout": bool(row.get("so") or (ga == 0 and sa >= 1)),
                "is_goalie": True,
                "position": "G",
            }
    return out


def apply_stats_ledger_to_storylines(session: Any, box: Dict[str, Any], rng: Any = None) -> Dict[str, Any]:
    """After a game box is written to the ledger, drive morale + published story beats."""
    if not isinstance(box, dict):
        return {"ingested": 0, "postgame": []}
    stats_map = player_stat_map_from_box(box)
    ingested = 0
    try:
        from app.sim_engine.franchise.storyline_coverage import ingest_game_box_storylines  # noqa: WPS433

        ingested = int(ingest_game_box_storylines(session, box) or 0)
    except Exception:
        ingested = 0
    hid = str(box.get("home_id") or "")
    aid = str(box.get("away_id") or "")
    hg = int(box.get("home_goals") or box.get("home_score") or 0)
    ag = int(box.get("away_goals") or box.get("away_score") or 0)
    postgame: list = []
    for tid, won in ((hid, hg > ag), (aid, ag > hg)):
        if not tid:
            continue
        payload = {
            "won": won,
            "result": "W" if won else "L",
            "game_id": box.get("game_id") or box.get("id"),
            "player_stats": stats_map,
            "home_id": hid,
            "away_id": aid,
            "home_goals": hg,
            "away_goals": ag,
            "home_skaters": box.get("home_skaters"),
            "away_skaters": box.get("away_skaters"),
            "home_goalie": box.get("home_goalie"),
            "away_goalie": box.get("away_goalie"),
        }
        try:
            from app.sim_engine.franchise.storyline_engine import apply_universe_postgame  # noqa: WPS433

            postgame.append(apply_universe_postgame(session, tid, payload, rng=rng))
        except Exception:
            continue
    return {"ingested": ingested, "postgame": postgame}


def build_franchise_game_stat_context(
    session: Any,
    home_team_id: str,
    away_team_id: str,
    game_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build per-player stat allocation modifiers for one matchup.

    Prefers narrative-universe V3 when available; otherwise maps active
    storyline conduct readiness modifiers onto stat fingerprints.
    """
    meta = dict(game_meta or {})
    try:
        from app.sim_engine.franchise.storyline_engine import build_universe_matchup_context  # noqa: WPS433

        return build_universe_matchup_context(session, home_team_id, away_team_id, meta)
    except (ImportError, AttributeError):
        pass

    team_by_id = getattr(session, "team_by_id", None) or {}
    home = team_by_id.get(str(home_team_id))
    away = team_by_id.get(str(away_team_id))
    home_mods = _roster_modifiers(home) if home is not None else {}
    away_mods = _roster_modifiers(away) if away is not None else {}
    base_id = str(meta.get("game_id") or f"conduct_{home_team_id}_{away_team_id}_{meta.get('calendar_day', 0)}")
    return {
        "id": base_id,
        "home_team_id": str(home_team_id),
        "away_team_id": str(away_team_id),
        "home": {"player_modifiers": home_mods},
        "away": {"player_modifiers": away_mods},
        "home_win_probability_delta": 0.0,
    }


def prime_franchise_game_stat_modifiers(
    session: Any,
    sim: Any,
    home_team_id: str,
    away_team_id: str,
    *,
    game_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach storyline/universe modifiers to the live SimEngine for one game."""
    context = build_franchise_game_stat_context(session, home_team_id, away_team_id, game_meta)
    home_mods = dict((context.get("home") or {}).get("player_modifiers") or {})
    away_mods = dict((context.get("away") or {}).get("player_modifiers") or {})
    fn = getattr(sim, "set_franchise_game_stat_modifiers", None)
    if callable(fn):
        fn(
            home_player_modifiers=home_mods,
            away_player_modifiers=away_mods,
            home_win_probability_delta=float(context.get("home_win_probability_delta", 0) or 0),
        )
    return context


def clear_franchise_game_stat_modifiers(sim: Any) -> None:
    fn = getattr(sim, "clear_franchise_game_stat_modifiers", None)
    if callable(fn):
        fn()


def apply_franchise_postgame_storyline(
    session: Any,
    team_id: str,
    game_result: Dict[str, Any],
    rng: Any = None,
) -> None:
    """Optional universe postgame tick when V3 helpers exist."""
    try:
        from app.sim_engine.franchise.storyline_engine import apply_universe_postgame  # noqa: WPS433
    except (ImportError, AttributeError):
        return
    try:
        apply_universe_postgame(session, team_id, game_result, rng=rng)
    except Exception:
        return
