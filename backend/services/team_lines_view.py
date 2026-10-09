"""Read-only view of any club's lines (NHL or AHL) for the Edit Lines scouting view.

NHL lines are the units the game engine actually dresses for that club (saved lineup
if it has one, otherwise the coach's best-first lines), plus its power play, penalty
kill, goalies and bench strategy. AHL lines come from the AHL league's lineup logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.lineup_integrity import player_key, player_name, player_ovr, position_bucket

STYLE_LABELS = {"rolling_four": "Rolls four lines", "balanced": "Balanced", "top_heavy": "Leans on top lines"}


def _tid(team: Any) -> str:
    raw = getattr(team, "team_id", None)
    if raw is None:
        raw = getattr(team, "id", None)
    return str(raw) if raw is not None else ""


def _row(p: Any) -> Optional[Dict[str, Any]]:
    if p is None:
        return None
    ident = getattr(p, "identity", None)
    pos = getattr(ident, "position", None) if ident is not None else getattr(p, "position", None)
    pos = str(getattr(pos, "value", pos) or "").split(".")[-1].upper()
    pot = getattr(p, "potential", None)
    try:
        pot = pot() if callable(pot) else pot
        pot = round(float(pot) * 99 if float(pot) <= 1.5 else float(pot)) if pot is not None else None
    except Exception:
        pot = None
    row: Dict[str, Any] = {
        "id": player_key(p), "name": player_name(p), "position": pos, "pos": position_bucket(p),
        "ovr": round(player_ovr(p)), "pot": pot,
        "age": int(getattr(ident, "age", 0) or getattr(p, "age", 0) or 0) or None,
        "injured": bool(getattr(p, "injury", None) or getattr(p, "injured", False)),
    }
    try:
        from app.sim_engine.generation.player_headshots import merge_headshot_into_row  # noqa: WPS433

        row = merge_headshot_into_row(row, p) or row
    except Exception:
        pass
    return row


def _unit(players: List[Any]) -> Dict[str, Any]:
    rows = [r for r in (_row(p) for p in players or []) if r]
    avg = round(sum(r["ovr"] for r in rows) / len(rows), 1) if rows else None
    return {"players": rows, "avg_ovr": avg}


def team_options(session: Any) -> List[Dict[str, Any]]:
    utid = str(getattr(session, "user_team_id", "") or "")
    out = []
    for t in (getattr(session, "team_by_id", None) or {}).values():
        abbr = str(getattr(t, "abbreviation", None) or getattr(t, "abbr", "") or "").upper()
        name = str(getattr(t, "full_name", None) or f"{getattr(t, 'city', '')} {getattr(t, 'name', '')}".strip() or abbr)
        out.append({"team_id": _tid(t), "abbr": abbr, "name": name, "is_user": _tid(t) == utid})
    out.sort(key=lambda r: (not r["is_user"], r["name"]))
    return out


def _nhl_view(session: Any, team: Any) -> Dict[str, Any]:
    sim = getattr(session, "sim", None)
    if sim is None:
        return {"ok": False, "reason": "No sim"}
    skaters = sim._gm_skaters(team)
    units = sim._gm_build_game_units(skaters, team)
    goalies = sorted(sim._gm_goalies(team), key=player_ovr, reverse=True)
    style = str(units.get("deployment_style") or "balanced")
    strength_rank = None
    try:
        sm = getattr(session, "strength_map", None) or {}
        order = sorted(sm.items(), key=lambda kv: -float(kv[1]))
        strength_rank = next((i + 1 for i, (k, _) in enumerate(order) if str(k) == _tid(team)), None)
    except Exception:
        strength_rank = None
    dressed = {player_key(p) for ln in units.get("lines") or [] for p in ln} | {player_key(p) for pr in units.get("pairs") or [] for p in pr}
    dressed |= {player_key(p) for p in goalies[:2]}
    extras = [r for r in (_row(p) for p in list(getattr(team, "roster", None) or [])) if r and r["id"] not in dressed]
    return {
        "ok": True,
        "forwards": [_unit(list(ln or [])[:3]) for ln in (units.get("lines") or [])[:4]],
        "defense": [_unit(pr) for pr in (units.get("pairs") or [])[:3]],
        "goalies": {"starter": _row(goalies[0]) if goalies else None, "backup": _row(goalies[1]) if len(goalies) > 1 else None},
        "power_play": [_unit(units.get("pp1") or []), _unit(units.get("pp2") or [])],
        "penalty_kill": [_unit(units.get("pk1") or []), _unit(units.get("pk2") or [])],
        "deployment_style": style,
        "deployment_label": STYLE_LABELS.get(style, style.replace("_", " ").title()),
        "strength_rank": strength_rank,
        "extras": sorted(extras, key=lambda r: -r["ovr"]),
    }


def _ahl_view(session: Any, team: Any) -> Dict[str, Any]:
    from services.ahl_league import _affiliate_name, _ahl_players, team_ahl_lines  # noqa: WPS433

    players = {player_key(p): p for p in _ahl_players(team)}
    lines = team_ahl_lines(session, team)

    def units(group: str) -> List[Dict[str, Any]]:
        out = []
        for u in (lines or {}).get(group) or []:
            out.append(_unit([players.get(str(pid)) for pid in ((u or {}).get("slots") or {}).values() if str(pid or "") in players]))
        return out

    g_units = units("goalies")
    g_rows = g_units[0]["players"] if g_units else []
    used = {r["id"] for grp in ("forwards", "defense") for u in units(grp) for r in u["players"]} | {r["id"] for r in g_rows}
    extras = [r for r in (_row(p) for p in players.values()) if r and r["id"] not in used]
    return {
        "ok": True,
        "affiliate_name": _affiliate_name(team),
        "forwards": units("forwards")[:4],
        "defense": units("defense")[:3],
        "goalies": {"starter": g_rows[0] if g_rows else None, "backup": g_rows[1] if len(g_rows) > 1 else None},
        "extras": sorted(extras, key=lambda r: -r["ovr"]),
    }


def build_team_lines_view(session: Any, team_id: str, level: str = "nhl") -> Dict[str, Any]:
    team = (getattr(session, "team_by_id", None) or {}).get(str(team_id))
    if team is None:
        raise ValueError("Team not found.")
    lvl = "ahl" if str(level).lower() == "ahl" else "nhl"
    body = _ahl_view(session, team) if lvl == "ahl" else _nhl_view(session, team)
    opt = next((o for o in team_options(session) if o["team_id"] == str(team_id)), {})
    return {**body, "level": lvl, "team_id": str(team_id), "team_name": opt.get("name"), "abbr": opt.get("abbr"),
            "is_user": bool(opt.get("is_user")), "teams": team_options(session)}
